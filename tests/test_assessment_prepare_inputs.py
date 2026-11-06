"""Synthetic cache/chunk recovery qualification; no production I/O or network."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import assessment_prepare_inputs as helper
from hydra_directflow.assessment_data import AssessmentReadAuthority
from hydra_directflow.assessment_runtime import AssessmentRuntime, LIMITS, MODEL_IDS, AssessmentIntegrityError
from hydra_directflow.local_data import LocalCycle, GAUGES, WEIGHTS_SHA256, USGS_MANIFEST_SHA256, basin_reduce, iso
from hydra_directflow.monthly import cycle_ids_for_origins
from hydra_directflow.reference import ASSESSMENT_START, HOUR, Observation, digest
from hydra_directflow.training import array_identity
from test_assessment_data import cycle as synthetic_cycle


SOURCE='c'*64


class FakeBudget:
    def __init__(self):self.bytes=0;self.gets=0
    def snapshot(self):
        return dict(observed_body_bytes=self.bytes,precharged_body_bytes=self.bytes,
                    attempted_gets=self.gets,body_cap=16*1024**3,get_cap=4000)


class FakeProvider:
    def __init__(self,directory,runtime,plan):
        self.directory=directory;self.runtime=runtime;self.plan=plan;self.owner='synthetic-owner'
        (directory/'raw').mkdir(parents=True);(directory/'commits').mkdir()
        self.budget=FakeBudget();self.calls=[];self.evictions=[];self.fail_eviction=False
    def __call__(self,cid,row):
        self.calls.append(cid);raw=('synthetic-archive:'+cid).encode();p=self.directory/'raw'/(cid+'.tar')
        p.write_bytes(raw);self.budget.bytes+=len(raw);self.budget.gets+=1
        return dict(status='READY',spec=dict(cycle_id=cid,archive_path=str(p)),
            ownership=dict(owned_archive=True,archive_path=str(p),owner_run_id=self.owner,
                           archive_sha256=hashlib.sha256(raw).hexdigest()))
    def on_cache_committed(self,proof,supplied):
        cid=supplied['spec']['cycle_id'];c=self.runtime.reuse('CYCLE/'+cid)
        assert c is not None and proof['cache_payload_verified'] is True
        assert proof['cache_commit_sha256']==digest(c)
        assert proof['cache_key']==c['metadata']['cache_key']
        assert proof['archive_sha256']==supplied['ownership']['archive_sha256']
        if self.fail_eviction:raise OSError('synthetic interruption after durable cache commit')
        self.evictions.append(cid);Path(supplied['ownership']['archive_path']).unlink()


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.calendar=tuple(ASSESSMENT_START+i*HOUR for i in range(240))
        ids=cycle_ids_for_origins(list(self.calendar))
        self.index={cid:dict(cycle=cid,source='synthetic') for cid in ids}
        rows=[dict(cycle_id=cid,sealed_row_sha256=digest(row),bundle_identity=None,
                   commit_identity=None,local_commit_candidates=[],local_archive_candidates=[])
              for cid,row in self.index.items()]
        self.stage=dict(schema='hydra-assessment-staging-plan/v1',status='APPROVED_BOUNDED_STAGING',
            cycles=rows,sources=dict(source_inventory=dict(sha256='4'*64)))
        self.stage['plan_sha256']=digest(self.stage)
        self.plan_rows={r['cycle_id']:r for r in rows}
        p=dict(run_directory=str(self.root/'run'),runtime_limits=dict(LIMITS),
            checkpoint_sha256_by_job={j:hashlib.sha256(j.encode()).hexdigest() for j in MODEL_IDS},
            input_source_identity=SOURCE,weights=dict(path=str(self.root/'weights.csv'),bytes=62601,sha256=WEIGHTS_SHA256))
        p['plan_sha256']=digest(p)
        self.runtime=AssessmentRuntime(p,resource_probe=lambda:dict(combined_rss_bytes=128*1024**2,
            mps_bytes=0,free_bytes=16*1024**3))
        self.authority=AssessmentReadAuthority(p['plan_sha256'],'2'*64,'3'*64,'4'*64,
            tuple((cid,digest(row)) for cid,row in self.index.items()),inputs_enabled=True)
        self.provider=FakeProvider(self.root/'sources',self.runtime,self.stage)
        self.weights=np.zeros((3,957));self.weights[:,0]=1
        self.binding=helper._identity(self.runtime,self.authority,SOURCE,self.stage)
        self.loaded=[]
        self.observations=tuple(Observation(g,t,None,20.+i,('A','R'),'usgs-wire-sha256:'+str(i)*64)
            for i,g in enumerate(GAUGES) for t in range(self.calendar[0]-169*HOUR,self.calendar[-1]-2*HOUR+1,HOUR))
    def tearDown(self):self.temp.cleanup()

    def load_cycle(self,spec,index_row,authority,*,read_budget):
        cid=spec['cycle_id'];self.loaded.append(cid)
        raw=Path(spec['archive_path']).read_bytes();read_budget(len(raw))
        ref=helper.assessment_cycle_reference(cid);_,cycle,_=synthetic_cycle(ref)
        values=cycle.values.copy()
        for i,row in enumerate(cycle.records):
            if row['group']=='soil_moisture':values[i]=.25
        cycle=replace(cycle,values=values,member_count=17,fields={'original':'retained'})
        receipt=dict(cycle_id=cid,sealed_row_sha256=digest(index_row),
            execution_plan_sha256=authority.execution_plan_sha256,archive_sha256=hashlib.sha256(raw).hexdigest(),
            archive_bytes=len(raw),manifest_sha256='d'*64,verification_target_arrays_read=False,
            numeric_records_decoded=len(cycle.records),lineage={'original':True})
        return cycle,receipt

    def usgs_commit(self):
        d=self.runtime.write('synthetic-usgs-receipt.json',b'{"synthetic":true}',numeric=False)
        return self.runtime.commit('USGS_FIXTURE',dict(receipt=d),metadata={'synthetic':True})

    def test_cycle_cache_restores_truthful_raw_provenance_and_reductions_without_redownload(self):
        cid=next(iter(self.index))
        with self.runtime.writer(),patch.object(helper,'load_assessment_cycle',side_effect=self.load_cycle):
            cycle,reduced,commit=helper._cached_cycle(self.runtime,self.authority,self.provider,self.plan_rows[cid],
                self.index[cid],self.weights,self.binding)
            self.assertEqual(self.provider.calls,[cid]);self.assertEqual(self.provider.evictions,[cid])
            self.assertFalse((self.provider.directory/'raw'/(cid+'.tar')).exists())
            restored,again,same=helper._cached_cycle(self.runtime,self.authority,self.provider,self.plan_rows[cid],
                self.index[cid],self.weights,self.weights_binding())
            self.assertEqual(same,commit);self.assertEqual(self.provider.calls,[cid]);self.assertEqual(self.loaded,[cid])
            self.assertEqual(restored.member_count,17);self.assertEqual(restored.records,cycle.records)
            self.assertEqual(restored.fields,cycle.fields);self.assertEqual(restored.availability,cycle.availability)
            np.testing.assert_array_equal(restored.values,cycle.values)
            np.testing.assert_array_equal(restored.present,cycle.present)
            for actual,expected in zip(again,reduced):np.testing.assert_array_equal(actual,expected)
            self.assertFalse(restored.values.flags.writeable)
            self.assertGreater(self.runtime.snapshot()['read_reserved'],0)
            self.assertEqual(self.runtime.snapshot()['entered_models'],0)

    def weights_binding(self):return dict(self.binding)

    def test_eviction_interruption_preserves_committed_cache_and_can_finish_without_redownload(self):
        cid=next(iter(self.index));self.provider.fail_eviction=True
        with self.runtime.writer(),patch.object(helper,'load_assessment_cycle',side_effect=self.load_cycle):
            with self.assertRaises(OSError):helper._cached_cycle(self.runtime,self.authority,self.provider,self.plan_rows[cid],
                self.index[cid],self.weights,self.binding)
            self.assertIsNotNone(self.runtime.reuse('CYCLE/'+cid))
            self.assertTrue((self.provider.directory/'raw'/(cid+'.tar')).exists())
            self.provider.fail_eviction=False
            helper._cached_cycle(self.runtime,self.authority,self.provider,self.plan_rows[cid],
                self.index[cid],self.weights,self.binding)
            self.assertEqual(self.loaded,[cid]);self.assertEqual(self.provider.calls,[cid]);self.assertEqual(self.provider.evictions,[cid])

    def test_no_eviction_if_cache_commit_does_not_complete(self):
        cid=next(iter(self.index))
        with self.runtime.writer(),patch.object(helper,'load_assessment_cycle',side_effect=self.load_cycle):
            with patch.object(self.runtime,'commit',side_effect=OSError('commit failed')):
                with self.assertRaises(OSError):helper._cached_cycle(self.runtime,self.authority,self.provider,self.plan_rows[cid],
                    self.index[cid],self.weights,self.binding)
            self.assertEqual(self.provider.evictions,[])
            self.assertTrue((self.provider.directory/'raw'/(cid+'.tar')).exists())

    def test_diagnosed_resume_uses_new_immutable_attempt_and_preserves_uncommitted_payloads(self):
        cid=next(iter(self.index))
        with self.assertRaises(OSError):
            with self.runtime.writer(),patch.object(helper,'load_assessment_cycle',side_effect=self.load_cycle):
                with patch.object(self.runtime,'commit',side_effect=OSError('synthetic fs interruption')):
                    helper._cached_cycle(self.runtime,self.authority,self.provider,self.plan_rows[cid],
                        self.index[cid],self.weights,self.binding)
        prior=set((self.runtime.directory/'artifacts/cycle-cache'/cid).glob('*'))
        self.assertEqual(len(prior),2);self.assertEqual(self.provider.evictions,[])
        with self.runtime.writer(),patch.object(helper,'load_assessment_cycle',side_effect=self.load_cycle):
            failed=self.runtime.snapshot()['failure']
            self.runtime.authorize_resume(dict(classification='INFRASTRUCTURE',failure_id=failed['id'],
                plan_sha256=self.runtime.plan['plan_sha256'],reason='Synthetic cache commit interruption',evidence_sha256=['d'*64]))
            helper._cached_cycle(self.runtime,self.authority,self.provider,self.plan_rows[cid],
                self.index[cid],self.weights,self.binding)
            after=set((self.runtime.directory/'artifacts/cycle-cache'/cid).glob('*'))
            self.assertEqual(len(after),4);self.assertTrue(prior<after)
            self.assertEqual(self.provider.evictions,[cid]);self.assertEqual(self.runtime.snapshot()['resume_id'],1)

    def test_changed_cycle_identity_or_corrupt_payload_fails_without_source_retry(self):
        cid=next(iter(self.index))
        with self.runtime.writer(),patch.object(helper,'load_assessment_cycle',side_effect=self.load_cycle):
            _,_,commit=helper._cached_cycle(self.runtime,self.authority,self.provider,self.plan_rows[cid],
                self.index[cid],self.weights,self.binding)
            with self.assertRaisesRegex(ValueError,'derivation'):
                helper._cached_cycle(self.runtime,self.authority,self.provider,self.plan_rows[cid],
                    dict(self.index[cid],changed=True),self.weights,self.binding)
            path=self.runtime.directory/commit['artifacts']['arrays']['path']
            raw=bytearray(path.read_bytes());raw[len(raw)//2]^=1;path.write_bytes(raw)
            with self.assertRaises(AssessmentIntegrityError):helper._cached_cycle(self.runtime,self.authority,self.provider,
                self.plan_rows[cid],self.index[cid],self.weights,self.binding)
            self.assertEqual(self.provider.calls,[cid])

    def test_reduced_cache_cannot_be_relabelled_as_full_or_renormalized(self):
        cid=next(iter(self.index));supplied=self.provider(cid,self.index[cid])
        cyc,receipt=self.load_cycle(supplied['spec'],self.index[cid],self.authority,read_budget=lambda n:None)
        means,masks,coverage=basin_reduce(cyc.values,cyc.present,self.weights)
        a=dict(values=cyc.values,present=cyc.present,means=means,masks=masks,coverage=coverage)
        b=helper._cycle_binding(self.binding,cid,self.index[cid])
        m=dict(schema=helper.CYCLE_SCHEMA,binding=b,arrays_sha256=array_identity(a),
            local_cycle=dict(reference=cyc.reference,records=cyc.records,member_count=cyc.member_count,
                fields=cyc.fields,availability=cyc.availability),loader_receipt=receipt,
            cache_key=digest(dict(b,archive_sha256=receipt['archive_sha256'],archive_bytes=receipt['archive_bytes'],
                                  original_manifest_sha256=receipt['manifest_sha256'])))
        for change in ('missing_raw','wrong_mean','weights','manifest'):
            aa=deepcopy(a);mm=deepcopy(m);weights=self.weights.copy()
            if change=='missing_raw':aa.pop('values')
            elif change=='wrong_mean':aa['means'][0,0]+=1
            elif change=='weights':weights[0,0]=.5
            else:mm['loader_receipt']['manifest_sha256']='e'*64
            mm['arrays_sha256']=array_identity(aa)
            with self.subTest(change=change),self.assertRaises(ValueError):helper._restore_cycle(aa,mm,b,weights)

    def test_two_chunks_reuse_overlapping_cycles_and_keep_compact_gauge_and_full_provenance(self):
        # One admissible-quality gap must reduce availability without removing
        # origins, changing other gauges, or introducing targets into bundles.
        obs=list(self.observations);row=next(i for i,o in enumerate(obs) if o.site==GAUGES[0]
            and o.valid_at==ASSESSMENT_START-2*HOUR);obs[row]=replace(obs[row],quality_flags=('P',))
        with self.runtime.writer(),patch.object(helper,'load_assessment_cycle',side_effect=self.load_cycle),redirect_stdout(io.StringIO()):
            usgs=self.usgs_commit()
            commits,pieces=helper._prepare_chunks(self.runtime,self.authority,self.provider,self.plan_rows,self.index,
                self.weights,obs,usgs,self.binding,self.calendar)
            self.assertEqual(len(commits),2);self.assertEqual(len(self.provider.calls),len(self.index))
            self.assertEqual(len(set(self.provider.calls)),len(self.index))
            self.assertEqual(len(self.loaded),len(self.index));self.assertEqual(len(self.provider.evictions),len(self.index))
            full=helper._arrays(self.runtime.read(commits[0]['artifacts']['arrays']))
            self.assertIn('source_metadata_json',full);self.assertIn('history_quality_flags',full)
            self.assertIn('history_source_ids',full);self.assertNotIn('target',full)
            self.assertEqual(json.loads(full['history_quality_flags'][0,-1]),['P'])
            self.assertFalse(full['input_admitted'][0]);self.assertTrue(full['input_admitted'][1:3].all())
            bundles=helper._gauge_bundles(self.runtime,commits,pieces,self.binding,self.calendar)
            for si,g in enumerate(GAUGES):
                a=helper._arrays(self.runtime.read(bundles[g]['arrays']))
                m=json.loads(self.runtime.read(bundles[g]['manifest']))
                self.assertEqual(set(a),set(helper.COMPACT_KEYS+helper.METADATA))
                self.assertEqual(len(a['origin_time']),240);np.testing.assert_array_equal(a['origin_time'],self.calendar)
                self.assertEqual(m['arrays_sha256'],array_identity(a))
                self.assertEqual(a['cohort_admission_digest'].item(),m['cohort_admission_digest'])
                self.assertEqual(len(m['full_provenance_chunks']),2)
                self.assertEqual(a['history'].shape,(240,168,2));self.assertFalse(m['target_values_read'])
                if si==0:self.assertLess(m['admitted_origins'],240)
                else:self.assertEqual(m['admitted_origins'],240)
            before=list(self.provider.calls)
            with patch.object(helper,'assemble_assessment_inputs',side_effect=AssertionError('Reassembled complete chunk')):
                again,newpieces=helper._prepare_chunks(self.runtime,self.authority,self.provider,self.plan_rows,self.index,
                    self.weights,obs,usgs,self.binding,self.calendar)
            self.assertEqual(again,commits);self.assertEqual(self.provider.calls,before)
            self.assertEqual(helper._gauge_bundles(self.runtime,again,newpieces,self.binding,self.calendar),bundles)
            progress=json.loads((self.runtime.directory/'progress.json').read_text())
            self.assertEqual(progress['completed_input_chunks'],2)
            self.assertEqual(progress['processed_candidate_cycles'],len(self.index))
            self.assertEqual(progress['source_download_observed_bytes'],self.provider.budget.bytes)
            self.assertFalse(progress['verification_targets_assembled'])

    def test_usgs_reader_is_called_only_for_exact_input_union_and_receipt_preserved(self):
        src=dict(observations=(),original_rows=(),metadata=({'month':'2024-07','sha256':'a'*64},),
                 mode='input',requested_windows=((helper.FIRST_INPUT,helper.LAST_INPUT),),
                 source_manifest_sha256=USGS_MANIFEST_SHA256,execution_plan_sha256=self.authority.execution_plan_sha256,
                 prediction_barrier_sha256=None,original_qualifiers_preserved=True,targets_imputed=False,source_files_modified=False)
        with self.runtime.writer(),patch.object(helper,'load_assessment_usgs',return_value=src) as reader:
            obs,commit=helper._usgs(self.runtime,self.authority,{},'/synthetic/manifest',self.binding)
            self.assertEqual(obs,());args,kwargs=reader.call_args
            self.assertEqual(args[2],[(helper.FIRST_INPUT,helper.LAST_INPUT)]);self.assertEqual(kwargs['mode'],'input')
            m=json.loads(self.runtime.read(commit['artifacts']['receipt']))
            self.assertNotIn('observations',m);self.assertNotIn('original_rows',m)
            self.assertTrue(m['original_qualifiers_preserved']);self.assertFalse(m['target_values_assembled'])
            _,again=helper._usgs(self.runtime,self.authority,{},'/synthetic/manifest',self.binding)
            self.assertEqual(again,commit)

    def test_disabled_release_wrong_source_or_changed_plan_rejects_before_source_access(self):
        for change in ('disabled','verify','source','plan','runtime'):
            authority=self.authority;stage=deepcopy(self.stage);source=SOURCE
            if change=='disabled':authority=replace(authority,inputs_enabled=False)
            elif change=='verify':authority=replace(authority,verification_enabled=True,prediction_barrier_sha256='f'*64)
            elif change=='source':source='f'*64
            elif change=='plan':stage['status']='NOT_APPROVED'
            else:authority=replace(authority,execution_plan_sha256='f'*64)
            with self.subTest(change=change),patch.object(helper,'load_assessment_usgs',side_effect=AssertionError('USGS read')):
                with self.assertRaises(ValueError):helper.prepare_inputs(self.runtime,authority,self.provider,stage,
                    self.index,{},'/no/manifest','/no/weights',source)
        self.assertEqual(self.provider.calls,[])

    def test_public_helper_requires_all1463_rows_before_numerical_read(self):
        with patch.object(helper,'load_assessment_usgs',side_effect=AssertionError('Numerical read')):
            with self.assertRaisesRegex(ValueError,'1463'):
                helper.prepare_inputs(self.runtime,self.authority,self.provider,self.stage,self.index,{},
                                      '/no/manifest',self.runtime.plan['weights']['path'],SOURCE)
        self.assertEqual(self.provider.calls,[])

    def test_public_helper_passes_exact_full_calendar_and_returns_controller_contract(self):
        calendar=helper.origins();ids=cycle_ids_for_origins(list(calendar))
        index={cid:dict(cycle=cid,source='synthetic') for cid in ids}
        stage=deepcopy(self.stage);stage['cycles']=[dict(cycle_id=cid,sealed_row_sha256=digest(row)) for cid,row in index.items()]
        stage['plan_sha256']=digest({k:v for k,v in stage.items() if k!='plan_sha256'})
        gate=replace(self.authority,cycle_row_hashes=tuple((cid,digest(row)) for cid,row in index.items()))
        self.provider.plan=stage
        expected=[{'synthetic_chunk':i} for i in range(73)]
        bundles={g:{'arrays':{'synthetic':True},'manifest':{'synthetic':True}} for g in GAUGES}
        with (self.runtime.writer(),
                patch('hydra_directflow.assessment_data.read_assessment_weights',return_value=(self.weights,{'synthetic':True})) as weight_read,
                patch.object(helper,'_usgs',return_value=((),{'synthetic':'usgs'})),
                patch.object(helper,'_prepare_chunks',return_value=(expected,{})) as chunker,
                patch.object(helper,'_gauge_bundles',return_value=bundles)):
            result=helper.prepare_inputs(self.runtime,gate,self.provider,stage,index,{},'/synthetic/manifest',
                self.runtime.plan['weights']['path'],SOURCE)
            self.assertEqual(result,dict(chunk_commits=expected,gauge_bundles=bundles,input_source_identity=SOURCE))
            self.assertEqual(chunker.call_args.args[-1],calendar);self.assertEqual(len(calendar),8760)
            self.assertEqual(len(ids),1463);self.assertEqual((len(calendar)+119)//120,73)
            self.assertEqual(weight_read.call_args.args[1],gate)
            self.assertEqual(self.provider.calls,[])

    def test_npz_object_duplicate_and_expansion_are_rejected(self):
        with self.assertRaises(ValueError):helper._arrays(helper._npz({'a':np.array([object()],dtype=object)}))
        with self.assertRaises(ValueError):helper._arrays(helper._npz({'a':np.ones(100)}),maximum=10)
        import zipfile
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w') as z:
            z.writestr('../a.npy',b'not-an-array')
        with self.assertRaises(ValueError):helper._arrays(buffer.getvalue())

    def test_cloud_only_stat_probe_never_opens_or_hydrates_candidate(self):
        p=self.root/'cloud-only';p.write_bytes(b'owned')
        info=p.lstat()
        class Cloud:
            st_mode=info.st_mode;st_size=info.st_size;st_flags=0x40000000
        with patch.object(Path,'lstat',return_value=Cloud()),patch.object(Path,'read_bytes',side_effect=AssertionError('Hydrated')):
            self.assertEqual(helper._resident_probe_size(p,1024),0)


if __name__=='__main__':unittest.main()
