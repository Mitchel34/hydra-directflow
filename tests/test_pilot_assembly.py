"""Synthetic provenance joins and immutable assembly; no models/fits/network."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import shutil

import numpy as np

from hydra_directflow import corpus, corpus_dataset as ds, pilot_assembly as assembly
from hydra_directflow.cache import DerivedCache, pack_month
from hydra_directflow.local_data import GAUGES, USGS_MANIFEST_SHA256, iso
from hydra_directflow.quality import PRIMARY_APPROVED
from hydra_directflow.reference import Observation, digest
from test_corpus_dataset import arrays_for

PYROOT = 'hydra_directflow.pilot_assembly.'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def value(t): return 10.+(int(t)//3600 % 500)/10.


class PilotAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.corpus=self.root/'corpus'; self.corpus.mkdir(); (self.corpus/'chunks').mkdir(); (self.corpus/'staging').mkdir()
        splits=[dict(name='split_'+str(i),start_utc=f'2021-08-{23+i:02d}T02:00:00Z',end_utc=f'2021-08-{24+i:02d}T02:00:00Z') for i in range(3)]
        units=[u for split in splits for u in corpus.partition_split(split,history_hours=168,chunk_hours=2)]
        core=dict(schema=ds.SCHEMA,splits=splits,history_hours=168,chunk_hours=2,input_quality_policy=PRIMARY_APPROVED,
            target_quality_policy=PRIMARY_APPROVED,qualified_index_sha256=ds.INDEX_SHA256,usgs_manifest_sha256=USGS_MANIFEST_SHA256,
            candidate_weights_sha256=ds.WEIGHTS_SHA256,units=units,required_cycle_ids=sorted({c for u in units for x in u['chunks'] for c in x['cycle_ids']}),
            calendar_unavailable_cycle_ids=[],implementation_sha256={'synthetic':'a'*64})
        self.sourceplan=dict(**core,plan_sha256=digest(core),requested_origins=72,candidate_origins=18,purged_origins=54)
        (self.corpus/'PLAN.json').write_text(json.dumps(self.sourceplan))
        self.roles={name:dict(start_utc=splits[i]['start_utc'],end_utc=splits[i]['end_utc']) for i,name in enumerate(assembly.ROLES)}
        self.refit=self.roles['inner_stop']['end_utc']; self.cache=self.root/'cache'
        start=ds._hour('2021-08-01T00:00:00Z'); end=ds._hour('2021-09-01T00:00:00Z')
        self.observations=[Observation(g,t,None,value(t),('A','R','A') if (t//3600)%2 else ('A',),'usgs-wire-sha256:'+'c'*64)
                           for t in range(start,end,3600) for g in GAUGES]
        self.cache_identity=dict(month='2021-08',source_manifest_sha256=USGS_MANIFEST_SHA256,
                                 source_partition={'bytes':123,'sha256':'b'*64},implementation_sha256={'fixture':'a'*64})
        with DerivedCache(self.cache).locked() as cache:
            _,_,self.key,_=cache.get_or_build('usgs_month',self.cache_identity,lambda:pack_month('2021-08',self.observations,{'fixture':True}))
        self.entry=self.cache/'entries'/self.key
        self.month_sources={'2021-08':dict(entry_path=str(self.entry),cache_key=self.key,commit_sha256=sha(self.entry/'COMMIT.json'))}
        expected={g:{role:[] for role in assembly.ROLES} for g in GAUGES}
        for ui,unit in enumerate(units):
            for chunk in unit['chunks']:
                arrays=arrays_for(chunk)
                arrays['history'][...,0]=np.array([[value(t) for t in row] for row in arrays['history_time']])
                arrays['target'][:]=np.array([[value(t) for t in row] for row in arrays['target_time']])
                corpus._shard_write(self.corpus,self.sourceplan['plan_sha256'],unit,chunk,arrays,
                    dict(case_source_digest=digest([ui,chunk['index']]),cache={'cycle_keys':{c:digest(c) for c in chunk['cycle_ids']},'month_keys':{'2021-08':self.key}},stage_seconds={'total':.1}))
                for g,t,case in zip(arrays['site_id'],arrays['origin_time'],arrays['case_id']):
                    role=assembly._selected_role(int(t),assembly._roles(self.roles),ds._hour(self.refit))
                    if role: expected[GAUGES[int(g)]][role].append(str(case))
        self.expected={g:{r:dict(cases=len(ids),case_ids_sha256=digest(ids)) for r,ids in rows.items()} for g,rows in expected.items()}
        snapshot=ds.inspect_corpus(self.corpus,expected_plan_sha256=self.sourceplan['plan_sha256'])
        self.receipt=self.root/'integrity.json'; self.receipt.write_text(json.dumps(snapshot.report()))
        self.kwargs=dict(corpus_root=self.corpus,integrity_receipt_path=self.receipt,expected_integrity_sha256=sha(self.receipt),
            expected_plan_sha256=self.sourceplan['plan_sha256'],expected_snapshot_sha256=snapshot.snapshot_sha256,
            month_sources=self.month_sources,role_time_bounds=self.roles,refit_at_utc=self.refit,expected_roles=self.expected)

    def tearDown(self): self.tmp.cleanup()
    def plan(self,**changes): return assembly.plan_metadata(**{**self.kwargs,**changes})
    def run_assembly(self):
        self.output=self.root/'assembled'; return assembly.execute(self.plan(),output_directory=self.output)

    def test_metadata_plan_never_loads_numeric_payloads(self):
        with patch(PYROOT+'ds._load_shard',side_effect=AssertionError('numeric')),patch(PYROOT+'unpack_month',side_effect=AssertionError('numeric')):
            plan=self.plan()
        self.assertEqual(plan['status'],'READY_LOCAL_ONLY'); self.assertEqual(len(plan['shards']),9)
        self.assertFalse(plan['scientific_fits_authorized'])

    def test_missing_or_dataless_provenance_blocks_without_open(self):
        plan=self.plan(month_sources={}); self.assertEqual(plan['status'],'BLOCKED_MISSING_RESIDENT_PROVENANCE')
        with self.assertRaisesRegex(ValueError,'incomplete'): assembly.execute(plan,output_directory=self.root/'out')
        original=assembly._resident
        with patch(PYROOT+'_resident',side_effect=lambda p:False if str(p).endswith('.npz') and self.key in str(p) else original(p)):
            plan=self.plan()
        self.assertEqual(plan['status'],'BLOCKED_MISSING_RESIDENT_PROVENANCE')

    def test_pins_duplicate_snapshot_and_protected_roles_reject(self):
        wrong=deepcopy(self.month_sources); wrong['2021-08']['commit_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'lineage'): self.plan(month_sources=wrong)
        roles=deepcopy(self.roles); roles['outer']['end_utc']='2024-08-01T00:00:00Z'
        with self.assertRaisesRegex(ValueError,'protected'): self.plan(role_time_bounds=roles)
        data=json.loads(self.receipt.read_text()); data['observed_commits'][1]=data['observed_commits'][0]
        self.receipt.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'Duplicate'): self.plan(expected_integrity_sha256=sha(self.receipt))

    def test_exact_join_preserves_original_flags_source_ids_and_tensors(self):
        result=self.run_assembly(); self.assertEqual(result['manifest_sha256'],sha(self.output/'MANIFEST.json'))
        bundle=assembly.load_training_bundle(self.output,expected_manifest_sha256=result['manifest_sha256'],gauge_id=GAUGES[0])
        self.assertEqual(len(bundle['train_indices']),6); self.assertEqual(len(bundle['inner_indices']),5)
        self.assertEqual(len(bundle['arrays']['case_id']),11)
        flags=bundle['history_provenance']['quality_flags'][0]
        self.assertIn(('A','R','A'),flags)
        self.assertEqual(len(set(bundle['history_provenance']['source_ids'][0])),1)
        np.testing.assert_equal(bundle['arrays']['history'][0,:,0],[value(t) for t in bundle['arrays']['history_time'][0]])
        self.assertTrue(all(not a.flags.writeable for a in bundle['arrays'].values()))
        self.assertNotIn(str(self.root),json.dumps(result['manifest']))
        self.assertNotIn('usgs-wire',json.dumps(result['manifest']))

    def test_outer_input_stream_never_opens_labels(self):
        result=self.run_assembly(); original=assembly._checked
        def guard(path,*a,**k):
            if Path(path).name=='labels.npz': raise AssertionError('outer labels read')
            return original(path,*a,**k)
        with patch(PYROOT+'_checked',side_effect=guard):
            batches=list(assembly.iter_outer_inputs(self.output,expected_manifest_sha256=result['manifest_sha256'],gauge_id=GAUGES[1],batch_size=1))
        self.assertEqual(len(batches),6); self.assertTrue(all('target' not in b['arrays'] and 'target_mask' not in b['arrays'] for b in batches))
        ids=[b['arrays']['case_id'][0] for b in batches]
        labels=assembly.load_outer_labels(self.output,expected_manifest_sha256=result['manifest_sha256'],gauge_id=GAUGES[1],case_ids=ids[:2])
        np.testing.assert_equal(labels['case_id'],ids[:2])
        with self.assertRaisesRegex(ValueError,'absent'):
            assembly.load_outer_labels(self.output,expected_manifest_sha256=result['manifest_sha256'],gauge_id=GAUGES[1],case_ids=['03164000:2021-08-23T02:00:00Z'])

    def test_corpus_value_and_mask_mismatch_fail_closed(self):
        original=ds._load_shard
        for field in ('history','target','history_mask','target_mask'):
            with self.subTest(field=field):
                def altered(*a,**k):
                    arrays,summary,sha=original(*a,**k)
                    if field.endswith('_mask'): arrays[field].flat[0]=False
                    else: arrays[field].flat[0]+=1
                    return arrays,summary,sha
                with patch(PYROOT+'ds._load_shard',side_effect=altered):
                    with self.assertRaisesRegex(ValueError,'mismatch'):
                        assembly.execute(self.plan(),output_directory=self.root/field)

    def test_rejected_quality_reason_is_not_transport_absence(self):
        arrays,meta=pack_month('2021-08',self.observations,{})
        arrays['quality_index'][0,0]=len(meta['qualities']); meta['qualities'].append(['P'])
        stores={'2021-08':(arrays,meta)}
        self.assertEqual(assembly._row(stores,0,int(arrays['times'][0]))[1],'QUALITY_NOT_ADMITTED')
        arrays['row_present'][0,1]=False; arrays['numeric_valid'][0,1]=False
        self.assertEqual(assembly._row(stores,0,int(arrays['times'][1]))[1],'MISSING_EXACT_HOUR')
        with self.assertRaisesRegex(ValueError,'transport absence'): assembly._row({},0,int(arrays['times'][0]))

    def test_ridge_series_only_gradient_unique_hours_and_no_fit(self):
        result=self.run_assembly(); ridge=assembly.load_ridge_series(self.output,expected_manifest_sha256=result['manifest_sha256'],gauge_id=GAUGES[0])
        a=ridge['arrays']; meta=ridge['metadata']
        self.assertEqual(len(a['observation_times']),24); self.assertEqual(meta['eligible_pairs'],17)
        self.assertTrue(np.all(a['observation_times']<ds._hour(self.roles['fit']['end_utc'])))
        self.assertEqual(meta['ridge_fits'],0); self.assertEqual(np.diff(a['observation_times']).tolist(),[3600]*23)
        np.testing.assert_equal(a['eligible_pair_stop_indices'],np.arange(7,24))

    def test_expected_membership_mismatch_never_completes_manifest(self):
        expected=deepcopy(self.expected); expected[GAUGES[0]]['fit']['cases']+=1
        with self.assertRaisesRegex(ValueError,'membership'):
            assembly.execute(self.plan(expected_roles=expected),output_directory=self.root/'badcounts')
        self.assertFalse((self.root/'badcounts'/'MANIFEST.json').exists())

    def test_resume_is_identical_no_overwrite_and_corruption_rejected(self):
        first=self.run_assembly(); old=(self.output/'MANIFEST.json').read_bytes()
        second=assembly.execute(self.plan(),output_directory=self.output)
        self.assertEqual(first['manifest_sha256'],second['manifest_sha256']); self.assertEqual(second['write_bytes'],0)
        self.assertEqual(old,(self.output/'MANIFEST.json').read_bytes())
        part=first['manifest']['parts'][0]; target=self.output/part['relative']/'inputs.npz'; target.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'identity'):
            assembly.execute(self.plan(),output_directory=self.output)
        self.assertEqual(target.read_bytes(),b'corrupt')

    def test_bounds_and_external_stop_abort_before_completion(self):
        with self.assertRaisesRegex(ValueError,'allowance'): self.plan(max_read_bytes=1)
        plan=self.plan(max_write_bytes=1)
        with self.assertRaisesRegex(ValueError,'allowance'): assembly.execute(plan,output_directory=self.root/'small')
        with self.assertRaisesRegex(RuntimeError,'stop'):
            assembly.execute(self.plan(),output_directory=self.root/'stopped',check=lambda:(_ for _ in ()).throw(RuntimeError('stop')))
        result=self.run_assembly()
        with self.assertRaisesRegex(ValueError,'allocation'):
            assembly.load_training_bundle(self.output,expected_manifest_sha256=result['manifest_sha256'],gauge_id=GAUGES[0],max_bytes=1)

    def test_no_model_imports_or_existing_source_writes(self):
        before={str(p):sha(p) for base in (self.corpus,self.cache) for p in base.rglob('*') if p.is_file()}
        self.run_assembly()
        self.assertEqual(before,{str(p):sha(p) for base in (self.corpus,self.cache) for p in base.rglob('*') if p.is_file()})
        self.assertEqual(assembly.execute.__module__,'hydra_directflow.pilot_assembly')

    def test_oversized_pinned_payload_rejects_before_open(self):
        path=self.root/'oversize'; path.write_bytes(b'12345')
        with patch('pathlib.Path.open',side_effect=AssertionError('body opened')):
            with self.assertRaisesRegex(ValueError,'byte bound'):
                assembly._checked(path,dict(bytes=3,sha256='a'*64),100)

    def test_unused_shard_tail_month_does_not_expand_source_scope(self):
        unit={'id':'candidate_training__2021-08__L168'}
        chunk=dict(index=0,first_origin_utc='2021-08-31T00:00:00Z',last_origin_utc='2021-08-31T23:00:00Z',origin_count=24)
        relative=unit['id']+'/000'; directory=self.corpus/'chunks'/relative; directory.mkdir(parents=True)
        summary=dict(input_quality_policy=PRIMARY_APPROVED,target_quality_policy=PRIMARY_APPROVED,
                     month_keys={'2021-08':self.key,'2021-09':'d'*64})
        summary_raw=json.dumps(summary).encode(); (directory/'case-index.json').write_bytes(summary_raw)
        (directory/'model.npz').write_bytes(b'not opened')
        commit=dict(plan_sha256=self.sourceplan['plan_sha256'],unit_id=unit['id'],chunk_index=0,
                    payloads={'case-index.json':assembly._identity(summary_raw),'model.npz':assembly._identity(b'not opened')})
        (directory/'COMMIT.json').write_text(json.dumps(commit))
        observed=[dict(chunk=relative,commit_sha256=sha(directory/'COMMIT.json'))]
        plan_file_sha=sha(self.corpus/'PLAN.json')
        snapshot=digest(dict(plan_sha256=self.sourceplan['plan_sha256'],plan_file_sha256=plan_file_sha,observed=observed,missing=[]))
        receipt=dict(status='COMPLETE_COMMITTED_CORPUS_SNAPSHOT',missing_chunks=[],stored_history_hours=168,
            plan_sha256=self.sourceplan['plan_sha256'],plan_file_sha256=plan_file_sha,snapshot_sha256=snapshot,
            numerical_payloads_checked=True,protected_assessment_access=False,observed_commits=observed)
        self.receipt.write_text(json.dumps(receipt))
        roles=dict(fit={'start_utc':'2021-08-29T00:00:00Z','end_utc':'2021-08-30T00:00:00Z'},
            inner_stop={'start_utc':'2021-08-30T00:00:00Z','end_utc':'2021-08-31T00:00:00Z'},
            outer={'start_utc':'2021-08-31T00:00:00Z','end_utc':'2021-09-01T00:00:00Z'})
        with patch(PYROOT+'ds._plan',return_value=(self.sourceplan,[(unit,chunk)],plan_file_sha)):
            plan=self.plan(expected_integrity_sha256=sha(self.receipt),expected_snapshot_sha256=snapshot,
                           role_time_bounds=roles,refit_at_utc='2021-08-31T00:00:00Z')
        self.assertEqual(plan['status'],'READY_LOCAL_ONLY'); self.assertEqual(set(plan['months']),{'2021-08'})
        self.assertIn('2021-09',plan['shards'][0]['month_keys'])

    def test_prepared_bad_clock_rejects_before_feature_member(self):
        result=self.run_assembly(); part=result['manifest']['parts'][0]
        arrays=assembly._load_npz(self.output,part,'inputs.npz'); arrays['target_time'][0,-1]=ds.ASSESSMENT_START
        class PoisonArchive:
            files=list(arrays)
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def __getitem__(self,key):
                if key not in ('origin_time','target_time','site_id','case_id'): raise AssertionError('feature read')
                return arrays[key]
        with patch(PYROOT+'np.load',return_value=PoisonArchive()):
            with self.assertRaisesRegex(ValueError,'coordinates'):
                assembly._load_npz(self.output,part,'inputs.npz')

    def test_coverage_retains_non_q_missing_case_without_changing_common_cohort(self):
        unit=self.sourceplan['units'][2]; chunk=unit['chunks'][0]; arrays=arrays_for(chunk)
        arrays['history'][...,0]=np.array([[value(t) for t in row] for row in arrays['history_time']])
        arrays['target'][:]=np.array([[value(t) for t in row] for row in arrays['target_time']])
        arrays['weather'][0]=np.nan; arrays['weather_mask'][0]=False; arrays['source_cycle'][0]=''
        arrays['weather_source_leads'][0]=-1; arrays['eligibility_reason'][0]='NO_COMPLETE_SINGLE_WEATHER_RUN'
        directory=self.corpus/'chunks'/unit['id']/f"{chunk['index']:03d}"; shutil.rmtree(directory)
        corpus._shard_write(self.corpus,self.sourceplan['plan_sha256'],unit,chunk,arrays,
            dict(case_source_digest=digest('missingweather'),cache={'cycle_keys':{c:digest(c) for c in chunk['cycle_ids']},'month_keys':{'2021-08':self.key}},stage_seconds={'total':.1}))
        snapshot=ds.inspect_corpus(self.corpus,expected_plan_sha256=self.sourceplan['plan_sha256'])
        self.receipt.write_text(json.dumps(snapshot.report())); self.kwargs.update(expected_integrity_sha256=sha(self.receipt),expected_snapshot_sha256=snapshot.snapshot_sha256)
        first=ds._hour(chunk['first_origin_utc']); ids=[GAUGES[0]+':'+iso(t) for t in range(first+3600,first+6*3600,3600)]
        self.expected[GAUGES[0]]['outer']=dict(cases=5,case_ids_sha256=digest(ids))
        result=self.run_assembly(); self.assertEqual(result['manifest']['coverage_outer'][GAUGES[0]]['cases'],6)
        strict=list(assembly.iter_outer_inputs(self.output,expected_manifest_sha256=result['manifest_sha256'],gauge_id=GAUGES[0]))
        self.assertEqual(sum(len(x['arrays']['case_id']) for x in strict),5)
        original=assembly._checked
        def no_labels(path,*args,**kwargs):
            if Path(path).name=='labels.npz': raise AssertionError('label opened')
            return original(path,*args,**kwargs)
        with patch(PYROOT+'_checked',side_effect=no_labels):
            candidates=list(assembly.iter_coverage_inputs(self.output,expected_manifest_sha256=result['manifest_sha256'],gauge_id=GAUGES[0]))
        self.assertEqual(sum(len(x['arrays']['case_id']) for x in candidates),6)
        self.assertFalse(candidates[0]['arrays']['weather_mask'][0].any()); self.assertTrue(np.isnan(candidates[0]['arrays']['weather'][0]).all())
        self.assertNotEqual(candidates[0]['cohort_admission_digest'],strict[0]['cohort_admission_digest'])
        labels=assembly.load_coverage_labels(self.output,expected_manifest_sha256=result['manifest_sha256'],gauge_id=GAUGES[0],case_ids=[GAUGES[0]+':'+iso(first)])
        self.assertTrue(labels['target_mask'].all())

    def test_gradient_shape_census_uses_original_fit_only_and_start_bands(self):
        cases=[([], 'clean'),([0], 'leading'),([2,3,4], 'long_or_many'),([166], 'nonrepresentable'),
               ([2,5], 'nonrepresentable'),([167], 'trailing_1'),([166,167], 'trailing_2'),
               ([55,56], 'interior_2_early'),([56], 'interior_1_middle'),([111,167], 'mixed_recent')]
        for missing,expected in cases:
            mask=np.ones(168,bool); mask[missing]=False
            self.assertEqual(assembly.original_gap_shape(mask),expected)
        result=self.run_assembly()
        for row in result['manifest']['original_gradient_gap_shape_census'].values():
            self.assertEqual(row['candidate_windows'],6); self.assertEqual(row['counts'],{'clean':6})
            self.assertFalse(row['recipe_adopted']); self.assertFalse(row['includes_inner_or_outer'])

    def test_resume_rejects_internally_rehashed_payload_before_manifest(self):
        result=self.run_assembly(); (self.output/'MANIFEST.json').unlink()
        part=result['manifest']['parts'][0]; directory=self.output/part['relative']
        path=directory/'inputs.npz'; values=assembly._load_npz(self.output,part,'inputs.npz')
        values['history'][0,0,0]+=1; raw=assembly._npz(values); path.write_bytes(raw)
        commit=json.loads((directory/'COMMIT.json').read_text()); commit['payloads']['inputs.npz']=assembly._identity(raw)
        (directory/'COMMIT.json').write_bytes(assembly._json(commit))
        with self.assertRaisesRegex(ValueError,'recomputed'):
            assembly.execute(self.plan(),output_directory=self.output)
        self.assertFalse((self.output/'MANIFEST.json').exists())


if __name__=='__main__': unittest.main()
