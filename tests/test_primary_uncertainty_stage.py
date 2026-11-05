"""Synthetic stage authentication/orchestration; no study values or model calls."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow.post_uncertainty import make_draws, evaluate_matched
from hydra_directflow.reference import digest
import test_post_data as data_tests

PATH=Path(__file__).resolve().parents[1]/'scripts/postprocess_primary_uncertainty.py'
SPEC=importlib.util.spec_from_file_location('primary_uncertainty_stage_tested',PATH)
stage=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(stage)


class FakeGuard:
    def __init__(self,root):
        self.run=Path(root).resolve();self.run.mkdir(exist_ok=True)
        self.writes=[];self.reads=[];self.checks=0;self.saves=0
    def check(self):self.checks+=1
    def save(self):self.saves+=1
    def read(self,path,maximum):
        raw=Path(path).read_bytes();assert len(raw)<=maximum
        self.reads.append(Path(path));return raw
    def write(self,relative,raw):
        path=self.run/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():assert path.read_bytes()==raw
        else:path.write_bytes(raw)
        self.writes.append(relative)
        return dict(path=relative,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


class PreparedGroupTests(unittest.TestCase):
    def setUp(self):
        self.fixture=data_tests.PostDataTests('test_complete_calendar_fixed_cohort_and_independent_metrics')
        self.fixture.setUp()
        self.arrays,self.manifest,self.audit=self.fixture.build()
        self.guard=FakeGuard(self.fixture.root/'post')
        self.original=self.fixture.plan
        self.plan=dict(plan_sha256='9'*64,original_plan_sha256=self.original['plan_sha256'],
                       protocol_sha256=self.original['protocol_sha256'])
        self.name=f'{self.fixture.fold}__{self.fixture.gauge}'
        self.pack()
    def tearDown(self):self.fixture.tearDown()
    def pack(self):
        # Test fixture replacement is explicit, outside production immutable writes.
        folder=self.guard.run/'groups'/self.name;folder.mkdir(parents=True,exist_ok=True)
        stream=io.BytesIO();np.savez_compressed(stream,**self.arrays)
        payloads={'arrays':stream.getvalue(),'manifest':stage._json(self.manifest),'audit':stage._json(self.audit)}
        artifacts={}
        for kind,raw in payloads.items():
            suffix='npz' if kind=='arrays' else 'json';path=folder/f'{kind}.{suffix}';path.write_bytes(raw)
            artifacts[kind]=dict(path=str(path.relative_to(self.guard.run)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        self.record=dict(group=self.name,fold=self.fixture.fold,gauge_id=self.fixture.gauge,
                         plan_sha256=self.plan['plan_sha256'],artifacts=artifacts)
        (folder/'COMMIT.json').write_bytes(stage._json(self.record))
    def load(self):return stage._load_group(self.plan,self.original,self.guard,self.record)

    def test_authenticated_group_preserves_full_calendar_and_nan_unavailable_rows(self):
        arrays,manifest,audit=self.load()
        self.assertEqual(arrays['predictions'].shape,(8,720,18))
        np.testing.assert_array_equal(arrays['origins'],self.arrays['origins'])
        self.assertTrue(np.isnan(arrays['labels'][~arrays['comparison_mask']]).all())
        self.assertEqual(manifest,self.manifest);self.assertEqual(audit,self.audit)

    def test_changed_archive_sha_fails_before_zip_decode(self):
        path=self.guard.run/self.record['artifacts']['arrays']['path']
        raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
        with patch.object(stage.zipfile,'ZipFile',side_effect=AssertionError('Decoded unauthenticated archive')):
            with self.assertRaisesRegex(ValueError,'artifact changed'):self.load()

    def test_group_commit_and_manifest_identity_are_not_inferred_from_filename(self):
        for mode in ('commit','manifest'):
            original=deepcopy(self.manifest)
            if mode=='commit':
                path=self.guard.run/'groups'/self.name/'COMMIT.json';value=json.loads(path.read_text());value['gauge_id']='03479000';path.write_text(json.dumps(value))
            else:
                self.manifest['protocol_sha256']='0'*64
                self.manifest['content_sha256']=digest({k:v for k,v in self.manifest.items() if k!='content_sha256'});self.pack()
            with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'group commit|scientific scope'):
                self.load()
            self.manifest=original;self.pack()

    def test_forecast_clock_failure_precedes_outcome_decode(self):
        self.arrays={**self.arrays,'forecast_time':self.arrays['forecast_time']+3600,
                     'predictions':np.array([object()],dtype=object)}
        self.pack()
        with self.assertRaisesRegex(ValueError,'Prepared clocks'):self.load()

    def test_mask_or_typed_values_cannot_change_under_valid_archive_sha(self):
        self.arrays={**self.arrays,'labels':self.arrays['labels'].copy()}
        self.arrays['labels'][0,0]+=1.;self.pack()
        with self.assertRaisesRegex(ValueError,'typed array digest'):self.load()

    def test_fixed_real_qualification_helpers_match_scalar_reference_on_synthetic_group(self):
        arrays,manifest,audit=self.load()
        draws=make_draws(arrays['origins'],start_utc=manifest['calendar_start_utc'],
            end_exclusive_utc=manifest['calendar_end_exclusive_utc'],fold=manifest['fold'],block_hours=168,replicates=1)
        result=evaluate_matched(draws,labels=arrays['labels'],label_mask=arrays['label_mask'],
            predictions=arrays['predictions'],prediction_mask=np.broadcast_to(arrays['comparison_mask'][None,:,None],arrays['predictions'].shape),
            comparison_mask=arrays['comparison_mask'],expected_comparison_mask_sha256=manifest['comparison_mask_sha256'])
        checked=stage._baseline_check(result,audit);self.assertEqual(checked['panels'],8)
        reference=stage._reference_check(draws,arrays,manifest,result)
        self.assertEqual(reference['draw_index'],0);self.assertEqual(reference['panel_index'],0)
        self.assertTrue(reference['original_coordinate_support_matches'])
        altered=dict(result,baseline_primary=result['baseline_primary'].copy());altered['baseline_primary'][0]+=.1
        with self.assertRaisesRegex(ValueError,'baseline primary differs'):stage._baseline_check(altered,audit)
        altered=dict(result,primary=result['primary'].copy());altered['primary'][0,0]=100.
        with self.assertRaisesRegex(ValueError,'primary qualification failed'):
            stage._reference_check(draws,arrays,manifest,altered)


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.guard=FakeGuard(Path(self.temp.name)/'run')
        original=dict(protocol_sha256='b'*64,jobs=[]);original['plan_sha256']=digest(original)
        path=Path(self.temp.name)/'original.json';path.write_bytes(stage._json(original))
        self.plan=dict(plan_sha256='a'*64,original_plan_sha256=original['plan_sha256'],
            original_plan_path=str(path),protocol_sha256='b'*64,
            uncertainty=dict(draws=2000,master_seed=20261102,block_hours=[168,336,672],fold_order=list(stage.FOLDS),
                stream_mapping='SeedSequence(master).spawn(12); block-major/fold-minor; one uint64 child seed',
                same_draws_for_all_gauges_models_seeds=True))
        groups=[dict(fold=f,gauge_id=g) for f in stage.FOLDS for g in stage.GAUGES]
        audit=dict(status='ALL180_PREDICTIONS_INDEPENDENTLY_AUDITED',plan_sha256='a'*64,
            groups_total=12,groups=groups,model_calls=0,protected_assessment_access=False)
        (self.guard.run/'AUDIT.json').write_bytes(stage._json(audit));self.seen=[];self.draws_created=[]
    def tearDown(self):self.temp.cleanup()
    def group(self,plan,original,guard,record):
        fold,gauge=record['fold'],record['gauge_id'];panels=[]
        seeds=(17,29,43) if fold=='final_development' else (17,)
        arms=(('full',168),('q_only',168),('full',24)) if fold=='final_development' else (('full',168),('q_only',168))
        for seed in seeds:
            for a in stage.ARCHITECTURES:
                for arm,length in arms:panels.append(dict(job_id=f'{fold}-{gauge}-{a}-{arm}-{length}-{seed}',architecture=a,input_arm=arm,history_hours=length,seed=seed))
        origin=np.arange(2,dtype=np.int64)
        arrays=dict(origins=origin,labels=np.ones((2,18)),label_mask=np.ones((2,18),bool),
            predictions=np.ones((len(panels),2,18)),comparison_mask=np.ones(2,bool))
        manifest=dict(panels=panels,fold=fold,gauge_id=gauge,calendar_start_utc='synthetic',calendar_end_exclusive_utc='synthetic',comparison_mask_sha256='c'*64)
        return arrays,manifest,{'synthetic':True}
    def draw(self,origins,**kwargs):
        value=SimpleNamespace(origins=origins,receipt={'synthetic':True,**kwargs},key=(kwargs['block_hours'],kwargs['fold']))
        self.draws_created.append(value);return value
    def evaluate(self,draws,**kwargs):
        panels=len(kwargs['predictions']);self.seen.append((draws.key,id(draws)))
        return dict(primary=np.full((2000,panels),.5),baseline_primary=np.full(panels,.5),
            per_lead_nse=np.full((2000,panels,18),.5),baseline_per_lead_nse=np.full((panels,18),.5))
    def invoke(self,reference=None):
        with patch.object(stage,'_load_group',side_effect=self.group),patch.object(stage,'make_draws',side_effect=self.draw),\
             patch.object(stage,'evaluate_matched',side_effect=self.evaluate),patch.object(stage,'_baseline_check',return_value={'passed':True}),\
             patch.object(stage,'_reference_check',side_effect=reference or (lambda *a:{'passed':True})):
            return stage.run_uncertainty(self.plan,self.guard)

    def test_one_draw_bundle_per_fold_block_reused_all_gauges_and_all_families_written(self):
        result=self.invoke()
        self.assertEqual(len(self.draws_created),12);self.assertEqual(len(self.seen),36)
        for i in range(0,36,3):self.assertEqual(len({x[1] for x in self.seen[i:i+3]}),1)
        self.assertEqual(result['fixed_scalar_reference_checks'],12)
        self.assertEqual([b['block_hours'] for b in result['blocks']],[168,336,672])
        for b in result['blocks']:
            for mode,count in (('cv',30),('final',42)):
                f=b['families'][mode];self.assertEqual(f['contrasts'],count);self.assertEqual(f['jointly_valid_draws'],2000)
                report=json.loads((self.guard.run/f['report']['path']).read_text())
                self.assertEqual(report['status'],'AVAILABLE_CONDITIONAL_POINTWISE')
                self.assertTrue(report['per_lead_mean_nse']['same_joint_family_valid_draw_pool'])
        self.assertIn('UNCERTAINTY.json',self.guard.writes)
        # Repeating this pure synthetic stage verifies deterministic immutable bytes.
        self.assertEqual(self.invoke(),result)

    def test_reference_failure_blocks_family_interval_publication(self):
        calls=[]
        def fail_last_fold(*args):
            calls.append(1)
            if len(calls)==4:raise ValueError('synthetic scalar qualification mismatch')
            return {'passed':True}
        with self.assertRaisesRegex(ValueError,'scalar qualification mismatch'):self.invoke(fail_last_fold)
        self.assertFalse(any(x.endswith('/cv.json') or x.endswith('/final.json') or x=='UNCERTAINTY.json' for x in self.guard.writes))

    def test_uncertainty_settings_or_group_inventory_cannot_silently_change(self):
        self.plan['uncertainty']['draws']=100
        with self.assertRaisesRegex(ValueError,'execution policy'):self.invoke()
        self.plan['uncertainty']['draws']=2000
        path=self.guard.run/'AUDIT.json';a=json.loads(path.read_text());a['groups'][-1]=a['groups'][0];path.write_bytes(stage._json(a))
        with self.assertRaisesRegex(ValueError,'All12 unique'):self.invoke()


if __name__=='__main__':unittest.main()
