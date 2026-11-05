"""Synthetic residual diagnostics only; no study values, source reads or fits."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow.reference import ASSESSMENT_START

PATH = Path(__file__).resolve().parents[1]/'scripts/postprocess_primary_residuals.py'
SPEC = importlib.util.spec_from_file_location('tested_primary_residual_stage', PATH)
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)


def iso(stamp):return datetime.fromtimestamp(int(stamp), timezone.utc).isoformat()


class FakeGuard:
    def __init__(self, run):self.run=run;self.reads=[];self.writes=[];self.checks=0
    def check(self):self.checks+=1
    def save(self):pass
    def read(self, path, maximum):
        self.reads.append(str(path));raw=Path(path).read_bytes()
        assert len(raw)<=maximum
        return raw
    def write(self, relative, raw):
        self.writes.append(relative);path=self.run/relative;path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():assert path.read_bytes()==raw
        else:path.write_bytes(raw)
        return dict(path=relative,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def group_fixture(fold='cv_1', gauge='03161000', *, n=48):
    start=int(datetime(2021,1,1,tzinfo=timezone.utc).timestamp())
    origins=start+np.arange(n,dtype=np.int64)*3600
    future=origins[:,None]+np.arange(1,19,dtype=np.int64)[None,:]*3600
    common=np.ones(n,dtype=bool);common[7:10]=False
    labels=np.full((n,18),10.,dtype=np.float64);labels[~common]=np.nan
    models=('LSTM','Transformer','GRU_Transformer','Mamba2_Transformer')
    arms=(('full',168),('q_only',168),('full',24)) if fold=='final_development' else (('full',168),('q_only',168))
    seeds=(17,29,43) if fold=='final_development' else (17,)
    panels=[]
    for model in models:
        for arm,length in arms:
            for seed in seeds:
                panels.append(dict(job_id=f'{fold}__{gauge}__{model}__{arm}_L{length}__seed{seed}',
                                   architecture=model,input_arm=arm,history_hours=length,seed=seed,device='synthetic'))
    prediction=np.stack([labels+(i+1)*np.arange(n,dtype=np.float64)[:,None]/10. for i in range(len(panels))])
    arrays=dict(origins=origins,forecast_time=future,labels=labels,
                label_mask=np.broadcast_to(common[:,None],(n,18)).copy(),comparison_mask=common,
                predictions=prediction,panel_ids=np.array([x['job_id'] for x in panels]))
    manifest=dict(fold=fold,gauge_id=gauge,calendar_start_utc=iso(start),calendar_end_exclusive_utc=iso(start+n*3600),
                  panels=panels,group_arrays_sha256='f'*64)
    return arrays,manifest


class ResidualMetricTests(unittest.TestCase):
    def test_perfect_linear_and_negative_correlations(self):
        linear=np.arange(30,dtype=np.float64);mask=np.ones(30,dtype=bool)
        result=stage.pairwise_lag_correlation(linear,mask,6)
        self.assertEqual(result['paired_original_hours'],24)
        self.assertAlmostEqual(result['correlation'],1.)
        alternating=np.tile(np.array([-1.,1.]),15)
        self.assertAlmostEqual(stage.pairwise_lag_correlation(alternating,mask,1)['correlation'],-1.)

    def test_pairwise_centering_matches_independent_correlation_calculation(self):
        values=np.array([2.,8.,4.,-3.,7.,1.,-6.,9.,2.])
        mask=np.array([True,True,False,True,True,True,False,True,True])
        selected=mask[:-1]&mask[1:]
        expected=float(np.corrcoef(values[:-1][selected],values[1:][selected])[0,1])
        actual=stage.pairwise_lag_correlation(values,mask,1)
        self.assertEqual(actual['paired_original_hours'],int(selected.sum()))
        self.assertAlmostEqual(actual['correlation'],expected)

    def test_missing_hour_does_not_create_false_adjacent_pairs(self):
        values=np.array([1.,np.nan,4.,8.]);mask=np.array([True,False,True,True])
        actual=stage.pairwise_lag_correlation(values,mask,1)
        self.assertEqual(actual['paired_original_hours'],1)
        self.assertIsNone(actual['correlation'])
        self.assertEqual(actual['reason'],'FEWER_THAN_TWO_PAIRS')
        self.assertEqual(stage.pairwise_lag_correlation(values,mask,6)['reason'],'ZERO_PAIRS')

    def test_constant_endpoints_are_explicitly_undefined(self):
        cases=[(np.ones(5),'BOTH_ENDPOINTS_CONSTANT'),(np.array([2.,2.,2.,3.]),'EARLIER_ENDPOINT_CONSTANT'),
               (np.array([3.,2.,2.,2.]),'LATER_ENDPOINT_CONSTANT')]
        for values,reason in cases:
            with self.subTest(reason=reason):
                result=stage.pairwise_lag_correlation(values,np.ones(len(values),bool),1)
                self.assertIsNone(result['correlation']);self.assertEqual(result['reason'],reason)

    def test_invalid_admitted_value_or_unplanned_lag_rejected(self):
        with self.assertRaisesRegex(ValueError,'Admitted residuals'):
            stage.pairwise_lag_correlation(np.array([1.,np.nan,3.]),np.ones(3,bool),1)
        with self.assertRaisesRegex(ValueError,'declared residual lag'):
            stage.pairwise_lag_correlation(np.arange(10,dtype=np.float64),np.ones(10,bool),2)

    def test_group_retains_every_model_seed_lead_lag_without_selection(self):
        arrays,manifest=group_fixture('final_development')
        result=stage.diagnose_group(arrays,manifest)
        self.assertEqual(len(result['panels']),36)
        self.assertEqual({x['panel']['seed'] for x in result['panels']},{17,29,43})
        self.assertEqual(result['lag_hours'],[1,6,12,24,48,72,168,336,672])
        self.assertEqual(len(result['panels'][0]['leads']),18)
        self.assertTrue(all(len(x['lags'])==9 for x in result['panels'][0]['leads']))
        self.assertFalse(result['block_length_selected'])
        self.assertFalse(result['significance_bands_computed'])
        self.assertFalse(result['paired_hours_are_independent_samples'])
        self.assertEqual(result['max_absolute_by_lag'][0]['defined_series'],36*18)

    def test_compacted_calendar_or_protected_clock_fails_before_outcomes(self):
        arrays,manifest=group_fixture()
        compacted={**arrays,'origins':np.delete(arrays['origins'],4)}
        with self.assertRaisesRegex(ValueError,'Complete hourly'):
            stage.diagnose_group(compacted,manifest)
        class ExplodingOutcome:
            def __array__(self,*args,**kwargs):raise AssertionError('Outcomes inspected')
        start=ASSESSMENT_START-20*3600;n=len(arrays['origins'])
        arrays['origins']=start+np.arange(n,dtype=np.int64)*3600
        arrays['forecast_time']=arrays['origins'][:,None]+np.arange(1,19)[None,:]*3600
        arrays['labels']=ExplodingOutcome()
        manifest.update(calendar_start_utc=iso(start),calendar_end_exclusive_utc=iso(start+n*3600))
        with self.assertRaisesRegex(ValueError,'Protected or crossing'):
            stage.diagnose_group(arrays,manifest)

    def test_filled_unavailable_calendar_or_missing_admitted_prediction_refused(self):
        for condition in ('filled','missing'):
            arrays,manifest=group_fixture()
            if condition=='filled':arrays['predictions'][:,~arrays['comparison_mask']]=0.
            else:arrays['predictions'][0,0,0]=np.nan
            with self.subTest(condition=condition),self.assertRaisesRegex(ValueError,'unavailable calendar gaps'):
                stage.diagnose_group(arrays,manifest)


class ResidualStageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();self.run=self.root/'postprocessing';self.run.mkdir()
        self.guard=FakeGuard(self.run)
        self.plan=dict(run_directory=str(self.run),plan_sha256='a'*64,original_plan_sha256='b'*64,
                       protocol_sha256='c'*64,protected_assessment_access=False)
        self.records=[]
        for fold in stage.FOLDS:
            for gauge in stage.GAUGES:
                name=f'{fold}__{gauge}'
                record=dict(group=name,fold=fold,gauge_id=gauge,plan_sha256=self.plan['plan_sha256'],artifacts={})
                directory=self.run/'groups'/name;directory.mkdir(parents=True)
                (directory/'COMMIT.json').write_text(json.dumps(record));self.records.append(record)
        self.audit=dict(status='ALL180_PREDICTIONS_INDEPENDENTLY_AUDITED',plan_sha256=self.plan['plan_sha256'],
                        protected_assessment_access=False,model_calls=0,groups=self.records)
        self.save_audit()

    def tearDown(self):self.temp.cleanup()
    def save_audit(self):(self.run/'AUDIT.json').write_text(json.dumps(self.audit))
    def loader(self,plan,guard,record):
        self.assertIs(plan,self.plan);self.assertIs(guard,self.guard)
        return group_fixture(record['fold'],record['gauge_id'])

    def test_full_stage_outputs_all180_without_model_calls_or_source_fetches(self):
        with patch.object(stage,'load_group',side_effect=self.loader) as loader:
            result=stage.run_diagnostics(self.plan,self.guard)
        self.assertEqual(loader.call_count,12)
        self.assertEqual(result['panel_count'],180)
        self.assertEqual(result['series_count'],3240)
        self.assertEqual(result['correlation_cells'],29160)
        self.assertEqual(len(result['groups']),12)
        self.assertEqual(len(self.guard.writes),13)
        self.assertEqual(self.guard.writes[-1],'RESIDUAL_DIAGNOSTICS.json')
        self.assertFalse(result['uncertainty_method_reselected'])
        self.assertFalse(result['independence_established'])
        self.assertEqual(result['model_calls'],0)
        self.assertTrue(self.guard.checks>180)
        stored=json.loads((self.run/'RESIDUAL_DIAGNOSTICS.json').read_text())
        self.assertEqual(stored,result)

    def test_inherited_loader_protected_guard_is_not_caught_or_bypassed(self):
        with patch.object(stage,'load_group',side_effect=ValueError('Protected or crossing forecast calendar')):
            with self.assertRaisesRegex(ValueError,'Protected or crossing'):
                stage.run_diagnostics(self.plan,self.guard)
        self.assertEqual(self.guard.writes,[])

    def test_missing_or_duplicate_audit_group_fails_before_value_reader(self):
        self.audit['groups']=self.records[:-1]+[self.records[0]];self.save_audit()
        with patch.object(stage,'load_group',side_effect=AssertionError('Values read')):
            with self.assertRaisesRegex(ValueError,'All12 unique'):
                stage.run_diagnostics(self.plan,self.guard)
        self.assertEqual(self.guard.writes,[])

    def test_changed_group_commit_fails_before_value_reader(self):
        first=self.records[0];path=self.run/'groups'/first['group']/'COMMIT.json'
        changed=deepcopy(first);changed['plan_sha256']='0'*64;path.write_text(json.dumps(changed))
        with patch.object(stage,'load_group',side_effect=AssertionError('Values read')):
            with self.assertRaisesRegex(ValueError,'immutable group commit'):
                stage.run_diagnostics(self.plan,self.guard)
        self.assertEqual(self.guard.writes,[])


if __name__=='__main__':unittest.main()
