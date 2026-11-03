"""Proposed scoring arithmetic and fake-inference orchestration; zero updates."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow.reference import ASSESSMENT_START
from hydra_directflow.scoring import (ScoringPolicy, ScoreIdentity, score_physical_h18,
    aggregate_cv_folds, aggregate_final_seeds, select_cv_configuration, run_development_score)
from hydra_directflow.training import FULL_FEATURE_SCHEMA, EarlyStopping

SMALL = ScoringPolicy(minimum_distinct_target_hours=2, minimum_distinct_utc_dates=1)
IDENTITY = ScoreIdentity('LSTM', 'fixed_v1', '03161000', FULL_FEATURE_SCHEMA, 24, 'cv_1', 17)


def epoch(value):
    return int(datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp())


def coordinates(n=4, start='2020-08-01T00:00:00Z', step_hours=1):
    origin = epoch(start) + np.arange(n, dtype=np.int64) * step_hours * 3600
    return dict(case_ids=np.array(['03161000:' + datetime.fromtimestamp(int(t), timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') for t in origin]),
                origins=origin, forecast_time=origin[:, None] + np.arange(1, 19)[None, :] * 3600)


def panel(*, n=4, start='2020-08-01T00:00:00Z', step_hours=1, identity=IDENTITY,
          policy=SMALL, error=1., labels=None, mask=None, **changes):
    labels = np.broadcast_to(np.arange(1, n + 1, dtype=np.float64)[:, None], (n, 18)).copy() if labels is None else labels
    args = dict(**coordinates(n, start, step_hours), labels=labels,
        label_mask=np.ones((n, 18), bool) if mask is None else mask,
        predictions=labels + error, identity=identity, policy=policy,
        input_cohort_digest='a'*64, label_policy_digest='b'*64, label_source_digest='c'*64)
    args.update(changes)
    return score_physical_h18(**args)


def cv(*, config='fixed_v1', errors=(1., 2., 3.)):
    return {f'cv_{i+1}': panel(n=4+i*2, start=f'{2020+i}-08-01T00:00:00Z', error=error,
                identity=replace(IDENTITY, fold_id=f'cv_{i+1}', configuration_id=config)) for i, error in enumerate(errors)}


def input_arrays():
    clock = coordinates()
    n = len(clock['origins'])
    values = dict(origin_time=clock['origins'], target_time=clock['forecast_time'], case_id=clock['case_ids'],
        history_time=clock['origins'][:, None] - np.arange(25, 1, -1) * 3600, site_id=np.zeros(n, dtype=np.int64),
        feature_schema=np.array(FULL_FEATURE_SCHEMA), cohort_admission_digest=np.array('d'*64),
        history=np.ones((n,24,2), np.float32), weather=np.ones((n,18,4),np.float32), state=np.ones((n,2),np.float32),
        target=np.broadcast_to(np.arange(1,n+1,dtype=np.float64)[:,None],(n,18)).copy())
    for name in ('history','weather','state','target'): values[name+'_mask']=np.ones(values[name].shape,bool)
    return values


def runner_args(arrays):
    return dict(arrays=arrays, outer_indices=(0,1,2,3), outer_start_utc='2020-08-01T00:00:00Z', outer_end_utc='2021-07-31T00:00:00Z',
        checkpoint_path='unused-synthetic.pt', checkpoint_sha256='1'*64, training_source_identity='2'*64,
        training_cohort_digest='3'*64, gradient_fit_case_ids=('03161000:2020-01-01T00:00:00Z',),
        inference_source_identity='4'*64, inference_cohort_digest='d'*64, identity=IDENTITY,
        label_policy_digest='5'*64, label_source_digest='6'*64, policy=SMALL, batch_size=2)


class ScoreTests(unittest.TestCase):
    def test_physical_metrics_equal_lead_nse_and_negative_predictions_are_unclipped(self):
        labels=np.broadcast_to(np.array([1.,2.,4.,8.])[:,None],(4,18)).copy()
        predicted=labels.copy();predicted[:,0]=[-1.,2.,4.,8.];predicted[:,1]+=4.
        value=panel(labels=labels,predictions=predicted)
        first=value['leads'][0]
        self.assertEqual(first['physical_mse'],1.);self.assertEqual(first['rmse'],1.)
        self.assertEqual(first['mae'],.5);self.assertEqual(first['bias'],-.5)
        self.assertEqual(first['negative_frequency'],.25)
        denominator=((labels[:,0]-labels[:,0].mean())**2).sum()
        expected=(1-4/denominator+1-64/denominator+16)/18
        self.assertAlmostEqual(value['primary_score'],expected)
        self.assertFalse(value['negative_predictions_clipped'])
        self.assertIn('standardized-Q MSE',value['loss_distinction'])
        with self.assertRaisesRegex(ValueError,'Physical discharge'):
            panel(label_units='standardized')

    def test_support_hours_and_dates_are_independent_reporting_gates(self):
        hours_only=panel(n=168,policy=ScoringPolicy())
        dates_only=panel(n=28,step_hours=24,policy=ScoringPolicy())
        both=panel(n=168,step_hours=4,policy=ScoringPolicy())
        self.assertIsNone(hours_only['primary_score']);self.assertIsNotNone(hours_only['leads'][0]['nse'])
        self.assertEqual(hours_only['leads'][0]['primary_ineligibility'],['TOO_FEW_DISTINCT_UTC_DATES'])
        self.assertEqual(dates_only['leads'][0]['primary_ineligibility'],['TOO_FEW_DISTINCT_TARGET_HOURS'])
        self.assertIsNotNone(both['primary_score'])
        self.assertIn('not independence',both['support_qualification'])

    def test_undefined_nse_and_zero_labels_are_retained_without_primary_omission(self):
        for values,reason in ((np.ones((4,18)),'CONSTANT_REFERENCE'),
                              (np.broadcast_to(np.array([0.,1e-200,2e-200,3e-200])[:,None],(4,18)).copy(),'REFERENCE_VARIANCE_NUMERIC_UNDERFLOW')):
            value=panel(labels=values,error=0.)
            self.assertIsNone(value['primary_score']);self.assertEqual(value['leads'][0]['undefined']['nse'],reason)
            self.assertEqual(value['leads'][0]['rmse'],0.)
        mask=np.ones((4,18),bool);mask[:,-1]=False
        value=panel(mask=mask)
        self.assertEqual(len(value['leads']),18);self.assertIsNone(value['primary_score'])
        self.assertEqual(value['leads'][-1]['n'],0);self.assertIsNone(value['leads'][-1]['negative_frequency'])
        self.assertEqual(value['leads'][0]['n'],4)

    def test_missing_predictions_fail_even_on_masked_labels(self):
        mask=np.ones((4,18),bool);mask[0,0]=False
        prediction=np.ones((4,18));prediction[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Missing/nonfinite'):panel(mask=mask,predictions=prediction)
        present=np.ones((4,18),bool);present[0,0]=False
        with self.assertRaisesRegex(ValueError,'Missing/nonfinite'):panel(mask=mask,prediction_mask=present)
        truth=np.ones((4,18));truth[0,0]=np.nan
        value=panel(labels=truth,mask=mask,predictions=np.ones((4,18)))
        self.assertEqual(value['leads'][0]['n'],3)

    def test_coordinates_reject_duplicate_unsigned_reverse_protected_and_wrong_gauge(self):
        for mutation in ('duplicate','unsigned_reverse','protected','wrong_gauge'):
            args=coordinates()
            if mutation=='duplicate':args['origins'][1]=args['origins'][0]
            elif mutation=='unsigned_reverse':args['origins']=args['origins'][::-1].astype(np.uint64)
            elif mutation=='protected':args['forecast_time'][0,0]=ASSESSMENT_START
            else:args['case_ids'][0]='03164000:2020-08-01T00:00:00Z'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):panel(**args)

    def test_fold_mean_weights_each_complete_fold_equally_not_sample_count(self):
        values=cv();value=aggregate_cv_folds(values,policy=SMALL)
        scores=[values[name]['primary_score'] for name in SMALL.expected_cv_folds]
        self.assertAlmostEqual(value['primary_score'],sum(scores)/3)
        self.assertNotAlmostEqual(value['primary_score'],sum(v*s for v,s in zip((4,6,8),scores))/18)
        self.assertEqual(value['per_lead_mean_metrics'][0]['rmse'],2.)
        with self.assertRaisesRegex(ValueError,'All three'):aggregate_cv_folds({k:v for k,v in values.items() if k!='cv_2'},policy=SMALL)
        values['cv_2']=panel(start='2021-08-01T00:00:00Z',identity=replace(IDENTITY,fold_id='cv_2'),labels=np.ones((4,18)))
        self.assertIsNone(aggregate_cv_folds(values,policy=SMALL)['primary_score'])

    def test_fold_overlap_and_forged_primary_reject(self):
        values=cv();values['cv_2']=panel(identity=replace(IDENTITY,fold_id='cv_2'))
        with self.assertRaisesRegex(ValueError,'overlap'):aggregate_cv_folds(values,policy=SMALL)
        values=cv();values['cv_1']['primary_score']=1.
        with self.assertRaisesRegex(ValueError,'all18'):aggregate_cv_folds(values,policy=SMALL)

    def test_all_seeds_mean_metrics_not_best_seed_or_ensemble_and_same_labels(self):
        values={seed:panel(identity=replace(IDENTITY,fold_id='final_development',seed=seed),error=error)
                for seed,error in zip(SMALL.final_seeds,(-2.,0.,2.))}
        result=aggregate_final_seeds(values,policy=SMALL)
        self.assertAlmostEqual(result['per_lead_mean_metrics'][0]['rmse'],4/3)
        self.assertEqual(result['per_lead_mean_metrics'][0]['bias'],0.)
        self.assertLess(result['primary_score'],1.) # The mean-prediction ensemble would be perfect.
        self.assertEqual(set(result['seed_reports']),{'17','29','43'})
        self.assertFalse(result['seed_selection_performed'])
        with self.assertRaisesRegex(ValueError,'All three'):aggregate_final_seeds({17:values[17],29:values[29]},policy=SMALL)
        changed=deepcopy(values);changed[43]['label_source_digest']='f'*64
        with self.assertRaisesRegex(ValueError,'same labels'):aggregate_final_seeds(changed,policy=SMALL)
        low_support={seed:panel(identity=replace(IDENTITY,fold_id='final_development',seed=seed),policy=ScoringPolicy())
                     for seed in SMALL.final_seeds}
        self.assertIsNone(aggregate_final_seeds(low_support)['primary_score'])
        self.assertEqual(aggregate_final_seeds(low_support)['undefined_seeds'],[17,29,43])

    def test_exact_ties_use_declared_order_without_rounding_or_seed_selection(self):
        a=aggregate_cv_folds(cv(config='a'),policy=SMALL)
        b=aggregate_cv_folds(cv(config='b'),policy=SMALL)
        value=select_cv_configuration({'a':a,'b':b},declared_configuration_order=('b','a'))
        self.assertEqual(value['selected_configuration'],'b');self.assertEqual(value['exactly_tied_configurations'],['b','a'])
        better=aggregate_cv_folds(cv(config='a',errors=(1.-1e-10,2.,3.)),policy=SMALL)
        value=select_cv_configuration({'a':better,'b':b},declared_configuration_order=('b','a'))
        self.assertEqual(value['selected_configuration'],'a');self.assertEqual(value['exactly_tied_configurations'],['a'])
        b['status']='PRIMARY_UNAVAILABLE'
        with self.assertRaisesRegex(ValueError,'Every candidate'):select_cv_configuration({'a':a,'b':b},declared_configuration_order=('b','a'))

    def test_stopping_tie_keeps_earliest_accepted_best_without_outer_metric(self):
        stopping=EarlyStopping(4,1,4,0.)
        self.assertEqual([stopping.observe(v) for v in (2.,1.,1.,1.)],[True,True,False,False])
        self.assertEqual(stopping.best_loss,1.)


class RunnerTests(unittest.TestCase):
    def test_plan_is_metadata_only_and_never_loads_checkpoint_or_labels(self):
        class MetadataOnly(dict):
            def __getitem__(self,key):
                if key in ('target','target_mask','history','weather','state'):raise AssertionError('Numerical read: '+key)
                return super().__getitem__(key)
        with patch('hydra_directflow.scoring._load_best_predictor',side_effect=AssertionError('Model load')):
            value=run_development_score(**runner_args(MetadataOnly(input_arrays())))
        self.assertEqual(value['status'],'PLANNED_NOT_EXECUTED');self.assertFalse(value['labels_read'])
        self.assertFalse(value['protected_assessment_enabled']);self.assertEqual(value['model_loads'],0)

    def test_fake_best_inference_finishes_all_batches_before_labels_and_binds_two_parents(self):
        tracker={'batches':0,'labels':[]}
        class DelayedLabels(dict):
            def __getitem__(self,key):
                if key in ('target','target_mask'):
                    if tracker['batches']!=2:raise AssertionError('Label fetched before all predictions')
                    tracker['labels'].append(key)
                return super().__getitem__(key)
        args=runner_args(DelayedLabels(input_arrays()));plan=run_development_score(**args)
        base=dict(checkpoint_sha256='1'*64,training_source_identity='2'*64,training_cohort_admission_digest='3'*64,
            gauge_id='03161000',feature_schema=FULL_FEATURE_SCHEMA,architecture='LSTM',seed=17,history_hours=24,
            prediction_units='m3/s',selected_state='best_model',optimizer_constructed=False,scaler_fitted=False,last_state_used=False,
            device='cpu',selected_best_epoch=1,completed_epochs=2)
        class FakePredictor:
            receipt=base
            def predict(self,inputs,**kwargs):
                selftest.assertNotIn('target',inputs);selftest.assertNotIn('target_mask',inputs)
                selftest.assertEqual(kwargs,dict(expected_inference_source_identity='4'*64,expected_inference_cohort_admission_digest='d'*64))
                tracker['batches']+=1
                return dict(prediction_q=np.ones((len(inputs['case_id']),18)),
                    **{key:inputs[key] for key in ('case_id','origin_time','history_time','forecast_time','site_id')},
                    receipt={**base,'inference_source_identity':'4'*64,'inference_cohort_admission_digest':'d'*64,
                        'target_values_read':False,'target_masks_read':False,'optimizer_updates':0,'posthoc_clipping':False})
        selftest=self
        with patch('hydra_directflow.scoring._load_best_predictor',return_value=FakePredictor()) as load:
            value=run_development_score(**args,execution_enabled=True,approved_plan_sha256=plan['plan_sha256'])
        self.assertEqual(tracker['labels'],['target','target_mask'])
        self.assertEqual(load.call_args.kwargs['expected_source_identity'],'2'*64)
        self.assertEqual(load.call_args.kwargs['expected_training_cohort_admission_digest'],'3'*64)
        self.assertEqual(value['execution']['inference_batches'],2)
        self.assertEqual(value['execution']['selected_best_epoch'],1)
        self.assertEqual(value['execution']['optimizer_updates'],0)

    def test_bad_plan_last_state_protected_or_fit_overlap_stops_before_labels(self):
        args=runner_args(input_arrays());plan=run_development_score(**args)
        with patch('hydra_directflow.scoring._load_best_predictor',side_effect=AssertionError('Model load')):
            with self.assertRaisesRegex(ValueError,'Exact caller-approved'):
                run_development_score(**args,execution_enabled=True,approved_plan_sha256='f'*64)
            crossed=dict(args,gradient_fit_case_ids=('03161000:2020-08-01T00:00:00Z',))
            with self.assertRaisesRegex(ValueError,'must precede'):run_development_score(**crossed)
            args['arrays']['target_time'][0,0]=ASSESSMENT_START
            with self.assertRaisesRegex(ValueError,'Protected'):run_development_score(**args)
        args=runner_args(input_arrays());plan=run_development_score(**args)
        with patch('hydra_directflow.scoring._load_best_predictor',return_value=SimpleNamespace(receipt={'selected_state':'last_model'})):
            with self.assertRaisesRegex(ValueError,'saved best state'):
                run_development_score(**args,execution_enabled=True,approved_plan_sha256=plan['plan_sha256'])

    def test_resource_bound_stops_before_checkpoint_load_without_retry(self):
        args=runner_args(input_arrays());args['max_combined_bytes']=1
        plan=run_development_score(**args)
        with patch('hydra_directflow.scoring._load_best_predictor',side_effect=AssertionError('Model load')):
            with self.assertRaisesRegex(ValueError,'resource guard'):
                run_development_score(**args,execution_enabled=True,approved_plan_sha256=plan['plan_sha256'])

    def test_unselected_protected_source_row_rejects_before_any_numeric_read(self):
        class PoisonValues(dict):
            def __getitem__(self,key):
                if key in ('history','weather','state','target','target_mask'):raise AssertionError('Numerical values read')
                return super().__getitem__(key)
        for mutation in ('protected_target','protected_origin_stale_target','development_inconsistent_target','fractional_hour_origin'):
            arrays=input_arrays()
            if mutation=='protected_target':arrays['target_time'][-1,-1]=ASSESSMENT_START
            elif mutation=='protected_origin_stale_target':arrays['origin_time'][-1]=ASSESSMENT_START
            elif mutation=='development_inconsistent_target':arrays['target_time'][-1,-1]+=3600
            else:arrays['origin_time'][-1]+=1
            args=runner_args(PoisonValues(arrays));args['outer_indices']=(0,1)
            with self.subTest(mutation=mutation),patch('hydra_directflow.scoring._load_best_predictor',side_effect=AssertionError('Model load')):
                with self.assertRaisesRegex(ValueError,'anywhere in the source'):run_development_score(**args)


if __name__=='__main__':unittest.main()
