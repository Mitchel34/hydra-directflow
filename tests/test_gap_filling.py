"""Synthetic Q values only; no actual sources, neural models or study fits."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np

from hydra_directflow.gap_filling import (fit_ridge_fallback,RidgeFallback,fill_history,RIDGE_POLICY_SHA256,
                                        filling_policy_fingerprint)
from hydra_directflow.gap_policies import POLICY_IDS
from hydra_directflow.local_data import utc_epoch
from hydra_directflow.reference import digest


def history(missing=()):
    cutoff=utc_epoch('2021-01-21T00:00:00Z');times=cutoff-np.arange(169,1,-1,dtype=np.int64)*3600
    mask=np.ones(168,bool);mask[list(missing)]=False
    values=np.arange(168,dtype=float)+10.
    return dict(q_values=values,approved_mask=mask,history_times=times,source_observation_times=tuple(int(t) for t in times),
        source_ids=tuple('synthetic:'+str(int(t)) for t in times),source_flags=tuple(('R','A','R') for _ in times),
        cutoff_epoch=cutoff,gauge_id='03161000',policy_id=POLICY_IDS['combined_2'])


def ridge_series():
    first=utc_epoch('2021-01-01T00:00:00Z');times=first+np.arange(96,dtype=np.int64)*3600
    return dict(observation_times=times,q_values=200.-.5*np.arange(96),approved_mask=np.ones(96,bool),
        source_observation_times=tuple(int(t) for t in times),source_ids=tuple('ridge:'+str(int(t)) for t in times),
        source_flags=tuple(('A',) for _ in times),gauge_id='03161000',gradient_start_epoch=first,
        gradient_end_epoch=first+72*3600,refit_epoch=utc_epoch('2021-01-20T00:00:00Z'),source_identity='a'*64)


def load(payload,**changes):
    params=dict(expected_source_identity=payload['source_identity'],expected_gauge_id=payload['gauge_id'],
        expected_gradient_start_epoch=payload['gradient_start_epoch'],expected_gradient_end_epoch=payload['gradient_end_epoch'],
        expected_refit_epoch=payload['refit_epoch'],expected_model_identity=payload['model_identity'])
    params.update(changes);return RidgeFallback.from_dict(payload,**params)


class GuardedValues:
    def __init__(self,values,forbidden):self.values=values;self.forbidden=set(forbidden);self.read=[]
    def __len__(self):return len(self.values)
    def __getitem__(self,index):
        if not isinstance(index,(int,np.integer)) or index in self.forbidden:raise AssertionError('Forbidden/nonselective Q read')
        self.read.append(int(index));return self.values[index]
    def __array__(self,*args,**kwargs):raise AssertionError('Whole Q array materialized')


class GapFillingTests(unittest.TestCase):
    def test_interior_linear_and_trailing_carry_preserve_observation_masks_and_provenance(self):
        args=history([50,167]);before=deepcopy(args);result=fill_history(**args)
        self.assertTrue(result['success']);self.assertEqual(result['numerical_fills_applied'],2)
        self.assertEqual(result['usable_q'][50],60.);self.assertEqual(result['usable_q'][167],176.)
        self.assertTrue(result['usable_mask'].all());self.assertFalse(result['supplied_approved_mask'][50])
        self.assertFalse(result['supplied_approved_mask'][167]);self.assertFalse(result['filled_values_are_observations'])
        provenance=result['slot_provenance'][50]
        self.assertEqual(provenance['support_source_ids'],(args['source_ids'][49],args['source_ids'][51]))
        self.assertEqual(provenance['support_source_flags'],(('R','A','R'),('R','A','R')))
        self.assertEqual(provenance['original_source_flags'],('R','A','R'));self.assertFalse(provenance['input_observation_available'])
        np.testing.assert_array_equal(args['q_values'],before['q_values']);np.testing.assert_array_equal(args['approved_mask'],before['approved_mask'])
        self.assertEqual(args['source_flags'],before['source_flags']);self.assertFalse(result['usable_q'].flags.writeable)

    def test_hidden_values_never_read_even_when_poisoned_or_changed(self):
        args=history([50,167]);guarded=GuardedValues(args['q_values'],[50,167]);args['q_values']=guarded
        first=fill_history(**args);self.assertTrue(first['success']);self.assertEqual(len(guarded.read),166)
        raw=history([50,167]);raw['q_values'][[50,167]]=[1e100,np.nan]
        second=fill_history(**raw);np.testing.assert_array_equal(first['usable_q'],second['usable_q'])
        self.assertEqual(first['policy_fingerprint'],second['policy_fingerprint'])

    def test_long_leading_and_total_cap_fail_without_partial_fills(self):
        for missing in ([0],[30,31,32],[30,60,167]):
            result=fill_history(**history(missing))
            self.assertFalse(result['success']);self.assertEqual(result['numerical_fills_applied'],0)
            self.assertTrue(np.isnan(result['usable_q'][missing]).all());self.assertFalse(result['usable_mask'][missing].any())
        args=history([30,167]);args['policy_id']=POLICY_IDS['combined_1']
        self.assertEqual(fill_history(**args)['failure_reasons'],['TOTAL_MISSING_EXCEEDS_CAP'])

    def test_future_endpoint_clock_bad_quality_or_missing_identity_rejects_before_Q_reads(self):
        for mutation in ('future','quality','missing_id','flags_not_tuples'):
            args=history([166]);args['q_values']=GuardedValues(args['q_values'],range(168))
            if mutation=='future':args['source_observation_times']=args['source_observation_times'][:-1]+(args['cutoff_epoch']-3600,)
            elif mutation=='quality':args['source_flags']=(('P',),)+args['source_flags'][1:]
            elif mutation=='missing_id':args['source_ids']=(None,)+args['source_ids'][1:]
            else:args['source_flags']=tuple(['A'] for _ in range(168))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):fill_history(**args)

    def test_ridge_fit_uses_only_approved_gradient_pair_values_and_cutoff_tail(self):
        args=ridge_series();guarded=GuardedValues(args['q_values'],range(72,96));args['q_values']=guarded
        fitted=fit_ridge_fallback(**args);payload=fitted.to_dict()
        self.assertEqual(payload['training_pairs'],65);self.assertEqual(max(guarded.read),71)
        self.assertEqual(payload['policy_sha256'],RIDGE_POLICY_SHA256)
        second=ridge_series();second['q_values'][72:]=np.nan
        self.assertEqual(payload['training_values_sha256'],fit_ridge_fallback(**second).to_dict()['training_values_sha256'])
        # Refit at gradient end excludes the final label at end-1h under the simulated2h delay.
        tight=ridge_series();tight['refit_epoch']=tight['gradient_end_epoch']
        guarded=GuardedValues(tight['q_values'],range(71,96));tight['q_values']=guarded
        result=fit_ridge_fallback(**tight);self.assertEqual(result.to_dict()['training_pairs'],64)
        self.assertEqual(max(guarded.read),70)

    def test_ridge_metadata_rejects_future_bounds_source_clocks_and_no_pairs_before_values(self):
        for mutation in ('refit_before_gradient_end','wrong_source_clock','no_pairs','unordered'):
            args=ridge_series();args['q_values']=GuardedValues(args['q_values'],range(96))
            if mutation=='refit_before_gradient_end':args['refit_epoch']=args['gradient_end_epoch']-3600
            elif mutation=='wrong_source_clock':args['source_observation_times']=(args['observation_times'][0]+3600,)+args['source_observation_times'][1:]
            elif mutation=='no_pairs':args['approved_mask'][:]=False
            else:args['observation_times']=args['observation_times'][::-1]
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):fit_ridge_fallback(**args)

    def test_unapproved_or_nonconsecutive_ridge_rows_are_never_features_or_outputs(self):
        args=ridge_series();args['approved_mask'][25]=False
        args['source_flags']=args['source_flags'][:25]+(('P',),)+args['source_flags'][26:]
        guarded=GuardedValues(args['q_values'],[25,*range(72,96)]);args['q_values']=guarded
        model=fit_ridge_fallback(**args);self.assertEqual(model.to_dict()['training_pairs'],57)
        self.assertNotIn(25,guarded.read)
        gap=ridge_series();gap['observation_times'][25:]+=3600
        gap['source_observation_times']=tuple(int(t) for t in gap['observation_times'])
        changed=fit_ridge_fallback(**gap)
        self.assertLess(changed.to_dict()['training_pairs'],65)

    def test_ridge_matches_independent_alpha1_unpenalized_intercept_solution(self):
        args=ridge_series();model=fit_ridge_fallback(**args);p=model.to_dict()
        q=args['q_values'][:72];x=np.array([q[i:i+6] for i in range(65)]);y=np.array([q[i+6:i+8] for i in range(65)])
        z=(x-x.mean(0))/x.std(0);design=np.column_stack((np.ones(65),z))
        penalty=np.diag([0.]+[1.]*6)
        solution=np.linalg.solve(design.T@design+penalty,design.T@y)
        np.testing.assert_allclose(p['intercept'],solution[0],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(p['coefficients'],solution[1:],rtol=1e-10,atol=1e-10)
        constant=ridge_series();constant['q_values'][:]=7.;c=fit_ridge_fallback(**constant).to_dict()
        self.assertEqual(c['constant_features'],[True]*6);self.assertEqual(c['feature_scale'],[1.]*6)
        self.assertEqual(c['intercept'],[7.,7.]);np.testing.assert_array_equal(c['coefficients'],np.zeros((6,2)))

    def test_ridge_serialization_binds_source_gauge_bounds_policy_and_never_refits(self):
        model=fit_ridge_fallback(**ridge_series());p=model.to_dict()
        with patch('numpy.linalg.solve',side_effect=AssertionError('Load refitted model')):
            restored=load(p)
        self.assertEqual(restored.identity,model.identity);self.assertEqual(restored.to_dict(),p)
        for changes in ({'expected_source_identity':'b'*64},{'expected_gauge_id':'03164000'},
                        {'expected_refit_epoch':p['refit_epoch']+3600},{'expected_gradient_end_epoch':p['gradient_end_epoch']+3600},
                        {'expected_model_identity':'f'*64}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):load(p,**changes)
        for key,value in (('coefficients',[[1.,2.]]),('feature_scale',[0.]*6),('policy_sha256','0'*64)):
            bad=deepcopy(p);bad[key]=value
            with self.assertRaises(ValueError):load(bad)

    def test_ridge_fill_is_direct_original_six_lags_and_refit_available(self):
        model=fit_ridge_fallback(**ridge_series());args=history([166,167]);args.update(trailing_method='ridge',ridge_model=model,expected_ridge_identity=model.identity)
        expected=model._predict_values(args['q_values'][160:166])
        result=fill_history(**args);self.assertTrue(result['success'])
        np.testing.assert_allclose(result['usable_q'][166:],expected,rtol=0,atol=0)
        self.assertEqual(result['slot_provenance'][167]['support_source_ids'],args['source_ids'][160:166])
        self.assertEqual(result['slot_provenance'][166]['support_source_ids'],result['slot_provenance'][167]['support_source_ids'])
        self.assertFalse(result['supplied_approved_mask'][166]);self.assertTrue(result['usable_mask'][166])

    def test_ridge_never_uses_filled_interior_endpoint_or_future_fitted_model(self):
        model=fit_ridge_fallback(**ridge_series());args=history([164,167]);args.update(trailing_method='ridge',ridge_model=model,expected_ridge_identity=model.identity)
        with patch.object(RidgeFallback,'_predict_values',side_effect=AssertionError('Used a filled lag')):
            result=fill_history(**args)
        self.assertFalse(result['success']);self.assertEqual(result['failure_reasons'],['RIDGE_ORIGINAL_SIX_LAG_CONTEXT_UNAVAILABLE'])
        self.assertTrue(np.isnan(result['usable_q'][[164,167]]).all());self.assertEqual(result['numerical_fills_applied'],0)
        late=deepcopy(ridge_series());late['refit_epoch']=utc_epoch('2021-02-01T00:00:00Z');future=fit_ridge_fallback(**late)
        args=history([167]);args.update(q_values=GuardedValues(args['q_values'],range(168)),trailing_method='ridge',ridge_model=future,expected_ridge_identity=future.identity)
        with self.assertRaisesRegex(ValueError,'not available'):fill_history(**args)

    def test_negative_or_nonfinite_derived_value_fails_atomically_without_carry_fallback(self):
        model=fit_ridge_fallback(**ridge_series());args=history([50,167]);args.update(trailing_method='ridge',ridge_model=model,expected_ridge_identity=model.identity)
        for value,reason in ((-1.,'NEGATIVE_DERIVED_VALUE'),(np.nan,'NONFINITE_DERIVED_VALUE')):
            with patch.object(RidgeFallback,'_predict_values',return_value=np.array([value,2.])):
                result=fill_history(**args)
            self.assertFalse(result['success']);self.assertIn(reason,result['failure_reasons'])
            self.assertEqual(result['numerical_fills_applied'],0);self.assertTrue(np.isnan(result['usable_q'][[50,167]]).all())
            self.assertFalse(result['usable_mask'][[50,167]].any())

    def test_no_target_or_weather_interface_and_ridge_policy_identity_is_explicit(self):
        args=history([167]);baseline=fill_history(**args)
        for key in ('targets','weather'):
            with self.assertRaises(TypeError):fill_history(**args,**{key:object()})
        self.assertFalse(baseline['targets_read_or_changed']);self.assertFalse(baseline['weather_read_or_changed'])
        self.assertEqual(baseline['policy_fingerprint'],filling_policy_fingerprint(POLICY_IDS['combined_2'],'carry'))
        with self.assertRaises(ValueError):fill_history(**{**args,'trailing_method':'ridge'})
        with self.assertRaises(ValueError):filling_policy_fingerprint(POLICY_IDS['combined_2'],'carry',ridge_identity='a'*64)


if __name__=='__main__':unittest.main()
