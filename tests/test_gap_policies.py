"""Synthetic exact-hour history masks/clocks only; no discharge values or fills."""
import copy
import unittest
import numpy as np

from hydra_directflow.gap_policies import (POLICY_IDS,NATURAL_L24_ID,POLICY_CATALOG_SHA256,policy_catalog,classify_history_window)
from hydra_directflow.local_data import utc_epoch
from hydra_directflow.reference import ASSESSMENT_START,digest


def fixture(missing=()):
    cutoff=utc_epoch('2023-08-17T05:00:00Z')
    times=cutoff-np.arange(169,1,-1,dtype=np.int64)*3600
    approved=np.ones(168,bool);approved[list(missing)]=False
    source=[int(t) if approved[i] else None for i,t in enumerate(times)]
    return dict(history_times=times,approved_mask=approved,source_observation_times=source,cutoff_epoch=cutoff)


def decision(result,alias):return result['candidates'][POLICY_IDS[alias]]


class GapPolicyTests(unittest.TestCase):
    def test_complete_history_passes_all_candidates_without_incremental_case_or_repairs(self):
        result=classify_history_window(**fixture())
        self.assertEqual(result['runs'],[]);self.assertEqual(result['missing_total'],0)
        self.assertTrue(all(row['history_admissible'] for row in result['candidates'].values()))
        self.assertTrue(all(not row['additional_history_candidate'] for row in result['candidates'].values()))
        self.assertTrue(result['natural_l24']['history_admissible']);self.assertEqual(result['natural_l24']['policy_id'],NATURAL_L24_ID)
        self.assertFalse(result['availability']['actual_delivery_or_revision_times_verified'])
        self.assertFalse(result['execution']['history_arrays_filled']);self.assertFalse(result['execution']['scientific_policy_adopted'])

    def test_single_interior_gap_has_two_original_approved_available_endpoints(self):
        args=fixture([166]);result=classify_history_window(**args);run=result['runs'][0]
        self.assertEqual((run['position'],run['start_index'],run['end_index_exclusive'],run['length_hours']),('interior',166,167,1))
        self.assertEqual(run['nearest_missing_age_hours'],3);self.assertEqual(run['farthest_missing_age_hours'],3)
        self.assertEqual(run['right_endpoint']['observation_epoch'],args['cutoff_epoch']-7200)
        self.assertEqual(run['right_endpoint']['simulated_available_epoch'],args['cutoff_epoch'])
        self.assertTrue(decision(result,'interior_1')['history_admissible'])
        self.assertTrue(decision(result,'combined_1')['history_admissible'])
        self.assertFalse(decision(result,'strict')['history_admissible']);self.assertFalse(decision(result,'trailing_2')['history_admissible'])
        self.assertEqual(decision(result,'trailing_2')['reasons'],['INTERIOR_GAP_NOT_ALLOWED'])

    def test_two_separate_one_hour_runs_need_total_cap_two_not_only_run_cap_one(self):
        result=classify_history_window(**fixture([20,40]))
        self.assertEqual(result['maximum_missing_run_hours'],1);self.assertEqual(result['missing_total'],2)
        self.assertFalse(decision(result,'interior_1')['history_admissible'])
        self.assertEqual(decision(result,'interior_1')['reasons'],['TOTAL_MISSING_EXCEEDS_CAP'])
        self.assertTrue(decision(result,'interior_2')['history_admissible'])

    def test_two_hour_and_longer_interior_runs_are_distinguished(self):
        two=classify_history_window(**fixture([20,21]));three=classify_history_window(**fixture([20,21,22]))
        self.assertTrue(decision(two,'interior_2')['history_admissible'])
        self.assertEqual(decision(two,'interior_1')['reasons'],['RUN_LENGTH_EXCEEDS_CAP','TOTAL_MISSING_EXCEEDS_CAP'])
        self.assertFalse(decision(three,'combined_2')['history_admissible'])
        self.assertIn('RUN_LENGTH_EXCEEDS_CAP',decision(three,'combined_2')['reasons'])

    def test_trailing_gap_uses_only_last_approved_history_reading_and_no_right_endpoint(self):
        args=fixture([166,167]);result=classify_history_window(**args);run=result['runs'][0]
        self.assertEqual(run['position'],'trailing');self.assertIsNone(run['right_endpoint'])
        self.assertEqual(run['left_endpoint']['index'],165)
        self.assertEqual(run['left_endpoint']['observation_epoch'],args['cutoff_epoch']-4*3600)
        self.assertEqual((run['nearest_missing_age_hours'],run['farthest_missing_age_hours']),(2,3))
        self.assertTrue(decision(result,'trailing_2')['history_admissible']);self.assertFalse(decision(result,'trailing_1')['history_admissible'])
        self.assertFalse(decision(result,'interior_2')['history_admissible']);self.assertFalse(result['natural_l24']['history_admissible'])
        one=classify_history_window(**fixture([167]));self.assertTrue(decision(one,'trailing_1')['history_admissible'])

    def test_combined_interior_and_trailing_require_combined_mode_and_total_cap(self):
        result=classify_history_window(**fixture([50,167]))
        self.assertTrue(decision(result,'combined_2')['history_admissible']);self.assertFalse(decision(result,'combined_1')['history_admissible'])
        self.assertIn('TRAILING_GAP_NOT_ALLOWED',decision(result,'interior_2')['reasons'])
        self.assertIn('INTERIOR_GAP_NOT_ALLOWED',decision(result,'trailing_2')['reasons'])
        longer=classify_history_window(**fixture([50,166,167]))
        self.assertFalse(decision(longer,'combined_2')['history_admissible'])
        self.assertEqual(decision(longer,'combined_2')['reasons'],['TOTAL_MISSING_EXCEEDS_CAP'])

    def test_leading_and_all_missing_are_never_repair_candidates(self):
        for missing in ([0],list(range(168))):
            result=classify_history_window(**fixture(missing))
            self.assertTrue(all(not row['history_admissible'] for row in result['candidates'].values()))
            self.assertIn('LEADING_GAP_UNSUPPORTED',decision(result,'combined_2')['reasons'])
            self.assertIsNone(result['runs'][0]['left_endpoint'])
        empty=classify_history_window(**fixture(range(168)))
        self.assertEqual(empty['runs'][0]['position'],'all_missing')
        self.assertIn('NO_APPROVED_HISTORY',decision(empty,'combined_2')['reasons'])

    def test_natural_L24_is_unrepaired_suffix_not_short_gap_readmission(self):
        old=classify_history_window(**fixture([0,20,50]));recent=classify_history_window(**fixture([144]))
        self.assertTrue(old['natural_l24']['history_admissible'])
        self.assertFalse(decision(old,'combined_2')['history_admissible'])
        self.assertFalse(recent['natural_l24']['history_admissible'])
        self.assertTrue(decision(recent,'interior_1')['history_admissible'])

    def test_future_or_wrong_source_clock_cannot_supply_an_endpoint(self):
        for change in ('future_right','unknown_approved','wrong_slot','rejected_future'):
            args=fixture([166])
            if change=='future_right':args['source_observation_times'][167]=args['cutoff_epoch']-3600
            elif change=='unknown_approved':args['source_observation_times'][165]=None
            elif change=='wrong_slot':args['source_observation_times'][165]-=3600
            else:args['source_observation_times'][166]=args['cutoff_epoch']+3600
            with self.subTest(change=change),self.assertRaises(ValueError):classify_history_window(**args)

    def test_rejected_record_timestamp_does_not_turn_into_approved_support(self):
        args=fixture([30]);args['source_observation_times'][30]=int(args['history_times'][30])
        result=classify_history_window(**args)
        self.assertEqual(result['missing_total'],1);self.assertFalse(decision(result,'strict')['history_admissible'])
        self.assertTrue(decision(result,'interior_1')['additional_history_candidate'])

    def test_exact_L168_Cminus2_clocks_boolean_mask_and_simulated2h_are_required(self):
        for change in ('short','clock_gap','shift','float_clock','mask_int','bad_lag','cutoff_fractional','protected','source_length'):
            args=fixture()
            if change=='short':args['history_times']=args['history_times'][-24:]
            elif change=='clock_gap':args['history_times'][30]+=3600
            elif change=='shift':args['history_times']+=3600
            elif change=='float_clock':args['history_times']=args['history_times'].astype(float)
            elif change=='mask_int':args['approved_mask']=args['approved_mask'].astype(int)
            elif change=='bad_lag':args['simulated_lag_hours']=0
            elif change=='cutoff_fractional':args['cutoff_epoch']+=1
            elif change=='protected':args['cutoff_epoch']=ASSESSMENT_START
            else:args['source_observation_times']=args['source_observation_times'][:-1]
            with self.subTest(change=change),self.assertRaises(ValueError):classify_history_window(**args)

    def test_no_history_mask_time_source_or_target_mutation_and_identity_is_deterministic(self):
        args=fixture([30,167]);before=copy.deepcopy(args);targets=np.arange(18,dtype=float);saved_targets=targets.copy()
        first=classify_history_window(**args);again=classify_history_window(**args)
        self.assertEqual(first,again)
        np.testing.assert_array_equal(args['history_times'],before['history_times'])
        np.testing.assert_array_equal(args['approved_mask'],before['approved_mask'])
        self.assertEqual(args['source_observation_times'],before['source_observation_times'])
        np.testing.assert_array_equal(targets,saved_targets)
        self.assertFalse(first['execution']['targets_read_or_changed']);self.assertFalse(first['execution']['numerical_Q_values_read'])
        with self.assertRaises(TypeError):classify_history_window(**args,targets=targets)

    def test_policy_catalog_binds_both_caps_and_is_defensive_copy(self):
        catalog=policy_catalog();self.assertEqual(digest(catalog),POLICY_CATALOG_SHA256)
        for alias,policy_id in POLICY_IDS.items():
            cap=0 if alias=='strict' else int(alias[-1])
            self.assertEqual(catalog[policy_id]['per_run_cap_hours'],cap)
            self.assertEqual(catalog[policy_id]['total_missing_cap_hours'],cap)
        catalog[POLICY_IDS['combined_2']]['total_missing_cap_hours']=200
        self.assertEqual(policy_catalog()[POLICY_IDS['combined_2']]['total_missing_cap_hours'],2)
        with self.assertRaises(TypeError):POLICY_IDS['strict']='changed'


if __name__=='__main__':unittest.main()
