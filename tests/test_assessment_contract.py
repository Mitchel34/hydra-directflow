"""Synthetic coordinate/identity tests; no protected values or model work."""
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import unittest
from hydra_directflow.assessment_contract import (
    origins, clock, expected_checkpoint_jobs, validate_checkpoint_roster,
    expected_primary_checkpoint_jobs, validate_primary_checkpoint_roster,
    nwm_support, validate_nwm_row_coordinates, paired_keys, NWM_REACHES)
from hydra_directflow.reference import ASSESSMENT_START, ASSESSMENT_END, HOUR


def utc(s):return int(datetime.fromisoformat(s.replace('Z','+00:00')).timestamp())


class AssessmentContractTests(unittest.TestCase):
    def test_full_origin_calendar_and_declared_tail(self):
        c=origins();self.assertEqual(len(c),8760);self.assertEqual(c[0],ASSESSMENT_START)
        self.assertEqual(c[-1],utc('2025-07-30T23:00:00Z'))
        self.assertEqual(clock(c[-1])['valid_times'][-1],utc('2025-07-31T17:00:00Z'))

    def test_history_is_exact_and_causal(self):
        c=clock(ASSESSMENT_START);self.assertEqual(len(c['history']),168)
        self.assertEqual(c['history'][0],utc('2024-07-23T23:00:00Z'))
        self.assertEqual(c['history'][-1],ASSESSMENT_START-2*HOUR)
        self.assertEqual(clock(ASSESSMENT_START,24)['history'][0],ASSESSMENT_START-25*HOUR)

    def test_origin_limits_and_no_silent_tail_purge(self):
        for c in [ASSESSMENT_START-HOUR,ASSESSMENT_END]:
            with self.assertRaises(ValueError):clock(c)
        self.assertTrue(clock(ASSESSMENT_END-HOUR))

    def test_non_hourly_or_ambiguous_types_rejected(self):
        for c in [ASSESSMENT_START+1,float(ASSESSMENT_START),True]:
            with self.assertRaises(ValueError):clock(c)
        for h in [12,169,True]:
            with self.assertRaises(ValueError):clock(ASSESSMENT_START,h)

    def test_all_existing_families_seeds_and_arms_retained(self):
        jobs=expected_checkpoint_jobs();self.assertEqual(len(jobs),108)
        self.assertEqual(len(validate_checkpoint_roster(jobs)),108)
        self.assertFalse(any('cv_' in x for x in jobs))

    def test_missing_duplicate_or_foreign_checkpoint_rejected(self):
        j=list(expected_checkpoint_jobs())
        for bad in [j[:-1],j[:-1]+[j[0]],j[:-1]+['cv_1__03161000__LSTM__full_L168__seed17']]:
            with self.assertRaises(ValueError):validate_checkpoint_roster(bad)

    def test_shortest_primary_pass_retains_all_models_gauges_and_seeds(self):
        jobs=expected_primary_checkpoint_jobs();self.assertEqual(len(jobs),36)
        self.assertEqual(len(validate_primary_checkpoint_roster(jobs)),36)
        self.assertTrue(all('__full_L168__' in j for j in jobs))
        with self.assertRaises(ValueError):validate_primary_checkpoint_roster(expected_checkpoint_jobs())

    def test_crosswalk_is_immutable(self):
        with self.assertRaises(TypeError):NWM_REACHES['03161000']=1

    def test_first_and_last_origin_all_gauges_and_supported_horizons(self):
        for origin in [ASSESSMENT_START,ASSESSMENT_END-HOUR]:
            for gauge,reach in NWM_REACHES.items():
                for h in range(1,17):
                    row=dict(feature_id=reach,ensemble=0,reference_time=origin-2*HOUR,
                             valid_time=origin+h*HOUR,lead_hours=h+2)
                    self.assertEqual(validate_nwm_row_coordinates(row,gauge_id=gauge,origin_horizon=h,origin=origin),
                                     (gauge,origin,h,origin+h*HOUR))

    def test_native_first_two_leads_are_not_future_origin_targets(self):
        out=nwm_support(ASSESSMENT_START,reference_time=ASSESSMENT_START-2*HOUR,native_leads=(1,2))
        self.assertFalse(any(r.supported for r in out))

    def test_two_hour_clock_supports_only_h1_to_h16(self):
        out=nwm_support(ASSESSMENT_START,reference_time=ASSESSMENT_START-2*HOUR,native_leads=range(1,19))
        self.assertEqual([r.origin_horizon for r in out if r.supported],list(range(1,17)))
        self.assertEqual((out[0].native_lead,out[15].native_lead),(3,18))
        self.assertEqual([r.reason for r in out[16:]],['NATIVE_HORIZON_UNAVAILABLE']*2)

    def test_missing_issue_is_not_called_a_noaa_upstream_gap(self):
        out=nwm_support(ASSESSMENT_START)
        self.assertEqual(out[0].reason,'ISSUE_ABSENT_IN_OWNED_SNAPSHOT')
        self.assertFalse(any(r.supported for r in out))

    def test_no_zero_lag_or_older_fallback_is_invented(self):
        for ref in [ASSESSMENT_START,ASSESSMENT_START-HOUR,ASSESSMENT_START-3*HOUR]:
            with self.assertRaises(ValueError):nwm_support(ASSESSMENT_START,reference_time=ref,native_leads=range(1,19))

    def test_invalid_native_leads_and_absent_issue_metadata_rejected(self):
        for leads in [(0,1),(18,19),(1,1),(True,),('1',)]:
            with self.assertRaises(ValueError):nwm_support(ASSESSMENT_START,reference_time=ASSESSMENT_START-2*HOUR,native_leads=leads)
        with self.assertRaises(ValueError):nwm_support(ASSESSMENT_START,native_leads=(1,))

    def test_one_missing_native_lead_only_masks_corresponding_horizon(self):
        out=nwm_support(ASSESSMENT_START,reference_time=ASSESSMENT_START-2*HOUR,native_leads=[h for h in range(1,19) if h!=5])
        self.assertFalse(out[2].supported);self.assertTrue(out[15].supported)
        self.assertEqual(out[2].reason,'NATIVE_LEAD_ABSENT_IN_OWNED_SNAPSHOT')

    def test_support_records_are_immutable(self):
        r=nwm_support(ASSESSMENT_START)[0]
        with self.assertRaises(FrozenInstanceError):r.supported=True

    def fixture(self):
        return dict(feature_id=6892192,ensemble=0,reference_time=ASSESSMENT_START-2*HOUR,
                    valid_time=ASSESSMENT_START+HOUR,lead_hours=3,streamflow='DO_NOT_READ')

    def test_metadata_check_never_reads_forecast_value(self):
        class Poison(dict):
            def __getitem__(self,k):
                if k=='streamflow':raise AssertionError('Numerical value accessed')
                return super().__getitem__(k)
        key=validate_nwm_row_coordinates(Poison(self.fixture()),gauge_id='03161000',origin_horizon=1,origin=ASSESSMENT_START)
        self.assertEqual(key,('03161000',ASSESSMENT_START,1,ASSESSMENT_START+HOUR))

    def test_cross_study_reach_or_ensemble_substitution_rejected(self):
        for field,value in [('feature_id',NWM_REACHES['03164000']),('ensemble',1),('ensemble',False)]:
            r=self.fixture();r[field]=value
            with self.assertRaises(ValueError):validate_nwm_row_coordinates(r,gauge_id='03161000',origin_horizon=1,origin=ASSESSMENT_START)

    def test_native_lead_cannot_be_mislabeled_origin_lead(self):
        for field,value in [('lead_hours',1),('valid_time',ASSESSMENT_START+3*HOUR),('reference_time',ASSESSMENT_START)]:
            r=self.fixture();r[field]=value
            with self.assertRaises(ValueError):validate_nwm_row_coordinates(r,gauge_id='03161000',origin_horizon=1,origin=ASSESSMENT_START)
        with self.assertRaises(ValueError):validate_nwm_row_coordinates(self.fixture(),gauge_id='03161000',origin_horizon=18,origin=ASSESSMENT_START)

    def test_nwm_and_target_missingness_never_changes_hydra_forecast_keys(self):
        keys=tuple(('03161000',ASSESSMENT_START,h,ASSESSMENT_START+h*HOUR) for h in range(1,19))
        out=paired_keys(keys,keys[:16],[k for k in keys if k[2]!=3])
        self.assertEqual(out['independent_hydra_forecast_keys'],frozenset(keys))
        self.assertEqual(len(out['paired_evaluable_keys']),15)
        self.assertIn(keys[17],out['independent_hydra_evaluable_keys'])

    def test_comparison_rejects_duplicate_shifted_or_unsupported_keys(self):
        k=('03161000',ASSESSMENT_START,1,ASSESSMENT_START+HOUR)
        for hydra,nwm,targets in [([k,k],[k],[k]),([k],[('03161000',ASSESSMENT_START,18,ASSESSMENT_START+18*HOUR)],[k]),
                                  ([('03161000',ASSESSMENT_START,1,ASSESSMENT_START+2*HOUR)],[k],[k]),
                                  ([('03161000',ASSESSMENT_START,1,float(ASSESSMENT_START+HOUR))],[k],[k]),
                                  ([list(k)],[k],[k])]:
            with self.assertRaises(ValueError):paired_keys(hydra,nwm,targets)


if __name__=='__main__':unittest.main()
