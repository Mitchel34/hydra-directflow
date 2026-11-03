"""Synthetic census arithmetic and policy composition; no user data or models."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from hydra_directflow.gap_policies import classify_history_window, POLICY_IDS
from hydra_directflow.coverage_analysis import reference_arrays, HOUR
from hydra_directflow.local_data import utc_epoch

SPEC=importlib.util.spec_from_file_location('gap_recovery_audit',Path(__file__).resolve().parents[1]/'scripts/audit_gap_recovery.py')
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)


class GapCensusTests(unittest.TestCase):
    def setUp(self): self.origin=utc_epoch('2023-08-10T00:00:00Z')

    def classification(self,missing):
        times=self.origin-np.arange(169,1,-1)*HOUR;mask=np.ones(168,bool);mask[missing]=False
        return classify_history_window(history_times=times,approved_mask=mask,
            source_observation_times=[int(t) if good else None for t,good in zip(times,mask)],cutoff_epoch=self.origin)

    def test_all_origin_denominator_does_not_hide_unknown_targets(self):
        origins=self.origin+np.arange(3)*HOUR;targets=np.ones((3,18),bool);targets[1,:]=False
        primary=np.array([True,False,False]);policies={'strict':np.array([True,True,False]),'perfect_q':np.ones(3,bool)}
        out=audit.summarize_support(origins,targets,primary,policies)
        strict=out['policies']['strict'];perfect=out['policies']['perfect_q']
        self.assertEqual(strict['forecast_available_by_lead'],[2]*18)
        self.assertEqual(strict['target_evaluable_by_lead'],[1]*18)
        self.assertEqual(strict['evaluable_fraction_all_origins_by_lead'],[1/3]*18)
        self.assertEqual(strict['evaluable_fraction_valid_targets_by_lead'],[.5]*18)
        self.assertEqual(perfect['evaluable_fraction_all_origins_by_lead'],[2/3]*18)
        self.assertEqual(perfect['all18_evaluable_cases'],2);self.assertEqual(out['unchanged_primary_complete_h18_cases'],1)

    def test_per_lead_evaluable_support_is_not_all18_case_support(self):
        origins=self.origin+np.arange(2)*HOUR;targets=np.ones((2,18),bool);targets[1,0]=False
        out=audit.summarize_support(origins,targets,np.array([True,False]),{'candidate':np.ones(2,bool)})
        self.assertEqual(out['policies']['candidate']['target_evaluable_by_lead'],[1]+[2]*17)
        self.assertEqual(out['policies']['candidate']['all18_evaluable_cases'],1)

    def test_unique_hours_deduplicate_overlapping_origin_lead_cells(self):
        origins=self.origin+np.arange(2)*HOUR;mask=np.ones((2,18),bool)
        out=audit.summarize_support(origins,mask,np.ones(2,bool),{'strict':np.ones(2,bool)})
        self.assertEqual(out['unique_requested_target_hours'],19)
        self.assertEqual(sum(out['policies']['strict']['target_evaluable_by_lead']),36)
        self.assertEqual(out['policies']['strict']['unique_evaluable_target_hours'],19)

    def test_zero_valid_target_denominator_is_explicitly_undefined(self):
        out=audit.summarize_support(np.array([self.origin]),np.zeros((1,18),bool),np.zeros(1,bool),{'candidate':np.ones(1,bool)})
        self.assertEqual(out['policies']['candidate']['evaluable_fraction_valid_targets_by_lead'],[None]*18)
        self.assertEqual(out['policies']['candidate']['evaluable_fraction_all_origins_by_lead'],[0.]*18)

    def test_history_run_position_length_and_cutoff_age_are_exact(self):
        mask=np.ones(168,bool);mask[[0,10,11,167]]=False;runs=audit.gap_runs(mask)
        self.assertEqual([(r['position'],r['length_hours']) for r in runs],[('LEADING',1),('INTERIOR',2),('TRAILING',1)])
        self.assertEqual(runs[0]['farthest_missing_age_hours'],169);self.assertEqual(runs[-1]['nearest_missing_age_hours'],2)
        self.assertEqual(audit.gap_runs(np.zeros(168,bool))[0]['position'],'ALL_MISSING')

    def test_policy_recovery_remains_separate_from_other_input_blockers(self):
        options=audit.history_options(self.classification([40]),POLICY_IDS)
        self.assertTrue(options['interior_1']);self.assertFalse(options['trailing_1']);self.assertFalse(options['strict_l168'])
        self.assertFalse(options['interior_1'] and False)  # Non-Q mask cannot be repaired by a Q option.
        self.assertEqual(audit.failure_combination(False,[False]*4,False,False),
            'Q_HISTORY|WEATHER_PRECIPITATION|WEATHER_TEMPERATURE|WEATHER_HUMIDITY|WEATHER_SNOW_WATER_EQUIVALENT|SOIL|CALENDAR')

    def test_optimistic_total_uncapped_ceiling_does_not_ignore_leading_or_long_gaps(self):
        repeated=audit.history_options(self.classification([10,30,50]),POLICY_IDS)
        self.assertFalse(repeated['combined_2']);self.assertTrue(repeated['optimistic_combined_run2_total_uncapped'])
        for missing in ([0,20,30],[10,11,12]):
            value=audit.history_options(self.classification(missing),POLICY_IDS)
            self.assertFalse(value['optimistic_combined_run2_total_uncapped'])

    def test_natural_l24_is_an_unrepaired_suffix_not_an_l168_fill(self):
        older=audit.history_options(self.classification([10,11,12]),POLICY_IDS)
        self.assertTrue(older['natural_l24']);self.assertFalse(older['combined_2'])
        trailing=audit.history_options(self.classification([167]),POLICY_IDS)
        self.assertFalse(trailing['natural_l24']);self.assertTrue(trailing['trailing_1'])

    def test_highflow_uses_only_known_approved_targets_and_keeps_unknown_count(self):
        origins=np.array([self.origin]);ref=reference_arrays([(self.origin+i*HOUR,float(i),('A',)) for i in range(20)],
            start=self.origin,end=self.origin+20*HOUR)
        ref['admitted'][18]=False;ref['values'][18]=np.nan;target=ref['admitted'][np.arange(1,19)][None,:]
        out=audit.high_flow_support(origins,target,np.array([False]),{'candidate':np.array([True])},ref,{'q95_cms':16.})
        self.assertEqual(sum(out['known_high_target_cells_by_lead']),1)
        self.assertEqual(out['unknown_target_cells_by_lead'][-1],1)
        self.assertIsNone(out['policies']['candidate']['known_high_coverage_fraction_by_lead'][-1])
        self.assertEqual(out['unique_known_high_target_hours'],1)

    def test_event_peak_without_forecast_input_remains_in_denominator(self):
        event=dict(event_id='observed',observed_peak_rank=1,representative_top3=True,start_utc='x',end_exclusive_utc='y',
            reference_requested_hours=120,reference_approved_hours=119,reference_unusable_hours=1,left_truncated=False,right_truncated=False,
            start_epoch=self.origin,end_epoch=self.origin+120*HOUR,peak_epoch=self.origin+HOUR)
        out=audit.event_support(event,np.array([self.origin]),np.ones((1,18),bool),np.array([False]),{'candidate':np.array([False])})
        self.assertFalse(out['policies']['candidate']['observed_peak_any_lead'])
        self.assertEqual(out['valid_target_cells_by_lead'],[1]*18)
        self.assertEqual(out['policies']['candidate']['target_evaluable_cells_by_lead'],[0]*18)
        self.assertEqual(out['observed_peak_requested_by_lead'],[True]+[False]*17)

    def test_event_detail_packaging_keeps_frontier_and_hash_bound_bounded_files(self):
        original=dict(source_plan_sha256='a'*64,snapshot_sha256='b'*64,coverage_report_sha256='c'*64,policy_ids={'strict':'v1'},
            split_coverage=[{'requested_origins':12}],fold_high_flow=[dict(gauge_id='03161000',fold='final_development',thresholds={},outer_window={},
                known_high_target_cells_by_lead=[1]*18,representative_top3=[{'event_id':'peak','policies':{'strict':{'count':1}}}])])
        summary,files=audit.package_report(original,'census')
        self.assertEqual(summary['split_coverage'],original['split_coverage'])
        self.assertEqual(summary['fold_high_flow'][0]['representative_top3_event_ids'],['peak'])
        self.assertIn('representative_top3',original['fold_high_flow'][0])
        self.assertNotIn('representative_top3',summary['fold_high_flow'][0])
        self.assertEqual(len(files),3)
        for name,value,identity in files:
            self.assertLessEqual(len(audit.report_bytes(value)),1024**2)
            self.assertEqual(identity['sha256'],audit.sha(audit.report_bytes(value)))
            self.assertEqual(identity['bytes'],len(audit.report_bytes(value)))

    def test_primary_cases_cannot_be_silently_removed_by_a_candidate(self):
        with self.assertRaisesRegex(ValueError,'removed'):
            audit.summarize_support(np.array([self.origin]),np.ones((1,18),bool),np.array([True]),{'bad':np.array([False])})


if __name__=='__main__':unittest.main()
