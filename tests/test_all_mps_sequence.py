"""All-MPS metadata planning; no Torch, optimizer or study-array reads."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

PATH=Path(__file__).resolve().parents[1]/"scripts/plan_all_mps_sequence.py"
SPEC=importlib.util.spec_from_file_location("all_mps_sequence_tests",PATH)
mps=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(mps)


class AllMPSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.parent,cls.full,cls.controls=mps.load_sources()

    def plan(self): return mps.build_plan(self.parent,self.full,self.controls)

    def test_all180_jobs_use_MPS_and_keep_membership_order_settings(self):
        result=self.plan()
        self.assertEqual(len(result['jobs']),180)
        for old,new in zip(self.parent['jobs'],result['jobs']):
            self.assertEqual(new['device'],'mps')
            for key in old:
                if key not in ('device','conditional_local_hours_by_epochs'): self.assertEqual(old[key],new[key],key)
        self.assertEqual([x['fits'] for x in result['seed_milestones']],[108,36,36])
        self.assertEqual(result['phases'],self.parent['phases'])
        self.assertEqual(result['required_worker_environment']['PYTORCH_ENABLE_MPS_FALLBACK'],'0')
        self.assertFalse(result['checkpoint_backend_migration_allowed'])

    def test_missing_MPS_controls_are_unknown_not_relabelled_CPU_rates(self):
        result=self.plan();coverage=result['timing_coverage']
        self.assertEqual((coverage['matching_geometry_jobs'],coverage['unmeasured_geometry_jobs']),(126,54))
        self.assertIsNone(coverage['complete_matched_geometry_campaign_estimate_hours'])
        self.assertEqual(len(coverage['missing_cells']),4)
        self.assertEqual([x['MPS_timing']['unmeasured_geometry_jobs'] for x in result['seed_milestones']],[30,12,12])
        for job in result['jobs']:
            self.assertNotIn('conditional_local_hours_by_epochs',job)
            self.assertNotIn('-cpu-',job['timing']['measurement_case_id'])
            self.assertFalse(job['timing']['proxy_is_upper_bound'])
            if job['timing']['matching_geometry_hours_by_epochs'] is None:
                self.assertIsNotNone(job['timing']['full_L168_proxy_hours_by_epochs'])

    def test_parent_and_benchmarks_are_not_modified(self):
        before=deepcopy((self.parent,self.full,self.controls));self.plan()
        self.assertEqual(before,(self.parent,self.full,self.controls))

    def test_current_rates_show_both_faster_and_slower_MPS_cases(self):
        rows={r['architecture']:r for r in mps.comparison(self.full)}
        self.assertGreater(rows['LSTM']['MPS_to_CPU_duration_ratio'],1)
        self.assertGreater(rows['GRU_Transformer']['MPS_to_CPU_duration_ratio'],1)
        self.assertLess(rows['Transformer']['MPS_to_CPU_duration_ratio'],1)
        self.assertLess(rows['Mamba2_Transformer']['MPS_to_CPU_duration_ratio'],1)

    def test_CPU_only_controls_do_not_enter_MPS_catalog(self):
        catalog=mps.matching_rates(self.full,self.controls)
        self.assertNotIn(('LSTM','q_only',168),catalog)
        self.assertNotIn(('GRU_Transformer','full',24),catalog)
        self.assertTrue(all(r['device']=='mps' for r in catalog.values()))

    def test_wrong_or_duplicated_measurement_rejects(self):
        altered=deepcopy(self.full)
        row=next(x for x in altered['cases'] if x['device']=='mps' and x['batch_size']==64)
        altered['cases'].append(deepcopy(row))
        with self.assertRaisesRegex(ValueError,'Duplicate'):mps.matching_rates(altered,self.controls)
        altered=deepcopy(self.full)
        row=next(x for x in altered['cases'] if x['device']=='mps' and x['batch_size']==64)
        row['environment']['device_fallback_enabled']=True
        with self.assertRaisesRegex(ValueError,'without fallback'):mps.matching_rates(altered,self.controls)

    def test_timing_cannot_cross_backends(self):
        job=deepcopy(self.parent['jobs'][0]);rate=mps.matching_rates(self.full,self.controls)[('LSTM','full',168)]
        with self.assertRaisesRegex(ValueError,'mismatch'):mps.hours(job,rate,50)
        job['device']='mps';self.assertGreater(mps.hours(job,rate,50),mps.hours(job,rate,20))

    def test_preflight_covers_exact_main_geometry_without_releasing_execution(self):
        result=self.plan();cells=result['required_main_MPS_preflight_geometries']
        self.assertEqual(len(cells),12)
        self.assertEqual({(c['architecture'],c['input_arm'],c['history_hours']) for c in cells},
                         {(a,arm,length) for a in mps.FAMILIES for arm,length in mps.ARMS})
        self.assertFalse(any(c['fresh_process_resume_and_best_inference_B64_qualified'] for c in cells))
        self.assertEqual(result['scientific_fits_started'],0)
        self.assertFalse(result['protected_assessment_access'])

    def test_supplemental_scenario_reconciles_to_jobs_without_becoming_estimate(self):
        result=self.plan()
        for epoch in ('20','50','100','200'):
            total=sum((j['timing']['matching_geometry_hours_by_epochs'] or j['timing']['full_L168_proxy_hours_by_epochs'])[epoch] for j in result['jobs'])
            self.assertAlmostEqual(total,result['timing_coverage']['supplemental_full_L168_proxy_scenario_hours'][epoch],places=10)
        self.assertFalse(any(m['MPS_timing']['supplemental_scenario_is_measured_complete_estimate'] for m in result['seed_milestones']))


if __name__=='__main__':unittest.main()
