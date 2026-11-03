"""Fastest demonstrated local device plan; pure metadata, no model calls."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

PATH=Path(__file__).resolve().parents[1]/'scripts/plan_fastest_local_sequence.py'
SPEC=importlib.util.spec_from_file_location('fastest_local_tests',PATH)
planner=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(planner)


class FastestLocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.sources=planner.load_sources()

    def test_current_winners_and_all180_matched_timings(self):
        result=planner.build_plan(*self.sources)
        self.assertEqual(result['selected_devices'],{'LSTM':'cpu','Transformer':'mps','GRU_Transformer':'cpu','Mamba2_Transformer':'mps'})
        self.assertEqual(len(result['jobs']),180)
        self.assertEqual(sum(j['device']=='mps' for j in result['jobs']),90)
        self.assertEqual(sum(j['device']=='cpu' for j in result['jobs']),90)
        self.assertTrue(all(j['timing_measurement_case_id'] and j['timing_status'].startswith('MATCHED') for j in result['jobs']))
        self.assertFalse(result['all_MPS_alternative_is_active'])

    def test_memberships_seed_order_and_existing_runtime_stay_identical(self):
        old=self.sources[0];result=planner.build_plan(*self.sources)
        self.assertEqual(old['phases'],result['phases']);self.assertEqual(old['seed_milestones'],result['seed_milestones'])
        for before,after in zip(old['jobs'],result['jobs']):
            for key in before:self.assertEqual(before[key],after[key],key)
        self.assertEqual(result['seed_order'],[17,29,43])

    def test_no_implicit_CPU_fallback_or_checkpoint_device_migration(self):
        result=planner.build_plan(*self.sources)
        self.assertEqual(result['required_worker_environment']['PYTORCH_ENABLE_MPS_FALLBACK'],'0')
        self.assertIn('NO_AUTOMATIC_BACKEND_SWITCH',result['backend_failure_policy'])
        self.assertFalse(result['checkpoint_backend_migration_allowed'])
        self.assertEqual(result['scientific_fits_started'],0)
        self.assertFalse(result['protected_assessment_access'])

    def test_changed_winner_cannot_reuse_old_runtime(self):
        parent,full,controls=deepcopy(self.sources)
        row=next(r for r in full['cases'] if r['architecture']=='LSTM' and r['device']=='mps' and r['batch_size']==64)
        for sample in row['training_samples']:sample['copy_seconds']=0.;sample['update_seconds']=.000001
        with self.assertRaisesRegex(ValueError,'new timing attribution'):planner.build_plan(parent,full,controls)

    def test_missing_matched_control_blocks_unlabelled_estimate(self):
        parent,full,controls=deepcopy(self.sources)
        controls['cases']=[r for r in controls['cases'] if r['id']!='Q_ONLY_L168-LSTM-cpu-B64']
        with self.assertRaisesRegex(ValueError,'matched actual-backend'):planner.build_plan(parent,full,controls)


if __name__=='__main__':unittest.main()
