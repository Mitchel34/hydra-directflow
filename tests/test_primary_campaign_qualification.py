"""Synthetic array / mocked-controller tests; zero model or optimizer calls."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import qualify_primary_campaign as q


class PrimaryQualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.stack = ExitStack()
        for name, value in {
            'ROOT': self.root, 'RUN': self.root / '.local/primary-campaign-qualification-v1',
            'REPORT': self.root / 'reports/primary-campaign-qualification-v1.json',
            'LEDGER': self.root / '.cache/primary-campaign-qualification-budget-v1.json',
            'LOCK': self.root / '.cache/primary-campaign-qualification-budget-v1.lock'}.items():
            self.stack.enter_context(patch.object(q, name, value))
        self.stack.enter_context(patch.object(q, 'implementation', return_value={'code': {'synthetic': 'a' * 64}, 'dependencies': {}}))
        self.prepared = q.plan()

    def tearDown(self):
        self.stack.close(); self.temp.cleanup()

    def prepare(self):
        record = q.precharge(self.prepared, q.RUN)
        q.mkdir_durable(q.RUN); q.atomic_json(q.RUN / 'plan.json', self.prepared)
        return record

    def test_scope_default_plan_has12_geometries36_updates12_scalers(self):
        self.assertEqual(len(q.cases()), 12)
        self.assertEqual(len(q.workers()), 36)
        self.assertEqual(sum(w['updates'] for w in q.workers()), 36)
        self.assertEqual(sum(w['scaler_fits'] for w in q.workers()), 12)
        self.assertEqual({(c['architecture'], c['device']) for c in q.cases()}, {
            ('LSTM', 'cpu'), ('Transformer', 'mps'), ('GRU_Transformer', 'cpu'), ('Mamba2_Transformer', 'mps')})
        self.assertEqual({(c['arm'], c['history_hours'], c['feature_schema']) for c in q.cases()}, {
            ('full168', 168, q.FULL), ('q168', 168, q.Q_ONLY), ('full24', 24, q.FULL)})
        self.assertEqual((q.MAX_SECONDS, q.MAX_BYTES, q.MPS_BYTES), (900, 4 * 1024**3, 2 * 1024**3))
        with patch.object(sys, 'argv', ['qualification']), patch.object(q, 'execute') as execute, patch('builtins.print'):
            q.main(); execute.assert_not_called()
        self.assertFalse(q.LEDGER.exists())

    def test_separate_consumed_allowance_is_durable_and_cannot_move_or_replay(self):
        old = self.root / 'old-ledger.json'; old.write_text('{"consumed":16}')
        before = old.read_bytes(); self.prepare(); charge = q.LEDGER.read_bytes()
        with self.assertRaisesRegex(ValueError, 'consumed'):
            q.precharge(self.prepared, q.RUN)
        with self.assertRaisesRegex(ValueError, 'fixed'):
            q.precharge(self.prepared, self.root / '.local/other')
        self.assertEqual(old.read_bytes(), before); self.assertEqual(q.LEDGER.read_bytes(), charge)
        first = q.workers()[0]; q.claim(q.RUN, first, self.prepared)
        record = q.read_json(q.LEDGER)
        self.assertEqual(record['workers'][first['id']]['status'], 'CLAIMED')
        with self.assertRaisesRegex(ValueError, 'replay'):
            q.claim(q.RUN, first, self.prepared)
        self.assertEqual(record['precharged_optimizer_updates'], 36)
        self.assertEqual(record['precharged_synthetic_scaler_fits'], 12)

    def test_resume_and_inference_require_authenticated_passed_dependencies(self):
        self.prepare(); base, resume, infer = q.workers()[:3]
        for worker in (resume, infer):
            with self.assertRaisesRegex(ValueError, 'has not passed'):
                q.claim(q.RUN, worker, self.prepared)
        directory = q.RUN / base['id']; q.mkdir_durable(directory)
        q.atomic_json(directory / 'result.json', {'synthetic': True})
        record = q.read_json(q.LEDGER)
        record['workers'][base['id']] = {'status': 'PASSED', 'result_sha256': q.sha((directory / 'result.json').read_bytes())}
        q.atomic_json(q.LEDGER, record)
        q.claim(q.RUN, resume, self.prepared)
        with self.assertRaisesRegex(ValueError, 'has not passed'):
            q.claim(q.RUN, infer, self.prepared)
        q.atomic_json(directory / 'result.json', {'changed': True})
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            q.claim(q.RUN, infer, self.prepared)

    def test_wrong_hash_expired_unreserved_worker_and_sync_failure_do_no_work(self):
        with patch.object(q, 'supervise') as launch:
            with self.assertRaisesRegex(ValueError, 'reviewed'):
                q.execute(self.prepared, 'b' * 64)
            launch.assert_not_called()
        with patch.object(q.os, 'fsync', side_effect=OSError('synthetic failure')), patch.object(q, 'supervise') as launch:
            with self.assertRaises(OSError):
                q.execute(self.prepared, self.prepared['plan_sha256'])
            launch.assert_not_called()
        # A separate temporary root avoids repairing or deleting any partial charge.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); run = root / 'run'
            with patch.object(q, 'ROOT', root), patch.object(q, 'RUN', run), patch.object(q, 'LEDGER', root / 'missing'):
                with self.assertRaises((FileNotFoundError, ValueError)):
                    q.claim(run, q.workers()[0], self.prepared)

    def test_expired_or_changed_plan_claim_stops_before_worker_imports(self):
        self.prepare(); record = q.read_json(q.LEDGER)
        record['deadline_epoch'] = time.time() - 1; q.atomic_json(q.LEDGER, record)
        with self.assertRaisesRegex(ValueError, 'deadline'):
            q.claim(q.RUN, q.workers()[0], self.prepared)
        record['deadline_epoch'] = time.time() + 10; q.atomic_json(q.LEDGER, record)
        with self.assertRaisesRegex(ValueError, 'precharged'):
            q.claim(q.RUN, q.workers()[0], {**self.prepared, 'plan_sha256': 'f' * 64})

    def test_fixed_paths_reject_symlink_and_external_outputs(self):
        with self.assertRaisesRegex(ValueError, 'fixed'):
            q.validate_paths(self.root / 'outside', q.REPORT)
        with self.assertRaisesRegex(ValueError, 'fixed'):
            q.validate_paths(q.RUN, self.root / 'outside.json')
        q.mkdir_durable(q.RUN.parent)
        outside = self.root / 'elsewhere'; outside.mkdir(); q.RUN.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'fixed'):
            q.validate_paths(q.RUN)

    def test_all_fixture_geometries_are_matched_label_free_outer_and_causal(self):
        from hydra_directflow.training import validate_development_bundle
        from hydra_directflow.inference import input_only_view, validate_inputs
        for case in q.cases()[:3]:
            with self.subTest(case=case['id']):
                data, source, cohort = q.fixture(case)
                self.assertEqual(validate_development_bundle(data, case['history_hours'], feature_schema=case['feature_schema']), 128)
                q.validate_roles(data)
                outer, outer_source, outer_cohort = q.fixture(case, outer=True)
                self.assertNotIn('target', outer); self.assertNotIn('target_mask', outer)
                self.assertNotEqual(source, outer_source); self.assertNotEqual(cohort, outer_cohort)
                self.assertFalse(set(data['case_id']) & set(outer['case_id']))
                inputs = input_only_view(outer, inference_source_identity=outer_source)
                self.assertEqual(validate_inputs(inputs, history_hours=case['history_hours'], feature_schema=case['feature_schema'],
                    expected_cohort_admission_digest=outer_cohort, expected_gauge_id=q.GAUGE), 64)
                if case['arm'] == 'q168':
                    self.assertEqual(data['weather'].shape, (128, 18, 1)); self.assertEqual(data['state'].shape, (128, 0))
        full, _, common = q.fixture(q.cases()[0]); short, _, short_cohort = q.fixture(q.cases()[2])
        np.testing.assert_array_equal(short['history'], full['history'][:, -24:])
        np.testing.assert_array_equal(short['target'], full['target']); self.assertEqual(common, short_cohort)

    def test_counters_reject_booleans_refits_and_overages(self):
        resume = q.workers()[1]
        self.assertEqual(q.checked_counts({'optimizer_updates_entered': 1, 'optimizer_updates_confirmed': 0}, resume), (1, 0, 0, 0))
        for bad in ({'optimizer_updates_entered': True}, {'optimizer_updates_entered': 2},
                    {'optimizer_updates_confirmed': 1}, {'scaler_fits_entered': 1}):
            with self.assertRaisesRegex(ValueError, 'counters'):
                q.checked_counts(bad, resume)

    def test_supervisor_interrupt_reaps_and_preserves_entered_confirmed_counts(self):
        self.prepare(); declaration = q.workers()[0]
        directory = q.RUN / declaration['id']; q.mkdir_durable(directory)
        q.atomic_json(directory / 'status.json', dict(optimizer_updates_entered=2, optimizer_updates_confirmed=1,
                     scaler_fits_entered=1, scaler_fits_confirmed=1, peak_rss_bytes=100, mps_driver_bytes=0))
        class Child:
            pid = 123; returncode = None
            def poll(self): return self.returncode
            def kill(self): self.returncode = -9
            def wait(self, timeout=None): return self.returncode
        child = Child(); process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100))
        with patch.object(q.subprocess, 'Popen', return_value=child), patch('psutil.Process', return_value=process), \
             patch.object(q.time, 'sleep', side_effect=KeyboardInterrupt):
            attempt = q.supervise(q.RUN, declaration, time.time() + 60, {}, self.prepared['plan_sha256'])
        self.assertEqual(child.returncode, -9); self.assertFalse(attempt['passed']); self.assertFalse(attempt['count_exact'])
        self.assertEqual((attempt['optimizer_updates_entered'], attempt['optimizer_updates_confirmed']), (2, 1))
        self.assertEqual(attempt['scaler_fits_confirmed'], 1); self.assertEqual(attempt['error_type'], 'KeyboardInterrupt')
        self.assertEqual(q.read_json(q.LEDGER)['precharged_optimizer_updates'], 36)

    def test_fast_worker_final_memory_violation_cannot_pass(self):
        self.prepare(); declaration = q.workers()[0]
        directory = q.RUN / declaration['id']; q.mkdir_durable(directory)
        value = dict(status='PASSED', pid=123, worker_id=declaration['id'], plan_sha256=self.prepared['plan_sha256'],
                     optimizer_updates_entered=2, optimizer_updates_confirmed=2, scaler_fits_entered=1, scaler_fits_confirmed=1,
                     peak_rss_bytes=q.MAX_BYTES, mps_driver_bytes=0)
        q.atomic_json(directory / 'result.json', value)
        child = SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)
        process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100))
        with patch.object(q.subprocess, 'Popen', return_value=child), patch('psutil.Process', return_value=process):
            attempt = q.supervise(q.RUN, declaration, time.time() + 60, {}, self.prepared['plan_sha256'])
        self.assertFalse(attempt['passed']); self.assertEqual(attempt['optimizer_updates_confirmed'], 2)
        self.assertEqual(attempt['error_type'], 'ValueError')

    def test_success_exit_with_wrong_result_identity_is_not_a_pass(self):
        self.prepare(); declaration = q.workers()[0]
        directory = q.RUN / declaration['id']; q.mkdir_durable(directory)
        result = dict(status='PASSED', pid=123, worker_id=declaration['id'], plan_sha256='wrong',
                      optimizer_updates_entered=2, optimizer_updates_confirmed=2,
                      scaler_fits_entered=1, scaler_fits_confirmed=1, peak_rss_bytes=100, mps_driver_bytes=0)
        q.atomic_json(directory / 'result.json', result)
        child = SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)
        process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100))
        with patch.object(q.subprocess, 'Popen', return_value=child), patch('psutil.Process', return_value=process):
            attempt = q.supervise(q.RUN, declaration, time.time() + 60, {}, self.prepared['plan_sha256'])
        self.assertFalse(attempt['passed']); self.assertFalse(attempt['count_exact'])
        self.assertEqual(attempt['optimizer_updates_confirmed'], 2)

    def test_comparison_requires_rng_exactness_and_best_not_last(self):
        keys = ('model_sha256', 'optimizer_sha256', 'sampler', 'scaler', 'stopping', 'best_model_sha256',
                'epoch_records', 'cpu_rng_sha256', 'mps_rng_sha256', 'array_sha256', 'source_identity', 'cohort_digest')
        base = dict.fromkeys(keys, 'a' * 64)
        base.update(trace=[{'indices': [1], 'loss': 2}, {'indices': [0], 'loss': 1}], pid=1,
                    best_prediction_sha256='p', best_differs_from_last=True, parameter_count=1,
                    update_seconds=[.1, .1], selected_best_epoch=1, completed_epochs=2)
        resumed = {**base, 'trace': base['trace'][1:], 'pid': 2}
        inferred = dict(pid=3, exact_fresh_best_predictions=True, best_prediction_sha256='p', synthetic_score_status='PRIMARY_UNAVAILABLE')
        self.assertTrue(q.compare_case(base, resumed, inferred)['exact_next_loss_samples_weights_optimizer_rng'])
        for key in ('cpu_rng_sha256', 'mps_rng_sha256', 'optimizer_sha256', 'sampler'):
            with self.assertRaisesRegex(ValueError, 'restored state'):
                q.compare_case(base, {**resumed, key: 'different'}, inferred)
        with self.assertRaisesRegex(ValueError, 'best-only'):
            q.compare_case({**base, 'best_differs_from_last': False}, resumed, inferred)


if __name__ == '__main__':
    unittest.main()
