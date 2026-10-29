"""Zero-update checks of fixed benchmark geometry, accounting and statistics."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PATH = Path(__file__).resolve().parents[1] / "scripts/benchmark_local.py"
SPEC = importlib.util.spec_from_file_location("local_benchmark_guard_tests", PATH)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class BenchmarkGuardTests(unittest.TestCase):
    def test_fixed_matrix_and_update_budget(self):
        cases = BENCHMARK.matrix()
        self.assertEqual(len(cases), 20)
        self.assertEqual(len({c['id'] for c in cases}), 20)
        self.assertEqual(sum(c['device'] == 'mps' for c in cases), 16)
        self.assertEqual(BENCHMARK.PLANNED_UPDATES, 800)
        self.assertLessEqual(BENCHMARK.PLANNED_UPDATES, BENCHMARK.MAX_UPDATES)
        self.assertEqual(BENCHMARK.specification()['width'], 64)
        self.assertEqual(BENCHMARK.specification()['horizons_hours'], list(range(1, 19)))

    def test_empirical_p95_and_throughput_use_measured_samples(self):
        samples = [{'copy_seconds': i / 1000, 'update_seconds': i / 100} for i in range(1, 31)]
        result = BENCHMARK.summaries(samples, 64, 64 * 18)
        self.assertAlmostEqual(result['update_seconds']['median'], .155)
        self.assertEqual(result['update_seconds']['empirical_p95'], .29)
        self.assertAlmostEqual(result['examples_per_second_at_median_update'], 64 / .155)
        self.assertAlmostEqual(result['valid_target_cells_per_second_at_median_update'], 64 * 18 / .155)
        with self.assertRaises(ValueError):
            BENCHMARK.summaries([{'copy_seconds': 0, 'update_seconds': 1}], 1, 18)

    def test_precharge_cannot_refund_or_repeat_full_matrix(self):
        previous = BENCHMARK.ROOT
        try:
            with tempfile.TemporaryDirectory() as directory:
                BENCHMARK.ROOT = Path(directory)
                report = BENCHMARK.ROOT / 'receipt.json'
                ledger = BENCHMARK.precharge(report, 'first')
                self.assertEqual(ledger['precharged_updates'], 800)
                path = BENCHMARK.ROOT / '.cache/local-throughput-update-budget.json'
                before = path.read_bytes()
                with self.assertRaisesRegex(RuntimeError, 'allowance exhausted'):
                    BENCHMARK.precharge(report, 'second')
                self.assertEqual(path.read_bytes(), before)
        finally:
            BENCHMARK.ROOT = previous

    def test_existing_receipt_with_missing_ledger_fails_closed(self):
        previous = BENCHMARK.ROOT
        try:
            with tempfile.TemporaryDirectory() as directory:
                BENCHMARK.ROOT = Path(directory)
                report = BENCHMARK.ROOT / 'receipt.json'
                report.write_text(json.dumps({'status': 'PARTIAL_OR_FAILED'}))
                with self.assertRaisesRegex(RuntimeError, 'missing ledger'):
                    BENCHMARK.precharge(report, 'retry')
        finally:
            BENCHMARK.ROOT = previous


if __name__ == '__main__':
    unittest.main()
