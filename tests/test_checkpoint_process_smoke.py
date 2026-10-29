"""Budget guards only; no Torch import, workers, or optimizer updates."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/checkpoint_process_smoke.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_process_smoke_guard_tests", MODULE_PATH)
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


class CheckpointProcessGuardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.previous_root = SMOKE.ROOT
        SMOKE.ROOT = Path(self.directory.name)

    def tearDown(self):
        SMOKE.ROOT = self.previous_root
        self.directory.cleanup()

    def test_precharge_consumes_failed_attempt_allowance_and_stops_at_cap(self):
        report = SMOKE.ROOT / "receipt.json"
        self.assertEqual(SMOKE.precharge(report, "attempt-1")["precharged_updates"], 6)
        self.assertEqual(SMOKE.precharge(report, "attempt-2")["precharged_updates"], 12)
        ledger = SMOKE.ROOT / ".cache/checkpoint-process-smoke-update-budget.json"
        before = ledger.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "allowance exhausted"):
            SMOKE.precharge(report, "attempt-3")
        self.assertEqual(ledger.read_bytes(), before)

    def test_missing_ledger_with_existing_receipt_cannot_reset_allowance(self):
        report = SMOKE.ROOT / "receipt.json"
        report.write_text(json.dumps({"status": "FAIL"}))
        with self.assertRaisesRegex(RuntimeError, "missing update ledger"):
            SMOKE.precharge(report, "retry")
        self.assertFalse((SMOKE.ROOT / ".cache/checkpoint-process-smoke-update-budget.json").exists())


if __name__ == "__main__":
    unittest.main()
