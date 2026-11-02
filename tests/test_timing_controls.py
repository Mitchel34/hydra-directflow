"""Zero-update timing-control geometry, cohort and allowance checks."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "scripts/benchmark_timing_controls.py"
SPEC = importlib.util.spec_from_file_location("timing_controls_tests", PATH)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def fixture():
    origins = np.array([1609459200 + i * 86400 for i in range(4)], dtype=np.int64)
    arrays = dict(site_id=np.zeros(4, dtype=np.int64), origin_time=origins,
                  history_time=origins[:, None] - np.arange(169, 1, -1)[None, :] * 3600,
                  target_time=origins[:, None] + np.arange(1, 19)[None, :] * 3600,
                  case_id=np.array(["case-" + str(i) for i in range(4)]), source_cycle=np.array(["20210101_00Z"] * 4))
    for name, shape in (("history", (4, 168, 2)), ("weather", (4, 18, 4)), ("state", (4, 2)), ("target", (4, 18))):
        arrays[name] = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 1
        arrays[name + "_mask"] = np.ones(shape, bool)
    return arrays


class TimingControlTests(unittest.TestCase):
    def test_fixed_eight_cases_and_separate200_update_allowance(self):
        cases = BENCH.matrix()
        self.assertEqual(len(cases), len({case["id"] for case in cases}))
        self.assertEqual(len(cases), 8)
        self.assertEqual(len(cases) * (BENCH.WARMUP + BENCH.SAMPLES), 200)
        self.assertEqual(sum(case["device"] == "cpu" for case in cases), 4)
        self.assertTrue(all(case["batch_size"] == 64 for case in cases))
        self.assertEqual({case["input_mode"] for case in cases}, {"Q_ONLY_L168", "FULL_L24"})

    def test_parameter_geometry_is_explicitly_not_matched(self):
        cases = BENCH.matrix()
        for architecture in BENCH.base.ARCHITECTURES:
            full = next(case for case in cases if case["architecture"] == architecture and case["input_mode"] == "FULL_L24")
            q = next(case for case in cases if case["architecture"] == architecture and case["input_mode"] == "Q_ONLY_L168")
            self.assertEqual(full["expected_parameter_count"] - q["expected_parameter_count"], 640)

    def test_L24_is_exact_original_suffix_with_identical_case_and_target_identities(self):
        arrays = fixture()
        before = BENCH.base._array_digest(arrays)
        selected, mapped, features = BENCH.prepare_mode(arrays, "FULL_L24", cohort_admission_digest="a" * 64)
        for name in ("history", "history_mask", "history_time"):
            np.testing.assert_equal(selected[name], arrays[name][:, -24:])
        for name in ("case_id", "target", "target_mask", "target_time", "origin_time"):
            np.testing.assert_equal(selected[name], arrays[name])
        baseline = BENCH.base._map_arrays(arrays)
        np.testing.assert_equal(mapped["history"], baseline["history"][:, -24:])
        np.testing.assert_equal(mapped["target"], baseline["target"])
        self.assertEqual(features["weather"], BENCH.base.FEATURES["weather"])
        self.assertEqual(BENCH.base._array_digest(arrays), before)

    def test_Q_only_never_fetches_original_weather_or_soil_and_uses_same_Q_units(self):
        arrays = fixture()
        baseline = BENCH.base._map_arrays(arrays)
        class Poison(dict):
            def __getitem__(self, name):
                if name in ("weather", "weather_mask", "state", "state_mask"):
                    raise AssertionError("Original weather/soil accessed")
                return super().__getitem__(name)
        selected, mapped, features = BENCH.prepare_mode(Poison(arrays), "Q_ONLY_L168", cohort_admission_digest="a" * 64)
        for name in ("history", "history_mask", "target", "target_mask"):
            np.testing.assert_equal(mapped[name], baseline[name])
        self.assertEqual(mapped["state"].shape, (4, 0))
        self.assertEqual(features["weather"], ("horizon_hours",))
        np.testing.assert_allclose(mapped["weather"][0, :, 0], np.arange(1, 19) / 18, rtol=1e-7)
        self.assertEqual(selected["cohort_admission_digest"].item(), "a" * 64)

    def test_both_modes_preserve_matching_cohort_digests(self):
        arrays = fixture()
        split = dict(train=[0, 1], validation=[2, 3], purged=[])
        before = BENCH.cohort_identities(arrays, split)
        for mode in BENCH.MODES:
            selected, _, _ = BENCH.prepare_mode(arrays, mode, cohort_admission_digest="a" * 64)
            self.assertEqual(BENCH.cohort_identities(selected, split), before)

    def test_target_changes_never_change_forward_inputs(self):
        arrays = fixture()
        _, first, _ = BENCH.prepare_mode(arrays, "Q_ONLY_L168", cohort_admission_digest="a" * 64)
        arrays["target"] *= 99
        _, second, _ = BENCH.prepare_mode(arrays, "Q_ONLY_L168", cohort_admission_digest="a" * 64)
        for name in BENCH.INPUT_NAMES:
            np.testing.assert_equal(first[name], second[name])
        self.assertFalse(np.array_equal(first["target"], second["target"]))

    def test_protected_coordinate_guard_precedes_value_access(self):
        arrays = fixture()
        shift = BENCH.base.epoch("2024-08-01T00:00:00Z") - arrays["origin_time"][0]
        for name in ("origin_time", "history_time", "target_time"):
            arrays[name] += shift
        class Poison(dict):
            def __getitem__(self, name):
                if name in ("history", "weather", "state", "target"):
                    raise AssertionError("Values accessed before coordinate gate")
                return super().__getitem__(name)
        for mode in BENCH.MODES:
            with self.assertRaisesRegex(ValueError, "Protected"):
                BENCH.prepare_mode(Poison(arrays), mode, cohort_admission_digest="a" * 64)

    def test_plan_only_never_charges_or_spawns(self):
        parent = {"plan_sha256": "b" * 64, "spec": {"source": {}, "fixture_receipt_sha256": "c" * 64, "source_receipts_sha256": "d" * 64}}
        with patch.object(BENCH.base, "plan", return_value=parent), patch.object(BENCH, "implementation_identity", return_value={}), \
             patch.object(BENCH.subprocess, "Popen") as spawn, patch.object(BENCH, "precharge") as charge:
            plan = BENCH.plan("unused")
            self.assertEqual(plan["status"], "PLANNED_NOT_EXECUTED")
            self.assertEqual(plan["optimizer_updates"], 0)
            self.assertEqual(plan["spec"]["maximum_optimizer_updates"], 200)
            spawn.assert_not_called(); charge.assert_not_called()

    def test_controls_budget_cannot_reset_relocate_or_use_prior600_allowance(self):
        previous = BENCH.ROOT
        try:
            with tempfile.TemporaryDirectory() as directory:
                BENCH.ROOT = Path(directory)
                plan = {"plan_sha256": "a" * 64}
                run = Path(directory) / "runs/control"
                ledger = BENCH.precharge(plan, run)
                self.assertEqual(ledger["precharged_updates"], 200)
                BENCH.verify_allowance(run, plan)
                self.assertFalse((Path(directory) / ".cache/real-batch-update-budget-v1.json").exists())
                before = (Path(directory) / ".cache/timing-controls-update-budget-v1.json").read_bytes()
                with self.assertRaises(ValueError): BENCH.precharge(plan, run)
                with self.assertRaises(ValueError): BENCH.verify_allowance(Path(directory) / "runs/other", plan)
                with self.assertRaises(ValueError): BENCH.verify_allowance(run, {"plan_sha256": "b" * 64})
                self.assertEqual(before, (Path(directory) / ".cache/timing-controls-update-budget-v1.json").read_bytes())
        finally:
            BENCH.ROOT = previous

    def test_interrupted_supervisor_retains_active_counts_and_charge(self):
        case = BENCH.matrix()[0]
        class Child:
            pid, returncode = 123, None
            def poll(self): return self.returncode
            def kill(self): self.returncode = -9
            def wait(self, timeout): return self.returncode
        class Process:
            def memory_info(self): raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / (case["id"] + ".status.json")).write_text(json.dumps({"optimizer_updates_entered": 3, "optimizer_updates_confirmed": 2}))
            with patch.object(BENCH.subprocess, "Popen", return_value=Child()), patch("psutil.Process", return_value=Process()):
                result = BENCH.supervise(case, run, time.monotonic() + 10)
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["optimizer_updates_entered"], 3)
            self.assertEqual(result["optimizer_updates_confirmed"], 2)
            self.assertEqual(result["optimizer_updates_precharged"], 25)
            self.assertFalse(result["update_count_exact"])


if __name__ == "__main__":
    unittest.main()
