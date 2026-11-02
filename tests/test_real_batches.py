"""Zero-optimizer tests for the sealed real-batch benchmark and its accounting."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timezone
import stat
import time

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "scripts/benchmark_real_batches.py"
SPEC = importlib.util.spec_from_file_location("real_batch_benchmark_tests", PATH)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def fixture(directory, change=None):
    start, end = BENCH.epoch("2023-08-01T00:00:00Z"), BENCH.epoch("2023-09-01T00:00:00Z")
    origins = np.arange(start, end, 3600, dtype=np.int64)
    history = np.full((744, 168, 2), 100., dtype=np.float64)
    history[..., 1] = np.arange(169, 1, -1)
    weather = np.ones((744, 18, 4), dtype=np.float64)
    weather[..., 1] = 273.15
    weather[..., 2] = .01
    state = np.full((744, 2), .5, dtype=np.float64)
    references = ((origins - 2 * 3600) // 21600) * 21600
    state[:, 1] = (origins - references) / 3600
    arrays = dict(history=history, history_mask=np.ones(history.shape, bool), weather=weather, weather_mask=np.ones(weather.shape, bool),
                  state=state, state_mask=np.ones(state.shape, bool), target=np.full((744, 18), 100.), target_mask=np.ones((744, 18), bool),
                  site_id=np.zeros(744, dtype=np.int64), origin_time=origins,
                  history_time=origins[:, None] - np.arange(169, 1, -1)[None, :] * 3600,
                  target_time=origins[:, None] + np.arange(1, 19)[None, :] * 3600,
                  case_id=np.array(["03161000:" + datetime.fromtimestamp(int(t), timezone.utc).isoformat().replace("+00:00", "Z") for t in origins]),
                  source_cycle=np.array([datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y%m%d_%HZ") for t in references]))
    if change:
        change(arrays)
    path = Path(directory) / "fixture.npz"
    np.savez_compressed(path, **arrays)
    raw = path.read_bytes()
    receipt = dict(schema="hydra-real-batch-fixture/v1", gauge_id="03161000", history_hours=168, horizons_hours=list(range(1, 19)),
                   input_quality_policy=BENCH.PRIMARY_APPROVED, target_quality_policy=BENCH.PRIMARY_APPROVED,
                   npz=dict(sha256=BENCH.sha(raw), bytes=len(raw)), case_source_digest="a" * 64,
                   arrays_sha256=BENCH._array_digest(arrays), source_receipts=[dict(schema="hydra-cached-development/v1",
                       scope="SYNTHETIC_UNIT_TEST_ONLY", input_quality_policy=BENCH.PRIMARY_APPROVED,
                       target_quality_policy=BENCH.PRIMARY_APPROVED, policy={"units": "PHYSICAL_UNSCALED"},
                       execution={"optimizer_updates": 0, "protected_assessment_values_decoded": 0})])
    return path, receipt, arrays


class RealBatchTests(unittest.TestCase):
    def test_fixed_matrix_counts_all_warmup_and_measured_updates(self):
        matrix = BENCH.matrix()
        self.assertEqual(len(matrix), 24)
        self.assertEqual(len({row["id"] for row in matrix}), 24)
        self.assertEqual({row["batch_size"] for row in matrix}, {16, 32, 64})
        self.assertEqual(len(matrix) * (BENCH.WARMUP + BENCH.SAMPLES), 600)
        self.assertEqual(len(matrix) * BENCH.WARMUP, 120)
        self.assertEqual(len(matrix) * BENCH.SAMPLES, 480)
        self.assertEqual(BENCH.MAX_WALL_SECONDS, 900)

    def test_batches_are_distinct_within_batch_and_repeat_explicitly_across_passes(self):
        for size in (16, 32, 64):
            visited = []
            for step in range(25):
                batch = BENCH.batch_indices(list(range(558)), size, step)
                self.assertEqual(len(batch), len(set(batch)))
                self.assertEqual(len(batch), size)
                visited.extend(batch)
            self.assertEqual(len(visited), 25 * size)
            self.assertEqual(len(set(visited)), min(558, 25 * size))
        for indices, size, step in (([0, 0], 1, 0), ([0], 2, 0), ([0], 1, -1), ([0], True, 0)):
            with self.assertRaises(ValueError):
                BENCH.batch_indices(indices, size, step)

    def test_fixture_has_exact_half_open_purge_and_model_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path, receipt, original = fixture(directory)
            arrays, split = BENCH.load_fixture(path, receipt)
            self.assertEqual({k: len(v) for k, v in split.items()}, {"train": 558, "validation": 168, "purged": 18})
            np.testing.assert_equal(arrays["target"], original["target"])
            boundary = BENCH.epoch(BENCH.VALIDATION_START)
            self.assertLess(arrays["target_time"][split["train"][-1], -1], boundary)
            self.assertEqual(arrays["target_time"][split["purged"][0], -1], boundary)
            self.assertEqual(arrays["origin_time"][split["validation"][0]], boundary)

    def test_primary_approved_policy_is_independent_for_input_and_target(self):
        with tempfile.TemporaryDirectory() as directory:
            _, receipt, _ = fixture(directory)
            for key in ("input_quality_policy", "target_quality_policy"):
                changed = {**receipt, key: "PROVISIONAL"}
                with self.assertRaisesRegex(ValueError, "PRIMARY_APPROVED"):
                    BENCH.receipt_identity(changed)

    def test_fixture_hash_tamper_is_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path, receipt, _ = fixture(directory)
            path.write_bytes(b"not a fixture")
            with self.assertRaisesRegex(ValueError, "SHA/size"):
                BENCH.load_fixture(path, receipt)

    def test_duplicate_origins_wrong_gauge_and_changed_clock_are_rejected(self):
        changes = (lambda a: a["origin_time"].__setitem__(1, a["origin_time"][0]),
                   lambda a: a["site_id"].__setitem__(0, 1),
                   lambda a: a["history_time"].__setitem__((0, -1), a["origin_time"][0] - 3600),
                   lambda a: a["target_time"].__setitem__((0, 0), a["origin_time"][0]))
        for change in changes:
            with tempfile.TemporaryDirectory() as directory:
                path, receipt, _ = fixture(directory, change)
                with self.assertRaises(ValueError):
                    BENCH.load_fixture(path, receipt)

    def test_fixed_complete_cohort_cannot_silently_promote_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path, receipt, _ = fixture(directory, lambda a: a["history_mask"].__setitem__((0, 0, 0), False))
            with self.assertRaisesRegex(ValueError, "source-supported"):
                BENCH.load_fixture(path, receipt)

    def test_unavailable_or_too_old_weather_cycle_is_rejected(self):
        for cycle in ("20230801_00Z", "20230730_00Z"):
            with tempfile.TemporaryDirectory() as directory:
                path, receipt, _ = fixture(directory, lambda a: a["source_cycle"].__setitem__(0, cycle))
                with self.assertRaisesRegex(ValueError, "single-run"):
                    BENCH.load_fixture(path, receipt)

    def test_default_plan_does_not_spawn_or_charge_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            _, receipt, _ = fixture(directory)
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps(receipt))
            with patch.object(BENCH, "implementation_identity", return_value={"test": "b" * 64}), patch.object(BENCH.subprocess, "Popen") as spawn:
                value = BENCH.plan(path)
                self.assertEqual(value["status"], "PLANNED_NOT_EXECUTED")
                self.assertEqual(value["optimizer_updates"], 0)
                self.assertEqual(value["spec"]["maximum_optimizer_updates"], 600)
                spawn.assert_not_called()

    def test_precharge_cannot_repeat_or_refund_failure(self):
        previous = BENCH.ROOT
        try:
            with tempfile.TemporaryDirectory() as directory:
                BENCH.ROOT = Path(directory)
                value = {"plan_sha256": "a" * 64}
                first = BENCH.precharge(value, Path(directory) / "runs/first")
                self.assertEqual(first["precharged_updates"], 600)
                path = Path(directory) / ".cache/real-batch-update-budget-v1.json"
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "already charged"):
                    BENCH.precharge(value, Path(directory) / "runs/retry")
                self.assertEqual(path.read_bytes(), before)
        finally:
            BENCH.ROOT = previous

    def test_worker_requires_bound_precharge_and_cannot_relocate_or_change_plan(self):
        previous = BENCH.ROOT
        try:
            with tempfile.TemporaryDirectory() as directory:
                BENCH.ROOT = Path(directory)
                value = {"plan_sha256": "a" * 64}
                run = Path(directory) / "runs/first"
                BENCH.precharge(value, run)
                BENCH.verify_worker_allowance(run, value)
                with self.assertRaises(ValueError):
                    BENCH.verify_worker_allowance(Path(directory) / "runs/second", value)
                with self.assertRaises(ValueError):
                    BENCH.verify_worker_allowance(run, {"plan_sha256": "b" * 64})
                with self.assertRaises(ValueError):
                    BENCH.verify_worker_allowance(Path(directory) / "public-run", value)
        finally:
            BENCH.ROOT = previous

    def test_claim_file_and_directory_are_synced_before_work_and_replay_is_denied(self):
        events = []
        real_fsync = BENCH.os.fsync
        def track_fsync(descriptor):
            events.append("directory" if stat.S_ISDIR(BENCH.os.fstat(descriptor).st_mode) else "file")
            real_fsync(descriptor)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.started.json"
            with patch.object(BENCH.os, "fsync", side_effect=track_fsync):
                BENCH.claim_worker_case(path, {"precharged_updates": 25})
                events.append("work_may_begin")
            self.assertEqual(events, ["file", "directory", "work_may_begin"])
            with self.assertRaises(FileExistsError):
                BENCH.claim_worker_case(path, {"precharged_updates": 25})

    def test_directory_sync_failure_blocks_budget_return_and_worker_claim_progress(self):
        previous = BENCH.ROOT
        try:
            with tempfile.TemporaryDirectory() as directory:
                BENCH.ROOT = Path(directory)
                synced = []
                def fail_cache_sync(path):
                    synced.append(Path(path).name)
                    if Path(path).name == ".cache":
                        raise OSError("Simulated directory flush failure")
                with patch.object(BENCH, "sync_directory", side_effect=fail_cache_sync):
                    with self.assertRaises(OSError):
                        BENCH.precharge({"plan_sha256": "a" * 64}, Path(directory) / "runs/first")
                ledger = json.loads((Path(directory) / ".cache/real-batch-update-budget-v1.json").read_text())
                self.assertEqual(ledger["precharged_updates"], 600)
                with self.assertRaisesRegex(ValueError, "already charged"):
                    BENCH.precharge({"plan_sha256": "a" * 64}, Path(directory) / "runs/retry")
                progressed = False
                with patch.object(BENCH, "sync_directory", side_effect=OSError("flush failed")):
                    with self.assertRaises(OSError):
                        BENCH.claim_worker_case(Path(directory) / "case.started.json", {"precharged_updates": 25})
                        progressed = True
                self.assertFalse(progressed)
                self.assertTrue((Path(directory) / "case.started.json").exists())
        finally:
            BENCH.ROOT = previous

    def test_supervisor_interruption_reaps_and_preserves_known_active_case_counts(self):
        case = BENCH.matrix()[0]
        class Child:
            pid = 123
            returncode = None
            kills = 0
            def poll(self):
                return self.returncode
            def kill(self):
                self.kills += 1
                self.returncode = -9
            def wait(self, timeout):
                return self.returncode
        child = Child()
        class Process:
            def memory_info(self):
                raise KeyboardInterrupt("Simulated supervisor interruption")
            def children(self, recursive=True):
                return []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / (case["id"] + ".status.json")).write_text(json.dumps({"optimizer_updates_entered": 3, "optimizer_updates_confirmed": 2}))
            with patch.object(BENCH.subprocess, "Popen", return_value=child), patch("psutil.Process", return_value=Process()):
                result = BENCH.supervise(case, root, time.monotonic() + 30)
            self.assertEqual(child.kills, 1)
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["optimizer_updates_entered"], 3)
            self.assertEqual(result["optimizer_updates_confirmed"], 2)
            self.assertEqual(result["optimizer_updates_precharged"], 25)
            self.assertFalse(result["update_count_exact"])
            self.assertTrue(result["supervisor_interrupted"])
            self.assertEqual(result["supervisor_error_type"], "KeyboardInterrupt")

    def test_source_receipts_cannot_alias_quality_or_hide_fitted_values(self):
        with tempfile.TemporaryDirectory() as directory:
            _, receipt, _ = fixture(directory)
            source = receipt["source_receipts"][0]
            for changed in ({**source, "input_quality_policy": "PRIMARY_APPROVED"},
                            {**source, "policy": {"units": "FITTED_VALUES"}},
                            {**source, "execution": {"optimizer_updates": 1, "protected_assessment_values_decoded": 0}}):
                with self.assertRaises(ValueError):
                    BENCH.receipt_identity({**receipt, "source_receipts": [changed]})

    def test_statistics_are_empirical_and_reject_invalid_values(self):
        result = BENCH.summarize(list(range(1, 21)))
        self.assertEqual(result["median"], 10.5)
        self.assertEqual(result["empirical_p95"], 19)
        for values in ([], [0.], [float("nan")], [True]):
            with self.assertRaises(ValueError):
                BENCH.summarize(values)

    def test_projection_uses_audited_counts_and_measured_batch_geometry(self):
        measurement = dict(status="PASS", update_count_exact=True, checkpoint_forward_exact=True, batch_size=16,
                           architecture="LSTM", device="cpu",
                           training_samples=[dict(cases=16, copy_seconds=.01, update_seconds=.09)] * 20,
                           warmup_samples=[dict(cases=16, copy_seconds=.01, update_seconds=.19)] * 5,
                           validation_samples=[dict(cases=16, copy_seconds=.01, forward_loss_seconds=.04)] * 20,
                           first_eval_sample=dict(cases=16, copy_seconds=.01, forward_loss_seconds=.14),
                           stage_seconds=dict(imports=.1, fixture_read_authentication=.1, fixed_unit_mapping_tensorization=.1,
                                              model_optimizer_setup=.1, checkpoint_write=.01))
        result = BENCH.project_runtime(measurement, train_cases=33, validation_cases=17, epochs=2, trials=1, seeds=1,
                                       count_audit_sha256="a" * 64, count_qualification="USGS_ONLY_HRRR_UNKNOWN")
        self.assertAlmostEqual(result["per_epoch_seconds"], .41)
        self.assertAlmostEqual(result["first_use_excess_seconds"], .6)
        self.assertAlmostEqual(result["total_hours"] * 3600, 1.82)
        self.assertEqual(result["count_qualification"], "USGS_ONLY_HRRR_UNKNOWN")
        with self.assertRaises(ValueError):
            BENCH.project_runtime(measurement, train_cases=33, validation_cases=17, epochs=2, trials=1, seeds=1,
                                  count_audit_sha256="unknown", count_qualification="unknown")


if __name__ == "__main__":
    unittest.main()
