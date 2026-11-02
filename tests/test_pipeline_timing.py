"""Zero-optimizer tests of timing plans, splits, cache geometry and projections."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from hydra_directflow.pipeline_timing import (
    PipelineTimingSpec, _array_digest, _map_arrays, _validate_arrays,
    environment_metadata, estimate_workflow, load_builder_request, plan_pipeline, run_pipeline_timing, sample_summary, split_cases, utc_epoch,
)


def coordinates(origins):
    return [0] * len(origins), origins, [[origin + lead * 3600 for lead in range(1, 19)] for origin in origins]


def measurement():
    return dict(status="COMPLETED_ENGINEERING_TIMING", spec=dict(architecture="LSTM", device="cpu"),
                training_batches=[dict(cases=8, copy_seconds=.01, update_seconds=.09)] * 3,
                validation_batches=[dict(cases=4, copy_seconds=.01, forward_loss_seconds=.04)] * 3,
                stage_seconds=dict(imports=1., cache_read_verified=.2, tensorization=.1, model_setup=.3,
                                   checkpoint_write=.4, preprocess_first_pass=2.))


def builder_request():
    return dict(history_hours=168,
                archive_specs=[dict(cycle_id=cycle, archive_path=cycle + ".tar", commit_path=cycle + ".json")
                               for cycle in ("20210101_00Z", "20220101_00Z")],
                qualified_index_path="index.jsonl", weights_path="weights.csv", usgs_manifest_path="manifest.json",
                usgs_paths={"2021-01": "2021-01.parquet", "2022-01": "2022-01.parquet"},
                origins_utc=["2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z"])


class TimingTests(unittest.TestCase):
    def test_default_does_not_allow_updates(self):
        self.assertEqual(PipelineTimingSpec().max_optimizer_steps, 0)
        for kwargs in (dict(max_optimizer_steps=True), dict(max_optimizer_steps=65), dict(device="cuda"),
                       dict(batch_size=0), dict(max_wall_seconds=901), dict(engineering_epochs=4)):
            with self.assertRaises(ValueError):
                PipelineTimingSpec(**kwargs)

    def test_chronological_target_disjoint_split_purges_boundary(self):
        boundary = utc_epoch("2022-01-03T00:00:00Z")
        origin = [boundary - 24 * 3600, boundary - 18 * 3600, boundary - 17 * 3600, boundary]
        result = split_cases(*coordinates(origin), gauge_id="03161000", validation_origin_utc="2022-01-03T00:00:00Z")
        self.assertEqual(result["train"], [0, 1])
        self.assertEqual(result["purged"], [2])
        self.assertEqual(result["validation"], [3])

    def test_split_rejects_empty_or_duplicate_or_bad_targets(self):
        origin = utc_epoch("2022-01-01T00:00:00Z")
        for stamps in ([origin], [origin, origin]):
            with self.assertRaises(ValueError):
                split_cases(*coordinates(stamps), gauge_id="03161000", validation_origin_utc="2022-01-02T00:00:00Z")
        site, origins, targets = coordinates([origin, origin + 48 * 3600])
        targets[0][0] += 1
        with self.assertRaises(ValueError):
            split_cases(site, origins, targets, gauge_id="03161000", validation_origin_utc="2022-01-02T00:00:00Z")

    def test_protected_label_tail_cannot_be_used(self):
        origin = utc_epoch("2024-07-30T07:00:00Z")
        with self.assertRaisesRegex(ValueError, "protected"):
            split_cases(*coordinates([origin, origin + 3600]), gauge_id="03161000", validation_origin_utc="2024-07-30T08:00:00Z")

    def test_split_selects_only_requested_gauge(self):
        origin = utc_epoch("2022-01-01T00:00:00Z")
        site, origins, targets = coordinates([origin, origin, origin + 48 * 3600, origin + 48 * 3600])
        result = split_cases([0, 1, 0, 1], origins, targets, gauge_id="03164000", validation_origin_utc="2022-01-02T00:00:00Z")
        self.assertEqual(result["train"], [1])
        self.assertEqual(result["validation"], [3])

    def test_clock_rejects_naive_offset_and_fractional(self):
        for value in ("2022-01-01T00:00:00", "2022-01-01T00:00:00-06:00", "2022-01-01T00:00:00.001Z", "2022-01-01T00:30:00Z"):
            with self.assertRaises(ValueError):
                utc_epoch(value)

    def test_plan_never_imports_torch_or_executes(self):
        request = builder_request()
        spec = PipelineTimingSpec(validation_origin_utc="2022-01-01T00:00:00Z")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(request))
            with patch("hydra_directflow.pipeline_timing._implementation", return_value={"test": "a" * 64}), patch("subprocess.Popen") as execute:
                plan = run_pipeline_timing(request_path=path, spec=spec)
                self.assertEqual(plan["status"], "PLANNED_NOT_EXECUTED")
                execute.assert_not_called()
                with self.assertRaises(ValueError):
                    run_pipeline_timing(request_path=path, spec=spec, execution_enabled=True, approved_plan_sha256=plan["plan_sha256"])

    def test_plan_identity_binds_input_and_spec(self):
        request = builder_request()
        spec = PipelineTimingSpec(validation_origin_utc="2022-01-01T00:00:00Z")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(request))
            with patch("hydra_directflow.pipeline_timing._implementation", return_value={}):
                first = plan_pipeline(path, spec)
                second = plan_pipeline(path, replace(spec, device="mps"))
                self.assertNotEqual(first["plan_sha256"], second["plan_sha256"])
                request["origins_utc"][0] = "2021-01-02T00:00:00Z"
                path.write_text(json.dumps(request))
                self.assertNotEqual(first["plan_sha256"], plan_pipeline(path, spec)["plan_sha256"])

    def test_request_paths_resolve_relative_to_request_not_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(builder_request()))
            _, request = load_builder_request(path)
            self.assertEqual(request["weights_path"], str(Path(directory) / "weights.csv"))
            self.assertEqual(request["usgs_paths"]["2021-01"], str(Path(directory) / "2021-01.parquet"))
            self.assertEqual(request["archive_specs"][0]["archive_path"], str(Path(directory) / "20210101_00Z.tar"))
            bad = builder_request()
            bad["unexpected"] = True
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                load_builder_request(path)

    def test_model_cache_geometry_requires_causal_history(self):
        origin = utc_epoch("2022-01-01T00:00:00Z")
        arrays = dict(origin_time=np.array([origin]), site_id=np.array([0]), case_id=np.array(["03161000:2022-01-01"]),
                      history_time=origin - np.arange(25, 1, -1)[None, :] * 3600,
                      target_time=origin + np.arange(1, 19)[None, :] * 3600)
        for name, shape in (("history", (1, 24, 2)), ("weather", (1, 18, 4)), ("state", (1, 2)), ("target", (1, 18))):
            arrays[name], arrays[name + "_mask"] = np.zeros(shape), np.ones(shape, bool)
        self.assertEqual(_validate_arrays(arrays, 24), 1)
        arrays["history_time"] += 3600
        with self.assertRaisesRegex(ValueError, "minus two"):
            _validate_arrays(arrays, 24)

    def test_timing_summary_is_empirical(self):
        result = sample_summary([1., 2., 3., 4.])
        self.assertEqual(result["median"], 2.5)
        self.assertEqual(result["empirical_p95"], 4.)
        for values in ([], [0.], [float("nan")], [True]):
            with self.assertRaises(ValueError):
                sample_summary(values)

    def test_environment_is_portable_and_binds_execution_geometry(self):
        fake_torch = SimpleNamespace(get_num_threads=lambda: 1, get_num_interop_threads=lambda: 1,
                                     backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
        with patch("importlib.metadata.version", return_value="test-version"):
            metadata = environment_metadata(fake_torch)
        self.assertEqual(metadata["model_dtype"], "float32")
        self.assertEqual(metadata["model_width"], 64)
        self.assertEqual(metadata["torch_intraop_threads"], 1)
        self.assertEqual(set(metadata["package_versions"]), {"torch", "numpy", "pyarrow", "numcodecs", "transformers"})
        self.assertFalse(metadata["device_fallback_enabled"])
        self.assertNotIn("hostname", metadata)
        self.assertNotIn("node", metadata)

    def test_projection_uses_measured_batch_geometry_and_all_fit_counts(self):
        result = estimate_workflow(measurement(), train_cases=17, validation_cases=5, epochs=2, trials=2, seeds=3, gauges=3)
        # Startup1.6 + train3batches*.1*2 + validation2batches*.05*2 + checkpoint.4*2.
        self.assertAlmostEqual(result["per_fit_seconds"], 3.2)
        self.assertEqual(result["total_fits"], 18)
        self.assertAlmostEqual(result["total_seconds"], 57.6)
        self.assertEqual(result["measured_train_batch_size"], 8)
        self.assertNotIn("conditional_compute_cost", result)

    def test_projection_requires_completed_consistent_samples_and_explicit_counts(self):
        for changed in ({"status": "FAILED"}, {"training_batches": []},
                        {"training_batches": [dict(cases=8, copy_seconds=.1, update_seconds=.1), dict(cases=4, copy_seconds=.1, update_seconds=.1)]}):
            with self.assertRaises(ValueError):
                estimate_workflow({**measurement(), **changed}, train_cases=8, validation_cases=4, epochs=1, trials=1, seeds=1)
        with self.assertRaises(ValueError):
            estimate_workflow(measurement(), train_cases=True, validation_cases=4, epochs=1, trials=1, seeds=1)

    def test_fixed_units_preserve_masks_and_do_not_fit(self):
        arrays = dict(history=np.array([[[100., 168.], [np.nan, np.nan]]]), history_mask=np.array([[[True, True], [False, False]]]),
                      weather=np.array([[[10., 303.15, .01, 100.]]]), weather_mask=np.ones((1, 1, 4), bool),
                      state=np.array([[.5, 24.]]), state_mask=np.ones((1, 2), bool),
                      target=np.array([[100., np.nan]]), target_mask=np.array([[True, False]]))
        before = _array_digest(arrays)
        mapped = _map_arrays(arrays)
        np.testing.assert_equal(mapped["history"], [[[1., 1.], [0., 0.]]])
        np.testing.assert_equal(mapped["weather"], np.ones((1, 1, 4)))
        np.testing.assert_equal(mapped["target"], [[1., 0.]])
        self.assertEqual(_array_digest(arrays), before)


if __name__ == "__main__":
    unittest.main()
