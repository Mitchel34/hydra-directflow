"""Date and measured-rate planning only; no model or source-value reads."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("workload_plan", Path(__file__).resolve().parents[1] / "scripts/plan_scientific_workload.py")
plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan)


def rates():
    return {a: dict(train_batch_seconds=1., stopping_batch_seconds=2., inference_batch_seconds=3.,
                   startup_seconds=4., checkpoint_seconds=5., batch_size=64, device=plan.DEVICES[a])
            for a in plan.ARCHITECTURES}


def receipt():
    cases = []
    for a in plan.ARCHITECTURES:
        for d in ("cpu", "mps"):
            for b in (16, 32, 64):
                cases.append(dict(architecture=a, device=d, batch_size=b, status="PASS", update_count_exact=True,
                    checkpoint_forward_exact=True,
                    training_samples=[dict(cases=b, copy_seconds=.1, update_seconds=.9)] * 20,
                    warmup_samples=[dict(cases=b, copy_seconds=.1, update_seconds=1.9)] * 5,
                    validation_samples=[dict(cases=b, copy_seconds=.2, forward_loss_seconds=1.8)] * 20,
                    inference_samples=[dict(cases=b, copy_seconds=.3, forward_to_cpu_seconds=2.7)] * 20,
                    first_eval_sample=dict(copy_seconds=.2, forward_loss_seconds=2.8),
                    stage_seconds=dict(imports=1., fixture_read_authentication=1., fixed_unit_mapping_tensorization=1.,
                        model_optimizer_setup=1., checkpoint_write=5.)))
    return dict(schema="hydra-real-batch-benchmark/v1", status="PASS", cases=cases,
                plan=dict(spec=dict(history_hours=168, horizons_hours=list(range(1, 19)), dtype="float32", model_width=64,
                    unit_mapping={"history": [[0, 1]] * 2, "weather": [[0, 1]] * 4,
                                  "state": [[0, 1]] * 2, "target": [[0, 1]]})))


class WorkloadPlanTests(unittest.TestCase):
    def test_target_tail_strictly_inside_partition(self):
        row = plan.calendar_window("2023-07-01T00:00:00Z", "2023-07-31T00:00:00Z")
        self.assertEqual(row["calendar_origins"], 720 - 18)
        self.assertEqual(row["last_origin_utc"], "2023-07-30T05:00:00Z")
        self.assertEqual(row["latest_target_utc"], "2023-07-30T23:00:00Z")

    def test_two_hour_label_delay_purges_one_more_origin_not_two(self):
        row = plan.calendar_window("2023-07-01T00:00:00Z", "2023-07-31T00:00:00Z", refit_at="2023-07-31T00:00:00Z")
        self.assertEqual(row["purged_origins"], 19)
        self.assertEqual(row["last_origin_utc"], "2023-07-30T04:00:00Z")
        self.assertEqual(row["latest_target_utc"], "2023-07-30T22:00:00Z")

    def test_gradient_fit_is_not_truncated_at_outer_boundary(self):
        fold = plan.fold_windows()[0]
        self.assertEqual(fold["gradient_fit"]["purged_origins"], 18)
        self.assertEqual(fold["inner_stopping"]["purged_origins"], 19)
        self.assertEqual(fold["outer_scoring"]["purged_origins"], 18)

    def test_zero_delay_still_preserves_strict_end(self):
        row = plan.fold_windows(simulated_label_delay_hours=0)[0]["inner_stopping"]
        self.assertEqual(row["purged_origins"], 18)

    def test_short_window_has_no_invented_origins(self):
        row = plan.calendar_window("2023-01-01T00:00:00Z", "2023-01-01T12:00:00Z")
        self.assertEqual((row["calendar_origins"], row["purged_origins"], row["latest_target_utc"]), (0, 12, None))

    def test_protected_or_nonhourly_windows_rejected(self):
        for start, end in (("2024-07-31T00:00:00Z", "2024-08-01T00:00:00Z"),
                           ("2023-01-01T00:00:01Z", "2023-01-02T00:00:00Z"),
                           ("2018-07-11T00:00:00Z", "2018-07-13T00:00:00Z"),
                           ("2023-01-01T00:00:00", "2023-01-02T00:00:00Z")):
            with self.assertRaises(ValueError): plan.calendar_window(start, end)

    def test_fold_dates_are_expanding_disjoint_and_preassessment(self):
        folds = plan.fold_windows()
        self.assertEqual(len(folds), 4)
        for fold in folds:
            self.assertEqual(fold["gradient_fit"]["start_utc"], plan.DEV_START)
            self.assertEqual(fold["gradient_fit"]["end_exclusive_utc"], fold["inner_stopping"]["start_utc"])
            self.assertEqual(fold["inner_stopping"]["end_exclusive_utc"], fold["outer_scoring"]["start_utc"])
            self.assertLess(plan.epoch(fold["outer_scoring"]["latest_target_utc"]), plan.epoch(plan.ASSESSMENT_START))

    def test_fit_matrix_and_paired_arms_have_equal_opportunity(self):
        base = plan.scenario(rates(), epochs=20, include_cv=False, input_arms=("full",))
        paired = plan.scenario(rates(), epochs=20, include_cv=False, input_arms=("full", "q_only"))
        cv = plan.scenario(rates(), epochs=20, include_cv=True, input_arms=("full", "q_only"))
        both = plan.scenario(rates(), epochs=20, include_cv=True, input_arms=("full", "q_only"), histories=(168, 24))
        self.assertEqual([x["fit_count"] for x in (base, paired, cv, both)], [36, 72, 144, 288])
        self.assertAlmostEqual(paired["total_hours"], 2 * base["total_hours"])
        self.assertAlmostEqual(both["total_hours"], 2 * cv["total_hours"])
        self.assertEqual(cv["optimizer_updates"], 0)

    def test_additional_configurations_apply_only_to_cv(self):
        row = plan.scenario(rates(), epochs=20, include_cv=True, input_arms=("full", "q_only"), cv_configurations=2)
        self.assertEqual(row["fit_count"], 216)
        for part in row["parts"]:
            self.assertEqual(part["configurations"], 1 if part["fold"] == "final_development" else 2)

    def test_final_lookback_contrast_adds_only_matched_full_l24_final_fits(self):
        for cv, count in ((False, 108), (True, 180)):
            value = plan.scope_with_final_lookback_contrast(rates(), epochs=50, include_cv=cv)
            self.assertEqual(value["fit_count"], count)
            self.assertEqual(value["final_lookback_contrast"]["fit_count"], 36)
            self.assertAlmostEqual(value["total_hours"], value["primary"]["total_hours"] + value["final_lookback_contrast"]["total_hours"])
            for part in value["final_lookback_contrast"]["parts"]:
                self.assertEqual((part["fold"], part["input_arm"], part["history_hours"]), ("final_development", "full", 24))
                self.assertEqual(part["throughput_source"], "FULL_L168_SIZING_PROXY_NOT_MEASURED")

    def test_partial_batches_outer_once_and_checkpoint_per_epoch(self):
        row = plan.scenario(rates(), epochs=20, include_cv=False, input_arms=("full",))
        part = row["parts"][0]; counts = part["calendar_cases_per_gauge"]
        seconds = 4 + 20 * (math.ceil(counts["gradient_fit"] / 64) + 2 * math.ceil(counts["inner_stopping"] / 64) + 5)
        seconds += 3 * math.ceil(counts["outer_scoring"] / 64)
        self.assertAlmostEqual(part["per_fit_hours"], seconds / 3600)
        self.assertEqual(part["fit_count"], 9)

    def test_measured_rates_keep_startup_and_warmup_excess(self):
        value = plan.rates_from_receipt(receipt())["LSTM"]
        self.assertEqual(value["train_batch_seconds"], 1.)
        self.assertEqual(value["startup_seconds"], 10.)

    def test_duplicate_matrix_invalid_timings_and_unqualified_case_fail(self):
        for mutation in ("duplicate", "nan", "bad_status", "shape"):
            data = receipt()
            if mutation == "duplicate": data["cases"][-1] = copy.deepcopy(data["cases"][-2])
            elif mutation == "nan": data["cases"][2]["training_samples"][0]["update_seconds"] = float("nan")
            elif mutation == "bad_status": data["cases"][2]["checkpoint_forward_exact"] = False
            else: data["cases"][2]["training_samples"].pop()
            with self.assertRaises(ValueError): plan.rates_from_receipt(data)

    def test_unmeasured_modes_are_labelled_as_proxies(self):
        row = plan.scenario(rates(), epochs=20, include_cv=False, input_arms=("full", "q_only"), histories=(168, 24))
        for part in row["parts"]:
            actual = part["input_arm"] == "full" and part["history_hours"] == 168
            self.assertEqual(part["throughput_source"] == "MEASURED_FULL_L168", actual)

    def test_other_receipt_geometry_cannot_be_called_measured_full_l168(self):
        for key, value in (("history_hours", 24), ("horizons_hours", [1]), ("dtype", "float16"),
                           ("model_width", 32), ("unit_mapping", {"history": [], "weather": [], "state": [], "target": []})):
            data = receipt(); data["plan"]["spec"][key] = value
            with self.assertRaises(ValueError): plan.rates_from_receipt(data)
        data = receipt(); data["schema"] = "unrecognized"
        with self.assertRaises(ValueError): plan.rates_from_receipt(data)


if __name__ == "__main__":
    unittest.main()
