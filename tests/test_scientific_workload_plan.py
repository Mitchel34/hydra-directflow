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


def matched_receipts():
    full = receipt()
    spec = full["plan"]["spec"]
    source = dict(schema="hydra-real-batch-fixture/v1", gauge_id="03161000", history_hours=168,
                  horizons_hours=list(range(1, 19)), input_quality_policy="PRIMARY_APPROVED_A_OPTIONAL_R_V1",
                  target_quality_policy="PRIMARY_APPROVED_A_OPTIONAL_R_V1", arrays_sha256="a" * 64,
                  case_source_digest="b" * 64, npz={"sha256": "c" * 64, "bytes": 1000})
    spec.update(source=source, fixture_receipt_sha256="d" * 64, source_receipts_sha256="e" * 64,
                maximum_optimizer_updates=600, expected_training_cases=558, expected_validation_cases=168,
                expected_purged_cases=18, validation_origin_utc="2023-08-25T00:00:00Z",
                implementation_sha256={"shared_helper": "f" * 64})
    full["plan"]["plan_sha256"] = plan._canonical_hash(spec)
    full.update(optimizer_updates_precharged=600, optimizer_updates_entered=600, optimizer_updates_confirmed=600,
                warmup_updates_confirmed=120, measured_updates_confirmed=480, unstarted_case_count=0,
                all_reported_update_counts_exact=True, cloud_jobs=0, network_requests=0)
    cohort = dict(case_ids_sha256="1" * 64, train_case_ids_sha256="2" * 64, validation_case_ids_sha256="3" * 64,
                  target_and_coordinates_sha256="4" * 64)
    parameters = dict(LSTM=67905, Transformer=85057, GRU_Transformer=76545, Mamba2_Transformer=86093)
    for row in full["cases"]:
        row.update(id=f'{row["device"]}-{row["architecture"]}-B{row["batch_size"]}',
                   cohort_counts={"train": 558, "validation": 168, "purged": 18}, source_array_sha256=source["arrays_sha256"],
                   parameter_count=parameters[row["architecture"]], train_case_ids_sha256=cohort["train_case_ids_sha256"],
                   validation_case_ids_sha256=cohort["validation_case_ids_sha256"],
                   environment={"model_dtype": "float32", "model_width": 64, "device_fallback_enabled": False,
                                "test_scope": "SYNTHETIC_RECEIPT_ONLY"})
    cases = []
    for mode, length, factor in (("Q_ONLY_L168", 168, .5), ("FULL_L24", 24, .25)):
        for architecture in plan.ARCHITECTURES:
            row = copy.deepcopy(next(r for r in full["cases"] if r["architecture"] == architecture
                                     and r["device"] == plan.DEVICES[architecture] and r["batch_size"] == 64))
            row.update(input_mode=mode, history_hours=length, id=f'{mode}-{architecture}',
                       parameter_count=parameters[architecture] - (640 if mode == "Q_ONLY_L168" else 0),
                       parameter_count_observed=True, features=copy.deepcopy(plan.Q_FEATURES if mode == "Q_ONLY_L168" else plan.FULL_FEATURES),
                       original_fixture_array_sha256=source["arrays_sha256"], optimizer_updates_precharged=25,
                       optimizer_updates_entered=25, optimizer_updates_confirmed=25, warmup_updates_confirmed=5, **cohort)
            for field in ("training_samples", "warmup_samples", "validation_samples", "inference_samples"):
                # Receipt fixture intentionally contains repeated object aliases;
                # replace each sample before adjusting its synthetic durations.
                row[field] = [{k: value * factor if k.endswith("seconds") else value for k, value in sample.items()}
                              for sample in row[field]]
            row["first_eval_sample"] = {key: value * factor for key, value in row["first_eval_sample"].items()}
            cases.append(row)
    control_spec = dict(source=copy.deepcopy(source), fixture_receipt_sha256=spec["fixture_receipt_sha256"],
                        source_receipts_sha256=spec["source_receipts_sha256"], parent_full_L168_plan_sha256=full["plan"]["plan_sha256"],
                        dtype="float32", width=64, fixed_unit_mapping=copy.deepcopy(spec["unit_mapping"]),
                        q_only_horizon_mapping="DECLARED_HORIZON_DIV18", original_full_cohort_preserved=True,
                        q_only_weather_soil_passed_to_model=False, labels_used_as_predictors=False, maximum_optimizer_updates=200,
                        expected_training_cases=558, expected_validation_cases=168, expected_purged_cases=18,
                        validation_origin_utc=spec["validation_origin_utc"], implementation_sha256=copy.deepcopy(spec["implementation_sha256"]),
                        matrix=[{key: row[key] for key in ("architecture", "input_mode", "device", "history_hours", "batch_size")} for row in cases])
    controls = dict(schema="hydra-timing-controls/v1", status="PASS", cases=cases, matched_cohort=cohort,
                    plan={"spec": control_spec, "plan_sha256": plan._canonical_hash(control_spec)},
                    optimizer_updates_precharged=200, optimizer_updates_entered=200, optimizer_updates_confirmed=200,
                    warmup_updates_confirmed=40, measured_updates_confirmed=160, unstarted_case_count=0,
                    all_reported_update_counts_exact=True, cloud_jobs=0, network_requests=0)
    return full, controls


def reseal(controls):
    controls["plan"]["plan_sha256"] = plan._canonical_hash(controls["plan"]["spec"])


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
        unmeasured = rates(); unmeasured["LSTM"]["device"] = "cuda:0"
        with self.assertRaisesRegex(ValueError, "CUDA"):
            plan.scenario(unmeasured, epochs=20, include_cv=False, input_arms=("full",))

    def test_measured_controls_dispatch_per_arm_and_history_without_q24_invention(self):
        full, controls = matched_receipts()
        measured = plan.rates_from_receipt(full, controls_receipt=controls)
        self.assertEqual(measured["LSTM"]["train_batch_seconds"], 1.)
        self.assertEqual(measured["LSTM"]["measured_controls"]["q_only_L168"]["train_batch_seconds"], .5)
        self.assertEqual(measured["LSTM"]["measured_controls"]["full_L24"]["train_batch_seconds"], .25)
        result = plan.scenario(measured, epochs=20, include_cv=False, input_arms=("full", "q_only"), histories=(168, 24))
        by = {(r["input_arm"], r["history_hours"]): r for r in result["parts"] if r["architecture"] == "LSTM"}
        self.assertEqual(by["q_only", 168]["throughput_source"], "MEASURED_Q_ONLY_L168")
        self.assertEqual(by["full", 24]["throughput_source"], "MEASURED_FULL_L24")
        self.assertEqual(by["q_only", 24]["throughput_source"], "FULL_L168_SIZING_PROXY_NOT_MEASURED")
        self.assertFalse(by["q_only", 24]["throughput_matches_requested_geometry"])
        self.assertEqual(by["q_only", 24]["per_fit_hours"], by["full", 168]["per_fit_hours"])
        self.assertLess(by["full", 24]["per_fit_hours"], by["q_only", 168]["per_fit_hours"])

    def test_measured_composite_counts_remain108_180_and_factorial288(self):
        full, controls = matched_receipts()
        measured = plan.rates_from_receipt(full, controls_receipt=controls)
        for include_cv, expected in ((False, 108), (True, 180)):
            scope = plan.scope_with_final_lookback_contrast(measured, epochs=20, include_cv=include_cv)
            self.assertEqual(scope["fit_count"], expected)
            self.assertTrue(all(p["throughput_matches_requested_geometry"] for p in scope["final_lookback_contrast"]["parts"]))
        factorial = plan.scenario(measured, epochs=20, include_cv=True, input_arms=("full", "q_only"), histories=(168, 24))
        self.assertEqual(factorial["fit_count"], 288)
        self.assertTrue(any(not p["throughput_matches_requested_geometry"] for p in factorial["parts"]))

    def test_controls_require_original_fixture_source_and_parent_plan(self):
        for mutation in ("source", "parent", "receipt", "mapping", "code"):
            full, controls = matched_receipts()
            s = controls["plan"]["spec"]
            if mutation == "source": s["source"]["arrays_sha256"] = "0" * 64
            elif mutation == "parent": s["parent_full_L168_plan_sha256"] = "0" * 64
            elif mutation == "receipt": s["fixture_receipt_sha256"] = "0" * 64
            elif mutation == "mapping": s["fixed_unit_mapping"]["target"][0][1] = 99
            else: s["implementation_sha256"]["shared_helper"] = "0" * 64
            reseal(controls)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                plan.rates_from_receipt(full, controls_receipt=controls)

    def test_controls_reject_target_or_cohort_substitution(self):
        for key in ("target_and_coordinates_sha256", "train_case_ids_sha256", "validation_case_ids_sha256", "case_ids_sha256", "original_fixture_array_sha256"):
            full, controls = matched_receipts()
            controls["cases"][0][key] = "0" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                plan.rates_from_receipt(full, controls_receipt=controls)

    def test_controls_reject_cuda_geometry_parameters_or_runtime_changes(self):
        for mutation in ("cuda", "history", "dtype", "width", "params", "features", "environment"):
            full, controls = matched_receipts()
            row = controls["cases"][0]
            if mutation == "cuda": row["device"] = "cuda:0"
            elif mutation == "history": row["history_hours"] = 24
            elif mutation == "dtype": controls["plan"]["spec"]["dtype"] = "float16"; reseal(controls)
            elif mutation == "width": controls["plan"]["spec"]["width"] = 128; reseal(controls)
            elif mutation == "params": row["parameter_count"] += 1
            elif mutation == "features": row["features"]["state"] = ["soil"]
            else: row["environment"]["model_width"] = 128
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                plan.rates_from_receipt(full, controls_receipt=controls)

    def test_controls_require_qualified_exact_nonduplicate_200_update_campaign(self):
        for mutation in ("status", "count", "duplicate", "checkpoint", "sample_count", "plan_hash"):
            full, controls = matched_receipts()
            if mutation == "status": controls["status"] = "PARTIAL_OR_FAILED"
            elif mutation == "count": controls["optimizer_updates_confirmed"] = 199
            elif mutation == "duplicate": controls["cases"][1] = copy.deepcopy(controls["cases"][0])
            elif mutation == "checkpoint": controls["cases"][0]["checkpoint_forward_exact"] = False
            elif mutation == "sample_count": controls["cases"][0]["inference_samples"].pop()
            else: controls["plan"]["plan_sha256"] = "0" * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                plan.rates_from_receipt(full, controls_receipt=controls)


if __name__ == "__main__":
    unittest.main()
