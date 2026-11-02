#!/usr/bin/env python3
"""Non-executing, calendar-only workload scenarios for the proposed study.

This is arithmetic over dates and existing timing receipts, not rolling-CV
implementation, protocol adoption, numerical data admission or model execution.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import stat
import re

ROOT = Path(__file__).resolve().parents[1]
HOUR = 3600
ARCHITECTURES = ("LSTM", "Transformer", "GRU_Transformer", "Mamba2_Transformer")
GAUGES = ("03161000", "03164000", "03479000")
DEV_START = "2018-07-12T00:00:00Z"
ASSESSMENT_START = "2024-07-31T00:00:00Z"
DEVICES = {"LSTM": "cpu", "Transformer": "mps", "GRU_Transformer": "cpu", "Mamba2_Transformer": "mps"}
FULL_FEATURES = {"history": ["discharge_cms", "age_hours"],
                 "weather": ["precip_mm", "temp_k", "specific_humidity", "swe_kg_m2"],
                 "state": ["soil_water_fraction", "soil_age_hours"]}
Q_FEATURES = {"history": ["discharge_cms", "age_hours"], "weather": ["horizon_hours"], "state": []}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def epoch(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0,
            "UTC timestamps required")
    stamp = parsed.timestamp()
    require(stamp.is_integer() and int(stamp) % HOUR == 0, "Exact hourly timestamps required")
    return int(stamp)


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def calendar_window(start, end, *, refit_at=None, simulated_label_delay_hours=2):
    first, after = epoch(start), epoch(end)
    require(epoch(DEV_START) <= first < after <= epoch(ASSESSMENT_START), "Development-only interval required")
    require(type(simulated_label_delay_hours) is int and simulated_label_delay_hours in (0, 2),
            "Explicit proposed label-delay convention required")
    # Targets are strictly inside [start,end): C+18h < end.
    last = after - 19 * HOUR
    if refit_at is not None:
        refit = epoch(refit_at)
        require(after <= refit <= epoch(ASSESSMENT_START), "Refit precedes admitted partition end")
        # A label at time t is hypothetically usable at t+delay, inclusive.
        last = min(last, refit - (18 + simulated_label_delay_hours) * HOUR)
    count = max(0, (last - first) // HOUR + 1)
    return dict(start_utc=start, end_exclusive_utc=end, calendar_origins=count,
                purged_origins=(after - first) // HOUR - count,
                last_origin_utc=iso(last) if count else None,
                latest_target_utc=iso(last + 18 * HOUR) if count else None,
                refit_at_utc=refit_at, simulated_label_delay_hours=simulated_label_delay_hours)


def fold_windows(*, simulated_label_delay_hours=2):
    folds = []
    for year in (2020, 2021, 2022, 2023):
        stopping = f"{year}-05-01T00:00:00Z"
        refit = f"{year}-07-31T00:00:00Z"
        end = f"{year + 1}-07-31T00:00:00Z"
        folds.append(dict(name=f"cv_{year - 2019}" if year < 2023 else "final_development",
            role="PROPOSED_OUTER_CV" if year < 2023 else "FINAL_DEVELOPMENT_CHECK_NOT_PROTECTED",
            gradient_fit=calendar_window(DEV_START, stopping, refit_at=refit,
                simulated_label_delay_hours=simulated_label_delay_hours),
            inner_stopping=calendar_window(stopping, refit, refit_at=refit,
                simulated_label_delay_hours=simulated_label_delay_hours),
            outer_scoring=calendar_window(refit, end, simulated_label_delay_hours=simulated_label_delay_hours)))
    return folds


def _seconds(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "Finite nonnegative timing required")
    return float(value)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _row_rates(row):
    require(row.get("status") == "PASS" and row.get("update_count_exact") is True
            and row.get("checkpoint_forward_exact") is True and row.get("batch_size") == 64
            and row.get("device") in ("cpu", "mps"), "Unqualified B64 local timing case")
    def mean_samples(key, duration, count):
        values = row[key]
        require(len(values) == count and all(v["cases"] == 64 for v in values), "Timing sample geometry differs")
        result = statistics.mean(_seconds(v["copy_seconds"]) + _seconds(v[duration]) for v in values)
        require(result > 0, "Positive measured batch duration required")
        return result
    train = mean_samples("training_samples", "update_seconds", 20)
    warmup = mean_samples("warmup_samples", "update_seconds", 5)
    stopping = mean_samples("validation_samples", "forward_loss_seconds", 20)
    inference = mean_samples("inference_samples", "forward_to_cpu_seconds", 20)
    first, stage = row["first_eval_sample"], row["stage_seconds"]
    startup = sum(_seconds(stage[key]) for key in
        ("imports", "fixture_read_authentication", "fixed_unit_mapping_tensorization", "model_optimizer_setup"))
    startup += max(0., 5 * (warmup - train))
    startup += max(0., _seconds(first["copy_seconds"]) + _seconds(first["forward_loss_seconds"]) - stopping)
    return dict(train_batch_seconds=train, stopping_batch_seconds=stopping, inference_batch_seconds=inference,
                startup_seconds=startup, checkpoint_seconds=_seconds(stage["checkpoint_write"]), device=row["device"], batch_size=64,
                parameter_count=row.get("parameter_count"), measurement_case_id=row.get("id"))


def _validate_controls(full, controls):
    require(controls.get("schema") == "hydra-timing-controls/v1" and controls.get("status") == "PASS"
            and len(controls.get("cases", [])) == 8, "Completed eight-case timing-controls receipt required")
    for receipt, updates, warmup, measured in ((full, 600, 120, 480), (controls, 200, 40, 160)):
        for key, expected in (("optimizer_updates_precharged", updates), ("optimizer_updates_entered", updates),
                              ("optimizer_updates_confirmed", updates), ("warmup_updates_confirmed", warmup),
                              ("measured_updates_confirmed", measured), ("unstarted_case_count", 0)):
            require(type(receipt.get(key)) is int and receipt[key] == expected, "Completed campaign update accounting differs")
        require(receipt.get("all_reported_update_counts_exact") is True, "Exact campaign counters required")
        value = receipt["plan"]
        require(_digest(value.get("plan_sha256")) and value["plan_sha256"] == _canonical_hash(value["spec"]), "Measurement plan digest differs")
        require(value["spec"].get("maximum_optimizer_updates") == updates and receipt.get("cloud_jobs") == 0
                and receipt.get("network_requests") == 0, "Qualified local campaign scope differs")
        require(all(row.get("status") == "PASS" and row.get("update_count_exact") is True
                    and row.get("checkpoint_forward_exact") is True for row in receipt["cases"]), "A campaign case is unqualified")
    primary, spec = full["plan"]["spec"], controls["plan"]["spec"]
    require(spec.get("parent_full_L168_plan_sha256") == full["plan"]["plan_sha256"], "Controls reference a different parent benchmark plan")
    source = primary.get("source", {})
    require(source.get("schema") == "hydra-real-batch-fixture/v1" and source.get("gauge_id") == "03161000"
            and source.get("history_hours") == 168 and source.get("horizons_hours") == list(range(1, 19))
            and source.get("input_quality_policy") == source.get("target_quality_policy") == "PRIMARY_APPROVED_A_OPTIONAL_R_V1"
            and _digest(source.get("arrays_sha256")) and _digest(source.get("case_source_digest"))
            and _digest(source.get("npz", {}).get("sha256")) and source.get("npz", {}).get("bytes", 0) > 0,
            "Original admitted fixture identity is incomplete")
    require(spec.get("source") == source, "Controls fixture/array/source identities differ from full benchmark")
    for key in ("fixture_receipt_sha256", "source_receipts_sha256"):
        require(_digest(primary.get(key)) and spec.get(key) == primary[key], "Controls preprocessing receipt identity differs")
    require(spec.get("dtype") == "float32" and spec.get("width") == 64
            and spec.get("fixed_unit_mapping") == primary["unit_mapping"]
            and spec.get("q_only_horizon_mapping") == "DECLARED_HORIZON_DIV18"
            and spec.get("original_full_cohort_preserved") is True
            and spec.get("q_only_weather_soil_passed_to_model") is False and spec.get("labels_used_as_predictors") is False,
            "Controls precision/width/fixed-map or predictor roles differ")
    for key, expected in (("expected_training_cases", 558), ("expected_validation_cases", 168), ("expected_purged_cases", 18)):
        require(primary.get(key) == spec.get(key) == expected, "Controls split case counts differ")
    require(spec.get("validation_origin_utc") == primary.get("validation_origin_utc") == "2023-08-25T00:00:00Z", "Controls split boundary differs")
    require(primary.get("implementation_sha256") and all(spec.get("implementation_sha256", {}).get(key) == value
            for key, value in primary["implementation_sha256"].items()), "Shared benchmark implementation changed")
    expected = {(architecture, mode, DEVICES[architecture], length, 64)
                for architecture in ARCHITECTURES for mode, length in (("Q_ONLY_L168", 168), ("FULL_L24", 24))}
    observed = [(row["architecture"], row["input_mode"], row["device"], row["history_hours"], row["batch_size"]) for row in controls["cases"]]
    planned = [(row["architecture"], row["input_mode"], row["device"], row["history_hours"], row["batch_size"]) for row in spec["matrix"]]
    require(len(observed) == len(set(observed)) == len(planned) == len(set(planned)) == 8 and set(observed) == set(planned) == expected,
            "Exact controls input/device/history matrix required; CUDA extrapolation is forbidden")
    cohort = controls.get("matched_cohort", {})
    require(set(cohort) == {"case_ids_sha256", "train_case_ids_sha256", "validation_case_ids_sha256", "target_and_coordinates_sha256"}
            and all(_digest(value) for value in cohort.values()), "Shared controls cohort/target identities required")
    baseline = {row["architecture"]: row for row in full["cases"]
                if row["batch_size"] == 64 and row["device"] == DEVICES[row["architecture"]]}
    for row in controls["cases"]:
        reference = baseline[row["architecture"]]
        require(row.get("status") == "PASS" and row.get("update_count_exact") is True and row.get("checkpoint_forward_exact") is True,
                "Unqualified controls execution")
        require(all(row.get(key) == 25 for key in ("optimizer_updates_precharged", "optimizer_updates_entered", "optimizer_updates_confirmed"))
                and row.get("warmup_updates_confirmed") == 5, "Controls per-case update accounting differs")
        require(row.get("cohort_counts") == reference.get("cohort_counts") == {"train": 558, "validation": 168, "purged": 18}, "Controls cohort counts differ")
        require(all(row.get(key) == value for key, value in cohort.items()), "Controls case/target identity differs")
        require(all(reference.get(key) == cohort[key] for key in ("train_case_ids_sha256", "validation_case_ids_sha256"))
                and reference.get("source_array_sha256") == row.get("original_fixture_array_sha256") == source["arrays_sha256"],
                "Full/control case membership or original target-bearing arrays differ")
        q_only = row["input_mode"] == "Q_ONLY_L168"
        require(row.get("features") == (Q_FEATURES if q_only else FULL_FEATURES), "Control predictor geometry differs")
        require(row.get("parameter_count_observed") is True and type(reference.get("parameter_count")) is int
                and row.get("parameter_count") == reference["parameter_count"] - (640 if q_only else 0), "Observed parameter geometry differs")
        environment = row.get("environment", {})
        require(environment == reference.get("environment") and environment.get("model_dtype") == "float32"
                and environment.get("model_width") == 64 and environment.get("device_fallback_enabled") is False,
                "Control/reference runtime metadata or FP32 width64 geometry differs")
        _row_rates(row)
    return controls["cases"]


def rates_from_receipt(receipt, *, controls_receipt=None):
    require(receipt.get("schema") == "hydra-real-batch-benchmark/v1"
            and receipt.get("status") == "PASS" and len(receipt.get("cases", [])) == 24,
            "Completed 24-case engineering timing receipt required")
    spec = receipt.get("plan", {}).get("spec", {})
    require(spec.get("history_hours") == 168 and spec.get("horizons_hours") == list(range(1, 19))
            and spec.get("dtype") == "float32" and spec.get("model_width") == 64,
            "Measured full-L168/H18 FP32 width64 geometry required")
    require({key: len(spec.get("unit_mapping", {}).get(key, [])) for key in ("history", "weather", "state", "target")}
            == {"history": 2, "weather": 4, "state": 2, "target": 1}, "Measured full-input feature geometry required")
    keys = [(r["architecture"], r["device"], r["batch_size"]) for r in receipt["cases"]]
    require(len(set(keys)) == 24 and set(keys) == {
        (a, d, b) for a in ARCHITECTURES for d in ("cpu", "mps") for b in (16, 32, 64)},
        "Exact nonduplicate timing matrix required")
    rates = {}
    for row in receipt["cases"]:
        architecture = row["architecture"]
        if row["batch_size"] != 64 or row["device"] != DEVICES[architecture]:
            continue
        rates[architecture] = _row_rates(row)
    require(len(rates) == 4, "Four measured architecture rates required")
    if controls_receipt is not None:
        for row in _validate_controls(receipt, controls_receipt):
            arm = "q_only" if row["input_mode"] == "Q_ONLY_L168" else "full"
            rates[row["architecture"]].setdefault("measured_controls", {})[f'{arm}_L{row["history_hours"]}'] = _row_rates(row)
    return rates


def _geometry_rate(rates, architecture, arm, history):
    baseline = rates[architecture]
    if arm == "full" and history == 168:
        return baseline, "MEASURED_FULL_L168", "full", 168, "primary"
    measured = baseline.get("measured_controls", {}).get(f"{arm}_L{history}")
    if measured is not None:
        return measured, f"MEASURED_{arm.upper()}_L{history}", arm, history, "controls"
    # Preserve the original explicit proxy convention. In particular Q-only
    # L24 is never silently inferred from either newly measured control arm.
    return baseline, "FULL_L168_SIZING_PROXY_NOT_MEASURED", "full", 168, "primary"


def scenario(rates, *, epochs, include_cv, input_arms, histories=(168,), cv_configurations=1,
             simulated_label_delay_hours=2):
    require(type(epochs) is int and epochs in (20, 50), "Explicit 20/50 epoch sizing scenario required")
    require(type(include_cv) is bool and input_arms in (("full",), ("full", "q_only")), "Explicit candidate scope required")
    require(histories in ((168,), (24,), (168, 24)) and type(cv_configurations) is int and cv_configurations in (1, 2),
            "Bounded candidate history/search scope required")
    require(set(rates) == set(ARCHITECTURES), "All four architecture rates required")
    folds = fold_windows(simulated_label_delay_hours=simulated_label_delay_hours)
    selected = folds if include_cv else folds[-1:]
    parts = []
    for fold in selected:
        is_cv = fold["role"] == "PROPOSED_OUTER_CV"
        repetitions = cv_configurations if is_cv else 3
        for architecture in ARCHITECTURES:
            nfit = fold["gradient_fit"]["calendar_origins"]
            nstop = fold["inner_stopping"]["calendar_origins"]
            nscore = fold["outer_scoring"]["calendar_origins"]
            for arm in input_arms:
                for history in histories:
                    rate, source, timing_arm, timing_history, receipt_role = _geometry_rate(rates, architecture, arm, history)
                    require(rate["device"] == DEVICES[architecture] and rate["batch_size"] == 64,
                            "Only the recorded local B64 device policy is implemented; no CUDA extrapolation")
                    per_epoch = (math.ceil(nfit / 64) * rate["train_batch_seconds"]
                                 + math.ceil(nstop / 64) * rate["stopping_batch_seconds"] + rate["checkpoint_seconds"])
                    per_fit = rate["startup_seconds"] + epochs * per_epoch + math.ceil(nscore / 64) * rate["inference_batch_seconds"]
                    multiplicity = len(GAUGES) * repetitions
                    parts.append(dict(fold=fold["name"], architecture=architecture, input_arm=arm,
                        history_hours=history, device=rate["device"], fit_count=multiplicity,
                        seeds=[17] if is_cv else [17, 29, 43], configurations=cv_configurations if is_cv else 1,
                        calendar_cases_per_gauge=dict(gradient_fit=nfit, inner_stopping=nstop, outer_scoring=nscore),
                        per_fit_hours=per_fit / 3600, total_hours=multiplicity * per_fit / 3600,
                        throughput_source=source, timing_input_arm=timing_arm, timing_history_hours=timing_history,
                        timing_receipt_role=receipt_role, timing_case_id=rate.get("measurement_case_id"),
                        timing_parameter_count=rate.get("parameter_count"),
                        throughput_matches_requested_geometry=arm == timing_arm and history == timing_history))
    return dict(status="PLANNING_ONLY_NO_PROTOCOL_ADOPTION", epochs=epochs, include_cv=include_cv,
        input_arms=list(input_arms), histories=list(histories), cv_configurations=cv_configurations,
        fit_count=sum(p["fit_count"] for p in parts), total_hours=sum(p["total_hours"] for p in parts),
        fold_windows=selected, parts=parts, source_values_read=0, optimizer_updates=0, cloud_jobs=0)


def scope_with_final_lookback_contrast(rates, *, epochs, include_cv):
    primary = scenario(rates, epochs=epochs, include_cv=include_cv, input_arms=("full", "q_only"))
    contrast = scenario(rates, epochs=epochs, include_cv=False, input_arms=("full",), histories=(24,))
    return dict(status="PLANNING_ONLY_NO_PROTOCOL_ADOPTION", epochs=epochs,
        scope="PAIRED_L168_WITH_FINAL_FULL_L24_CONTRAST", include_cv_for_paired_l168=include_cv,
        common_case_policy="All input/lookback contrasts use common L168-admissible cases; calendar counts here precede admission",
        primary=primary, final_lookback_contrast=contrast,
        fit_count=primary["fit_count"] + contrast["fit_count"], total_hours=primary["total_hours"] + contrast["total_hours"],
        source_values_read=0, optimizer_updates=0, cloud_jobs=0)


def read_receipt(path):
    path = Path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not getattr(before, "st_flags", 0) & 0x40000000
            and 0 < before.st_size < 2 * 1024**2, "Bounded resident regular timing receipt required")
    with path.open("rb") as stream:
        raw = stream.read(2 * 1024**2 + 1)
    after = path.lstat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) and len(raw) == before.st_size,
            "Timing receipt changed while reading")
    return raw, json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=ROOT / "reports/real-batch-benchmark.json")
    parser.add_argument("--controls", type=Path, help="Optional completed Q-only L168/full L24 timing receipt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), "Output must be new")
    raw, measurements = read_receipt(args.measurements)
    controls_raw, controls = read_receipt(args.controls) if args.controls is not None else (None, None)
    rates = rates_from_receipt(measurements, controls_receipt=controls)
    rows = [scenario(rates, epochs=e, include_cv=cv, input_arms=arms, histories=histories)
        for e in (20, 50) for cv, arms, histories in (
            (False, ("full",), (168,)),
            (False, ("full", "q_only"), (168,)),
            (True, ("full", "q_only"), (168,)),
            (True, ("full", "q_only"), (168, 24)))]
    result = dict(schema="hydra-proposed-scientific-workload/v2", created_at_utc=datetime.now(timezone.utc).isoformat(),
        status="CALENDAR_COUNT_SCENARIOS_NOT_COHORT_ADMISSION_OR_CV_IMPLEMENTATION",
        measurements_sha256=hashlib.sha256(raw).hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        controls_measurements_sha256=hashlib.sha256(controls_raw).hexdigest() if controls_raw is not None else None,
        controls_validation="MATCHED_FIXTURE_COHORT_TARGET_RUNTIME_AND_GEOMETRY_RECEIPTS" if controls is not None else "NOT_SUPPLIED",
        timing_geometry_coverage={"full_L168": "MEASURED", "q_only_L168": "MEASURED" if controls is not None else "FULL_L168_PROXY",
                                  "full_L24": "MEASURED" if controls is not None else "FULL_L168_PROXY", "q_only_L24": "UNMEASURED_FULL_L168_PROXY"},
        source_values_read=0, optimizer_updates=0, cloud_jobs=0, cuda_runtime_extrapolated=False,
        rates=rates, scenarios=rows,
        composite_scopes=[scope_with_final_lookback_contrast(rates, epochs=e, include_cv=cv) for e in (20, 50) for cv in (False, True)],
        assumptions=[
            "Candidate folds and seeds from scientific-study-design.md; unapproved, no model selection performed",
            "Calendar counts before USGS/HRRR eligibility; same cases assumed for four architectures and paired input arms",
            "All targets stay inside assigned partition; inner-stop labels hypothetically available by refit with simulated2h lag",
            "Independent annual CV refits; hourly forecasts do not imply hourly refitting",
            "One final selected configuration per family; three final seeds, no favorable-seed selection",
            "Train-only transforms in each fit; last training-period checkpoints frozen through outer scoring, no combined-data refit",
            "Sequential mixed CPU/MPS at measured B64 rates; partial batches charged as full; outer inference once per fit",
            "Full-input L168, Q-only L168 and full-input L24 use geometry-specific measurements when supplied; Q-only L24 remains an explicit full-L168 proxy",
            "Control arms retain the original matched cohort and fixed engineering maps; not exact parameter matching or a scientific transform choice",
            "Recorded source/runtime identities must match; the prior mixed device policy is fixed, not optimized anew for each control; no assumed CUDA speedup"], exclusions=[
            "Actual missing-data admission and common-cohort counts", "Early-stopping convergence and failed/retried fits",
            "Historical fitting/tuning/selection elapsed time; counts do not prove model readiness at the first outer origin",
            "Full-cohort startup and train-only scaling", "Guard/scoring/resampling/figure costs beyond measured batch calls",
            "Thermal, sleep, travel, contention and operator delays", "Standalone component controls and expanded search",
            "Assessment access/scoring, cloud launch and spending"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    for row in rows:
        print(row["epochs"], row["fit_count"], round(row["total_hours"], 2), row["include_cv"], row["histories"])


if __name__ == "__main__":
    main()
