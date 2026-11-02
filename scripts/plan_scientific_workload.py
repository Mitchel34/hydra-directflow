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

ROOT = Path(__file__).resolve().parents[1]
HOUR = 3600
ARCHITECTURES = ("LSTM", "Transformer", "GRU_Transformer", "Mamba2_Transformer")
GAUGES = ("03161000", "03164000", "03479000")
DEV_START = "2018-07-12T00:00:00Z"
ASSESSMENT_START = "2024-07-31T00:00:00Z"
DEVICES = {"LSTM": "cpu", "Transformer": "mps", "GRU_Transformer": "cpu", "Mamba2_Transformer": "mps"}


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


def rates_from_receipt(receipt):
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
        require(row.get("status") == "PASS" and row.get("update_count_exact") is True
                and row.get("checkpoint_forward_exact") is True, "Unqualified measured case")
        def mean_samples(key, duration, count):
            values = row[key]
            require(len(values) == count and all(v["cases"] == 64 for v in values), "Timing sample geometry differs")
            return statistics.mean(_seconds(v["copy_seconds"]) + _seconds(v[duration]) for v in values)
        train = mean_samples("training_samples", "update_seconds", 20)
        warmup = mean_samples("warmup_samples", "update_seconds", 5)
        stopping = mean_samples("validation_samples", "forward_loss_seconds", 20)
        inference = mean_samples("inference_samples", "forward_to_cpu_seconds", 20)
        first = row["first_eval_sample"]
        stage = row["stage_seconds"]
        startup = sum(_seconds(stage[key]) for key in
            ("imports", "fixture_read_authentication", "fixed_unit_mapping_tensorization", "model_optimizer_setup"))
        startup += max(0., 5 * (warmup - train))
        startup += max(0., _seconds(first["copy_seconds"]) + _seconds(first["forward_loss_seconds"]) - stopping)
        rates[architecture] = dict(train_batch_seconds=train, stopping_batch_seconds=stopping,
            inference_batch_seconds=inference, startup_seconds=startup,
            checkpoint_seconds=_seconds(stage["checkpoint_write"]), device=row["device"], batch_size=64)
    require(len(rates) == 4, "Four measured architecture rates required")
    return rates


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
            rate = rates[architecture]
            nfit = fold["gradient_fit"]["calendar_origins"]
            nstop = fold["inner_stopping"]["calendar_origins"]
            nscore = fold["outer_scoring"]["calendar_origins"]
            per_epoch = (math.ceil(nfit / 64) * rate["train_batch_seconds"]
                         + math.ceil(nstop / 64) * rate["stopping_batch_seconds"] + rate["checkpoint_seconds"])
            per_fit = rate["startup_seconds"] + epochs * per_epoch + math.ceil(nscore / 64) * rate["inference_batch_seconds"]
            for arm in input_arms:
                for history in histories:
                    multiplicity = len(GAUGES) * repetitions
                    parts.append(dict(fold=fold["name"], architecture=architecture, input_arm=arm,
                        history_hours=history, device=rate["device"], fit_count=multiplicity,
                        seeds=[17] if is_cv else [17, 29, 43], configurations=cv_configurations if is_cv else 1,
                        calendar_cases_per_gauge=dict(gradient_fit=nfit, inner_stopping=nstop, outer_scoring=nscore),
                        per_fit_hours=per_fit / 3600, total_hours=multiplicity * per_fit / 3600,
                        throughput_source="MEASURED_FULL_L168" if arm == "full" and history == 168 else "FULL_L168_SIZING_PROXY_NOT_MEASURED"))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=ROOT / "reports/real-batch-benchmark.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), "Output must be new")
    raw = args.measurements.read_bytes()
    require(len(raw) < 2 * 1024**2, "Bounded timing receipt required")
    rates = rates_from_receipt(json.loads(raw))
    rows = [scenario(rates, epochs=e, include_cv=cv, input_arms=arms, histories=histories)
        for e in (20, 50) for cv, arms, histories in (
            (False, ("full",), (168,)),
            (False, ("full", "q_only"), (168,)),
            (True, ("full", "q_only"), (168,)),
            (True, ("full", "q_only"), (168, 24)))]
    result = dict(schema="hydra-proposed-scientific-workload/v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
        status="CALENDAR_COUNT_SCENARIOS_NOT_COHORT_ADMISSION_OR_CV_IMPLEMENTATION",
        measurements_sha256=hashlib.sha256(raw).hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
            "Full-input L168 rates proxy unmeasured Q-only/L24 rates; no assumed CUDA speedup"], exclusions=[
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
