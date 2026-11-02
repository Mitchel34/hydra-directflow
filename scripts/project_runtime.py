#!/usr/bin/env python3
"""Project explicit local scenarios from completed timing receipts; never train."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hydra_directflow.contract import ARCHITECTURES, require
from hydra_directflow.pipeline_timing import estimate_workflow


def later_epoch_projection(measurement, scenario):
    """Steady-batch sensitivity plus one positive first-use excess per fit."""
    later = copy.deepcopy(measurement)
    fields = (("training_batches", "update_seconds"), ("validation_batches", "forward_loss_seconds"))
    first_use_excess = 0.0
    for field, duration in fields:
        samples = measurement[field]
        epochs = sorted({row["epoch"] for row in samples})
        require(epochs == [1, 2, 3], "Three completed engineering epochs required for this sensitivity")
        cases = [sum(row["cases"] for row in samples if row["epoch"] == epoch) for epoch in epochs]
        require(len(set(cases)) == 1, "Epochs must cover the same case count")
        seconds = [sum(row["copy_seconds"] + row[duration] for row in samples if row["epoch"] == epoch)
                   for epoch in epochs]
        first_use_excess += max(0.0, seconds[0] - statistics.mean(seconds[1:]))
        later[field] = [row for row in samples if row["epoch"] > 1]
    result = estimate_workflow(later, **scenario)
    result["basis"] = "EPOCHS_2_AND_3_BATCH_COSTS_PLUS_POSITIVE_FIRST_EPOCH_EXCESS_ONCE_PER_FIT"
    result["first_use_excess_seconds_per_fit"] = first_use_excess
    result["per_fit_components_seconds"]["first_use_excess"] = first_use_excess
    result["per_fit_seconds"] += first_use_excess
    result["total_seconds"] += first_use_excess * result["total_fits"]
    result["total_hours"] = result["total_seconds"] / 3600
    return result


def project_campaign(campaign, scenario):
    require(campaign.get("status") == "COMPLETED_ALL_EIGHT_LOCAL_TIMING_CASES", "Completed eight-case campaign required")
    attempts = campaign["attempts"]
    expected = {(architecture, device) for architecture in ARCHITECTURES for device in ("cpu", "mps")}
    observed = [(row["spec"]["architecture"], row["spec"]["device"]) for row in attempts]
    require(len(observed) == len(set(observed)) == 8 and set(observed) == expected, "Exact CPU/MPS four-model matrix required")
    require(set(scenario) == {"train_cases", "validation_cases", "epochs", "trials", "seeds", "gauges",
                              "validation_every", "checkpoint_every"}, "Exact scenario counts required")
    rows = []
    for attempt in attempts:
        require(attempt["optimizer_count_exact"] and attempt["checkpoint_forward_exact"], "Exact execution receipts required")
        all_samples = estimate_workflow(attempt, **scenario)
        all_samples["basis"] = "ALL_THREE_EPOCHS_MEAN_BATCH_COST_INCLUDING_FIRST_USE"
        rows.append(dict(architecture=attempt["spec"]["architecture"], device=attempt["spec"]["device"],
                         plan_sha256=attempt["plan_sha256"], all_samples=all_samples,
                         later_epoch_sensitivity=later_epoch_projection(attempt, scenario)))
    totals = {device: {basis: sum(row[basis]["total_hours"] for row in rows if row["device"] == device)
                       for basis in ("all_samples", "later_epoch_sensitivity")} for device in ("cpu", "mps")}
    return dict(schema="hydra-local-runtime-sensitivity/v1", status="CONDITIONAL_UNFROZEN_SCENARIO",
                created_at_utc=datetime.now(timezone.utc).isoformat(), scenario=scenario, projections=rows,
                all_four_models_total_hours=totals, scientific_choices_frozen=False, optimizer_updates=0,
                cloud_jobs=0, cloud_runtime_estimated=False, cloud_campaign_budget=None,
                qualification="Two sensitivity calculations, not confidence bounds or a convergence forecast. Small measured batch geometry and engineering checks are retained. Full-corpus preprocessing, scientific evaluation and cloud hardware remain unmeasured.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=ROOT / "reports/pipeline-timing.json")
    parser.add_argument("--scenario", type=Path, default=ROOT / "configs/runtime-scenario.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), "Output must be new")
    raw, scenario_raw = args.measurements.read_bytes(), args.scenario.read_bytes()
    result = project_campaign(json.loads(raw), json.loads(scenario_raw))
    result["input_sha256"] = dict(measurements=hashlib.sha256(raw).hexdigest(), scenario=hashlib.sha256(scenario_raw).hexdigest())
    result["implementation_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in (Path(__file__), ROOT / "src/hydra_directflow/pipeline_timing.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["all_four_models_total_hours"], sort_keys=True))


if __name__ == "__main__":
    main()
