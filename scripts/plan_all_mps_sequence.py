#!/usr/bin/env python3
"""Apply the user's all-MPS direction to the unchanged seed-ordered matrix.

Metadata-only: matching measured geometries and explicit proxies stay separate.
There is no model import, execution entry point or automatic CPU fallback.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plan_seed_sequence import digest, require, FAMILIES, ARMS
from plan_scientific_workload import _row_rates

PINS = {
    "reports/local-seed-sequence-v1.json": "87556571ea8d709da243d3b1339bad9747c32cbf71cbc53e9b02bb3d9f19e8fa",
    "reports/real-batch-benchmark.json": "008364afd040be959d30949197b561596bc0c2953937c51b52be9474ffe0b801",
    "reports/timing-controls.json": "63df6e08489a26768f3777aaff26576d1c39f5104989f31e7f380eea6430971f",
}
EPOCHS = (20, 50, 100, 200)


def load_sources():
    result = []
    for name, expected in PINS.items():
        raw = (ROOT / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, "Published source hash differs: " + name)
        result.append(json.loads(raw))
    return result


def matching_rates(full, controls):
    catalog = {}
    for receipt, kind in ((full, "FULL_L168"), (controls, "CONTROL")):
        for row in receipt["cases"]:
            if row["batch_size"] != 64 or row["device"] != "mps":
                continue
            architecture = row["architecture"]
            require(architecture in FAMILIES, "Unknown architecture in timing source")
            arm, length = ("full", 168) if kind == "FULL_L168" else (
                ("q_only", 168) if row["input_mode"] == "Q_ONLY_L168" else ("full", 24))
            require(kind == "FULL_L168" or row["input_mode"] in ("Q_ONLY_L168", "FULL_L24"),
                    "Unexpected control geometry")
            require(row["environment"]["device_fallback_enabled"] is False
                    and row["environment"]["model_dtype"] == "float32", "FP32 MPS without fallback required")
            identity = (architecture, arm, length)
            require(identity not in catalog, "Duplicate timing geometry")
            rates = _row_rates(row)
            require(rates["device"] == "mps", "CPU timing cannot describe an MPS job")
            catalog[identity] = rates
    require(all((a, "full", 168) in catalog for a in FAMILIES), "All four full-L168 MPS measurements required")
    return catalog


def hours(job, rates, epochs):
    require(rates["device"] == job["device"] == "mps" and job["batch_size"] == 64,
            "Timing/backend/batch mismatch")
    counts = job["actual_common_cases"]
    train, inner, outer = (math.ceil(counts[role] / 64) for role in ("gradient_fit", "inner_stopping", "outer_scoring"))
    require(train == job["gradient_batches_per_epoch"], "Source training-batch accounting differs")
    return (rates["startup_seconds"] + epochs * (train * rates["train_batch_seconds"]
        + inner * rates["stopping_batch_seconds"] + rates["checkpoint_seconds"])
        + outer * rates["inference_batch_seconds"]) / 3600


def build_plan(parent, full, controls):
    require(parent["schema"] == "hydra-local-seed-sequence/v1" and parent["fits"] == 180
            and parent["plan_sha256"] == digest({k:v for k,v in parent.items() if k != "plan_sha256"}),
            "Exact seed-ordered parent plan required")
    catalog = matching_rates(full, controls)
    result = deepcopy(parent)
    result.pop("plan_sha256")
    result.update(schema="hydra-local-seed-sequence/v2", status="USER_REQUESTED_ALL_MPS_PREPARED_NOT_EXECUTED",
        parent_plan_sha256=parent["plan_sha256"], evidence_sha256=PINS,
        device_policy="MPS_FOR_ALL_NEURAL_TRAINING_AND_INFERENCE",
        required_worker_environment={"PYTORCH_ENABLE_MPS_FALLBACK":"0"},
        backend_failure_policy="STOP_AND_PRESERVE_ATTEMPT_NO_AUTOMATIC_CPU_FALLBACK",
        non_neural_CPU_work="Source loading, preprocessing, bookkeeping and retrospective metric aggregation remain CPU work",
        checkpoint_backend_migration_allowed=False)
    unmatched = set()
    for job in result["jobs"]:
        job.pop("conditional_local_hours_by_epochs")  # Never relabel old CPU costs as MPS.
        job["device"] = "mps"
        identity = (job["architecture"], job["input_arm"], job["history_hours"])
        matched = identity in catalog
        rates = catalog[identity] if matched else catalog[(job["architecture"], "full", 168)]
        if not matched:
            unmatched.add(identity)
        projected = {str(e): hours(job, rates, e) for e in EPOCHS}
        job["timing"] = dict(status="MATCHED_GEOMETRY_ENGINEERING_RATE" if matched else "UNMEASURED_MPS_CONTROL_GEOMETRY",
            measurement_case_id=rates["measurement_case_id"], measured_device="mps",
            measurement_feature_geometry="MATCHES_JOB" if matched else "FULL_L168_PROXY_ONLY",
            matching_geometry_hours_by_epochs=projected if matched else None,
            full_L168_proxy_hours_by_epochs=None if matched else projected,
            proxy_is_upper_bound=False, full_scientific_fit_duration_measured=False)
    for milestone in result["seed_milestones"]:
        milestone.pop("conditional_local_hours_by_epochs")
        selected = [j for j in result["jobs"] if j["seed"] == milestone["seed"]]
        missing = [j for j in selected if j["timing"]["matching_geometry_hours_by_epochs"] is None]
        milestone["MPS_timing"] = dict(matching_geometry_jobs=len(selected)-len(missing), unmeasured_geometry_jobs=len(missing),
            complete_matched_geometry_estimate_hours=None if missing else {
                str(e): sum(j["timing"]["matching_geometry_hours_by_epochs"][str(e)] for j in selected) for e in EPOCHS},
            measured_geometry_subtotal_hours={str(e): sum(j["timing"]["matching_geometry_hours_by_epochs"][str(e)]
                for j in selected if j["timing"]["matching_geometry_hours_by_epochs"] is not None) for e in EPOCHS},
            supplemental_full_L168_proxy_scenario_hours={str(e): sum((j["timing"]["matching_geometry_hours_by_epochs"]
                or j["timing"]["full_L168_proxy_hours_by_epochs"])[str(e)] for j in selected) for e in EPOCHS},
            supplemental_scenario_is_measured_complete_estimate=False)
    measured = sum(j["timing"]["matching_geometry_hours_by_epochs"] is not None for j in result["jobs"])
    result["timing_coverage"] = dict(matching_geometry_jobs=measured, unmeasured_geometry_jobs=180-measured,
        complete_matched_geometry_campaign_estimate_hours=None,
        missing_cells=[dict(architecture=a,input_arm=arm,history_hours=length,device="mps",batch_size=64)
                       for a,arm,length in sorted(unmatched)],
        supplemental_full_L168_proxy_scenario_hours={str(e): sum(m["MPS_timing"]["supplemental_full_L168_proxy_scenario_hours"][str(e)]
            for m in result["seed_milestones"]) for e in EPOCHS})
    result["required_main_MPS_preflight_geometries"] = [dict(architecture=a, input_arm=arm, history_hours=length,
        device="mps", batch_size=64, precision="FP32", fresh_process_resume_and_best_inference_B64_qualified=False)
        for a in FAMILIES for arm,length in ARMS]
    result["runtime_scope"] = ("All-MPS engineering-rate projections;54 jobs lack matching controls. "
        "Supplemental full-L168 proxies are neither measured complete estimates nor upper bounds. "
        "No expected early-stop duration, deadline guarantee, or whole-campaign qualification is asserted.")
    result["remaining_gates"] = [*parent["remaining_gates"],
        "Qualify current trainer restart and best-state inference at actual all-MPS B64 primary geometries",
        "Measure missing MPS LSTM/GRU Q-onlyL168 and fullL24 controls before a complete matched runtime estimate",
        "Require MPS availability, explicit memory bounds and disabled CPU fallback before fresh worker imports",
        "Any separately retained H4 pilot also needs MPS adaptation; its historical CPU-only qualification is not repurposed"]
    result["plan_sha256"] = digest(result)
    return result


def comparison(full):
    rows = {(r["architecture"],r["device"]):_row_rates(r) for r in full["cases"] if r["batch_size"]==64}
    return [dict(architecture=a, CPU_mean_copy_plus_update_seconds=rows[(a,"cpu")]["train_batch_seconds"],
        MPS_mean_copy_plus_update_seconds=rows[(a,"mps")]["train_batch_seconds"],
        MPS_to_CPU_duration_ratio=rows[(a,"mps")]["train_batch_seconds"]/rows[(a,"cpu")]["train_batch_seconds"],
        scope="FULL_H2_L168_B64_FIXED_UNIT_ENGINEERING_20_TIMED_UPDATES_AFTER5_WARMUPS") for a in FAMILIES]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    args = parser.parse_args()
    parent, full, controls = load_sources()
    result = build_plan(parent, full, controls)
    benchmark = dict(schema="hydra-all-MPS-timing-review/v1", evidence_sha256=PINS,
        all_MPS_plan_sha256=result["plan_sha256"], source_benchmark_comparison=comparison(full),
        timing_coverage=result["timing_coverage"], current_study_uniform_MPS_speedup_established=False,
        user_device_choice_honored=True, new_optimizer_updates=0, new_model_calls=0,
        scientific_fits_started=0, protected_assessment_access=False, cloud_jobs=0)
    for path, value in ((args.output,result),(args.comparison,benchmark)):
        with path.open("x") as stream:
            json.dump(value,stream,indent=2,sort_keys=True); stream.write("\n")
    print(json.dumps(dict(status=result["status"],all_MPS_jobs=len(result["jobs"]),
        matching_timing_jobs=result["timing_coverage"]["matching_geometry_jobs"],
        unmeasured_timing_jobs=result["timing_coverage"]["unmeasured_geometry_jobs"],plan_sha256=result["plan_sha256"])))


if __name__ == "__main__":
    main()
