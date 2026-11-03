#!/usr/bin/env python3
"""Build the confirmed local 180-fit roster in seed order; never execute a fit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/confirmed-180-fit-admitted-runtime.json"
SOURCE_SHA = "9a8b5526b0956125aecc2ebca9a29d20e90e6b262c136225505cde18cd92b2ec"
FAMILIES = ("LSTM", "Transformer", "GRU_Transformer", "Mamba2_Transformer")
GAUGES = ("03161000", "03164000", "03479000")
FOLDS = ("cv_1", "cv_2", "cv_3", "final_development")
ARMS = (("full", 168), ("q_only", 168), ("full", 24))
PHASES = (("cv_1", 17), ("cv_2", 17), ("cv_3", 17),
          ("final_development", 17), ("final_development", 29), ("final_development", 43))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_report():
    raw = SOURCE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SOURCE_SHA, "Published admitted-runtime authority differs")
    return json.loads(raw)


def key(row):
    return (row["fold"], row["gauge_id"], row["architecture"], row["input_arm"], row["history_hours"])


def selected_scenario(report, epochs):
    scenarios = [s for s in report["scenarios"] if s["epochs"] == epochs and s["simulated_label_delay_hours"] == 2]
    require(len(scenarios) == 1, "One exact two-hour timing/count scenario required")
    value = scenarios[0]
    require(value["fit_count"] == 180 and report["scope_fit_count"] == 180, "Confirmed 180-fit scope required")
    rows = value["parts"]
    expected = {(f, g, a, arm, length) for f in FOLDS for g in GAUGES for a in FAMILIES
                for arm, length in (ARMS if f == "final_development" else ARMS[:2])}
    require(len(rows) == len(expected) and {key(row) for row in rows} == expected,
            "Missing, duplicated or expanded fit matrix")
    memberships = {}
    for row in rows:
        require(row["configurations"] == 1 and row["batch_size"] == 64,
                "One configuration and unchanged B64 required")
        seeds = [17, 29, 43] if row["fold"] == "final_development" else [17]
        require(row["seeds"] == seeds and row["fit_count"] == len(seeds), "Fold/seed count differs")
        require(row["device"] == ("cpu" if row["architecture"] in ("LSTM", "GRU_Transformer") else "mps"),
                "Published local backend assignment changed")
        count_keys = {"gradient_fit", "inner_stopping", "outer_scoring"}
        require(set(row["actual_common_cases"]) == set(row["role_case_ids_sha256"]) == count_keys,
                "Explicit role counts and hashes required")
        identity = (row["actual_common_cases"], row["role_case_ids_sha256"])
        group = (row["fold"], row["gauge_id"])
        require(group not in memberships or memberships[group] == identity, "Paired cohort membership changed")
        memberships[group] = identity
    require(sum(row["fit_count"] for row in rows) == 180, "Expanded fit count differs")
    return value, {key(row): row for row in rows}


def build_plan(report):
    low, low_rows = selected_scenario(report, 20)
    high, high_rows = selected_scenario(report, 50)
    require(low["fold_windows"] == high["fold_windows"], "Timing scenarios use different temporal roles")
    windows = {w["name"]: w for w in low["fold_windows"]}
    require(set(windows) == set(FOLDS), "All declared development folds required")
    for window in windows.values():
        require(window["outer_scoring"]["end_exclusive_utc"] <= "2024-07-31T00:00:00Z",
                "Protected assessment is outside this schedule")
    jobs, phases = [], []
    for phase_index, (fold, seed) in enumerate(PHASES):
        phase_id = f"{phase_index + 1:02d}_{fold}_seed{seed}"
        first = len(jobs)
        for gauge in GAUGES:
            for architecture in FAMILIES:
                for arm, history in (ARMS if fold == "final_development" else ARMS[:2]):
                    row_key = (fold, gauge, architecture, arm, history)
                    small, row = low_rows[row_key], high_rows[row_key]
                    require(row["actual_common_cases"] == small["actual_common_cases"]
                            and row["role_case_ids_sha256"] == small["role_case_ids_sha256"],
                            "Timing source membership differs")
                    h20, h50 = small["per_fit_hours"], row["per_fit_hours"]
                    timings = {str(e): h20 + (e - 20) * (h50 - h20) / 30 for e in (20, 50, 100, 200)}
                    job_id = f"{fold}__{gauge}__{architecture}__{arm}_L{history}__seed{seed}"
                    jobs.append(dict(job_id=job_id, order=len(jobs), phase_id=phase_id, fold=fold, seed=seed,
                        gauge_id=gauge, architecture=architecture, input_arm=arm, history_hours=history,
                        batch_size=64, precision="FP32", device=row["device"],
                        actual_common_cases=row["actual_common_cases"], role_case_ids_sha256=row["role_case_ids_sha256"],
                        gradient_batches_per_epoch=row["batches"]["gradient_fit_per_epoch"],
                        update_ceiling_scenarios={str(e): e * row["batches"]["gradient_fit_per_epoch"] for e in (50, 100, 200)},
                        conditional_local_hours_by_epochs=timings,
                        evaluation_scope="DECLARED_OUTER_DEVELOPMENT_ONLY", selected_weights="BEST_INNER_CHECKPOINT",
                        status="PLANNED_NOT_STARTED"))
        phases.append(dict(phase_id=phase_id, fold=fold, seed=seed, fits=len(jobs)-first,
            job_ids=[j["job_id"] for j in jobs[first:]],
            predecessor_complete_phase=None if not phases else phases[-1]["phase_id"],
            completion_requires="AUTHENTICATED_TRAINING_AND_MATCHING_DEVELOPMENT_EVALUATION_FOR_EVERY_JOB"))
    require(len(jobs) == len({j["job_id"] for j in jobs}) == 180, "Unique 180 jobs required")
    milestones = []
    for seed in (17, 29, 43):
        selected = [j for j in jobs if j["seed"] == seed]
        milestones.append(dict(seed=seed, fits=len(selected), cv_fits=sum(j["fold"] != "final_development" for j in selected),
            final_development_fits=sum(j["fold"] == "final_development" for j in selected),
            cumulative_completed_fits=sum(j["seed"] <= seed for j in jobs),
            conditional_local_hours_by_epochs={str(e): sum(j["conditional_local_hours_by_epochs"][str(e)] for j in selected)
                                               for e in (20, 50, 100, 200)},
            publish_complete_seed_report=True, review_may_overlap_next_seed=True,
            extra_human_approval_pause_between_unchanged_seeds=False))
    result = dict(schema="hydra-local-seed-sequence/v1", status="USER_REQUESTED_ORDER_PREPARED_NOT_EXECUTED",
        source_report="reports/confirmed-180-fit-admitted-runtime.json", source_report_sha256=SOURCE_SHA,
        source_plan_sha256=report["source_plan_sha256"], snapshot_sha256=report["snapshot_sha256"],
        seed_order=[17, 29, 43], fits=180, cv_seed=17, cv_repeated_for_later_seeds=False,
        local_only=True, architectures=list(FAMILIES), phases=phases, seed_milestones=milestones, jobs=jobs,
        role_windows=windows, scientific_epoch_ceiling_selected=None,
        stopping_proposal=dict(max_epochs=200, min_epochs=10, patience=5, min_improvement=0,
            monitor="INNER_STANDARDIZED_Q_MSE", numeric_ceiling_frozen=False),
        report_requirements=["Every expected training terminal state, best checkpoint and source/cohort/role identity",
            "Matching development predictions and physical-unit metrics for every expected job",
            "CV fold and per-gauge/per-lead comparisons; coverage, exclusions, negative frequency and event support",
            "Actual completed/best epochs, stopping reason, runtime, resource and recovery records",
            "Single-seed development labels until all three final seed reports exist; retain every seed"],
        advance_rules=["Missing training, checkpoint, prediction or metric evidence blocks a complete report and next seed",
            "Poor finite forecast skill alone is a valid result, not a reason to skip a model or seed",
            "Undefined primary support remains explicit; it never becomes a zero score or omitted row",
            "Settings remain frozen during early-result review; changes require a new version and consistent reruns"],
        implementation_status="METADATA_ROSTER_ONLY_NOT_A_DURABLE_EXECUTION_CONTROLLER",
        remaining_gates=["Adopt concrete protocol, epoch limit and actual per-fit/global resource ledger",
            "Qualify full main-study role-limited bundles and their local execution integration",
            "Implement and qualify durable job/checkpoint/evaluation receipts and seed-report barriers"],
        runtime_scope="Per-fit affine extension of measured20/50-epoch local proxies; no expected early-stop epoch or deadline guarantee",
        scientific_fits_started=0, protected_assessment_access=False, paid_cloud_jobs_started=0)
    result["plan_sha256"] = digest(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_plan(source_report())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True); stream.write("\n")
    print(json.dumps(dict(status=result["status"], fits=180,
        milestone_fits=[m["fits"] for m in result["seed_milestones"]], plan_sha256=result["plan_sha256"])))


if __name__ == "__main__":
    main()
