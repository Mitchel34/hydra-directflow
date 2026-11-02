#!/usr/bin/env python3
"""Size the confirmed180-fit scope from pinned actual fold counts; never fit.

Only JSON count/audit/timing receipts are read. The source corpus and numerical
arrays are not opened. Existing calendar/proxy reports are preserved.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hydra_workload_rate_helpers", ROOT / "scripts/plan_scientific_workload.py")
workload = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workload)
require = workload.require
ROLES = ("gradient_fit", "inner_stopping", "outer_scoring")
SPLITS = ("candidate_training", "candidate_validation")


def bounded_count(value, *, maximum):
    require(type(value) is int and 0 <= value <= maximum, "Nonnegative integer count within its authority required")
    return value


def validate_authority(counts, audit, *, admission_report_sha256):
    require(workload._digest(admission_report_sha256), "Pinned admission file digest required")
    require(audit.get("schema") == "hydra-corpus-admission-audit/v1" and audit.get("status") == "COMPLETE_SNAPSHOT_CANDIDATE_AUDIT",
            "Complete corpus admission audit required")
    require(audit.get("audit_content_sha256") == workload._canonical_hash({k: v for k, v in audit.items() if k not in ("audit_content_sha256", "runtime")}),
            "Admission content identity differs")
    snapshot = audit["snapshot"]
    require(snapshot.get("status") == "COMPLETE_COMMITTED_CORPUS_SNAPSHOT" and snapshot.get("missing_chunks") == []
            and snapshot.get("numerical_payloads_checked") is True and snapshot.get("stored_history_hours") == 168,
            "Complete numerically checked L168 snapshot required")
    require(snapshot.get("planned_cases") == snapshot.get("available_candidate_cases") == audit.get("all_candidate_cases")
            and snapshot.get("final_scientific_cohort") is False and audit.get("physical_domain_review_needed") is False,
            "Candidate snapshot coverage/domain scope differs")
    quality = dict(input_policy="PRIMARY_APPROVED_A_OPTIONAL_R_V1", target_policy="PRIMARY_APPROVED_A_OPTIONAL_R_V1",
                   original_flags_reconstructed=False)
    require(audit.get("quality") == quality, "Admission quality policy differs")
    execution = audit["execution"]
    require(execution.get("protected_assessment_access") is False and execution.get("scientific_cohort_adopted") is False
            and execution.get("optimizer_updates") == execution.get("scaler_fits") == 0, "Audit execution/adoption scope differs")
    require(counts.get("schema") == "hydra-proposed-fold-admitted-counts/v1"
            and counts.get("status") == "COMPLETE_SNAPSHOT_PROPOSED_FOLD_COUNTS"
            and counts.get("scope_fit_count") == 180 and counts.get("scope_status") == "USER_CONFIRMED"
            and counts.get("window_policy_status") == "PROPOSED_UNRESOLVED"
            and counts.get("final_scientific_cohort") is False and counts.get("cohort_mode") == "common_l168_complete",
            "Confirmed180 scope and unresolved proposed-window common cohort required")
    require(counts.get("fold_counts_sha256") == workload._canonical_hash({k: v for k, v in counts.items() if k not in ("fold_counts_sha256", "runtime")}),
            "Fold count content identity differs")
    require(counts.get("admission_report_sha256") == admission_report_sha256
            and counts.get("audit_content_sha256") == audit["audit_content_sha256"]
            and counts.get("snapshot_sha256") == snapshot["snapshot_sha256"]
            and counts.get("source_plan_sha256") == snapshot["plan_sha256"]
            and counts.get("source_plan_file_sha256") == snapshot["plan_file_sha256"], "Count/admission snapshot or source-plan identity differs")
    require(all(workload._digest(counts.get(key)) for key in ("snapshot_sha256", "source_plan_sha256", "source_plan_file_sha256")), "Bound source identities required")
    require(counts.get("source_identities") == audit["source_identities"] and counts.get("quality") == quality
            and counts.get("producer_implementation_sha256") == audit["producer_implementation_sha256"]
            and counts.get("observed_shard_commits") == snapshot["observed_commits"]
            and counts.get("physical_domain_review_needed") is False, "Count source/quality/producer authority differs")
    require(all(counts.get("implementation_sha256", {}).get(k) == v for k, v in audit["implementation_sha256"].items()),
            "Count reader authority differs from the numerical audit")
    require(counts["implementation_sha256"].get("scripts/plan_scientific_workload.py") == hashlib.sha256((ROOT / "scripts/plan_scientific_workload.py").read_bytes()).hexdigest(),
            "Proposed-window planning implementation differs")
    execute = counts["execution"]
    require(all(execute.get(k) == 0 for k in ("raw_source_reads", "network_requests", "scaler_fits", "optimizer_updates", "cloud_jobs"))
            and execute.get("protected_assessment_access") is False and execute.get("cohort_or_window_policy_adopted") is False,
            "Fold counting must not fit, access assessment or adopt a window policy")
    require(counts.get("case_id_hash_encoding") == "CANONICAL_JSON_LIST_IN_CHRONOLOGICAL_ORDER_SHA256", "Case-ID hash convention differs")
    parents = {}
    for row in audit["results"]:
        key = row["gauge_id"], row["split"]
        require(key not in parents and key[0] in workload.GAUGES and key[1] in SPLITS, "Duplicate or unknown audit selection")
        selection = row["candidate_selections"]["common_L168"]
        require(selection["status"] == "COMPLETE_SNAPSHOT_CANDIDATE_SELECTION" and selection["cohort_mode"] == "common_l168_complete"
                and selection["history_hours"] == 168 and selection["gauge_id"] == key[0] and selection["split"] == key[1]
                and selection["snapshot_sha256"] == snapshot["snapshot_sha256"] and selection["source_plan_sha256"] == snapshot["plan_sha256"]
                and selection["case_count"] == row["candidate_counts"]["common_L168"] and selection["final_scientific_cohort"] is False,
                "Common-L168 parent selection differs")
        bounded_count(selection["case_count"], maximum=row["candidate_cases"])
        parents[key] = selection
    require(set(parents) == {(g, s) for g in workload.GAUGES for s in SPLITS}, "All gauge/split parent selections required")
    require(sum(p["case_count"] for p in parents.values()) == snapshot["source_admitted_cases"], "Global common-case count differs")
    require(len(counts["parent_selections"]) == 6 and {(p["gauge_id"], p["split"]) for p in counts["parent_selections"]} == set(parents)
            and all(p == parents[p["gauge_id"], p["split"]] for p in counts["parent_selections"]), "Fold count parent selection bindings differ")
    return parents


def validate_counts(counts, audit, *, admission_report_sha256):
    parents = validate_authority(counts, audit, admission_report_sha256=admission_report_sha256)
    scenarios = counts["scenarios"]
    require(len(scenarios) == 2 and {s["simulated_label_delay_hours"] for s in scenarios} == {0, 2}, "Both distinct proposed label-delay conventions required")
    result = {}
    for scenario in scenarios:
        delay = scenario["simulated_label_delay_hours"]
        require(type(delay) is int, "Integer label-delay convention required")
        expected = {f["name"]: f for f in workload.fold_windows(simulated_label_delay_hours=delay)}
        require(len(scenario["folds"]) == 4 and {f["name"] for f in scenario["folds"]} == set(expected), "Exact four nonduplicate proposed folds required")
        rows = {}
        for fold in scenario["folds"]:
            template = expected[fold["name"]]
            require(fold["role"] == template["role"] and len(fold["gauge_results"]) == 3
                    and {g["gauge_id"] for g in fold["gauge_results"]} == set(workload.GAUGES), "Fold role/gauge coverage differs")
            for gauge in fold["gauge_results"]:
                require(set(gauge["roles"]) == set(ROLES), "Fit/stopping/scoring role counts are required")
                for role in ROLES:
                    row, window = gauge["roles"][role], template[role]
                    require(row["window"] == window, "Count window differs from exact proposed calendar/availability rules")
                    split = "candidate_validation" if fold["name"] == "final_development" and role == "outer_scoring" else "candidate_training"
                    parent = parents[gauge["gauge_id"], split]
                    require(row["source_split"] == split and row["parent_selection_digest"] == parent["selection_digest"]
                            and row["parent_cohort_admission_digest"] == parent["cohort_admission_digest"]
                            and row["parent_selection_cases"] == parent["case_count"], "Role count parent/source split differs")
                    n = bounded_count(row["admitted_cases"], maximum=min(window["calendar_origins"], parent["case_count"]))
                    require(n > 0 and row["zero_row_support"] is False, "Confirmed fits require nonempty fit, stopping and scoring support")
                    require(workload._digest(row["case_ids_sha256"]), "Each role needs its exact case-ID identity")
                    inside = bounded_count(row["origin_window_common_cases"], maximum=parent["case_count"])
                    strict = bounded_count(row["strict_target_window_excluded_common_cases"], maximum=inside)
                    lagged = bounded_count(row["refit_label_delay_excluded_common_cases"], maximum=inside)
                    require(n + strict + lagged == inside and row["calendar_positions_without_common_case"] == window["calendar_origins"] - n,
                            "Role exclusions/counts do not reconcile")
                    first, last, target = (workload.epoch(row[name]) for name in
                        ("first_admitted_origin_utc", "last_admitted_origin_utc", "last_admitted_target_utc"))
                    require(workload.epoch(window["start_utc"]) <= first <= last <= workload.epoch(window["last_origin_utc"])
                            and target == last + 18 * 3600 and target <= workload.epoch(window["latest_target_utc"])
                            and n <= (last - first) // 3600 + 1, "Observed count endpoints exceed the declared role window")
                    rows[fold["name"], gauge["gauge_id"], role] = row
        require(len(rows) == 36, "Exact per-gauge role inventory required")
        result[delay] = rows
    for key, zero in result[0].items():
        two = result[2][key]
        difference = zero["admitted_cases"] - two["admitted_cases"]
        require(difference in (0, 1) if key[2] == "inner_stopping" else difference == 0,
                "Only the single boundary stopping origin may differ between label-delay conventions")
        require((zero["case_ids_sha256"] == two["case_ids_sha256"]) == (difference == 0),
                "Unchanged delay-scenario membership must retain its case identity")
    return result


def scenario(rates, role_counts, *, epochs, label_delay_hours):
    require(type(epochs) is int and epochs in (20, 50), "Explicit20/50 epoch runtime scenario required")
    folds = workload.fold_windows(simulated_label_delay_hours=label_delay_hours)
    parts = []
    for fold in folds:
        final = fold["name"] == "final_development"
        arms = [("full", 168), ("q_only", 168)] + ([("full", 24)] if final else [])
        seeds = [17, 29, 43] if final else [17]
        for gauge in workload.GAUGES:
            counts = {role: role_counts[fold["name"], gauge, role]["admitted_cases"] for role in ROLES}
            identities = {role: role_counts[fold["name"], gauge, role]["case_ids_sha256"] for role in ROLES}
            for architecture in workload.ARCHITECTURES:
                for arm, history in arms:
                    rate, source, timing_arm, timing_history, receipt_role = workload._geometry_rate(rates, architecture, arm, history)
                    require(timing_arm == arm and timing_history == history and source.startswith("MEASURED_")
                            and rate["batch_size"] == 64 and rate["device"] == workload.DEVICES[architecture],
                            "The confirmed180 scope requires measured full168/Q168/full24 local B64 rates; no proxy or CUDA extrapolation")
                    train_batches, stop_batches, score_batches = (math.ceil(counts[role] / 64) for role in ROLES)
                    train_seconds = epochs * train_batches * rate["train_batch_seconds"]
                    stop_seconds = epochs * stop_batches * rate["stopping_batch_seconds"]
                    checkpoint_seconds = epochs * rate["checkpoint_seconds"]
                    score_seconds = score_batches * rate["inference_batch_seconds"]
                    per_fit = rate["startup_seconds"] + train_seconds + stop_seconds + checkpoint_seconds + score_seconds
                    parts.append(dict(fold=fold["name"], gauge_id=gauge, architecture=architecture, input_arm=arm,
                        history_hours=history, device=rate["device"], batch_size=64, seeds=seeds, configurations=1, fit_count=len(seeds),
                        actual_common_cases=counts, role_case_ids_sha256=identities,
                        batches=dict(gradient_fit_per_epoch=train_batches, inner_stopping_per_epoch=stop_batches, outer_scoring_once=score_batches),
                        per_fit_components_seconds=dict(startup=rate["startup_seconds"], training=train_seconds, inner_stopping=stop_seconds,
                                                        checkpoint_writes=checkpoint_seconds, outer_inference_once=score_seconds),
                        per_fit_hours=per_fit / 3600, total_hours=len(seeds) * per_fit / 3600,
                        throughput_source=source, timing_receipt_role=receipt_role, timing_case_id=rate.get("measurement_case_id"),
                        timing_parameter_count=rate.get("parameter_count"), common_case_policy="SAME_L168_COMMON_MEMBERSHIP_FOR_ALL_INPUT_AND_HISTORY_ARMS"))
    require(sum(p["fit_count"] for p in parts) == 180, "Confirmed scope arithmetic differs")
    by_gauge = {g: sum(p["total_hours"] for p in parts if p["gauge_id"] == g) for g in workload.GAUGES}
    by_phase = {name: sum(p["total_hours"] for p in parts if (p["fold"] == "final_development") == final)
                for name, final in (("cross_validation", False), ("final_development_and_L24_contrast", True))}
    return dict(status="CONFIRMED_SCOPE_RUNTIME_SCENARIO_NOT_FIT_AUTHORIZATION", fit_count=180, epochs=epochs,
                simulated_label_delay_hours=label_delay_hours, label_delay_policy_adopted=False,
                fold_windows=folds, total_hours=sum(p["total_hours"] for p in parts), hours_by_gauge=by_gauge,
                hours_by_phase=by_phase, parts=parts, optimizer_updates_executed=0, source_values_read=0, cloud_jobs=0)


def build_report(counts, audit, full, controls, *, input_sha256):
    role_counts = validate_counts(counts, audit, admission_report_sha256=input_sha256["admission"])
    rates = workload.rates_from_receipt(full, controls_receipt=controls)
    return dict(schema="hydra-confirmed180-admitted-runtime/v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
        status="CONFIRMED_SCOPE_CONDITIONAL_RUNTIME_ONLY", scope_fit_count=180, scope_status="USER_CONFIRMED",
        remaining_protocol_choices_pending=True, final_scientific_cohort_adopted=False, fold_window_policy_adopted=False,
        input_sha256=input_sha256, snapshot_sha256=counts["snapshot_sha256"], source_plan_sha256=counts["source_plan_sha256"],
        fold_counts_content_sha256=counts["fold_counts_sha256"], admission_audit_content_sha256=audit["audit_content_sha256"],
        all_candidate_cases=audit["all_candidate_cases"], complete_common_L168_cases=audit["snapshot"]["source_admitted_cases"],
        cohort_mode="common_l168_complete", rates=rates,
        scenarios=[scenario(rates, role_counts[delay], epochs=epochs, label_delay_hours=delay) for delay in (0, 2) for epochs in (20, 50)],
        implementation_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ("scripts/plan_admitted_workload.py", "scripts/plan_scientific_workload.py")},
        execution=dict(source_values_read=0, models_executed=0, optimizer_updates=0, network_requests=0, cloud_jobs=0),
        assumptions=["Confirmed scope: paired full/Q-only L168 CV and final fits plus final full-L24 common-case contrast",
                     "Each gauge uses its own actual role counts; no blanket multiplication of one gauge's count",
                     "One CV seed/configuration and three final seeds; no favorable-seed selection",
                     "20/50 epochs are sizing scenarios, not selected convergence or stopping rules",
                     "Both0/2-hour label-delay windows remain unresolved proposals",
                     "Allthree input/history geometries use matched measured local B64 rates; no Q24 or CUDA proxy",
                     "Inner stopping and checkpoint writes each epoch; outer prediction once; partial batches charged as full",
                     "Sequential execution; fixed engineering benchmark maps do not select the scientific normalization"],
        exclusions=["Convergence and actual early stopping", "Full-cohort loading/scaling and operational fit/selection latency",
                    "Scoring, resampling, outage studies and figures beyond measured prediction batches",
                    "Retries, thermal contention, sleep/travel/operator delays", "Protected assessment and cloud execution"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, default=ROOT / "reports/real-batch-benchmark.json")
    parser.add_argument("--controls", type=Path, default=ROOT / "reports/timing-controls.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), "Use a new report; previous proposals are never overwritten")
    loaded = {name: workload.read_receipt(path) for name, path in
              (("counts", args.counts), ("admission", args.admission), ("measurements", args.measurements), ("controls", args.controls))}
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, (raw, _) in loaded.items()}
    result = build_report(*(loaded[name][1] for name in ("counts", "admission", "measurements", "controls")), input_sha256=hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    for row in result["scenarios"]:
        print(json.dumps({key: row[key] for key in ("simulated_label_delay_hours", "epochs", "fit_count", "total_hours")}))


if __name__ == "__main__":
    main()
