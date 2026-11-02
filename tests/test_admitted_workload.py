"""Synthetic receipt/count arithmetic only; no corpus, model or network use."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/plan_admitted_workload.py"
SPEC = importlib.util.spec_from_file_location("admitted_workload_tests", PATH)
plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan)
w = plan.workload


def reseal(audit, counts):
    audit["audit_content_sha256"] = w._canonical_hash({k: v for k, v in audit.items() if k not in ("audit_content_sha256", "runtime")})
    admission_sha = w._canonical_hash(audit)
    counts["admission_report_sha256"] = admission_sha
    counts["audit_content_sha256"] = audit["audit_content_sha256"]
    counts["fold_counts_sha256"] = w._canonical_hash({k: v for k, v in counts.items() if k not in ("fold_counts_sha256", "runtime")})
    return admission_sha


def fixtures():
    snapshot = dict(status="COMPLETE_COMMITTED_CORPUS_SNAPSHOT", missing_chunks=[], numerical_payloads_checked=True,
                    stored_history_hours=168, planned_cases=159084, available_candidate_cases=159084,
                    final_scientific_cohort=False, source_admitted_cases=0, snapshot_sha256="a" * 64,
                    plan_sha256="b" * 64, plan_file_sha256="c" * 64, observed_commits=[])
    quality = dict(input_policy="PRIMARY_APPROVED_A_OPTIONAL_R_V1", target_policy="PRIMARY_APPROVED_A_OPTIONAL_R_V1", original_flags_reconstructed=False)
    audit = dict(schema="hydra-corpus-admission-audit/v1", status="COMPLETE_SNAPSHOT_CANDIDATE_AUDIT", snapshot=snapshot,
                 all_candidate_cases=159084, quality=quality, physical_domain_review_needed=False,
                 execution=dict(protected_assessment_access=False, scientific_cohort_adopted=False, optimizer_updates=0, scaler_fits=0),
                 source_identities={"qualified_index_sha256": "d" * 64}, producer_implementation_sha256={"producer": "e" * 64},
                 implementation_sha256={"reader": "f" * 64}, results=[])
    parents = {}
    for split in plan.SPLITS:
        for i, gauge in enumerate(w.GAUGES):
            count = 40000 - i * 1000 if split == "candidate_training" else 8000 - i * 500
            selection = dict(status="COMPLETE_SNAPSHOT_CANDIDATE_SELECTION", cohort_mode="common_l168_complete", history_hours=168,
                             gauge_id=gauge, split=split, snapshot_sha256=snapshot["snapshot_sha256"], source_plan_sha256=snapshot["plan_sha256"],
                             case_count=count, final_scientific_cohort=False, selection_digest=w._canonical_hash([gauge, split, "selection"]),
                             cohort_admission_digest=w._canonical_hash([gauge, split, "cohort"]))
            audit["results"].append(dict(gauge_id=gauge, split=split, candidate_cases=44262 if split == "candidate_training" else 8766,
                                         candidate_counts={"common_L168": count}, candidate_selections={"common_L168": selection}))
            parents[gauge, split] = selection
    snapshot["source_admitted_cases"] = sum(p["case_count"] for p in parents.values())
    counts = dict(schema="hydra-proposed-fold-admitted-counts/v1", status="COMPLETE_SNAPSHOT_PROPOSED_FOLD_COUNTS",
                  scope_fit_count=180, scope_status="USER_CONFIRMED", window_policy_status="PROPOSED_UNRESOLVED",
                  final_scientific_cohort=False, cohort_mode="common_l168_complete", snapshot_sha256=snapshot["snapshot_sha256"],
                  source_plan_sha256=snapshot["plan_sha256"], source_plan_file_sha256=snapshot["plan_file_sha256"],
                  source_identities=audit["source_identities"], quality=quality, producer_implementation_sha256=audit["producer_implementation_sha256"],
                  observed_shard_commits=[], physical_domain_review_needed=False, parent_selections=list(parents.values()),
                  implementation_sha256={**audit["implementation_sha256"], "scripts/plan_scientific_workload.py": hashlib.sha256((plan.ROOT / "scripts/plan_scientific_workload.py").read_bytes()).hexdigest()},
                  case_id_hash_encoding="CANONICAL_JSON_LIST_IN_CHRONOLOGICAL_ORDER_SHA256",
                  execution=dict(raw_source_reads=0, network_requests=0, scaler_fits=0, optimizer_updates=0, cloud_jobs=0,
                                 protected_assessment_access=False, cohort_or_window_policy_adopted=False), scenarios=[])
    for delay in (0, 2):
        folds = []
        for fold in w.fold_windows(simulated_label_delay_hours=delay):
            gauges = []
            for i, gauge in enumerate(w.GAUGES):
                roles = {}
                for role in plan.ROLES:
                    window = fold[role]
                    split = "candidate_validation" if fold["name"] == "final_development" and role == "outer_scoring" else "candidate_training"
                    parent = parents[gauge, split]
                    n = min(parent["case_count"], window["calendar_origins"] - i * 100)
                    first = w.epoch(window["start_utc"]); last = first + (n - 1) * 3600
                    roles[role] = dict(window=window, admitted_cases=n, parent_selection_cases=parent["case_count"],
                        case_ids_sha256=w._canonical_hash([fold["name"], gauge, role, n]), source_split=split,
                        parent_selection_digest=parent["selection_digest"], parent_cohort_admission_digest=parent["cohort_admission_digest"],
                        origin_window_common_cases=n, strict_target_window_excluded_common_cases=0, refit_label_delay_excluded_common_cases=0,
                        calendar_positions_without_common_case=window["calendar_origins"] - n,
                        first_admitted_origin_utc=w.iso(first), last_admitted_origin_utc=w.iso(last), last_admitted_target_utc=w.iso(last + 18 * 3600),
                        zero_row_support=False)
                gauges.append(dict(gauge_id=gauge, roles=roles))
            folds.append(dict(name=fold["name"], role=fold["role"], gauge_results=gauges))
        counts["scenarios"].append(dict(simulated_label_delay_hours=delay, folds=folds))
    admission_sha = reseal(audit, counts)
    return counts, audit, admission_sha


def rates():
    result = {}
    for architecture in w.ARCHITECTURES:
        rate = dict(train_batch_seconds=1., stopping_batch_seconds=2., inference_batch_seconds=3.,
                    startup_seconds=4., checkpoint_seconds=5., device=w.DEVICES[architecture], batch_size=64)
        result[architecture] = {**rate, "measured_controls": {"q_only_L168": {**rate, "train_batch_seconds": .5},
                                                           "full_L24": {**rate, "train_batch_seconds": .25}}}
    return result


class AdmittedWorkloadTests(unittest.TestCase):
    def test_complete_authority_and_both_proposed_delay_conventions(self):
        counts, audit, sha = fixtures()
        rows = plan.validate_counts(counts, audit, admission_report_sha256=sha)
        self.assertEqual(set(rows), {0, 2})
        self.assertEqual(len(rows[0]), 36)
        for key in rows[0]:
            difference = rows[0][key]["admitted_cases"] - rows[2][key]["admitted_cases"]
            self.assertEqual(difference, 1 if key[2] == "inner_stopping" else 0)

    def test_exact180_scope_uses_each_gauges_counts_and_measured_geometries(self):
        counts, audit, sha = fixtures()
        rows = plan.validate_counts(counts, audit, admission_report_sha256=sha)[2]
        result = plan.scenario(rates(), rows, epochs=20, label_delay_hours=2)
        self.assertEqual(result["fit_count"], 180)
        self.assertEqual(len(result["parts"]), 108)
        self.assertEqual(sum(p["fit_count"] for p in result["parts"] if p["history_hours"] == 24), 36)
        self.assertTrue(all(p["fold"] == "final_development" and p["input_arm"] == "full" for p in result["parts"] if p["history_hours"] == 24))
        self.assertGreater(result["hours_by_gauge"]["03161000"], result["hours_by_gauge"]["03164000"])
        self.assertGreater(result["hours_by_gauge"]["03164000"], result["hours_by_gauge"]["03479000"])
        first = result["parts"][0]; n = first["actual_common_cases"]
        expected = 4 + 20 * (math.ceil(n["gradient_fit"] / 64) + 2 * math.ceil(n["inner_stopping"] / 64) + 5) + 3 * math.ceil(n["outer_scoring"] / 64)
        self.assertAlmostEqual(first["per_fit_hours"] * 3600, expected)
        self.assertAlmostEqual(sum(result["hours_by_gauge"].values()), result["total_hours"])
        self.assertFalse(result["label_delay_policy_adopted"])

    def test_source_snapshot_admission_file_and_scope_must_match(self):
        for mutation in ("snapshot", "admission", "scope", "cohort", "adopted"):
            counts, audit, sha = fixtures()
            if mutation == "snapshot": counts["snapshot_sha256"] = "0" * 64
            elif mutation == "admission": counts["admission_report_sha256"] = "0" * 64
            elif mutation == "scope": counts["scope_fit_count"] = 288
            elif mutation == "cohort": counts["cohort_mode"] = "natural_complete"
            else: counts["final_scientific_cohort"] = True
            counts["fold_counts_sha256"] = w._canonical_hash({k:v for k,v in counts.items() if k not in ("fold_counts_sha256", "runtime")})
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                plan.validate_counts(counts, audit, admission_report_sha256=sha)

    def test_missing_duplicate_gauges_or_roles_fail(self):
        for mutation in ("delay", "fold", "gauge", "role"):
            counts, audit, sha = fixtures()
            if mutation == "delay": counts["scenarios"][1] = copy.deepcopy(counts["scenarios"][0])
            elif mutation == "fold": counts["scenarios"][0]["folds"][1] = copy.deepcopy(counts["scenarios"][0]["folds"][0])
            elif mutation == "gauge": counts["scenarios"][0]["folds"][0]["gauge_results"][1] = copy.deepcopy(counts["scenarios"][0]["folds"][0]["gauge_results"][0])
            else: del counts["scenarios"][0]["folds"][0]["gauge_results"][0]["roles"]["inner_stopping"]
            reseal(audit, counts)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                plan.validate_counts(counts, audit, admission_report_sha256=sha)

    def test_counts_windows_parent_membership_and_endpoints_are_bounded(self):
        for mutation in ("above_calendar", "bool", "zero", "window", "parent", "endpoint", "reconciliation"):
            counts, audit, sha = fixtures()
            row = counts["scenarios"][0]["folds"][0]["gauge_results"][0]["roles"]["gradient_fit"]
            if mutation == "above_calendar": row["admitted_cases"] = row["window"]["calendar_origins"] + 1
            elif mutation == "bool": row["admitted_cases"] = True
            elif mutation == "zero": row["admitted_cases"] = 0; row["zero_row_support"] = True
            elif mutation == "window": row["window"]["end_exclusive_utc"] = "2020-05-02T00:00:00Z"
            elif mutation == "parent": row["parent_selection_digest"] = "0" * 64
            elif mutation == "endpoint": row["last_admitted_target_utc"] = "2024-08-01T00:00:00Z"
            else: row["strict_target_window_excluded_common_cases"] = 1
            reseal(audit, counts)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                plan.validate_counts(counts, audit, admission_report_sha256=sha)

    def test_partial_or_unreviewed_audit_is_not_full_count_authority(self):
        for mutation in ("partial", "physical", "quality", "content"):
            counts, audit, sha = fixtures()
            if mutation == "partial": audit["snapshot"]["missing_chunks"] = ["missing"]
            elif mutation == "physical": audit["physical_domain_review_needed"] = True
            elif mutation == "quality": audit["quality"] = {"input_policy": "OTHER"}
            else: audit["audit_content_sha256"] = "0" * 64
            if mutation != "content": sha = reseal(audit, counts)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                plan.validate_counts(counts, audit, admission_report_sha256=sha)

    def test_confirmed_scope_rejects_proxy_or_cuda_rates(self):
        counts, audit, sha = fixtures()
        rows = plan.validate_counts(counts, audit, admission_report_sha256=sha)[2]
        for mutation in ("proxy", "cuda"):
            timing = rates()
            if mutation == "proxy": del timing["LSTM"]["measured_controls"]["full_L24"]
            else: timing["LSTM"]["device"] = "cuda:0"
            with self.assertRaisesRegex(ValueError, "requires measured"):
                plan.scenario(timing, rows, epochs=20, label_delay_hours=2)


if __name__ == "__main__":
    unittest.main()
