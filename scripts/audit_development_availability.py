#!/usr/bin/env python3
"""Metadata-only USGS L24/L168 availability audit of pinned August 2021 data.

No discharge/source-value columns, models, protected data, or weather sources
are decoded. Whole local files are hash-checked before metadata projection.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hydra_directflow.local_data import (GAUGES, USGS_MANIFEST_SHA256, USGS_SHA256,
                                         iso, utc_epoch, verified_local_bytes)
from hydra_directflow.reference import require

HOUR = 3600
MONTH_START = utc_epoch("2021-08-01T00:00:00Z")
MONTH_END = utc_epoch("2021-09-01T00:00:00Z")
ORIGIN_START = utc_epoch("2021-08-08T02:00:00Z")
ORIGIN_END = utc_epoch("2021-08-31T05:00:00Z")
HISTORY_LAG = 2
HORIZON = 18
HISTORY_OPTIONS = (24, 168)
POLICIES = ("exact_A_only", "approved_A_optional_R_no_e_or_P")
METADATA_COLUMNS = ("site", "sample_epoch", "sample_time_utc", "is_exact_utc_hour",
                    "numeric_valid", "source_qualifiers_json", "is_estimated", "is_core")
REASONS = ("missing_exact_hour", "numeric_valid_false", "estimated_flag_true", "qualifiers_not_admitted")


def load_metadata(native_path, manifest_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    manifest_raw = verified_local_bytes(manifest_path, USGS_MANIFEST_SHA256, max_bytes=2 * 1024**2)
    manifest = json.loads(manifest_raw)
    parts = [p for p in manifest["native_partitions"] if Path(p["path"]).name == "2021-08.parquet"]
    require(len(parts) == 1 and parts[0]["sha256"] == USGS_SHA256, "Pinned USGS partition binding differs")
    raw = verified_local_bytes(native_path, USGS_SHA256, expected_bytes=parts[0]["bytes"], max_bytes=128 * 1024)
    # Explicit projection is the only Parquet data read. Numeric QA is retained
    # from acquisition; it is not recomputed from flow values in this audit.
    table = pq.read_table(pa.BufferReader(raw), columns=list(METADATA_COLUMNS), use_threads=False)
    require(tuple(table.column_names) == METADATA_COLUMNS and table.num_rows <= 10000,
            "Unexpected metadata projection or bounded row count")
    return table.to_pylist(), {"native_partition": {"logical_id": "native/2021-08.parquet",
                                                   "sha256": USGS_SHA256, "bytes": len(raw)},
                              "source_manifest": {"logical_id": "usgs-source-manifest",
                                                  "sha256": USGS_MANIFEST_SHA256, "bytes": len(manifest_raw)},
                              "pyarrow_version": pa.__version__}


def index_metadata_rows(rows):
    """Validate native keys before restricting to core exact UTC hours."""
    exact = {site: {} for site in GAUGES}
    profile = {site: {"native_rows": 0, "core_native_rows": 0, "padding_native_rows": 0,
                      "exact_core_rows": 0, "core_qualifier_counts_verbatim": Counter(),
                      "exact_core_qualifier_counts_verbatim": Counter(),
                      "exact_core_numeric_invalid": 0, "exact_core_estimated": 0,
                      "estimated_flag_qualifier_disagreements": 0} for site in GAUGES}
    seen = set()
    for row in rows:
        site, stamp = row["site"], row["sample_epoch"]
        require(site in GAUGES and type(stamp) is int, "Unknown site or noninteger source timestamp")
        parsed = datetime.fromisoformat(row["sample_time_utc"].replace("Z", "+00:00"))
        require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
                "Source sample_time_utc must declare UTC")
        require(utc_epoch(row["sample_time_utc"]) == stamp, "Source timestamp coordinates disagree")
        require((site, stamp) not in seen, "Duplicate native gauge/timestamp key")
        seen.add((site, stamp))
        for key in ("is_exact_utc_hour", "numeric_valid", "is_estimated", "is_core"):
            require(type(row[key]) is bool, "Source validity/quality flags must be booleans")
        require(row["is_exact_utc_hour"] == (stamp % HOUR == 0), "Incorrect exact UTC hour indicator")
        flags = json.loads(row["source_qualifiers_json"])
        require(type(flags) is list and all(isinstance(flag, str) and flag for flag in flags)
                and len(set(flags)) == len(flags), "Malformed or duplicate qualifier flags")
        q = profile[site]
        q["native_rows"] += 1
        if not row["is_core"]:
            q["padding_native_rows"] += 1
            continue
        require(MONTH_START <= stamp < MONTH_END, "Core record lies outside August 2021")
        q["core_native_rows"] += 1
        q["core_qualifier_counts_verbatim"][row["source_qualifiers_json"]] += 1
        q["estimated_flag_qualifier_disagreements"] += int(row["is_estimated"] != ("e" in flags))
        if row["is_exact_utc_hour"]:
            q["exact_core_rows"] += 1
            q["exact_core_qualifier_counts_verbatim"][row["source_qualifiers_json"]] += 1
            q["exact_core_numeric_invalid"] += int(not row["numeric_valid"])
            q["exact_core_estimated"] += int(row["is_estimated"])
            exact[site][stamp] = {"numeric_valid": row["numeric_valid"], "is_estimated": row["is_estimated"],
                                  "qualifiers": tuple(flags), "qualifiers_verbatim": row["source_qualifiers_json"]}
    for site, q in profile.items():
        q["expected_exact_core_hours"] = (MONTH_END - MONTH_START) // HOUR
        q["missing_exact_core_hours"] = q["expected_exact_core_hours"] - len(exact[site])
        for key in ("core_qualifier_counts_verbatim", "exact_core_qualifier_counts_verbatim"):
            q[key] = dict(sorted(q[key].items()))
    return exact, profile


def exclusion_reasons(row, policy):
    require(policy in POLICIES, "Unknown declared quality policy")
    if row is None:
        return ("missing_exact_hour",)
    flags = row["qualifiers"]
    qualifier_admitted = (flags == ("A",) if policy == "exact_A_only"
                          else "A" in flags and set(flags) <= {"A", "R"})
    reasons = []
    if not row["numeric_valid"]:
        reasons.append("numeric_valid_false")
    if row["is_estimated"]:
        reasons.append("estimated_flag_true")
    if not qualifier_admitted:
        reasons.append("qualifiers_not_admitted")
    return tuple(reasons)


def origin_axis():
    origins = tuple(range(ORIGIN_START, ORIGIN_END + HOUR, HOUR))
    require(len(origins) == 556 and origins[0] - 169 * HOUR >= MONTH_START
            and origins[-1] + HORIZON * HOUR < MONTH_END, "Common interior-origin bounds differ")
    return origins


def analyze_gauge(exact, origins, *, history_hours, policy):
    require(history_hours in HISTORY_OPTIONS and policy in POLICIES, "Unknown lookback or policy")
    require(type(origins) is tuple and len(origins) == len(set(origins))
            and all(type(c) is int and c % HOUR == 0 for c in origins), "Unique exact UTC origins required")
    count = {"requested_origins": len(origins), "history_complete": 0, "targets_complete": 0,
             "joint_USGS_complete": 0, "history_incomplete": 0, "targets_incomplete": 0,
             "requested_history_slot_occurrences": len(origins) * history_hours,
             "requested_target_slot_occurrences": len(origins) * HORIZON}
    cells = {name: Counter({reason: 0 for reason in REASONS}) for name in ("history", "targets")}
    origin_rejections = {name: Counter({reason: 0 for reason in REASONS}) for name in ("history", "targets")}
    jointly_complete = set()
    for origin in origins:
        last_history = origin - HISTORY_LAG * HOUR
        history_times = tuple(last_history - h * HOUR for h in range(history_hours - 1, -1, -1))
        target_times = tuple(origin + h * HOUR for h in range(1, HORIZON + 1))
        require(history_times[0] >= MONTH_START and target_times[-1] < MONTH_END,
                "Requested lookback/target window leaves the pinned month")
        complete = {}
        for kind, times in (("history", history_times), ("targets", target_times)):
            failures = [exclusion_reasons(exact.get(t), policy) for t in times]
            for reasons in failures:
                cells[kind].update(reasons)
            origin_rejections[kind].update({r for reasons in failures for r in reasons})
            complete[kind] = not any(failures)
            count[kind + "_complete"] += int(complete[kind])
            count[kind + "_incomplete"] += int(not complete[kind])
        joint = complete["history"] and complete["targets"]
        count["joint_USGS_complete"] += int(joint)
        if joint:
            jointly_complete.add(origin)
    count["joint_USGS_complete_fraction"] = count["joint_USGS_complete"] / len(origins) if origins else None
    count["origin_exclusions_by_reason_overlap"] = {kind: dict(values) for kind, values in origin_rejections.items()}
    count["slot_exclusion_occurrences_by_reason_overlap"] = {kind: dict(values) for kind, values in cells.items()}
    return count, jointly_complete


def prefix_check(exact, origins, *, history_hours, policy):
    """Independent window-count arithmetic on the complete 744-hour month axis."""
    prefix = [0]
    for t in range(MONTH_START, MONTH_END, HOUR):
        prefix.append(prefix[-1] + int(not exclusion_reasons(exact.get(t), policy)))
    result = {"history_complete": 0, "targets_complete": 0, "joint_USGS_complete": 0}
    for origin in origins:
        i = (origin - MONTH_START) // HOUR
        end_h = i - HISTORY_LAG + 1
        history_ok = prefix[end_h] - prefix[end_h - history_hours] == history_hours
        target_ok = prefix[i + HORIZON + 1] - prefix[i + 1] == HORIZON
        result["history_complete"] += int(history_ok)
        result["targets_complete"] += int(target_ok)
        result["joint_USGS_complete"] += int(history_ok and target_ok)
    return result


def build_report(rows, provenance):
    exact, profiles = index_metadata_rows(rows)
    origins = origin_axis()
    results, policy_counts, comparisons = [], [], []
    for site in GAUGES:
        for policy in POLICIES:
            counts = Counter({reason: 0 for reason in REASONS})
            admitted = 0
            for row in exact[site].values():
                reasons = exclusion_reasons(row, policy)
                admitted += int(not reasons)
                counts.update(reasons)
            policy_counts.append({"site": site, "policy": policy, "exact_core_rows": len(exact[site]),
                                  "admitted_exact_core_hours": admitted,
                                  "rejected_present_exact_core_hours": len(exact[site]) - admitted,
                                  "present_hour_exclusions_by_reason_overlap": dict(counts)})
            cohorts = {}
            for length in HISTORY_OPTIONS:
                count, cohort = analyze_gauge(exact[site], origins, history_hours=length, policy=policy)
                alternative = prefix_check(exact[site], origins, history_hours=length, policy=policy)
                require(all(count[key] == value for key, value in alternative.items()), "Independent prefix-window count differs")
                results.append({"site": site, "history_hours": length, "policy": policy, **count,
                                "prefix_window_count_crosscheck": "PASS"})
                cohorts[length] = cohort
            require(cohorts[168] <= cohorts[24], "Long-lookback completeness unexpectedly exceeds nested short history")
            comparisons.append({"site": site, "policy": policy, "common_requested_origins": len(origins),
                                "joint_complete_both_lookbacks": len(cohorts[168] & cohorts[24]),
                                "joint_complete_L24_only": len(cohorts[24] - cohorts[168]),
                                "joint_complete_L168_only": len(cohorts[168] - cohorts[24])})
    return {"schema": "hydra-development-usgs-metadata-availability/v1", "status": "COMPLETED_METADATA_ONLY_AUDIT",
            "artifact_created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "USGS-only metadata availability; not full HRRR eligibility or forecast performance",
            "sources": provenance, "decoded_columns": list(METADATA_COLUMNS),
            "source_integrity": {"whole_file_bytes_hashed": True, "only_selected_metadata_columns_decoded": True,
                                 "numeric_valid_is_retained_acquisition_QA_not_recomputed_from_discharge": True,
                                 "duplicate_native_keys": 0, "qualifier_strings_preserved_verbatim": True},
            "common_origin_window": {"start_inclusive_utc": iso(ORIGIN_START), "end_inclusive_utc": iso(ORIGIN_END),
                                     "origins_per_gauge": len(origins), "timezone": "UTC",
                                     "history_lag_hours": HISTORY_LAG, "target_horizons_hours": list(range(1, 19)),
                                     "earliest_L168_history_utc": iso(ORIGIN_START - 169 * HOUR),
                                     "latest_target_utc": iso(ORIGIN_END + 18 * HOUR)},
            "policies": {"exact_A_only": "Qualifier tuple exactly (A,); numeric_valid true and is_estimated false",
                         "approved_A_optional_R_no_e_or_P": "A required; only A/R flags allowed, order independent; numeric_valid true and is_estimated false",
                         "unknown_and_duplicate_flags": "Unknown flags rejected by both policies; duplicate/empty/malformed flag entries fail metadata validation",
                         "same_input_and_target_policy": True, "historical_publication_assumption": "Two-hour history lag is a simulation, not historical publication proof"},
            "monthly_metadata_profiles": profiles, "policy_counts": policy_counts, "results": results,
            "lookback_comparisons_on_identical_origins": comparisons,
            "counting_notes": ["Each origin is tested on all exact history slots and all H18 target slots",
                               "Missing exact hours are never replaced by quarter-hour samples",
                               "Slot occurrences count repeated use across overlapping windows; they are not unique source-hour counts",
                               "Reason counts may overlap; history/target incomplete totals count each origin once",
                               "Source padding rows are counted but excluded from August core availability"],
            "decision": "No lookback or quality policy selected automatically; this single-month audit does not establish whole-study coverage",
            "execution": {"discharge_or_source_value_columns_decoded": False, "protected_rows_or_outcomes_read": False,
                          "model_operations": 0, "scoring_operations": 0, "network_requests": 0,
                          "imputation": False, "source_files_modified": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=ROOT / ".local/inputs/2021-08.parquet")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / ".local/inputs/usgs-source-manifest.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/development-availability.json")
    args = parser.parse_args()
    require(args.report.resolve() not in {args.native.resolve(), args.source_manifest.resolve()}
            and not args.report.is_symlink(), "Report cannot overwrite/link to source inputs")
    started = time.monotonic()
    rows, provenance = load_metadata(args.native, args.source_manifest)
    report = build_report(rows, provenance)
    report["execution"]["elapsed_seconds"] = time.monotonic() - started
    report["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "comparisons": len(report["results"]),
                      "origins_per_gauge": len(origin_axis()), "discharge_columns_decoded": False}))


if __name__ == "__main__":
    main()
