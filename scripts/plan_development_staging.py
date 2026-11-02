#!/usr/bin/env python3
"""Construct a bounded retained-development staging plan; never execute it."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hydra_directflow.development_export import inspect_development_export
from hydra_directflow.eligibility import (DEVELOPMENT_START, DEVELOPMENT_END, inspect_local,
                                         months_between, read_required_index, required_six_hour_cycles)
from hydra_directflow.local_data import USGS_MANIFEST_SHA256, verified_local_bytes
from hydra_directflow.multicycle import INDEX_SHA256, INDEX_BYTES

ARCHIVE_BOUND = 16 * 1024**2
COMMIT_BOUND = 16 * 1024
BODY_CAP = 100 * 1024**3
GET_CAP = 20000
HOTSET_CAP = 2 * 1024**3
ALLOCATIONS = {"development_staging": 35, "cuda_qualification_pilot": 10, "scientific_compute": 75, "contingency": 30}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_json(path, maximum=16 * 1024**2):
    path = Path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not getattr(before, "st_flags", 0) & 0x40000000
            and 0 < before.st_size <= maximum, "Plan input must be a bounded resident regular file")
    raw = path.read_bytes()
    after = path.lstat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "Plan input changed while reading")
    return json.loads(raw), {"path": str(path.resolve()), "sha256": sha(raw), "bytes": len(raw)}


def validate_approval(value):
    require(value.get("schema") == "hydra-execution-budget-approval/v1"
            and value.get("approved_total_incremental_usd") == 150
            and value.get("allocations_usd") == ALLOCATIONS
            and sum(value["allocations_usd"].values()) == value["approved_total_incremental_usd"], "Combined approved allocation differs")
    require(value.get("phase_release", {}).get("development_staging") == "AUTHORIZED_WITH_DECLARED_LIMITS"
            and value.get("data_response_body_cap_bytes") == BODY_CAP
            and value.get("maximum_get_attempts") == GET_CAP
            and value.get("raw_staging_cap_bytes") == HOTSET_CAP, "Staging phase or limits differ")
    require(value.get("protected_assessment_access_authorized") is False
            and value.get("protected_scoring_authorized") is False
            and value.get("iam_or_security_changes_authorized") is False
            and value.get("automatic_budget_increase") is False, "Protected/IAM/budget scope differs")
    require(value["phase_release"].get("cuda_pilot") == "CONDITIONAL_ON_MEASURED_LOCAL_NEED_AND_BOUNDED_QUALIFICATION"
            and value["phase_release"].get("scientific_compute") == "CONDITIONAL_ON_PROTOCOL_COHORT_AND_CUDA_READINESS",
            "Conditional cloud phases must remain conditional")


def object_identity(value, *, maximum, optional_bytes=False):
    require(isinstance(value, dict) and re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", "")), "Exact object digest required")
    parsed = urllib.parse.urlsplit(value.get("durable_uri", ""))
    require(parsed.scheme == "gs" and parsed.netloc and parsed.path.startswith("/")
            and len(parsed.path) > 1 and not parsed.query and not parsed.fragment, "Pinned retained gs object required")
    generation = str(value.get("generation", ""))
    require(re.fullmatch(r"[1-9][0-9]{0,24}", generation), "Pinned object generation required")
    size = value.get("bytes")
    require((optional_bytes and size is None) or (type(size) is int and 0 < size <= maximum), "Object byte bound differs")
    return {"durable_uri": value["durable_uri"], "generation": generation, "sha256": value["sha256"], "bytes": size}


def local_candidate(path, expected_bytes):
    return {"path": str(Path(path).absolute()), **inspect_local(path, expected_bytes=expected_bytes)}


def build_cycle_manifest(rows, required_ids, inventory, augmentation, extra_archives=None):
    original = {item["cycle_id"]: item for item in inventory}
    augmented = {item["cycle_id"]: item for item in augmentation}
    require(len(original) == len(inventory) and len(augmented) == len(augmentation), "Duplicate inventory/proof cycle")
    require(set(original) == set(augmented) == set(required_ids), "Inventory/proof calendar differs from required sources")
    extras = {}
    for entry in extra_archives or []:
        extras.setdefault(entry["cycle"], []).append(entry["archive_path"])
    cycles, excluded = [], []
    for cycle_id in required_ids:
        row, prior, proof = rows.get(cycle_id), original[cycle_id], augmented[cycle_id]
        if row is None:
            require(prior["index_status"] == "ABSENT_WARMUP_OR_CALENDAR_UNKNOWN" and proof.get("bundle_identity") is None,
                    "Missing sealed source cannot have an invented bundle")
            excluded.append({"cycle_id": cycle_id, "reason": "OUTSIDE_SEALED_ACQUISITION_CALENDAR", "fetch_authorized": False})
            continue
        require(prior["source_locations"] == row["source_locations"] and prior["data_complete"] is row["data_complete"],
                "Inventory source differs from sealed index")
        locations = row["source_locations"]
        commit = object_identity(locations["commit"], maximum=COMMIT_BOUND, optional_bytes=True) if "commit" in locations else None
        bundle = object_identity(proof["bundle_identity"], maximum=ARCHIVE_BOUND) if proof.get("bundle_identity") else None
        direct = object_identity(locations["bundle"], maximum=ARCHIVE_BOUND) if locations.get("bundle") else None
        proof_route = "UNKNOWN_NEEDS_COMMIT_FETCH"
        if direct is not None:
            require(bundle == direct, "Recorded bundle differs from sealed bundle identity")
            proof_route = "SEALED_INDEX_BUNDLE"
        elif bundle is not None:
            require(commit is not None and proof.get("commit_sha256") == commit["sha256"]
                    and proof.get("commit_path") and proof.get("identity_source") == "AUTHENTICATED_RESIDENT_ORIGINAL_COMMIT",
                    "Known bundle lacks its authenticated original-commit proof")
            proof_route = "HASH_BOUND_RECORDED_AUTHENTICATED_COMMIT"
        commit_paths = [item["commit_path"] for item in prior.get("local_candidates", []) if item.get("commit_path")]
        if proof.get("commit_path"):
            commit_paths.append(proof["commit_path"])
        archive_paths = [item["archive_path"] for item in prior.get("local_candidates", []) if item.get("archive_path")]
        archive_paths += extras.get(cycle_id, [])
        commit_candidates, current_verified = [], False
        for path in dict.fromkeys(commit_paths):
            state = local_candidate(path, commit.get("bytes") if commit else None)
            if commit and state["status"] == "RESIDENT_UNVERIFIED_SHA":
                raw = verified_local_bytes(path, commit["sha256"], expected_bytes=commit["bytes"], max_bytes=COMMIT_BOUND)
                content = json.loads(raw)
                require(content["cycle"] == cycle_id, "Original commit cycle differs")
                confirmed = object_identity(content["bundle"], maximum=ARCHIVE_BOUND)
                require(bundle in (None, confirmed), "Original commit and recorded bundle conflict")
                bundle = confirmed
                commit["bytes"] = len(raw)
                state["status"] = "VERIFIED_COMMIT_METADATA"
                state["sha256"] = commit["sha256"]
                current_verified = True
            commit_candidates.append(state)
        if current_verified:
            proof_route = "CURRENT_AUTHENTICATED_LOCAL_COMMIT"
        if bundle is not None:
            require(bundle["durable_uri"].endswith("/cycles/" + cycle_id + ".tar"), "Bundle object does not name its source cycle")
        archives = [local_candidate(path, bundle["bytes"] if bundle else None) for path in dict.fromkeys(archive_paths)]
        if commit is not None:
            commit.update(max_response_body_bytes=COMMIT_BOUND, bytes_source="EXACT_BOUND_IDENTITY" if commit["bytes"] is not None else "UNKNOWN_BOUNDED_16_KIB")
        cycles.append({"cycle_id": cycle_id, "sealed_row_sha256": sha(canonical(row)),
                       "data_complete": row["data_complete"], "groups_available": row.get("groups_available"),
                       "groups_missing": row.get("groups_missing", {}), "record_count": row.get("record_count"),
                       "index_location_kind": row["location_kind"], "commit_identity": commit,
                       "bundle_identity": bundle, "bundle_proof_route": proof_route,
                       "commit_verification_required_before_bundle_use": proof_route in ("UNKNOWN_NEEDS_COMMIT_FETCH", "HASH_BOUND_RECORDED_AUTHENTICATED_COMMIT"),
                       "maximum_archive_response_body_bytes": ARCHIVE_BOUND,
                       "local_commit_candidates": commit_candidates, "local_archive_candidates": archives,
                       "archive_payloads_read_by_planner": False})
    return cycles, excluded


def build_usgs_manifest(source_manifest, eligibility_inventory, export_reference):
    originals = {Path(item["path"]).stem: item for item in source_manifest["native_partitions"]}
    prior = {item["month"]: item for item in eligibility_inventory}
    months = months_between(DEVELOPMENT_START - 169 * 3600, DEVELOPMENT_END)
    require(set(export_reference) == {"2024-07"}, "Exactly the verified July development export is required")
    result = []
    for month in months:
        original = originals[month]
        identity = {"month": month, "sha256": original["sha256"], "bytes": original["bytes"]}
        if month == "2024-07":
            exported = inspect_development_export(export_reference[month], original)
            payload_path = Path(export_reference[month]["directory"]) / "data.parquet"
            payload = exported["payload"]
            # Only committed development-export bytes are hashed, never raw July.
            verified_local_bytes(payload_path, payload["sha256"], expected_bytes=payload["bytes"], max_bytes=4 * 1024**2)
            result.append({"month": month, "stage_kind": "VERIFIED_DEVELOPMENT_EXPORT_ONLY",
                           "original_source_identity": identity, "development_export_reference": export_reference[month],
                           "export_identity": exported, "payload_identity": {"sha256": payload["sha256"], "bytes": payload["bytes"]},
                           "local_candidates": [{"path": str(payload_path), "status": "VERIFIED_EXPORT_PAYLOAD_HASH", "bytes": payload["bytes"]}],
                           "raw_mixed_month_staging_forbidden": True})
        else:
            candidates = []
            if prior.get(month, {}).get("path"):
                candidates.append(prior[month]["path"])
            candidates.append(original["path"])
            result.append({"month": month, "stage_kind": "RETAINED_NATIVE_DEVELOPMENT_ONLY",
                           "original_source_identity": identity, "payload_identity": identity,
                           "local_candidates": [local_candidate(path, original["bytes"]) for path in dict.fromkeys(candidates)],
                           "new_upstream_reacquisition_authorized_by_plan": False})
    return result


def source_bound_summary(cycles, usgs):
    known = [item for item in cycles if item["bundle_identity"] is not None]
    unknown = len(cycles) - len(known)
    archive_bytes = sum(item["bundle_identity"]["bytes"] for item in known) + unknown * ARCHIVE_BOUND
    commits = [item["commit_identity"] for item in cycles if item["commit_identity"]]
    commit_bytes = sum(item["bytes"] if item["bytes"] is not None else COMMIT_BOUND for item in commits)
    usgs_bytes = sum(item["payload_identity"]["bytes"] for item in usgs)
    # Include a one-byte response-overrun probe per conservatively counted GET.
    get_upper = len(cycles) + len(commits) + len(usgs)
    body_upper = archive_bytes + commit_bytes + usgs_bytes + get_upper
    require(get_upper <= GET_CAP and body_upper <= BODY_CAP, "Complete-source conservative bounds exceed the approved staging limits")
    return {"sealed_development_cycles": len(cycles), "known_bundle_identities": len(known),
            "known_bundle_bytes": sum(item["bundle_identity"]["bytes"] for item in known),
            "unresolved_bundle_identities": unknown, "archive_body_upper_bound_bytes": archive_bytes,
            "commit_body_upper_bound_bytes": commit_bytes, "usgs_encoded_payload_bytes": usgs_bytes,
            "all_sources_body_upper_bound_bytes": body_upper, "all_sources_get_upper_bound": get_upper,
            "bounds_ignore_local_hits_conservatively": True,
            "local_archive_status_counts": dict(Counter(item["status"] for cycle in cycles for item in cycle["local_archive_candidates"])),
            "local_commit_status_counts": dict(Counter(item["status"] for cycle in cycles for item in cycle["local_commit_candidates"])),
            "usgs_stage_kinds": dict(Counter(item["stage_kind"] for item in usgs)),
            "usgs_months_with_resident_candidate": sum(any(entry["status"] in ("RESIDENT_UNVERIFIED_SHA", "VERIFIED_EXPORT_PAYLOAD_HASH") for entry in item["local_candidates"]) for item in usgs)}


def make_plan(*, index_path, usgs_manifest_path, eligibility_inventory_path, byte_augmentation_path,
              july_export_reference_path, approval_path, resident_archive_inventory_path=None):
    inputs = {}
    inventory, inputs["eligibility_inventory"] = read_json(eligibility_inventory_path)
    augmentation, inputs["resident_commit_proof"] = read_json(byte_augmentation_path)
    manifest, inputs["usgs_manifest"] = read_json(usgs_manifest_path, 2 * 1024**2)
    export_reference, inputs["july_export_reference"] = read_json(july_export_reference_path, 65536)
    approval, inputs["budget_approval"] = read_json(approval_path, 65536)
    validate_approval(approval)
    require(inputs["usgs_manifest"]["sha256"] == USGS_MANIFEST_SHA256
            and inventory["source_identities"]["usgs_manifest_sha256"] == USGS_MANIFEST_SHA256
            and inventory["source_identities"]["qualified_index_sha256"] == INDEX_SHA256
            and augmentation["qualified_index_sha256"] == INDEX_SHA256, "Source authorities differ")
    required = required_six_hour_cycles()
    rows = read_required_index(index_path, required)
    require(len(required) == 8844 and len(rows) == 8841, "Fixed development source calendar differs")
    inputs["qualified_index"] = {"path": str(Path(index_path).resolve()), "sha256": INDEX_SHA256, "bytes": INDEX_BYTES}
    extra = []
    if resident_archive_inventory_path is not None:
        value, inputs["resident_archive_header_inventory"] = read_json(resident_archive_inventory_path)
        extra = value["resident_cycle_archives"]
    cycles, outside = build_cycle_manifest(rows, required, inventory["hrrr_cycles"], augmentation["cycles"], extra)
    require([item["cycle_id"] for item in outside] == ["20180711_06Z", "20180711_12Z", "20180711_18Z"], "Unexpected source-calendar exclusion")
    usgs = build_usgs_manifest(manifest, inventory["usgs_partitions"], export_reference)
    summary = source_bound_summary(cycles, usgs)
    limits = {"download_body_bytes": BODY_CAP, "get_attempts": GET_CAP, "raw_hotset_bytes": HOTSET_CAP,
              "archive_response_body_bytes": ARCHIVE_BOUND, "commit_response_body_bytes": COMMIT_BOUND}
    private = {"schema": "hydra-development-staging-plan/v1", "status": "PLAN_ONLY_NOT_EXECUTED",
               "sources": inputs, "limits": limits, "approved_staging_ceiling_usd": 35,
               "combined_ceiling_usd": 150, "allocations_usd": ALLOCATIONS,
               "active_phase": "DEVELOPMENT_STAGING_ONLY", "conditional_cloud_phases_released_by_plan": False,
               "protected_assessment_access_authorized": False, "iam_changes_authorized": False,
               "development_start_utc": "2018-07-12T00:00:00Z", "development_end_exclusive_utc": "2024-07-31T00:00:00Z",
               "cycles": cycles, "excluded_outside_calendar": outside, "usgs_partitions": usgs,
               "summary": summary, "planner_sha256": sha(Path(__file__).read_bytes()),
               "dependency_sha256": {name: sha((ROOT / "src/hydra_directflow" / name).read_bytes())
                                      for name in ("eligibility.py", "development_export.py", "local_data.py", "reference.py")},
               "execution_contract": ["Existing retained objects only; generation and SHA must be verified before consumption",
                    "Authenticate unresolved commits before admitting their bundle identities; charge failed/partial attempts to persistent global limits",
                    "Prefer verified basin cache hits; acquire raw data only on misses and evict only owned raw staging after durable basin commit",
                    "Keep owned raw archives, temporary partials, and retained monthly encoded inputs within the shared 2 GiB hotset",
                    "Unknown transport/residency must stop the chunk; it must never be relabeled as an upstream weather gap",
                    "Never fetch the three outside-calendar warmup IDs or raw July2024 USGS partition",
                    "Use only the pinned July development export; no protected scoring, new upstream USGS vintage, or IAM changes"]}
    private["plan_sha256"] = sha(canonical(private))
    private["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    public = {"schema": "hydra-development-staging-plan-summary/v1", "status": private["status"],
              "created_at_utc": private["created_at_utc"], "plan_sha256": private["plan_sha256"],
              "sources": {name: {key: value for key, value in identity.items() if key != "path"} for name, identity in inputs.items()},
              "limits": limits, "approved_staging_ceiling_usd": 35, "combined_ceiling_usd": 150,
              "allocations_usd": ALLOCATIONS, "summary": summary, "excluded_outside_calendar": outside,
              "july2024_route": "VERIFIED_DEVELOPMENT_EXPORT_ONLY_RAW_MIXED_MONTH_EXCLUDED",
              "planner_sha256": private["planner_sha256"], "dependency_sha256": private["dependency_sha256"],
              "execution": {"network_requests": 0, "authentication_requests": 0, "hydrations": 0,
                            "weather_payload_reads": 0, "flow_columns_decoded": False, "cloud_jobs": 0},
              "qualification": "Plan construction only; dollar ceilings are approved limits, not an invoice or cloud phase release"}
    return private, public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "usgs-manifest", "eligibility-inventory", "byte-augmentation", "july-export-reference", "approval", "private-plan", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resident-archive-inventory", type=Path)
    args = parser.parse_args()
    try:
        args.private_plan.resolve().relative_to((ROOT / ".local").resolve())
    except ValueError as error:
        raise ValueError("Private source plan must stay in this checkout's ignored .local directory") from error
    require(args.private_plan.resolve() != args.report.resolve()
            and not any(path.exists() or path.is_symlink() for path in (args.private_plan, args.report)), "Outputs must be distinct new files")
    private, public = make_plan(index_path=args.index, usgs_manifest_path=args.usgs_manifest,
        eligibility_inventory_path=args.eligibility_inventory, byte_augmentation_path=args.byte_augmentation,
        july_export_reference_path=args.july_export_reference, approval_path=args.approval,
        resident_archive_inventory_path=args.resident_archive_inventory)
    raw = (json.dumps(private, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    public["private_plan_identity"] = {"sha256": sha(raw), "bytes": len(raw)}
    args.private_plan.parent.mkdir(parents=True, exist_ok=True)
    with args.private_plan.open("xb") as stream:
        stream.write(raw)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(public, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": private["status"], "plan_sha256": private["plan_sha256"], **private["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
