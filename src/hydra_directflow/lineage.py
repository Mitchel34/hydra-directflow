"""Read old and current sealed-index lineage without rewriting source records.

A missing summary key is resolved only from the SHA256-authenticated bundle.
Missing byte counts never disable hashing. The adapter describes lineage and
manifest metadata; numerical source verification remains in multicycle.py.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re

from .local_data import MAX_BYTES, read_cycle_tar, sha256, verified_local_bytes
from .reference import ASSESSMENT_START, require

SCHEMA = "hydra-directflow-lineage-adapter/v1"
MAX_COMMIT_BYTES = 16384


def _cycle_guard(cycle_id):
    require(isinstance(cycle_id, str) and re.fullmatch(r"\d{8}_\d{2}Z", cycle_id), "Invalid cycle ID")
    try:
        reference = int(datetime.strptime(cycle_id, "%Y%m%d_%HZ").replace(tzinfo=timezone.utc).timestamp())
    except ValueError as exc:
        raise ValueError("Invalid cycle clock") from exc
    require(reference + 48 * 3600 < ASSESSMENT_START, "Protected assessment archive access is disabled")


def _identity(identity, *, name, byte_limit, bytes_required):
    require(isinstance(identity, dict) and re.fullmatch(r"[0-9a-f]{64}", str(identity.get("sha256", ""))),
            name + " requires an exact SHA256")
    size = identity.get("bytes")
    require((size is None and not bytes_required) or (type(size) is int and 0 < size <= byte_limit),
            name + " byte count is missing or outside the bound")
    return identity["sha256"], size


def _same_bundle(first, second):
    require(first["sha256"] == second["sha256"] and first["bytes"] == second["bytes"],
            "Index and commit bundle identities conflict")


def authenticate_bundle_identity(spec, index_row):
    """Resolve a bundle identity through exact original sealed source identities.

    A commit path is required for commit-kind rows. Bundle-kind rows can use the
    direct sealed bundle hash without opening a commit; if a commit is supplied,
    it is authenticated and must agree with the direct bundle identity.
    """
    cycle_id = spec["cycle_id"]
    _cycle_guard(cycle_id)  # Before any local source opens.
    require(index_row.get("cycle") == cycle_id and index_row.get("location_kind") in ("commit", "bundle")
            and index_row.get("original_quality_flags_rewritten") is False
            and type(index_row.get("data_complete")) is bool, "Unsupported qualified cycle lineage")
    locations = index_row.get("source_locations")
    require(isinstance(locations, dict), "Sealed source locations are required")
    kind = index_row["location_kind"]
    indexed_bundle = locations.get("bundle")
    if indexed_bundle is not None:
        _identity(indexed_bundle, name="Sealed bundle", byte_limit=MAX_BYTES, bytes_required=True)
        require(indexed_bundle.get("verified") is True, "Sealed bundle must retain verified identity")
    if kind == "bundle":
        require(indexed_bundle is not None, "Bundle-kind row lacks its sealed bundle identity")
    use_commit = kind == "commit" or spec.get("commit_path") is not None
    worker, commit_receipt, bundle = None, None, indexed_bundle
    if use_commit:
        require(spec.get("commit_path") is not None and "commit" in locations,
                "Commit-kind lineage requires the original local commit")
        commit_identity = locations["commit"]
        commit_sha, expected_bytes = _identity(commit_identity, name="Sealed commit",
                                              byte_limit=MAX_COMMIT_BYTES, bytes_required=False)
        raw = verified_local_bytes(spec["commit_path"], commit_sha, expected_bytes=expected_bytes,
                                   max_bytes=MAX_COMMIT_BYTES)
        commit = json.loads(raw)
        require(commit.get("schema") == "hrrr-verified-cloud-cycle/v1" and commit.get("cycle") == cycle_id,
                "Commit cycle/schema differs")
        bundle = commit.get("bundle")
        _identity(bundle, name="Committed bundle", byte_limit=MAX_BYTES, bytes_required=True)
        require(bundle.get("verified") is True, "Committed bundle was not verified")
        worker = commit.get("worker_manifest")
        require(isinstance(worker, dict) and worker.get("cycle") == cycle_id
                and worker.get("bundle_bytes") == bundle["bytes"]
                and worker.get("bundle_sha256") == bundle["sha256"]
                and re.fullmatch(r"[0-9a-f]{64}", str(worker.get("manifest_sha256", ""))),
                "Commit worker manifest binding differs")
        require(type(worker.get("record_count")) is int and 0 <= worker["record_count"] <= 145,
                "Commit record count is outside the supported bound")
        if indexed_bundle is not None:
            _same_bundle(indexed_bundle, bundle)
        commit_receipt = {"sha256": commit_sha, "bytes": len(raw),
                          "byte_count_source": "SEALED_INDEX" if expected_bytes is not None else "AUTHENTICATED_LOCAL_BYTES"}
    receipt = {"schema": SCHEMA, "cycle_id": cycle_id, "index_location_kind": kind,
               "binding_route": "SEALED_INDEX_COMMIT_BUNDLE" if use_commit else "SEALED_INDEX_BUNDLE",
               "commit": commit_receipt,
               "archive_sha256": bundle["sha256"], "archive_bytes": bundle["bytes"],
               "source_summaries_rewritten": False}
    return bundle, worker, receipt


def availability_view(index_row, manifest, members):
    """Keep original availability or label a complete legacy manifest-derived view.

    Legacy producers had no AVAILABILITY.json. The derived view is explicitly
    adapter metadata; source field/record quality and publication flags stay
    untouched. It is allowed only for a sealed complete outcome with the exact
    original group/lead inventory, never to fill a source gap.
    """
    if "AVAILABILITY.json" in members:
        return json.loads(members["AVAILABILITY.json"])
    require(manifest.get("schema") == "HRRR-raw-cycle-acquisition/v1"
            and "availability" not in manifest and index_row["data_complete"] is True,
            "Legacy availability can only derive from a sealed complete legacy outcome")
    cycle_id = index_row["cycle"]
    selected_six_hour = int(cycle_id[9:11]) % 6 == 0
    weather = ("temperature", "humidity", "precipitation", "snow_water_equivalent")
    expected = {group: list(range(1, 37 if selected_six_hour else 19)) for group in weather}
    if selected_six_hour:
        expected["soil_moisture"] = [0]
    require(sorted(manifest.get("groups", [])) == sorted(expected), "Legacy available group inventory is incomplete")
    records = json.loads(members["RAW_PROVENANCE.json"])
    require(isinstance(records, list) and len(records) == manifest["record_count"] <= 145,
            "Legacy original record inventory differs")
    found = {group: [] for group in expected}
    for index, record in enumerate(records):
        require(record.get("group") in expected and type(record.get("lead")) is int
                and record.get("record_index") == index, "Invalid legacy original record identity")
        found[record["group"]].append(record["lead"])
    require(all(sorted(found[group]) == leads for group, leads in expected.items()),
            "Legacy available group lead inventory is incomplete or duplicated")
    origin = "ADAPTER_DERIVED_FROM_ORIGINAL_MANIFEST_AND_RECORDS"
    return {"schema": "hydra-legacy-derived-availability/v1", "cycle": cycle_id,
            "metadata_origin": origin, "data_complete": True,
            "coverage_status": "ALL_REQUESTED_GROUPS_AVAILABLE", "expected_groups": sorted(expected),
            "groups": [{"cycle": cycle_id, "group": group, "source_key": None,
                        "status": "ADAPTER_DERIVED_COMPLETE_GROUP", "record_count": len(leads),
                        "leads": leads, "metadata_origin": origin} for group, leads in expected.items()]}


def normalize_manifest_summary(index_row, manifest, availability, worker=None):
    """Compare supplied summaries; derive omitted summaries from original bytes."""
    cycle_id = index_row["cycle"]
    require(manifest.get("cycle") == cycle_id and availability.get("cycle") == cycle_id,
            "Manifest/availability cycle differs")
    legacy = availability.get("schema") == "hydra-legacy-derived-availability/v1"
    require((legacy and manifest.get("schema") == "HRRR-raw-cycle-acquisition/v1" and "availability" not in manifest)
            or (not legacy and manifest.get("availability") == availability),
            "Manifest and original availability differ")
    n = manifest.get("record_count")
    require(type(n) is int and 0 <= n <= 145, "Manifest record count is outside supported bound")
    require(type(availability.get("data_complete")) is bool and
            availability["data_complete"] is index_row["data_complete"] and
            ((legacy and "data_complete" not in manifest) or manifest.get("data_complete") is index_row["data_complete"]),
            "Manifest/index completeness conflict")
    groups = availability.get("groups")
    require(isinstance(groups, list) and all(isinstance(g, dict) and isinstance(g.get("group"), str) for g in groups),
            "Original availability groups are required")
    require(len({g["group"] for g in groups}) == len(groups), "Duplicate original availability group")
    complete_status = "ADAPTER_DERIVED_COMPLETE_GROUP" if legacy else "AVAILABLE_VERIFIED"
    available = sorted(g["group"] for g in groups if g.get("status") == complete_status)
    missing = {g["group"]: g["status"] for g in groups if g.get("status") != complete_status}
    require(all(type(g.get("record_count")) is int and g["record_count"] >= 0 for g in groups)
            and sum(g["record_count"] for g in groups) == n, "Availability record inventory differs")
    require(sorted(manifest.get("groups", [])) == available, "Manifest available groups differ")
    require((legacy and "coverage_status" not in manifest) or
            manifest.get("coverage_status") == availability.get("coverage_status"), "Manifest coverage status differs")
    supplied = {"record_count": n, "groups_available": available,
                "coverage_status": availability.get("coverage_status"), "groups_missing": missing}
    derived = []
    for key, value in supplied.items():
        if key not in index_row:
            derived.append(key)
        else:
            actual = sorted(index_row[key]) if key == "groups_available" else index_row[key]
            require(actual == value, "Sealed index " + key + " conflicts with authenticated manifest")
    if worker is not None:
        require(worker["record_count"] == n, "Commit and original manifest record counts differ")
    return {**supplied, "data_complete": availability["data_complete"],
            "derived_summary_keys": sorted(derived)}


def read_authenticated_bundle(spec, index_row):
    """Return members, manifest, and a portable lineage receipt for all variants."""
    bundle, worker, receipt = authenticate_bundle_identity(spec, index_row)
    require(Path(spec["archive_path"]).lstat().st_size == bundle["bytes"], "Archive byte count differs")
    members, manifest = read_cycle_tar(spec["archive_path"], bundle["sha256"], cycle_id=spec["cycle_id"], allow_legacy=True)
    manifest_sha = sha256(members["CYCLE_MANIFEST.json"])
    if worker is not None:
        require(manifest_sha == worker["manifest_sha256"], "Commit manifest binding differs")
    availability = availability_view(index_row, manifest, members)
    summary = normalize_manifest_summary(index_row, manifest, availability, worker)
    receipt.update({"manifest_sha256": manifest_sha, "normalized_summary": summary,
                    "payload_schema": manifest.get("schema"),
                    "availability_metadata_origin": availability.get("metadata_origin", "ORIGINAL_AVAILABILITY_JSON"),
                    "summary_authority": "ORIGINAL_SHA256_AUTHENTICATED_MANIFEST_AND_RECORDS" if
                         availability.get("schema") == "hydra-legacy-derived-availability/v1" else
                         "ORIGINAL_SHA256_AUTHENTICATED_MANIFEST_AND_AVAILABILITY"})
    return members, manifest, receipt
