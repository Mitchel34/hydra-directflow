#!/usr/bin/env python3
"""Rerun frozen eligibility helpers using owned copies and July's safe export.

The active corpus-bound core is never edited or monkey-patched. This adapter
never opens raw July 2024 and projects only metadata from its verified export.
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import stat
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hydra_directflow import eligibility as core
from hydra_directflow.development_export import inspect_development_export, _validate_development_coordinates
from hydra_directflow.local_data import GAUGES, USGS_MANIFEST_SHA256, verified_local_bytes


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path, maximum=16 * 1024**2):
    path = Path(path)
    state = path.lstat()
    require(stat.S_ISREG(state.st_mode) and not getattr(state, "st_flags", 0) & 0x40000000
            and 0 < state.st_size <= maximum, "Audit input must be a bounded resident regular file")
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest(), len(raw)


def metadata_with_export(manifest_path, paths, export_reference):
    """Withhold raw July before calling the unchanged native metadata loader."""
    require(set(export_reference) == {"2024-07"}, "Exactly one pinned July development export required")
    safe_paths = {**paths, "2024-07": None}
    exact, known, inventory = core.inventory_usgs(manifest_path, safe_paths)
    require("2024-07" not in known, "Raw July must remain withheld from the native loader")
    manifest = json.loads(verified_local_bytes(manifest_path, USGS_MANIFEST_SHA256, max_bytes=2 * 1024**2))
    sources = [item for item in manifest["native_partitions"] if Path(item["path"]).name == "2024-07.parquet"]
    require(len(sources) == 1, "July source-manifest identity must be unique")
    identity = inspect_development_export(export_reference["2024-07"], sources[0])
    payload = identity["payload"]
    path = Path(export_reference["2024-07"]["directory"]) / payload["name"]
    raw = verified_local_bytes(path, payload["sha256"], expected_bytes=payload["bytes"], max_bytes=4 * 1024**2)
    from pyarrow import BufferReader
    import pyarrow.parquet as pq
    table = pq.read_table(BufferReader(raw), columns=list(core.METADATA_COLUMNS), use_threads=False)
    require(tuple(table.column_names) == core.METADATA_COLUMNS and table.num_rows == identity["exported_rows"] <= 30000,
            "Development-export metadata projection/count differs")
    rows = table.to_pylist()
    _validate_development_coordinates(rows)  # Check every exported coordinate, before eligibility use.
    indexed, profile = core.metadata_rows(rows, "2024-07")
    for site in GAUGES:
        require(not (set(exact[site]) & set(indexed[site])), "Duplicate source hour across metadata routes")
        exact[site].update(indexed[site])
    known.add("2024-07")
    replacement = {"month": "2024-07", "path": str(path), "sha256": payload["sha256"],
                   "expected_bytes": payload["bytes"], "bytes": len(raw), "metadata_known": True,
                   "status": "VERIFIED_DEVELOPMENT_EXPORT_METADATA_READ", "profile": profile,
                   "returned_metadata_rows": table.num_rows, "development_export_identity": identity,
                   "original_source_identity": {"sha256": sources[0]["sha256"], "bytes": sources[0]["bytes"]}}
    inventory = [replacement if item["month"] == "2024-07" else item for item in inventory]
    return exact, known, inventory


OWNED_MONTHS = ("2018-12", "2019-06", "2019-09", "2021-06", "2021-07", "2021-11", "2021-12", "2022-02", "2022-04", "2023-03", "2023-12")


def validate_owned_entries(receipt, manifest):
    """Validate every logical source identity before forming any owned path."""
    entries = receipt.get("files")
    require(type(entries) is list and len(entries) == len(OWNED_MONTHS), "Exactly eleven owned receipt entries required")
    months = [item.get("month") for item in entries]
    require(all(type(month) is str and re.fullmatch(r"\d{4}-\d{2}", month) for month in months)
            and len(set(months)) == len(months) and set(months) == set(OWNED_MONTHS),
            "Owned receipt months must be the exact unique eleven approved native months")
    native = {Path(item["path"]).stem: item for item in manifest["native_partitions"]}
    require(len(native) == len(manifest["native_partitions"]), "Duplicate source-manifest partition")
    for item in entries:
        original = native.get(item["month"])
        require(original is not None and type(item.get("bytes")) is int and item["bytes"] == original["bytes"]
                and type(item.get("sha256")) is str and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                and item["sha256"] == original["sha256"], "Owned receipt per-file identity differs from pinned source manifest")
    require(sum(item["bytes"] for item in entries) == 774216, "Owned receipt total differs")
    return entries


def coverage_statement(known, inventory):
    expected = set(core.months_between(core.DEVELOPMENT_START - 169 * 3600, core.DEVELOPMENT_END))
    require(len(inventory) == len(expected) and {item["month"] for item in inventory} == expected
            and set(known) <= expected, "Owned audit development month inventory differs")
    complete = len(known) == len(expected)
    text = ("All development USGS admission metadata is known through72 native months and the verified July export"
            if complete else f"{len(known)} of {len(expected)} development USGS metadata months are known; remaining coverage stays UNKNOWN")
    return complete, text + "; HRRR structural counts reuse the earlier inventory and remain separate from numeric case admission."


def audit_owned(*, base_report_path, base_inventory_path, owned_directory, materialization_receipt_path,
                usgs_manifest_path, july_export_reference_path):
    started = time.monotonic()
    base, base_sha, _ = read_json(base_report_path, 1024**2)
    private_base, private_sha, private_bytes = read_json(base_inventory_path)
    require(base["private_inventory_identity"] == {"sha256": private_sha, "bytes": private_bytes}, "Base public/private inventory binding differs")
    core_before = hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
    require(base["implementation_sha256"]["eligibility.py"] == core_before, "Base report and frozen core differ")
    receipt, receipt_sha, _ = read_json(materialization_receipt_path, 128 * 1024)
    require(receipt["status"] == "VERIFIED_ALL_11_OWNED_COPIES" and receipt["verified_files"] == 11
            and receipt["verified_encoded_bytes"] == 774216 and receipt["source_manifest_sha256"] == USGS_MANIFEST_SHA256,
            "Owned materialization receipt differs")
    manifest_raw = verified_local_bytes(usgs_manifest_path, USGS_MANIFEST_SHA256, max_bytes=2 * 1024**2)
    owned_entries = validate_owned_entries(receipt, json.loads(manifest_raw))
    paths = {item["month"]: item["path"] for item in private_base["usgs_partitions"] if item.get("path")}
    for item in owned_entries:
        require(item["month"] != "2024-07", "Owned materialization cannot admit raw July")
        path = Path(owned_directory) / (item["month"] + ".parquet")
        verified_local_bytes(path, item["sha256"], expected_bytes=item["bytes"], max_bytes=512 * 1024)
        paths[item["month"]] = str(path)
    exports, export_ref_sha, _ = read_json(july_export_reference_path, 65536)
    exact, known, usgs = metadata_with_export(usgs_manifest_path, paths, exports)
    splits = core.validate_splits(base["splits"])
    hrrr = private_base["hrrr_cycles"]
    results = [row for split in splits for length in (24, 168)
               for row in core.count_split(split, exact, known, hrrr, history_hours=length)]
    from collections import Counter
    report = deepcopy(base)
    all_known, coverage_text = coverage_statement(known, usgs)
    report.update(schema="hydra-owned-development-eligibility-metadata/v1",
        status="COMPLETED_ALL_DEVELOPMENT_USGS_METADATA_KNOWN" if all_known else "COMPLETED_WITH_EXPLICIT_UNKNOWN_COVERAGE",
        created_at_utc=datetime.now(timezone.utc).isoformat(), results=results,
        source_base_report_sha256=base_sha, source_base_inventory_sha256=private_sha,
        materialization_receipt_sha256=receipt_sha, july_export_reference_sha256=export_ref_sha,
        hrrr_inventory_reused_from_authenticated_base=True, hrrr_inventory_snapshot_at_utc=base["created_at_utc"],
        frozen_core_unchanged=True,
        usgs_inventory={"required_months": len(usgs), "verified_metadata_months": len(known),
                        "unknown_metadata_months": len(usgs) - len(known),
                        "status_counts": dict(Counter(item["status"] for item in usgs)),
                        "months": [{key: value for key, value in item.items() if key != "path"} for item in usgs]},
        aggregate_admission_proofs={"enabled": False, "proofs": [], "proven_gauge_months": 0,
                                   "reason": "DIRECT_NATIVE_OR_VERIFIED_EXPORT_METADATA_NOW_AVAILABLE"},
        execution={"elapsed_seconds": time.monotonic() - started, "discharge_value_columns_decoded": False,
                   "protected_numeric_values_decoded": False, "raw_july2024_read": False,
                   "hrrr_payloads_read": 0, "network_requests": 0, "hydrations": 0,
                   "model_operations": 0, "scaler_fits": 0, "imputation": False, "source_files_modified": False,
                   "whole_file_encoded_bytes_hashed_for_identity": True})
    report["implementation_sha256"]["scripts/audit_owned_development_eligibility.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["implementation_sha256"]["development_export.py"] = hashlib.sha256((ROOT / "src/hydra_directflow/development_export.py").read_bytes()).hexdigest()
    report["interpretation"].append(coverage_text)
    private = {"schema": "hydra-private-owned-development-metadata-inventory/v1", "created_at_utc": report["created_at_utc"],
               "source_identities": private_base["source_identities"], "usgs_partitions": usgs, "hrrr_cycles": hrrr,
               "materialization_receipt_sha256": receipt_sha, "source_base_inventory_sha256": private_sha,
               "development_export_reference": exports}
    require(hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest() == core_before, "Frozen eligibility core changed during audit")
    return report, private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base-report", "base-inventory", "owned-directory", "materialization-receipt", "usgs-manifest", "july-export-reference", "report", "private-inventory"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    require(args.private_inventory.resolve().is_relative_to((ROOT / ".local").resolve()), "Private inventory must remain ignored")
    require(args.report.resolve() != args.private_inventory.resolve() and
            not any(path.exists() or path.is_symlink() for path in (args.report, args.private_inventory)), "Outputs must be distinct new files")
    report, private = audit_owned(base_report_path=args.base_report, base_inventory_path=args.base_inventory,
        owned_directory=args.owned_directory, materialization_receipt_path=args.materialization_receipt,
        usgs_manifest_path=args.usgs_manifest, july_export_reference_path=args.july_export_reference)
    raw = (json.dumps(private, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    report["private_inventory_identity"] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    args.private_inventory.parent.mkdir(parents=True, exist_ok=True)
    with args.private_inventory.open("xb") as stream:
        stream.write(raw)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": report["status"], "known_months": report["usgs_inventory"]["verified_metadata_months"],
                      "unknown_months": report["usgs_inventory"]["unknown_metadata_months"], "raw_july2024_read": False,
                      "flow_columns_decoded": False, "frozen_core_unchanged": True}))


if __name__ == "__main__":
    main()
