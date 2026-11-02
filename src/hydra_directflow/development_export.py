"""Separately gated administrative export of the mixed July 2024 USGS month.

Planning never decodes source values. Execution can internally decode protected
rows while filtering Parquet row groups; it returns only artifact identities and
counts, never source values. The execution token is a software guard, not human
permission and not a protected-scoring permit. Obtain separate authorization
before executing against real mixed-month data.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import time
import uuid

from .local_data import GAUGES, USGS_MANIFEST_SHA256, iso, utc_epoch, verified_local_bytes
from .reference import ASSESSMENT_START, digest, require

MONTH = "2024-07"
MONTH_START = int(datetime(2024, 7, 1, tzinfo=timezone.utc).timestamp())
CUTOFF = ASSESSMENT_START
PLAN_SCHEMA = "hydra-development-boundary-export-plan/v1"
EXPORT_SCHEMA = "hydra-development-boundary-export/v1"
ADMIN_PREFIX = "ADMIN_DEVELOPMENT_EXPORT:"
PRIVATE_ROOT = Path(__file__).resolve().parents[2] / ".local"
MAX_SOURCE_BYTES = 512 * 1024
MAX_EXPORT_BYTES = 4 * 1024**2
MAX_ROWS = 30000
USGS_COLUMNS = ("site", "sample_epoch", "sample_time_utc", "source_units", "discharge_cfs",
                "source_qualifiers_json", "numeric_valid", "is_exact_utc_hour", "is_core",
                "source_wire_sha256", "historical_available_at_utc")
DECODE_SCOPE = "MIXED_SOURCE_ROW_GROUPS_MAY_INCLUDE_PROTECTED_VALUES"
ROW_RULE = "ORIGINAL_CORE_ROWS_IN_MONTH_STRICTLY_BEFORE_PROTECTED_START"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hex(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _implementation():
    directory = Path(__file__).resolve().parent
    return {name: _sha((directory / name).read_bytes())
            for name in ("development_export.py", "local_data.py", "reference.py")}


def _private_output(path):
    path = Path(path).absolute()
    require(not path.is_symlink(), "Export directory cannot be a symlink")
    try:
        path.resolve().relative_to(PRIVATE_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Development export must remain in this checkout's ignored .local directory") from exc
    require(path.resolve() != PRIVATE_ROOT.resolve(), "Use a dedicated export directory")
    return path


def _source_partition(manifest_path):
    raw = verified_local_bytes(manifest_path, USGS_MANIFEST_SHA256, max_bytes=2 * 1024**2)
    manifest = json.loads(raw)
    matches = [entry for entry in manifest["native_partitions"] if Path(entry["path"]).name == MONTH + ".parquet"]
    require(len(matches) == 1, "Boundary month must have one pinned original partition")
    entry = matches[0]
    require(_hex(entry.get("sha256")) and type(entry.get("bytes")) is int
            and 0 < entry["bytes"] <= MAX_SOURCE_BYTES, "Invalid pinned source partition identity")
    return {"month": MONTH, "sha256": entry["sha256"], "bytes": entry["bytes"]}


def _source_residency(path, expected_bytes):
    try:
        info = Path(path).lstat()
    except FileNotFoundError:
        return "MISSING"
    if not stat.S_ISREG(info.st_mode):
        return "NOT_REGULAR"
    if getattr(info, "st_flags", 0) & 0x40000000:
        return "NONRESIDENT_NO_HYDRATION"
    return "RESIDENT_UNVERIFIED" if info.st_size == expected_bytes else "WRONG_BYTE_COUNT"


def plan_development_export(*, source_path, source_manifest_path, output_directory):
    """Inspect pinned metadata and local file metadata only; create no artifacts."""
    partition = _source_partition(source_manifest_path)
    output = _private_output(output_directory)
    identity = {"schema": PLAN_SCHEMA, "source_manifest_sha256": USGS_MANIFEST_SHA256,
                "source_partition": partition, "month": MONTH, "start_inclusive_utc": iso(MONTH_START),
                "cutoff_exclusive_utc": iso(CUTOFF), "selection_rule": ROW_RULE,
                "source_location_sha256": _sha(str(Path(source_path).absolute()).encode()),
                "output_location_sha256": _sha(str(output.resolve()).encode()),
                "columns": list(USGS_COLUMNS), "implementation_sha256": _implementation(),
                "physical_decode_scope": DECODE_SCOPE, "protected_scoring_permitted": False}
    plan_sha = digest(identity)
    residency = _source_residency(source_path, partition["bytes"])
    return {**identity, "status": "PLAN_ONLY_NOT_EXECUTED", "plan_sha256": plan_sha,
            "administrative_plan_token": ADMIN_PREFIX + plan_sha,
            "token_is_human_authorization": False, "source_residency": residency,
            "ready_for_separately_authorized_execution": residency == "RESIDENT_UNVERIFIED" and not output.exists(),
            "output_already_exists": output.exists(), "source_values_decoded": False,
            "network_requests": 0, "cloud_operations": 0}


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path):
    path = Path(path)
    missing = []
    cursor = path
    while not cursor.exists():
        require(not cursor.is_symlink(), "Export parent cannot be a dangling symlink")
        missing.append(cursor)
        cursor = cursor.parent
    require(cursor.is_dir(), "Export parent is not a directory")
    for directory in reversed(missing):
        directory.mkdir()
        _sync_directory(directory.parent)


def _write(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_development_coordinates(rows):
    """Metadata checks use no discharge values and never include a row in errors."""
    require(len(rows) <= MAX_ROWS, "Development export row bound exceeded")
    seen = set()
    for row in rows:
        instant, site = row["sample_epoch"], row["site"]
        require(type(instant) is int and MONTH_START <= instant < CUTOFF
                and utc_epoch(row["sample_time_utc"]) == instant and row["is_core"] is True,
                "Development export contains a nondevelopment coordinate")
        require(site in GAUGES and (site, instant) not in seen, "Development export site/coordinate inventory differs")
        seen.add((site, instant))


def execute_development_export(*, source_path, source_manifest_path, output_directory,
                               execute=False, administrative_plan_token=None):
    """Plan by default; explicit execution needs its separate administrative token.

    This function cannot verify human consent. Real July 2024 execution requires
    separate user authorization outside this API, even if the token is known.
    """
    require(type(execute) is bool, "Execution mode must be an explicit boolean")
    plan = plan_development_export(source_path=source_path, source_manifest_path=source_manifest_path,
                                   output_directory=output_directory)
    if not execute:
        return plan
    require(type(administrative_plan_token) is str and administrative_plan_token == plan["administrative_plan_token"],
            "Exact administrative export token required; scoring permits are not accepted")
    require(plan["ready_for_separately_authorized_execution"], "Export source/output is not ready for administrative execution")
    source = plan["source_partition"]
    raw = verified_local_bytes(source_path, source["sha256"], expected_bytes=source["bytes"], max_bytes=MAX_SOURCE_BYTES)
    # Import and decode only after token, residency and complete source-hash checks.
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    try:
        table = pq.read_table(pa.BufferReader(raw), columns=list(USGS_COLUMNS))
        require(table.num_rows <= MAX_ROWS, "Source native row bound exceeded")
        selection = pc.and_(pc.and_(pc.greater_equal(table["sample_epoch"], MONTH_START),
                                   pc.less(table["sample_epoch"], CUTOFF)), pc.equal(table["is_core"], True))
        development = table.filter(selection).replace_schema_metadata(None)
    except Exception:
        # Third-party decode exceptions may contain input fragments. Never expose
        # those fragments from a mixed-month administrative operation.
        raise ValueError("Administrative mixed-month decoding or selection failed") from None
    coordinates = development.select(["site", "sample_epoch", "sample_time_utc", "is_core"]).to_pylist()
    _validate_development_coordinates(coordinates)
    buffer = io.BytesIO()
    pq.write_table(development, buffer, compression="zstd")
    payload = buffer.getvalue()
    require(0 < len(payload) <= MAX_EXPORT_BYTES, "Development export payload bound exceeded")
    commit = {"schema": EXPORT_SCHEMA, "status": "COMPLETE_DEVELOPMENT_ONLY", "month": MONTH,
              "start_inclusive_utc": iso(MONTH_START), "cutoff_exclusive_utc": iso(CUTOFF),
              "selection_rule": ROW_RULE, "source_manifest_sha256": USGS_MANIFEST_SHA256,
              "source_partition": source, "plan_sha256": plan["plan_sha256"],
              "administrative_token_sha256": _sha(administrative_plan_token.encode()),
              "administrative_scope": "EXPORT_ONLY_NO_SCORING_AUTHORIZATION",
              "token_is_human_authorization": False, "protected_scoring_permitted": False,
              "physical_decode_scope": DECODE_SCOPE, "protected_values_returned_or_logged": False,
              "artifact_created_at_utc": iso(int(time.time())),
              "source_rows": table.num_rows, "exported_rows": development.num_rows,
              "excluded_rows": table.num_rows - development.num_rows,
              "columns": list(USGS_COLUMNS), "implementation_sha256": plan["implementation_sha256"],
              "payload": {"name": "data.parquet", "sha256": _sha(payload), "bytes": len(payload)}}
    commit_raw = _canonical(commit)
    output = _private_output(output_directory)
    _mkdir_durable(output.parent)
    require(not output.exists() and not output.is_symlink(), "Refusing to overwrite an export")
    stage = output.with_name(output.name + ".staging-" + uuid.uuid4().hex)
    stage.mkdir()
    _write(stage / "data.parquet", payload)
    _write(stage / "COMMIT.json", commit_raw)
    _sync_directory(stage)
    require(not output.exists() and not output.is_symlink(), "Export target appeared during staging")
    stage.rename(output)
    _sync_directory(output.parent)
    # No rows, source values, dataframes or model outcomes leave this function.
    return {"schema": EXPORT_SCHEMA, "status": commit["status"], "month": MONTH,
            "commit_sha256": _sha(commit_raw), "payload_sha256": _sha(payload),
            "exported_rows": development.num_rows, "excluded_rows": commit["excluded_rows"],
            "cutoff_exclusive_utc": iso(CUTOFF), "physical_decode_scope": DECODE_SCOPE,
            "protected_values_returned_or_logged": False, "protected_scoring_permitted": False,
            "token_is_human_authorization": False, "network_requests": 0, "cloud_operations": 0}


def inspect_development_export(export_reference, source_partition):
    """Authenticate an externally pinned export commitment without value decode."""
    require(isinstance(export_reference, dict) and set(export_reference) == {"directory", "commit_sha256"}
            and _hex(export_reference["commit_sha256"]), "Pinned development export directory and commit SHA256 required")
    directory = _private_output(export_reference["directory"])
    info = directory.lstat()
    require(stat.S_ISDIR(info.st_mode) and not getattr(info, "st_flags", 0) & 0x40000000,
            "Export directory must be resident and real")
    require({p.name for p in directory.iterdir()} == {"COMMIT.json", "data.parquet"}, "Development export inventory differs")
    raw = verified_local_bytes(directory / "COMMIT.json", export_reference["commit_sha256"], max_bytes=65536)
    commit = json.loads(raw)
    expected_source = {"month": MONTH, "sha256": source_partition["sha256"], "bytes": source_partition["bytes"]}
    require(commit.get("schema") == EXPORT_SCHEMA and commit.get("status") == "COMPLETE_DEVELOPMENT_ONLY"
            and commit.get("month") == MONTH and commit.get("source_manifest_sha256") == USGS_MANIFEST_SHA256
            and commit.get("source_partition") == expected_source, "Development export source lineage differs")
    require(commit.get("start_inclusive_utc") == iso(MONTH_START) and commit.get("cutoff_exclusive_utc") == iso(CUTOFF)
            and commit.get("selection_rule") == ROW_RULE and commit.get("columns") == list(USGS_COLUMNS),
            "Development export cutoff or selection contract differs")
    require(_hex(commit.get("plan_sha256")) and commit.get("administrative_token_sha256") ==
            _sha((ADMIN_PREFIX + commit["plan_sha256"]).encode())
            and commit.get("administrative_scope") == "EXPORT_ONLY_NO_SCORING_AUTHORIZATION"
            and commit.get("token_is_human_authorization") is False
            and commit.get("protected_scoring_permitted") is False
            and commit.get("protected_values_returned_or_logged") is False
            and commit.get("physical_decode_scope") == DECODE_SCOPE,
            "Development export administrative scope differs")
    require(type(commit.get("exported_rows")) is int and 0 <= commit["exported_rows"] <= MAX_ROWS
            and type(commit.get("source_rows")) is int and commit["exported_rows"] <= commit["source_rows"] <= MAX_ROWS
            and commit.get("excluded_rows") == commit["source_rows"] - commit["exported_rows"],
            "Development export row accounting differs")
    implementations = commit.get("implementation_sha256")
    require(isinstance(implementations, dict) and set(implementations) ==
            {"development_export.py", "local_data.py", "reference.py"}
            and all(_hex(value) for value in implementations.values()), "Export implementation identities missing")
    payload = commit.get("payload", {})
    require(payload.get("name") == "data.parquet" and _hex(payload.get("sha256"))
            and type(payload.get("bytes")) is int and 0 < payload["bytes"] <= MAX_EXPORT_BYTES,
            "Development export payload identity differs")
    return {"commit_sha256": export_reference["commit_sha256"], "source_partition": expected_source,
            "source_manifest_sha256": USGS_MANIFEST_SHA256, "month": MONTH,
            "start_inclusive_epoch": MONTH_START, "cutoff_exclusive_epoch": CUTOFF,
            "payload": payload, "exported_rows": commit["exported_rows"], "plan_sha256": commit["plan_sha256"],
            "physical_decode_scope": DECODE_SCOPE, "protected_scoring_permitted": False}


def read_development_export(export_reference, source_partition):
    """Read only an admitted export, checking its complete time axis before Q."""
    identity = inspect_development_export(export_reference, source_partition)
    path = Path(export_reference["directory"]) / "data.parquet"
    raw = verified_local_bytes(path, identity["payload"]["sha256"], expected_bytes=identity["payload"]["bytes"],
                               max_bytes=MAX_EXPORT_BYTES)
    from pyarrow import BufferReader
    import pyarrow.parquet as pq
    coordinates = pq.read_table(BufferReader(raw), columns=["site", "sample_epoch", "sample_time_utc", "is_core"]).to_pylist()
    require(len(coordinates) == identity["exported_rows"], "Development export row count differs")
    _validate_development_coordinates(coordinates)
    rows = pq.read_table(BufferReader(raw), columns=list(USGS_COLUMNS)).to_pylist()
    require(len(rows) == len(coordinates), "Development export data/coordinate counts differ")
    return rows, identity
