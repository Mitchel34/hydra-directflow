"""Resumable development corpus staging through caller-owned source transport.

This module performs no network calls or raw-file deletion. A source provider is
called only on a verified basin-cache miss. A post-commit callback may finalize
an explicitly owned raw stage after the derived cache has been verified. Its
transport budget, download authorization and ownership checks remain external.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
import uuid
import zipfile

import numpy as np

from .cache import (CachedDevelopmentSession, _read, _sync_directory, _write_durable,
                    implementation_identity, pack_cycle, unpack_cycle)
from .contract import require, DevelopmentProtocol
from .eligibility import read_required_index
from .local_data import (GAUGES, USGS_MANIFEST_SHA256, WEIGHTS_SHA256, DEVELOPMENT_QUALIFICATION,
                         basin_reduce, bounded_npy, iso, utc_epoch, load_candidate_weights, verified_local_bytes)
from .monthly import cycle_ids_for_origins
from .multicycle import INDEX_SHA256, load_authenticated_cycle, _months_for_windows, assemble_cases
from .quality import PRIMARY_APPROVED
from .reference import ASSESSMENT_START, digest
from .telemetry import canonical

HOUR = 3600
ACQUISITION_START = utc_epoch("2018-07-12T00:00:00Z")
SCHEMA = "hydra-development-corpus/v1"
DEFAULT_SPLITS = (
    {"name": "candidate_training", "start_utc": "2018-07-12T00:00:00Z", "end_utc": "2023-07-31T00:00:00Z"},
    {"name": "candidate_validation", "start_utc": "2023-07-31T00:00:00Z", "end_utc": "2024-07-31T00:00:00Z"},
)
SHARD_KEYS = ("history", "history_mask", "weather", "weather_mask", "state", "state_mask", "target", "target_mask",
              "site_id", "origin_time", "history_time", "target_time", "case_id", "source_cycle", "soil_source_cycle",
              "weather_source_leads", "eligibility_reason")
MAX_SHARD_BYTES = 16 * 1024**2


_INDEX_FACTORY_TOKEN = object()


class AuthenticatedIndexSnapshot:
    """Factory-only snapshot of the exact sealed index, authenticated once."""
    def __init__(self, path, rows, *, _token=None):
        require(_token is _INDEX_FACTORY_TOKEN, "Use AuthenticatedIndexSnapshot.from_file; arbitrary row maps are not admitted")
        self._path = str(Path(path).resolve())
        self._rows = deepcopy(rows)

    @classmethod
    def from_file(cls, path, cycle_ids):
        rows = read_required_index(path, cycle_ids)
        require(set(rows) == set(cycle_ids), "Interior acquired calendar identities missing; transport cannot repair index metadata")
        return cls(path, rows, _token=_INDEX_FACTORY_TOKEN)

    def subset(self, path, cycle_ids):
        require(str(Path(path).resolve()) == self._path and set(cycle_ids) <= set(self._rows),
                "Source session differs from the authenticated index snapshot")
        return {cycle: deepcopy(self._rows[cycle]) for cycle in cycle_ids}


class SourceUnavailable(RuntimeError):
    """Transport/admission failures are distinct from authenticated upstream gaps."""
    def __init__(self, cycle_id, status, error_type=None):
        self.cycle_id, self.status, self.error_type = cycle_id, status, error_type
        super().__init__(status + ": " + cycle_id)


def _epoch_cycle(cycle_id):
    return int(datetime.strptime(cycle_id, "%Y%m%d_%HZ").replace(tzinfo=timezone.utc).timestamp())


def _origins(chunk):
    first = utc_epoch(chunk["first_origin_utc"])
    return [first + i * HOUR for i in range(chunk["origin_count"])]


def partition_split(split, *, history_hours=168, chunk_hours=120):
    """Split only at calendar/storage boundaries; purge only scientific ends."""
    require(history_hours in (24, 168) and type(chunk_hours) is int and 1 <= chunk_hours <= 128,
            "Explicit bounded L24/L168 and chunk geometry required")
    start, end = utc_epoch(split["start_utc"]), utc_epoch(split["end_utc"])
    require(ACQUISITION_START <= start < end <= ASSESSMENT_START and start % HOUR == end % HOUR == 0,
            "Exact pre-assessment development split required")
    require(re.fullmatch(r"[a-zA-Z0-9_]+", split["name"]), "Portable split name required")
    cursor = datetime.fromtimestamp(start, timezone.utc).replace(day=1, hour=0)
    units = []
    while int(cursor.timestamp()) < end:
        following = cursor.replace(year=cursor.year + 1, month=1) if cursor.month == 12 else cursor.replace(month=cursor.month + 1)
        first, after = max(start, int(cursor.timestamp())), min(end, int(following.timestamp()))
        candidate_after = max(first, min(after, end - 18 * HOUR))
        origins = list(range(first, candidate_after, HOUR))
        month = cursor.strftime("%Y-%m")
        unit = dict(id=f'{split["name"]}__{month}__L{history_hours}', split=split["name"], month=month,
                    history_hours=history_hours, requested_origins=(after - first) // HOUR,
                    candidate_origins=len(origins), purged_origins=(after - candidate_after) // HOUR,
                    split_end_exclusive_utc=iso(end), chunks=[])
        for index, offset in enumerate(range(0, len(origins), chunk_hours)):
            batch = origins[offset:offset + chunk_hours]
            candidates = cycle_ids_for_origins(batch)
            require(len(candidates) <= 32, "Chunk exceeds authenticated cycle-reader bound")
            outside = [cycle for cycle in candidates if _epoch_cycle(cycle) < ACQUISITION_START]
            unit["chunks"].append(dict(index=index, first_origin_utc=iso(batch[0]), last_origin_utc=iso(batch[-1]),
                origin_count=len(batch), cycle_ids=[cycle for cycle in candidates if cycle not in outside],
                calendar_unavailable_cycle_ids=outside))
        unit["required_usgs_months"] = sorted(_months_for_windows(
            [(origins[0] - (history_hours + 1) * HOUR, origins[-1] + 18 * HOUR)],
            development_export_months=("2024-07",))) if origins else []
        units.append(unit)
        cursor = following
    return units


def plan_corpus(*, history_hours=168, chunk_hours=120):
    """Pure plan for the exact existing candidate split dates; no source reads."""
    units = [unit for split in DEFAULT_SPLITS for unit in partition_split(split, history_hours=history_hours, chunk_hours=chunk_hours)]
    cycles = sorted({cycle for unit in units for chunk in unit["chunks"] for cycle in chunk["cycle_ids"]})
    outside = sorted({cycle for unit in units for chunk in unit["chunks"] for cycle in chunk["calendar_unavailable_cycle_ids"]})
    code = implementation_identity()
    code["corpus.py"] = hashlib.sha256(_read(Path(__file__), 256 * 1024)).hexdigest()
    code["monthly.py"] = hashlib.sha256(_read(Path(__file__).with_name("monthly.py"), 256 * 1024)).hexdigest()
    code["eligibility.py"] = hashlib.sha256(_read(Path(__file__).with_name("eligibility.py"), 256 * 1024)).hexdigest()
    identity = dict(schema=SCHEMA, splits=list(DEFAULT_SPLITS), history_hours=history_hours, chunk_hours=chunk_hours,
                    input_quality_policy=PRIMARY_APPROVED, target_quality_policy=PRIMARY_APPROVED,
                    qualified_index_sha256=INDEX_SHA256, usgs_manifest_sha256=USGS_MANIFEST_SHA256,
                    candidate_weights_sha256=WEIGHTS_SHA256, units=units, required_cycle_ids=cycles,
                    calendar_unavailable_cycle_ids=outside, implementation_sha256=code)
    return dict(**identity, plan_sha256=digest(identity), status="PLANNED_NOT_EXECUTED",
                required_usgs_months=sorted({month for unit in units for month in unit["required_usgs_months"]}),
                requires_july2024_development_export=True,
                requested_origins=sum(unit["requested_origins"] for unit in units),
                candidate_origins=sum(unit["candidate_origins"] for unit in units),
                purged_origins=sum(unit["purged_origins"] for unit in units),
                qualification="Candidate split cache preparation; no frozen model selection or scientific training")


class SourceBackedSession(CachedDevelopmentSession):
    """Reuse ordinary cache identities; transport is lazy and owned by the caller."""
    def __init__(self, *, source_provider, on_cache_committed=None, guard=None, authenticated_index=None, **kwargs):
        super().__init__(**kwargs)
        self.source_provider, self.on_cache_committed = source_provider, on_cache_committed
        self.guard = guard or (lambda: None)
        require(authenticated_index is None or type(authenticated_index) is AuthenticatedIndexSnapshot,
                "Only a factory-authenticated index snapshot can be reused")
        self.authenticated_index = authenticated_index
        self.provider_calls, self.finalization_calls = 0, 0

    def __enter__(self):
        if self.authenticated_index is None:
            return super().__enter__()
        started = time.monotonic()
        self._lock = self.cache.locked()
        self._lock.__enter__()
        try:
            self.code = implementation_identity()
            self.index = self.authenticated_index.subset(self.index_path, list(self.specs))
            self.weights = load_candidate_weights(self.weights_path, qualification=DEVELOPMENT_QUALIFICATION)
            manifest = json.loads(verified_local_bytes(self.manifest_path, USGS_MANIFEST_SHA256, max_bytes=2 * 1024**2))
            self.partitions = {}
            for entry in manifest["native_partitions"]:
                month = Path(entry["path"]).stem
                require(month not in self.partitions, "Duplicate source month in manifest")
                self.partitions[month] = {"sha256": entry["sha256"], "bytes": entry["bytes"]}
            self.policy = {"geometry": DEVELOPMENT_QUALIFICATION, "quality": "PRESERVE_ALL_NATIVE_FLAGS_ADMISSION_AT_CASE_ASSEMBLY",
                           "units": "PHYSICAL_UNSCALED", "availability": "UNKNOWN_HISTORICAL_PUBLICATION_SIMULATED_LAG_2H"}
            self.opened = True
            self.initialization_seconds = time.monotonic() - started
            return self
        except BaseException:
            self._lock.__exit__(*__import__("sys").exc_info())
            raise

    def _cycle(self, cycle_id):
        self.guard()
        if cycle_id in self.cycles:
            self.cache.counts["memory_hits"] += 1
            return self.cycles[cycle_id]
        identity = dict(cycle_id=cycle_id, qualified_index_sha256=INDEX_SHA256,
                        sealed_row_sha256=digest(self.index[cycle_id]), candidate_weights_sha256=WEIGHTS_SHA256,
                        implementation_sha256=self.code, policy=self.policy)
        lease = {}
        def build():
            if not callable(self.source_provider):
                raise SourceUnavailable(cycle_id, "SOURCE_PROVIDER_REQUIRED")
            self.guard()
            self.provider_calls += 1
            try:
                supplied = self.source_provider(cycle_id, deepcopy(self.index[cycle_id]))
            except Exception as error:
                raise SourceUnavailable(cycle_id, "SOURCE_TRANSPORT_FAILED", type(error).__name__) from error
            if not isinstance(supplied, dict) or supplied.get("status") != "READY":
                raise SourceUnavailable(cycle_id, "SOURCE_TRANSPORT_NOT_READY")
            spec, ownership = supplied.get("spec"), supplied.get("ownership")
            try:
                require(isinstance(spec, dict) and spec.get("cycle_id") == cycle_id and spec.get("archive_path"),
                        "Source provider cycle/spec differs")
                require(isinstance(ownership, dict) and type(ownership.get("owned_archive")) is bool
                        and isinstance(ownership.get("owner_run_id"), str) and ownership["owner_run_id"]
                        and Path(ownership["archive_path"]).resolve() == Path(spec["archive_path"]).resolve(),
                        "Explicit matching source-stage ownership required")
            except Exception as error:
                raise SourceUnavailable(cycle_id, "INVALID_SOURCE_PROVIDER_RESULT", type(error).__name__) from error
            try:
                cycle, receipt = load_authenticated_cycle(spec, self.index[cycle_id])
                result = pack_cycle(cycle, basin_reduce(cycle.values, cycle.present, self.weights), receipt)
            except Exception as error:
                raise SourceUnavailable(cycle_id, "SOURCE_ADMISSION_FAILED", type(error).__name__) from error
            lease.update(supplied)
            return result
        arrays, metadata, key, reuse = self.cache.get_or_build("basin_cycle", identity, build, validator=unpack_cycle)
        cycle, reduced = unpack_cycle(arrays, metadata)
        result = cycle, reduced, metadata["source_receipt"], key
        self.cycles[cycle_id] = result
        if reuse == "built" and self.on_cache_committed is not None:
            require(callable(self.on_cache_committed), "Post-commit finalization must be callable")
            committed = _read(self.cache.path / "entries" / key / "COMMIT.json", 128 * 1024)
            receipt = {**metadata["source_receipt"], "cache_key": key,
                       "cache_commit_sha256": hashlib.sha256(committed).hexdigest(), "cache_payload_verified": True}
            try:
                self.on_cache_committed(receipt, lease)
                self.finalization_calls += 1
            except Exception as error:
                raise SourceUnavailable(cycle_id, "OWNED_STAGE_FINALIZATION_FAILED_AFTER_CACHE_COMMIT", type(error).__name__) from error
        self.guard()
        return result

    def _month(self, month):
        self.guard()
        try:
            result = super()._month(month)
        except Exception as error:
            raise SourceUnavailable("USGS:" + month, "DISCHARGE_SOURCE_OR_EXPORT_ADMISSION_FAILED", type(error).__name__) from error
        self.guard()
        return result


def build_corpus_chunk(session, unit, chunk):
    origins = _origins(chunk)
    if chunk["cycle_ids"]:
        return session.build(origins_utc=[iso(value) for value in origins], history_hours=unit["history_hours"],
            archive_cycle_ids=chunk["cycle_ids"], input_quality_policy=PRIMARY_APPROVED, target_quality_policy=PRIMARY_APPROVED)
    # Very small initial chunks can precede every available acquisition cycle.
    # They still retain observations and explicit unresolved source coverage.
    required = cycle_ids_for_origins(origins)
    require(required == chunk["calendar_unavailable_cycle_ids"] and
            all(_epoch_cycle(cycle) < ACQUISITION_START for cycle in required),
            "Empty source inventory is permitted only for pre-acquisition warmup")
    started = time.monotonic()
    windows = [(origin - (unit["history_hours"] + 1) * HOUR, origin + 18 * HOUR) for origin in origins]
    observations, month_keys = [], {}
    for month in sorted(_months_for_windows(windows, development_export_months=session.exports)):
        rows, _, key = session._month(month)
        observations.extend(row for row in rows if any(start <= row.valid_at <= end for start, end in windows))
        month_keys[month] = key
    arrays = assemble_cases({}, {}, observations, origins, DevelopmentProtocol(history_hours=unit["history_hours"]),
                            input_quality_policy=PRIMARY_APPROVED, target_quality_policy=PRIMARY_APPROVED)
    return arrays, dict(cache={"cycle_keys": {}, "month_keys": month_keys},
        case_source_digest=digest({"case_ids": arrays["case_id"].tolist(), "month_keys": month_keys,
                                   "unavailable_calendar_cycles": required, "quality": PRIMARY_APPROVED}),
        stage_seconds={"total": time.monotonic() - started}, source_calendar_scope="PRE_ACQUISITION_WARMUP")


def compact_shard(arrays, chunk):
    """Keep numeric inputs and case coordinates; shared stores retain provenance."""
    selected = {name: np.asarray(arrays[name]) for name in SHARD_KEYS}
    count = len(selected["case_id"])
    require(count == 3 * chunk["origin_count"], "Every requested gauge case must survive assembly")
    missing = [_epoch_cycle(cycle) for cycle in chunk["calendar_unavailable_cycle_ids"]]
    calendar_status = []
    for i, origin in enumerate(selected["origin_time"]):
        issues = []
        for field, max_age, name in (("source_cycle", 18, "WEATHER"), ("soil_source_cycle", 8, "SOIL")):
            relevant = [reference for reference in missing if 2 * HOUR <= int(origin) - reference <= max_age * HOUR]
            if relevant and not str(selected[field][i]):
                issues.append(name)
        calendar_status.append("PRE_ACQUISITION_WARMUP_UNRESOLVED_" + "_AND_".join(issues) if issues else "RESOLVED_WITHIN_ACQUIRED_CALENDAR")
    selected["source_calendar_status"] = np.array(calendar_status)
    selected["admission_mask"] = (selected["eligibility_reason"] == "COMPLETE") & (selected["source_calendar_status"] == "RESOLVED_WITHIN_ACQUIRED_CALENDAR")
    require(set(selected) == set(SHARD_KEYS) | {"source_calendar_status", "admission_mask"}, "Unexpected model shard fields")
    require(all(value.dtype.kind in "fbiuUS" for value in selected.values()), "No object or JSON dictionaries in model shards")
    return selected


def _validate_shard(model, unit, chunk):
    require(set(model) == set(SHARD_KEYS) | {"admission_mask", "source_calendar_status"}, "Corpus model fields missing")
    n, length = 3 * chunk["origin_count"], unit["history_hours"]
    expected = np.repeat(np.array(_origins(chunk), np.int64), 3)
    require(np.array_equal(model["origin_time"], expected)
            and np.array_equal(model["site_id"], np.tile(np.arange(3), chunk["origin_count"])),
            "Corpus gauge/origin order differs")
    require(model["case_id"].tolist() == [site + ":" + iso(int(origin)) for origin in _origins(chunk) for site in GAUGES],
            "Corpus case identities differ")
    require(np.array_equal(model["history_time"], expected[:, None] - np.arange(length + 1, 1, -1) * HOUR)
            and np.array_equal(model["target_time"], expected[:, None] + np.arange(1, 19) * HOUR)
            and np.all(model["target_time"] < utc_epoch(unit["split_end_exclusive_utc"])),
            "Corpus history/targets differ or cross the scientific split")
    complete = np.ones(n, dtype=bool)
    for name, shape in (("history", (n, length, 2)), ("weather", (n, 18, 4)), ("state", (n, 2)), ("target", (n, 18))):
        value, mask = model[name], model[name + "_mask"]
        require(value.shape == mask.shape == shape and value.dtype.kind == "f" and mask.dtype == np.dtype(bool),
                "Corpus value/mask shape or type differs")
        require(np.isfinite(value[mask]).all() and np.isnan(value[~mask]).all(), "Physical missing values/masks differ")
        complete &= mask.reshape(n, -1).all(axis=1)
    require(model["eligibility_reason"].shape == (n,) and np.array_equal(model["eligibility_reason"] == "COMPLETE", complete),
            "Corpus complete-case reason differs from source masks")
    require(model["weather_source_leads"].shape == (n, 18), "Corpus source lead shape differs")
    for field in ("source_cycle", "soil_source_cycle"):
        require(model[field].shape == (n,) and set(model[field].tolist()) <= set(chunk["cycle_ids"]) | {""},
                "Corpus selected source lies outside authenticated candidates")
    for i, origin in enumerate(expected):
        cycle = str(model["source_cycle"][i])
        if cycle:
            reference = _epoch_cycle(cycle)
            require(2 * HOUR <= origin - reference <= 18 * HOUR
                    and model["weather_mask"][i].all()
                    and np.array_equal(model["weather_source_leads"][i], (model["target_time"][i] - reference) // HOUR),
                    "Corpus weather selection/lead clocks differ")
        else:
            require(not model["weather_mask"][i].any() and (model["weather_source_leads"][i] == -1).all(),
                    "Missing weather run cannot acquire observed values")
    expected_model = compact_shard(model, chunk)
    require(model["admission_mask"].dtype == np.dtype(bool)
            and np.array_equal(model["admission_mask"], expected_model["admission_mask"])
            and np.array_equal(model["source_calendar_status"], expected_model["source_calendar_status"]),
            "Corpus calendar coverage/admission differs")


def _shard_write(root, plan_sha, unit, chunk, arrays, receipt):
    target = root / "chunks" / unit["id"] / f'{chunk["index"]:03d}'
    require(not target.exists() and not target.is_symlink(), "Corpus chunks are immutable")
    target.parent.mkdir(exist_ok=True)
    stage = root / "staging" / uuid.uuid4().hex
    stage.mkdir()
    model = compact_shard(arrays, chunk)
    _validate_shard(model, unit, chunk)
    stream = io.BytesIO()
    np.savez_compressed(stream, **model)
    payload = stream.getvalue()
    require(len(payload) <= MAX_SHARD_BYTES and sum(value.nbytes for value in model.values()) <= MAX_SHARD_BYTES,
            "Typed model shard exceeds bound")
    summary = dict(unit_id=unit["id"], chunk_index=chunk["index"], first_origin_utc=chunk["first_origin_utc"],
                   last_origin_utc=chunk["last_origin_utc"], origin_count=chunk["origin_count"], case_count=len(model["case_id"]),
                   admitted_cases=int(model["admission_mask"].sum()), eligibility_counts=dict(Counter(model["eligibility_reason"].tolist())),
                   source_calendar_counts=dict(Counter(model["source_calendar_status"].tolist())),
                   calendar_unavailable_cycle_ids=chunk["calendar_unavailable_cycle_ids"],
                   input_quality_policy=PRIMARY_APPROVED, target_quality_policy=PRIMARY_APPROVED,
                   case_source_digest=receipt["case_source_digest"], cycle_keys=receipt["cache"]["cycle_keys"],
                   month_keys=receipt["cache"]["month_keys"], stage_seconds=receipt["stage_seconds"],
                   provenance="Original per-cycle records/fields and hourly qualifiers/IDs remain in referenced shared stores")
    summary_raw = canonical(summary)
    _write_durable(stage / "model.npz", payload)
    _write_durable(stage / "case-index.json", summary_raw)
    commit = dict(schema=SCHEMA, plan_sha256=plan_sha, unit_id=unit["id"], chunk_index=chunk["index"],
                  payloads={"model.npz": {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)},
                            "case-index.json": {"sha256": hashlib.sha256(summary_raw).hexdigest(), "bytes": len(summary_raw)}},
                  arrays={name: {"shape": list(value.shape), "dtype": value.dtype.str} for name, value in model.items()})
    _write_durable(stage / "COMMIT.json", canonical(commit))
    _sync_directory(stage)
    stage.rename(target)
    _sync_directory(target.parent)
    _sync_directory(root / "chunks")
    return _shard_read(target, plan_sha, unit, chunk)


def _shard_read(path, plan_sha, unit, chunk):
    require(not path.is_symlink(), "Corpus chunk cannot be a symlink")
    commit = json.loads(_read(path / "COMMIT.json", 65536))
    require(commit["schema"] == SCHEMA and commit["plan_sha256"] == plan_sha and commit["unit_id"] == unit["id"]
            and commit["chunk_index"] == chunk["index"], "Corpus chunk plan identity differs")
    require(set(commit["payloads"]) == {"model.npz", "case-index.json"}
            and {item.name for item in path.iterdir()} == {"COMMIT.json", "model.npz", "case-index.json"}, "Corpus chunk inventory differs")
    blobs = {}
    for name, maximum in (("model.npz", MAX_SHARD_BYTES), ("case-index.json", 256 * 1024)):
        value = _read(path / name, maximum)
        require(len(value) == commit["payloads"][name]["bytes"] and hashlib.sha256(value).hexdigest() == commit["payloads"][name]["sha256"],
                "Corpus chunk payload SHA256 differs")
        blobs[name] = value
    summary = json.loads(blobs["case-index.json"])
    require(summary["unit_id"] == unit["id"] and summary["chunk_index"] == chunk["index"]
            and summary["origin_count"] == chunk["origin_count"] and summary["case_count"] == 3 * chunk["origin_count"]
            and summary["first_origin_utc"] == chunk["first_origin_utc"] and summary["last_origin_utc"] == chunk["last_origin_utc"]
            and summary["calendar_unavailable_cycle_ids"] == chunk["calendar_unavailable_cycle_ids"]
            and set(summary["cycle_keys"]) == set(chunk["cycle_ids"])
            and summary["input_quality_policy"] == summary["target_quality_policy"] == PRIMARY_APPROVED,
            "Corpus case index differs from planned coordinates/policies")
    with zipfile.ZipFile(io.BytesIO(blobs["model.npz"])) as archive:
        entries = archive.infolist()
        require(len(entries) == len(commit["arrays"]) and len({entry.filename for entry in entries}) == len(entries)
                and {entry.filename for entry in entries} == {name + ".npy" for name in commit["arrays"]}
                and sum(entry.file_size for entry in entries) <= MAX_SHARD_BYTES, "Corpus NPZ expansion inventory differs")
        arrays = {}
        for name, geometry in commit["arrays"].items():
            require(name in set(SHARD_KEYS) | {"admission_mask", "source_calendar_status"}, "Unapproved corpus array")
            arrays[name] = bounded_npy(archive.read(name + ".npy"), shape=tuple(geometry["shape"]), dtype=geometry["dtype"])
    _validate_shard(arrays, unit, chunk)
    require(int(arrays["admission_mask"].sum()) == summary["admitted_cases"]
            and dict(Counter(arrays["eligibility_reason"].tolist())) == summary["eligibility_counts"]
            and dict(Counter(arrays["source_calendar_status"].tolist())) == summary["source_calendar_counts"],
            "Corpus case/admission accounting differs")
    return summary


def run_corpus(*, qualified_index_path, weights_path, usgs_paths, usgs_manifest_path, cache_directory,
               run_directory, source_provider=None, on_cache_committed=None, history_hours=168, chunk_hours=120,
               usgs_development_exports=None, execution_enabled=False, approved_plan_sha256=None,
               max_wall_seconds=900, max_rss_bytes=4 * 1024**3, stop_after_chunks=None):
    """Resume immutable monthly shards; stop rather than invent failed sources.

    Limits are cooperative, including before/after each provider call. The
    caller must separately supervise and authorize any transport callback.
    """
    plan = plan_corpus(history_hours=history_hours, chunk_hours=chunk_hours)
    if not execution_enabled:
        return plan
    require(type(execution_enabled) is bool and approved_plan_sha256 == plan["plan_sha256"], "Exact reviewed corpus plan required")
    require(type(max_wall_seconds) is int and 1 <= max_wall_seconds <= 1800
            and type(max_rss_bytes) is int and 0 < max_rss_bytes <= 4 * 1024**3, "Bounded corpus resources required")
    require(stop_after_chunks is None or type(stop_after_chunks) is int and stop_after_chunks > 0, "Positive chunk stop required")
    checkout = Path(__file__).resolve().parents[2]
    root, cache = Path(run_directory).absolute(), Path(cache_directory).absolute()
    require(any(root.resolve().is_relative_to((checkout / name).resolve()) for name in ("runs", ".local"))
            and any(cache.resolve().is_relative_to((checkout / name).resolve()) for name in (".cache", ".local")),
            "Corpus run/cache paths must remain private")
    snapshot = AuthenticatedIndexSnapshot.from_file(qualified_index_path, plan["required_cycle_ids"])
    root.mkdir(parents=True, exist_ok=True)
    require(not root.is_symlink(), "Corpus run must be a real directory")
    descriptor = os.open(root / ".writer.lock", os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    started, built, reused, peak = time.monotonic(), 0, 0, 0
    payloads, failure, provider_calls, finalization_calls = [], None, 0, 0
    admitted_run, locked = False, False
    import psutil
    process = psutil.Process()
    def guard():
        nonlocal peak
        peak = max(peak, process.memory_info().rss)
        require(peak <= max_rss_bytes and time.monotonic() - started < max_wall_seconds, "Corpus resource bound reached")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        manifest = root / "PLAN.json"
        if manifest.exists():
            require(json.loads(_read(manifest, 4 * 1024**2))["plan_sha256"] == plan["plan_sha256"], "Corpus directory belongs to another plan")
        else:
            require({p.name for p in root.iterdir()} <= {".writer.lock"}, "Refusing unrelated corpus directory")
            _write_durable(manifest, canonical(plan))
        for name in ("chunks", "staging"):
            path = root / name; path.mkdir(exist_ok=True)
            require(not path.is_symlink(), "Corpus stage must be a real directory")
        _sync_directory(root)
        admitted_run = True
        stop = False
        for unit in plan["units"]:
            if not unit["chunks"]:
                continue
            guard()
            cycles = sorted({cycle for chunk in unit["chunks"] for cycle in chunk["cycle_ids"]})
            specs = [{"cycle_id": cycle, "archive_path": "SOURCE_PROVIDER_NOT_ENTERED"} for cycle in cycles]
            with SourceBackedSession(archive_specs=specs, qualified_index_path=qualified_index_path, weights_path=weights_path,
                    usgs_paths=usgs_paths, usgs_manifest_path=usgs_manifest_path, cache_directory=cache,
                    usgs_development_exports=usgs_development_exports, source_provider=source_provider,
                    on_cache_committed=on_cache_committed, guard=guard, authenticated_index=snapshot) as session:
                try:
                    # Fail before paid-provider callbacks if discharge inputs or
                    # the boundary export are unavailable for this month piece.
                    for month in unit["required_usgs_months"]:
                        session._month(month)
                    for chunk in unit["chunks"]:
                        guard()
                        path = root / "chunks" / unit["id"] / f'{chunk["index"]:03d}'
                        if path.exists() or path.is_symlink():
                            payload = _shard_read(path, plan["plan_sha256"], unit, chunk)
                            for cycle, key in payload["cycle_keys"].items():
                                require(session._cycle(cycle)[3] == key, "Dependent basin cache identity differs")
                            for month, key in payload["month_keys"].items():
                                require(session._month(month)[2] == key, "Dependent discharge cache identity differs")
                            reused += 1
                        else:
                            arrays, receipt = build_corpus_chunk(session, unit, chunk)
                            guard()
                            payload = _shard_write(root, plan["plan_sha256"], unit, chunk, arrays, receipt)
                            del arrays
                            built += 1
                        payloads.append(payload)
                        guard()
                        if stop_after_chunks is not None and built >= stop_after_chunks:
                            stop = True; break
                finally:
                    provider_calls += session.provider_calls
                    finalization_calls += session.finalization_calls
            if stop:
                break
    except BaseException as error:
        if not admitted_run:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            raise
        failure = {"error_type": type(error).__name__}
        if isinstance(error, SourceUnavailable):
            failure.update(cycle_id=error.cycle_id, source_status=error.status, cause_type=error.error_type)
    total_chunks = sum(len(unit["chunks"]) for unit in plan["units"])
    complete = failure is None and len(payloads) == total_chunks
    result = dict(schema=SCHEMA, status="COMPLETED_DEVELOPMENT_CORPUS" if complete else "STOPPED_WITH_EXPLICIT_FAILURE" if failure else "STOPPED_AT_CHUNK_BOUNDARY",
                  plan_sha256=plan["plan_sha256"], created_at_utc=datetime.now(timezone.utc).isoformat(), failure=failure,
                  history_hours=history_hours, requested_origins=plan["requested_origins"], candidate_origins=plan["candidate_origins"],
                  purged_origins=plan["purged_origins"], completed_origins=sum(p["origin_count"] for p in payloads),
                  completed_cases=sum(p["case_count"] for p in payloads), admitted_cases=sum(p["admitted_cases"] for p in payloads),
                  chunks_built=built, chunks_reused=reused, total_chunks=total_chunks,
                  source_provider_calls=provider_calls, post_cache_commit_callback_calls=finalization_calls,
                  sealed_index_authentication_passes=1,
                  calendar_unavailable_cycle_ids=plan["calendar_unavailable_cycle_ids"],
                  source_calendar_counts=dict(sum((Counter(p["source_calendar_counts"]) for p in payloads), Counter())),
                  elapsed_seconds=time.monotonic() - started, observed_boundary_peak_rss_bytes=peak,
                  source_transport_budget_owned_by_caller=True, raw_files_deleted_by_corpus=False,
                  optimizer_updates=0, protected_assessment_scoring=False)
    try:
        # This summary is replaceable; immutable source/cache/chunk commitments are not.
        tmp = root / ("RESULT.json.writing-" + uuid.uuid4().hex)
        _write_durable(tmp, canonical(result)); tmp.replace(root / "RESULT.json"); _sync_directory(root)
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    return result
