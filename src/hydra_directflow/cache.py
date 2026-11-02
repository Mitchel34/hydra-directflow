"""Immutable local basin/month stores with atomic commits and resumable reuse.

Cache keys bind source identities, code, and policies. Only committed entries
are reusable; payloads are content addressed and checked on every disk load.
The cache never fetches data or changes source files. Corruption fails closed
unless the caller explicitly requests quarantine-and-rebuild.
"""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import time
import uuid
import zipfile

import numpy as np

from .contract import DevelopmentProtocol
from .local_data import (DEVELOPMENT_QUALIFICATION, GAUGES, USGS_MANIFEST_SHA256, WEIGHTS_SHA256,
                         LocalCycle, basin_reduce, bounded_npy, iso, load_candidate_weights, verified_local_bytes)
from .multicycle import (INDEX_SHA256, INDEX_BYTES, SELECTION_SCOPE, _months_for_windows, assemble_cases,
                         load_authenticated_cycle, load_usgs_windows, qualified_cycle_rows, validate_request)
from .reference import ASSESSMENT_START, Observation, digest, require
from .telemetry import canonical

MAX_OBJECT_BYTES = 16 * 1024**2
MAX_METADATA_BYTES = 4 * 1024**2
SCHEMA = "hydra-derived-cache/v1"


class CacheCorruption(ValueError):
    pass


class CacheBusy(RuntimeError):
    pass


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(path, maximum):
    path = Path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not getattr(before, "st_flags", 0) & 0x40000000,
            "Cache inputs must be resident regular files, never symlinks")
    require(0 < before.st_size <= maximum, "Cache file exceeds bound")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    after = path.lstat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(stamp(before) == stamp(after) and len(data) == before.st_size, "Cache file changed during read")
    return data


def _directory(path, *, create=False):
    path = Path(path)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    info = path.lstat()
    require(stat.S_ISDIR(info.st_mode) and not getattr(info, "st_flags", 0) & 0x40000000,
            "Cache directories must be resident real directories")
    return path


def _write_durable(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def implementation_identity():
    root = Path(__file__).resolve().parent
    names = ("cache.py", "lineage.py", "multicycle.py", "local_data.py", "contract.py", "reference.py", "telemetry.py")
    return {name: _sha(_read(root / name, 256 * 1024)) for name in names}


class DerivedCache:
    """One writer lock protects immutable entries and explicit repair.

    Uncommitted staging directories are retained as crash evidence and ignored
    on resume. Complete entries are never edited in place or silently replaced.
    """
    def __init__(self, directory, *, repair_corrupt=False):
        require(type(repair_corrupt) is bool, "Repair must be an explicit boolean")
        self.path = Path(directory).absolute()
        self.repair_corrupt = repair_corrupt
        self._held = False
        self.counts = Counter(hits=0, misses=0, repairs=0, memory_hits=0, bytes_read=0, bytes_written=0)

    @contextmanager
    def locked(self):
        require(not self._held, "Cache lock cannot be nested")
        _directory(self.path, create=True)
        lock_path = self.path / ".writer.lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            require(stat.S_ISREG(os.fstat(descriptor).st_mode), "Cache lock must be a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise CacheBusy("Another process owns the cache writer lock") from error
            marker = self.path / "CACHE_ROOT.json"
            if marker.exists():
                require(json.loads(_read(marker, 4096)) == {"schema": SCHEMA}, "Unknown cache root format")
            else:
                require(set(p.name for p in self.path.iterdir()) <= {".writer.lock"}, "Refusing to adopt an unrelated nonempty directory")
                _write_durable(marker, canonical({"schema": SCHEMA}))
            for name in ("entries", "staging", "quarantine"):
                _directory(self.path / name, create=True)
            self._held = True
            yield self
        finally:
            self._held = False
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def key(kind, identity):
        require(kind in ("basin_cycle", "usgs_month"), "Unknown derived cache kind")
        return digest({"schema": SCHEMA, "kind": kind, "identity": identity})

    def _load(self, directory, kind, identity, key):
        try:
            _directory(directory)
            commit_raw = _read(directory / "COMMIT.json", 128 * 1024)
            commit = json.loads(commit_raw)
            require(commit["schema"] == SCHEMA and commit["kind"] == kind and commit["key"] == key
                    and commit["identity"] == identity, "Cache derivation identity differs")
            payloads = commit["payloads"]
            require(set(payloads) == {"arrays", "metadata"}, "Cache payload inventory differs")
            require(set(p.name for p in directory.iterdir()) == {"COMMIT.json", payloads["arrays"]["name"], payloads["metadata"]["name"]},
                    "Unexpected cache entry members")
            raw = {}
            for label, maximum, suffix in (("arrays", MAX_OBJECT_BYTES, ".npz"), ("metadata", MAX_METADATA_BYTES, ".json")):
                item = payloads[label]
                require(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) and item["name"] == item["sha256"] + suffix,
                        "Cache content-addressed filename differs")
                data = _read(directory / item["name"], maximum)
                require(len(data) == item["bytes"] and _sha(data) == item["sha256"], "Cache payload SHA256 differs")
                raw[label] = data
            require(isinstance(commit["arrays"], dict) and 0 < len(commit["arrays"]) <= 16, "Cache array inventory bound exceeded")
            arrays = {}
            with zipfile.ZipFile(io.BytesIO(raw["arrays"])) as archive:
                members = archive.infolist()
                require(len(members) == len(commit["arrays"]) and len({m.filename for m in members}) == len(members)
                        and {m.filename for m in members} == {name + ".npy" for name in commit["arrays"]}
                        and sum(m.file_size for m in members) <= MAX_OBJECT_BYTES,
                        "Cache NPZ inventory/expansion bound differs")
                for name, geometry in commit["arrays"].items():
                    require(re.fullmatch(r"[a-z][a-z0-9_]*", name), "Unsafe cache array name")
                    dtype, shape = np.dtype(geometry["dtype"]), tuple(geometry["shape"])
                    require(dtype.kind in "fiub" and len(shape) <= 3 and all(type(x) is int and 0 <= x <= 100000 for x in shape),
                            "Unsupported cache array encoding")
                    arrays[name] = bounded_npy(archive.read(name + ".npy"), shape=shape, dtype=dtype)
            metadata = json.loads(raw["metadata"])
            require(isinstance(metadata, dict), "Cache metadata must be an object")
            self.counts["bytes_read"] += len(commit_raw) + sum(map(len, raw.values()))
            return arrays, metadata
        except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as error:
            raise CacheCorruption("Committed cache entry failed verification: " + key) from error

    def get_or_build(self, kind, identity, builder, validator=None):
        require(self._held, "Cache writer lock is required")
        key = self.key(kind, identity)
        target = self.path / "entries" / key
        if target.exists() or target.is_symlink():
            try:
                arrays, metadata = self._load(target, kind, identity, key)
                if validator is not None:
                    try:
                        validator(arrays, metadata)
                    except (ValueError, KeyError, TypeError) as error:
                        raise CacheCorruption("Cache semantic validation failed: " + key) from error
                self.counts["hits"] += 1
                return arrays, metadata, key, "hit"
            except CacheCorruption:
                if not self.repair_corrupt:
                    raise
                target.rename(self.path / "quarantine" / (key + "-" + uuid.uuid4().hex))
                _sync_directory(self.path / "entries")
                _sync_directory(self.path / "quarantine")
                self.counts["repairs"] += 1
        self.counts["misses"] += 1
        stage = self.path / "staging" / (key + "-" + uuid.uuid4().hex)
        stage.mkdir()
        arrays, metadata = builder()
        if validator is not None:
            validator(arrays, metadata)
        require(isinstance(arrays, dict) and 0 < len(arrays) <= 16 and isinstance(metadata, dict), "Bounded arrays and object metadata required")
        geometry = {}
        total = 0
        for name, value in arrays.items():
            require(re.fullmatch(r"[a-z][a-z0-9_]*", name) and isinstance(value, np.ndarray)
                    and value.dtype.kind in "fiub" and value.ndim <= 3, "Unsupported derived array")
            total += value.nbytes
            geometry[name] = {"shape": list(value.shape), "dtype": value.dtype.str}
        require(total <= MAX_OBJECT_BYTES, "Derived array memory bound exceeded")
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        contents = {"arrays": stream.getvalue(), "metadata": canonical(metadata)}
        require(len(contents["arrays"]) <= MAX_OBJECT_BYTES and len(contents["metadata"]) <= MAX_METADATA_BYTES, "Derived payload bound exceeded")
        payloads = {}
        for label, raw in contents.items():
            suffix = ".npz" if label == "arrays" else ".json"
            name = _sha(raw) + suffix
            _write_durable(stage / name, raw)
            payloads[label] = {"name": name, "sha256": _sha(raw), "bytes": len(raw)}
        commit = {"schema": SCHEMA, "kind": kind, "key": key, "identity": identity,
                  "arrays": geometry, "payloads": payloads}
        commit_raw = canonical(commit)
        _write_durable(stage / "COMMIT.json", commit_raw)
        _sync_directory(stage)
        # Same-filesystem rename publishes all payloads and the manifest together.
        stage.rename(target)
        _sync_directory(self.path / "entries")
        self.counts["bytes_written"] += len(commit_raw) + sum(map(len, contents.values()))
        checked_arrays, checked_metadata = self._load(target, kind, identity, key)
        return checked_arrays, checked_metadata, key, "built"


def _month_bounds(month):
    require(isinstance(month, str) and re.fullmatch(r"\d{4}-\d{2}", month), "Explicit month required")
    year, number = map(int, month.split("-"))
    start = int(datetime(year, number, 1, tzinfo=timezone.utc).timestamp())
    following = datetime(year + (number == 12), 1 if number == 12 else number + 1, 1, tzinfo=timezone.utc)
    end = int(following.timestamp()) - 3600
    require(end < ASSESSMENT_START, "Entire cached monthly partition must precede protected assessment")
    return start, end


def pack_month(month, observations, receipt):
    start, end = _month_bounds(month)
    times = np.arange(start, end + 1, 3600, dtype=np.int64)
    shape = (3, len(times))
    values = np.full(shape, np.nan)
    present, valid = np.zeros(shape, bool), np.zeros(shape, bool)
    quality, source = np.full(shape, -1, dtype=np.int64), np.full(shape, -1, dtype=np.int64)
    qualities, sources = [], []
    for row in observations:
        require(row.site in GAUGES and start <= row.valid_at <= end and row.valid_at % 3600 == 0,
                "Monthly observation lies outside its store")
        i, j = GAUGES.index(row.site), (row.valid_at - start) // 3600
        require(not present[i, j] and row.available_at is None, "Duplicate or unexpected observation vintage")
        present[i, j] = True
        if row.value is not None:
            require(np.isfinite(row.value) and row.value >= 0, "Invalid unmasked discharge")
            values[i, j], valid[i, j] = row.value, True
        flags = list(row.quality_flags)
        if flags not in qualities:
            qualities.append(flags)
        if row.source_id not in sources:
            sources.append(row.source_id)
        quality[i, j], source[i, j] = qualities.index(flags), sources.index(row.source_id)
    return dict(times=times, values=values, row_present=present, numeric_valid=valid, quality_index=quality, source_index=source), \
           dict(month=month, qualities=qualities, sources=sources, source_receipt=receipt)


def unpack_month(arrays, metadata):
    start, end = _month_bounds(metadata["month"])
    expected_times = np.arange(start, end + 1, 3600, dtype=np.int64)
    require(set(arrays) == {"times", "values", "row_present", "numeric_valid", "quality_index", "source_index"}, "Hourly store arrays differ")
    require(np.array_equal(arrays["times"], expected_times) and arrays["times"].dtype == np.dtype("int64"), "Monthly time axis differs")
    shape = (3, len(expected_times))
    for name in ("values", "row_present", "numeric_valid", "quality_index", "source_index"):
        require(arrays[name].shape == shape, "Monthly store shape differs")
    require(arrays["row_present"].dtype == arrays["numeric_valid"].dtype == np.dtype(bool), "Explicit monthly boolean masks required")
    present, valid = arrays["row_present"], arrays["numeric_valid"]
    require(not (valid & ~present).any() and np.isfinite(arrays["values"][valid]).all()
            and (arrays["values"][valid] >= 0).all() and np.isnan(arrays["values"][~valid]).all(), "Monthly numeric masks/values differ")
    qualities, sources = metadata["qualities"], metadata["sources"]
    require(isinstance(qualities, list) and isinstance(sources, list)
            and all(isinstance(flags, list) and all(isinstance(flag, str) for flag in flags) for flags in qualities)
            and all(isinstance(source, str) and source for source in sources), "Original USGS dictionary metadata differs")
    result = []
    for name, dictionary in (("quality_index", qualities), ("source_index", sources)):
        values = arrays[name]
        require(values.dtype == np.dtype("int64") and (values[~present] == -1).all()
                and (values[present] >= 0).all() and (values[present] < len(dictionary)).all(), "Monthly provenance dictionary index differs")
    for i, j in zip(*np.where(present)):
        result.append(Observation(GAUGES[i], int(expected_times[j]), None,
                                  float(arrays["values"][i, j]) if valid[i, j] else None,
                                  tuple(qualities[arrays["quality_index"][i, j]]), sources[arrays["source_index"][i, j]]))
    return result


def pack_cycle(cycle, reduced, receipt):
    means, masks, coverage = reduced
    return dict(means=means, masks=masks, coverage=coverage), \
           dict(reference=cycle.reference, records=list(cycle.records), member_count=cycle.member_count,
                fields=cycle.fields, availability=cycle.availability, source_receipt=receipt)


def unpack_cycle(arrays, metadata):
    require(set(arrays) == {"means", "masks", "coverage"}, "Basin store arrays differ")
    records = metadata["records"]
    require(isinstance(records, list) and 0 <= len(records) <= 145, "Basin record inventory differs")
    shape = (len(records), 3)
    require(all(value.shape == shape for value in arrays.values()) and arrays["masks"].dtype == np.dtype(bool), "Basin reduction geometry differs")
    require(np.isfinite(arrays["means"][arrays["masks"]]).all() and np.isnan(arrays["means"][~arrays["masks"]]).all(), "Basin mean masks differ")
    require(np.isfinite(arrays["coverage"]).all() and (arrays["coverage"] >= 0).all() and (arrays["coverage"] <= 1 + 1e-10).all(), "Basin coverage differs")
    # Assembly only uses reference/record dictionaries. Full native-cell arrays
    # deliberately do not survive basin reduction in this compact store.
    cycle = LocalCycle(metadata["reference"], np.empty((len(records), 0)), np.empty((len(records), 0), bool),
                       tuple(records), metadata["member_count"], metadata["fields"], metadata["availability"])
    return cycle, (arrays["means"], arrays["masks"], arrays["coverage"])


class CachedDevelopmentSession:
    """Authenticate index once, then assemble bounded chunks from shared stores."""
    def __init__(self, *, archive_specs, qualified_index_path, weights_path, usgs_paths,
                 usgs_manifest_path, cache_directory, repair_corrupt=False):
        require(isinstance(archive_specs, (list, tuple)) and 1 <= len(archive_specs) <= 1024, "One to 1024 local session cycles required")
        self.specs = {item["cycle_id"]: dict(item) for item in archive_specs}
        require(len(self.specs) == len(archive_specs), "Duplicate session cycle")
        self.index_path, self.weights_path = qualified_index_path, weights_path
        self.usgs_paths, self.manifest_path = dict(usgs_paths), usgs_manifest_path
        self.cache = DerivedCache(cache_directory, repair_corrupt=repair_corrupt)
        self.cycles, self.months, self.opened = {}, {}, False

    def __enter__(self):
        started = time.monotonic()
        self._lock = self.cache.locked()
        self._lock.__enter__()
        try:
            self.code = implementation_identity()
            self.index = qualified_cycle_rows(self.index_path, list(self.specs))
            self.weights = load_candidate_weights(self.weights_path, qualification=DEVELOPMENT_QUALIFICATION)
            manifest = json.loads(verified_local_bytes(self.manifest_path, USGS_MANIFEST_SHA256, max_bytes=2 * 1024**2))
            self.partitions = {}
            for entry in manifest["native_partitions"]:
                month = Path(entry["path"]).stem
                require(month not in self.partitions, "Duplicate source month in manifest")
                self.partitions[month] = {"sha256": entry["sha256"], "bytes": entry["bytes"]}
            self.policy = {"geometry": DEVELOPMENT_QUALIFICATION, "quality": "PRESERVE_NATIVE_FLAGS_STRICT_A_AT_CASE_ASSEMBLY",
                           "units": "PHYSICAL_UNSCALED", "availability": "UNKNOWN_HISTORICAL_PUBLICATION_SIMULATED_LAG_2H"}
            self.opened = True
            self.initialization_seconds = time.monotonic() - started
            return self
        except BaseException:
            self._lock.__exit__(*__import__("sys").exc_info())
            raise

    def __exit__(self, exc_type, exc, traceback):
        self.opened = False
        return self._lock.__exit__(exc_type, exc, traceback)

    def _cycle(self, cycle_id):
        if cycle_id in self.cycles:
            self.cache.counts["memory_hits"] += 1
            return self.cycles[cycle_id]
        identity = dict(cycle_id=cycle_id, qualified_index_sha256=INDEX_SHA256,
                        sealed_row_sha256=digest(self.index[cycle_id]), candidate_weights_sha256=WEIGHTS_SHA256,
                        implementation_sha256=self.code, policy=self.policy)
        def build():
            cycle, receipt = load_authenticated_cycle(self.specs[cycle_id], self.index[cycle_id])
            return pack_cycle(cycle, basin_reduce(cycle.values, cycle.present, self.weights), receipt)
        arrays, metadata, key, reuse = self.cache.get_or_build("basin_cycle", identity, build, validator=unpack_cycle)
        cycle, reduced = unpack_cycle(arrays, metadata)
        result = cycle, reduced, metadata["source_receipt"], key
        self.cycles[cycle_id] = result
        return result

    def _month(self, month):
        if month in self.months:
            self.cache.counts["memory_hits"] += 1
            return self.months[month]
        require(month in self.usgs_paths and month in self.partitions, "Required local monthly USGS source missing")
        start, end = _month_bounds(month)
        identity = dict(month=month, source_manifest_sha256=USGS_MANIFEST_SHA256, source_partition=self.partitions[month],
                        implementation_sha256=self.code, policy=self.policy)
        def build():
            observations, receipts = load_usgs_windows(self.usgs_paths, self.manifest_path, [(start, end)])
            require(len(receipts) == 1 and receipts[0]["month"] == month, "Monthly source receipt differs")
            return pack_month(month, observations, receipts[0])
        arrays, metadata, key, reuse = self.cache.get_or_build("usgs_month", identity, build, validator=unpack_month)
        result = unpack_month(arrays, metadata), metadata["source_receipt"], key
        self.months[month] = result
        return result

    def build(self, *, origins_utc, history_hours=24, archive_cycle_ids=None):
        require(self.opened, "Use the session as a context manager")
        selected = list(self.specs) if archive_cycle_ids is None else list(archive_cycle_ids)
        require(all(cycle in self.specs for cycle in selected), "Unknown session cycle selected")
        protocol, origins = validate_request([self.specs[cycle] for cycle in selected], origins_utc, history_hours)
        started, before = time.monotonic(), self.cache.counts.copy()
        cycles, reduced, source_receipts, cycle_keys = {}, {}, [], {}
        for cycle_id in sorted(selected):
            cycle, reduction, receipt, key = self._cycle(cycle_id)
            cycles[cycle_id], reduced[cycle_id] = cycle, reduction
            source_receipts.append(receipt)
            cycle_keys[cycle_id] = key
        cycle_seconds = time.monotonic() - started
        windows = [(origin - (history_hours + 1) * 3600, origin + 18 * 3600) for origin in origins]
        mark = time.monotonic()
        observations, monthly_receipts, month_keys = [], [], {}
        for month in sorted(_months_for_windows(windows)):
            rows, receipt, key = self._month(month)
            observations.extend(row for row in rows if any(start <= row.valid_at <= end for start, end in windows))
            monthly_receipts.append(receipt)
            month_keys[month] = key
        month_seconds = time.monotonic() - mark
        mark = time.monotonic()
        arrays = assemble_cases(cycles, reduced, observations, origins, protocol)
        assembly_seconds = time.monotonic() - mark
        complete = arrays["eligibility_reason"] == "COMPLETE"
        report = dict(schema="hydra-cached-development/v1", status="COMPLETED_WITH_EXPLICIT_MASKS" if not complete.all() else "PASSED_COMPLETE_DEVELOPMENT_CASES",
                      artifact_created_at_utc=iso(int(time.time())), origin_count=len(origins), case_count=len(complete), complete_case_count=int(complete.sum()),
                      history_hours=history_hours, horizons_hours=list(protocol.horizons_hours), gauges=list(GAUGES),
                      origins_utc=[iso(origin) for origin in origins], selection_scope=SELECTION_SCOPE,
                      selected_cycle_case_counts=dict(Counter(arrays["source_cycle"].tolist())),
                      eligibility_counts=dict(Counter(arrays["eligibility_reason"].tolist())),
                      stage_seconds=dict(basin_store=cycle_seconds, discharge_store=month_seconds, case_assembly=assembly_seconds, total=time.monotonic() - started),
                      session_initialization_seconds=self.initialization_seconds,
                      cache=dict(delta={key: self.cache.counts[key] - before[key] for key in self.cache.counts}, cumulative=dict(self.cache.counts), cycle_keys=cycle_keys, month_keys=month_keys,
                                 uncommitted_staging_directories=sum(1 for _ in (self.cache.path / "staging").iterdir())),
                      sources=dict(qualified_index_sha256=INDEX_SHA256, qualified_index_bytes=INDEX_BYTES,
                                   cycles=source_receipts, usgs_manifest_sha256=USGS_MANIFEST_SHA256, usgs_partitions=monthly_receipts,
                                   candidate_weights_sha256=WEIGHTS_SHA256), implementation_sha256=self.code,
                      policy=self.policy, execution=dict(network_requests=0, cloud_jobs=0, optimizer_updates=0,
                                                         protected_assessment_values_decoded=0, source_files_modified=False),
                      qualification="Development only; private immutable derived stores; original candidate geometry and provenance remain unchanged")
        report["case_source_digest"] = digest(dict(case_ids=arrays["case_id"].tolist(), weather_cycles=arrays["source_cycle"].tolist(),
                                                 soil_cycles=arrays["soil_source_cycle"].tolist(), history_hours=history_hours, sources=report["sources"]))
        return arrays, report


def build_cached_development(*, archive_specs, qualified_index_path, weights_path, usgs_paths,
                             usgs_manifest_path, origins_utc, history_hours=24, cache_directory, repair_corrupt=False):
    with CachedDevelopmentSession(archive_specs=archive_specs, qualified_index_path=qualified_index_path, weights_path=weights_path,
                                  usgs_paths=usgs_paths, usgs_manifest_path=usgs_manifest_path, cache_directory=cache_directory,
                                  repair_corrupt=repair_corrupt) as session:
        return session.build(origins_utc=origins_utc, history_hours=history_hours)
