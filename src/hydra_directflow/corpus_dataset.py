"""Read-only snapshots and bounded per-gauge development-corpus views.

Verifies plan/shard bytes, schemas and coordinates. This does not re-audit
upstream source/cache payloads, fetch data, fit transforms/models, authorize
scientific execution, or treat a partial snapshot as a final cohort.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import stat
import zipfile

import numpy as np

from .local_data import GAUGES, USGS_MANIFEST_SHA256, WEIGHTS_SHA256, bounded_npy, iso
from .multicycle import INDEX_SHA256
from .quality import PRIMARY_APPROVED
from .reference import ASSESSMENT_START, digest, require

HOUR = 3600
MIN_ORIGIN = int(datetime(2018, 7, 12, tzinfo=timezone.utc).timestamp())
SCHEMA = "hydra-development-corpus/v1"
MAX_SHARD_BYTES = 16 * 1024**2
PLAN_KEYS = ("schema", "splits", "history_hours", "chunk_hours", "input_quality_policy", "target_quality_policy",
             "qualified_index_sha256", "usgs_manifest_sha256", "candidate_weights_sha256", "units",
             "required_cycle_ids", "calendar_unavailable_cycle_ids", "implementation_sha256")
MODEL_KEYS = ("history", "history_mask", "weather", "weather_mask", "state", "state_mask", "target", "target_mask")
COORD_KEYS = ("site_id", "origin_time", "history_time", "target_time", "case_id", "source_cycle", "soil_source_cycle",
              "weather_source_leads", "eligibility_reason", "source_calendar_status", "admission_mask")
ALL_KEYS = frozenset(MODEL_KEYS + COORD_KEYS)
COHORT_MODES = ("all_candidates", "natural_complete", "common_l168_complete")
FULL_FEATURE_SCHEMA = "FULL_HRRR_H2_W4_S2_V1"
Q_ONLY_FEATURE_SCHEMA = "Q_ONLY_H2_HORIZON1_S0_V1"


class IncompleteCorpusError(ValueError):
    pass


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hex(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read(path, maximum):
    path = Path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not getattr(before, "st_flags", 0) & 0x40000000,
            "Corpus inputs must be resident regular files; no symlinks or hydration")
    require(0 < before.st_size <= maximum, "Corpus input byte bound exceeded")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    after = path.lstat()
    key = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    require(key(before) == key(after) and len(raw) == before.st_size, "Corpus input changed during read")
    return raw


def _directory(path):
    info = Path(path).lstat()
    require(stat.S_ISDIR(info.st_mode) and not getattr(info, "st_flags", 0) & 0x40000000,
            "Corpus directories must be resident real directories")


def _hour(stamp):
    value = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    require(value.tzinfo is not None and value.utcoffset().total_seconds() == 0
            and value.minute == value.second == value.microsecond == 0, "Exact UTC hour required")
    return int(value.timestamp())


def _cycle_epoch(cycle):
    require(isinstance(cycle, str) and re.fullmatch(r"\d{8}_\d{2}Z", cycle), "Explicit source cycle required")
    return int(datetime.strptime(cycle, "%Y%m%d_%HZ").replace(tzinfo=timezone.utc).timestamp())


def _candidate_cycles(first, last):
    step = 6 * HOUR
    lower, upper = ((first - 18 * HOUR + step - 1) // step) * step, ((last - 2 * HOUR) // step) * step
    values = [datetime.fromtimestamp(t, timezone.utc).strftime("%Y%m%d_%HZ") for t in range(lower, upper + 1, step)]
    return ([cycle for cycle in values if _cycle_epoch(cycle) >= MIN_ORIGIN],
            [cycle for cycle in values if _cycle_epoch(cycle) < MIN_ORIGIN])


def _months(start, last):
    cursor = datetime.fromtimestamp(start, timezone.utc).replace(day=1, hour=0)
    result = []
    while int(cursor.timestamp()) <= last:
        result.append(cursor.strftime("%Y-%m"))
        cursor = cursor.replace(year=cursor.year + 1, month=1) if cursor.month == 12 else cursor.replace(month=cursor.month + 1)
    return result


def _plan(root, expected_sha):
    require(_hex(expected_sha), "An externally expected corpus plan SHA256 is required")
    raw = _read(root / "PLAN.json", 4 * 1024**2)
    plan = json.loads(raw)
    require(plan["schema"] == SCHEMA and plan["plan_sha256"] == expected_sha
            and digest({key: plan[key] for key in PLAN_KEYS}) == expected_sha, "Corpus plan SHA256 differs")
    require(plan["qualified_index_sha256"] == INDEX_SHA256 and plan["usgs_manifest_sha256"] == USGS_MANIFEST_SHA256
            and plan["candidate_weights_sha256"] == WEIGHTS_SHA256, "Corpus source authority differs")
    require(plan["input_quality_policy"] == plan["target_quality_policy"] == PRIMARY_APPROVED
            and plan["history_hours"] in (24, 168) and type(plan["chunk_hours"]) is int and 1 <= plan["chunk_hours"] <= 128,
            "Corpus quality or tensor contract differs")
    require(isinstance(plan["implementation_sha256"], dict) and plan["implementation_sha256"]
            and all(_hex(value) for value in plan["implementation_sha256"].values()), "Producer code identities missing")
    splits = {}
    previous_end = MIN_ORIGIN
    for split in plan["splits"]:
        name, start, end = split["name"], _hour(split["start_utc"]), _hour(split["end_utc"])
        require(re.fullmatch(r"[A-Za-z0-9_]+", name) and name not in splits
                and previous_end <= start < end <= ASSESSMENT_START, "Overlapping, unordered or protected corpus split")
        splits[name] = (start, end); previous_end = end
    require(splits, "Explicit corpus splits required")
    descriptors, unit_names, cursors = [], set(), {name: start for name, (start, _) in splits.items()}
    cycles, outside = set(), set()
    requested = candidates = purged = 0
    previous_unit_end = MIN_ORIGIN
    for unit in plan["units"]:
        name, month = unit["split"], unit["month"]
        require(name in splits and re.fullmatch(r"\d{4}-\d{2}", month)
                and unit["id"] == f"{name}__{month}__L{plan['history_hours']}" and unit["id"] not in unit_names
                and unit["history_hours"] == plan["history_hours"], "Invalid or duplicate corpus month unit")
        unit_names.add(unit["id"])
        start, end = splits[name]
        month_start = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
        following = month_start.replace(year=month_start.year + 1, month=1) if month_start.month == 12 else month_start.replace(month=month_start.month + 1)
        first, after = max(start, int(month_start.timestamp())), min(end, int(following.timestamp()))
        eligible_after = max(first, min(after, end - 18 * HOUR))
        require(first < after and first == cursors[name] and first >= previous_unit_end
                and _hour(unit["split_end_exclusive_utc"]) == end,
                "Corpus units are unordered, overlapping or omit a calendar interval")
        cursors[name] = after
        previous_unit_end = after
        require(unit["requested_origins"] == (after - first) // HOUR
                and unit["candidate_origins"] == (eligible_after - first) // HOUR
                and unit["purged_origins"] == (after - eligible_after) // HOUR, "Corpus split-purge accounting differs")
        cursor = first
        for index, chunk in enumerate(unit["chunks"]):
            count = chunk["origin_count"]
            require(chunk["index"] == index and type(count) is int and 1 <= count <= plan["chunk_hours"]
                    and _hour(chunk["first_origin_utc"]) == cursor
                    and _hour(chunk["last_origin_utc"]) == cursor + (count - 1) * HOUR
                    and cursor + count * HOUR <= eligible_after, "Corpus chunk order, overlap or bounds differ")
            expected_cycles, expected_outside = _candidate_cycles(cursor, cursor + (count - 1) * HOUR)
            require(chunk["cycle_ids"] == expected_cycles and chunk["calendar_unavailable_cycle_ids"] == expected_outside,
                    "Corpus candidate source calendar differs")
            cycles.update(expected_cycles); outside.update(expected_outside)
            descriptors.append((unit, chunk))
            cursor += count * HOUR
        require(cursor == eligible_after, "Corpus chunk calendar is incomplete within its plan")
        requested += unit["requested_origins"]; candidates += unit["candidate_origins"]; purged += unit["purged_origins"]
    require(all(cursors[name] == end for name, (_, end) in splits.items()), "Corpus plan omits a split interval")
    require((requested, candidates, purged) == (plan["requested_origins"], plan["candidate_origins"], plan["purged_origins"])
            and sorted(cycles) == plan["required_cycle_ids"] and sorted(outside) == plan["calendar_unavailable_cycle_ids"],
            "Corpus total/calendar accounting differs")
    return plan, descriptors, _sha(raw)


def _shapes(n, length):
    return dict(history=(n, length, 2), history_mask=(n, length, 2), weather=(n, 18, 4), weather_mask=(n, 18, 4),
                state=(n, 2), state_mask=(n, 2), target=(n, 18), target_mask=(n, 18), site_id=(n,), origin_time=(n,),
                history_time=(n, length), target_time=(n, 18), case_id=(n,), source_cycle=(n,), soil_source_cycle=(n,),
                weather_source_leads=(n, 18), eligibility_reason=(n,), source_calendar_status=(n,), admission_mask=(n,))


def _decode(archive, commit, name, shape):
    geometry = commit["arrays"][name]
    require(tuple(geometry["shape"]) == shape, "Corpus numerical schema shape differs: " + name)
    dtype = np.dtype(geometry["dtype"])
    if name in ("history", "weather", "state", "target"):
        require(dtype == np.dtype("<f8"), "Physical source model arrays must retain float64")
    elif name.endswith("_mask"):
        require(dtype == np.dtype(bool), "Explicit Boolean admission/missingness masks required")
    elif name in ("case_id", "source_cycle", "soil_source_cycle", "eligibility_reason", "source_calendar_status"):
        require(dtype.kind == "U" and 0 < dtype.itemsize <= 1024, "Bounded Unicode coordinates required")
    else:
        require(dtype == np.dtype("<i8"), "Exact int64 source coordinates required")
    return bounded_npy(archive.read(name + ".npy"), shape=shape, dtype=dtype)


def _calendar_status(arrays, chunk):
    absent = [_cycle_epoch(cycle) for cycle in chunk["calendar_unavailable_cycle_ids"]]
    result = []
    for i, origin in enumerate(arrays["origin_time"]):
        reasons = []
        for name, max_age, label in (("source_cycle", 18, "WEATHER"), ("soil_source_cycle", 8, "SOIL")):
            if not arrays[name][i] and any(2 * HOUR <= int(origin) - reference <= max_age * HOUR for reference in absent):
                reasons.append(label)
        result.append("PRE_ACQUISITION_WARMUP_UNRESOLVED_" + "_AND_".join(reasons) if reasons else "RESOLVED_WITHIN_ACQUIRED_CALENDAR")
    return np.array(result)


def _complete(arrays, history_hours):
    n = len(arrays["origin_time"])
    result = arrays["history_mask"][:, -history_hours:].reshape(n, -1).all(axis=1)
    for name in ("weather", "state", "target"):
        result &= arrays[name + "_mask"].reshape(n, -1).all(axis=1)
    return result & (arrays["source_calendar_status"] == "RESOLVED_WITHIN_ACQUIRED_CALENDAR")


def _validate_coordinates(arrays, unit, chunk):
    length, count = unit["history_hours"], chunk["origin_count"]
    origins = [_hour(chunk["first_origin_utc"]) + i * HOUR for i in range(count)]
    expected = np.repeat(np.array(origins, np.int64), 3)
    require(np.array_equal(arrays["origin_time"], expected) and np.array_equal(arrays["site_id"], np.tile(np.arange(3), count)),
            "Shard gauge/origin order or uniqueness differs")
    require(arrays["case_id"].tolist() == [site + ":" + iso(origin) for origin in origins for site in GAUGES],
            "Shard case ID order, identity or duplicates differ")
    require(np.array_equal(arrays["history_time"], expected[:, None] - np.arange(length + 1, 1, -1) * HOUR)
            and np.array_equal(arrays["target_time"], expected[:, None] + np.arange(1, 19) * HOUR)
            and np.all(arrays["target_time"] < min(_hour(unit["split_end_exclusive_utc"]), ASSESSMENT_START)),
            "Shard target/history coordinates violate development boundaries")
    allowed = set(chunk["cycle_ids"]) | {""}
    require(set(arrays["source_cycle"].tolist()) <= allowed and set(arrays["soil_source_cycle"].tolist()) <= allowed,
            "Selected source outside the planned candidate inventory")
    for i, origin in enumerate(expected):
        cycle = str(arrays["source_cycle"][i])
        if cycle:
            reference = _cycle_epoch(cycle)
            require(2 * HOUR <= origin - reference <= 18 * HOUR
                    and np.array_equal(arrays["weather_source_leads"][i], (arrays["target_time"][i] - reference) // HOUR),
                    "Selected weather availability/source leads differ")
        else:
            require((arrays["weather_source_leads"][i] == -1).all(), "Absent source cannot have observed forecast leads")
    require(np.array_equal(arrays["source_calendar_status"], _calendar_status(arrays, chunk)), "Source calendar status differs")


def _validate_values(arrays, summary):
    n = len(arrays["origin_time"])
    complete = np.ones(n, bool)
    for name in ("history", "weather", "state", "target"):
        value, mask = arrays[name], arrays[name + "_mask"]
        require(np.isfinite(value[mask]).all() and np.isnan(value[~mask]).all(), "Shard physical value/mask semantics differ")
        complete &= mask.reshape(n, -1).all(axis=1)
    require(np.array_equal(arrays["history_mask"][..., 0], arrays["history_mask"][..., 1])
            and np.array_equal(arrays["state_mask"][:, 0], arrays["state_mask"][:, 1]), "History/state feature masks differ")
    ages = (arrays["origin_time"][:, None] - arrays["history_time"]) / HOUR
    admitted = arrays["history_mask"][..., 1]
    require(np.array_equal(arrays["history"][..., 1][admitted], ages[admitted]), "History age values disagree with coordinates")
    require((arrays["history"][..., 0][arrays["history_mask"][..., 0]] >= 0).all()
            and (arrays["target"][arrays["target_mask"]] >= 0).all(), "Negative admitted discharge")
    for i, origin in enumerate(arrays["origin_time"]):
        if arrays["source_cycle"][i]:
            require(arrays["weather_mask"][i].all(), "Selected weather cycle must supply complete H18 features")
        else:
            require(not arrays["weather_mask"][i].any(), "Missing weather source cannot be admitted")
        if arrays["state_mask"][i].any():
            cycle = str(arrays["soil_source_cycle"][i])
            require(cycle, "Observed soil requires its original source cycle")
            age = (origin - _cycle_epoch(cycle)) / HOUR
            require(2 <= age <= 8 and arrays["state"][i, 1] == age and 0 <= arrays["state"][i, 0] <= 1,
                    "Soil state age/value differs from admission contract")
    require(np.array_equal(arrays["eligibility_reason"] == "COMPLETE", complete)
            and np.array_equal(arrays["admission_mask"], _complete(arrays, arrays["history"].shape[1])),
            "Stored admission differs from original masks and source-calendar support")
    require(int(arrays["admission_mask"].sum()) == summary["admitted_cases"]
            and dict(Counter(arrays["eligibility_reason"].tolist())) == summary["eligibility_counts"]
            and dict(Counter(arrays["source_calendar_status"].tolist())) == summary["source_calendar_counts"],
            "Shard admission accounting differs")


def _load_shard(root, unit, chunk, plan_sha, *, expected_commit_sha=None):
    path = root / "chunks" / unit["id"] / f'{chunk["index"]:03d}'
    _directory(path)
    require({item.name for item in path.iterdir()} == {"COMMIT.json", "case-index.json", "model.npz"}, "Unexpected committed shard members")
    raw = _read(path / "COMMIT.json", 65536)
    commit_sha, commit = _sha(raw), json.loads(raw)
    require(expected_commit_sha is None or commit_sha == expected_commit_sha, "Snapshot shard commitment changed")
    require(commit["schema"] == SCHEMA and commit["plan_sha256"] == plan_sha
            and commit["unit_id"] == unit["id"] and commit["chunk_index"] == chunk["index"], "Shard plan identity differs")
    require(set(commit["payloads"]) == {"case-index.json", "model.npz"} and set(commit["arrays"]) == ALL_KEYS,
            "Shard payload/array inventory differs")
    payloads = {}
    for name, maximum in (("case-index.json", 256 * 1024), ("model.npz", MAX_SHARD_BYTES)):
        data = _read(path / name, maximum)
        identity = commit["payloads"][name]
        require(_hex(identity["sha256"]) and type(identity["bytes"]) is int and len(data) == identity["bytes"]
                and _sha(data) == identity["sha256"], "Committed shard payload hash/size differs")
        payloads[name] = data
    summary = json.loads(payloads["case-index.json"])
    require(summary["unit_id"] == unit["id"] and summary["chunk_index"] == chunk["index"]
            and summary["origin_count"] == chunk["origin_count"] and summary["case_count"] == 3 * chunk["origin_count"]
            and summary["first_origin_utc"] == chunk["first_origin_utc"] and summary["last_origin_utc"] == chunk["last_origin_utc"]
            and summary["calendar_unavailable_cycle_ids"] == chunk["calendar_unavailable_cycle_ids"]
            and summary["input_quality_policy"] == summary["target_quality_policy"] == PRIMARY_APPROVED
            and _hex(summary["case_source_digest"]), "Shard summary or policy differs")
    start, end = _hour(chunk["first_origin_utc"]), _hour(chunk["last_origin_utc"])
    require(set(summary["cycle_keys"]) == set(chunk["cycle_ids"])
            and set(summary["month_keys"]) == set(_months(start - (unit["history_hours"] + 1) * HOUR, end + 18 * HOUR))
            and all(_hex(key) for values in (summary["cycle_keys"], summary["month_keys"]) for key in values.values()),
            "Shard upstream reference inventory differs")
    shapes, arrays = _shapes(summary["case_count"], unit["history_hours"]), {}
    with zipfile.ZipFile(io.BytesIO(payloads["model.npz"])) as archive:
        entries = archive.infolist()
        require(len(entries) == len(ALL_KEYS) and len({entry.filename for entry in entries}) == len(entries)
                and {entry.filename for entry in entries} == {name + ".npy" for name in ALL_KEYS}
                and sum(entry.file_size for entry in entries) <= MAX_SHARD_BYTES, "NPZ inventory/expansion bound differs")
        for name in COORD_KEYS:
            arrays[name] = _decode(archive, commit, name, shapes[name])
        _validate_coordinates(arrays, unit, chunk)  # Before any flow/weather/state value decoding.
        for name in MODEL_KEYS:
            arrays[name] = _decode(archive, commit, name, shapes[name])
    _validate_values(arrays, summary)
    return arrays, summary, commit_sha


@dataclass(frozen=True)
class ShardSnapshot:
    unit_json: str
    chunk_json: str
    commit_sha256: str
    natural24: tuple[bool, ...]
    source_admitted: tuple[bool, ...]

    @property
    def unit(self): return json.loads(self.unit_json)
    @property
    def chunk(self): return json.loads(self.chunk_json)
    @property
    def identifier(self): return self.unit["id"] + "/" + f'{self.chunk["index"]:03d}'


@dataclass(frozen=True)
class DatasetSelection:
    inventory: "CorpusSnapshot"
    gauge_id: str
    split: str
    history_hours: int
    cohort_mode: str
    allow_partial: bool
    selected_rows: tuple[tuple[int, tuple[int, ...]], ...]
    cohort_admission_digest: str
    selection_digest: str

    @property
    def case_count(self): return sum(len(rows) for _, rows in self.selected_rows)

    def report(self):
        return dict(status="COMPLETE_SNAPSHOT_CANDIDATE_SELECTION" if self.inventory.complete else "PARTIAL_SNAPSHOT_CANDIDATE_SELECTION",
                    snapshot_sha256=self.inventory.snapshot_sha256, source_plan_sha256=self.inventory.plan_sha256,
                    gauge_id=self.gauge_id, split=self.split, history_hours=self.history_hours, cohort_mode=self.cohort_mode,
                    case_count=self.case_count, cohort_admission_digest=self.cohort_admission_digest,
                    selection_digest=self.selection_digest, final_scientific_cohort=False, upstream_sources_reaudited=False)


@dataclass(frozen=True)
class CorpusSnapshot:
    root: str
    plan_sha256: str
    plan_file_sha256: str
    snapshot_sha256: str
    stored_history_hours: int
    splits: tuple[str, ...]
    shards: tuple[ShardSnapshot, ...]
    missing_chunks: tuple[str, ...]
    planned_cases: int

    @property
    def complete(self): return not self.missing_chunks

    def report(self):
        return dict(status="COMPLETE_COMMITTED_CORPUS_SNAPSHOT" if self.complete else "PARTIAL_COMMITTED_CORPUS_SNAPSHOT",
                    snapshot_sha256=self.snapshot_sha256, plan_sha256=self.plan_sha256, plan_file_sha256=self.plan_file_sha256,
                    observed_commits=[{"chunk": shard.identifier, "commit_sha256": shard.commit_sha256} for shard in self.shards],
                    missing_chunks=list(self.missing_chunks), planned_cases=self.planned_cases,
                    available_candidate_cases=sum(len(shard.source_admitted) for shard in self.shards),
                    source_admitted_cases=sum(sum(shard.source_admitted) for shard in self.shards),
                    natural_l24_complete_cases=sum(sum(shard.natural24) for shard in self.shards),
                    stored_history_hours=self.stored_history_hours, final_scientific_cohort=False,
                    upstream_sources_reaudited=False, numerical_payloads_checked=True,
                    protected_assessment_access=False, fitting_performed=False)

    def select(self, *, gauge_id, split, history_hours, cohort_mode, allow_partial=False):
        require(gauge_id in GAUGES and split in self.splits and history_hours in (24, 168)
                and history_hours <= self.stored_history_hours and cohort_mode in COHORT_MODES and type(allow_partial) is bool,
                "Explicit supported gauge/split/history/cohort selection required")
        if not self.complete and not allow_partial:
            raise IncompleteCorpusError("Partial corpus selection requires explicit allow_partial=True; it is not a final cohort")
        require(cohort_mode != "common_l168_complete" or self.stored_history_hours == 168,
                "L168 common cohort cannot be inferred from an L24-only store")
        gauge = GAUGES.index(gauge_id)
        selected = []
        for index, shard in enumerate(self.shards):
            if shard.unit["split"] != split:
                continue
            natural = shard.natural24 if history_hours == 24 else shard.source_admitted
            policy = (True,) * len(natural) if cohort_mode == "all_candidates" else shard.source_admitted if cohort_mode == "common_l168_complete" else natural
            rows = tuple(i for i in range(gauge, len(policy), 3) if policy[i])
            if rows:
                selected.append((index, rows))
        membership = dict(source_plan_sha256=self.plan_sha256, snapshot_sha256=self.snapshot_sha256,
                          gauge_id=gauge_id, split=split,
                          selected=[{"commit_sha256": self.shards[index].commit_sha256, "rows": list(rows)} for index, rows in selected])
        cohort = digest(membership)
        selection = digest(dict(cohort_admission_digest=cohort, history_hours=history_hours, cohort_mode=cohort_mode,
                                allow_partial=allow_partial, complete_snapshot=self.complete))
        return DatasetSelection(self, gauge_id, split, history_hours, cohort_mode, allow_partial, tuple(selected), cohort, selection)


def inspect_corpus(root, *, expected_plan_sha256):
    """Capture only committed directories seen in this scan; never follow appends."""
    root = Path(root).absolute(); _directory(root)
    plan, descriptors, plan_file_sha = _plan(root, expected_plan_sha256)
    chunks_root = root / "chunks"; _directory(chunks_root)
    planned_paths = {unit["id"] + "/" + f'{chunk["index"]:03d}' for unit, chunk in descriptors}
    allowed_units = {unit["id"] for unit in plan["units"]}
    observed = set()
    for unit_path in chunks_root.iterdir():
        _directory(unit_path)
        require(unit_path.name in allowed_units, "Unplanned corpus unit directory")
        for chunk_path in unit_path.iterdir():
            _directory(chunk_path)
            name = unit_path.name + "/" + chunk_path.name
            require(name in planned_paths, "Unplanned or duplicate corpus chunk directory")
            observed.add(name)
    shards, missing = [], []
    for unit, chunk in descriptors:
        name = unit["id"] + "/" + f'{chunk["index"]:03d}'
        if name not in observed:
            missing.append(name); continue
        arrays, _, commit_sha = _load_shard(root, unit, chunk, expected_plan_sha256)
        shards.append(ShardSnapshot(json.dumps(unit, sort_keys=True), json.dumps(chunk, sort_keys=True), commit_sha,
                                    tuple(bool(value) for value in _complete(arrays, 24)),
                                    tuple(bool(value) for value in arrays["admission_mask"])))
        del arrays
    require(_sha(_read(root / "PLAN.json", 4 * 1024**2)) == plan_file_sha, "Corpus plan changed during snapshot")
    identity = dict(plan_sha256=expected_plan_sha256, plan_file_sha256=plan_file_sha,
                    observed=[{"chunk": shard.identifier, "commit_sha256": shard.commit_sha256} for shard in shards], missing=missing)
    return CorpusSnapshot(str(root), expected_plan_sha256, plan_file_sha, digest(identity), plan["history_hours"],
                          tuple(split["name"] for split in plan["splits"]), tuple(shards), tuple(missing), 3 * plan["candidate_origins"])


def _view_reason(arrays):
    result = []
    for i in range(len(arrays["origin_time"])):
        reasons = [label for name, label in (("weather", "NO_COMPLETE_SINGLE_WEATHER_RUN"), ("history", "HISTORY_INCOMPLETE"),
                    ("state", "SOIL_UNAVAILABLE_OR_STALE"), ("target", "TARGET_INCOMPLETE")) if not arrays[name + "_mask"][i].all()]
        result.append("|".join(reasons) or "COMPLETE")
    return np.array(result)


def iter_batches(selection, *, feature_view, batch_size=128):
    """Yield bounded immutable copies; NPZ compression is not zero-copy memmap."""
    require(type(selection) is DatasetSelection and feature_view in ("full", "q_only")
            and type(batch_size) is int and 1 <= batch_size <= 128, "Explicit selection/view and bounded batch size required")
    snapshot = selection.inventory
    root = Path(snapshot.root)
    require(_sha(_read(root / "PLAN.json", 4 * 1024**2)) == snapshot.plan_file_sha256, "Snapshot corpus plan changed")
    for shard_index, rows in selection.selected_rows:
        shard = snapshot.shards[shard_index]
        arrays, _, _ = _load_shard(root, shard.unit, shard.chunk, snapshot.plan_sha256, expected_commit_sha=shard.commit_sha256)
        for offset in range(0, len(rows), batch_size):
            chosen = list(rows[offset:offset + batch_size])
            data = {name: np.array(value[chosen], copy=True) for name, value in arrays.items()}
            data["source_eligibility_reason"] = data.pop("eligibility_reason")
            data["source_admission_mask"] = data.pop("admission_mask")
            for name in ("history", "history_mask", "history_time"):
                data[name] = np.array(data[name][:, -selection.history_hours:], copy=True)
            data["view_eligibility_reason"] = _view_reason(data)
            data["view_complete_mask"] = _complete(data, selection.history_hours)
            if feature_view == "q_only":
                from .ablations import q_only_view
                batch = {**data, "case_ids": tuple(data["case_id"].tolist()), "history_features": ("discharge_cms", "age_hours")}
                view = q_only_view(batch, cohort_admission_digest=selection.cohort_admission_digest)
                for name in MODEL_KEYS:
                    data[name] = np.array(getattr(view, name), copy=True)
            data["feature_schema"] = np.array(Q_ONLY_FEATURE_SCHEMA if feature_view == "q_only" else FULL_FEATURE_SCHEMA)
            data["cohort_admission_digest"] = np.array(selection.cohort_admission_digest)
            data["selection_digest"] = np.array(selection.selection_digest)
            data["source_plan_sha256"] = np.array(snapshot.plan_sha256)
            data["source_shard_commit_sha256"] = np.array(shard.commit_sha256)
            for value in data.values():
                value.setflags(write=False)
            yield data
        del arrays
