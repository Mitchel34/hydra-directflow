"""Bounded typed arrays from explicitly selected, verified development shards.

No fitting, scaler statistics, network, source writes, role inference or
scientific execution authorization. Role membership follows caller-declared
half-open UTC windows, with all other selected rows retained as unassigned.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from types import MappingProxyType
import time

import numpy as np

from .corpus_dataset import DatasetSelection, iter_batches, FULL_FEATURE_SCHEMA, Q_ONLY_FEATURE_SCHEMA
from .local_data import GAUGES, iso
from .reference import ASSESSMENT_START, digest, require

ROLES = ("fit", "inner_stop", "outer")
MODEL_KEYS = ("history", "history_mask", "weather", "weather_mask", "state", "state_mask", "target", "target_mask")
COORD_KEYS = ("origin_time", "history_time", "target_time", "site_id", "case_id")
MAX_RSS = 4 * 1024**3
WORKING_RESERVE = 64 * 1024**2


def _hour(stamp):
    value = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    require(value.tzinfo is not None and value.utcoffset().total_seconds() == 0
            and value.minute == value.second == value.microsecond == 0, "Caller role bounds require exact UTC hours")
    return int(value.timestamp())


def _roles(role_time_bounds):
    require(isinstance(role_time_bounds, dict) and set(role_time_bounds) <= set(ROLES), "Explicit fit/inner_stop/outer role windows required")
    result, previous_end = {}, None
    for name in ROLES:
        if name not in role_time_bounds:
            continue
        item = role_time_bounds[name]
        require(isinstance(item, dict) and set(item) == {"start_utc", "end_utc"}, "Role bounds must be explicit half-open UTC intervals")
        start, end = _hour(item["start_utc"]), _hour(item["end_utc"])
        require(start < end <= ASSESSMENT_START and (previous_end is None or start >= previous_end),
                "Role windows overlap, are out of order, or cross protected assessment")
        result[name] = (start, end); previous_end = end
    return result


def _selection_rows(selection):
    require(type(selection) is DatasetSelection, "Verified corpus_dataset selection required")
    snapshot = selection.inventory
    gauge = GAUGES.index(selection.gauge_id)
    rows, descriptors = [], []
    for shard_index, local_rows in selection.selected_rows:
        require(type(shard_index) is int and 0 <= shard_index < len(snapshot.shards), "Selection shard index differs")
        shard = snapshot.shards[shard_index]
        require(shard.unit["split"] == selection.split and tuple(sorted(set(local_rows))) == local_rows,
                "Selection shard/split or row order differs")
        first, count = _hour(shard.chunk["first_origin_utc"]), shard.chunk["origin_count"]
        for local in local_rows:
            require(type(local) is int and 0 <= local < 3 * count and local % 3 == gauge, "Selection gauge/row differs")
            rows.append(first + (local // 3) * 3600)
        descriptors.append({"source_shard": shard.identifier, "commit_sha256": shard.commit_sha256,
                            "source_rows": list(local_rows), "source_split": selection.split})
    require(rows == sorted(set(rows)), "Selection contains duplicate or unordered gauge/origin cases")
    membership = dict(source_plan_sha256=snapshot.plan_sha256, snapshot_sha256=snapshot.snapshot_sha256,
                      gauge_id=selection.gauge_id, split=selection.split,
                      selected=[{"commit_sha256": snapshot.shards[i].commit_sha256, "rows": list(r)} for i, r in selection.selected_rows])
    require(digest(membership) == selection.cohort_admission_digest, "Selection membership digest differs")
    return rows, descriptors


def _geometry(count, length, feature_view):
    weather, state = (4, 2) if feature_view == "full" else (1, 0)
    result = {}
    for name, shape in (("history", (count, length, 2)), ("weather", (count, 18, weather)),
                        ("state", (count, state)), ("target", (count, 18))):
        result[name] = (shape, np.dtype("<f8")); result[name + "_mask"] = (shape, np.dtype(bool))
    for name, shape in (("origin_time", (count,)), ("site_id", (count,)),
                        ("history_time", (count, length)), ("target_time", (count, 18))):
        result[name] = (shape, np.dtype("<i8"))
    result["case_id"] = ((count,), np.dtype("<U29"))
    return result


def _array_digest(arrays):
    result = hashlib.sha256()
    for name in sorted(arrays):
        value = arrays[name]
        result.update(json.dumps([name, value.dtype.str, value.shape], separators=(",", ":")).encode())
        flat = value.reshape(-1)
        step = max(1, 1024**2 // value.dtype.itemsize)
        for offset in range(0, len(flat), step):
            result.update(flat[offset:offset + step].tobytes())
    return result.hexdigest()


def _ranges(indices):
    result = []
    for value in indices:
        if result and value == result[-1][1]:
            result[-1][1] += 1
        else:
            result.append([int(value), int(value) + 1])
    return result


@dataclass(frozen=True)
class TrainerBundle:
    arrays: MappingProxyType
    manifest_json: str
    construction_report_json: str

    @property
    def manifest(self): return json.loads(self.manifest_json)
    @property
    def construction_report(self): return json.loads(self.construction_report_json)
    @property
    def source_identity(self): return self.manifest["bundle_identity"]

    def index_ranges(self, role):
        roles = self.manifest["roles"]
        require(role in roles, "Role was not declared; no outer-to-stopping fallback")
        return tuple(tuple(bounds) for bounds in roles[role]["index_ranges"])

    def indices_for(self, role):
        """Explicit bounded, reiterable Python-int tuple for trainer mechanics."""
        ranges = self.index_ranges(role)
        count = len(self.arrays["origin_time"])
        require(all(0 <= start <= end <= count for start, end in ranges)
                and sum(end - start for start, end in ranges) <= count,
                "Role index ranges exceed the typed bundle")
        return tuple(index for start, end in ranges for index in range(start, end))


def assemble_bundle(selections, *, feature_view, role_time_bounds, max_total_bytes,
                    max_batch_reads, allow_partial=False, batch_size=128, max_rss_bytes=MAX_RSS):
    """Combine selections chronologically under explicit bounds; retain all rows.

    Only strict origin/target-window bounds are implemented. No historical
    publication lag or refit-label cutoff is inferred from observed row positions.
    """
    require(isinstance(selections, (list, tuple)) and 1 <= len(selections) <= 32, "Explicit bounded selections required")
    require(feature_view in ("full", "q_only") and type(allow_partial) is bool, "Explicit supported feature/partial mode required")
    require(type(max_total_bytes) is int and 0 < max_total_bytes <= MAX_RSS
            and type(max_batch_reads) is int and 0 <= max_batch_reads <= 100000
            and type(batch_size) is int and 1 <= batch_size <= 128
            and type(max_rss_bytes) is int and 0 < max_rss_bytes <= MAX_RSS, "Explicit bounded memory/read allowance required")
    roles = _roles(role_time_bounds)
    inventory = []
    for position, selection in enumerate(selections):
        origins, shards = _selection_rows(selection)
        require(allow_partial or selection.inventory.complete, "Partial snapshots require explicit engineering allow_partial=True")
        inventory.append((position, selection, origins, shards))
    identity = {(s.gauge_id, s.history_hours, s.inventory.plan_sha256, s.inventory.snapshot_sha256) for _, s, _, _ in inventory}
    require(len(identity) == 1, "Selections must share gauge, history view, source plan and immutable snapshot")
    gauge_id, length, plan_sha, snapshot_sha = next(iter(identity))
    inventory.sort(key=lambda item: item[2][0] if item[2] else float("inf"))
    ordered_origins = [origin for _, _, origins, _ in inventory for origin in origins]
    require(ordered_origins == sorted(set(ordered_origins)), "Selections overlap or interleave; duplicate gauge cases cannot be combined")
    count = len(ordered_origins)
    geometry = _geometry(count, length, feature_view)
    feature_schema = FULL_FEATURE_SCHEMA if feature_view == "full" else Q_ONLY_FEATURE_SCHEMA
    predicted_bytes = sum(int(np.prod(shape)) * dtype.itemsize for shape, dtype in geometry.values()) + 4 * (len(feature_schema) + 64)
    require(predicted_bytes <= max_total_bytes, "Dense array byte allowance exceeded before allocation")
    predicted_reads = sum((len(rows) + batch_size - 1) // batch_size for _, selection, _, _ in inventory for _, rows in selection.selected_rows)
    require(predicted_reads <= max_batch_reads, "Batch-read allowance exceeded before data reads")
    import psutil
    process, started, peak = psutil.Process(), time.monotonic(), 0
    def guard():
        nonlocal peak
        peak = max(peak, process.memory_info().rss)
        require(peak <= max_rss_bytes, "Typed bundle process RSS bound reached")
    guard()
    require(peak + predicted_bytes + (WORKING_RESERVE if count else 0) <= max_rss_bytes,
            "Insufficient process memory headroom before allocation")
    arrays = {name: np.empty(shape, dtype=dtype) for name, (shape, dtype) in geometry.items()}
    parts, provenance, cursor, reads = [], [], 0, 0
    for original_position, selection, origins, shards in inventory:
        start = cursor
        for batch in iter_batches(selection, feature_view=feature_view, batch_size=batch_size):
            reads += 1; require(reads <= max_batch_reads, "Observed batch-read allowance exceeded")
            n = len(batch["case_id"])
            require(n <= batch_size and cursor + n <= count, "Reader changed declared batch geometry")
            require(batch["cohort_admission_digest"].item() == selection.cohort_admission_digest
                    and batch["feature_schema"].item() == feature_schema, "Reader changed selected cohort or feature mode")
            for name in MODEL_KEYS + COORD_KEYS:
                value = batch[name]
                require(value.dtype == arrays[name].dtype and value.shape == (n,) + arrays[name].shape[1:], "Typed reader field differs: " + name)
                arrays[name][cursor:cursor + n] = value
            cursor += n; guard()
        require(cursor - start == len(origins), "Reader changed selected row count")
        parts.append({"caller_position": original_position, "split": selection.split,
                      "selection_digest": selection.selection_digest, "source_cohort_digest": selection.cohort_admission_digest,
                      "cohort_mode": selection.cohort_mode, "index_range": [start, cursor], "case_count": len(origins)})
        offset = start
        for shard in shards:
            stop = offset + len(shard["source_rows"])
            provenance.append({**shard, "output_index_range": [offset, stop]}); offset = stop
    require(cursor == count and reads == predicted_reads, "Typed bundle row/read accounting differs")
    origins = np.array(ordered_origins, dtype=np.int64)
    require(np.array_equal(arrays["origin_time"], origins) and np.all(arrays["site_id"] == GAUGES.index(gauge_id)),
            "Typed bundle gauge/origin order changed")
    require(arrays["case_id"].tolist() == [gauge_id + ":" + iso(int(t)) for t in origins]
            and np.array_equal(arrays["target_time"], origins[:, None] + np.arange(1, 19) * 3600)
            and np.all(arrays["target_time"] < ASSESSMENT_START), "Typed bundle case/target identity or boundary differs")
    cohort = digest({"source_plan_sha256": plan_sha, "snapshot_sha256": snapshot_sha, "gauge_id": gauge_id,
                     "members": [{"commit_sha256": p["commit_sha256"], "rows": p["source_rows"]} for p in provenance]})
    assigned, role_manifest = np.zeros(count, bool), {}
    for name, (start, end) in roles.items():
        origin_inside = (origins >= start) & (origins < end)
        target_inside = (arrays["target_time"] >= start).all(axis=1) & (arrays["target_time"] < end).all(axis=1)
        admitted = origin_inside & target_inside
        require(not (assigned & admitted).any(), "Declared role assignments overlap")
        indices = np.flatnonzero(admitted)
        assigned |= admitted
        role_manifest[name] = {"start_utc": iso(start), "end_exclusive_utc": iso(end), "index_ranges": _ranges(indices),
                               "case_count": int(admitted.sum()), "origin_window_cases": int(origin_inside.sum()),
                               "target_boundary_purged_cases": int((origin_inside & ~target_inside).sum()),
                               "case_ids_sha256": digest(arrays["case_id"][indices].tolist()), "zero_row_support": len(indices) == 0,
                               "cases_with_observed_labels": int(arrays["target_mask"][indices].any(axis=1).sum()),
                               "complete_target_cases": int(arrays["target_mask"][indices].all(axis=1).sum()),
                               "observed_target_cells": int(arrays["target_mask"][indices].sum())}
    arrays["feature_schema"] = np.array(feature_schema)
    arrays["cohort_admission_digest"] = np.array(cohort)
    exact_bytes = sum(value.nbytes for value in arrays.values())
    require(exact_bytes == predicted_bytes and exact_bytes <= max_total_bytes, "Dense array byte accounting differs")
    core = dict(schema="hydra-trainer-typed-bundle/v1", source_plan_sha256=plan_sha, snapshot_sha256=snapshot_sha,
                gauge_id=gauge_id, history_hours=length, feature_schema=feature_schema, feature_view=feature_view,
                cohort_admission_digest=cohort, candidate_selection_parts=parts, shard_provenance=provenance,
                roles=role_manifest, role_assignment_digest=digest(role_manifest), unassigned_index_ranges=_ranges(np.flatnonzero(~assigned)),
                unassigned_cases=int((~assigned).sum()), case_count=count, arrays_sha256=_array_digest(arrays),
                complete_source_snapshot=all(s.inventory.complete for _, s, _, _ in inventory),
                scientific_admission=False, fitted_statistics=False,
                time_policy="CALLER_DECLARED_STRICT_ORIGIN_AND_H18_TARGET_WINDOWS_ONLY",
                simulated_refit_label_cutoff_implemented=False)
    manifest = {**core, "bundle_identity": digest(core)}
    for value in arrays.values(): value.setflags(write=False)
    guard()
    construction = dict(dense_array_bytes=exact_bytes, batch_reads=reads, max_total_bytes=max_total_bytes,
                        max_batch_reads=max_batch_reads, max_rss_bytes=max_rss_bytes, observed_peak_rss_bytes=peak,
                        working_read_reserve_bytes=WORKING_RESERVE if count else 0,
                        elapsed_seconds=time.monotonic() - started, source_files_written=False,
                        network_requests=0, scaler_fits=0, optimizer_updates=0)
    return TrainerBundle(MappingProxyType(arrays), json.dumps(manifest, sort_keys=True), json.dumps(construction, sort_keys=True))
