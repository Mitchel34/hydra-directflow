"""Authenticated, bounded offline development preprocessing across HRRR cycles.

Selection is latest eligible among the supplied local candidates, never a claim
of globally latest coverage. Source gaps remain masks. No network, hydration,
scaler fitting, optimizer updates, or protected assessment decoding occurs.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import time

import numpy as np

from .contract import DevelopmentProtocol, TensorSpec
from .local_data import (
    GAUGES, FEATURE_GROUPS, GRID_SHA256, USGS_MANIFEST_SHA256, WEIGHTS_SHA256,
    DEVELOPMENT_QUALIFICATION, CFS_TO_CMS, LocalCycle, basin_reduce, bounded_npy,
    iso, utc_epoch, sha256, read_cycle_tar, verified_local_bytes,
    load_candidate_weights, _source_bytes, _verify_grid_and_clocks,
    _verify_soil_grid, _verify_weather_crop,
)
from .reference import (
    ASSESSMENT_START, AvailabilityPolicy, Observation, SoilInitialState,
    WeatherRun, digest, history_window, latest_weather_run, require, soil_state,
    target_window,
)

INDEX_SHA256 = "75469df20f79d94ffb058f439f1ac873f0cf4164c1417eec7298a6641a35baa5"
INDEX_BYTES = 112202255
MAX_CYCLES = 32
MAX_ORIGINS = 128
MAX_PARTITIONS = 24
SELECTION_SCOPE = "OFFLINE_SUPPLIED_CANDIDATES"
UNITS = {"temperature": "K", "humidity": "kg kg-1", "precipitation": "kg m-2", "snow_water_equivalent": "kg m-2"}


def cycle_reference(cycle_id):
    require(isinstance(cycle_id, str) and re.fullmatch(r"\d{8}_\d{2}Z", cycle_id), "Invalid cycle ID")
    try:
        reference = int(datetime.strptime(cycle_id, "%Y%m%d_%HZ").replace(tzinfo=timezone.utc).timestamp())
    except ValueError as exc:
        raise ValueError("Invalid cycle ID clock") from exc
    require(reference + 48 * 3600 < ASSESSMENT_START, "Protected assessment archive access is disabled")
    return reference


def validate_request(archive_specs, origins_utc, history_hours):
    protocol = DevelopmentProtocol(history_hours=history_hours)
    require(isinstance(archive_specs, (list, tuple)) and 1 <= len(archive_specs) <= MAX_CYCLES,
            "Supply one to 32 explicitly local cycle archives")
    ids = [spec["cycle_id"] for spec in archive_specs]
    require(len(ids) == len(set(ids)), "Duplicate supplied cycle ID")
    for cycle_id in ids:
        cycle_reference(cycle_id)
    require(isinstance(origins_utc, (list, tuple)) and 1 <= len(origins_utc) <= MAX_ORIGINS,
            "Supply one to 128 development origins")
    origins = [utc_epoch(stamp) for stamp in origins_utc]
    require(origins == sorted(set(origins)) and all(origin % 3600 == 0 for origin in origins),
            "Origins must be unique, increasing exact UTC hours")
    require(all(origin + 18 * 3600 < ASSESSMENT_START for origin in origins),
            "Protected assessment and boundary-crossing target windows are disabled")
    return protocol, origins


def qualified_cycle_rows(path, cycle_ids):
    """One bounded streaming pass verifies the sealed index before using rows."""
    p = Path(path)
    before = p.lstat()
    require(stat.S_ISREG(before.st_mode) and not getattr(before, "st_flags", 0) & 0x40000000,
            "Qualified index must be a resident regular file")
    require(before.st_size == INDEX_BYTES, "Qualified index byte count differs")
    wanted = set(cycle_ids)
    rows, count, hasher = {}, 0, hashlib.sha256()
    with p.open("rb") as stream:
        while line := stream.readline(65537):
            require(len(line) <= 65536 and line.endswith(b"\n"), "Index line exceeds bounded encoding")
            hasher.update(line)
            count += 1
            require(count <= 61824, "Qualified index row bound exceeded")
            row = json.loads(line)
            if row["cycle"] in wanted:
                require(row["cycle"] not in rows, "Duplicate cycle in qualified index")
                rows[row["cycle"]] = row
    after = p.lstat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "Qualified index changed while reading")
    require(count == 61824 and hasher.hexdigest() == INDEX_SHA256, "Sealed qualified index SHA256 differs")
    require(set(rows) == wanted, "Requested cycle absent from sealed qualified index")
    return rows


def load_authenticated_cycle(spec, index_row):
    """Bind sealed index -> original commit -> tar -> members -> source clocks/tiles."""
    cycle_id = spec["cycle_id"]
    reference = cycle_reference(cycle_id)
    require(index_row["cycle"] == cycle_id and index_row["location_kind"] == "commit"
            and index_row["original_quality_flags_rewritten"] is False, "Unsupported qualified cycle lineage")
    identity = index_row["source_locations"]["commit"]
    commit_raw = verified_local_bytes(spec["commit_path"], identity["sha256"],
                                      expected_bytes=identity["bytes"], max_bytes=16384)
    commit = json.loads(commit_raw)
    require(commit["schema"] == "hrrr-verified-cloud-cycle/v1" and commit["cycle"] == cycle_id,
            "Commit cycle/schema differs")
    bundle, worker = commit["bundle"], commit["worker_manifest"]
    require(bundle["verified"] is True and bundle["bytes"] == worker["bundle_bytes"]
            and bundle["sha256"] == worker["bundle_sha256"] and worker["cycle"] == cycle_id,
            "Commit bundle identity differs")
    # read_cycle_tar authenticates exact bytes plus all whitelisted members.
    require(Path(spec["archive_path"]).lstat().st_size == bundle["bytes"], "Archive byte count differs")
    members, manifest = read_cycle_tar(spec["archive_path"], bundle["sha256"], cycle_id=cycle_id)
    require(sha256(members["CYCLE_MANIFEST.json"]) == worker["manifest_sha256"], "Commit manifest binding differs")
    n = manifest["record_count"]
    require(type(n) is int and 1 <= n <= 145 and n == worker["record_count"] == index_row["record_count"],
            "Cycle record count differs from sealed lineage")
    require(manifest["raw_shape"] == [n, 957] and manifest["mask_shape"] == [n, 120]
            and manifest["data_complete"] is index_row["data_complete"], "Cycle shapes/completeness differ")
    require(manifest["assets"]["grid_sha256"] == GRID_SHA256, "Candidate acquisition grid differs")
    values = bounded_npy(members["raw_values.npy"], shape=(n, 957), dtype="<f8")
    bits = bounded_npy(members["raw_missing_bits.npy"], shape=(n, 120), dtype="u1")
    missing = np.unpackbits(bits, axis=1, bitorder="little")
    require(not missing[:, 957:].any(), "Nonzero unused mask padding")
    present = ~missing[:, :957].astype(bool)
    require(np.isfinite(values[present]).all(), "Nonfinite admitted source value")
    records = json.loads(members["RAW_PROVENANCE.json"])
    fields = json.loads(members["FIELD_SOURCES.json"])
    availability = json.loads(members["AVAILABILITY.json"])
    require(len(records) == n and availability == manifest["availability"] and availability["cycle"] == cycle_id,
            "Original source/availability inventory differs")
    expected_leads = 36 if reference % (6 * 3600) == 0 else 18
    _verify_grid_and_clocks(members, fields, reference, cycle_id=cycle_id, minimum_leads=expected_leads)
    seen, groups = set(), {g["group"]: g for g in availability["groups"]}
    require(len(groups) == len(availability["groups"]), "Duplicate availability group")
    for index, record in enumerate(records):
        group, lead = record["group"], record["lead"]
        require(record["record_index"] == index and (group, lead) not in seen, "Duplicate/reordered source record")
        seen.add((group, lead))
        require(utc_epoch(record["source_reference_UTC"]) == reference
                and utc_epoch(record["source_valid_UTC"]) == reference + lead * 3600, "Record source clock differs")
        require(sha256(values[index].astype("<f8", copy=False).tobytes()) == record["source_values_sha256_float64_le"]
                and sha256(bits[index].tobytes()) == record["missing_bits_sha256"], "Record value/mask hash differs")
        require(int((~present[index]).sum()) == record["missing_cells"], "Record missing count differs")
        if group == "soil_moisture":
            require(reference % (6 * 3600) == 0 and lead == 0 and record["native_metadata"]["units"] == "Proportion",
                    "Unexpected initial soil definition")
            _verify_soil_grid(_source_bytes(members, record["source"], cycle_id=cycle_id))
        else:
            require(group in UNITS and 1 <= lead <= expected_leads and record["units"] == UNITS[group],
                    "Unsupported weather group, lead or units")
            require(record["APCP_interval_hours"] == ([lead - 1, lead] if group == "precipitation" else None),
                    "Unexpected precipitation interval")
            require(record["field_source_id"] in fields, "Missing original field source")
    present_groups = {group for group, _ in seen}
    require(present_groups == set(index_row["groups_available"]), "Available groups differ from sealed index")
    for group in present_groups:
        leads = [0] if group == "soil_moisture" else list(range(1, expected_leads + 1))
        require(sorted(lead for g, lead in seen if g == group) == leads,
                "Available group has incomplete source lead support")
        require(groups[group]["leads"] == leads and groups[group]["record_count"] == len(leads)
                and groups[group]["status"] == "AVAILABLE_VERIFIED", "Original available group contract differs")
    _verify_weather_crop(members, fields, records, values, present, cycle_id=cycle_id)
    cycle = LocalCycle(reference, values, present, tuple(records), len(members), fields, availability)
    receipt = {"cycle_id": cycle_id, "archive_sha256": bundle["sha256"], "archive_bytes": bundle["bytes"],
               "commit_sha256": identity["sha256"], "manifest_sha256": worker["manifest_sha256"],
               "record_count": n, "data_complete": manifest["data_complete"],
               "weather_source_slots": sorted({f["spec"]["chunks"][0] for f in fields.values()}),
               "groups_available": sorted(present_groups),
               "weather_tiles_redecoded": bool(fields), "soil_values_record_hash_verified": "soil_moisture" in present_groups}
    return cycle, receipt


def _months_for_windows(windows):
    result = set()
    for start, end in windows:
        require(start <= end < ASSESSMENT_START, "Protected USGS window disabled")
        year, month = datetime.fromtimestamp(start, timezone.utc).timetuple()[:2]
        while True:
            stamp = f"{year:04d}-{month:02d}"
            result.add(stamp)
            next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
            next_start = int(datetime(next_year, next_month, 1, tzinfo=timezone.utc).timestamp())
            require(next_start <= ASSESSMENT_START, "Monthly partition overlaps protected assessment")
            if next_start > end:
                break
            year, month = next_year, next_month
    require(len(result) <= MAX_PARTITIONS, "USGS partition count exceeds bound")
    return result


def load_usgs_windows(paths, manifest_path, windows):
    """Authenticate native monthly partitions and return only requested windows.

    Parquet may decode complete row groups before filtering. All admitted months
    therefore precede the protected assessment, even for pre-boundary windows.
    """
    from pyarrow import BufferReader
    import pyarrow.parquet as pq
    needed = _months_for_windows(windows)
    require(needed <= set(paths), "Required local USGS month is absent")
    manifest = json.loads(verified_local_bytes(manifest_path, USGS_MANIFEST_SHA256, max_bytes=2 * 1024 * 1024))
    partition_map = {}
    for entry in manifest["native_partitions"]:
        name = Path(entry["path"]).name
        require(name not in partition_map, "Duplicate USGS partition name in source manifest")
        partition_map[name] = entry
    columns = ["site", "sample_epoch", "sample_time_utc", "source_units", "discharge_cfs",
               "source_qualifiers_json", "numeric_valid", "is_exact_utc_hour", "is_core",
               "source_wire_sha256", "historical_available_at_utc"]
    observations, metadata, seen = [], [], set()
    # OR of bounded windows controls returned rows, not Parquet page decoding.
    filters = [[("sample_epoch", ">=", start), ("sample_epoch", "<=", end)] for start, end in windows]
    for month in sorted(needed):
        entry = partition_map.get(month + ".parquet")
        require(entry is not None, "Month absent from pinned USGS source manifest")
        raw = verified_local_bytes(paths[month], entry["sha256"], expected_bytes=entry["bytes"], max_bytes=512 * 1024)
        rows = pq.read_table(BufferReader(raw), columns=columns, filters=filters).to_pylist()
        require(len(rows) <= 30000, "USGS decoded row bound exceeded")
        exact = 0
        for row in rows:
            t, site = row["sample_epoch"], row["site"]
            require(any(start <= t <= end for start, end in windows) and t < ASSESSMENT_START
                    and utc_epoch(row["sample_time_utc"]) == t, "USGS sample/window clock differs")
            require(site in GAUGES and row["source_units"] == "ft3/s"
                    and row["historical_available_at_utc"] is None, "USGS site, units or vintage differs")
            require(row["is_exact_utc_hour"] == (t % 3600 == 0), "USGS exact-hour indicator differs")
            if not row["is_core"] or not row["is_exact_utc_hour"]:
                continue
            # Padding rows reside in adjacent partitions but is_core prevents duplicate admission.
            require(iso(t).startswith(month), "Core USGS row is outside native month")
            require((site, t) not in seen, "Duplicate native exact-hour record")
            seen.add((site, t))
            exact += 1
            value = row["discharge_cfs"]
            usable = row["numeric_valid"] and type(value) in (int, float) and math.isfinite(value) and value >= 0
            flags = tuple(json.loads(row["source_qualifiers_json"]))
            observations.append(Observation(site, t, None, float(value) * CFS_TO_CMS if usable else None,
                                            flags, "usgs-wire-sha256:" + row["source_wire_sha256"]))
        metadata.append({"month": month, "sha256": entry["sha256"], "bytes": entry["bytes"],
                         "returned_rows": len(rows), "exact_core_rows": exact})
    return observations, metadata


def _metadata_record(cycle_id, cycle, record):
    """Original complete record plus original group/field metadata, kept privately."""
    group = next(g for g in cycle.availability["groups"] if g["group"] == record["group"])
    result = {"cycle_id": cycle_id, "record": record, "group_availability": group}
    if "field_source_id" in record:
        result["field_source"] = cycle.fields[record["field_source_id"]]
    return json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)


def assemble_cases(cycles, reduced, observations, origins, protocol):
    """Pure case assembly; no files or fitted transforms. Missing cases survive."""
    require(protocol == DevelopmentProtocol(history_hours=protocol.history_hours),
            "Only the fixed development candidate protocol is implemented")
    policy = AvailabilityPolicy("development_simulated_lag_2h_v1", "retrospective_simulated", 7200)
    strict_a = lambda flags: flags == ("A",)
    metadata, metadata_indices, row_indexes = [], {}, {}
    for cycle_id, cycle in sorted(cycles.items()):
        row_indexes[cycle_id] = {(r["group"], r["lead"]): i for i, r in enumerate(cycle.records)}
        for r in cycle.records:
            metadata_indices[cycle_id, r["group"], r["lead"]] = len(metadata)
            metadata.append(_metadata_record(cycle_id, cycle, r))
    result = {name: [] for name in ("history", "history_mask", "weather", "weather_mask", "state", "state_mask",
              "site_id", "origin_time", "case_id", "target", "target_mask", "history_time", "target_time",
              "history_quality_flags", "target_quality_flags", "history_source_ids", "target_source_ids",
              "source_cycle", "soil_source_cycle", "weather_source_leads", "weather_metadata_index",
              "soil_metadata_index", "eligibility_reason")}
    runs_by_site, soil_by_site, soil_lookup = {}, {}, {}
    for site_index, site in enumerate(GAUGES):
        runs, soils = [], []
        for cycle_id, cycle in sorted(cycles.items()):
            means, masks, _ = reduced[cycle_id]
            indexes = row_indexes[cycle_id]
            leads = sorted({lead for group, lead in indexes if group in UNITS})
            support = tuple((feature, tuple(bool(masks[indexes[group, lead], site_index])
                                             if (group, lead) in indexes else False for lead in leads))
                            for feature, group in FEATURE_GROUPS)
            runs.append(WeatherRun(cycle_id, cycle.reference, None,
                                   tuple(cycle.reference + lead * 3600 for lead in leads), support))
            if ("soil_moisture", 0) in indexes:
                row = indexes["soil_moisture", 0]
                record = cycle.records[row]
                source_id = record["source"]["key"]
                soils.append(SoilInitialState(source_id, cycle.reference, None,
                             float(means[row, site_index]) if masks[row, site_index] else None,
                             (record["availability_status"],)))
                soil_lookup[source_id] = cycle_id
        runs_by_site[site], soil_by_site[site] = runs, soils
    for origin in origins:
        for site_index, site in enumerate(GAUGES):
            h = history_window(observations, site=site, origin_anchor=origin, cutoff=origin,
                               lookback_hours=protocol.history_hours, history_lag_hours=2,
                               policy=policy, admit_quality=strict_a)
            t = target_window(observations, site=site, origin_anchor=origin,
                              horizons_hours=protocol.horizons_hours, admit_quality=strict_a)
            chosen = latest_weather_run(runs_by_site[site], cutoff=origin, origin_anchor=origin,
                         policy=policy, horizons_hours=protocol.horizons_hours,
                         allow_older_extended_cycles=True)
            soil = soil_state(soil_by_site[site], cutoff=origin, policy=policy,
                              max_age_seconds=protocol.max_soil_age_hours * 3600)
            weather, weather_mask = np.full((18, 4), np.nan), np.zeros((18, 4), dtype=bool)
            provenance = np.full((18, 4), -1, dtype=np.int64)
            leads = np.full(18, -1, dtype=np.int64)
            cycle_id = chosen.run_id if chosen else ""
            if chosen:
                means, masks, _ = reduced[cycle_id]
                indexes = row_indexes[cycle_id]
                leads = np.array([(valid - chosen.reference) // 3600 for valid in t["times"]], dtype=np.int64)
                for j, lead in enumerate(leads):
                    for k, (_, group) in enumerate(FEATURE_GROUPS):
                        weather[j, k] = means[indexes[group, int(lead)], site_index]
                        weather_mask[j, k] = masks[indexes[group, int(lead)], site_index]
                        provenance[j, k] = metadata_indices[cycle_id, group, int(lead)]
            soil_cycle = soil_lookup.get(soil["source_id"], "")
            reasons = [reason for condition, reason in ((chosen is None, "NO_COMPLETE_SINGLE_WEATHER_RUN"),
                        (not all(h["mask"]), "HISTORY_INCOMPLETE"), (not soil["mask"], "SOIL_UNAVAILABLE_OR_STALE"),
                        (not all(t["mask"]), "TARGET_INCOMPLETE")) if condition]
            values = dict(history=list(zip(h["values"], h["age_hours"])),
                history_mask=np.repeat(np.array(h["mask"], dtype=bool)[:, None], 2, axis=1),
                weather=weather, weather_mask=weather_mask,
                state=[soil["value"], soil["age_hours"] if soil["mask"] else None], state_mask=[soil["mask"]] * 2,
                site_id=site_index, origin_time=origin, case_id=site + ":" + iso(origin),
                target=t["values"], target_mask=t["mask"], history_time=h["times"], target_time=t["times"],
                history_quality_flags=[json.dumps(f) for f in h["quality_flags"]],
                target_quality_flags=[json.dumps(f) for f in t["quality_flags"]],
                history_source_ids=[s or "" for s in h["source_ids"]], target_source_ids=[s or "" for s in t["source_ids"]],
                source_cycle=cycle_id, soil_source_cycle=soil_cycle, weather_source_leads=leads,
                weather_metadata_index=provenance,
                soil_metadata_index=metadata_indices.get((soil_cycle, "soil_moisture", 0), -1),
                eligibility_reason="|".join(reasons) or "COMPLETE")
            for name, value in values.items():
                result[name].append(value)
    arrays = {}
    float_keys = {"history", "weather", "state", "target"}
    int_keys = {"site_id", "origin_time", "history_time", "target_time", "weather_source_leads",
                "weather_metadata_index", "soil_metadata_index"}
    for name, values in result.items():
        dtype = np.float64 if name in float_keys else bool if name.endswith("_mask") else np.int64 if name in int_keys else str
        arrays[name] = np.array(values, dtype=dtype)
    arrays["source_metadata_json"] = np.array(metadata, dtype=str)
    arrays["development_labels"] = np.array([DEVELOPMENT_QUALIFICATION, "RETROSPECTIVE_SIMULATED_LAG_2H", SELECTION_SCOPE])
    spec = TensorSpec(protocol.history_hours, protocol.horizons_hours, ("discharge_cms", "age_hours"),
                      tuple(f for f, _ in FEATURE_GROUPS), ("soil_water_fraction", "soil_age_hours"), GAUGES)
    target_shape = spec.validate_shapes(**{key: arrays[key].shape for key in
        ("history", "history_mask", "weather", "weather_mask", "state", "state_mask", "site_id")})
    require(arrays["target"].shape == arrays["target_mask"].shape == target_shape, "Target shapes differ")
    return arrays


def build_multicycle_development(*, archive_specs, qualified_index_path, weights_path,
                                usgs_paths, usgs_manifest_path, origins_utc, history_hours=24):
    """Return private arrays and a portable aggregate preprocessing receipt."""
    started = time.monotonic()
    protocol, origins = validate_request(archive_specs, origins_utc, history_hours)
    stages = {}
    mark = time.monotonic()
    index = qualified_cycle_rows(qualified_index_path, [s["cycle_id"] for s in archive_specs])
    stages["index_auth"] = time.monotonic() - mark
    mark = time.monotonic()
    weights = load_candidate_weights(weights_path, qualification=DEVELOPMENT_QUALIFICATION)
    cycles, reduced, source_receipts = {}, {}, []
    for source in sorted(archive_specs, key=lambda s: s["cycle_id"]):
        cycle, receipt = load_authenticated_cycle(source, index[source["cycle_id"]])
        cycles[source["cycle_id"]] = cycle
        reduced[source["cycle_id"]] = basin_reduce(cycle.values, cycle.present, weights)
        source_receipts.append(receipt)
    stages["cycle_auth_decode_reduce"] = time.monotonic() - mark
    mark = time.monotonic()
    windows = [(origin - (history_hours + 1) * 3600, origin + 18 * 3600) for origin in origins]
    observations, partitions = load_usgs_windows(usgs_paths, usgs_manifest_path, windows)
    stages["usgs_load"] = time.monotonic() - mark
    mark = time.monotonic()
    arrays = assemble_cases(cycles, reduced, observations, origins, protocol)
    stages["case_assembly"] = time.monotonic() - mark
    stages["total"] = time.monotonic() - started
    complete = arrays["eligibility_reason"] == "COMPLETE"
    report = {
        "schema": "hydra-directflow-multicycle-development/v1",
        "status": "PASSED_COMPLETE_DEVELOPMENT_CASES" if complete.all() else "COMPLETED_WITH_EXPLICIT_MASKS",
        "artifact_created_at_utc": iso(int(time.time())), "stage_seconds": stages,
        "selection_scope": SELECTION_SCOPE, "supplied_cycle_ids": sorted(cycles),
        "selected_cycle_case_counts": dict(Counter(arrays["source_cycle"].tolist())),
        "selected_soil_cycle_case_counts": dict(Counter(arrays["soil_source_cycle"].tolist())),
        "origin_count": len(origins), "case_count": len(complete), "complete_case_count": int(complete.sum()),
        "origins_utc": [iso(origin) for origin in origins], "gauges": list(GAUGES),
        "history_hours": history_hours, "horizons_hours": list(protocol.horizons_hours),
        "quality_policy": "STRICT_EXACT_A_ONLY_DEVELOPMENT", "availability_policy": "RETROSPECTIVE_SIMULATED_LAG_2H",
        "history_end": "C_MINUS_2_HOURS_INCLUSIVE", "max_soil_age_hours": protocol.max_soil_age_hours,
        "missing_case_policy": "RETAIN_REQUESTED_CASES_WITH_MASKS_NO_SILENT_DROP",
        "eligibility_counts": dict(Counter(arrays["eligibility_reason"].tolist())),
        "sources": {"qualified_index_sha256": INDEX_SHA256, "qualified_index_bytes": INDEX_BYTES,
                    "cycles": source_receipts, "usgs_manifest_sha256": USGS_MANIFEST_SHA256,
                    "usgs_partitions": partitions, "candidate_weights_sha256": WEIGHTS_SHA256},
        "tensors": {key: {"shape": list(arrays[key].shape), "mask_true": int(arrays[key + "_mask"].sum()),
                          "mask_total": int(arrays[key + "_mask"].size)} for key in ("history", "weather", "state", "target")},
        "provenance": "Original records/group/field metadata in private dictionary; per-case weather/soil indices; original USGS qualifiers and wire IDs",
        "execution": {"network_requests": 0, "cloud_operations": 0, "training_steps": 0, "scaler_fits": 0,
                      "protected_assessment_values_decoded": 0, "source_files_modified": False},
        "qualification": "Development engineering only; approximate independent non-nesting basins; local candidates only; no historical real-time availability, global archive coverage, trained model or forecast-skill claim",
    }
    report["case_source_digest"] = digest({"case_ids": arrays["case_id"].tolist(), "weather_cycles": arrays["source_cycle"].tolist(),
                "soil_cycles": arrays["soil_source_cycle"].tolist(), "history_hours": history_hours, "sources": report["sources"]})
    return arrays, report
