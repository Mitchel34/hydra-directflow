"""Bounded, offline-only development ingestion of one authenticated local cycle.

This reader cannot fetch missing files, hydrate dataless macOS files, or admit
candidate geometry to a frozen study. Its outputs are unscaled physical units.
Historical publication clocks remain unknown; the smoke explicitly simulates
two-hour availability. No scaler fitting, training, or assessment access occurs.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tarfile
from collections import Counter
import time

import numpy as np

from .reference import require

GAUGES = ("03161000", "03164000", "03479000")
CYCLE_ID = "20210823_00Z"
ARCHIVE_SHA256 = "6f34151174311054173b03dc32bcb4080b9244537c24f9a45f7746c6be6cdf00"
WEIGHTS_SHA256 = "9eaaf3b221360ed6faaa0376e953fd0c2d0356fb419df96daed9107404283a74"
USGS_SHA256 = "16f1dcbc063fed46a72ce286571c3b78fc2c199fce24e07aa76cdd03318c6d2b"
USGS_MANIFEST_SHA256 = "4c494b4587c394b1819261f6df0386eba19500a28f0d73aa06543870f933ea2a"
GRID_SHA256 = "f706418ce30b51f44214802eaf8a90b0a671840212c05418865ec15a1eb468b2"
AXIS_SHA256 = {
    "projection_x_coordinate": "993a76cf606f879a6263b3d28e5d79ec0b46aec55518e6ff9baeba68e6e79a07",
    "projection_y_coordinate": "f8b955dbfc985a08b76e5643c2f7fe09a9fc617cf81d341a9b1ee53d9a14c70e",
}
DECODED_AXIS_SHA256 = {
    "projection_x_coordinate": "5c3bbcf48178c5dc34d46063b2946ed39679a10706a0130f56aec08ee8a7308f",
    "projection_y_coordinate": "909b6a5c7c11317424a87acdbd274f015c286bce95fa5117e4d3d3fd243feab0",
}
SOIL_SECTION3_SHA256 = "a3d66291c0a46d57625377318f0404bc2cea9837bf49ae59e28a23a1f8884558"
DEVELOPMENT_QUALIFICATION = "DEVELOPMENT_ONLY_APPROXIMATE_BASINS_SEPARATE_NON_NESTING_GAUGES"
FEATURE_GROUPS = (("precip_mm", "precipitation"), ("temp_k", "temperature"),
                  ("specific_humidity", "humidity"), ("swe_kg_m2", "snow_water_equivalent"))
MAX_BYTES = 16 * 1024 * 1024
CFS_TO_CMS = 0.028316846592


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def utc_epoch(stamp):
    t = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    require(t.tzinfo is not None and t.microsecond == 0, "Explicit integral UTC timestamp required")
    return int(t.timestamp())


def iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def verified_local_bytes(path, expected_sha256, *, expected_bytes=None, max_bytes=MAX_BYTES):
    """Check metadata before opening; reject symlinks and macOS dataless inputs."""
    p = Path(path)
    st = p.lstat()
    require(stat.S_ISREG(st.st_mode), "Input must be a regular local file, never a symlink")
    require(not (getattr(st, "st_flags", 0) & 0x40000000), "Dataless input cannot be hydrated")
    require(0 < st.st_size <= max_bytes, "Input exceeds the explicit local read bound")
    require(expected_bytes is None or st.st_size == expected_bytes, "Input byte count differs")
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None, "An exact SHA256 is required")
    with p.open("rb") as f:
        data = f.read(max_bytes + 1)
    require(len(data) == st.st_size and sha256(data) == expected_sha256, "Input SHA256 mismatch")
    return data


def read_cycle_tar(path, expected_sha256=ARCHIVE_SHA256, *, cycle_id=CYCLE_ID):
    """Never extract; accept only bounded regular members in the cycle whitelist."""
    raw = verified_local_bytes(path, expected_sha256)
    fixed = {"CYCLE_MANIFEST.json", "AVAILABILITY.json", "FIELD_SOURCES.json",
             "RAW_PROVENANCE.json", "POST_FETCH_CACHE_PROVENANCE.json",
             "raw_values.npy", "raw_missing_bits.npy"}
    result = {}
    seen = set()
    total = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for n, member in enumerate(archive):
            require(n < 100, "Archive member count exceeds bound")
            name = member.name
            pure = PurePosixPath(name)
            require(not pure.is_absolute() and ".." not in pure.parts and "\\" not in name,
                    "Unsafe archive path")
            require(name not in seen, "Duplicate archive member")
            seen.add(name)
            if member.isdir():
                require(name in (cycle_id, cycle_id + "/source_bytes"), "Unknown archive directory")
                continue
            require(member.isfile() and not member.issparse(), "Only regular nonsparse archive files allowed")
            require(name.startswith(cycle_id + "/"), "Unexpected cycle member")
            relative = name[len(cycle_id) + 1:]
            require(relative in fixed or re.fullmatch(r"source_bytes/[0-9a-f]{64}\.bin", relative),
                    "Archive member outside whitelist")
            total += member.size
            require(0 <= member.size <= 8 * 1024 * 1024 and total <= MAX_BYTES,
                    "Archive expanded byte bound exceeded")
            stream = archive.extractfile(member)
            require(stream is not None, "Unreadable archive member")
            value = stream.read(member.size + 1)
            require(len(value) == member.size, "Truncated archive member")
            result[relative] = value
    require(fixed <= result.keys(), "Required archive members missing")
    manifest = json.loads(result["CYCLE_MANIFEST.json"])
    require(manifest["cycle"] == cycle_id, "Manifest cycle differs")
    require(set(manifest["files"]) == set(result) - {"CYCLE_MANIFEST.json"},
            "Manifest inventory differs from archive")
    for name, identity in manifest["files"].items():
        require(len(result[name]) == identity["bytes"] and sha256(result[name]) == identity["sha256"],
                "Archive member hash mismatch")
    return result, manifest


def bounded_npy(data, *, shape, dtype):
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    require(version in ((1, 0), (2, 0)), "Unsupported NumPy format")
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    actual_shape, fortran, actual_dtype = reader(stream)
    require(actual_shape == shape and not fortran and actual_dtype == np.dtype(dtype),
            "NumPy shape, order or dtype differs from bounded contract")
    require(len(data) - stream.tell() == math.prod(shape) * actual_dtype.itemsize, "NumPy byte count differs")
    return np.load(io.BytesIO(data), allow_pickle=False)


def _source_bytes(members, source):
    prefix = f"cycles/{CYCLE_ID}/"
    require(source["relative_path"].startswith(prefix), "Source member belongs to a different cycle")
    value = members[source["relative_path"][len(prefix):]]
    require(len(value) == source["bytes"] and sha256(value) == source["sha256"], "Source hash differs")
    return value


def _coordinate(members, field, metadata, name):
    """Decode only bounded 1D clocks/axes, never arbitrary Zarr objects."""
    from numcodecs import Blosc
    spec = metadata[name + "/.zarray"]
    data = _source_bytes(members, field["coordinate_sources"][name])
    shape, dtype = spec["shape"], np.dtype(spec["dtype"])
    require(dtype.str in ("<f8", "<i8") and len(shape) <= 1 and spec["order"] == "C"
            and spec["filters"] is None, "Unsupported coordinate encoding")
    count = math.prod(shape)
    require(0 < count <= 1799, "Coordinate allocation bound exceeded")
    compressor = spec["compressor"]
    if compressor is not None:
        require(compressor["id"] == "blosc" and len(data) >= 16, "Unsupported coordinate compression")
        require(struct.unpack_from("<I", data, 4)[0] == count * dtype.itemsize,
                "Compressed coordinate expansion bound differs")
        data = Blosc().decode(data)
    require(len(data) == count * dtype.itemsize, "Coordinate length differs")
    return np.frombuffer(data, dtype=dtype)


def _verify_grid_and_clocks(members, fields, reference):
    for field in fields.values():
        metadata = json.loads(_source_bytes(members, field["metadata_sources"][0]))["metadata"]
        require(field["spec"]["shape"][1:] == [1059, 1799], "Weather native grid shape differs")
        require(field["spec"]["order"] == "C", "Weather grid scan order differs")
        require(utc_epoch(field["source_reference_UTC"]) == reference, "Field reference time differs")
        for name, expected_hash in AXIS_SHA256.items():
            require(field["coordinate_sources"][name]["sha256"] == expected_hash, "Candidate grid axis hash differs")
            axis = _coordinate(members, field, metadata, name)
            require(sha256(axis.astype("<f8", copy=False).tobytes()) == DECODED_AXIS_SHA256[name],
                    "Decoded candidate grid coordinates differ")
            require(np.all(np.diff(axis) == 3000), "Native grid spacing differs")
        lead = _coordinate(members, field, metadata, "forecast_period")
        valid = _coordinate(members, field, metadata, "time")
        ref = _coordinate(members, field, metadata, "forecast_reference_time")
        require(metadata["forecast_period/.zattrs"]["units"] == "hours", "Forecast period units differ")
        for name in ("time", "forecast_reference_time"):
            attrs = metadata[name + "/.zattrs"]
            require(attrs["units"] == "hours since 1970-01-01" and attrs["calendar"] == "gregorian",
                    "Numeric source clock units differ")
        require(ref.shape == (1,) and ref[0] * 3600 == reference, "Numeric initialization clock differs")
        require(np.array_equal(lead, np.arange(1, len(lead) + 1)), "Numeric source lead axis differs")
        require(np.array_equal(valid * 3600, reference + lead * 3600), "Numeric source valid clock differs")
        require(len(lead) >= 36, "Source does not support the retained extended forecast")


def _verify_soil_grid(data):
    require(data[:4] == b"GRIB" and data[7] == 2, "Expected native soil GRIB2 message")
    require(int.from_bytes(data[8:16], "big") == len(data) and data[-4:] == b"7777", "Soil GRIB length differs")
    pos = 16
    found = []
    while pos < len(data) - 4:
        length = int.from_bytes(data[pos:pos + 4], "big")
        require(length >= 5 and pos + length <= len(data) - 4, "Invalid GRIB section length")
        if data[pos + 4] == 3:
            found.append(sha256(data[pos:pos + length]))
        pos += length
    require(pos == len(data) - 4 and found == [SOIL_SECTION3_SHA256], "Soil grid definition differs from candidate weights")


def _verify_weather_crop(members, fields, records, values, present):
    """Reconstruct the 29x33 crop from authenticated tiles to prove cell order."""
    from numcodecs import Blosc
    for field_id, field in fields.items():
        spec = field["spec"]
        require(spec["chunks"] == [47, 150, 150] and spec["dtype"] == "<f2"
                and spec["filters"] is None and spec["compressor"]["id"] == "blosc",
                "Unexpected bounded source tile encoding")
        require(field["source"]["url"].endswith("/0.3.9"), "Source tile indices differ from candidate crop")
        raw = _source_bytes(members, field["source"])
        expected_size = 47 * 150 * 150 * 2
        require(len(raw) >= 16 and struct.unpack_from("<I", raw, 4)[0] == expected_size,
                "Source tile expansion bound differs")
        decoded = Blosc().decode(raw)
        require(len(decoded) == expected_size, "Decoded source tile size differs")
        tile = np.frombuffer(decoded, dtype="<f2").reshape(47, 150, 150)
        # Tile (j=3, i=9) starts at native (j=450, i=1350).
        crop = tile[:, 28:61, 14:43].reshape(47, 957)
        for index, record in enumerate(records):
            if record.get("field_source_id") != field_id:
                continue
            source = crop[record["lead"] - 1]
            missing = np.isnan(source) | (source == spec["fill_value"])
            require(np.array_equal(~missing, present[index]), "Raw crop masks differ from original source tile")
            require(np.array_equal(source.astype("<f8"), values[index], equal_nan=True),
                    "Raw crop values/cell order differ from original source tile")


@dataclass(frozen=True)
class LocalCycle:
    reference: int
    values: np.ndarray
    present: np.ndarray
    records: tuple[dict, ...]
    member_count: int
    fields: dict
    availability: dict


def load_local_cycle(path):
    members, manifest = read_cycle_tar(path)
    require(manifest["raw_shape"] == [145, 957] and manifest["mask_shape"] == [145, 120]
            and manifest["record_count"] == 145 and manifest["data_complete"] is True,
            "Unexpected bounded cycle shape/completeness")
    require(manifest["assets"]["grid_sha256"] == GRID_SHA256, "Acquisition grid asset differs")
    values = bounded_npy(members["raw_values.npy"], shape=(145, 957), dtype="<f8")
    bits = bounded_npy(members["raw_missing_bits.npy"], shape=(145, 120), dtype="u1")
    missing = np.unpackbits(bits, axis=1, bitorder="little")
    require(not missing[:, 957:].any(), "Nonzero unused mask padding")
    present = ~missing[:, :957].astype(bool)
    require(np.isfinite(values[present]).all(), "Nonfinite admitted weather values")
    records = json.loads(members["RAW_PROVENANCE.json"])
    require(len(records) == 145, "Unexpected record inventory")
    reference = utc_epoch("2021-08-23T00:00:00Z")
    fields = json.loads(members["FIELD_SOURCES.json"])
    availability = json.loads(members["AVAILABILITY.json"])
    require(availability == manifest["availability"] and availability["cycle"] == CYCLE_ID,
            "Original availability metadata differs from manifest")
    _verify_grid_and_clocks(members, fields, reference)
    seen = set()
    units = {"temperature": "K", "humidity": "kg kg-1", "precipitation": "kg m-2", "snow_water_equivalent": "kg m-2"}
    for index, record in enumerate(records):
        group, lead = record["group"], record["lead"]
        require(record["record_index"] == index and (group, lead) not in seen, "Duplicate/reordered source record")
        seen.add((group, lead))
        require(utc_epoch(record["source_reference_UTC"]) == reference
                and utc_epoch(record["source_valid_UTC"]) == reference + lead * 3600,
                "Record forecast valid clock differs")
        require(sha256(values[index].astype("<f8", copy=False).tobytes()) == record["source_values_sha256_float64_le"]
                and sha256(bits[index].tobytes()) == record["missing_bits_sha256"], "Record value/mask hash differs")
        require(int((~present[index]).sum()) == record["missing_cells"], "Record missing count differs")
        if group == "soil_moisture":
            require(lead == 0 and record["native_metadata"]["units"] == "Proportion", "Unexpected soil definition")
            _verify_soil_grid(_source_bytes(members, record["source"]))
        else:
            require(group in units and record["units"] == units[group], "Unknown field or units")
            require(record["APCP_interval_hours"] == ([lead - 1, lead] if group == "precipitation" else None),
                    "Unexpected accumulation interval")
    expected = {(group, lead) for group in units for lead in range(1, 37)} | {("soil_moisture", 0)}
    require(seen == expected, "Incomplete source group/lead support")
    groups = availability["groups"]
    require(len(groups) == 5 and {g["group"] for g in groups} == set(units) | {"soil_moisture"},
            "Original group availability inventory differs")
    for group in groups:
        expected_leads = [0] if group["group"] == "soil_moisture" else list(range(1, 37))
        require(group["leads"] == expected_leads and group["record_count"] == len(expected_leads)
                and group["status"] == "AVAILABLE_VERIFIED", "Original group availability support differs")
    _verify_weather_crop(members, fields, records, values, present)
    return LocalCycle(reference, values, present, tuple(records), len(members), fields, availability)


def hrrr_provenance_arrays(cycle, leads, *, site_count=3):
    """Retain actual source metadata separately from newly declared labels.

    HRRR artifacts do not define a generic quality_flags column. The private
    metadata JSON preserves selected original keys, including null publication
    times, exact availability statuses, and original record/source identities.
    Repetition across gauges explicitly reflects reuse of one source crop.
    """
    records = {(r["group"], r["lead"]): r for r in cycle.records}
    groups = {g["group"]: g for g in cycle.availability["groups"]}
    record_keys = ("group", "lead", "record_index", "source_route", "source_dtype",
                   "source_reference_UTC", "source_valid_UTC", "source_values_sha256_float64_le",
                   "missing_bits_sha256", "missing_cells")

    def original_metadata(record):
        original = {key: record[key] for key in record_keys}
        original["group_availability"] = {key: groups[record["group"]][key]
                                          for key in ("status", "source_key", "record_count", "leads")}
        if record["group"] == "soil_moisture":
            original.update({key: record[key] for key in
                             ("availability_status", "historical_available_at_UTC", "soil_definition")})
            original["native_units"] = record["native_metadata"]["units"]
            original["source_identity"] = {key: record["source"][key] for key in ("key", "sha256", "bytes")}
        else:
            original.update({key: record[key] for key in
                             ("field_source_id", "storage_dtype", "units", "APCP_interval_hours")})
            field = cycle.fields[record["field_source_id"]]
            original["field_availability"] = {key: field[key] for key in
                                              ("availability_status", "historical_available_at_UTC")}
            original["source_identity"] = {key: field["source"][key] for key in ("key", "sha256", "bytes")}
        return json.dumps(original, sort_keys=True, separators=(",", ":"), allow_nan=False)

    selected = [[records[group, lead] for _, group in FEATURE_GROUPS] for lead in leads]
    weather_ids = [[record["field_source_id"] for record in row] for row in selected]
    weather_indices = [[record["record_index"] for record in row] for row in selected]
    weather_metadata = [[original_metadata(record) for record in row] for row in selected]
    soil = records["soil_moisture", 0]
    return {
        "weather_source_ids": np.array([weather_ids] * site_count),
        "weather_record_indices": np.array([weather_indices] * site_count, dtype=np.int64),
        "weather_source_metadata": np.array([weather_metadata] * site_count),
        "weather_development_labels": np.array([DEVELOPMENT_QUALIFICATION, "RETROSPECTIVE_SIMULATED_LAG_2H"]),
        "soil_source_ids": np.array([soil["source"]["key"]] * site_count),
        "soil_record_indices": np.array([soil["record_index"]] * site_count, dtype=np.int64),
        "soil_source_metadata": np.array([original_metadata(soil)] * site_count),
        "soil_development_labels": np.array([DEVELOPMENT_QUALIFICATION, "RETROSPECTIVE_SIMULATED_LAG_2H"]),
    }


def load_candidate_weights(path, *, qualification):
    require(qualification == DEVELOPMENT_QUALIFICATION, "Explicit development geometry qualification required")
    raw = verified_local_bytes(path, WEIGHTS_SHA256, expected_bytes=62601, max_bytes=128 * 1024)
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    require(len(rows) == 514, "Candidate weight row count differs")
    weights = np.zeros((3, 957), dtype=np.float64)
    seen = set()
    for row in rows:
        site, index = row["gauge_id"], int(row["crop_index"])
        i, j = int(row["native_i"]), int(row["native_j"])
        require(site in GAUGES and 0 <= index < 957 and (site, index) not in seen, "Invalid/duplicate weighted cell")
        require(1364 <= i <= 1392 and 478 <= j <= 510 and index == (j - 478) * 29 + i - 1364
                and int(row["native_flat_index"]) == j * 1799 + i, "Candidate weights grid indices differ")
        require(row["status"] == "CANDIDATE_SCIENTIFIC_HOLD" and row["admitted"] == "false",
                "Original candidate admission metadata changed")
        weight = float(row["weight"])
        require(math.isfinite(weight) and weight > 0, "Invalid candidate weight")
        weights[GAUGES.index(site), index] = weight
        seen.add((site, index))
    require(np.all(np.abs(weights.sum(axis=1) - 1) <= 1e-10), "Full-basin candidate weights do not sum to one")
    return weights


def basin_reduce(values, present, weights):
    """Missing positive-weight cells mask the mean; never renormalize coverage."""
    require(values.shape == present.shape and values.shape[-1] == weights.shape[-1], "Spatial reduction shape differs")
    require(present.dtype == np.dtype(bool), "Spatial masks must be booleans")
    require(np.isfinite(values[present]).all(), "Nonfinite observed value")
    coverage = present.astype(float) @ weights.T
    complete = np.stack([np.all(present[..., row > 0], axis=-1) for row in weights], axis=-1)
    mean = np.where(present, values, 0.0) @ weights.T
    return np.where(complete, mean, np.nan), complete, coverage


def load_usgs_window(path, manifest_path, *, start, end):
    """Only the pinned development partition and requested UTC window are decoded."""
    from pyarrow import BufferReader
    import pyarrow.parquet as pq
    require(utc_epoch("2021-08-01T00:00:00Z") <= start <= end < utc_epoch("2021-09-01T00:00:00Z"),
            "This offline smoke reader is limited to August 2021")
    require(end - start <= 200 * 3600, "USGS requested window exceeds the bounded development chunk")
    manifest_raw = verified_local_bytes(manifest_path, USGS_MANIFEST_SHA256, max_bytes=2 * 1024 * 1024)
    manifest = json.loads(manifest_raw)
    matches = [p for p in manifest["native_partitions"] if Path(p["path"]).name == "2021-08.parquet"]
    require(len(matches) == 1 and matches[0]["sha256"] == USGS_SHA256, "USGS partition identity differs")
    raw = verified_local_bytes(path, USGS_SHA256, expected_bytes=matches[0]["bytes"], max_bytes=128 * 1024)
    columns = ["site", "sample_epoch", "sample_time_utc", "source_units", "discharge_cfs",
               "source_qualifiers_json", "numeric_valid", "is_exact_utc_hour", "is_core",
               "source_wire_sha256", "historical_available_at_utc"]
    rows = pq.read_table(BufferReader(raw), columns=columns,
                         filters=[("sample_epoch", ">=", start), ("sample_epoch", "<=", end)]).to_pylist()
    require(len(rows) <= 2500, "USGS decoded row bound exceeded")
    exact = []
    seen = set()
    for row in rows:
        t = row["sample_epoch"]
        require(start <= t <= end and utc_epoch(row["sample_time_utc"]) == t, "USGS sample clock differs")
        require(row["site"] in GAUGES and row["source_units"] == "ft3/s", "USGS site/units differ")
        require(row["historical_available_at_utc"] is None, "Unexpected documented availability vintage")
        require(row["is_exact_utc_hour"] == (t % 3600 == 0), "USGS exact-hour indicator differs")
        if row["is_core"] and row["is_exact_utc_hour"]:
            key = row["site"], t
            require(key not in seen, "Duplicate USGS exact-hour record")
            seen.add(key)
            exact.append(row)
    return exact, len(rows)


def build_development_smoke(*, archive_path, weights_path, usgs_path, usgs_manifest_path,
                            origin_utc="2021-08-23T02:00:00Z", history_hours=24):
    """Return private arrays plus a portable aggregate receipt for ONE origin."""
    from .contract import DevelopmentProtocol, TensorSpec
    from .reference import (AvailabilityPolicy, Observation, SoilInitialState, WeatherRun,
                            history_window, latest_weather_run, soil_state, target_window)
    started = time.monotonic()
    protocol = DevelopmentProtocol(history_hours=history_hours)
    origin = utc_epoch(origin_utc)
    require(origin % 3600 == 0 and utc_epoch("2021-08-23T02:00:00Z") <= origin <= utc_epoch("2021-08-23T07:00:00Z"),
            "This one-cycle development reader admits only August 23 origins 02Z through 07Z")
    policy = AvailabilityPolicy("development_simulated_lag_2h_v1", "retrospective_simulated", 7200)
    cycle = load_local_cycle(archive_path)
    weights = load_candidate_weights(weights_path, qualification=DEVELOPMENT_QUALIFICATION)
    means, masks, coverage = basin_reduce(cycle.values, cycle.present, weights)
    row_index = {(r["group"], r["lead"]): index for index, r in enumerate(cycle.records)}
    history_start = origin - (history_hours + protocol.history_lag_hours - 1) * 3600
    target_times = tuple(origin + h * 3600 for h in protocol.horizons_hours)
    rows, native_rows = load_usgs_window(usgs_path, usgs_manifest_path, start=history_start, end=target_times[-1])
    observations = []
    for row in rows:
        value = row["discharge_cfs"]
        usable = row["numeric_valid"] and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
        flags = tuple(json.loads(row["source_qualifiers_json"]))
        observations.append(Observation(row["site"], row["sample_epoch"], None,
                                        float(value) * CFS_TO_CMS if usable else None,
                                        flags, "usgs-wire-sha256:" + row["source_wire_sha256"]))
    strict_a = lambda flags: flags == ("A",)
    histories, targets, soils, selected = [], [], [], []
    valid_times = tuple(cycle.reference + lead * 3600 for lead in range(1, 37))
    source_flags = tuple(sorted({f["availability_status"] for f in cycle.fields.values()}
                                | {g["status"] for g in cycle.availability["groups"]
                                   if g["group"] != "soil_moisture"}))
    for site_index, site in enumerate(GAUGES):
        feature_support = tuple((feature, tuple(bool(masks[row_index[group, lead], site_index])
                                                for lead in range(1, 37)))
                                for feature, group in FEATURE_GROUPS)
        run = WeatherRun(CYCLE_ID, cycle.reference, None, valid_times, feature_support, source_flags)
        chosen = latest_weather_run((run,), cutoff=origin, origin_anchor=origin, policy=policy,
                                    horizons_hours=protocol.horizons_hours,
                                    required_features=tuple(f for f, _ in FEATURE_GROUPS),
                                    allow_older_extended_cycles=protocol.allow_older_extended_cycles)
        require(chosen is not None, "No single cycle supplies every requested weather feature/target")
        selected.append(chosen.run_id)
        histories.append(history_window(observations, site=site, origin_anchor=origin, cutoff=origin,
                                         lookback_hours=history_hours, history_lag_hours=2, policy=policy,
                                         admit_quality=strict_a))
        targets.append(target_window(observations, site=site, origin_anchor=origin,
                                     horizons_hours=protocol.horizons_hours, admit_quality=strict_a))
        soil_index = row_index["soil_moisture", 0]
        soil_record = cycle.records[soil_index]
        soil_group = next(g for g in cycle.availability["groups"] if g["group"] == "soil_moisture")
        state = SoilInitialState(soil_record["source"]["key"], cycle.reference, None,
                                 float(means[soil_index, site_index]) if masks[soil_index, site_index] else None,
                                 (soil_record["availability_status"], soil_group["status"]))
        soils.append(soil_state((state,), cutoff=origin, policy=policy,
                                max_age_seconds=protocol.max_soil_age_hours * 3600))
    # None becomes NaN only in the private physical-unit artifact, with its mask.
    history = np.array([list(zip(h["values"], h["age_hours"])) for h in histories], dtype=np.float64)
    history_mask = np.repeat(np.array([h["mask"] for h in histories], dtype=bool)[..., None], 2, axis=2)
    leads = tuple((t - cycle.reference) // 3600 for t in target_times)
    weather = np.array([[[means[row_index[group, lead], site_index] for _, group in FEATURE_GROUPS]
                         for lead in leads] for site_index in range(3)], dtype=np.float64)
    weather_mask = np.array([[[masks[row_index[group, lead], site_index] for _, group in FEATURE_GROUPS]
                              for lead in leads] for site_index in range(3)], dtype=bool)
    state = np.array([[s["value"], s["age_hours"] if s["mask"] else None] for s in soils], dtype=np.float64)
    state_mask = np.repeat(np.array([s["mask"] for s in soils], dtype=bool)[:, None], 2, axis=1)
    tensors = dict(history=history, history_mask=history_mask, weather=weather, weather_mask=weather_mask,
                   state=state, state_mask=state_mask, site_id=np.arange(3, dtype=np.int64),
                   target=np.array([t["values"] for t in targets], dtype=np.float64),
                   target_mask=np.array([t["mask"] for t in targets], dtype=bool),
                   history_time=np.array(histories[0]["times"], dtype=np.int64),
                   target_time=np.array(target_times, dtype=np.int64),
                   history_quality_flags=np.array([[json.dumps(f) for f in h["quality_flags"]] for h in histories]),
                   target_quality_flags=np.array([[json.dumps(f) for f in t["quality_flags"]] for t in targets]),
                   history_source_ids=np.array([[s or "" for s in h["source_ids"]] for h in histories]),
                   target_source_ids=np.array([[s or "" for s in t["source_ids"]] for t in targets]))
    tensors.update(hrrr_provenance_arrays(cycle, leads))
    spec = TensorSpec(history_hours, protocol.horizons_hours, ("discharge_cms", "age_hours"),
                      tuple(f for f, _ in FEATURE_GROUPS), ("soil_water_fraction", "soil_age_hours"), GAUGES)
    expected_target = spec.validate_shapes(**{key: tensors[key].shape for key in
        ("history", "history_mask", "weather", "weather_mask", "state", "state_mask", "site_id")})
    require(tensors["target"].shape == tensors["target_mask"].shape == expected_target, "Target tensor shape differs")
    all_complete = all(tensors[key].all() for key in ("history_mask", "weather_mask", "state_mask", "target_mask"))
    report = {
        "schema": "hydra-directflow-local-smoke/v1",
        "status": "PASSED_DEVELOPMENT_SMOKE" if all_complete else "COMPLETED_WITH_EXPLICIT_MASKS",
        "scope": "ONE_REAL_LOCAL_ORIGIN_THREE_GAUGES_NO_TRAINING",
        "artifact_created_at_utc": iso(int(time.time())),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "origin_utc": iso(origin), "source_cycle": CYCLE_ID,
        "source_reference_utc": iso(cycle.reference), "source_leads_hours": list(leads),
        "target_start_utc": iso(target_times[0]), "target_end_utc": iso(target_times[-1]),
        "history_start_utc": iso(history_start), "history_end_utc": iso(origin - 7200),
        "history_hours": history_hours, "horizons_hours": list(protocol.horizons_hours), "gauge_ids": list(GAUGES),
        "availability": {"mode": policy.mode, "policy_id": policy.policy_id,
                         "lag_seconds": 7200, "historical_first_publication": "UNKNOWN",
                         "retrieval_time_used_as_availability": False,
                         "forecast_delivery_or_inference_completion": "NOT_RUN"},
        "geometry": {"qualification": DEVELOPMENT_QUALIFICATION, "scientific_admission": False,
                     "original_candidate_status_unchanged": True, "weights_sha256": WEIGHTS_SHA256,
                     "sparse_rows": 514, "matrix_shape": [3, 957], "cell_order": "j-major i-minor",
                     "weight_sums": weights.sum(axis=1).tolist(), "renormalized": False,
                     "missing_cell_policy": "MASK_IF_ANY_POSITIVE_WEIGHT_CELL_IS_MISSING",
                     "coverage_min": float(coverage.min()), "coverage_max": float(coverage.max()),
                     "grid_asset_sha256": GRID_SHA256, "soil_section3_sha256": SOIL_SECTION3_SHA256,
                     "axis_source_sha256": AXIS_SHA256, "decoded_axis_sha256": DECODED_AXIS_SHA256,
                     "limitations": ["Approximate study-basin polygons; no surveyed outlet claim",
                                     "Jefferson and Galax treated separately; no nesting or differencing claim",
                                     "CRS binding corroborated; original provider Zarr CRS attribute absent"]},
        "sources": {"hrrr": {"logical_id": CYCLE_ID, "sha256": ARCHIVE_SHA256,
                              "bytes": 5150720, "regular_members_verified": cycle.member_count,
                              "record_count": len(cycle.records), "grid_cells": 957,
                              "numeric_forecast_clocks_verified": True, "native_grid_verified": True,
                              "weather_crop_redecoded_from_original_source_tiles": True,
                              "soil_values_verified_against_original_record_hash": True},
                    "usgs": {"logical_id": "native/2021-08.parquet", "sha256": USGS_SHA256,
                             "manifest_sha256": USGS_MANIFEST_SHA256, "native_rows_in_requested_window": native_rows,
                             "exact_hour_rows_in_requested_window": len(rows)}},
        "features": {"history": list(spec.history_features), "weather": list(spec.weather_features),
                     "state": list(spec.state_features), "target": "discharge_cms",
                     "cfs_to_cms": CFS_TO_CMS, "quality_policy": "STRICT_EXACT_A_ONLY",
                     "transform": "PHYSICAL_UNITS_UNSCALED", "soil_definition": "modeled nominal 1cm node",
                     "precipitation": "original one-hour accumulation; kg/m2 numerically equals mm; no differencing"},
        "tensors": {key: {"shape": list(tensors[key].shape), "mask_true": int(tensors[mask].sum()),
                          "mask_total": int(tensors[mask].size)}
                    for key, mask in (("history", "history_mask"), ("weather", "weather_mask"),
                                      ("state", "state_mask"), ("target", "target_mask"))},
        "source_quality_counts": dict(Counter(row["source_qualifiers_json"] for row in rows)),
        "provenance_retention": {
            "usgs_original_qualifiers_and_source_wire_ids_retained": True,
            "hrrr_generic_quality_flags_column_exists": False,
            "hrrr_original_metadata_retained": ["group availability status and source key",
                                                "publication availability status and null timestamp",
                                                "field source ID or native soil source key", "record index",
                                                "source route and dtype", "source units and accumulation interval",
                                                "reference and valid clocks", "record values and mask hashes",
                                                "original source bytes SHA256 and byte count", "original missing cell count"],
            "weather_metadata_alignment": "gauge x target hour x weather feature",
            "soil_metadata_alignment": "gauge",
            "development_labels_stored_separately_from_original_metadata": True,
        },
        "selected_single_cycle_per_gauge": selected,
        "execution": {"network_requests": 0, "cloud_operations": 0, "training_steps": 0,
                      "scaler_fits": 0, "assessment_rows_read": 0, "source_files_modified": False},
        "qualification": "Development preprocessing receipt only; no trained model, forecast skill, frozen study, or historical real-time availability claim",
    }
    return tensors, report
