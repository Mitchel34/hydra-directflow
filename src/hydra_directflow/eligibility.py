"""Metadata-only development eligibility and local source inventory.

USGS numeric_valid is retained acquisition QA, not a recomputation from flow.
An unread monthly partition makes its windows UNKNOWN. Sealed HRRR summaries
establish structural support only; numeric basin masks still need validation.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import stat
import time

from .local_data import GAUGES, USGS_MANIFEST_SHA256, iso, verified_local_bytes
from .multicycle import INDEX_BYTES, INDEX_SHA256
from .quality import PRIMARY_APPROVED, quality_admitter, quality_policy_binding
from .reference import ASSESSMENT_START, require

HOUR = 3600
DEVELOPMENT_START = int(datetime(2018, 7, 12, tzinfo=timezone.utc).timestamp())
DEVELOPMENT_END = ASSESSMENT_START
METADATA_COLUMNS = ("site", "sample_epoch", "sample_time_utc", "is_exact_utc_hour",
                    "numeric_valid", "source_qualifiers_json", "is_core")
WEATHER_GROUPS = frozenset(("temperature", "humidity", "precipitation", "snow_water_equivalent"))
ELIGIBLE, INELIGIBLE, UNKNOWN = "ELIGIBLE", "INELIGIBLE", "UNKNOWN"
MAX_PARTITION_BYTES = 512 * 1024


def utc_hour(stamp):
    require(type(stamp) is str, "Explicit UTC timestamp required")
    value = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    require(value.tzinfo is not None and value.utcoffset().total_seconds() == 0
            and value.minute == value.second == value.microsecond == 0, "Exact UTC hour required")
    return int(value.timestamp())


def month_bounds(month):
    require(type(month) is str and re.fullmatch(r"\d{4}-\d{2}", month), "YYYY-MM month required")
    year, number = map(int, month.split("-"))
    start = datetime(year, number, 1, tzinfo=timezone.utc)
    end = datetime(year + (number == 12), 1 if number == 12 else number + 1, 1, tzinfo=timezone.utc)
    return int(start.timestamp()), int(end.timestamp())


def months_between(start, end):
    require(type(start) is int and type(end) is int and start < end, "Nonempty integer interval required")
    first = datetime.fromtimestamp(start, timezone.utc).strftime("%Y-%m")
    result = []
    while True:
        result.append(first)
        _, following = month_bounds(first)
        if following >= end:
            return result
        first = datetime.fromtimestamp(following, timezone.utc).strftime("%Y-%m")


def validate_splits(splits):
    require(isinstance(splits, (list, tuple)) and 1 <= len(splits) <= 32, "One to 32 named development splits required")
    output, seen = [], set()
    for split in splits:
        require(type(split) is dict and set(split) == {"name", "start_utc", "end_utc"}, "Split keys must be name/start_utc/end_utc")
        name = split["name"]
        require(type(name) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name) and name not in seen,
                "Unique portable split name required")
        seen.add(name)
        start, end = utc_hour(split["start_utc"]), utc_hour(split["end_utc"])
        require(DEVELOPMENT_START <= start < end <= DEVELOPMENT_END, "Split must stay inside the pre-assessment development interval")
        output.append({"name": name, "start": start, "end": end})
    return output


def inspect_local(path, *, expected_bytes=None):
    if path is None:
        return {"status": "NOT_SUPPLIED", "bytes": None}
    try:
        info = Path(path).lstat()
    except FileNotFoundError:
        return {"status": "MISSING", "bytes": None}
    if not stat.S_ISREG(info.st_mode):
        return {"status": "NONREGULAR_NOT_ADMITTED", "bytes": info.st_size}
    if getattr(info, "st_flags", 0) & 0x40000000:
        return {"status": "DATALESS_NOT_READ", "bytes": info.st_size}
    if expected_bytes is not None and info.st_size != expected_bytes:
        return {"status": "RESIDENT_SIZE_MISMATCH", "bytes": info.st_size}
    return {"status": "RESIDENT_UNVERIFIED_SHA", "bytes": info.st_size}


def metadata_rows(rows, month):
    """Index only exact core observations; preserve duplicate qualifier tokens."""
    start, end = month_bounds(month)
    exact = {site: {} for site in GAUGES}
    seen = set()
    profiles = {site: Counter() for site in GAUGES}
    qualifiers = {site: Counter() for site in GAUGES}
    for row in rows:
        require(set(row) == set(METADATA_COLUMNS), "Only the declared metadata columns may enter this audit")
        site, stamp = row["site"], row["sample_epoch"]
        require(site in GAUGES and type(stamp) is int and (site, stamp) not in seen, "Unknown site or duplicate/nonintegral native timestamp")
        seen.add((site, stamp))
        parsed = datetime.fromisoformat(row["sample_time_utc"].replace("Z", "+00:00"))
        require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0
                and parsed.microsecond == 0 and int(parsed.timestamp()) == stamp, "Native source clocks differ")
        for flag in ("is_exact_utc_hour", "numeric_valid", "is_core"):
            require(type(row[flag]) is bool, "Boolean source metadata required")
        require(row["is_exact_utc_hour"] == (stamp % HOUR == 0), "Exact-hour source marker differs")
        flags = json.loads(row["source_qualifiers_json"])
        require(type(flags) is list and all(type(flag) is str for flag in flags), "Original qualifiers must be a JSON array of strings")
        profiles[site]["native_rows_returned"] += 1
        if not row["is_core"]:
            profiles[site]["padding_rows_returned"] += 1
            continue
        require(start <= stamp < end, "Core observation lies outside its native month")
        if stamp >= DEVELOPMENT_END:
            continue  # No protected-period metadata enters eligibility calculations.
        profiles[site]["core_native_rows_returned"] += 1
        if row["is_exact_utc_hour"]:
            exact[site][stamp] = {"numeric_valid": row["numeric_valid"], "qualifiers": tuple(flags)}
            qualifiers[site][row["source_qualifiers_json"]] += 1
            profiles[site]["exact_core_rows"] += 1
            profiles[site]["numeric_valid_false"] += int(not row["numeric_valid"])
    report = {}
    for site in GAUGES:
        expected = max(0, min(end, DEVELOPMENT_END) - start) // HOUR
        report[site] = {**profiles[site], "expected_exact_core_hours_before_assessment": expected,
                        "missing_exact_core_hours": expected - len(exact[site]),
                        "exact_qualifier_counts_verbatim": dict(sorted(qualifiers[site].items()))}
    return exact, report


def inventory_usgs(manifest_path, paths=None):
    """Hash resident monthly files, then project metadata columns only."""
    from pyarrow import BufferReader
    import pyarrow.parquet as pq
    raw = verified_local_bytes(manifest_path, USGS_MANIFEST_SHA256, max_bytes=2 * 1024**2)
    manifest = json.loads(raw)
    native = {}
    for entry in manifest["native_partitions"]:
        month = Path(entry["path"]).stem
        require(month not in native, "Duplicate USGS month in source manifest")
        native[month] = entry
    overrides = dict(paths or {})
    required = months_between(DEVELOPMENT_START - 169 * HOUR, DEVELOPMENT_END)
    known_months, exact, inventory = set(), {site: {} for site in GAUGES}, []
    for month in required:
        identity = native.get(month)
        if identity is None:
            inventory.append({"month": month, "status": "ABSENT_FROM_PINNED_MANIFEST", "path": None,
                              "sha256": None, "expected_bytes": None, "metadata_known": False})
            continue
        path = overrides.get(month, identity["path"])
        state = inspect_local(path, expected_bytes=identity["bytes"])
        item = {"month": month, "path": str(path), "sha256": identity["sha256"],
                "expected_bytes": identity["bytes"], "metadata_known": False, **state}
        if state["status"] == "RESIDENT_UNVERIFIED_SHA":
            try:
                body = verified_local_bytes(path, identity["sha256"], expected_bytes=identity["bytes"], max_bytes=MAX_PARTITION_BYTES)
            except ValueError:
                item["status"] = "RESIDENT_IDENTITY_CHECK_FAILED"
            else:
                # Projection excludes discharge_cfs/source_value and every other value column.
                # Filters are logical: Parquet may decode other rows of these metadata columns.
                start, end = month_bounds(month)
                table = pq.read_table(BufferReader(body), columns=list(METADATA_COLUMNS),
                                      filters=[("sample_epoch", ">=", start), ("sample_epoch", "<", min(end, DEVELOPMENT_END))],
                                      use_threads=False)
                require(tuple(table.column_names) == METADATA_COLUMNS and table.num_rows <= 10000,
                        "Metadata projection or row bound differs")
                indexed, profile = metadata_rows(table.to_pylist(), month)
                for site in GAUGES:
                    require(not (set(exact[site]) & set(indexed[site])), "Duplicate exact core time across native months")
                    exact[site].update(indexed[site])
                known_months.add(month)
                item.update(status="VERIFIED_METADATA_READ", metadata_known=True, profile=profile,
                            returned_metadata_rows=table.num_rows)
        inventory.append(item)
    return exact, known_months, inventory


def required_six_hour_cycles(start=DEVELOPMENT_START, end=DEVELOPMENT_END):
    """Sources for H18-purged origins: age at least two and at most 18 hours."""
    require(start % HOUR == end % HOUR == 0 and start < end, "Exact hourly interval required")
    last_origin = end - 19 * HOUR
    if last_origin < start:
        return []
    first = ((start - 18 * HOUR + 6 * HOUR - 1) // (6 * HOUR)) * 6 * HOUR
    last = ((last_origin - 2 * HOUR) // (6 * HOUR)) * 6 * HOUR
    return [datetime.fromtimestamp(value, timezone.utc).strftime("%Y%m%d_%HZ")
            for value in range(first, last + 1, 6 * HOUR)]


def read_required_index(path, cycle_ids):
    path = Path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not getattr(before, "st_flags", 0) & 0x40000000
            and before.st_size == INDEX_BYTES, "Sealed index must be the exact resident file")
    wanted, rows, hasher, count = set(cycle_ids), {}, hashlib.sha256(), 0
    with path.open("rb") as stream:
        while line := stream.readline(65537):
            require(len(line) <= 65536 and line.endswith(b"\n"), "Index line encoding bound exceeded")
            hasher.update(line)
            count += 1
            require(count <= 61824, "Index row bound exceeded")
            row = json.loads(line)
            if row["cycle"] in wanted:
                require(row["cycle"] not in rows, "Duplicate requested index cycle")
                rows[row["cycle"]] = row
    after = path.lstat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "Index changed during read")
    require(count == 61824 and hasher.hexdigest() == INDEX_SHA256, "Sealed index digest differs")
    # Warmup can precede the acquired calendar. Keep absent identities UNKNOWN.
    return rows


def cycle_epoch(cycle_id):
    require(type(cycle_id) is str and re.fullmatch(r"\d{8}_\d{2}Z", cycle_id), "Explicit source cycle ID required")
    return int(datetime.strptime(cycle_id, "%Y%m%d_%HZ").replace(tzinfo=timezone.utc).timestamp())


def group_support(row, required):
    if row is None:
        return UNKNOWN
    groups = row.get("groups_available")
    if groups is not None:
        require(type(groups) is list and all(type(group) is str for group in groups), "Invalid sealed group summary")
        return ELIGIBLE if set(required) <= set(groups) else INELIGIBLE
    missing = row.get("groups_missing", {})
    if set(required) & set(missing):
        return INELIGIBLE
    return UNKNOWN  # Legacy complete flag alone is not a decoded group/lead contract.


def _bundle_identity(value):
    if not isinstance(value, dict):
        return None
    if not (type(value.get("bytes")) is int and 0 < value["bytes"] <= 16 * 1024**2
            and re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", ""))
            and type(value.get("durable_uri")) is str and value["durable_uri"].startswith("gs://")
            and str(value.get("generation", "")).isdigit()):
        return None
    return {key: value[key] for key in ("bytes", "sha256", "durable_uri", "generation")}


def inventory_hrrr(rows, cycle_ids, local_archive_specs=None):
    """Read bounded resident commit JSON only; never open weather payloads."""
    supplied = {}
    for spec in local_archive_specs or []:
        cycle = spec["cycle_id"]
        supplied.setdefault(cycle, []).append(spec)
    inventory = []
    for cycle in cycle_ids:
        row = rows.get(cycle)
        item = {"cycle_id": cycle, "reference_epoch": cycle_epoch(cycle),
                "index_status": "PRESENT" if row else "ABSENT_WARMUP_OR_CALENDAR_UNKNOWN",
                "data_complete": row.get("data_complete") if row else None,
                "weather_group_support": group_support(row, WEATHER_GROUPS),
                "soil_group_support": group_support(row, {"soil_moisture"}),
                "source_locations": row.get("source_locations", {}) if row else {},
                "local_candidates": [], "bundle_identity": None, "bundle_bytes_source": "UNKNOWN"}
        if row:
            direct = _bundle_identity(row["source_locations"].get("bundle"))
            if direct:
                item.update(bundle_identity=direct, bundle_bytes_source="SEALED_INDEX")
        for spec in supplied.get(cycle, []):
            candidate = {"archive_path": spec.get("archive_path"), "commit_path": spec.get("commit_path")}
            commit_identity = row.get("source_locations", {}).get("commit") if row else None
            if commit_identity and spec.get("commit_path"):
                state = inspect_local(spec["commit_path"], expected_bytes=commit_identity.get("bytes"))
                candidate["commit"] = state
                if state["status"] == "RESIDENT_UNVERIFIED_SHA":
                    try:
                        raw = verified_local_bytes(spec["commit_path"], commit_identity["sha256"],
                                                   expected_bytes=commit_identity.get("bytes"), max_bytes=16384)
                    except ValueError:
                        candidate["commit"]["status"] = "RESIDENT_IDENTITY_CHECK_FAILED"
                    else:
                        commit = json.loads(raw)
                        require(commit.get("cycle") == cycle, "Authenticated commit belongs to another cycle")
                        bundle = _bundle_identity(commit.get("bundle"))
                        require(bundle is not None, "Authenticated commit lacks a bounded bundle identity")
                        require(item["bundle_identity"] in (None, bundle), "Sealed bundle identities conflict")
                        item.update(bundle_identity=bundle, bundle_bytes_source="AUTHENTICATED_LOCAL_COMMIT")
                        candidate["commit"]["status"] = "VERIFIED_COMMIT_METADATA"
            else:
                candidate["commit"] = inspect_local(spec.get("commit_path"))
            candidate["archive"] = inspect_local(spec.get("archive_path"),
                expected_bytes=item["bundle_identity"]["bytes"] if item["bundle_identity"] else None)
            item["local_candidates"].append(candidate)
        states = [entry["archive"]["status"] for entry in item["local_candidates"]]
        item["local_archive_status"] = ("RESIDENT_STAT_ONLY_PAYLOAD_NOT_READ" if "RESIDENT_UNVERIFIED_SHA" in states else
                                        "DATALESS_NOT_READ" if "DATALESS_NOT_READ" in states else
                                        "MISSING_OR_NOT_ADMITTED" if states else "NO_LOCAL_PATH_SUPPLIED")
        inventory.append(item)
    return inventory


def combine_status(*values):
    require(values and all(value in (ELIGIBLE, INELIGIBLE, UNKNOWN) for value in values), "Explicit three-state eligibility required")
    if UNKNOWN in values:
        return UNKNOWN
    return ELIGIBLE if all(value == ELIGIBLE for value in values) else INELIGIBLE


def prefix(values):
    result = [0]
    for value in values:
        result.append(result[-1] + int(value))
    return result


def window_status(known, admitted, first, after):
    require(0 <= first < after < len(known) and len(known) == len(admitted), "Window leaves metadata axes")
    size = after - first
    if known[after] - known[first] != size:
        return UNKNOWN
    return ELIGIBLE if admitted[after] - admitted[first] == size else INELIGIBLE


def build_prefixes(exact, known_months, start, end):
    require(start % HOUR == end % HOUR == 0 and start < end, "Exact hourly metadata axis required")
    times = range(start, end, HOUR)
    known = [datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m") in known_months for value in times]
    admit = quality_admitter(PRIMARY_APPROVED)
    admitted = {}
    for site in GAUGES:
        admitted[site] = prefix(bool((row := exact[site].get(value)) is not None
                                    and row["numeric_valid"] and admit(row["qualifiers"])) for value in times)
    return prefix(known), admitted


def hrrr_structural_status(origin, by_reference):
    def available_support(max_age, key):
        references = range(((origin - max_age * HOUR + 6 * HOUR - 1) // (6 * HOUR)) * 6 * HOUR,
                           ((origin - 2 * HOUR) // (6 * HOUR)) * 6 * HOUR + 1, 6 * HOUR)
        states = [by_reference.get(reference, {}).get(key, UNKNOWN) for reference in references]
        if ELIGIBLE in states:
            return ELIGIBLE
        return UNKNOWN if not states or UNKNOWN in states else INELIGIBLE
    weather = available_support(18, "weather_group_support")
    soil = available_support(8, "soil_group_support")
    return weather, soil


def count_split(split, exact, known_months, hrrr_inventory, *, history_hours):
    require(history_hours in (24, 168), "L24 or L168 required")
    start, end = split["start"], split["end"]
    origins = range(start, max(start, end - 18 * HOUR), HOUR)
    axis_start = start - (history_hours + 1) * HOUR
    known, admitted = build_prefixes(exact, known_months, axis_start, end)
    hrrr = {item["reference_epoch"]: item for item in hrrr_inventory}
    structure = {origin: hrrr_structural_status(origin, hrrr) for origin in origins}
    output = []
    for site in GAUGES:
        counts = {name: Counter({ELIGIBLE: 0, INELIGIBLE: 0, UNKNOWN: 0}) for name in
                  ("input_history_metadata", "target_metadata", "joint_usgs_metadata", "weather_structural_support", "soil_structural_support", "usgs_and_hrrr_structure")}
        for origin in origins:
            center = (origin - axis_start) // HOUR
            history = window_status(known, admitted[site], center - history_hours - 1, center - 1)
            target = window_status(known, admitted[site], center + 1, center + 19)
            joint = combine_status(history, target)
            weather, soil = structure[origin]
            values = {"input_history_metadata": history, "target_metadata": target, "joint_usgs_metadata": joint,
                      "weather_structural_support": weather, "soil_structural_support": soil,
                      "usgs_and_hrrr_structure": combine_status(joint, weather, soil)}
            for name, value in values.items():
                counts[name][value] += 1
        requested = (end - start) // HOUR
        record = {"split": split["name"], "gauge": site, "history_hours": history_hours,
                  "requested_origins": requested, "purged_h18_boundary_origins": requested - len(origins),
                  "candidate_origins_after_purge": len(origins),
                  **{name: dict(value) for name, value in counts.items()},
                  "full_model_numeric_input_eligibility": "NOT_ESTABLISHED_REQUIRES_HRRR_BASIN_MASKS_AND_NUMERIC_SOURCE_QA"}
        require(all(sum(value.values()) == len(origins) for value in counts.values()), "Eligibility states do not partition candidate origins")
        output.append(record)
    return output


def audit_development_eligibility(*, qualified_index_path, usgs_manifest_path, splits,
                                  usgs_paths=None, local_archive_specs=None):
    started = time.monotonic()
    parsed = validate_splits(splits)
    exact, known, usgs = inventory_usgs(usgs_manifest_path, usgs_paths)
    required = required_six_hour_cycles()
    rows = read_required_index(qualified_index_path, required)
    weather = inventory_hrrr(rows, required, local_archive_specs)
    results = [row for split in parsed for length in (24, 168)
               for row in count_split(split, exact, known, weather, history_hours=length)]
    code = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("eligibility.py", "quality.py")}
    monthly_hrrr = {}
    for item in weather:
        month = item["cycle_id"][:4] + "-" + item["cycle_id"][4:6]
        summary = monthly_hrrr.setdefault(month, {"cycles": 0, "index_absent": 0, "complete_summary": 0,
              "explicit_gap_summary": 0, "known_bundle_count": 0, "known_bundle_bytes": 0,
              "local_status_counts": Counter(), "weather_structure_counts": Counter(), "soil_structure_counts": Counter()})
        summary["cycles"] += 1
        summary["index_absent"] += item["index_status"] != "PRESENT"
        summary["complete_summary"] += item["data_complete"] is True
        summary["explicit_gap_summary"] += item["data_complete"] is False
        if item["bundle_identity"]:
            summary["known_bundle_count"] += 1
            summary["known_bundle_bytes"] += item["bundle_identity"]["bytes"]
        summary["local_status_counts"][item["local_archive_status"]] += 1
        summary["weather_structure_counts"][item["weather_group_support"]] += 1
        summary["soil_structure_counts"][item["soil_group_support"]] += 1
    public_months = [{key: value for key, value in item.items() if key != "path"} for item in usgs]
    report = {"schema": "hydra-development-eligibility-metadata/v1", "status": "COMPLETED_WITH_EXPLICIT_UNKNOWN_COVERAGE" if len(known) < len(usgs) else "COMPLETED_USGS_METADATA_INVENTORY",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "USGS metadata eligibility and HRRR structural inventory; not a frozen scientific cohort or forecast skill",
              "development_interval": {"start_utc": iso(DEVELOPMENT_START), "end_exclusive_utc": iso(DEVELOPMENT_END)},
              "splits": splits, "split_status": "CALLER_SUPPLIED_COUNTING_RANGES_NOT_A_PROTOCOL_FREEZE",
              **quality_policy_binding(input_quality_policy=PRIMARY_APPROVED, target_quality_policy=PRIMARY_APPROVED),
              "history_end": "C_MINUS_2_HOURS_INCLUSIVE", "target_horizons_hours": list(range(1, 19)),
              "purge_rule": "C+18h must be strictly before each split end; prior observations may precede split start",
              "usgs_inventory": {"required_months": len(usgs), "verified_metadata_months": len(known),
                                 "unknown_metadata_months": len(usgs) - len(known),
                                 "status_counts": dict(Counter(item["status"] for item in usgs)), "months": public_months},
              "hrrr_inventory": {"required_six_hour_cycles": len(required), "sealed_rows_present": len(rows),
                                 "warmup_or_calendar_rows_absent": len(required) - len(rows),
                                 "known_bundle_count": sum(item["bundle_identity"] is not None for item in weather),
                                 "known_bundle_bytes_lower_bound": sum(item["bundle_identity"]["bytes"] for item in weather if item["bundle_identity"]),
                                 "local_status_counts": dict(Counter(item["local_archive_status"] for item in weather)),
                                 "by_month": monthly_hrrr,
                                 "ordinary_18h_runs_excluded": "A two-hour availability lag leaves at most16 target hours; native18 runs cannot supply direct H18",
                                 "numeric_weather_masks_verified": False, "local_payload_presence_is_stat_only": True},
              "results": results, "metadata_columns_decoded": list(METADATA_COLUMNS),
              "source_identities": {"usgs_manifest_sha256": USGS_MANIFEST_SHA256, "qualified_index_sha256": INDEX_SHA256,
                                    "qualified_index_bytes": INDEX_BYTES}, "implementation_sha256": code,
              "interpretation": ["UNKNOWN includes unread/dataless/missing or identity-failed partitions; it is never converted to an absent observation or zero eligible count",
                                 "Known missing exact hours are distinguished from unknown monthly coverage",
                                 "numeric_valid is retained acquisition metadata; discharge sign, magnitude and units are not recomputed",
                                 "HRRR structural support may use an older single extended run; unknown summaries are not inferred from a legacy complete flag",
                                 "Different lookbacks are counted on identical split origins; compare their common eligible cohort before scientific contrasts",
                                 "Known byte totals are lower bounds, not full-corpus download estimates; no local path supplied is not a claim of cloud absence"],
              "execution": {"elapsed_seconds": time.monotonic() - started, "discharge_value_columns_decoded": False,
                            "protected_numeric_values_decoded": False, "whole_file_encoded_bytes_hashed_for_identity": True, "hrrr_payloads_read": 0, "network_requests": 0,
                            "model_operations": 0, "scaler_fits": 0, "imputation": False, "source_files_modified": False}}
    private = {"schema": "hydra-private-development-source-inventory/v1", "created_at_utc": report["created_at_utc"],
               "source_identities": report["source_identities"], "usgs_partitions": usgs, "hrrr_cycles": weather}
    return report, private
