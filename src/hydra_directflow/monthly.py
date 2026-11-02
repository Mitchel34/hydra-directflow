"""Plan and checkpoint one real development calendar month without model fitting."""
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

from .contract import DevelopmentProtocol, require
from .local_data import iso
from .multicycle import _months_for_windows, cycle_reference
from .reference import digest
from .cache import (_read, _write_durable, _sync_directory, implementation_identity,
                    CachedDevelopmentSession)
from .pipeline_timing import load_builder_request, _array_digest
from .telemetry import atomic_json, canonical

SCHEMA = "hydra-month-preprocessing/v1"
HOUR = 3600


def cycle_ids_for_origins(origins):
    """All retained 36-lead runs capable of H18 under the fixed two-hour lag."""
    require(origins and origins == sorted(set(origins)), "Increasing unique origins required")
    step = 6 * HOUR
    first = ((origins[0] - 18 * HOUR + step - 1) // step) * step
    last = ((origins[-1] - 2 * HOUR) // step) * step
    return [datetime.fromtimestamp(t, timezone.utc).strftime("%Y%m%d_%HZ") for t in range(first, last + 1, step)]


def month_calendar(month, history_hours=168, chunk_hours=120):
    DevelopmentProtocol(history_hours=history_hours)
    require(isinstance(month, str) and re.fullmatch(r"\d{4}-\d{2}", month), "Explicit YYYY-MM required")
    require(type(chunk_hours) is int and 1 <= chunk_hours <= 128, "Chunk must contain one to 128 origins")
    first = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
    following = first.replace(year=first.year + 1, month=1) if first.month == 12 else first.replace(month=first.month + 1)
    origins = list(range(int(first.timestamp()), int(following.timestamp()), HOUR))
    months = sorted(_months_for_windows([(origins[0] - (history_hours + 1) * HOUR, origins[-1] + 18 * HOUR)]))
    cycles = cycle_ids_for_origins(origins)
    for cycle in cycles:
        cycle_reference(cycle)
    chunks = []
    for index, offset in enumerate(range(0, len(origins), chunk_hours)):
        batch = origins[offset:offset + chunk_hours]
        candidates = cycle_ids_for_origins(batch)
        require(len(candidates) <= 32, "Chunk exceeds bounded cycle reader")
        chunks.append(dict(index=index, origins_utc=[iso(t) for t in batch], cycle_ids=candidates))
    return dict(month=month, history_hours=history_hours, chunk_hours=chunk_hours,
                origin_count=len(origins), gauge_case_count=3 * len(origins),
                origins_utc=[iso(t) for t in origins], required_cycle_ids=cycles,
                required_usgs_months=months, chunks=chunks)


def plan_month(request_path, *, month, chunk_hours=120):
    raw, request = load_builder_request(request_path)
    calendar = month_calendar(month, request["history_hours"], chunk_hours)
    require(request["origins_utc"] == calendar["origins_utc"], "Request must contain every month origin exactly once")
    supplied = [row["cycle_id"] for row in request["archive_specs"]]
    require(len(set(supplied)) == len(supplied), "Duplicate source cycle")
    missing = sorted(set(calendar["required_cycle_ids"]) - set(supplied))
    absent_months = sorted(set(calendar["required_usgs_months"]) - set(request["usgs_paths"]))
    code = implementation_identity()
    code["monthly.py"] = hashlib.sha256(_read(Path(__file__), 256 * 1024)).hexdigest()
    code["pipeline_timing.py"] = hashlib.sha256(_read(Path(__file__).with_name("pipeline_timing.py"), 256 * 1024)).hexdigest()
    identity = dict(schema=SCHEMA, calendar=calendar, request_sha256=hashlib.sha256(raw).hexdigest(),
                    resolved_request_sha256=digest(request), implementation_sha256=code)
    return dict(**identity, plan_sha256=digest(identity), status="BLOCKED_MISSING_SOURCE_SPECIFICATIONS" if missing or absent_months else "PLANNED_NOT_EXECUTED",
                missing_cycle_ids=missing, missing_usgs_months=absent_months,
                qualification="Complete calendar and candidate specification only; execution still verifies local sources or committed derived caches"), request


def _chunk_read(directory, plan_sha256, chunk):
    commit = json.loads(_read(directory / "COMMIT.json", 8192))
    require(commit["plan_sha256"] == plan_sha256 and commit["chunk_index"] == chunk["index"], "Chunk belongs to a different plan")
    name = commit["payload_sha256"] + ".json"
    require(re.fullmatch(r"[0-9a-f]{64}\.json", name), "Invalid chunk payload name")
    require(set(p.name for p in directory.iterdir()) == {"COMMIT.json", name}, "Unexpected committed chunk files")
    raw = _read(directory / name, 2 * 1024**2)
    require(len(raw) == commit["payload_bytes"] and hashlib.sha256(raw).hexdigest() == commit["payload_sha256"], "Chunk payload hash differs")
    payload = json.loads(raw)
    require(payload["index"] == chunk["index"] and payload["origins_utc"] == chunk["origins_utc"], "Chunk origin calendar differs")
    require(payload["case_count"] == 3 * len(chunk["origins_utc"])
            and sum(payload["eligibility_counts"].values()) == payload["case_count"], "Chunk case accounting differs")
    require(set(payload["cache"]["cycle_keys"]) == set(chunk["cycle_ids"]), "Chunk candidate inventory differs")
    return payload


def _chunk_write(root, plan_sha256, chunk, payload):
    target = root / "chunks" / f'{chunk["index"]:03d}'
    require(not target.exists() and not target.is_symlink(), "Committed chunks are immutable")
    stage = root / "staging" / uuid.uuid4().hex
    stage.mkdir()
    raw = canonical(payload)
    require(len(raw) <= 2 * 1024**2, "Chunk receipt exceeds bound")
    sha = hashlib.sha256(raw).hexdigest()
    _write_durable(stage / (sha + ".json"), raw)
    _write_durable(stage / "COMMIT.json", canonical(dict(plan_sha256=plan_sha256, chunk_index=chunk["index"], payload_sha256=sha, payload_bytes=len(raw))))
    _sync_directory(stage)
    stage.rename(target)
    _sync_directory(root / "chunks")


def run_month(*, request_path, month, cache_directory, run_directory, chunk_hours=120,
              execution_enabled=False, approved_plan_sha256=None, max_wall_seconds=900,
              max_rss_bytes=4 * 1024**3, stop_after_chunks=None):
    """Checkpoint after each bounded chunk; resumes verify all committed chunks.

    Resource checks are cooperative at chunk boundaries, not OS containment.
    stop_after_chunks is a deliberate resumability check, never a completion claim.
    """
    plan, request = plan_month(request_path, month=month, chunk_hours=chunk_hours)
    if not execution_enabled:
        return plan
    require(type(execution_enabled) is bool and plan["status"] == "PLANNED_NOT_EXECUTED"
            and approved_plan_sha256 == plan["plan_sha256"], "Complete calendar and exact reviewed plan required")
    require(type(max_wall_seconds) is int and 1 <= max_wall_seconds <= 1800
            and type(max_rss_bytes) is int and 0 < max_rss_bytes <= 4 * 1024**3, "Explicit bounded resources required")
    require(stop_after_chunks is None or type(stop_after_chunks) is int and stop_after_chunks > 0, "Positive explicit stopping chunk count required")
    root = Path(run_directory).absolute()
    checkout = Path(__file__).resolve().parents[2]
    require(any(root.resolve().is_relative_to((checkout / name).resolve()) for name in ("runs", ".local")), "Private month output must stay under ignored runs/ or .local/")
    require(any(Path(cache_directory).resolve().is_relative_to((checkout / name).resolve()) for name in (".cache", ".local")), "Private derived cache must stay under ignored .cache/ or .local/")
    root.mkdir(parents=True, exist_ok=True)
    require(not root.is_symlink(), "Month output must be a real directory")
    descriptor = os.open(root / ".writer.lock", os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    started, peak, resumed, built = time.monotonic(), 0, 0, 0
    import psutil
    process = psutil.Process()
    payloads, newly_built = [], []
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = root / "PLAN.json"
        if manifest.exists():
            require(json.loads(_read(manifest, 256 * 1024))["plan_sha256"] == plan["plan_sha256"], "Run directory belongs to a different month plan")
        else:
            require(set(p.name for p in root.iterdir()) <= {".writer.lock"}, "Refusing unrelated output directory")
            _write_durable(manifest, canonical(plan))
        for name in ("chunks", "staging"):
            path = root / name
            path.mkdir(exist_ok=True)
            require(not path.is_symlink(), "Month stage must be a real directory")
        shared = {key: request[key] for key in ("archive_specs", "qualified_index_path", "weights_path", "usgs_paths", "usgs_manifest_path")}
        with CachedDevelopmentSession(**shared, cache_directory=cache_directory) as session:
            for chunk in plan["calendar"]["chunks"]:
                peak = max(peak, process.memory_info().rss)
                require(peak <= max_rss_bytes and time.monotonic() - started < max_wall_seconds, "Month resource bound reached")
                target = root / "chunks" / f'{chunk["index"]:03d}'
                if target.exists() or target.is_symlink():
                    require(not target.is_symlink(), "Committed chunk cannot be a symlink")
                    payload = _chunk_read(target, plan["plan_sha256"], chunk)
                    # Verify dependent store payloads, even when no windows need rebuilding.
                    for cycle in payload["cache"]["cycle_keys"]:
                        require(session._cycle(cycle)[3] == payload["cache"]["cycle_keys"][cycle], "Dependent basin cache identity differs")
                    for month_key in payload["cache"]["month_keys"]:
                        require(session._month(month_key)[2] == payload["cache"]["month_keys"][month_key], "Dependent discharge cache identity differs")
                    resumed += 1
                else:
                    arrays, receipt = session.build(origins_utc=chunk["origins_utc"], history_hours=request["history_hours"], archive_cycle_ids=chunk["cycle_ids"])
                    index = [dict(case_id=str(arrays["case_id"][i]), source_cycle=str(arrays["source_cycle"][i]),
                                  soil_source_cycle=str(arrays["soil_source_cycle"][i]), eligibility_reason=str(arrays["eligibility_reason"][i]))
                             for i in range(len(arrays["case_id"]))]
                    payload = dict(index=chunk["index"], origins_utc=chunk["origins_utc"], case_count=receipt["case_count"],
                                   complete_case_count=receipt["complete_case_count"], eligibility_counts=receipt["eligibility_counts"],
                                   stage_seconds=receipt["stage_seconds"], array_sha256=_array_digest(arrays),
                                   cache=receipt["cache"], case_index=index)
                    peak = max(peak, process.memory_info().rss)
                    require(peak <= max_rss_bytes and time.monotonic() - started < max_wall_seconds, "Month resource bound reached")
                    _chunk_write(root, plan["plan_sha256"], chunk, payload)
                    newly_built.append(payload)
                    built += 1
                    del arrays
                peak = max(peak, process.memory_info().rss)
                require(peak <= max_rss_bytes and time.monotonic() - started < max_wall_seconds, "Month resource bound reached")
                payloads.append(payload)
                if stop_after_chunks is not None and built >= stop_after_chunks:
                    break
            cache_counts = dict(session.cache.counts)
            initialization_seconds = session.initialization_seconds
        peak = max(peak, process.memory_info().rss)
        require(peak <= max_rss_bytes and time.monotonic() - started < max_wall_seconds, "Month resource bound reached")
        complete = len(payloads) == len(plan["calendar"]["chunks"])
        counts = Counter()
        for payload in payloads:
            counts.update(payload["eligibility_counts"])
        result = dict(schema=SCHEMA, status="COMPLETED_FULL_MONTH" if complete else "STOPPED_AT_EXPLICIT_CHUNK_BOUNDARY",
                      created_at_utc=datetime.now(timezone.utc).isoformat(), month=month, history_hours=request["history_hours"],
                      plan_sha256=plan["plan_sha256"], implementation_sha256=plan["implementation_sha256"],
                      requested_origins=plan["calendar"]["origin_count"], requested_cases=plan["calendar"]["gauge_case_count"],
                      completed_origins=sum(len(p["origins_utc"]) for p in payloads), completed_cases=sum(p["case_count"] for p in payloads),
                      complete_case_count=sum(p["complete_case_count"] for p in payloads), eligibility_counts=dict(counts),
                      required_source_cycles=len(plan["calendar"]["required_cycle_ids"]), required_usgs_months=plan["calendar"]["required_usgs_months"],
                      chunks_built=built, chunks_reused=resumed, chunk_array_digests=[p["array_sha256"] for p in payloads],
                      elapsed_seconds=time.monotonic() - started, observed_boundary_peak_rss_bytes=peak,
                      session_initialization_seconds=initialization_seconds, cache_counts=cache_counts,
                      stage_seconds_new_chunks={name: sum(p["stage_seconds"][name] for p in newly_built)
                                                for name in ("basin_store", "discharge_store", "case_assembly", "total")},
                      selection_scope="ALL_REQUIRED_SIX_HOUR_CANDIDATES_FOR_FIXED_2H_LAG_H18_36_RETAINED_LEADS" if complete else "PARTIAL_CHUNK_CALENDAR",
                      resource_qualification="Cooperative wall/RSS checks at chunk boundaries; peaks between checks may be higher",
                      optimizer_updates=0, cloud_jobs=0, network_requests=0, scientific_training=False,
                      qualification="One actual development calendar month; masks retained. Runtime does not establish scientific skill or every-season throughput.")
        atomic_json(root / "RESULT.json", result)
        return result
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
