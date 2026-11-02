#!/usr/bin/env python3
"""Read committed corpus metadata, a read-only budget snapshot and optional ps.

No model/feature/target payload is opened. No lockfile is used as liveness
evidence. This program does not import corpus writers, settle budget attempts,
fetch sources, hydrate files, create outputs or start background work.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sqlite3
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hydra-development-corpus/v1"
PLAN_KEYS = ("schema", "splits", "history_hours", "chunk_hours", "input_quality_policy", "target_quality_policy",
             "qualified_index_sha256", "usgs_manifest_sha256", "candidate_weights_sha256", "units",
             "required_cycle_ids", "calendar_unavailable_cycle_ids", "implementation_sha256")
MAX_UNITS, MAX_CHUNKS, MAX_ORIGINS = 256, 2048, 200000
MODEL_LIMIT = 16 * 1024**2


class SnapshotError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise SnapshotError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def is_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value, *, hourly=False):
    require(type(value) is str, "INVALID_TIMESTAMP")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0, "UTC_TIMESTAMP_REQUIRED")
    require(not hourly or stamp.minute == stamp.second == stamp.microsecond == 0, "EXACT_HOUR_REQUIRED")
    return stamp.timestamp()


def count(value, *, maximum=MAX_ORIGINS * 3):
    require(type(value) is int and 0 <= value <= maximum, "INVALID_BOUNDED_COUNT")
    return value


def resident_directory(path):
    info = Path(path).lstat()
    require(stat.S_ISDIR(info.st_mode) and not getattr(info, "st_flags", 0) & 0x40000000, "DIRECTORY_NOT_RESIDENT_OR_REGULAR")
    return info


def directory_entries(path, maximum):
    result = []
    for item in Path(path).iterdir():
        require(len(result) < maximum, "DIRECTORY_ENTRY_BOUND_EXCEEDED")
        result.append(item)
    return result


def read_json(path, maximum):
    """Bounded no-follow metadata read with a stable immutable-file identity."""
    path = Path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not getattr(before, "st_flags", 0) & 0x40000000
            and 0 < before.st_size <= maximum, "METADATA_NOT_RESIDENT_REGULAR_OR_BOUNDED")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(maximum + 1)
    after = path.lstat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(stamp(before) == stamp(opened) == stamp(after) and len(raw) == before.st_size, "METADATA_CHANGED_DURING_READ")
    value = json.loads(raw)
    require(isinstance(value, dict), "METADATA_OBJECT_REQUIRED")
    return value, raw, after


def load_plan(directory, expected_sha256):
    require(is_digest(expected_sha256), "PINNED_PLAN_DIGEST_REQUIRED")
    resident_directory(directory)
    plan, raw, info = read_json(Path(directory) / "PLAN.json", 4 * 1024**2)
    require(plan.get("schema") == SCHEMA and all(key in plan for key in PLAN_KEYS), "UNSUPPORTED_CORPUS_PLAN")
    require(plan.get("plan_sha256") == expected_sha256 == digest({key: plan[key] for key in PLAN_KEYS}), "CORPUS_PLAN_DIGEST_MISMATCH")
    require(type(plan["history_hours"]) is int and plan["history_hours"] in (24, 168), "INVALID_PLAN_HISTORY")
    require(type(plan["chunk_hours"]) is int and 1 <= plan["chunk_hours"] <= 128, "INVALID_PLAN_CHUNK_LIMIT")
    require(isinstance(plan["units"], list) and 0 < len(plan["units"]) <= MAX_UNITS, "PLAN_UNIT_BOUND")
    units, entries, intervals = {}, [], []
    for unit in plan["units"]:
        identifier = unit["id"]
        require(type(identifier) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identifier)
                and identifier not in units, "DUPLICATE_OR_UNSAFE_PLAN_UNIT")
        require(unit["history_hours"] == plan["history_hours"] and isinstance(unit["chunks"], list), "PLAN_UNIT_GEOMETRY_MISMATCH")
        units[identifier] = unit
        indexes, total = set(), 0
        for chunk in unit["chunks"]:
            index = count(chunk["index"], maximum=MAX_CHUNKS)
            require(index not in indexes, "DUPLICATE_PLAN_CHUNK")
            indexes.add(index)
            n = count(chunk["origin_count"], maximum=128)
            require(n > 0 and n <= plan["chunk_hours"], "INVALID_CHUNK_ORIGINS")
            first, last = timestamp(chunk["first_origin_utc"], hourly=True), timestamp(chunk["last_origin_utc"], hourly=True)
            require(last == first + (n - 1) * 3600 and last + 18 * 3600 < timestamp(unit["split_end_exclusive_utc"], hourly=True),
                    "CHUNK_CLOCK_OR_SPLIT_MISMATCH")
            require(len(chunk["cycle_ids"]) == len(set(chunk["cycle_ids"])) <= 32, "CHUNK_CANDIDATE_INVENTORY_INVALID")
            entries.append((identifier, index, unit, chunk))
            intervals.append((first, last))
            total += n
        require(indexes == set(range(len(indexes))) and total == count(unit["candidate_origins"])
                and count(unit["requested_origins"]) == total + count(unit["purged_origins"]), "PLAN_UNIT_ACCOUNTING_MISMATCH")
    require(0 < len(entries) <= MAX_CHUNKS, "PLAN_CHUNK_BOUND")
    ordered = sorted(intervals)
    require(all(a[1] < b[0] for a, b in zip(ordered, ordered[1:])), "DUPLICATE_OR_OVERLAPPING_PLAN_ORIGINS")
    require(sum(c[3]["origin_count"] for c in entries) == count(plan["candidate_origins"], maximum=MAX_ORIGINS)
            and sum(u["requested_origins"] for u in units.values()) == count(plan["requested_origins"], maximum=MAX_ORIGINS)
            and sum(u["purged_origins"] for u in units.values()) == count(plan["purged_origins"], maximum=MAX_ORIGINS),
            "PLAN_TOTAL_ACCOUNTING_MISMATCH")
    entries.sort(key=lambda item: timestamp(item[3]["first_origin_utc"], hourly=True))
    return plan, entries, dict(file_sha256=sha(raw), file_mtime_utc=datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat())


def bounded_counts(mapping, expected_total):
    require(isinstance(mapping, dict) and len(mapping) <= 64, "INVALID_COUNT_INVENTORY")
    for key, value in mapping.items():
        require(type(key) is str and re.fullmatch(r"[A-Z0-9_|]{1,160}", key), "UNSAFE_COUNT_LABEL")
        count(value)
    require(sum(mapping.values()) == expected_total, "COUNT_INVENTORY_TOTAL_DIFFERS")
    return mapping


def read_chunk(directory, plan, unit, chunk):
    resident_directory(directory)
    commit, raw, info = read_json(directory / "COMMIT.json", 65536)
    require(commit.get("schema") == SCHEMA and commit.get("plan_sha256") == plan["plan_sha256"]
            and commit.get("unit_id") == unit["id"] and commit.get("chunk_index") == chunk["index"], "CHUNK_PLAN_IDENTITY_MISMATCH")
    require(set(commit["payloads"]) == {"model.npz", "case-index.json"}, "CHUNK_PAYLOAD_INVENTORY_MISMATCH")
    require({p.name for p in directory_entries(directory, 4)} == {"COMMIT.json", "case-index.json", "model.npz"}, "CHUNK_DIRECTORY_INVENTORY_MISMATCH")
    for name, maximum in (("model.npz", MODEL_LIMIT), ("case-index.json", 256 * 1024)):
        identity = commit["payloads"][name]
        require(is_digest(identity.get("sha256")) and type(identity.get("bytes")) is int and 0 < identity["bytes"] <= maximum,
                "INVALID_CHUNK_PAYLOAD_IDENTITY")
    # Intentionally stat only: no NPZ bytes, hashes, arrays or values are read.
    model_info = (directory / "model.npz").lstat()
    require(stat.S_ISREG(model_info.st_mode) and model_info.st_size == commit["payloads"]["model.npz"]["bytes"],
            "MODEL_PAYLOAD_MISSING_OR_STAT_MISMATCH")
    summary, body, _ = read_json(directory / "case-index.json", 256 * 1024)
    identity = commit["payloads"]["case-index.json"]
    require(len(body) == identity["bytes"] and sha(body) == identity["sha256"], "CASE_INDEX_HASH_MISMATCH")
    require(summary.get("unit_id") == unit["id"] and summary.get("chunk_index") == chunk["index"]
            and summary.get("origin_count") == chunk["origin_count"] and summary.get("case_count") == 3 * chunk["origin_count"]
            and summary.get("first_origin_utc") == chunk["first_origin_utc"] and summary.get("last_origin_utc") == chunk["last_origin_utc"]
            and summary.get("calendar_unavailable_cycle_ids") == chunk["calendar_unavailable_cycle_ids"]
            and set(summary["cycle_keys"]) == set(chunk["cycle_ids"])
            and summary.get("input_quality_policy") == plan["input_quality_policy"]
            and summary.get("target_quality_policy") == plan["target_quality_policy"], "CASE_INDEX_PLAN_MISMATCH")
    n = count(summary["case_count"])
    bounded_counts(summary["eligibility_counts"], n)
    bounded_counts(summary["source_calendar_counts"], n)
    require(count(summary["admitted_cases"]) <= summary["eligibility_counts"].get("COMPLETE", 0), "ADMITTED_CASE_COUNT_DIFFERS")
    require(all(is_digest(value) for value in summary["cycle_keys"].values()) and all(is_digest(value) for value in summary["month_keys"].values())
            and is_digest(summary.get("case_source_digest")), "INVALID_CASE_SOURCE_IDENTITIES")
    return dict(unit_id=unit["id"], chunk_index=chunk["index"], origin_count=chunk["origin_count"], case_count=n,
                admitted_cases=summary["admitted_cases"], eligibility_counts=summary["eligibility_counts"],
                source_calendar_counts=summary["source_calendar_counts"], first_origin_utc=chunk["first_origin_utc"],
                last_origin_utc=chunk["last_origin_utc"], commit_sha256=sha(raw), model_claimed_sha256=commit["payloads"]["model.npz"]["sha256"],
                model_resident=not bool(getattr(model_info, "st_flags", 0) & 0x40000000), commit_mtime_epoch=info.st_mtime)


def issue_code(error):
    return str(error) if isinstance(error, SnapshotError) else type(error).__name__


def scan_progress(directory, plan, entries):
    chunks = Path(directory) / "chunks"
    if chunks.exists() or chunks.is_symlink():
        resident_directory(chunks)
    expected = {(unit_id, index): (unit, chunk) for unit_id, index, unit, chunk in entries}
    valid, invalid, absent, issues = {}, [], [], []
    for unit_id, index, unit, chunk in entries:
        path = chunks / unit_id / f"{index:03d}"
        if not path.exists() and not path.is_symlink():
            absent.append((unit_id, index))
            continue
        try:
            # Prevent an expected child path from traversing a substituted unit link.
            resident_directory(path.parent)
            valid[unit_id, index] = read_chunk(path, plan, unit, chunk)
        except (OSError, ValueError, KeyError, TypeError) as error:
            invalid.append((unit_id, index))
            issues.append(dict(unit_id=unit_id, chunk_index=index, code=issue_code(error)))
    # Unexpected/duplicate directory spellings cannot be counted as progress.
    if chunks.exists():
        unit_paths = directory_entries(chunks, MAX_UNITS)
        unit_ids = {item[0] for item in entries}
        for unit_path in unit_paths:
            if unit_path.name not in unit_ids:
                issues.append(dict(code="UNPLANNED_UNIT_ENTRY", entry_name_sha256=sha(unit_path.name.encode())))
                continue
            try:
                resident_directory(unit_path)
                paths = directory_entries(unit_path, MAX_CHUNKS)
                names = {f"{index:03d}" for identifier, index in expected if identifier == unit_path.name}
                for path in paths:
                    if path.name not in names:
                        issues.append(dict(unit_id=unit_path.name, code="UNPLANNED_OR_DUPLICATE_CHUNK_ENTRY", entry_name_sha256=sha(path.name.encode())))
            except (OSError, ValueError) as error:
                issues.append(dict(unit_id=unit_path.name, code=issue_code(error)))
    payload_claims = defaultdict(list)
    for key, value in valid.items():
        payload_claims[value["model_claimed_sha256"]].append(key)
    for repeated in payload_claims.values():
        if len(repeated) > 1:
            for key in repeated:
                valid.pop(key)
                invalid.append(key)
                issues.append(dict(unit_id=key[0], chunk_index=key[1], code="DUPLICATE_MODEL_PAYLOAD_CLAIM"))
    values = list(valid.values())
    eligibility, calendar = Counter(), Counter()
    units = defaultdict(lambda: dict(committed_chunks=0, committed_origins=0, committed_cases=0, admitted_cases=0))
    for value in values:
        eligibility.update(value["eligibility_counts"])
        calendar.update(value["source_calendar_counts"])
        u = units[value["unit_id"]]
        u["committed_chunks"] += 1
        for target, source in (("committed_origins", "origin_count"), ("committed_cases", "case_count"), ("admitted_cases", "admitted_cases")):
            u[target] += value[source]
    prefix, through = 0, None
    for unit_id, index, _, _ in entries:
        if (unit_id, index) not in valid:
            break
        prefix += 1
        through = valid[unit_id, index]["last_origin_utc"]
    newest = max((value["commit_mtime_epoch"] for value in values), default=None)
    return dict(status="METADATA_INTEGRITY_ISSUES" if issues else "ALL_PLANNED_CHUNK_METADATA_COMMITTED" if len(values) == len(entries) else "PARTIAL_COMMITTED_METADATA",
                planned_chunks=len(entries), committed_chunks=len(values), missing_chunks=len(absent), invalid_chunks=len(invalid),
                planned_candidate_origins=plan["candidate_origins"], planned_candidate_cases=3 * plan["candidate_origins"],
                committed_origins=sum(value["origin_count"] for value in values), committed_cases=sum(value["case_count"] for value in values),
                admitted_cases_from_case_indices=sum(value["admitted_cases"] for value in values), eligibility_counts=dict(eligibility),
                source_calendar_counts=dict(calendar), contiguous_committed_chunks=prefix, contiguous_committed_through_utc=through,
                last_committed_origin_utc=max((value["last_origin_utc"] for value in values), default=None),
                latest_commit_file_mtime_utc=datetime.fromtimestamp(newest, timezone.utc).isoformat() if newest is not None else None,
                dataless_model_payloads=sum(not value["model_resident"] for value in values), units=dict(units), issues=issues,
                verification_scope="Pinned PLAN and COMMIT/case-index metadata; model.npz existence/size only; no payload hash or numeric validation",
                model_payload_bytes_read=0, feature_or_target_arrays_decoded=False)


def read_budget(database, expected_approval_sha256=None):
    observed = utc_now()
    if database is None:
        return dict(status="NOT_REQUESTED", observed_at_utc=observed)
    path = Path(database)
    connection = None
    try:
        for candidate in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
            if candidate == path or candidate.exists() or candidate.is_symlink():
                info = candidate.lstat()
                require(stat.S_ISREG(info.st_mode) and not getattr(info, "st_flags", 0) & 0x40000000
                        and info.st_size <= 256 * 1024**2, "BUDGET_FILE_NOT_RESIDENT_REGULAR_OR_BOUNDED")
        require(expected_approval_sha256 is None or is_digest(expected_approval_sha256), "INVALID_EXPECTED_BUDGET_APPROVAL")
        connection = sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True, timeout=.5)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        deadline = time.monotonic() + 2
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        connection.execute("BEGIN")
        records = connection.execute("SELECT approval,body_cap,get_cap,cost_cap_micro,charged_body,attempts FROM budget WHERE id=1").fetchall()
        require(len(records) == 1, "BUDGET_SINGLETON_INVALID")
        approval, body_cap, get_cap, cost_cap, charged, attempts = records[0]
        require(is_digest(approval), "INVALID_STORED_BUDGET_APPROVAL")
        require(expected_approval_sha256 is None or approval == expected_approval_sha256, "BUDGET_APPROVAL_MISMATCH")
        require(all(type(x) is int and x >= 0 for x in (body_cap, get_cap, cost_cap, charged, attempts))
                and body_cap > 0 and get_cap > 0 and cost_cap > 0 and charged <= body_cap and attempts <= get_cap,
                "BUDGET_LIMIT_OR_TOTAL_INVALID")
        sums = connection.execute("SELECT COALESCE(SUM(reserved),0),COUNT(*),COALESCE(SUM(observed),0),MAX(created_utc) FROM attempts").fetchone()
        require((charged, attempts) == sums[:2], "BUDGET_TOTALS_DO_NOT_RECONCILE")
        bad = connection.execute("SELECT COUNT(*) FROM attempts WHERE typeof(reserved)!='integer' OR typeof(observed)!='integer' OR reserved<=0 OR observed<0 OR observed>reserved OR status NOT IN ('PRECHARGED','VERIFIED','FAILED') OR reserved IS NULL OR observed IS NULL OR status IS NULL").fetchone()[0]
        require(bad == 0 and type(sums[2]) is int and sums[2] >= 0, "INVALID_ATTEMPT_ACCOUNTING")
        states = dict(connection.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status").fetchall())
        pending = connection.execute("SELECT COALESCE(SUM(reserved),0) FROM attempts WHERE status='PRECHARGED'").fetchone()[0]
        latest_created = datetime.fromtimestamp(timestamp(sums[3]), timezone.utc).isoformat() if sums[3] is not None else None
        connection.commit()
        estimate_micro = (charged * 280000 + 1024**3 - 1) // 1024**3 + attempts * 65
        return dict(status="RECONCILED_READ_ONLY" if expected_approval_sha256 else "RECONCILED_AUTHORITY_UNPINNED",
                    observed_at_utc=observed, approval_sha256=approval, approval_pinned=expected_approval_sha256 is not None,
                    body_cap_bytes=body_cap, get_cap=get_cap, cost_estimate_cap_usd=cost_cap / 1e6,
                    precharged_body_bytes=charged, observed_settled_body_bytes=sums[2], attempted_gets=attempts,
                    attempt_states=states, pending_precharged_bytes=pending, latest_attempt_created_utc=latest_created,
                    conservative_precharge_estimate_usd=estimate_micro / 1e6,
                    estimate_basis="Existing staging rate model:0.28USD/GiB plus0.000065USD/GET; not a provider invoice",
                    scope="This SQLite retained-object staging budget only; other combined-envelope allocations and FileProvider transport are excluded",
                    pending_attempts_preserved=True, settlements_performed=0, billing_invoice_reconciled=False, sqlite_mode="ro/query_only")
    except (OSError, ValueError, sqlite3.Error) as error:
        return dict(status="BUDGET_UNAVAILABLE_OR_INVALID", observed_at_utc=observed, error_code=issue_code(error),
                    totals=None, pending_attempts_preserved=True, settlements_performed=0)
    finally:
        if connection is not None:
            connection.close()


def last_increment(path, expected_plan, progress, *, stale_after_seconds=300):
    observed = utc_now()
    if path is None:
        return dict(status="NOT_REQUESTED", observed_at_utc=observed)
    try:
        value, raw, info = read_json(path, 2 * 1024**2)
        corpus = value.get("corpus", value)
        require(isinstance(corpus, dict) and corpus.get("schema") == SCHEMA and corpus.get("plan_sha256") == expected_plan,
                "INCREMENT_CORPUS_PLAN_MISMATCH")
        created = value.get("created_at_utc", corpus.get("created_at_utc"))
        stamp = timestamp(created)
        age = timestamp(observed) - stamp
        status = corpus.get("status")
        require(status in ("COMPLETED_DEVELOPMENT_CORPUS", "STOPPED_WITH_EXPLICIT_FAILURE", "STOPPED_AT_CHUNK_BOUNDARY"), "UNRECOGNIZED_INCREMENT_STATUS")
        failure = corpus.get("failure")
        safe = None
        if failure is not None:
            require(isinstance(failure, dict), "INVALID_INCREMENT_FAILURE")
            safe = {}
            for name in ("error_type", "source_status", "cause_type", "cycle_id"):
                item = failure.get(name)
                if item is None:
                    continue
                pattern = r"(?:USGS:\d{4}-\d{2}|\d{8}_\d{2}Z)" if name == "cycle_id" else r"[A-Za-z0-9_]{1,100}"
                safe[name] = item if isinstance(item, str) and re.fullmatch(pattern, item) else "REDACTED_UNSAFE_VALUE"
        reported = count(corpus["completed_origins"], maximum=MAX_ORIGINS)
        scanned = progress.get("committed_origins") if progress else None
        return dict(status="READ_LAST_INCREMENT_SNAPSHOT", observed_at_utc=observed, artifact_created_at_utc=created,
                    file_mtime_utc=datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(), file_sha256=sha(raw),
                    age_seconds=max(0., age), timestamp_in_future=age < -1,
                    stale_by_age=age > stale_after_seconds, committed_progress_after_snapshot=scanned is not None and scanned > reported,
                    receipt_progress_exceeds_scan=scanned is not None and reported > scanned,
                    reported_completed_origins=reported, reported_corpus_status=status, reported_failure=safe,
                    interpretation="Attributed terminal receipt from an earlier increment; not current process liveness or proof that a later attempt failed")
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(status="INCREMENT_UNAVAILABLE_OR_INVALID", observed_at_utc=observed, error_code=issue_code(error))


def process_cwd(pid):
    link = Path(f"/proc/{pid}/cwd")
    if link.is_symlink():
        return os.readlink(link)
    executable = shutil.which("lsof")
    if executable is None:
        return None
    result = subprocess.run([executable, "-a", "-p", str(pid), "-d", "cwd", "-Fn"], capture_output=True, text=True, timeout=3)
    require(len(result.stdout) <= 65536, "CWD_OBSERVATION_OUTPUT_BOUND")
    names = [line[1:] for line in result.stdout.splitlines() if line.startswith("n")]
    return names[0] if result.returncode == 0 and len(names) == 1 else None


def observe_worker(*, expected_pid=None, expected_command_sha256=None, expected_cwd=None, corpus_run_name=None):
    observed = utc_now()
    if expected_pid is None:
        return dict(status="UNVERIFIED_NOT_REQUESTED", observed_at_utc=observed)
    require(type(expected_pid) is int and 0 < expected_pid < 2**31, "INVALID_EXPECTED_PID")
    try:
        result = subprocess.run(["ps", "-ww", "-p", str(expected_pid), "-o", "command="], capture_output=True, text=True, timeout=3)
        require(len(result.stdout) <= 65536, "PROCESS_COMMAND_OUTPUT_BOUND")
        command = result.stdout.strip()
        if not command and result.returncode in (0, 1):
            return dict(status="PROCESS_NOT_OBSERVED", expected_pid=expected_pid, observed_at_utc=observed)
        require(result.returncode == 0 and command, "PROCESS_OBSERVATION_FAILED")
        actual_sha = sha(command.encode())
        base = dict(status="PROCESS_OBSERVED_IDENTITY_UNVERIFIED", expected_pid=expected_pid,
                    observed_at_utc=observed, observed_command_sha256=actual_sha, command_text_returned=False)
        if not is_digest(expected_command_sha256) or expected_cwd is None:
            return {**base, "reason": "Exact caller command digest and expected checkout cwd are required"}
        try:
            cwd = process_cwd(expected_pid)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            return {**base, "reason": "Process cwd was unavailable", "error_code": issue_code(error)}
        if cwd is None:
            return {**base, "reason": "Process cwd could not be associated with the expected checkout"}
        cwd_match = Path(cwd).resolve() == Path(expected_cwd).resolve()
        args = shlex.split(command)
        index = 1
        while index < len(args) and args[index] in ("-B", "-u"):
            index += 1
        script = Path(args[index]) if index < len(args) else None
        actual_script = (Path(cwd) / script).resolve() if script is not None and not script.is_absolute() else script
        expected_script = (Path(expected_cwd) / "scripts/stage_development.py").resolve()
        run_matches = False
        if args.count("--corpus-run-name") == 1:
            position = args.index("--corpus-run-name")
            run_matches = position + 1 < len(args) and args[position + 1] == corpus_run_name
        def argument(name):
            if args.count(name) != 1:
                return None
            position = args.index(name)
            return args[position + 1] if position + 1 < len(args) else None
        stage_plan, execution_plan = argument("--staging-plan"), argument("--approved-execution-plan-sha256")
        execution_shape = args.count("--execute") == 1 and isinstance(stage_plan, str) and stage_plan.endswith(".json") and is_digest(execution_plan)
        matched = actual_sha == expected_command_sha256 and cwd_match and actual_script == expected_script and run_matches and execution_shape
        return {**base, "status": "VERIFIED_CALLER_PROCESS_AT_SNAPSHOT" if matched else "PROCESS_IDENTITY_MISMATCH",
                "command_digest_matches": actual_sha == expected_command_sha256, "checkout_cwd_matches": cwd_match,
                "script_matches": actual_script == expected_script, "corpus_run_argument_matches": run_matches,
                "staging_execution_arguments_present": execution_shape,
                "qualification": "One ps/cwd observation only; no assertion of progress, health, lock ownership or future liveness"}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return dict(status="PROCESS_OBSERVATION_UNAVAILABLE", expected_pid=expected_pid, observed_at_utc=observed, error_code=issue_code(error))


def inspect_progress(*, corpus_directory, expected_plan_sha256, budget_database=None, expected_budget_approval_sha256=None,
                     last_increment_path=None, stale_after_seconds=300, expected_pid=None,
                     expected_command_sha256=None, expected_cwd=None):
    started = utc_now()
    require(type(stale_after_seconds) is int and 0 < stale_after_seconds <= 86400, "BOUNDED_STALENESS_THRESHOLD_REQUIRED")
    root = Path(corpus_directory).absolute()
    result = dict(schema="hydra-staging-progress-snapshot/v1", snapshot_started_at_utc=started,
                  plan_sha256=expected_plan_sha256, corpus_location_sha256=digest(str(root.resolve())),
                  source_numeric_payload_reads=0, network_calls=0, budget_mutations=0, inspection_is_not_global_atomic=True)
    try:
        plan, entries, identity = load_plan(root, expected_plan_sha256)
        result.update(plan_identity=identity, progress=scan_progress(root, plan, entries))
        result["status"] = "METADATA_SNAPSHOT_WITH_ISSUES" if result["progress"]["issues"] else "METADATA_SNAPSHOT_OK"
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(status="PLAN_OR_PROGRESS_UNAVAILABLE", progress=None, error_code=issue_code(error))
    result["budget"] = read_budget(budget_database, expected_budget_approval_sha256)
    result["last_increment"] = last_increment(last_increment_path if last_increment_path is not None else root / "RESULT.json",
                                               expected_plan_sha256, result["progress"], stale_after_seconds=stale_after_seconds)
    result["worker_observation"] = observe_worker(expected_pid=expected_pid, expected_command_sha256=expected_command_sha256,
                                                  expected_cwd=expected_cwd, corpus_run_name=root.name)
    result["snapshot_finished_at_utc"] = utc_now()
    result["qualification"] = "Metadata verification only; numeric shards/shared-cache payloads are not reverified. Missing chunks are uncommitted, not upstream gaps. Worker and budget observations are separate timestamped snapshots."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-directory", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--budget-database", type=Path)
    parser.add_argument("--expected-budget-approval-sha256")
    parser.add_argument("--last-increment", type=Path)
    parser.add_argument("--stale-after-seconds", type=int, default=300)
    parser.add_argument("--expected-pid", type=int)
    parser.add_argument("--expected-command-sha256")
    parser.add_argument("--expected-cwd", type=Path)
    args = parser.parse_args()
    try:
        result = inspect_progress(corpus_directory=args.corpus_directory, expected_plan_sha256=args.expected_plan_sha256,
                                  budget_database=args.budget_database, expected_budget_approval_sha256=args.expected_budget_approval_sha256,
                                  last_increment_path=args.last_increment, stale_after_seconds=args.stale_after_seconds,
                                  expected_pid=args.expected_pid, expected_command_sha256=args.expected_command_sha256,
                                  expected_cwd=args.expected_cwd)
    except (OSError, ValueError, TypeError) as error:
        result = dict(schema="hydra-staging-progress-snapshot/v1", status="INVALID_INSPECTION_REQUEST", error_code=issue_code(error))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result["status"] not in ("METADATA_SNAPSHOT_OK",):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
