#!/usr/bin/env python3
"""Plan or explicitly materialize 11 exact, already-owned development files.

FileProvider may hydrate an owned source during an explicitly enabled read.
Its internal HTTP traffic is not observed by this program. No GCS API, upstream
USGS query, credential command, protected mixed-month read, or model is used.
"""
from datetime import datetime, timezone
from collections import Counter
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hydra_directflow.staging import atomic_json, durable_directory, sync_directory, RAW_HOTSET
from hydra_directflow.local_data import USGS_MANIFEST_SHA256, verified_local_bytes

MONTHS = ("2018-12", "2019-06", "2019-09", "2021-06", "2021-07", "2021-11", "2021-12", "2022-02", "2022-04", "2023-03", "2023-12")
TOTAL_BYTES = 774216
MAX_ATTEMPTS_PER_FILE = 3
MAX_RESERVED_ENCODED_BYTES = 3 * (TOTAL_BYTES + len(MONTHS))
TIMEOUT_SECONDS = 60


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def local_json(path, maximum):
    path = Path(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not getattr(info, "st_flags", 0) & 0x40000000
            and 0 < info.st_size <= maximum, "Control input must be a bounded resident regular file")
    raw = path.read_bytes()
    return json.loads(raw), sha(raw)


def plan_materialization(staging_plan_path, approval_path, output_directory):
    staging, staging_sha = local_json(staging_plan_path, 32 * 1024**2)
    core = {key: value for key, value in staging.items() if key not in ("plan_sha256", "created_at_utc")}
    require(staging["schema"] == "hydra-development-staging-plan/v1" and sha(canonical(core)) == staging["plan_sha256"],
            "Staging plan identity differs")
    approval, approval_sha = local_json(approval_path, 65536)
    require(approval_sha == staging["sources"]["budget_approval"]["sha256"]
            and approval.get("phase_release", {}).get("development_staging") == "AUTHORIZED_WITH_DECLARED_LIMITS"
            and approval.get("approved_total_incremental_usd") == 150
            and approval.get("allocations_usd", {}).get("development_staging") == 35
            and approval.get("protected_assessment_access_authorized") is False,
            "Materialization must use the exact approved development staging envelope")
    require(staging.get("protected_assessment_access_authorized") is False, "Protected source access is forbidden")
    source = staging["sources"]["usgs_manifest"]
    raw = verified_local_bytes(source["path"], USGS_MANIFEST_SHA256, expected_bytes=source["bytes"], max_bytes=2 * 1024**2)
    require(sha(raw) == source["sha256"], "USGS source-manifest authority differs")
    native = {Path(entry["path"]).stem: entry for entry in json.loads(raw)["native_partitions"]}
    output = Path(output_directory).absolute()
    require(output.resolve().is_relative_to((ROOT / ".local").resolve()) and output.name == "usgs"
            and not output.is_symlink(), "Owned outputs must use an ignored provider/usgs directory")
    selected = [entry for entry in staging["usgs_partitions"] if entry["stage_kind"] == "RETAINED_NATIVE_DEVELOPMENT_ONLY"
                and not any(candidate["status"] == "RESIDENT_UNVERIFIED_SHA" for candidate in entry["local_candidates"])]
    require(tuple(entry["month"] for entry in selected) == MONTHS, "Only the exact 11 planned native sources are admitted")
    files = []
    for entry in selected:
        month = entry["month"]
        original = native[month]
        expected = entry["original_source_identity"]
        require(month != "2024-07" and expected["sha256"] == original["sha256"] and expected["bytes"] == original["bytes"]
                and original["path"] in [candidate["path"] for candidate in entry["local_candidates"]], "Native source identity/path differs")
        require(type(original["bytes"]) is int and 0 < original["bytes"] <= 512 * 1024, "Native file exceeds materialization bound")
        files.append({"month": month, "source_path": original["path"], "bytes": original["bytes"],
                      "sha256": original["sha256"], "destination": str(output / (month + ".parquet"))})
    require(sum(item["bytes"] for item in files) == TOTAL_BYTES, "Planned total source bytes differ")
    identity = {"schema": "hydra-owned-usgs-materialization/v1", "staging_plan_sha256": staging["plan_sha256"],
                "staging_plan_file_sha256": staging_sha, "approval_sha256": approval_sha,
                "source_manifest_sha256": USGS_MANIFEST_SHA256, "files": files,
                "max_attempts_per_file": MAX_ATTEMPTS_PER_FILE, "max_reserved_encoded_bytes": MAX_RESERVED_ENCODED_BYTES,
                "timeout_seconds_per_file": TIMEOUT_SECONDS, "script_sha256": sha(Path(__file__).read_bytes()),
                "dependency_sha256": {name: sha((ROOT / "src/hydra_directflow" / name).read_bytes())
                                      for name in ("staging.py", "local_data.py", "reference.py")}}
    plan_sha = sha(canonical(identity))
    public = {"schema": identity["schema"], "status": "PLAN_ONLY_NOT_EXECUTED", "plan_sha256": plan_sha,
              "staging_plan_sha256": staging["plan_sha256"], "approval_sha256": approval_sha,
              "file_count": len(files), "months": list(MONTHS), "expected_encoded_bytes": TOTAL_BYTES,
              "max_reserved_encoded_bytes": MAX_RESERVED_ENCODED_BYTES, "max_attempts_per_file": MAX_ATTEMPTS_PER_FILE,
              "timeout_seconds_per_file": TIMEOUT_SECONDS, "script_sha256": identity["script_sha256"],
              "dependency_sha256": identity["dependency_sha256"],
              "raw_july2024_access": False, "direct_gcs_calls": 0, "source_contents_written": False,
              "qualification": "Reserve encoded-read bytes separately in the combined envelope; FileProvider HTTP request/transfer counts are unobserved"}
    return identity, public


def verified_copy(path, item):
    try:
        verified_local_bytes(path, item["sha256"], expected_bytes=item["bytes"], max_bytes=512 * 1024)
        return True
    except (OSError, ValueError):
        return False


def copy_exact_owned_source(item, *, execution_enabled=False):
    """Explicit worker primitive: opaque bytes only, no Parquet value decode."""
    require(execution_enabled is True, "Owned source reads require explicit materialization execution")
    require(item["month"] in MONTHS and type(item["bytes"]) is int and 0 < item["bytes"] <= 512 * 1024,
            "Worker may read only a bounded approved native month")
    source, target = Path(item["source_path"]), Path(item["destination"])
    info = source.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_size == item["bytes"], "Original owned source metadata differs")
    require(not target.exists() and not target.is_symlink(), "Owned copy must be new")
    require(shutil.disk_usage(target.parent).free > item["bytes"] * 2 + 64 * 1024**2, "Insufficient materialization disk headroom")
    hotset = 0
    for folder in (target.parent, target.parent.parent / "raw"):
        if folder.exists():
            require(folder.is_dir() and not folder.is_symlink(), "Hotset directories must be real")
            hotset += sum(path.lstat().st_size for path in folder.iterdir() if stat.S_ISREG(path.lstat().st_mode))
    require(hotset + item["bytes"] <= RAW_HOTSET, "Shared raw/USGS hotset limit reached")
    initially_dataless = bool(getattr(info, "st_flags", 0) & 0x40000000)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        require(stat.S_ISREG(opened.st_mode) and opened.st_size == item["bytes"], "Opened owned source differs")
        raw = stream.read(item["bytes"] + 1)
    require(len(raw) == item["bytes"] and sha(raw) == item["sha256"], "Owned source bytes differ from the original pinned object")
    temporary = target.with_name(target.name + ".partial-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as output:
            output.write(raw); output.flush(); os.fsync(output.fileno())
        # Atomic no-overwrite publication; both names are in the same filesystem.
        os.link(temporary, target)
        temporary.unlink(); sync_directory(target.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"status": "VERIFIED_OWNED_COPY", "month": item["month"], "sha256": item["sha256"],
            "encoded_bytes_read": len(raw), "source_was_dataless": initially_dataless,
            "source_currently_resident": not bool(getattr(source.lstat(), "st_flags", 0) & 0x40000000),
            "flow_columns_decoded": False}


def claim_attempt(control, plan_sha256, attempt_id, item):
    """Durably consume one precharged worker lease before any owned source open."""
    control = Path(control)
    lock_path = control / "attempt-claims.lock"
    require(not lock_path.is_symlink(), "Attempt claim lock cannot be a symlink")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = control / "ledger.json"
        ledger, _ = local_json(ledger_path, 128 * 1024)
        require(ledger["plan_sha256"] == plan_sha256 and ledger["reserved_encoded_bytes"] ==
                sum(row["reserved_encoded_bytes"] for row in ledger["attempts"])
                and ledger["reserved_encoded_bytes"] <= MAX_RESERVED_ENCODED_BYTES, "Worker budget ledger differs")
        matches = [row for row in ledger["attempts"] if row.get("attempt_id") == attempt_id]
        require(len(matches) == 1, "Worker requires one existing precharged attempt")
        attempt = matches[0]
        require(attempt["month"] == item["month"] and attempt["source_sha256"] == item["sha256"]
                and attempt["reserved_encoded_bytes"] == item["bytes"] + 1 and attempt["status"] == "PRECHARGED"
                and sum(row["month"] == item["month"] for row in ledger["attempts"]) <= MAX_ATTEMPTS_PER_FILE,
                "Worker attempt is unreserved, already claimed, settled, or mismatched")
        attempt.update(status="WORKER_CLAIMED", worker_pid=os.getpid(), claimed_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(ledger_path, ledger)


def worker(request_path):
    request, _ = local_json(request_path, 65536)
    identity, public = plan_materialization(request["staging_plan_path"], request["approval_path"], request["output_directory"])
    require(public["plan_sha256"] == request["plan_sha256"], "Worker materialization plan changed")
    selected = [item for item in identity["files"] if item["month"] == request["month"]]
    require(len(selected) == 1, "Worker must select one planned month")
    output = Path(request["output_directory"]).absolute()
    control = output.parent / "usgs-materialization-control"
    require(Path(request_path).resolve().parent == control.resolve(), "Worker request must belong to its private controller")
    descriptor = request.get("provider_lock_fd")
    require(type(descriptor) is int and descriptor >= 0, "Inherited provider lock descriptor required")
    held, expected = os.fstat(descriptor), (output.parent / "provider.lock").lstat()
    require(stat.S_ISREG(held.st_mode) and stat.S_ISREG(expected.st_mode)
            and (held.st_dev, held.st_ino) == (expected.st_dev, expected.st_ino), "Inherited provider lock identity differs")
    # The child retains this shared open-file description until process exit,
    # keeping the hotset lock even if its parent dies. Never unlock it here.
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    claim_attempt(control, public["plan_sha256"], request.get("attempt_id"), selected[0])
    return {**copy_exact_owned_source(selected[0], execution_enabled=True), "attempt_id": request["attempt_id"]}


def reserve_attempt(ledger, path, item):
    attempts = sum(entry["month"] == item["month"] for entry in ledger["attempts"])
    charge = item["bytes"] + 1
    require(attempts < MAX_ATTEMPTS_PER_FILE and ledger["reserved_encoded_bytes"] + charge <= MAX_RESERVED_ENCODED_BYTES,
            "Persistent owned-file materialization allowance exhausted")
    ledger["reserved_encoded_bytes"] += charge
    ledger["attempts"].append({"attempt_id": uuid.uuid4().hex, "month": item["month"], "source_sha256": item["sha256"], "reserved_encoded_bytes": charge,
                               "status": "PRECHARGED", "observed_encoded_bytes": None})
    atomic_json(path, ledger)
    return len(ledger["attempts"]) - 1


def execute_materialization(staging_plan_path, approval_path, output_directory, *, approved_plan_sha256):
    identity, public = plan_materialization(staging_plan_path, approval_path, output_directory)
    require(approved_plan_sha256 == public["plan_sha256"], "Exact reviewed materialization plan digest required")
    output = Path(output_directory).absolute()
    durable_directory(output)
    provider_lock = output.parent / "provider.lock"
    require(not provider_lock.is_symlink(), "Shared provider lock cannot be a symlink")
    with provider_lock.open("a+") as exclusive:
        fcntl.flock(exclusive.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return _materialize_locked(identity, public, staging_plan_path, approval_path, output, exclusive.fileno())
        finally:
            fcntl.flock(exclusive.fileno(), fcntl.LOCK_UN)


def _materialize_locked(identity, public, staging_plan_path, approval_path, output, provider_lock_fd):
    control = output.parent / "usgs-materialization-control"
    durable_directory(control)
    lock_path = control / "writer.lock"
    require(not lock_path.is_symlink(), "Materialization lock cannot be a symlink")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = control / "ledger.json"
        if ledger_path.exists():
            ledger, _ = local_json(ledger_path, 128 * 1024)
            require(ledger["plan_sha256"] == public["plan_sha256"] and
                    ledger["reserved_encoded_bytes"] == sum(row["reserved_encoded_bytes"] for row in ledger["attempts"])
                    and ledger["reserved_encoded_bytes"] <= MAX_RESERVED_ENCODED_BYTES, "Materialization ledger differs; cannot reset charges")
        else:
            ledger = {"schema": "hydra-owned-materialization-ledger/v1", "plan_sha256": public["plan_sha256"],
                      "reserved_encoded_bytes": 0, "attempts": []}
            atomic_json(ledger_path, ledger)
        complete, reused = [], 0
        for item in identity["files"]:
            target = Path(item["destination"])
            if verified_copy(target, item):
                complete.append(item["month"]); reused += 1; continue
            require(not target.exists() and not target.is_symlink(), "Existing owned copy failed authentication; no overwrite")
            index = reserve_attempt(ledger, ledger_path, item)
            attempt_id = ledger["attempts"][index]["attempt_id"]
            request_path = control / ("request-" + item["month"] + "-" + uuid.uuid4().hex + ".json")
            atomic_json(request_path, {"staging_plan_path": str(Path(staging_plan_path).resolve()),
                        "approval_path": str(Path(approval_path).resolve()), "output_directory": str(output),
                        "plan_sha256": public["plan_sha256"], "month": item["month"],
                        "attempt_id": attempt_id, "provider_lock_fd": provider_lock_fd})
            try:
                child = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(request_path)],
                                       capture_output=True, text=True, timeout=TIMEOUT_SECONDS, pass_fds=(provider_lock_fd,))
                require(child.returncode == 0, "Owned-file worker stopped; its error is sanitized")
                result = json.loads(child.stdout)
                require(result["status"] == "VERIFIED_OWNED_COPY" and result["attempt_id"] == attempt_id and result["month"] == item["month"]
                        and result["encoded_bytes_read"] == item["bytes"] and verified_copy(target, item), "Worker copy receipt differs")
                ledger, _ = local_json(ledger_path, 128 * 1024)
                require(ledger["attempts"][index]["attempt_id"] == attempt_id and ledger["attempts"][index]["status"] == "WORKER_CLAIMED",
                        "Worker did not durably claim its charged attempt")
                ledger["attempts"][index].update(status="VERIFIED", observed_encoded_bytes=item["bytes"], source_was_dataless=result["source_was_dataless"])
                complete.append(item["month"])
            except BaseException as error:
                ledger, _ = local_json(ledger_path, 128 * 1024)
                require(ledger["attempts"][index]["attempt_id"] == attempt_id, "Attempt ledger changed unexpectedly")
                ledger["attempts"][index].update(status="FAILED_OR_UNSETTLED", error_type=type(error).__name__)
                atomic_json(ledger_path, ledger)
                summary = {**public, "status": "STOPPED_WITH_RETAINED_CHARGES", "completed_files": len(complete),
                           "failed_month": item["month"], "error_type": type(error).__name__,
                           "reserved_encoded_bytes": ledger["reserved_encoded_bytes"],
                           "attempt_states": dict(Counter(row["status"] for row in ledger["attempts"])), "fileprovider_http_requests": None,
                           "fileprovider_activity_after_timeout": "NOT_OBSERVED_MAY_CONTINUE", "direct_gcs_calls": 0}
                atomic_json(control / "summary.json", summary)
                return summary
            atomic_json(ledger_path, ledger)
        summary = {**public, "status": "VERIFIED_OWNED_USGS_COPIES_READY", "completed_files": len(complete),
                   "reused_files": reused, "reserved_encoded_bytes": ledger["reserved_encoded_bytes"],
                   "verified_copy_bytes": TOTAL_BYTES, "attempt_states": dict(Counter(row["status"] for row in ledger["attempts"])),
                   "fileprovider_http_requests": None,
                   "fileprovider_wire_bytes": None, "direct_gcs_calls": 0,
                   "created_at_utc": datetime.now(timezone.utc).isoformat()}
        atomic_json(control / "summary.json", summary)
        return summary


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        try:
            result = worker(sys.argv[2])
        except BaseException as error:
            print(json.dumps({"status": "FAILED", "error_type": type(error).__name__}))
            raise SystemExit(1)
        print(json.dumps(result, sort_keys=True))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-plan", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved-plan-sha256")
    args = parser.parse_args()
    if args.execute:
        if not args.approved_plan_sha256:
            parser.error("Execution requires the exact reviewed materialization plan digest")
        result = execute_materialization(args.staging_plan, args.approval, args.output_directory,
                                         approved_plan_sha256=args.approved_plan_sha256)
    else:
        result = plan_materialization(args.staging_plan, args.approval, args.output_directory)[1]
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"].startswith("STOPPED"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
