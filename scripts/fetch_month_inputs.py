#!/usr/bin/env python3
"""Plan or execute bounded, generation-pinned reads of existing HRRR objects.

No cloud compute or cloud writes. Planning and ordinary tests do not request
credentials or contact Google. Request bodies, paths and tokens stay private.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hydra_directflow.local_data import USGS_MANIFEST_SHA256, WEIGHTS_SHA256
from hydra_directflow.multicycle import INDEX_SHA256, qualified_cycle_rows

PRIVATE_ROOT = ROOT / ".local"
BYTE_CAP = 1024**3
OBJECT_CAP = 137
WALL_CAP = 900
ARCHIVE_CAP = 16 * 1024**2
COMMIT_CAP = 16384
BUCKET_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def resident_bytes(path, maximum):
    path = Path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not getattr(before, "st_flags", 0) & 0x40000000,
            "Input must be a resident regular file")
    require(0 < before.st_size <= maximum, "Local input exceeds byte bound")
    raw = path.read_bytes()
    after = path.lstat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "Local input changed")
    require(len(raw) == before.st_size, "Local input length changed")
    return raw


def verified_hit(path, identity, maximum):
    if not path:
        return False
    try:
        raw = resident_bytes(path, maximum)
        return len(raw) == identity["bytes"] and digest(raw) == identity["sha256"]
    except (OSError, ValueError):
        return False


def object_identity(identity, maximum):
    require(isinstance(identity, dict), "Object identity required")
    require(type(identity.get("bytes")) is int and 0 < identity["bytes"] <= maximum,
            "Object byte bound missing or exceeded")
    require(re.fullmatch(r"[0-9a-f]{64}", identity.get("sha256", "")), "Exact object SHA256 required")
    uri = urllib.parse.urlsplit(identity.get("durable_uri", ""))
    require(uri.scheme == "gs" and BUCKET_PATTERN.fullmatch(uri.netloc) and uri.path.startswith("/")
            and len(uri.path) > 1 and not uri.query and not uri.fragment, "Exact gs object URI required")
    generation = str(identity.get("generation", ""))
    require(re.fullmatch(r"[1-9][0-9]{0,24}", generation), "Pinned positive object generation required")
    return {"bytes": identity["bytes"], "sha256": identity["sha256"],
            "durable_uri": identity["durable_uri"], "generation": generation}


def media_url(identity):
    obj = urllib.parse.urlsplit(identity["durable_uri"])
    return "https://storage.googleapis.com/" + obj.netloc + "/" + urllib.parse.quote(obj.path[1:], safe="/") + "?generation=" + identity["generation"]


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_directory(path):
    """Publish newly created directory links durably before using a ledger."""
    path = Path(path)
    pending, current = [], path
    while not current.exists():
        require(not current.is_symlink(), "Cache directory cannot be a dangling symlink")
        pending.append(current)
        current = current.parent
    for item in reversed(pending):
        item.mkdir(exist_ok=True)
        require(item.is_dir() and not item.is_symlink(), "Cache directory must be real")
        sync_directory(item.parent)
    require(path.is_dir() and not path.is_symlink(), "Cache directory must be real")


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".writing")
    require(not tmp.exists() and not tmp.is_symlink(), "Stale journal staging file requires review")
    with tmp.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)


def private_destination(path):
    path = Path(path).absolute()
    require(not path.is_symlink(), "Cache directory cannot be a symlink")
    try:
        path.resolve().relative_to(PRIVATE_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Cache must remain in this checkout's ignored .local directory") from exc
    require(path.resolve() != PRIVATE_ROOT.resolve(), "Use a dedicated cache subdirectory")
    return path


def load_inventory(path, index_path):
    raw = resident_bytes(path, 2 * 1024**2)
    inventory = json.loads(raw)
    require(inventory.get("schema") == "hydra-private-month-retrieval-inventory/v1"
            and inventory.get("qualified_index_sha256") == INDEX_SHA256, "Unsupported inventory/index binding")
    require(inventory.get("month") == "2023-08", "This bounded retrieval package is August 2023 only")
    rows = inventory["cycles"]
    start = datetime(2023, 7, 31, 6, tzinfo=timezone.utc)
    expected = [(start + timedelta(hours=6 * i)).strftime("%Y%m%d_%HZ") for i in range(127)]
    require([row["cycle_id"] for row in rows] == expected, "Exact 127-cycle month plus warmup required")
    sealed = qualified_cycle_rows(index_path, expected)
    for row in rows:
        current = sealed[row["cycle_id"]]
        require(row["source_locations"] == current["source_locations"]
                and row["location_kind"] == current["location_kind"] == "commit"
                and row["data_complete"] is current["data_complete"], "Inventory differs from sealed index")
        object_identity(current["source_locations"]["commit"], COMMIT_CAP)
    return raw, inventory, sealed


def make_plan(inventory_path, index_path, cache_directory, *, weights_path, usgs_manifest_path):
    raw, inventory, sealed = load_inventory(inventory_path, index_path)
    directory = private_destination(cache_directory)
    require(digest(resident_bytes(weights_path, 128 * 1024)) == WEIGHTS_SHA256, "Candidate weights differ")
    manifest_raw = resident_bytes(usgs_manifest_path, 2 * 1024**2)
    require(digest(manifest_raw) == USGS_MANIFEST_SHA256, "USGS source manifest differs")
    native = {Path(entry["path"]).stem: entry for entry in json.loads(manifest_raw)["native_partitions"]}
    partitions = inventory["usgs_partitions"]
    require([part["month"] for part in partitions] == ["2023-07", "2023-08", "2023-09"], "Three supporting USGS months required")
    for part in partitions:
        original = native[part["month"]]
        require(part["sha256"] == original["sha256"] and part["bytes"] == original["bytes"]
                and verified_hit(part["path"], original, 512 * 1024), "USGS partition identity/residency differs")
    identity = {"inventory_sha256": digest(raw), "qualified_index_sha256": INDEX_SHA256,
                "usgs_manifest_sha256": USGS_MANIFEST_SHA256, "weights_sha256": WEIGHTS_SHA256,
                "script_sha256": digest(resident_bytes(__file__, 256 * 1024)),
                "reader_sha256": digest(resident_bytes(ROOT / "src/hydra_directflow/multicycle.py", 256 * 1024)),
                "cache_location_sha256": digest(str(directory.resolve()).encode()),
                "response_body_byte_cap": BYTE_CAP, "object_attempt_cap": OBJECT_CAP,
                "wall_seconds_cap": WALL_CAP}
    public = {"schema": "hydra-bounded-existing-object-read/v1", "status": "PLANNED_NOT_EXECUTED",
              "month": "2023-08", "cycle_count": len(sealed), "origin_count": 744,
              "plan_sha256": digest(canonical(identity)), **identity,
              "cloud_jobs": 0, "cloud_writes": 0, "credential_requests": 0, "network_requests": 0,
              "budget_semantics": "Persistent precharge of expected response body bytes plus one before every attempted GET, including failed reads; headers and transport overhead excluded",
              "cost_limit_semantics": "Byte/attempt bounds only; provider dollar charges require separately approved rates and billing reconciliation"}
    return public, inventory, sealed, directory


@contextmanager
def wall_alarm(seconds=WALL_CAP):
    """Interrupt blocked socket/token calls at the wall deadline on local Unix."""
    require(signal.getitimer(signal.ITIMER_REAL)[0] == 0, "An existing process alarm must not be replaced")
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError("Transfer wall deadline reached")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class Budget:
    def __init__(self, directory, plan_sha256, clock=time.monotonic):
        self.path = Path(directory) / "transfer-ledger.json"
        self.clock, self.started = clock, clock()
        if self.path.exists():
            self.value = json.loads(resident_bytes(self.path, 256 * 1024))
            require(self.value["plan_sha256"] == plan_sha256 and self.value["body_byte_cap"] == BYTE_CAP
                    and self.value["attempt_cap"] == OBJECT_CAP, "Existing transfer ledger belongs to another plan")
        else:
            self.value = {"schema": "hydra-storage-read-budget/v1", "plan_sha256": plan_sha256,
                          "body_byte_cap": BYTE_CAP, "attempt_cap": OBJECT_CAP,
                          "charged_body_bytes": 0, "attempts": []}
            atomic_json(self.path, self.value)

    def guard(self):
        require(self.clock() - self.started < WALL_CAP, "Transfer wall limit reached")

    def reserve(self, identity):
        self.guard()
        charge = identity["bytes"] + 1
        require(self.value["charged_body_bytes"] + charge <= BYTE_CAP, "Persistent response byte budget exhausted")
        require(len(self.value["attempts"]) < OBJECT_CAP, "Persistent object-attempt budget exhausted")
        entry = {"object_identity_sha256": digest(canonical(identity)), "precharged_body_bytes": charge,
                 "expected_sha256": identity["sha256"], "status": "PRECHARGED_UNSETTLED"}
        self.value["charged_body_bytes"] += charge
        self.value["attempts"].append(entry)
        atomic_json(self.path, self.value)
        return len(self.value["attempts"]) - 1

    def settle(self, index, status, observed_bytes):
        self.value["attempts"][index].update(status=status, observed_body_bytes=observed_bytes)
        atomic_json(self.path, self.value)


class DenyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise ValueError("Storage redirect prohibited")


class StorageClient:
    def __init__(self):
        self.token = None
        self.credential_requests = 0
        self.network_requests = 0
        self.opener = urllib.request.build_opener(DenyRedirect())

    def open(self, identity):
        if self.token is None:
            executable = shutil.which("gcloud")
            require(executable is not None, "gcloud is required for explicitly enabled transfers")
            self.credential_requests += 1
            result = subprocess.run([executable, "auth", "print-access-token"], capture_output=True, text=True, timeout=20)
            require(result.returncode == 0, "Could not obtain an access token; authenticate locally")
            self.token = result.stdout.strip()
            require(20 <= len(self.token) <= 16384 and not any(char.isspace() for char in self.token), "Unexpected access-token format")
        request = urllib.request.Request(media_url(identity), headers={"Authorization": "Bearer " + self.token,
                                                                    "Accept-Encoding": "identity"}, method="GET")
        self.network_requests += 1
        return self.opener.open(request, timeout=30)


def transfer_object(identity, destination, budget, client):
    identity = object_identity(identity, ARCHIVE_CAP)
    destination = Path(destination)
    if verified_hit(destination, identity, ARCHIVE_CAP):
        return False
    require(not destination.exists() and not destination.is_symlink(), "Existing cache object failed authentication")
    require(shutil.disk_usage(destination.parent).free >= 2 * identity["bytes"] + 64 * 1024**2, "Insufficient cache disk headroom")
    staging = destination.with_name(destination.name + ".partial")
    require(not staging.exists() and not staging.is_symlink(), "Unsettled partial file requires review; no automatic retry")
    index = budget.reserve(identity)
    observed, complete = 0, False
    try:
        with client.open(identity) as response:
            require(response.status == 200, "Storage media response was not HTTP 200")
            require(response.headers.get("x-goog-generation") == identity["generation"], "Storage generation differs")
            require(response.headers.get("Content-Encoding", "identity") == "identity", "Encoded media response prohibited")
            require(response.headers.get("Content-Length") == str(identity["bytes"]), "Storage response size differs")
            hasher = hashlib.sha256()
            with staging.open("xb") as output:
                while True:
                    budget.guard()
                    block = response.read(min(1024**2, identity["bytes"] + 1 - observed))
                    if not block:
                        break
                    observed += len(block)
                    require(observed <= identity["bytes"], "Storage body exceeds authenticated size")
                    hasher.update(block)
                    output.write(block)
                require(observed == identity["bytes"] and hasher.hexdigest() == identity["sha256"], "Downloaded object failed authentication")
                output.flush()
                os.fsync(output.fileno())
            os.replace(staging, destination)
            sync_directory(destination.parent)
            complete = True
        budget.settle(index, "VERIFIED", observed)
        return True
    except Exception:
        budget.settle(index, "FAILED_PRECHARGE_RETAINED", observed)
        raise
    finally:
        # Remove only the partial staging file created by this call, never sources.
        if not complete and staging.exists():
            staging.unlink()


def authenticated_commit(row, sealed_row, directory, budget, client):
    identity = object_identity(sealed_row["source_locations"]["commit"], COMMIT_CAP)
    target = directory / "commits" / (row["cycle_id"] + ".json")
    candidates = [target, row.get("commit_path")]
    chosen = next((path for path in candidates if verified_hit(path, identity, COMMIT_CAP)), None)
    fetched = False
    if chosen is None:
        fetched = transfer_object(identity, target, budget, client)
        chosen = target
    commit = json.loads(resident_bytes(chosen, COMMIT_CAP))
    require(commit["schema"] == "hrrr-verified-cloud-cycle/v1" and commit["cycle"] == row["cycle_id"], "Original commit schema/cycle differs")
    bundle = object_identity(commit["bundle"], ARCHIVE_CAP)
    worker = commit["worker_manifest"]
    require(commit["bundle"].get("verified") is True and worker["cycle"] == row["cycle_id"]
            and worker["bundle_sha256"] == bundle["sha256"] and worker["bundle_bytes"] == bundle["bytes"], "Commit/bundle binding differs")
    require(bundle["durable_uri"].endswith("/cycles/" + row["cycle_id"] + ".tar"), "Bundle object cycle suffix differs")
    return Path(chosen), bundle, fetched


def execute(inventory_path, index_path, cache_directory, *, weights_path, usgs_manifest_path,
            approved_plan_sha256, client=None):
    plan, inventory, sealed, directory = make_plan(inventory_path, index_path, cache_directory,
                                                 weights_path=weights_path, usgs_manifest_path=usgs_manifest_path)
    require(approved_plan_sha256 == plan["plan_sha256"], "Exact current plan digest required before execution")
    durable_directory(directory)
    for sub in ("commits", "archives"):
        child = directory / sub
        require(not child.is_symlink(), "Private cache subdirectory cannot be a symlink")
        durable_directory(child)
    lock_path = directory / "transfer.lock"
    require(not lock_path.is_symlink(), "Transfer lock cannot be a symlink")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another transfer process owns the shared budget") from error
        try:
            with wall_alarm():
                return _execute_locked(plan, inventory, sealed, directory, index_path=index_path,
                                       weights_path=weights_path, usgs_manifest_path=usgs_manifest_path, client=client)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _execute_locked(plan, inventory, sealed, directory, *, index_path, weights_path, usgs_manifest_path, client):
    budget = Budget(directory, plan["plan_sha256"])
    client = client if client is not None else StorageClient()
    specs, fetched, hits = [], 0, 0
    try:
        for row in inventory["cycles"]:
            budget.guard()
            commit_path, bundle, got_commit = authenticated_commit(row, sealed[row["cycle_id"]], directory, budget, client)
            target = directory / "archives" / (row["cycle_id"] + ".tar")
            candidates = [target] + [entry["archive_path"] for entry in row["local_archive_candidates"]]
            chosen = next((path for path in candidates if verified_hit(path, bundle, ARCHIVE_CAP)), None)
            got_archive = False
            if chosen is None:
                got_archive = transfer_object(bundle, target, budget, client)
                chosen = target
            fetched += int(got_commit) + int(got_archive)
            hits += int(not got_commit) + int(not got_archive)
            specs.append({"cycle_id": row["cycle_id"], "archive_path": str(Path(chosen).resolve()), "commit_path": str(commit_path.resolve())})
        start = datetime(2023, 8, 1, tzinfo=timezone.utc)
        request = {"archive_specs": specs, "qualified_index_path": str(Path(index_path).resolve()),
                   "weights_path": str(Path(weights_path).resolve()), "usgs_manifest_path": str(Path(usgs_manifest_path).resolve()),
                   "usgs_paths": {entry["month"]: str(Path(entry["path"]).resolve()) for entry in inventory["usgs_partitions"]},
                   "origins_utc": [(start + timedelta(hours=i)).isoformat().replace("+00:00", "Z") for i in range(744)], "history_hours": 168}
        request_path = directory / "request.json"
        if request_path.exists():
            require(json.loads(resident_bytes(request_path, 256 * 1024)) == request, "Existing completed request differs")
        else:
            atomic_json(request_path, request)
        result = {**plan, "status": "VERIFIED_MONTH_INPUTS_READY", "request_sha256": digest(resident_bytes(request_path, 256 * 1024)),
                  "objects_fetched_this_execution": fetched, "verified_local_hits_this_execution": hits}
    except Exception as error:
        result = {**plan, "status": "STOPPED_INCOMPLETE", "error_type": type(error).__name__,
                  "verified_cycles_this_execution": len(specs), "objects_fetched_this_execution": fetched}
        result.pop("network_requests", None)
        result.update(credential_requests=getattr(client, "credential_requests", None),
                      network_requests_this_execution=getattr(client, "network_requests", None))
        atomic_json(directory / "transfer-summary.json", {**result, "precharged_body_bytes": budget.value["charged_body_bytes"],
                    "object_attempts": len(budget.value["attempts"]),
                    "network_requests_upper_bound": len(budget.value["attempts"]),
                    "wall_seconds": time.monotonic() - budget.started})
        raise
    result.pop("network_requests", None)
    result.update(precharged_body_bytes=budget.value["charged_body_bytes"], object_attempts=len(budget.value["attempts"]),
                  network_requests_upper_bound=len(budget.value["attempts"]),
                  network_requests_this_execution=getattr(client, "network_requests", None),
                  credential_requests=getattr(client, "credential_requests", None),
                  wall_seconds=time.monotonic() - budget.started)
    atomic_json(directory / "transfer-summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--usgs-manifest", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved-plan-sha256")
    args = parser.parse_args()
    kwargs = dict(weights_path=args.weights, usgs_manifest_path=args.usgs_manifest)
    if args.execute:
        if not args.approved_plan_sha256:
            parser.error("Execution requires the exact prior plan digest")
        result = execute(args.inventory, args.index, args.cache_directory, **kwargs,
                         approved_plan_sha256=args.approved_plan_sha256)
    else:
        result = make_plan(args.inventory, args.index, args.cache_directory, **kwargs)[0]
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
