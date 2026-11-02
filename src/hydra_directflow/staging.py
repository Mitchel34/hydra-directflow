"""Budgeted reads of existing, generation-pinned development source objects.

No compute provisioning, cloud writes, permission changes or protected scoring.
The persistent SQLite precharge commits before each GET and survives failures.
Only provider-owned raw copies can be evicted after verified basin-cache commit.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import stat
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from .reference import digest, require
from .local_data import verified_local_bytes
from .lineage import authenticate_bundle_identity

GIB = 1024**3
MAX_BODY = 100 * GIB
MAX_GETS = 20000
MAX_ARCHIVE = 16 * 1024**2
MAX_COMMIT = 16384
RAW_HOTSET = 2 * GIB


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_directory(path):
    path = Path(path)
    missing, cursor = [], path
    while not cursor.exists():
        require(not cursor.is_symlink(), "Dangling staging directory link")
        missing.append(cursor); cursor = cursor.parent
    require(cursor.is_dir(), "Staging parent is not a directory")
    for item in reversed(missing):
        item.mkdir(mode=0o700)
        sync_directory(item.parent)
    require(not path.is_symlink() and stat.S_ISDIR(path.lstat().st_mode), "Staging directory must be real")


def atomic_json(path, value):
    path = Path(path)
    stage = path.with_name(path.name + ".writing-" + uuid.uuid4().hex)
    with stage.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    stage.replace(path); sync_directory(path.parent)


def conservative_cost_microusd(body_bytes, attempts):
    # $0.23/GiB transfer + $0.05/GiB retrieval; $0.065/1000 GETs.
    return (body_bytes * 280000 + GIB - 1) // GIB + attempts * 65


class SourceBudget:
    def __init__(self, directory, approval_sha256, *, body_cap=MAX_BODY, get_cap=MAX_GETS, usd_cap=35):
        require(re.fullmatch(r"[0-9a-f]{64}", approval_sha256), "Approved envelope identity required")
        require(type(body_cap) is int and 0 < body_cap <= MAX_BODY and type(get_cap) is int and 0 < get_cap <= MAX_GETS,
                "Bounded body/attempt allowances required")
        require(type(usd_cap) in (int, float) and 0 < usd_cap <= 35, "Staging estimate allowance cannot exceed approved35USD")
        self.path = Path(directory).absolute(); durable_directory(self.path)
        database = self.path / "source-budget.sqlite3"
        for item in (database, database.with_name(database.name + "-wal"), database.with_name(database.name + "-shm")):
            if item.exists() or item.is_symlink():
                info = item.lstat()
                require(stat.S_ISREG(info.st_mode) and not getattr(info, "st_flags", 0) & 0x40000000,
                        "Budget files must be resident regular files, never links")
        self.connection = sqlite3.connect(database, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY CHECK(id=1), approval TEXT NOT NULL, body_cap INTEGER, get_cap INTEGER, cost_cap_micro INTEGER, charged_body INTEGER, attempts INTEGER)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, identity_sha TEXT, object_sha TEXT, kind TEXT, reserved INTEGER, observed INTEGER, status TEXT, error_type TEXT, created_utc TEXT)")
        row = self.connection.execute("SELECT approval,body_cap,get_cap,cost_cap_micro FROM budget WHERE id=1").fetchone()
        identity = (approval_sha256, body_cap, get_cap, round(usd_cap * 1000000))
        if row is None:
            self.connection.execute("INSERT INTO budget VALUES(1,?,?,?,?,0,0)", identity)
            self.connection.commit(); sync_directory(self.path)
        else:
            require(row == identity, "Persistent staging budget cannot be reset or enlarged")
        recorded = self.connection.execute("SELECT charged_body,attempts FROM budget WHERE id=1").fetchone()
        summed = self.connection.execute("SELECT COALESCE(SUM(reserved),0),COUNT(*) FROM attempts").fetchone()
        require(recorded == summed and 0 <= recorded[0] <= body_cap and 0 <= recorded[1] <= get_cap,
                "Persistent source budget totals do not reconcile")
        os.chmod(self.path / "source-budget.sqlite3", 0o600)

    def reserve(self, identity, reserved_bytes, kind):
        require(type(reserved_bytes) is int and reserved_bytes > 0, "Positive read reservation required")
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            row = self.connection.execute("SELECT body_cap,get_cap,cost_cap_micro,charged_body,attempts FROM budget WHERE id=1").fetchone()
            proposed_body, proposed_gets = row[3] + reserved_bytes, row[4] + 1
            require(proposed_body <= row[0] and proposed_gets <= row[1]
                    and conservative_cost_microusd(proposed_body, proposed_gets) <= row[2], "Persistent source-read allowance exhausted")
            self.connection.execute("UPDATE budget SET charged_body=?,attempts=? WHERE id=1", (proposed_body, proposed_gets))
            cursor = self.connection.execute("INSERT INTO attempts(identity_sha,object_sha,kind,reserved,observed,status,created_utc) VALUES(?,?,?,?,0,'PRECHARGED',?)",
                                            (digest(identity), identity["sha256"], kind, reserved_bytes, datetime.now(timezone.utc).isoformat()))
            self.connection.commit()
            return cursor.lastrowid
        except BaseException:
            self.connection.rollback(); raise

    def settle(self, attempt, status, observed_bytes, error_type=None):
        require(status in ("VERIFIED", "FAILED") and type(observed_bytes) is int and observed_bytes >= 0, "Bounded attempt settlement required")
        self.connection.execute("UPDATE attempts SET observed=?,status=?,error_type=? WHERE id=?", (observed_bytes, status, error_type, attempt))
        self.connection.commit()

    def snapshot(self):
        row = self.connection.execute("SELECT body_cap,get_cap,cost_cap_micro,charged_body,attempts FROM budget WHERE id=1").fetchone()
        states = dict(self.connection.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status").fetchall())
        observed = self.connection.execute("SELECT COALESCE(SUM(observed),0) FROM attempts").fetchone()[0]
        return dict(body_cap=row[0], get_cap=row[1], cost_estimate_cap_usd=row[2] / 1e6,
                    precharged_body_bytes=row[3], attempted_gets=row[4], observed_body_bytes=observed,
                    conservative_precharge_estimate_usd=conservative_cost_microusd(row[3], row[4]) / 1e6,
                    attempt_states=states, billing_invoice_reconciled=False)

    def close(self):
        self.connection.close()


class DenyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise ValueError("Source redirects are prohibited")


@contextmanager
def network_deadline(seconds=180):
    require(signal.getitimer(signal.ITIMER_REAL)[0] == 0, "Existing process deadline must not be replaced")
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError("Bounded source attempt deadline reached")
    signal.signal(signal.SIGALRM, expired); signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)


def object_identity(value, maximum, *, bytes_required, allowed_buckets):
    require(isinstance(value, dict) and re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))), "Exact source SHA required")
    size = value.get("bytes")
    require(size is None and not bytes_required or type(size) is int and 0 < size <= maximum, "Source byte size exceeds reader bound")
    uri = value.get("durable_uri")
    if uri is None and value.get("uri_generation"):
        uri = value["uri_generation"].rsplit("#", 1)[0]
    parsed = urllib.parse.urlsplit(str(uri))
    require(parsed.scheme == "gs" and parsed.netloc in allowed_buckets and parsed.path.startswith("/")
            and len(parsed.path) > 1 and not parsed.query and not parsed.fragment, "Source URI is outside approved sealed buckets")
    generation = "" if value.get("generation") is None else str(value["generation"])
    if not generation and value.get("uri_generation"):
        generation = value["uri_generation"].rsplit("#", 1)[-1]
    require(re.fullmatch(r"[1-9][0-9]{0,24}", generation), "Immutable source generation required")
    return dict(durable_uri=uri, generation=generation, sha256=value["sha256"], bytes=size)


class GCSClient:
    def __init__(self):
        self.token, self.token_time = None, 0.
        self.credential_requests = self.network_requests = 0
        self.opener = urllib.request.build_opener(DenyRedirect())

    def invalidate_token(self):
        """Discard only this process's token; the next open uses the same CLI."""
        self.token, self.token_time = None, 0.

    def open(self, identity):
        if self.token is None or time.monotonic() - self.token_time > 2700:
            executable = shutil.which("gcloud")
            require(executable is not None, "Existing gcloud authentication is required")
            self.credential_requests += 1
            result = subprocess.run([executable, "auth", "print-access-token"], capture_output=True, text=True, timeout=20)
            require(result.returncode == 0, "Existing cloud authentication could not refresh")
            token = result.stdout.strip()
            require(20 <= len(token) <= 16384 and not any(c.isspace() for c in token), "Unexpected token format")
            self.token, self.token_time = token, time.monotonic()
        source = urllib.parse.urlsplit(identity["durable_uri"])
        url = "https://storage.googleapis.com/" + source.netloc + "/" + urllib.parse.quote(source.path[1:], safe="/") + "?generation=" + identity["generation"]
        request = urllib.request.Request(url, headers={"Authorization": "Bearer " + self.token, "Accept-Encoding": "identity"})
        self.network_requests += 1
        return self.opener.open(request, timeout=30)


def verified_hit(path, identity, maximum):
    if not path:
        return False
    try:
        verified_local_bytes(path, identity["sha256"], expected_bytes=identity.get("bytes"), max_bytes=maximum)
        return True
    except (OSError, ValueError):
        return False


class DevelopmentSourceProvider:
    """Single-writer source delivery callback for corpus cache misses."""
    def __init__(self, *, plan, directory, budget, client=None, max_attempts_per_object=3):
        require(plan.get("schema") == "hydra-development-staging-plan/v1", "Fixed development staging plan required")
        self.plan, self.directory, self.budget = plan, Path(directory).absolute(), budget
        root = Path(__file__).resolve().parents[2] / ".local"
        require(self.directory.resolve().is_relative_to(root.resolve()), "Provider files must remain in ignored local storage")
        durable_directory(self.directory)
        self.client = client if client is not None else GCSClient()
        self.cycles = {row["cycle_id"]: row for row in plan["cycles"]}
        require(len(self.cycles) == len(plan["cycles"]), "Duplicate planned cycle")
        self.allowed_buckets = set()
        for row in plan["cycles"]:
            for key in ("commit_identity", "bundle_identity"):
                value = row.get(key)
                if value:
                    uri = value.get("durable_uri") or value.get("uri_generation", "").rsplit("#", 1)[0]
                    self.allowed_buckets.add(urllib.parse.urlsplit(uri).netloc)
        require(self.allowed_buckets and "" not in self.allowed_buckets, "Planned source buckets missing")
        require(type(max_attempts_per_object) is int and 1 <= max_attempts_per_object <= 3, "Bounded source retries required")
        self.max_attempts_per_object = max_attempts_per_object
        self.owner = digest({"directory": str(self.directory.resolve()), "approval": plan["sources"]["budget_approval"]["sha256"],
                             "index": plan["sources"]["qualified_index"]["sha256"]})
        self.lock = None
        self.hits = self.downloads = self.evictions = 0

    def __enter__(self):
        lock_path = self.directory / "provider.lock"
        require(not lock_path.is_symlink(), "Provider lock cannot be a symlink")
        self.lock = lock_path.open("a+")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close(); self.lock = None; raise
        for name in ("commits", "raw"):
            durable_directory(self.directory / name)
        return self

    def __exit__(self, *unused):
        if self.lock:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN); self.lock.close()
            self.lock = None

    def _transfer(self, identity, destination, maximum, kind):
        if verified_hit(destination, identity, maximum):
            self.hits += 1; return
        require(not destination.exists() and not destination.is_symlink(), "Existing staged source failed verification")
        limit = identity["bytes"] if identity["bytes"] is not None else maximum
        raw_bytes = sum(p.stat().st_size for folder in (self.directory / "raw", self.directory / "usgs") if folder.exists()
                        for p in folder.iterdir() if p.is_file())
        require(raw_bytes + limit <= RAW_HOTSET and shutil.disk_usage(self.directory).free > limit * 2 + 256 * 1024**2,
                "Raw staging or free disk limit reached")
        refreshed_after_401 = False
        for retry in range(self.max_attempts_per_object):
            attempt = self.budget.reserve(identity, limit + 1, kind)
            temporary = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
            observed = 0
            try:
                with network_deadline(), self.client.open(identity) as response:
                    require(response.status == 200 and response.headers.get("x-goog-generation") == identity["generation"], "Source response/generation differs")
                    require(response.headers.get("Content-Encoding", "identity") == "identity", "Encoded source responses prohibited")
                    length = response.headers.get("Content-Length", "")
                    require(length.isdigit() and 0 < int(length) <= limit, "Source content length exceeds reservation")
                    require(identity["bytes"] is None or int(length) == identity["bytes"], "Source size differs")
                    hasher = hashlib.sha256()
                    with temporary.open("xb") as stream:
                        while True:
                            block = response.read(min(1024**2, limit + 1 - observed))
                            if not block: break
                            observed += len(block)
                            require(observed <= limit, "Source exceeded reserved bytes")
                            hasher.update(block); stream.write(block)
                        require(observed == int(length) and hasher.hexdigest() == identity["sha256"], "Source SHA256 or size differs")
                        stream.flush(); os.fsync(stream.fileno())
                    temporary.rename(destination); sync_directory(destination.parent)
                self.budget.settle(attempt, "VERIFIED", observed)
                self.downloads += 1; return
            except BaseException as error:
                http_status = error.code if isinstance(error, urllib.error.HTTPError) else None
                error_type = f"HTTPError_{http_status}" if type(http_status) is int and 100 <= http_status <= 599 else type(error).__name__
                self.budget.settle(attempt, "FAILED", observed, error_type)
                if temporary.exists(): temporary.unlink()
                if isinstance(error, urllib.error.HTTPError):
                    # Do not read/log the error body, URL, headers or token.
                    error.close()
                if http_status == 401 and not refreshed_after_401 and retry + 1 < self.max_attempts_per_object:
                    invalidate = getattr(self.client, "invalidate_token", None)
                    if callable(invalidate):
                        invalidate()
                        refreshed_after_401 = True
                        # Retry only through the next normal iteration: reserve
                        # first, then refresh with the existing credential CLI.
                        continue
                # Integrity, disk, interruption and budget failures never retry.
                transient = isinstance(error, (TimeoutError, ConnectionError, urllib.error.URLError))
                if isinstance(error, urllib.error.HTTPError):
                    transient = error.code in (408, 429, 500, 502, 503, 504)
                if not transient or retry + 1 == self.max_attempts_per_object: raise
                time.sleep((1, 3, 5)[retry])

    def __call__(self, cycle_id, sealed_row):
        require(self.lock is not None, "Source provider writer lock required")
        row = self.cycles[cycle_id]
        require(row["sealed_row_sha256"] == digest(sealed_row), "Planned cycle differs from authenticated sealed index")
        spec = {"cycle_id": cycle_id}
        if sealed_row["location_kind"] == "commit":
            identity = object_identity(sealed_row["source_locations"]["commit"], MAX_COMMIT, bytes_required=False, allowed_buckets=self.allowed_buckets)
            target = self.directory / "commits" / (cycle_id + ".json")
            candidates = [str(target)] + [item["path"] if isinstance(item, dict) else item for item in row.get("local_commit_candidates", [])]
            chosen = next((path for path in candidates if verified_hit(path, identity, MAX_COMMIT)), None)
            if chosen is None:
                self._transfer(identity, target, MAX_COMMIT, "commit"); chosen = str(target)
            else:
                self.hits += 1
            spec["commit_path"] = chosen
        bundle, _, _ = authenticate_bundle_identity(spec, sealed_row)
        identity = object_identity(bundle, MAX_ARCHIVE, bytes_required=True, allowed_buckets=self.allowed_buckets)
        target = self.directory / "raw" / (cycle_id + ".tar")
        candidates = [str(target)] + [item["path"] if isinstance(item, dict) else item for item in row.get("local_archive_candidates", [])]
        chosen = next((path for path in candidates if verified_hit(path, identity, MAX_ARCHIVE)), None)
        if chosen is None:
            self._transfer(identity, target, MAX_ARCHIVE, "archive"); chosen = str(target)
        else:
            self.hits += 1
        spec["archive_path"] = chosen
        return {"status": "READY", "spec": spec,
                "ownership": {"owned_archive": Path(chosen).absolute() == target, "archive_path": chosen,
                              "owner_run_id": self.owner, "archive_sha256": identity["sha256"]}}

    def on_cache_committed(self, receipt, provider_result):
        require(self.lock is not None, "Source provider writer lock required for owned-stage cleanup")
        ownership = provider_result["ownership"]
        if not ownership["owned_archive"]: return
        require(receipt.get("cache_payload_verified") is True
                and re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("cache_key", "")))
                and re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("cache_commit_sha256", ""))),
                "Verified durable cache commitment required before raw eviction")
        path = Path(ownership["archive_path"])
        require(ownership["owner_run_id"] == self.owner and path.parent.resolve() == (self.directory / "raw").resolve()
                and receipt["archive_sha256"] == ownership["archive_sha256"], "Owned-stage deletion proof differs")
        verified_local_bytes(path, ownership["archive_sha256"], max_bytes=MAX_ARCHIVE)
        path.unlink(); sync_directory(path.parent); self.evictions += 1

    def summary(self):
        return dict(local_hits=self.hits, objects_downloaded=self.downloads, owned_raw_files_evicted=self.evictions,
                    budget=self.budget.snapshot(), network_requests=getattr(self.client, "network_requests", None),
                    credential_requests=getattr(self.client, "credential_requests", None), cloud_jobs=0, cloud_writes=0)
