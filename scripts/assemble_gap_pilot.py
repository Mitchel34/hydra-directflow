#!/usr/bin/env python3
"""Plan-first local CV1 assembly; no model/scaler fits, downloads or scoring.

Every execution has a fresh controller receipt and a fixed wall/memory bound.
The assembly module authenticates immutable sources and enforces byte limits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MAX_SECONDS = 900
MAX_RSS = 4 * 1024**3
MAX_READ = 1024**3
MAX_WRITE = 1024**3
SOURCE_PLAN = "0dad073824984fd7eeba88e4c313d77b4015dcccb94e8eb27297fcf7c6b39b0f"
SNAPSHOT = "b41630b7df9a216874eb4ba38a10176d4b72092bd6f1ce2ad4c4f369db56948c"
AUTHORITIES = {
    "reports/development-corpus-residency-integrity-v1.json": "155f563238b8bcbdb549dc761775c2b24d4171d3f1f7785acba23b4e1c956d94",
    "reports/development-coverage-regimes-v2.json": "8e82f613165a5d452c74a30783b02e1c0ae5fb5d93dbe2e338d0010c215e8c0a",
    "reports/confirmed-180-fit-admitted-runtime.json": "9a8b5526b0956125aecc2ebca9a29d20e90e6b262c136225505cde18cd92b2ec",
}
CODE = ("scripts/assemble_gap_pilot.py", "src/hydra_directflow/pilot_assembly.py",
        "src/hydra_directflow/corpus_dataset.py", "src/hydra_directflow/cache.py",
        "src/hydra_directflow/quality.py", "src/hydra_directflow/reference.py",
        "src/hydra_directflow/local_data.py", "src/hydra_directflow/trainer_bundle.py")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def private_path(path):
    path = Path(path).absolute()
    local = Path.home() / ".codex" / "hydra-directflow-runtime"
    require(path == path.resolve() and path != local and path.is_relative_to(local),
            "Private artifacts require a real path inside stable local storage")
    require(not path.is_relative_to(ROOT) or path.is_relative_to(ROOT / ".local")
            or path.is_relative_to(ROOT / ".cache"), "Private artifacts cannot enter tracked repository paths")
    return path


def public_report_path(path):
    path = Path(path).absolute()
    require(path == path.resolve() and path.parent == ROOT / "reports" and path.suffix == ".json",
            "Public summary requires an explicit reports JSON path")
    return path


def write_new(path, value):
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def authority(cache_root):
    from hydra_directflow.corpus_dataset import _read
    values = {}
    for name, expected in AUTHORITIES.items():
        raw = _read(ROOT / name, 4 * 1024**2)
        require(sha(raw) == expected, "Published assembly authority changed: " + name)
        values[name] = json.loads(raw)
    counts = values["reports/confirmed-180-fit-admitted-runtime.json"]
    cases = [x for x in counts["scenarios"] if x["epochs"] == 20 and x["simulated_label_delay_hours"] == 2]
    require(len(cases) == 1, "Unique published two-hour CV1 authority required")
    case = cases[0]
    window = next(x for x in case["fold_windows"] if x["name"] == "cv_1")
    role_names = (("fit", "gradient_fit"), ("inner_stop", "inner_stopping"), ("outer", "outer_scoring"))
    bounds = {short: {"start_utc": window[long]["start_utc"], "end_utc": window[long]["end_exclusive_utc"]}
              for short, long in role_names}
    expected = {}
    for row in case["parts"]:
        if (row["fold"], row["architecture"], row["input_arm"], row["history_hours"]) == ("cv_1", "LSTM", "full", 168):
            require(row["gauge_id"] not in expected, "Duplicate published CV1 gauge")
            expected[row["gauge_id"]] = {short: {"cases": row["actual_common_cases"][long],
                "case_ids_sha256": row["role_case_ids_sha256"][long]} for short, long in role_names}
    require(set(expected) == {"03161000", "03164000", "03479000"}, "Exact three pilot gauges required")
    monthly = values["reports/development-coverage-regimes-v2.json"]["monthly_reference_identities"]
    sources = {x["month"]: {"entry_path": str(Path(cache_root).absolute() / "entries" / x["cache_key"]),
               "cache_key": x["cache_key"], "commit_sha256": x["commit_sha256"]}
               for x in monthly if "2018-07" <= x["month"] <= "2021-07"}
    require(len(sources) == 37, "Exact 37 development months required")
    return sources, bounds, window["inner_stopping"]["refit_at_utc"], expected


def implementation():
    from hydra_directflow.corpus_dataset import _read
    return {name: sha(_read(ROOT / name, 1024**2)) for name in CODE}


def prepare(corpus, cache, output):
    from hydra_directflow.pilot_assembly import plan_metadata
    sources, roles, refit, expected = authority(cache)
    assembly = plan_metadata(corpus_root=corpus,
        integrity_receipt_path=ROOT / "reports/development-corpus-residency-integrity-v1.json",
        expected_integrity_sha256=AUTHORITIES["reports/development-corpus-residency-integrity-v1.json"],
        expected_plan_sha256=SOURCE_PLAN, expected_snapshot_sha256=SNAPSHOT,
        month_sources=sources, role_time_bounds=roles, refit_at_utc=refit,
        expected_roles=expected, max_read_bytes=MAX_READ, max_write_bytes=MAX_WRITE)
    output = private_path(output)
    require(not output.exists() and not output.is_symlink(), "Fresh assembly destination required")
    require(output.parent.is_dir(), "Existing local output parent required")
    for source in (corpus, cache):
        source = Path(source).resolve()
        require(not output.is_relative_to(source) and not source.is_relative_to(output),
                "Output must be disjoint from source directories")
    value = dict(schema="hydra-cv1-assembly-controller-plan/v1", assembly=assembly,
        output_directory=str(output), implementation=implementation(), authorities=AUTHORITIES,
        maximum_seconds=MAX_SECONDS, maximum_parent_child_rss_bytes=MAX_RSS,
        maximum_source_read_bytes=MAX_READ, maximum_output_bytes=MAX_WRITE,
        model_fits=0, scaler_fits=0, optimizer_updates=0, downloads=0,
        protected_assessment_access=False, scientific_execution_authorized=False)
    return {**value, "plan_sha256": sha(canonical(value))}


def validate_plan(value, expected):
    require(value.get("schema") == "hydra-cv1-assembly-controller-plan/v1"
            and expected == value.get("plan_sha256")
            and expected == sha(canonical({k: v for k, v in value.items() if k != "plan_sha256"})),
            "Exact assembly plan required")
    require(value["implementation"] == implementation() and value["authorities"] == AUTHORITIES,
            "Assembly implementation or authority changed")
    require((value["maximum_seconds"], value["maximum_parent_child_rss_bytes"],
             value["maximum_source_read_bytes"], value["maximum_output_bytes"]) == (MAX_SECONDS, MAX_RSS, MAX_READ, MAX_WRITE)
            and value["model_fits"] == value["scaler_fits"] == value["optimizer_updates"] == value["downloads"] == 0
            and value["protected_assessment_access"] is False and value["scientific_execution_authorized"] is False,
            "Assembly scope or limits changed")
    if "output_directory" in value:
        private_path(value["output_directory"])


def deny_network_and_models(event, args):
    if event in ("socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.bind"):
        raise RuntimeError("Assembly worker is offline")
    if event == "import" and args and str(args[0]).split(".")[0] in ("torch", "transformers"):
        raise RuntimeError("Assembly worker cannot import model runtimes")


def validated_manifest(value, manifest_raw):
    require(value["manifest_sha256"] == sha(manifest_raw) and value["manifest"] == json.loads(manifest_raw),
            "Worker result does not match the actual immutable manifest")
    return value["manifest"]


def worker(plan_path, control):
    from hydra_directflow.corpus_dataset import _read
    plan_path, control = private_path(plan_path), private_path(control)
    plan = json.loads(_read(plan_path, 16 * 1024**2))
    receipt = json.loads(_read(control / "precharged.json", 1024**2))
    validate_plan(plan, receipt["plan_sha256"])
    require(time.time() < receipt["deadline_epoch"], "Assembly allowance expired")
    write_new(control / "claimed.json", dict(plan_sha256=plan["plan_sha256"], pid=os.getpid(), claimed_at_utc=utc()))
    sys.addaudithook(deny_network_and_models)
    import psutil
    process = psutil.Process()
    def check():
        require(time.time() < receipt["deadline_epoch"], "Assembly wall deadline exceeded")
        require(process.memory_info().rss <= MAX_RSS, "Assembly memory limit exceeded")
    def deadline(*unused):
        raise TimeoutError("Assembly wall deadline exceeded")
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, max(.001, receipt["deadline_epoch"] - time.time()))
    from hydra_directflow.pilot_assembly import execute
    result = execute(plan["assembly"], output_directory=plan["output_directory"], check=check)
    check()
    write_new(control / "worker-result.json", result)


def execute_controller(plan_path, *, approved_plan_sha256, control, report):
    from hydra_directflow.corpus_dataset import _read
    plan_path = private_path(plan_path)
    plan = json.loads(_read(plan_path, 16 * 1024**2))
    validate_plan(plan, approved_plan_sha256)
    control, report = private_path(control), public_report_path(report)
    require(not control.exists() and not control.is_symlink() and not report.exists() and not report.is_symlink(),
            "Fresh controller and report required; preserve earlier attempts")
    require(not Path(plan["output_directory"]).exists(), "Existing output needs explicit recovery review")
    control.mkdir(parents=False)
    sync_directory(control.parent)
    start = time.monotonic()
    write_new(control / "precharged.json", dict(plan_sha256=approved_plan_sha256,
        started_at_utc=utc(), deadline_epoch=time.time() + MAX_SECONDS,
        maximum_seconds=MAX_SECONDS, model_fits=0, optimizer_updates=0))
    import psutil
    parent = psutil.Process()
    peak = 0
    reason = None
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    with (control / "worker.log").open("xb") as log:
        child = subprocess.Popen([sys.executable, "-I", "-B", str(Path(__file__).resolve()),
            "--worker", "--plan", str(plan_path), "--control", str(control)],
            cwd=ROOT, env=env, stdout=log, stderr=log)
        observed = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                peak = max(peak, parent.memory_info().rss + observed.memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            if peak > MAX_RSS or time.monotonic() - start > MAX_SECONDS:
                reason = "RESOURCE_LIMIT"; child.kill(); break
            time.sleep(.05)
        child.wait(timeout=10)
        log.flush(); os.fsync(log.fileno())
    passed = child.returncode == 0 and (control / "worker-result.json").is_file()
    result = dict(schema="hydra-cv1-assembly-execution/v1", status="PASSED" if passed else "FAILED_PRESERVED_ATTEMPT",
        created_at_utc=utc(), plan_sha256=approved_plan_sha256, implementation=plan["implementation"],
        authority_sha256=AUTHORITIES, elapsed_seconds=time.monotonic() - start,
        peak_parent_child_rss_bytes=peak, maximum_seconds=MAX_SECONDS, maximum_parent_child_rss_bytes=MAX_RSS,
        maximum_source_read_bytes=MAX_READ, maximum_output_bytes=MAX_WRITE,
        worker_exit_code=child.returncode, failure_reason=reason,
        private_worker_log_sha256=sha((control / "worker.log").read_bytes()),
        model_fits=0, scaler_fits=0, optimizer_updates=0, downloads=0, paid_cloud_jobs=0,
        protected_assessment_access=False, scientific_execution_authorized=False)
    if passed:
        raw = _read(control / "worker-result.json", 32 * 1024**2)
        value = json.loads(raw)
        manifest_raw = _read(Path(plan["output_directory"]) / "MANIFEST.json", 8 * 1024**2)
        manifest = validated_manifest(value, manifest_raw)
        result["worker_result_sha256"] = sha(raw)
        result["assembly_manifest_sha256"] = value["manifest_sha256"]
        result["assembly"] = {key: manifest[key] for key in ("schema", "status", "content_sha256",
            "snapshot_sha256", "source_plan_sha256", "role_time_bounds", "refit_at_utc", "quality_policy",
            "cohort_mode", "roles", "join_slot_counts", "execution", "historical_delivery_vintage_known",
            "scientific_fit_authorization", "coverage_outer", "original_gradient_gap_shape_census")}
        result["assembly"].update(parts=len(manifest["parts"]), provenance_months=len(manifest["provenance"]),
            ridge_series=[p["identity"] for p in manifest["ridge"]],
            read_bytes=value["read_bytes"], write_bytes=value["write_bytes"])
    write_new(control / "result.json", result)
    write_new(report, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "cache", "output", "plan", "control", "report"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--approved-plan-sha256")
    args = parser.parse_args()
    if args.worker:
        require(args.plan and args.control, "Explicit worker control required")
        worker(args.plan, args.control); return
    if args.execute:
        require(args.plan and args.control and args.report and args.approved_plan_sha256,
                "Explicit reviewed plan, controller and report required")
        result = execute_controller(args.plan, approved_plan_sha256=args.approved_plan_sha256,
                                    control=args.control, report=args.report)
        print(json.dumps({k: result[k] for k in ("status", "plan_sha256", "elapsed_seconds")}))
        require(result["status"] == "PASSED", "Assembly failed; retain receipts and review before retry")
        return
    require(args.corpus and args.cache and args.output and args.plan, "Explicit local source/output/plan paths required")
    private_path(args.plan)
    result = prepare(args.corpus, args.cache, args.output)
    write_new(args.plan, result)
    print(json.dumps(dict(status="PLAN_ONLY_NO_NUMERIC_READS_OR_FITS", plan_sha256=result["plan_sha256"])))


if __name__ == "__main__":
    main()
