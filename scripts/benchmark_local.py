#!/usr/bin/env python3
"""Opt-in synthetic local throughput measurement; not an epoch/science benchmark.

The fixed matrix plans 800 updates: 16 MPS cases and four CPU references, each
with 10 warmup and 30 measured steps. A separate 1,000-update ledger, 1,200-second
global wall cap, 720-second aggregate child-CPU cap, and 4 GiB active-job cap
apply. Default invocation prints the plan without importing Torch or executing.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ARCHITECTURES = ("LSTM", "Transformer", "GRU_Transformer", "Mamba2_Transformer")
HISTORY_FEATURES = ("discharge_cms", "age_hours")
WEATHER_FEATURES = ("precip_mm", "temp_k", "specific_humidity", "swe_kg_m2")
STATE_FEATURES = ("soil_water_fraction", "soil_age_hours")
WARMUP = 10
SAMPLES = 30
UPDATES_PER_CASE = WARMUP + SAMPLES
MAX_UPDATES = 1000
MAX_WALL_SECONDS = 1200
MAX_CPU_SECONDS = 720
MAX_BYTES = 4 * 1024**3
MPS_BYTES = 2 * 1024**3
MODEL_WIDTH = 64


def matrix():
    cases = []
    for architecture in ARCHITECTURES:
        for length in (24, 168):
            for batch in (32, 64):
                cases.append({"id": f"mps-{architecture}-L{length}-B{batch}", "device": "mps",
                              "architecture": architecture, "history_hours": length, "batch_size": batch})
    for architecture in ARCHITECTURES:
        cases.append({"id": f"cpu-{architecture}-L168-B64", "device": "cpu", "architecture": architecture,
                      "history_hours": 168, "batch_size": 64})
    return cases


PLANNED_UPDATES = len(matrix()) * UPDATES_PER_CASE


def specification():
    return {"matrix": matrix(), "horizons_hours": list(range(1, 19)), "history_features": HISTORY_FEATURES,
        "weather_features": WEATHER_FEATURES, "state_features": STATE_FEATURES,
        "width": MODEL_WIDTH, "dtype": "float32", "dropout": .1, "seed": 1729,
        "optimizer": "AdamW", "learning_rate": 1e-4, "foreach": False,
        "loss": "masked_mse", "mask_case": "all input masks true and all target masks true",
        "warmup_updates_per_case": WARMUP, "measured_updates_per_case": SAMPLES,
        "planned_optimizer_updates": PLANNED_UPDATES, "new_allowance_maximum_updates": MAX_UPDATES,
        "global_wall_seconds": MAX_WALL_SECONDS, "aggregate_child_cpu_seconds": MAX_CPU_SECONDS,
        "combined_active_job_rss_mps_bytes": MAX_BYTES, "mps_driver_cap_bytes": MPS_BYTES,
        "source_policy": "fixed synthetic tensors only; no source loader, research data, or assessment",
        "timing_scope": "separate copy and synchronized forward+masked-loss+backward+AdamW update; model/loss validation included, post-step finite gradient/parameter checks excluded",
        "not_measured": ["source reading", "preprocessing", "epoch duration", "cloud hardware", "forecast skill"]}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def portable(value):
    if isinstance(value, str):
        return value.replace(str(ROOT), "[repo]").replace(str(Path.home()), "[home]")
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    return value


def precharge(report_path, attempt_id):
    """Whole matrix is charged before spawning; failure/skip never refunds it."""
    directory = ROOT / ".cache"
    directory.mkdir(exist_ok=True)
    ledger_path = directory / "local-throughput-update-budget.json"
    with (directory / "local-throughput-budget.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text())
        else:
            if Path(report_path).exists():
                raise RuntimeError("Existing benchmark receipt but missing ledger; refusing reset")
            ledger = {"scope": "separate local synthetic throughput allowance",
                      "maximum_updates": MAX_UPDATES, "precharged_updates": 0, "attempts": []}
        if ledger["maximum_updates"] != MAX_UPDATES or ledger["precharged_updates"] + PLANNED_UPDATES > MAX_UPDATES:
            raise RuntimeError("Synthetic benchmark allowance exhausted; no automatic retry or increase")
        ledger["precharged_updates"] += PLANNED_UPDATES
        ledger["attempts"].append({"attempt_id": attempt_id, "precharged_updates": PLANNED_UPDATES,
                                  "spec_sha256": canonical_hash(specification()),
                                  "created_at_utc": datetime.now(timezone.utc).isoformat()})
        atomic_json(ledger_path, ledger)
    return ledger


def summaries(samples, batch_size, valid_cells):
    """Nearest-rank empirical p95: sample 29 of 30, no distributional claim."""
    result = {}
    for name in ("copy_seconds", "update_seconds"):
        values = [sample[name] for sample in samples]
        if not values or any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Positive finite measured durations required")
        result[name] = {"median": statistics.median(values), "empirical_p95": sorted(values)[math.ceil(.95 * len(values)) - 1],
                        "minimum": min(values), "maximum": max(values)}
    median = result["update_seconds"]["median"]
    result.update(examples_per_second_at_median_update=batch_size / median,
                  valid_target_cells_per_second_at_median_update=valid_cells / median,
                  percentile_method="nearest_rank", sample_count=len(samples))
    return result


def cpu_seconds(children=False):
    usage = resource.getrusage(resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def worker(args):
    started = time.monotonic()
    def deny_network(event, unused):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            raise RuntimeError("Network prohibited during synthetic throughput measurement")
    sys.addaudithook(deny_network)
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Worker directory must be inside ignored runs/")
    permit = json.loads((run_dir / "authorization.json").read_text())
    if permit != {"spec_sha256": canonical_hash(specification()), "precharged_updates": PLANNED_UPDATES}:
        raise ValueError("Worker requires exact fixed matrix precharge")
    case = next(case for case in matrix() if case["id"] == args.case)
    # Each case can consume its precharge once, including failed/import-killed
    # workers. Reusing the private worker command cannot silently retry a case.
    with (run_dir / (args.case + ".started.json")).open("x") as claim:
        json.dump({"case": args.case, "optimizer_updates_precharged": UPDATES_PER_CASE,
                   "pid": os.getpid(), "created_at_utc": datetime.now(timezone.utc).isoformat()}, claim)
    if not 0 < args.cpu_limit <= MAX_CPU_SECONDS or not 0 < args.wall_limit <= MAX_WALL_SECONDS:
        raise ValueError("Worker caps cannot exceed the total benchmark caps")
    # POSIX fallback if the parent is interrupted. The parent uses finer-grained
    # aggregate checks and a global wall deadline across every child.
    hard_cpu = max(1, math.floor(args.cpu_limit - 1))
    resource.setrlimit(resource.RLIMIT_CPU, (hard_cpu, hard_cpu))
    import torch
    import psutil
    from hydra_directflow.models.directflow import DirectFlowModel, ModelConfig, masked_mse
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = case["device"]
    result_path = run_dir / (args.case + ".json")
    telemetry_path = run_dir / (args.case + ".telemetry.json")
    result = {**case, "status": "FAIL", "optimizer_updates_confirmed": 0, "warmup_updates_completed": 0,
              "samples": [], "optimizer_updates_precharged": UPDATES_PER_CASE,
              "pid": os.getpid(), "cpu_fallback": False}
    if device == "mps" and not torch.backends.mps.is_available():
        result.update(status="UNAVAILABLE", reason="MPS unavailable; no replacement CPU run", startup_wall_seconds=time.monotonic() - started)
        atomic_json(result_path, result)
        return
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(min(1., MPS_BYTES / torch.mps.recommended_max_memory()))
    stop = threading.Event()
    monitor_errors = []
    peak = {"worker_rss_bytes": 0, "mps_driver_bytes": 0, "mps_current_bytes": 0}
    process = psutil.Process()

    def sample_resources():
        try:
            while not stop.is_set():
                sample = {"worker_rss_bytes": process.memory_info().rss,
                          "mps_driver_bytes": torch.mps.driver_allocated_memory() if device == "mps" else 0,
                          "mps_current_bytes": torch.mps.current_allocated_memory() if device == "mps" else 0,
                          "optimizer_updates_confirmed": result["optimizer_updates_confirmed"]}
                for key in peak:
                    peak[key] = max(peak[key], sample[key])
                atomic_json(telemetry_path, sample)
                stop.wait(.05)
        except Exception as error:
            monitor_errors.append(type(error).__name__)
    watcher = threading.Thread(target=sample_resources, daemon=True)
    watcher.start()

    def synchronize():
        if device == "mps":
            torch.mps.synchronize()

    def guard():
        if monitor_errors:
            raise RuntimeError("Resource telemetry failed")
        if time.monotonic() - started >= args.wall_limit or cpu_seconds() >= args.cpu_limit - 1:
            raise RuntimeError("Worker remaining wall/CPU budget reached")
        driver = torch.mps.driver_allocated_memory() if device == "mps" else 0
        if process.memory_info().rss + driver > MAX_BYTES:
            raise RuntimeError("Worker RSS plus MPS driver memory limit reached")

    try:
        torch.manual_seed(1729)
        config = ModelConfig(case["architecture"], HISTORY_FEATURES, WEATHER_FEATURES, STATE_FEATURES,
                             case["history_hours"], tuple(range(1, 19)), width=MODEL_WIDTH, dropout=.1)
        model = DirectFlowModel(config).to(device=device, dtype=torch.float32).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
        generator = torch.Generator().manual_seed(317)
        batch = case["batch_size"]
        host = []
        for shape in ((batch, case["history_hours"], 2), (batch, 18, 4), (batch, 2)):
            value = torch.randn(shape, dtype=torch.float32, generator=generator)
            host.extend((value, torch.ones_like(value, dtype=torch.bool)))
        target = torch.randn((batch, 18), dtype=torch.float32, generator=generator)
        host.extend((target, torch.ones_like(target, dtype=torch.bool)))
        result.update(parameter_count=sum(p.numel() for p in model.parameters()),
            input_copy_bytes=sum(t.numel() * t.element_size() for t in host), valid_target_cells=batch * 18,
            startup_wall_seconds=time.monotonic() - started,
            copy_semantics="CPU-to-MPS synchronous copy including targets/masks" if device == "mps" else "CPU-to-CPU explicit clone including targets/masks",
            tensors_created_outside_timing=True, post_step_integrity_checks_outside_timing=True,
            model_and_masked_loss_validation_inside_timing=True,
            torch_version=str(torch.__version__), mps_available=torch.backends.mps.is_available())
        for index in range(UPDATES_PER_CASE):
            guard()
            synchronize()
            copy_started = time.perf_counter()
            transferred = [tensor.to(device=device, copy=True, non_blocking=False) for tensor in host]
            synchronize()
            copy_elapsed = time.perf_counter() - copy_started
            update_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*transferred[:6])
            loss = masked_mse(prediction, transferred[6], transferred[7])
            loss.backward()
            optimizer.step()
            synchronize()
            update_elapsed = time.perf_counter() - update_started
            result["optimizer_updates_confirmed"] += 1
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(prediction).all()):
                raise RuntimeError("Nonfinite loss/prediction")
            if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise RuntimeError("Missing/nonfinite parameter gradients")
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise RuntimeError("Nonfinite updated parameters")
            result["last_synthetic_loss"] = float(loss.detach().cpu())
            if index < WARMUP:
                result["warmup_updates_completed"] += 1
            else:
                result["samples"].append({"sample": index - WARMUP + 1,
                                          "copy_seconds": copy_elapsed, "update_seconds": update_elapsed})
            guard()
        result.update(status="PASS", checks="finite loss/predictions, every parameter gradient, every updated parameter",
                      summary=summaries(result["samples"], batch, batch * 18))
    except Exception as error:
        result.update(error_type=type(error).__name__, error=portable(str(error))[:2000])
    finally:
        stop.set()
        watcher.join(timeout=1)
        if watcher.is_alive() or monitor_errors:
            result.update(status="FAIL", error="Resource telemetry did not complete")
        result.update(worker_observed_peaks=peak, worker_wall_seconds=time.monotonic() - started,
                      worker_cpu_seconds_before_exit=cpu_seconds(), update_count_exact_on_success=result["status"] == "PASS")
        atomic_json(result_path, result)


def supervise(case, run_dir, deadline, cpu_baseline):
    import psutil
    remaining_wall = deadline - time.monotonic()
    spent_cpu = cpu_seconds(children=True) - cpu_baseline
    remaining_cpu = MAX_CPU_SECONDS - spent_cpu
    if min(remaining_wall, remaining_cpu) <= 1:
        return {**case, "status": "NOT_RUN_RESOURCE_BUDGET", "optimizer_updates_confirmed": 0}
    result_path = run_dir / (case["id"] + ".json")
    telemetry = run_dir / (case["id"] + ".telemetry.json")
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0"})
    command = [sys.executable, str(Path(__file__).resolve()), "--execute", "--worker", "--case", case["id"],
               "--run-dir", str(run_dir), "--wall-limit", str(remaining_wall), "--cpu-limit", str(remaining_cpu)]
    peak, reason, child = 0, None, None
    started = time.monotonic()
    last_telemetry = {}
    try:
        with (run_dir / (case["id"] + ".log")).open("w") as log:
            child = subprocess.Popen(command, env=environment, stdout=log, stderr=log, cwd=ROOT)
            process, supervisor = psutil.Process(child.pid), psutil.Process()
            while child.poll() is None:
                try:
                    rss = supervisor.memory_info().rss + process.memory_info().rss
                    rss += sum(p.memory_info().rss for p in process.children(recursive=True))
                    if telemetry.exists():
                        try:
                            last_telemetry = json.loads(telemetry.read_text())
                        except (OSError, json.JSONDecodeError):
                            pass
                    combined = rss + last_telemetry.get("mps_driver_bytes", 0)
                    peak = max(peak, combined)
                    current_cpu = process.cpu_times()
                    aggregate = spent_cpu + current_cpu.user + current_cpu.system
                    if combined > MAX_BYTES:
                        reason = "combined_active_job_RSS_and_MPS_driver_limit"
                    elif time.monotonic() >= deadline:
                        reason = "global_1200_second_wall_limit"
                    elif aggregate >= MAX_CPU_SECONDS - 1:
                        reason = "aggregate_720_second_child_CPU_limit"
                    if reason:
                        child.kill()
                        break
                except psutil.NoSuchProcess:
                    break
                time.sleep(.05)
            child.wait(timeout=5)
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
    result = json.loads(result_path.read_text()) if result_path.exists() else {
        **case, "status": "FAIL", "error": "Worker produced no result",
        "optimizer_updates_confirmed": last_telemetry.get("optimizer_updates_confirmed", 0), "update_count_exact_on_success": False}
    result.update(combined_observed_peak_bytes=peak, supervised_wall_seconds=time.monotonic() - started,
                  aggregate_reaped_child_cpu_seconds=cpu_seconds(children=True) - cpu_baseline)
    if reason or child.returncode:
        result.update(status="STOPPED" if reason else "FAIL", stop_reason=reason, exit_code=child.returncode)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run the explicitly bounded fixed synthetic matrix")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/local-throughput.json")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=[case["id"] for case in matrix()], help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--wall-limit", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--cpu-limit", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "DISABLED_REQUIRES_EXECUTE", "planned_updates": PLANNED_UPDATES,
                          "maximum_new_allowance": MAX_UPDATES, "cases": len(matrix()), "spec_sha256": canonical_hash(specification())}))
        return 0
    if args.worker:
        worker(args)
        return 0
    import psutil
    from hydra_directflow.telemetry import RunJournal
    if args.report.is_symlink() or not args.report.resolve().is_relative_to((ROOT / "reports").resolve()):
        parser.error("Public receipt must be a regular path within reports/")
    started = time.monotonic()
    deadline = started + MAX_WALL_SECONDS - 1  # Reserve one second for terminal receipts.
    cpu_baseline = cpu_seconds(children=True)
    attempt_id = "local-throughput-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    ledger = precharge(args.report, attempt_id)
    run_dir = ROOT / "runs" / attempt_id
    spec = specification()
    journal = RunJournal(run_dir, spec, max_wall_seconds=MAX_WALL_SECONDS, max_rss_bytes=MAX_BYTES)
    atomic_json(run_dir / "authorization.json", {"spec_sha256": canonical_hash(spec), "precharged_updates": PLANNED_UPDATES})
    journal.event("optimizer_updates_precharged", precharged_updates=PLANNED_UPDATES, allowance_maximum=MAX_UPDATES)
    prior_path = ROOT / "reports/checkpoint-process-smoke.json"
    prior = json.loads(prior_path.read_text())
    prior_updates = prior["update_accounting"]["cumulative_precharged_updates_including_prior_qualification"]
    files = [Path(__file__).resolve(), ROOT / "requirements/model-lock.txt", ROOT / "src/hydra_directflow/telemetry.py",
             ROOT / "src/hydra_directflow/models/directflow.py", ROOT / "src/hydra_directflow/models/mamba2_math.py"]
    report = {"schema": "hydra-local-synthetic-throughput/v1", "status": "RUNNING", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "local synthetic update microbenchmark only; no epoch duration or forecast skill evidence",
        "spec": spec, "spec_sha256": canonical_hash(spec), "implementation_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in files},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "logical_cpus": os.cpu_count(), "host_memory_bytes": psutil.virtual_memory().total,
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "psutil")},
            "mps_cpu_fallback": False, "thread_count": 1, "network_access": "blocked by worker audit hook"},
        "update_accounting": {"new_allowance": ledger, "prior_synthetic_qualification_precharged_updates": prior_updates,
            "prior_qualification_receipt_sha256": file_hash(prior_path), "prior_receipts_modified": False,
            "cumulative_precharged_synthetic_updates": prior_updates + ledger["precharged_updates"],
            "real_data_engineering_updates_included": False,
            "failed_or_unstarted_cases": "remain fully charged; no refund, automatic retry, or allocation increase"},
        "cases": []}
    if args.report.exists():
        report["prior_attempts"] = [json.loads(args.report.read_text())]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, portable(report))
    try:
        for case in matrix():
            remaining_wall = deadline - time.monotonic()
            remaining_cpu = MAX_CPU_SECONDS - (cpu_seconds(children=True) - cpu_baseline)
            if min(remaining_wall, remaining_cpu) <= (20 if case["device"] == "cpu" else 1):
                result = {**case, "status": "NOT_RUN_RESOURCE_BUDGET", "optimizer_updates_confirmed": 0}
                report["cases"].append(result)
            else:
                with journal.stage("case_" + case["id"].lower().replace("-", "_")):
                    result = supervise(case, run_dir, deadline, cpu_baseline)
                    journal.event("case_completed", case_id=case["id"], status=result["status"],
                                  optimizer_updates_confirmed=result.get("optimizer_updates_confirmed", 0),
                                  combined_observed_peak_bytes=result.get("combined_observed_peak_bytes", 0))
                    report["cases"].append(result)
                    # Retain the worker receipt before stage-exit deadline checks.
                    atomic_json(args.report, portable(report))
            atomic_json(args.report, portable(report))
            summary = result.get("summary", {})
            median = summary.get("update_seconds", {}).get("median")
            print(f"{case['id']}: {result['status']}" + (f"; median update {median:.4f}s" if median else ""), flush=True)
        report["status"] = "PASS" if all(case["status"] == "PASS" for case in report["cases"]) else "PARTIAL_OR_FAILED"
        journal.finish("COMPLETED" if report["status"] == "PASS" else "STOPPED", result_status=report["status"])
    except BaseException as error:
        report.update(status="PARTIAL_OR_FAILED", error_type=type(error).__name__, error=portable(str(error))[:2000])
        if journal.status == "RUNNING":
            journal.finish("STOPPED", error_type=type(error).__name__)
    report["totals"] = {"optimizer_updates_confirmed": sum(case.get("optimizer_updates_confirmed", 0) for case in report["cases"]),
        "warmup_updates_confirmed": sum(case.get("warmup_updates_completed", 0) for case in report["cases"]),
        "measured_samples_retained": sum(len(case.get("samples", [])) for case in report["cases"]),
        "wall_seconds": time.monotonic() - started, "aggregate_reaped_child_cpu_seconds": cpu_seconds(children=True) - cpu_baseline,
        "maximum_observed_combined_bytes": max((case.get("combined_observed_peak_bytes", 0) for case in report["cases"]), default=0)}
    report["journal"] = {"status": journal.status, "spec_sha256": journal.spec_sha256,
                         "state_sha256": file_hash(run_dir / "state.json"), "events_sha256": file_hash(run_dir / "events.jsonl")}
    atomic_json(args.report, portable(report))
    print(json.dumps({"status": report["status"], **report["totals"]}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
