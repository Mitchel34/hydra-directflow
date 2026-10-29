#!/usr/bin/env python3
"""Opt-in, offline fresh-process checkpoint mechanics with synthetic tensors.

Default invocation prints the disabled plan and performs no optimizer updates.
Use --execute for the separately bounded six-update run (allowance ceiling 12).
No research datasets, fitted scalers, downloaded weights, or forecast scores.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MAX_SECONDS = 300
MAX_BYTES = 4 * 1024**3
PLANNED_UPDATES = 6
MAX_UPDATES = 12
CASES = {"cpu-lstm": ("LSTM", "cpu"), "mps-mamba2": ("Mamba2_Transformer", "mps")}
HISTORY = ("discharge_cms", "age_hours")
WEATHER = ("precip_mm", "temp_k", "specific_humidity", "swe_kg_m2")
STATE = ("soil_water_fraction", "soil_age_hours")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def portable(value):
    if isinstance(value, str):
        return value.replace(str(ROOT), "[repo]").replace(str(Path.home()), "[home]")
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    return value


def precharge(report_path, attempt_id):
    """A separate ledger: previous model-qualification receipts are read-only."""
    directory = ROOT / ".cache"
    directory.mkdir(exist_ok=True)
    ledger_path = directory / "checkpoint-process-smoke-update-budget.json"
    with (directory / "checkpoint-process-smoke-budget.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text())
        else:
            if Path(report_path).exists():
                raise RuntimeError("Existing checkpoint receipt but missing update ledger; refusing reset")
            ledger = {"scope": "separately authorized fresh-process synthetic recovery",
                      "maximum_updates": MAX_UPDATES, "precharged_updates": 0, "attempts": []}
        if ledger["maximum_updates"] != MAX_UPDATES or ledger["precharged_updates"] + PLANNED_UPDATES > MAX_UPDATES:
            raise RuntimeError("Fresh-process recovery allowance exhausted; no automatic increase")
        ledger["precharged_updates"] += PLANNED_UPDATES
        ledger["attempts"].append({"attempt_id": attempt_id, "precharged_updates": PLANNED_UPDATES,
                                    "created_at_utc": datetime.now(timezone.utc).isoformat()})
        atomic_json(ledger_path, ledger)
    return ledger


def tensor_tree_hash(value):
    """Device-independent digest of structure, dtype, shape, and tensor bytes."""
    import torch
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(json.dumps(["tensor", str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for element in item:
                visit(element)
        else:
            digest.update(json.dumps([type(item).__name__, item], sort_keys=True, allow_nan=False).encode())
    visit(value)
    return digest.hexdigest()


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def worker(args):
    def deny_network(event, unused):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            raise RuntimeError("Network access prohibited during checkpoint smoke")
    sys.addaudithook(deny_network)
    import torch
    import psutil
    from hydra_directflow.models.directflow import DirectFlowModel, ModelConfig, masked_mse
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Worker journal must be inside ignored runs/")
    permit = json.loads((run_dir / "authorization.json").read_text())
    if permit != {"planned_updates": PLANNED_UPDATES, "maximum_updates": MAX_UPDATES}:
        raise ValueError("Missing bounded parent precharge")
    architecture, device = CASES[args.case]
    result_path = run_dir / f"{args.case}-{args.phase}.json"
    telemetry_path = run_dir / f"{args.case}-{args.phase}.telemetry.json"
    checkpoint_path = run_dir / f"{args.case}-checkpoint.pt"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = {"status": "FAIL", "architecture": architecture, "device": device,
              "phase": args.phase, "pid": os.getpid(), "optimizer_updates_completed": 0}
    if device == "mps" and not torch.backends.mps.is_available():
        result.update(status="UNAVAILABLE", error="MPS unavailable; no CPU fallback")
        atomic_json(result_path, result)
        return
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(min(1., (2 * 1024**3) / torch.mps.recommended_max_memory()))
    stop = threading.Event()
    peak = {"rss_bytes": 0, "mps_driver_bytes": 0}

    def monitor():
        process = psutil.Process()
        while not stop.is_set():
            sample = {"rss_bytes": process.memory_info().rss,
                      "mps_driver_bytes": torch.mps.driver_allocated_memory() if device == "mps" else 0}
            for key, value in sample.items():
                peak[key] = max(peak[key], value)
            atomic_json(telemetry_path, sample)
            stop.wait(.05)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        torch.manual_seed(526)
        config = ModelConfig(architecture, HISTORY, WEATHER, STATE, 24, tuple(range(1, 19)), dropout=.1)
        model = DirectFlowModel(config).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
        generator = torch.Generator().manual_seed(925)
        inputs = []
        for shape in ((3, 24, 2), (3, 18, 4), (3, 2)):
            values = torch.randn(shape, generator=generator).to(device)
            mask = torch.ones_like(values, dtype=torch.bool)
            mask.reshape(-1)[1] = False
            values.reshape(-1)[1] = 0
            inputs.extend((values, mask))
        targets = torch.linspace(-.5, .5, 18, device=device)[None].expand(3, -1).clone()
        target_mask = torch.ones_like(targets, dtype=torch.bool)
        target_mask[:, 5] = False
        targets[:, 5] = float("nan")
        input_digest = tensor_tree_hash(inputs)
        target_digest = tensor_tree_hash((targets, target_mask))

        def update():
            optimizer.zero_grad(set_to_none=True)
            output = model(*inputs)
            loss = masked_mse(output, targets, target_mask)
            assert bool(torch.isfinite(loss)) and bool(torch.isfinite(output).all())
            loss.backward()
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            optimizer.step()
            result["optimizer_updates_completed"] += 1
            return output.detach().cpu(), loss.detach().cpu()

        if args.phase == "original":
            update()
            checkpoint = {"model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
                          "cpu_rng": torch.get_rng_state(), "input_sha256": input_digest,
                          "target_sha256": target_digest}
            if device == "mps":
                checkpoint["device_rng"] = torch.mps.get_rng_state().cpu()
            torch.save(checkpoint, checkpoint_path)
            saved_state_hash = tensor_tree_hash((checkpoint["model"], checkpoint["optimizer"]))
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            assert checkpoint["input_sha256"] == input_digest
            assert checkpoint["target_sha256"] == target_digest
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            torch.set_rng_state(checkpoint["cpu_rng"])
            if device == "mps":
                torch.mps.set_rng_state(checkpoint["device_rng"])
            saved_state_hash = tensor_tree_hash((model.state_dict(), optimizer.state_dict()))
            assert saved_state_hash == tensor_tree_hash((checkpoint["model"], checkpoint["optimizer"]))
        output, loss = update()
        if device == "mps":
            torch.mps.synchronize()
        result.update(status="PASS", input_sha256=input_digest, target_sha256=target_digest,
            checkpoint_sha256=file_hash(checkpoint_path), restored_state_sha256=saved_state_hash,
            output_sha256=tensor_tree_hash(output), loss_sha256=tensor_tree_hash(loss), loss=float(loss),
            model_state_sha256=tensor_tree_hash(model.state_dict()),
            optimizer_state_sha256=tensor_tree_hash(optimizer.state_dict()),
            cpu_rng_sha256=tensor_tree_hash(torch.get_rng_state()), output_shape=list(output.shape),
            finite_all_parameter_gradients=True, torch_version=torch.__version__)
        if device == "mps":
            result["device_rng_sha256"] = tensor_tree_hash(torch.mps.get_rng_state())
    except Exception as error:
        result.update(error_type=type(error).__name__, error=portable(str(error))[:2000])
    finally:
        stop.set()
        watcher.join(timeout=1)
        result["observed_peaks"] = peak
        atomic_json(result_path, result)


def supervise(case, phase, run_dir, deadline):
    import psutil
    result_path = run_dir / f"{case}-{phase}.json"
    telemetry = run_dir / f"{case}-{phase}.telemetry.json"
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "0"})
    command = [sys.executable, str(Path(__file__).resolve()), "--execute", "--worker", "--case", case,
               "--phase", phase, "--run-dir", str(run_dir)]
    peak, stop_reason = 0, None
    started = time.monotonic()
    with (run_dir / f"{case}-{phase}.log").open("w") as log:
        child = subprocess.Popen(command, env=environment, stdout=log, stderr=log, cwd=ROOT)
        process, supervisor = psutil.Process(child.pid), psutil.Process()
        while child.poll() is None:
            try:
                rss = supervisor.memory_info().rss + process.memory_info().rss
                rss += sum(p.memory_info().rss for p in process.children(recursive=True))
                driver = 0
                if telemetry.exists():
                    try:
                        driver = json.loads(telemetry.read_text())["mps_driver_bytes"]
                    except (OSError, json.JSONDecodeError, KeyError):
                        pass
                peak = max(peak, rss + driver)
                if rss + driver > MAX_BYTES:
                    stop_reason = "combined_RSS_and_MPS_driver_limit"
                if time.monotonic() >= deadline:
                    stop_reason = "global_300_second_wall_limit"
                if stop_reason:
                    child.kill()
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(.05)
        child.wait(timeout=5)
    result = json.loads(result_path.read_text()) if result_path.exists() else {"status": "FAIL", "error": "Worker produced no receipt"}
    result.update(wall_seconds=round(time.monotonic() - started, 3), combined_observed_peak_bytes=peak)
    if stop_reason or child.returncode:
        result.update(status="FAIL", stop_reason=stop_reason, exit_code=child.returncode)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Explicitly run the bounded synthetic recovery proof")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/checkpoint-process-smoke.json")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=("original", "resumed"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "DISABLED_REQUIRES_EXECUTE", "planned_synthetic_updates": PLANNED_UPDATES,
                          "separate_allowance_maximum": MAX_UPDATES, "cases": list(CASES)}))
        return 0
    if args.worker:
        worker(args)
        return 0
    from hydra_directflow.telemetry import RunJournal
    if args.report.is_symlink() or not args.report.resolve().is_relative_to((ROOT / "reports").resolve()):
        parser.error("Public receipt must be a regular path within reports/")
    deadline = time.monotonic() + MAX_SECONDS
    attempt_id = "checkpoint-process-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    ledger = precharge(args.report, attempt_id)
    run_dir = ROOT / "runs" / attempt_id
    spec = {"scope": "synthetic_fresh_process_checkpoint_recovery", "cases": CASES,
            "history_hours": 24, "horizons_hours": list(range(1, 19)), "batch_size": 3,
            "history_features": HISTORY, "weather_features": WEATHER, "state_features": STATE,
            "dropout": .1, "optimizer": "AdamW", "learning_rate": 1e-4,
            "planned_updates": PLANNED_UPDATES, "max_seconds": MAX_SECONDS, "max_combined_bytes": MAX_BYTES}
    journal = RunJournal(run_dir, spec, max_wall_seconds=MAX_SECONDS, max_rss_bytes=MAX_BYTES)
    atomic_json(run_dir / "authorization.json", {"planned_updates": PLANNED_UPDATES, "maximum_updates": MAX_UPDATES})
    journal.event("optimizer_updates_precharged", planned=PLANNED_UPDATES, cumulative=ledger["precharged_updates"], maximum=MAX_UPDATES)
    qualification_path = ROOT / "reports/model-qualification.json"
    previous = json.loads(qualification_path.read_text())
    previous_charged = previous["update_accounting"]["cumulative_precharged_updates"]
    report = {"schema": "hydra-directflow-checkpoint-process-smoke/v1", "status": "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "spec": spec,
        "evidence_class": "synthetic checkpoint mechanics; no research fit or forecast skill",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": importlib.metadata.version("torch"), "transformers": importlib.metadata.version("transformers")},
        "limits": {"global_seconds": MAX_SECONDS, "combined_rss_mps_bytes": MAX_BYTES,
                   "mps_driver_bytes": 2 * 1024**3, "threads": 1, "poll_seconds": .05,
                   "cpu_fallback": False, "automatic_increase": False},
        "update_accounting": {"new_allowance": ledger, "previous_qualification_charged_updates": previous_charged,
            "cumulative_precharged_updates_including_prior_qualification": previous_charged + ledger["precharged_updates"],
            "prior_qualification_sha256": file_hash(qualification_path), "prior_receipt_modified": False},
        "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in
            [Path(__file__).resolve(), ROOT / "src/hydra_directflow/telemetry.py",
             ROOT / "src/hydra_directflow/models/directflow.py", ROOT / "src/hydra_directflow/models/mamba2_math.py"]},
        "cases": []}
    if args.report.exists():
        report["prior_attempts"] = [json.loads(args.report.read_text())]
    try:
        for case in CASES:
            pair = {"case": case}
            report["cases"].append(pair)
            for phase in ("original", "resumed"):
                with journal.stage(case.replace("-", "_") + "_" + phase):
                    result = supervise(case, phase, run_dir, deadline)
                    pair[phase] = result
                    journal.event("process_result", case=case, **result)
                    if result["status"] != "PASS":
                        raise RuntimeError("Checkpoint worker failed; receipt retained")
            keys = ("input_sha256", "target_sha256", "checkpoint_sha256", "restored_state_sha256", "output_sha256",
                    "loss_sha256", "model_state_sha256", "optimizer_state_sha256", "cpu_rng_sha256", "device_rng_sha256")
            pair["matched_hashes"] = {key: pair["original"].get(key) == pair["resumed"].get(key) for key in keys
                                      if key in pair["original"] or key in pair["resumed"]}
            pair["fresh_process"] = pair["original"]["pid"] != pair["resumed"]["pid"]
            pair["status"] = "PASS" if pair["fresh_process"] and all(pair["matched_hashes"].values()) else "FAIL"
            journal.event("fresh_process_parity", case=case, status=pair["status"], matched_hashes=pair["matched_hashes"])
            journal.artifact(run_dir / f"{case}-checkpoint.pt")
            if pair["status"] != "PASS":
                raise RuntimeError("Fresh-process state or output parity failed")
            print(case + ": PASS (fresh process, exact tensor hashes)", flush=True)
        report["status"] = "PASS"
        journal.finish()
    except Exception as error:
        report.update(error_type=type(error).__name__, error=portable(str(error)))
        if journal.status == "RUNNING":
            journal.finish("FAILED", error_type=type(error).__name__)
    report["optimizer_updates_completed"] = sum(pair.get(phase, {}).get("optimizer_updates_completed", 0)
                                                   for pair in report["cases"] for phase in ("original", "resumed"))
    report["journal"] = {"spec_sha256": journal.spec_sha256, "status": journal.status,
        "state_sha256": file_hash(run_dir / "state.json"), "events_sha256": file_hash(run_dir / "events.jsonl")}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, portable(report))
    print(json.dumps({"status": report["status"], "optimizer_updates_completed": report["optimizer_updates_completed"],
                      "new_allowance_precharged": ledger["precharged_updates"]}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
