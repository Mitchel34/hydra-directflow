#!/usr/bin/env python3
"""Bounded real-input to initialized-model smoke; zero updates or skill scores."""
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ARCHITECTURES = ("LSTM", "Transformer", "GRU_Transformer", "Mamba2_Transformer")
HISTORY = ("discharge_cms", "age_hours")
WEATHER = ("precip_mm", "temp_k", "specific_humidity", "swe_kg_m2")
STATE = ("soil_water_fraction", "soil_age_hours")
MAX_SECONDS = 300
MAX_BYTES = 4 * 1024**3
UNIT_MAPPING = {
    "history.discharge_cms": "x / 100", "history.age_hours": "x / 168",
    "weather.precip_mm": "x / 10", "weather.temp_k": "(x - 273.15) / 30",
    "weather.specific_humidity": "x / 0.01", "weather.swe_kg_m2": "x / 100",
    "state.soil_water_fraction": "x", "state.soil_age_hours": "x / 24",
    "missing_inputs": "map only observed values; encode false masks with zero placeholders",
}
QUALIFICATION = ("ENGINEERING_ONLY_FIXED_UNIT_MAPPING_NOT_STUDY_TRANSFORMS; initialized random weights; "
                 "no trained model, optimizer updates, fitted scalers, prediction publication, or forecast skill claim")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def load_inputs(npz_path, receipt_path):
    """Authenticate the private artifact and decode ONLY six model input arrays."""
    import numpy as np
    from hydra_directflow.local_data import (ARCHIVE_SHA256, CYCLE_ID, GAUGES, USGS_SHA256,
                                             bounded_npy, verified_local_bytes)
    path = Path(receipt_path)
    metadata = path.lstat()
    if (not path.is_file() or path.is_symlink() or metadata.st_size > 64 * 1024
            or getattr(metadata, "st_flags", 0) & 0x40000000):
        raise ValueError("Source receipt must be a bounded regular local file")
    receipt_raw = path.read_bytes()
    receipt = json.loads(receipt_raw)
    if (receipt["status"] != "PASSED_DEVELOPMENT_SMOKE" or receipt["source_cycle"] != CYCLE_ID
            or receipt["sources"]["hrrr"]["sha256"] != ARCHIVE_SHA256
            or receipt["sources"]["usgs"]["sha256"] != USGS_SHA256
            or tuple(receipt["gauge_ids"]) != GAUGES or receipt["history_hours"] != 24
            or receipt["origin_utc"] != "2021-08-23T02:00:00Z"):
        raise ValueError("Only the verified one-origin L24 development fixture is admitted")
    for name, expected in (("history", HISTORY), ("weather", WEATHER), ("state", STATE)):
        if tuple(receipt["features"][name]) != expected:
            raise ValueError("Receipt feature order differs")
    identity = receipt["private_tensor_artifact"]
    raw = verified_local_bytes(npz_path, identity["sha256"], expected_bytes=identity["bytes"], max_bytes=2 * 1024**2)
    shapes = {"history": (3, 24, 2), "weather": (3, 18, 4), "state": (3, 2)}
    inputs = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if len(members) > 64 or len({m.filename for m in members}) != len(members):
            raise ValueError("NPZ inventory is not bounded/unique")
        if any(m.file_size > 4 * 1024**2 for m in members) or sum(m.file_size for m in members) > 16 * 1024**2:
            raise ValueError("NPZ expanded size exceeds bound")
        for name, shape in shapes.items():
            values = bounded_npy(archive.read(name + ".npy"), shape=shape, dtype="<f8")
            mask = bounded_npy(archive.read(name + "_mask.npy"), shape=shape, dtype="?")
            if not np.isfinite(values[mask]).all():
                raise ValueError("Nonfinite admitted physical input")
            mapped = values.copy()
            if name == "history":
                mapped[..., 0] /= 100
                mapped[..., 1] /= 168
            elif name == "weather":
                mapped[..., 0] /= 10
                mapped[..., 1] = (mapped[..., 1] - 273.15) / 30
                mapped[..., 2] /= .01
                mapped[..., 3] /= 100
            else:
                mapped[..., 1] /= 24
            inputs[name] = np.where(mask, mapped, 0).astype(np.float32)
            inputs[name + "_mask"] = mask
    # target.npy/target_mask.npy and quality metadata are never decoded here.
    provenance = {"private_npz": identity, "source_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
                  "hrrr_archive_sha256": ARCHIVE_SHA256, "usgs_partition_sha256": USGS_SHA256,
                  "origin_utc": receipt["origin_utc"], "source_cycle": CYCLE_ID,
                  "source_qualification": receipt["qualification"]}
    return inputs, provenance


def worker(args):
    def deny_network(event, values):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            raise RuntimeError("Network access prohibited during real-input model smoke")
    sys.addaudithook(deny_network)
    import torch
    import psutil
    from hydra_directflow.models.directflow import DirectFlowModel, ModelConfig
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS device is unavailable")
        torch.mps.set_per_process_memory_fraction(min(1., 2 * 1024**3 / torch.mps.recommended_max_memory()))
    stop = threading.Event()
    process = psutil.Process()
    peak = {"host_rss_bytes": 0, "mps_driver_bytes": 0}

    def sample():
        while not stop.is_set():
            current = {"host_rss_bytes": process.memory_info().rss,
                       "mps_driver_bytes": torch.mps.driver_allocated_memory() if args.device == "mps" else 0}
            for name, value in current.items():
                peak[name] = max(peak[name], value)
            atomic_json(args.telemetry, current)
            stop.wait(.05)

    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    result = {"status": "FAILED", "architecture": args.architecture, "device": args.device,
              "optimizer_updates": 0, "scaler_fits": 0, "targets_decoded_or_passed_to_forward": False}
    try:
        inputs, provenance = load_inputs(args.npz, args.receipt)
        model = DirectFlowModel(ModelConfig(args.architecture, HISTORY, WEATHER, STATE,
                                             history_hours=24, horizon_hours=tuple(range(1, 19)),
                                             width=64, dropout=0)).to(args.device).eval()
        tensors = [torch.from_numpy(inputs[name]).to(args.device) for name in
                   ("history", "history_mask", "weather", "weather_mask", "state", "state_mask")]

        def state_digest():
            h = hashlib.sha256()
            for name, value in model.state_dict().items():
                h.update(name.encode())
                h.update(value.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()

        before = state_digest()
        started = time.monotonic()
        with torch.no_grad():
            output = model(*tensors)
        if args.device == "mps":
            torch.mps.synchronize()
        seconds = time.monotonic() - started
        after = state_digest()
        unchanged = before == after and all(p.grad is None for p in model.parameters())
        if tuple(output.shape) != (3, 18) or not bool(torch.isfinite(output).all()) or not unchanged:
            raise ValueError("Finite shape or unchanged-model invariant failed")
        if output.requires_grad or output.device.type != args.device or any(t.device.type != args.device for t in tensors):
            raise ValueError("Unexpected gradient/device execution")
        result.update(status="PASSED", output_shape=[3, 18], output_finite=True,
                      parameters_and_buffers_unchanged=True, model_state_sha256=after,
                      output_requires_grad=False, forward_seconds=seconds,
                      mps_available=torch.backends.mps.is_available(), torch_version=torch.__version__,
                      parameter_count=sum(p.numel() for p in model.parameters()),
                      input_npz_sha256=provenance["private_npz"]["sha256"])
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    finally:
        stop.set()
        watcher.join(timeout=1)
        result["observed_peaks"] = peak
        atomic_json(args.result, result)


def supervise_case(args, architecture, device, run_dir, deadline):
    import psutil
    identifier = device + "-" + architecture.lower()
    result_path = run_dir / (identifier + ".json")
    telemetry = run_dir / (identifier + ".telemetry.json")
    log_path = run_dir / (identifier + ".log")
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "VECLIB_MAXIMUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                        "PYTORCH_ENABLE_MPS_FALLBACK": "0"})
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--npz", str(args.npz),
               "--receipt", str(args.receipt), "--architecture", architecture, "--device", device,
               "--result", str(result_path), "--telemetry", str(telemetry)]
    peak = 0
    reason = None
    with log_path.open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=log, env=environment)
        watched = psutil.Process(child.pid)
        supervisor = psutil.Process()
        while child.poll() is None:
            try:
                rss = supervisor.memory_info().rss + watched.memory_info().rss
                rss += sum(p.memory_info().rss for p in watched.children(recursive=True))
                driver = 0
                if telemetry.exists():
                    try:
                        driver = json.loads(telemetry.read_text())["mps_driver_bytes"]
                    except (OSError, json.JSONDecodeError, KeyError):
                        pass
                peak = max(peak, rss + driver)
                if rss + driver > MAX_BYTES:
                    reason = "combined_RSS_and_MPS_driver_limit"
                elif time.monotonic() >= deadline:
                    reason = "overall_300_second_wall_limit"
                if reason:
                    child.kill()
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(.05)
        child.wait(timeout=5)
    result = json.loads(result_path.read_text()) if result_path.exists() else {
        "status": "FAILED", "architecture": architecture, "device": device, "error_type": "NoWorkerReceipt"}
    result["combined_observed_peak_bytes"] = peak
    if reason or child.returncode:
        result.update(status="FAILED", stop_reason=reason, exit_code=child.returncode)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True, help="Verified private NPZ; never included in Git")
    parser.add_argument("--receipt", type=Path, default=ROOT / "reports/local-smoke.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/input-model-smoke.json")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--architecture", choices=ARCHITECTURES, help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("cpu", "mps"), help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--telemetry", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return 0
    from hydra_directflow.telemetry import RunJournal
    created = datetime.now(timezone.utc).isoformat()
    run_dir = args.run_dir or ROOT / "runs" / ("input-model-smoke-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if args.report.resolve() in {args.npz.resolve(), args.receipt.resolve()} or args.report.is_symlink():
        parser.error("Report cannot overwrite or link to an input")
    if not run_dir.resolve().is_relative_to((ROOT / "runs").resolve()):
        parser.error("Run journal must live inside the ignored runs directory")
    spec = {"scope": "one_real_origin_three_gauges_L24_H18", "architectures": list(ARCHITECTURES),
            "qualification": QUALIFICATION, "unit_mapping": UNIT_MAPPING, "max_seconds": MAX_SECONDS,
            "max_combined_bytes": MAX_BYTES, "optimizer_updates": 0, "scaler_fits": 0}
    journal = RunJournal(run_dir, spec, max_wall_seconds=MAX_SECONDS, max_rss_bytes=MAX_BYTES)
    deadline = time.monotonic() + MAX_SECONDS
    report = {"schema": "hydra-directflow-input-model-smoke/v1", "artifact_created_at_utc": created,
              "status": "FAILED", "qualification": QUALIFICATION, "unit_mapping": UNIT_MAPPING,
              "input_features": {"history": list(HISTORY), "weather": list(WEATHER), "state": list(STATE)},
              "limits": {"overall_wall_seconds": MAX_SECONDS, "combined_RSS_MPS_bytes": MAX_BYTES,
                         "mps_driver_cap_bytes": 2 * 1024**3, "threads": 1, "poll_interval_seconds": .05,
                         "mps_cpu_fallback": False, "automatic_limit_increases": False},
              "execution": {"optimizer_updates": 0, "scaler_fits": 0, "network_requests": 0,
                            "assessment_rows_read": 0, "targets_decoded_or_passed_to_forward": False,
                            "performance_scores_computed": False, "outputs_saved_as_forecasts": False}, "cases": []}
    try:
        with journal.stage("load"):
            _, provenance = load_inputs(args.npz, args.receipt)
            report["input_provenance"] = provenance
            journal.event("authenticated_inputs", **provenance)
        devices = ["cpu"]
        for device in devices:
            for architecture in ARCHITECTURES:
                with journal.stage("forward_" + device + "_" + architecture.lower()):
                    result = supervise_case(args, architecture, device, run_dir, deadline)
                    report["cases"].append(result)
                    journal.event("model_forward_result", **result)
                    journal.metric("completed_forwards", len(report["cases"]))
                    journal.metric("combined_observed_peak_bytes", max(c["combined_observed_peak_bytes"] for c in report["cases"]))
                    if result["status"] != "PASSED":
                        raise RuntimeError("Forward smoke failed; see private journal")
            if device == "cpu" and report["cases"][0]["mps_available"]:
                devices.append("mps")
        report["mps_available"] = "mps" in devices
        report["status"] = "PASSED_REAL_INPUT_FORWARD_SMOKE"
        journal.finish()
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        if journal.status == "RUNNING":
            journal.finish("FAILED", error_type=type(exc).__name__)
    report["journal"] = {"spec_sha256": journal.spec_sha256, "status": journal.status,
                         "state_sha256": hashlib.sha256((run_dir / "state.json").read_bytes()).hexdigest(),
                         "events_sha256": hashlib.sha256((run_dir / "events.jsonl").read_bytes()).hexdigest(),
                         "metrics_sha256": hashlib.sha256((run_dir / "metrics.prom").read_bytes()).hexdigest()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, report)
    print(json.dumps({"status": report["status"], "forward_cases": len(report["cases"]), "optimizer_updates": 0}))
    return 0 if report["status"] == "PASSED_REAL_INPUT_FORWARD_SMOKE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
