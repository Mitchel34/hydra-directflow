#!/usr/bin/env python3
"""Bounded offline synthetic CPU/MPS qualification; never a research fit.

Run with the prepared interpreter: .venv/bin/python scripts/qualify_models.py
Each model/device/lookback case makes exactly three synthetic optimizer updates.
The full run uses 48 updates. MPS failure is recorded, never retried on CPU.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("LSTM", "Transformer", "GRU_Transformer", "Mamba2_Transformer")
LIMIT_BYTES = 4 * 1024**3
TIMEOUT_SECONDS = 300


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def portable(value):
    if isinstance(value, str):
        return value.replace(str(ROOT), "[repo]").replace(str(Path.home()), "[home]")
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    return value


def worker(arguments):
    # Network access is prohibited even if a transitive library ignores offline flags.
    def deny_network(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            raise RuntimeError("Network access prohibited during synthetic qualification")
    sys.addaudithook(deny_network)
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    import psutil
    spec = importlib.util.spec_from_file_location("model_test_helpers", ROOT / "tests/test_models.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    helpers.configure_threads()
    if arguments.kind == "probe":
        atomic_json(arguments.result, {"status": "PASS", "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(), "torch": torch.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(), "mps_cpu_fallback": False})
        return
    if arguments.device == "mps" and not torch.backends.mps.is_available():
        atomic_json(arguments.result, {"status": "UNAVAILABLE", "reason": "MPS is not available"})
        return
    if arguments.device == "mps":
        recommended = torch.mps.recommended_max_memory()
        # A fixed 2 GiB driver-allocation cap; the supervisor also caps RSS+driver
        # at 4 GiB. This never increases in response to failure or retries.
        torch.mps.set_per_process_memory_fraction(min(1.0, (2 * 1024**3) / recommended))
    stop = threading.Event()
    peak = {"rss_bytes": 0, "mps_driver_bytes": 0, "mps_current_bytes": 0}
    process = psutil.Process()

    def sample():
        while not stop.is_set():
            sample_data = {"rss_bytes": process.memory_info().rss,
                "mps_driver_bytes": torch.mps.driver_allocated_memory() if arguments.device == "mps" else 0,
                "mps_current_bytes": torch.mps.current_allocated_memory() if arguments.device == "mps" else 0}
            for key, value in sample_data.items():
                peak[key] = max(peak[key], value)
            atomic_json(arguments.telemetry, sample_data)
            stop.wait(.05)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    result = {"status": "FAIL", "device": arguments.device, "optimizer_updates_planned":
              3 if arguments.kind == "model" else 0}
    try:
        if arguments.kind in ("model", "geometry"):
            weather = None
            if arguments.kind == "geometry":
                development = json.loads((ROOT / "configs/development.json").read_text())
                weather = tuple(development["weather"]["required_features"])
            result["forward_gradient"] = helpers.forward_gradient_case(
                arguments.architecture, arguments.history_hours, arguments.device, weather_features=weather)
            if arguments.kind == "model":
                result["checkpoint"] = helpers.checkpoint_case(
                    arguments.architecture, arguments.history_hours, arguments.device)
        else:
            result["oracle"] = helpers.ssd_oracle_case(arguments.device)
        if arguments.device == "mps":
            torch.mps.synchronize()
        result["status"] = "PASS"
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)[:3000]
        result["traceback"] = traceback.format_exc(limit=5)[-5000:]
    finally:
        stop.set()
        monitor.join(timeout=1)
        result["worker_observed_peaks"] = peak
        atomic_json(arguments.result, result)


def run_case(case, temporary):
    import psutil
    identifier = case["id"]
    result_path = temporary / (identifier + ".json")
    telemetry = temporary / (identifier + ".telemetry.json")
    log_path = temporary / (identifier + ".log")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--kind", case["kind"],
               "--device", case.get("device", "cpu"), "--result", str(result_path),
               "--telemetry", str(telemetry)]
    if case["kind"] in ("model", "geometry"):
        command.extend(["--architecture", case["architecture"], "--history-hours", str(case["history_hours"])])
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "0"})
    started = time.monotonic()
    peak_combined = 0
    reason = None
    with log_path.open("w") as log:
        child = subprocess.Popen(command, env=environment, stdout=log, stderr=log, cwd=ROOT)
        watched = psutil.Process(child.pid)
        supervisor = psutil.Process()
        while child.poll() is None:
            try:
                rss = supervisor.memory_info().rss + watched.memory_info().rss
                rss += sum(p.memory_info().rss for p in watched.children(recursive=True))
                mps = 0
                if telemetry.exists():
                    try:
                        mps = json.loads(telemetry.read_text())["mps_driver_bytes"]
                    except (OSError, json.JSONDecodeError, KeyError):
                        pass
                peak_combined = max(peak_combined, rss + mps)
                if rss + mps > LIMIT_BYTES:
                    reason = "Combined supervisor+worker RSS and MPS driver memory exceeded 4 GiB"
                if time.monotonic() - started > TIMEOUT_SECONDS:
                    reason = "Case exceeded 300 second wall-clock cap"
                if reason:
                    child.kill()
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(.05)
        child.wait(timeout=5)
    result = dict(case)
    if result_path.exists():
        result.update(json.loads(result_path.read_text()))
    else:
        result.update(status="FAIL", error="Worker produced no result", exit_code=child.returncode)
    if reason:
        result.update(status="FAIL", error=reason)
    if child.returncode:
        result.update(status="FAIL", exit_code=child.returncode)
    result.update(wall_seconds=round(time.monotonic() - started, 3),
                  combined_observed_peak_bytes=peak_combined)
    log = log_path.read_text().strip()
    if log:
        result["worker_log_tail"] = log[-2500:]
    print(f"{identifier}: {result['status']} ({result['wall_seconds']} s)", flush=True)
    return result


def provenance():
    paths = [ROOT / "requirements/model-lock.txt", ROOT / "tests/test_models.py", Path(__file__).resolve()]
    paths.extend(sorted((ROOT / "src/hydra_directflow/models").glob("*.py")))
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    versions = {}
    for line in (ROOT / "requirements/model-lock.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, pinned = line.split("==")
            actual = importlib.metadata.version(name)
            versions[name] = {"pinned": pinned, "installed": actual, "matches": actual == pinned}
    wheels = [{"filename": path.name, "size_bytes": path.stat().st_size,
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
              for path in sorted((ROOT / ".cache/wheels").glob("*.whl"))]
    # --ignore-installed proves the entire environment resolves from the local
    # cache; an inherited global install cannot hide a missing wheel.
    resolution = subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run",
        "--ignore-installed", "--no-index", "--find-links", str(ROOT / ".cache/wheels"),
        "--requirement", str(ROOT / "requirements/model-lock.txt")],
        capture_output=True, text=True, timeout=120, cwd=ROOT)
    return {"source_sha256": hashes, "packages": versions, "wheel_inventory": wheels,
            "wheel_cache_relative_path": ".cache/wheels",
            "offline_dependency_resolution": {"status": "PASS" if resolution.returncode == 0 else "FAIL",
                "network_disabled_by_pip": True, "installed_packages_ignored": True,
                "exit_code": resolution.returncode,
                "output_tail": portable((resolution.stdout + resolution.stderr)[-2000:])},
            "wheel_scope": "Current interpreter/platform wheels; other OS/architectures require matching wheels"}


def precharge_updates(mode, report_path):
    """Fail closed before any model step; failures consume the planned allowance."""
    additional = mode == "mps-l24-recovery"
    ledger_path = ROOT / (".cache/model-qualification-additional-recovery-budget.json" if additional
                          else ".cache/model-qualification-update-budget.json")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
    else:
        if additional and report_path.exists():
            # Do not recreate the added allowance after a lost local ledger.
            # Older receipts may retain it under prior_runs or prior_update_budget.
            def contains_additional_receipt(value):
                if isinstance(value, dict):
                    if "previous_allowance_charged_updates" in value:
                        return True
                    return any(contains_additional_receipt(v) for v in value.values())
                if isinstance(value, list):
                    return any(contains_additional_receipt(v) for v in value)
                return False
            if contains_additional_receipt(json.loads(report_path.read_text())):
                raise RuntimeError("Additional recovery ledger missing; refusing to reconstruct or reset charges")
        # Import the original full-run receipt if upgrading a previously run
        # harness. Missing checkpoint output does not refund failed-case steps.
        charged = 0
        if report_path.exists() and not additional:
            previous = json.loads(report_path.read_text())
            if previous.get("prior_runs"):
                raise RuntimeError("Budget ledger missing after composite recovery history; refusing to reconstruct or reset charges")
            charged = sum(case.get("optimizer_updates_planned", 0) for case in previous.get("cases", []))
        ledger = {"maximum_updates": 16 if additional else 64, "charged_updates": charged, "runs": []}
        if additional:
            ledger["authorization_scope"] = "Additional bounded 16-update correction of checkpoint-harness restore; original 60-update receipt retained"
            ledger["previous_allowance_charged_updates"] = 60
    planned = {"full": 48, "mps-recovery": 12, "mps-l24-recovery": 12, "schema-check": 0}[mode]
    if ledger["charged_updates"] + planned > ledger["maximum_updates"]:
        raise RuntimeError("Synthetic update allowance exhausted; obtain a newly bounded authorization before further runs")
    ledger["charged_updates"] += planned
    ledger["runs"].append({"mode": mode, "updates_precharged": planned,
                          "created_at_utc": datetime.now(timezone.utc).isoformat()})
    atomic_json(ledger_path, ledger)
    return ledger


def update_accounting():
    ledgers = []
    for name in ("model-qualification-update-budget.json", "model-qualification-additional-recovery-budget.json"):
        path = ROOT / ".cache" / name
        if path.exists():
            ledgers.append({"ledger_relative_path": str(path.relative_to(ROOT)), **json.loads(path.read_text())})
    return {"ledgers": ledgers, "cumulative_precharged_updates": sum(x["charged_updates"] for x in ledgers),
            "note": "Failed attempts remain charged; 48 updates is the full-invocation plan, not the composite total",
            "checkpoint_scope": "Same-process serialize/load/RNG restore and exact next-update parity; fresh-process restart not tested"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(ROOT / "reports/model-qualification.json"))
    parser.add_argument("--mode", choices=("full", "mps-recovery", "mps-l24-recovery", "schema-check"), default="full",
                        help="full: 48 updates; MPS recovery: 12 updates; schema-check: no optimizer updates")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--kind", choices=("probe", "model", "oracle", "geometry"), default="probe", help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu", help=argparse.SUPPRESS)
    parser.add_argument("--architecture", choices=ARCHITECTURES, help=argparse.SUPPRESS)
    parser.add_argument("--history-hours", type=int, choices=(24, 168), help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    parser.add_argument("--telemetry", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker:
        worker(arguments)
        return 0
    destination = Path(arguments.report)
    previous = json.loads(destination.read_text()) if destination.exists() else None
    if arguments.mode != "full" and previous is None:
        raise ValueError("MPS recovery requires the original report")
    update_budget = precharge_updates(arguments.mode, destination)
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "synthetic engineering qualification; not research fitting or forecast skill",
        "scope": {"horizons_hours": list(range(1, 19)), "history_hours": [24, 168],
                  "gauges": ["03161000", "03164000", "03479000"],
                  "gauge_scope_note": "Per-gauge shared architecture contract; no gauge observations consumed",
                  "real_data_consumed": False, "pretrained_weights_downloaded": False},
        "limits": {"seconds_per_case": TIMEOUT_SECONDS, "combined_rss_mps_bytes": LIMIT_BYTES,
                   "mps_driver_cap_bytes": 2 * 1024**3, "threads": 1,
                   "optimizer_updates_maximum": 48, "automatic_allocation_growth": False,
                   "cpu_fallback_for_mps": False, "memory_poll_interval_seconds": .05},
        "dependencies": provenance(), "update_budget": update_budget, "cases": []}
    with tempfile.TemporaryDirectory(prefix="directflow-qualification-") as directory:
        temporary = Path(directory)
        probe = run_case({"id": "environment", "kind": "probe"}, temporary)
        report["environment"] = probe
        for device in ("cpu", "mps"):
            if arguments.mode in ("mps-recovery", "mps-l24-recovery") and device == "cpu":
                continue
            for architecture in ARCHITECTURES:
                for length in (24, 168):
                    if arguments.mode == "mps-recovery" and length == 24:
                        continue
                    if arguments.mode == "mps-l24-recovery" and length == 168:
                        continue
                    case = {"id": f"{device}-{architecture}-L{length}-H18", "kind":
                            "geometry" if arguments.mode == "schema-check" else "model",
                            "device": device, "architecture": architecture, "history_hours": length}
                    report["cases"].append(run_case(case, temporary))
            if arguments.mode == "full":
                report["cases"].append(run_case({"id": f"{device}-ssd-oracle", "kind": "oracle",
                                                  "device": device}, temporary))
    if previous is not None:
        report["prior_runs"] = [previous]
    if arguments.mode in ("mps-recovery", "mps-l24-recovery"):
        recovered = {case["id"]: case for case in report["cases"]}
        report["recovery"] = {"change": "Load checkpoints on CPU, then move parameters/moments via load_state_dict",
            "rerun_scope": "Four MPS models at " + ("L168/H18" if arguments.mode == "mps-recovery" else "L24/H18"),
            "optimizer_updates": 12,
            "historical_failure": "Initial MPS optimizer-state device assertions failed; corrected checkpoints load on CPU before state placement"}
        report["cases"] = [recovered.get(case["id"], case) for case in previous["cases"]]
    if arguments.mode == "schema-check":
        report["development_schema_cases"] = report["cases"]
        report["cases"] = previous["cases"]
        report["prior_update_budget"] = previous["update_budget"]
    statuses = [case["status"] for case in report["cases"] + report.get("development_schema_cases", [])]
    dependency_pass = (report["dependencies"]["offline_dependency_resolution"]["status"] == "PASS"
                       and all(p["matches"] for p in report["dependencies"]["packages"].values()))
    report["status"] = "PASS" if all(s == "PASS" for s in statuses) and probe["status"] == "PASS" and dependency_pass else "INCOMPLETE_OR_FAILED"
    report["synthetic_optimizer_updates_completed_in_passing_cases"] = sum(
        case.get("checkpoint", {}).get("optimizer_updates", 0) for case in report["cases"])
    report["failed_case_update_counts"] = "Up to 3 each; failed attempts remain precharged. The full-invocation plan is 48; cumulative recovery charges are in update_accounting."
    report["update_accounting"] = update_accounting()
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, portable(report))
    print(f"Overall: {report['status']}; report: {destination.relative_to(ROOT)}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
