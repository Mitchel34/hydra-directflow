#!/usr/bin/env python3
"""Explicit twelve-update full/Q-only fresh-process trainer recovery qualification."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    from hydra_directflow.telemetry import atomic_json as write_atomic
    write_atomic(Path(path), value)
    descriptor = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def identity():
    names = ["scripts/qualify_training_recovery.py", "src/hydra_directflow/training.py",
             "src/hydra_directflow/models/directflow.py", "src/hydra_directflow/models/mamba2_math.py"]
    return {name: file_hash(ROOT / name) for name in names}


def synthetic_arrays():
    import numpy as np
    from hydra_directflow.local_data import utc_epoch, iso
    rng = np.random.default_rng(1729)
    origins = np.array([utc_epoch("2021-01-01T00:00:00Z") + h * 3600 for h in range(8)] +
                       [utc_epoch("2021-01-03T00:00:00Z") + h * 3600 for h in range(4)], dtype=np.int64)
    data = dict(origin_time=origins, history_time=origins[:, None] - np.arange(25, 1, -1) * 3600,
                target_time=origins[:, None] + np.arange(1, 19) * 3600, site_id=np.zeros(12, np.int64),
                case_id=np.array(["03161000:" + iso(int(c)) for c in origins]))
    for name, shape in (("history", (12, 24, 2)), ("weather", (12, 18, 4)), ("state", (12, 2)), ("target", (12, 18))):
        data[name], data[name + "_mask"] = rng.uniform(1., 10., shape), np.ones(shape, bool)
    return data


def tree_digest(value):
    import torch
    out = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            x = item.detach().cpu().contiguous()
            out.update(str((str(x.dtype), tuple(x.shape))).encode()); out.update(x.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                out.update((type(key).__name__ + str(key)).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item: visit(child)
        else:
            out.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return out.hexdigest()


def worker(mode, directory, architecture, device, feature_schema):
    def deny_network(event, unused):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            raise RuntimeError("Network is prohibited in synthetic qualification")
    sys.addaudithook(deny_network)
    precharge = json.loads((directory.parent / "precharge.json").read_text())
    assert precharge["precharged_optimizer_updates"] == precharge["maximum_optimizer_updates"] == 12
    assert precharge["implementation_sha256"] == identity()
    assert dict(architecture=architecture, device=device, feature_schema=feature_schema) in precharge["cases"]
    assert directory.resolve().is_relative_to((ROOT / ".local").resolve())
    with (directory / (mode + ".started")).open("x") as stream:
        stream.write(str(os.getpid())); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(directory, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    import torch
    from hydra_directflow.training import (DevelopmentTrainer, TrainerSpec, EarlyStopping, array_identity,
        q_only_training_bundle, Q_ONLY_FEATURE_SCHEMA)
    from hydra_directflow.local_data import utc_epoch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(min(1., 2 * 1024**3 / torch.mps.recommended_max_memory()))
    arrays = synthetic_arrays(); source = array_identity(arrays)
    cohort = source if feature_schema == Q_ONLY_FEATURE_SCHEMA else None
    if cohort is not None:
        arrays = q_only_training_bundle(arrays, cohort_admission_digest=cohort)
    spec = TrainerSpec(architecture, device, 24, 4, .0001, 17, True, 2, feature_schema=feature_schema)
    checkpoint = directory / "mid-epoch.pt"
    status = dict(optimizer_updates_entered=0, optimizer_updates_confirmed=0, pid=os.getpid(), mps_driver_bytes=0)
    def update(trainer):
        status["mps_driver_bytes"] = torch.mps.driver_allocated_memory() if device == "mps" else 0
        status["optimizer_updates_entered"] += 1; atomic_json(directory / (mode + "-status.json"), status)
        result = trainer.update()
        status["optimizer_updates_confirmed"] += 1; atomic_json(directory / (mode + "-status.json"), status)
        return result
    if mode == "reference":
        trainer = DevelopmentTrainer(arrays, train_indices=list(range(8)), validation_indices=list(range(8, 12)),
                                     boundary_epoch=utc_epoch("2021-01-03T00:00:00Z"), spec=spec, source_identity=source)
        update(trainer)
        sha = trainer.checkpoint(checkpoint)
        atomic_json(directory / "checkpoint-identity.json", dict(sha256=sha, source=source))
    else:
        recorded = json.loads((directory / "checkpoint-identity.json").read_text())
        trainer = DevelopmentTrainer.restore(checkpoint, arrays, expected_sha256=recorded["sha256"], expected_source_identity=source,
                                              expected_feature_schema=feature_schema, expected_cohort_admission_digest=cohort)
    step = update(trainer)
    # The second batch ends the sole epoch. This evaluates only and adds no step.
    trainer.fit_epochs(EarlyStopping(1, 1, 1, 0.))
    completed = directory / (mode + "-completed.pt")
    completed_sha = trainer.checkpoint(completed)
    restored = DevelopmentTrainer.restore(completed, arrays, expected_sha256=completed_sha, expected_source_identity=source,
                                          expected_feature_schema=feature_schema, expected_cohort_admission_digest=cohort)
    assert asdict(restored.stopping) == asdict(trainer.stopping) and restored.sampler.state() == trainer.sampler.state()
    result = dict(pid=os.getpid(), feature_schema=feature_schema, comparison_cohort_sha256=cohort,
                  second_update=step, scaler=asdict(trainer.scaler), sampler=trainer.sampler.state(),
                  stopping=asdict(trainer.stopping), model_sha256=tree_digest(trainer.model.state_dict()),
                  optimizer_sha256=tree_digest(trainer.optimizer.state_dict()), best_model_sha256=tree_digest(trainer.best_model_state),
                  cpu_rng_sha256=tree_digest(torch.get_rng_state()),
                  mps_rng_sha256=tree_digest(torch.mps.get_rng_state()) if device == "mps" else None,
                  source_array_sha256=source, completed_checkpoint_restore_passed=True,
                  actual_updates=status["optimizer_updates_confirmed"], mps_driver_bytes=torch.mps.driver_allocated_memory() if device == "mps" else 0)
    atomic_json(directory / (mode + "-result.json"), result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", choices=("reference", "replay"))
    parser.add_argument("--architecture")
    parser.add_argument("--device")
    parser.add_argument("--feature-schema", choices=("FULL_HRRR_H2_W4_S2_V1", "Q_ONLY_H2_HORIZON1_S0_V1"))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.run_directory, args.architecture, args.device, args.feature_schema); return
    plan = dict(schema="hydra-scientific-trainer-qualification/v1", status="PLANNED_NOT_EXECUTED",
                source="SYNTHETIC_ONLY", maximum_optimizer_updates=12, maximum_wall_seconds=300,
                maximum_worker_bytes=4 * 1024**3, implementation_sha256=identity(),
                cases=[dict(architecture=architecture, device=device, feature_schema=feature)
                       for feature in ("FULL_HRRR_H2_W4_S2_V1", "Q_ONLY_H2_HORIZON1_S0_V1")
                       for architecture, device in (("LSTM", "cpu"), ("Mamba2_Transformer", "mps"))])
    if not args.execute:
        print(json.dumps(plan, indent=2)); return
    import psutil
    if args.run_directory is None or args.report is None: parser.error("Execution requires new run directory and receipt")
    directory = args.run_directory.absolute()
    if not directory.resolve().is_relative_to((ROOT / ".local").resolve()) or directory.exists() or args.report.exists():
        parser.error("Use a new ignored .local run directory and new receipt")
    directory.mkdir(parents=True)
    receipt = {**plan, "status": "RUNNING", "precharged_optimizer_updates": 12, "confirmed_optimizer_updates": 0,
               "attempts": [], "created_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(args.report, receipt); atomic_json(directory / "precharge.json", receipt)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "VECLIB_MAXIMUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    started = time.monotonic()
    try:
        for case in plan["cases"]:
            local = directory / (case["architecture"] + "-" + case["device"] + "-" + case["feature_schema"]); local.mkdir()
            for mode in ("reference", "replay"):
                status_path = local / (mode + "-status.json")
                child, code, peak = None, None, 0
                try:
                    with (local / (mode + ".log")).open("x") as log:
                        child = subprocess.Popen([sys.executable, "-B", __file__, "--worker", mode, "--run-directory", str(local),
                                                  "--architecture", case["architecture"], "--device", case["device"],
                                                  "--feature-schema", case["feature_schema"]], env=env, stdout=log, stderr=log)
                        process = psutil.Process(child.pid)
                        while child.poll() is None:
                            current = json.loads(status_path.read_text()) if status_path.exists() else {}
                            try: peak = max(peak, process.memory_info().rss + current.get("mps_driver_bytes", 0))
                            except psutil.NoSuchProcess: pass
                            if peak > 4 * 1024**3 or time.monotonic() - started > 300:
                                raise RuntimeError("Qualification resource limit reached")
                            time.sleep(.05)
                        code = child.wait()
                finally:
                    if child is not None and child.poll() is None:
                        child.kill(); child.wait(timeout=5)
                    if child is not None: code = child.returncode
                    state = json.loads(status_path.read_text()) if status_path.exists() else {}
                    count = state.get("optimizer_updates_confirmed", 0)
                    receipt["confirmed_optimizer_updates"] += count
                    receipt["attempts"].append(dict(**case, mode=mode, exit_code=code, confirmed_updates=count,
                          entered_updates=state.get("optimizer_updates_entered", 0), count_exact=code == 0,
                          peak_observed_worker_rss_plus_reported_mps=peak))
                    atomic_json(args.report, receipt)
                if code: raise RuntimeError("Qualification child failed; inspect private log")
            reference = json.loads((local / "reference-result.json").read_text()); replay = json.loads((local / "replay-result.json").read_text())
            assert reference["pid"] != replay["pid"]
            for key in reference:
                if key not in ("pid", "actual_updates", "mps_driver_bytes"):
                    assert reference[key] == replay[key], key
            receipt.setdefault("recovery_checks", []).append(dict(**case, distinct_pids=True, exact_next_update_and_sample_order=True,
                        exact_scaler_model_optimizer_rng_and_stopping=True, source_array_sha256=reference["source_array_sha256"]))
        assert receipt["confirmed_optimizer_updates"] == 12
        receipt["status"] = "PASSED_FRESH_PROCESS_TRAINER_RECOVERY"
    except BaseException as error:
        receipt.update(status="FAILED_QUALIFICATION", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.report, receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "confirmed_optimizer_updates", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
