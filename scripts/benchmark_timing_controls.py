#!/usr/bin/env python3
"""Plan-first 200-update Q-only/L24 timing controls on the sealed August cohort.

The completed 600-update benchmark is imported for immutable admission/statistic
helpers only. Its script, allowance and receipts are not modified or reused.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_SPEC = importlib.util.spec_from_file_location("hydra_original_real_batch_helpers", ROOT / "scripts/benchmark_real_batches.py")
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)
require, atomic_json, canonical = base.require, base.atomic_json, base.canonical
WARMUP, SAMPLES, MAX_UPDATES, MAX_WALL_SECONDS = 5, 20, 200, 900
MAX_BYTES, MPS_CAP = 4 * 1024**3, 2 * 1024**3
INPUT_NAMES = ("history", "history_mask", "weather", "weather_mask", "state", "state_mask")
MODES = ("Q_ONLY_L168", "FULL_L24")
DEVICE_POLICY = {"LSTM": "cpu", "Transformer": "mps", "GRU_Transformer": "cpu", "Mamba2_Transformer": "mps"}
FULL_PARAMETERS = {"LSTM": 67905, "Transformer": 85057, "GRU_Transformer": 76545, "Mamba2_Transformer": 86093}
QUALIFICATION = "MATCHED_COHORT_TIMING_CONTROLS_NOT_FORECAST_SKILL_OR_SCIENTIFIC_SELECTION"


def matrix():
    return [dict(id=f"{mode}-{architecture}-{DEVICE_POLICY[architecture]}-B64", input_mode=mode,
                 architecture=architecture, device=DEVICE_POLICY[architecture], batch_size=64,
                 history_hours=168 if mode == "Q_ONLY_L168" else 24,
                 expected_parameter_count=FULL_PARAMETERS[architecture] - (640 if mode == "Q_ONLY_L168" else 0))
            for mode in MODES for architecture in base.ARCHITECTURES]


def implementation_identity():
    result = base.implementation_identity()
    for name in ("scripts/benchmark_timing_controls.py", "src/hydra_directflow/training.py", "src/hydra_directflow/ablations.py"):
        result[name] = base.sha(base._local_bytes(ROOT / name, 256 * 1024))
    return result


def plan(receipt_path):
    admitted = base.plan(receipt_path)
    specification = dict(matrix=matrix(), source=admitted["spec"]["source"],
        fixture_receipt_sha256=admitted["spec"]["fixture_receipt_sha256"],
        source_receipts_sha256=admitted["spec"]["source_receipts_sha256"],
        parent_full_L168_plan_sha256=admitted["plan_sha256"], implementation_sha256=implementation_identity(),
        warmup_updates=WARMUP, measured_updates=SAMPLES, validation_samples=20, inference_samples=20,
        maximum_optimizer_updates=MAX_UPDATES, maximum_campaign_wall_seconds=MAX_WALL_SECONDS,
        combined_parent_worker_rss_mps_bytes=MAX_BYTES, mps_driver_cap_bytes=MPS_CAP,
        validation_origin_utc=base.VALIDATION_START, expected_training_cases=558, expected_validation_cases=168,
        expected_purged_cases=18, dtype="float32", width=64, dropout=.1, seed=1729,
        optimizer="AdamW_foreach_false", learning_rate=1e-4, fixed_unit_mapping=base.UNIT_MAPPING,
        q_only_horizon_mapping="DECLARED_HORIZON_DIV18", original_full_cohort_preserved=True,
        q_only_weather_soil_passed_to_model=False, labels_used_as_predictors=False,
        L24_rule="Exact last24 original history values, masks and UTC slots; no case readmission or relabeling",
        device_policy="Prior full-L168 B64 recommendations held fixed; not a new device-optimality search",
        parameter_matching="Not exact across input arms: Q-only removes640 projection parameters at width64",
        qualification=QUALIFICATION)
    require(len(matrix()) * (WARMUP + SAMPLES) == MAX_UPDATES, "Control update accounting differs")
    return dict(schema="hydra-timing-controls-plan/v1", status="PLANNED_NOT_EXECUTED", spec=specification,
                plan_sha256=base.sha(canonical(specification)), optimizer_updates=0, network_requests=0, cloud_jobs=0)


def prepare_mode(arrays, input_mode, *, cohort_admission_digest):
    """Pure matched-cohort views and the original fixed engineering unit maps."""
    import numpy as np
    from hydra_directflow.training import q_only_training_bundle, validate_coordinates, feature_schema_fields, FULL_FEATURE_SCHEMA, Q_ONLY_FEATURE_SCHEMA
    require(input_mode in MODES, "Explicit prepared timing input mode required")
    validate_coordinates(arrays, 168)
    if input_mode == "Q_ONLY_L168":
        selected = q_only_training_bundle(arrays, cohort_admission_digest=cohort_admission_digest)
        mapped = {}
        # Read only Q history/age and labels. Original weather/state data never
        # enters this branch's transform or the model inputs.
        for name in ("history", "target"):
            source, mask = selected[name], selected[name + "_mask"]
            value = np.zeros(source.shape, dtype=np.float32)
            if name == "target":
                value[mask] = source[mask] / base.UNIT_MAPPING[name][0][1]
            else:
                for column, (offset, scale) in enumerate(base.UNIT_MAPPING[name]):
                    use = mask[..., column]
                    value[..., column][use] = (source[..., column][use] - offset) / scale
            mapped[name], mapped[name + "_mask"] = value, mask.copy()
        mapped["weather"] = (selected["weather"] / 18).astype(np.float32)
        mapped["weather_mask"] = selected["weather_mask"].copy()
        mapped["state"] = selected["state"].astype(np.float32)
        mapped["state_mask"] = selected["state_mask"].copy()
        features = feature_schema_fields(Q_ONLY_FEATURE_SCHEMA)
    else:
        selected = {}
        for name in base.MODEL_ARRAYS + base.COORDINATES:
            value = arrays[name]
            if name in ("history", "history_mask", "history_time"):
                value = value[:, -24:]
            selected[name] = np.array(value, copy=True)
        mapped = base._map_arrays(selected)
        features = feature_schema_fields(FULL_FEATURE_SCHEMA)
        validate_coordinates(selected, 24)
    for name in ("site_id", "origin_time", "target_time", "case_id", "target", "target_mask"):
        require(np.array_equal(selected[name], arrays[name], equal_nan=True) if np.asarray(arrays[name]).dtype.kind == "f"
                else np.array_equal(selected[name], arrays[name]), "Timing control changed original labels or case coordinates")
    require(all(np.isfinite(value).all() for name, value in mapped.items() if not name.endswith("_mask")), "Fixed mapping produced nonfinite model values")
    return selected, mapped, features


def cohort_identities(arrays, split):
    return dict(case_ids_sha256=base.sha(canonical(arrays["case_id"].tolist())),
                train_case_ids_sha256=base.sha(canonical(arrays["case_id"][split["train"]].tolist())),
                validation_case_ids_sha256=base.sha(canonical(arrays["case_id"][split["validation"]].tolist())),
                target_and_coordinates_sha256=base._array_digest({name: arrays[name] for name in
                    ("target", "target_mask", "site_id", "origin_time", "target_time", "case_id")}))


def precharge(plan_value, run_dir):
    directory = ROOT / ".cache"
    directory.mkdir(exist_ok=True)
    base.sync_directory(ROOT)
    ledger_path = directory / "timing-controls-update-budget-v1.json"
    with (directory / "timing-controls-update-budget-v1.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        require(not ledger_path.exists() and not ledger_path.is_symlink(), "The200-update controls allowance is already charged; no retry/increase")
        ledger = dict(schema="hydra-timing-controls-budget/v1", maximum_updates=MAX_UPDATES, precharged_updates=MAX_UPDATES,
                      plan_sha256=plan_value["plan_sha256"], run_directory=str(Path(run_dir).resolve()),
                      cases=[dict(id=case["id"], precharged_updates=25) for case in matrix()],
                      created_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(ledger_path, ledger)
        base.sync_directory(directory)
        return {key: value for key, value in ledger.items() if key != "run_directory"}


def verify_allowance(run_dir, plan_value):
    run_dir = Path(run_dir).resolve()
    require(any(run_dir.is_relative_to((ROOT / name).resolve()) for name in ("runs", ".local")), "Controls workers require an ignored run directory")
    ledger = json.loads(base._local_bytes(ROOT / ".cache/timing-controls-update-budget-v1.json", 65536))
    require(ledger["plan_sha256"] == plan_value["plan_sha256"] and Path(ledger["run_directory"]).resolve() == run_dir
            and ledger["maximum_updates"] == ledger["precharged_updates"] == MAX_UPDATES
            and ledger["cases"] == [dict(id=case["id"], precharged_updates=25) for case in matrix()],
            "Controls worker requires its exact separate precharged allowance")


def worker(args):
    started = time.monotonic()
    def deny_network(event, unused):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            raise RuntimeError("Network prohibited in local timing controls")
    sys.addaudithook(deny_network)
    run_dir = args.run_directory.resolve()
    require(any(run_dir.is_relative_to((ROOT / name).resolve()) for name in ("runs", ".local")), "Controls workers require an ignored run directory")
    request = json.loads(base._local_bytes(run_dir / "private-request.json", 2 * 1024**2))
    actual = plan(request["receipt_path"])
    require(actual["plan_sha256"] == request["plan"]["plan_sha256"], "Control source/implementation/plan changed")
    verify_allowance(run_dir, actual)
    require(0 < args.wall_limit <= MAX_WALL_SECONDS, "Invalid worker wall allowance")
    case = next(case for case in matrix() if case["id"] == args.case)
    base.claim_worker_case(run_dir / (args.case + ".started.json"), dict(pid=os.getpid(), plan_sha256=actual["plan_sha256"], precharged_updates=25))
    result = dict(**case, status="FAILED", plan_sha256=actual["plan_sha256"], optimizer_updates_precharged=25,
                  optimizer_updates_entered=0, optimizer_updates_confirmed=0, warmup_updates_confirmed=0,
                  update_count_exact=False, training_samples=[], warmup_samples=[], validation_samples=[], inference_samples=[],
                  stage_seconds={}, forecast_skill_reported=False, scientific_weights_selected=False)
    stage = result["stage_seconds"]
    status_path, result_path = run_dir / (args.case + ".status.json"), run_dir / (args.case + ".json")
    stopped, monitor_errors, watcher = threading.Event(), [], None
    try:
        tick = time.monotonic()
        import torch
        import psutil
        from hydra_directflow.models.directflow import DirectFlowModel, ModelConfig, masked_mse
        stage["imports"] = time.monotonic() - tick
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(1729)
        device, batch_size = case["device"], case["batch_size"]
        if device == "mps":
            require(torch.backends.mps.is_available(), "MPS unavailable; no fallback")
            torch.mps.set_per_process_memory_fraction(min(1., MPS_CAP / torch.mps.recommended_max_memory()))
        process = psutil.Process()
        peaks = dict(worker_rss_bytes=0, mps_driver_bytes=0)
        def snapshot():
            rss, driver = process.memory_info().rss, torch.mps.driver_allocated_memory() if device == "mps" else 0
            peaks["worker_rss_bytes"], peaks["mps_driver_bytes"] = max(peaks["worker_rss_bytes"], rss), max(peaks["mps_driver_bytes"], driver)
            return dict(worker_rss_bytes=rss, mps_driver_bytes=driver, optimizer_updates_entered=result["optimizer_updates_entered"],
                        optimizer_updates_confirmed=result["optimizer_updates_confirmed"])
        def monitor():
            try:
                while not stopped.is_set():
                    atomic_json(status_path, snapshot())
                    stopped.wait(.05)
            except Exception as error:
                monitor_errors.append(type(error).__name__)
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        def sync():
            if device == "mps":
                torch.mps.synchronize()
        def guard():
            require(not monitor_errors and time.monotonic() - started < args.wall_limit, "Control wall/telemetry guard failed")
            current = snapshot()
            require(current["worker_rss_bytes"] + current["mps_driver_bytes"] <= MAX_BYTES, "Control memory cap reached")
        def timed(callback):
            guard(); sync()
            tick = time.monotonic()
            value = callback()
            sync()
            duration = time.monotonic() - tick
            guard()
            return value, duration
        receipt = json.loads(base._local_bytes(request["receipt_path"], 2 * 1024**2))
        (full, split), stage["fixture_read_authentication"] = timed(lambda: base.load_fixture(request["fixture_path"], receipt))
        def transform():
            selected, mapped, features = prepare_mode(full, case["input_mode"], cohort_admission_digest=receipt["case_source_digest"])
            return selected, {name: torch.from_numpy(value) for name, value in mapped.items()}, features
        (selected, host, features), stage["fixed_unit_mapping_tensorization"] = timed(transform)
        original_cohort, control_cohort = cohort_identities(full, split), cohort_identities(selected, split)
        require(original_cohort == control_cohort, "Control changed matched cohort/target identities")
        config = ModelConfig(case["architecture"], features["history"], features["weather"], features["state"],
                             case["history_hours"], tuple(range(1, 19)), width=64, dropout=.1)
        def setup():
            model = DirectFlowModel(config).to(device)
            return model, torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
        (model, optimizer), stage["model_optimizer_setup"] = timed(setup)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        require(parameters == case["expected_parameter_count"], "Observed input-geometry parameter count differs")
        result.update(parameter_count=parameters, parameter_count_observed=True, features=features, environment=base.environment_metadata(torch),
                      cohort_counts={key: len(value) for key, value in split.items()}, **control_cohort,
                      prepared_array_sha256=base._array_digest(selected), original_fixture_array_sha256=receipt["arrays_sha256"],
                      exact_parameter_matching_across_arms=False,
                      copy_semantics="CPU gather and explicit copy including labels for training/validation; inference excludes labels")
        def transfer(indices, labels=True):
            names = base.MODEL_ARRAYS if labels else INPUT_NAMES
            return {name: host[name][indices].to(device=device, copy=True) for name in names}
        model.train()
        mark = time.monotonic()
        for step in range(WARMUP + SAMPLES):
            indices = base.batch_indices(split["train"], batch_size, step)
            batch, copy_seconds = timed(lambda: transfer(indices))
            def update():
                optimizer.zero_grad(set_to_none=True)
                prediction = model(*(batch[name] for name in INPUT_NAMES))
                loss = masked_mse(prediction, batch["target"], batch["target_mask"])
                require(bool(torch.isfinite(loss)), "Nonfinite loss")
                loss.backward()
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "Invalid gradient")
                result["optimizer_updates_entered"] += 1
                optimizer.step(); sync()
                result["optimizer_updates_confirmed"] += 1
                require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Nonfinite updated parameter")
            _, seconds = timed(update)
            sample = dict(step=step + 1, cases=batch_size, copy_seconds=copy_seconds, update_seconds=seconds)
            if step < WARMUP:
                result["warmup_samples"].append(sample)
                result["warmup_updates_confirmed"] += 1
            else:
                result["training_samples"].append(sample)
        stage["training_loops_including_guards"] = time.monotonic() - mark
        model.eval()
        with torch.inference_mode():
            mark = time.monotonic()
            for step in range(21):
                batch, copy_seconds = timed(lambda: transfer(base.batch_indices(split["validation"], batch_size, step)))
                def validation():
                    prediction = model(*(batch[name] for name in INPUT_NAMES))
                    loss = masked_mse(prediction, batch["target"], batch["target_mask"])
                    require(bool(torch.isfinite(loss)), "Nonfinite validation mechanics loss")
                _, seconds = timed(validation)
                sample = dict(cases=batch_size, copy_seconds=copy_seconds, forward_loss_seconds=seconds)
                if step == 0:
                    result["first_eval_sample"] = sample
                else:
                    result["validation_samples"].append(sample)
            stage["validation_loops_including_guards"] = time.monotonic() - mark
            mark = time.monotonic()
            for step in range(20):
                inputs, copy_seconds = timed(lambda: transfer(base.batch_indices(split["validation"], batch_size, step), labels=False))
                prediction, seconds = timed(lambda: model(*(inputs[name] for name in INPUT_NAMES)).detach().cpu())
                require(prediction.shape == (batch_size, 18) and bool(torch.isfinite(prediction).all()), "Invalid inference output")
                result["inference_samples"].append(dict(cases=batch_size, copy_seconds=copy_seconds, forward_to_cpu_seconds=seconds))
                reference_inputs, reference_prediction = inputs, prediction
            stage["inference_loops_including_guards"] = time.monotonic() - mark
        checkpoint = run_dir / (args.case + ".private-checkpoint.pt")
        def save():
            state = dict(plan_sha256=actual["plan_sha256"], input_mode=case["input_mode"], cohort=control_cohort,
                         model=base._cpu_tree(model.state_dict(), torch), optimizer=base._cpu_tree(optimizer.state_dict(), torch),
                         cpu_rng=torch.get_rng_state(), mps_rng=torch.mps.get_rng_state().cpu() if device == "mps" else None,
                         optimizer_updates=result["optimizer_updates_confirmed"])
            with checkpoint.open("xb") as stream:
                torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
            base.sync_directory(checkpoint.parent)
        _, stage["checkpoint_write"] = timed(save)
        checkpoint_sha = base.sha(base._local_bytes(checkpoint, 32 * 1024**2))
        def restore():
            raw = base._local_bytes(checkpoint, 32 * 1024**2)
            require(base.sha(raw) == checkpoint_sha, "Checkpoint bytes changed")
            state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            require(state["plan_sha256"] == actual["plan_sha256"] and state["input_mode"] == case["input_mode"] and state["cohort"] == control_cohort,
                    "Checkpoint input-mode/cohort differs")
            restored, restored_optimizer = setup()
            restored.load_state_dict(state["model"]); restored_optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["cpu_rng"])
            if device == "mps": torch.mps.set_rng_state(state["mps_rng"])
            return restored.eval(), restored_optimizer
        (restored, restored_optimizer), stage["checkpoint_read_restore"] = timed(restore)
        with torch.inference_mode():
            replay, stage["checkpoint_forward_verification"] = timed(lambda: restored(*(reference_inputs[name] for name in INPUT_NAMES)).detach().cpu())
        require(torch.equal(reference_prediction, replay), "Control checkpoint forward replay differs")
        result.update(status="PASS", update_count_exact=True, checkpoint_forward_exact=True,
                      checkpoint=dict(bytes=checkpoint.stat().st_size, sha256=checkpoint_sha),
                      worker_observed_peaks=peaks, warmup_sample_visits=5 * batch_size, timed_training_sample_visits=20 * batch_size,
                      unique_training_cases_visited=len({i for step in range(25) for i in base.batch_indices(split["train"], batch_size, step)}),
                      validation_sample_visits=20 * batch_size, inference_sample_visits=20 * batch_size,
                      scaler_fits=0, optimizer_updates_in_validation_inference=0, protected_assessment_scores=0)
        result["summary"] = {label: base.summarize([sample["copy_seconds"] + sample[duration] for sample in result[field]])
            for label, field, duration in (("train_batch_total_seconds", "training_samples", "update_seconds"),
                 ("validation_batch_total_seconds", "validation_samples", "forward_loss_seconds"),
                 ("inference_batch_total_seconds", "inference_samples", "forward_to_cpu_seconds"))}
        result["examples_per_second_at_median_train_batch"] = batch_size / result["summary"]["train_batch_total_seconds"]["median"]
    except Exception as error:
        result.update(status="FAILED", error_type=type(error).__name__, error=base.portable(str(error))[:2000])
    finally:
        stopped.set()
        if watcher is not None:
            watcher.join(timeout=1)
            if watcher.is_alive() or monitor_errors:
                result.update(status="FAILED", update_count_exact=False, error="Control telemetry did not finish")
        result["worker_wall_seconds"] = time.monotonic() - started
        atomic_json(result_path, base.portable(result))
        atomic_json(status_path, {key: result[key] for key in ("optimizer_updates_entered", "optimizer_updates_confirmed")})


def supervise(case, run_dir, deadline):
    import psutil
    remaining = deadline - time.monotonic()
    require(remaining > 0, "Control campaign wall allowance exhausted")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    status_path = run_dir / (case["id"] + ".status.json")
    child, peak, last, reason, interrupted = None, 0, {}, None, None
    started = time.monotonic()
    try:
        with (run_dir / (case["id"] + ".log")).open("x") as log:
            child = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--case", case["id"],
                                      "--run-directory", str(run_dir), "--wall-limit", str(remaining)],
                                     cwd=ROOT, env=env, stdout=log, stderr=log)
            process, parent = psutil.Process(child.pid), psutil.Process()
            while child.poll() is None:
                try:
                    rss = process.memory_info().rss + parent.memory_info().rss + sum(p.memory_info().rss for p in process.children(recursive=True))
                    if status_path.exists(): last = json.loads(base._local_bytes(status_path, 65536))
                    peak = max(peak, rss + last.get("mps_driver_bytes", 0))
                    if peak > MAX_BYTES or time.monotonic() >= deadline:
                        reason = "Control campaign memory/wall limit reached"; child.kill(); break
                except psutil.NoSuchProcess: break
                time.sleep(.05)
            child.wait(timeout=5)
    except BaseException as error:
        reason, interrupted = "Supervisor interrupted: " + type(error).__name__, type(error).__name__
    finally:
        try:
            if child is not None and child.poll() is None:
                child.kill(); child.wait(timeout=5)
        except BaseException as error:
            reason, interrupted = reason or "Child reaping not confirmed", interrupted or type(error).__name__
    try:
        if status_path.exists(): last = json.loads(base._local_bytes(status_path, 65536))
    except (OSError, ValueError): reason = reason or "Final status unreadable"
    path = run_dir / (case["id"] + ".json")
    try: result = json.loads(base._local_bytes(path, 256 * 1024)) if path.exists() else None
    except (OSError, ValueError): result = None; reason = reason or "Worker result unreadable"
    if result is None:
        result = dict(**case, status="FAILED", optimizer_updates_precharged=25, update_count_exact=False,
                      optimizer_updates_entered=0, optimizer_updates_confirmed=0)
    for key in ("optimizer_updates_entered", "optimizer_updates_confirmed"):
        result[key] = max([v for v in (last.get(key, 0), result.get(key, 0)) if type(v) is int and 0 <= v <= 25], default=0)
    result.update(combined_observed_peak_bytes=peak, supervised_wall_seconds=time.monotonic() - started)
    if reason or child is None or child.returncode:
        result.update(status="FAILED", update_count_exact=False, stop_reason=reason, supervisor_interruption_type=interrupted,
                      exit_code=child.returncode if child is not None else None)
    return result


def run_campaign(*, receipt_path, fixture_path, run_directory, report_path, approved_plan_sha256):
    planned = plan(receipt_path)
    require(planned["plan_sha256"] == approved_plan_sha256, "Exact reviewed controls plan required")
    directory, report_path = Path(run_directory).resolve(), Path(report_path).absolute()
    require(any(directory.is_relative_to((ROOT / name).resolve()) for name in ("runs", ".local")), "Private controls directory must be ignored")
    require(not directory.exists() and not report_path.exists() and not report_path.is_symlink(), "Controls directory and receipt must be new")
    receipt = json.loads(base._local_bytes(receipt_path, 2 * 1024**2))
    full, split = base.load_fixture(fixture_path, receipt)
    original = cohort_identities(full, split)
    for mode in MODES:
        selected, _, _ = prepare_mode(full, mode, cohort_admission_digest=receipt["case_source_digest"])
        require(cohort_identities(selected, split) == original, "Preflight changed matched cohort/labels")
    del full, selected
    ledger = precharge(planned, directory)
    journal = base.RunJournal(directory, planned, max_wall_seconds=MAX_WALL_SECONDS, max_rss_bytes=MAX_BYTES)
    base.sync_directory(directory.parent)
    atomic_json(directory / "private-request.json", dict(plan=planned, receipt_path=str(Path(receipt_path).resolve()), fixture_path=str(Path(fixture_path).resolve())))
    started, deadline = time.monotonic(), time.monotonic() + MAX_WALL_SECONDS - 1
    result = dict(schema="hydra-timing-controls/v1", status="RUNNING", qualification=QUALIFICATION, plan=planned, allowance=ledger,
                  optimizer_updates_precharged=MAX_UPDATES, cases=[], matched_cohort=original,
                  created_at_utc=datetime.now(timezone.utc).isoformat(), scientific_selection=False, network_requests=0, cloud_jobs=0)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(report_path, result)
    try:
        for case in matrix():
            with journal.stage("case_" + case["id"].lower().replace("-", "_")):
                completed = supervise(case, directory, deadline)
                result["cases"].append(completed)
                journal.event("case_finished", case_id=case["id"], status=completed["status"], optimizer_updates_confirmed=completed["optimizer_updates_confirmed"])
                atomic_json(report_path, base.portable(result))
            print(json.dumps({key: completed.get(key) for key in ("id", "status", "parameter_count", "optimizer_updates_confirmed")}), flush=True)
            require(completed["status"] == "PASS", "Controls campaign stopped after failed case; no retry")
        result["status"] = "PASS"; journal.finish()
    except BaseException as error:
        result.update(status="PARTIAL_OR_FAILED", error_type=type(error).__name__, error=base.portable(str(error))[:2000])
        if journal.status == "RUNNING": journal.finish("STOPPED", error_type=type(error).__name__)
    result.update(optimizer_updates_entered=sum(row.get("optimizer_updates_entered", 0) for row in result["cases"]),
                  optimizer_updates_confirmed=sum(row.get("optimizer_updates_confirmed", 0) for row in result["cases"]),
                  warmup_updates_confirmed=sum(row.get("warmup_updates_confirmed", 0) for row in result["cases"]),
                  measured_updates_confirmed=sum(len(row.get("training_samples", [])) for row in result["cases"]),
                  all_reported_update_counts_exact=all(row.get("update_count_exact", False) for row in result["cases"]),
                  unstarted_case_count=len(matrix()) - len(result["cases"]), failed_and_unstarted_allowances_retained=True,
                  elapsed_seconds=time.monotonic() - started, observed_combined_peak_bytes=max((row.get("combined_observed_peak_bytes", 0) for row in result["cases"]), default=0),
                  journal_status=journal.status)
    atomic_json(report_path, base.portable(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=[c["id"] for c in matrix()], help=argparse.SUPPRESS)
    parser.add_argument("--wall-limit", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args); return
    if args.receipt is None: parser.error("Sealed original fixture receipt required")
    if not args.execute:
        print(json.dumps(plan(args.receipt), indent=2, sort_keys=True, allow_nan=False)); return
    if any(v is None for v in (args.fixture, args.approved_plan_sha256, args.run_directory, args.report)):
        parser.error("Execution needs fixture, exact approved plan and new private run directory/receipt")
    result = run_campaign(receipt_path=args.receipt, fixture_path=args.fixture, run_directory=args.run_directory,
                          report_path=args.report, approved_plan_sha256=args.approved_plan_sha256)
    if result["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
