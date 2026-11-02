#!/usr/bin/env python3
"""Plan-first, fixed 600-update local real-batch engineering benchmark.

Twenty-four fresh workers use the same sealed August 2023 fixture. Each has
five warmup and twenty measured updates; every allowance is charged before the
campaign. No retries, network, scientific selection, or cloud path is provided.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hydra_directflow.contract import require
from hydra_directflow.engineering import FEATURES, UNIT_MAPPING, _cpu_tree, _local_bytes
from hydra_directflow.pipeline_timing import _array_digest, _map_arrays, environment_metadata
from hydra_directflow.quality import PRIMARY_APPROVED
from hydra_directflow.telemetry import RunJournal, atomic_json, canonical

ARCHITECTURES = ("LSTM", "Transformer", "GRU_Transformer", "Mamba2_Transformer")
BATCHES = (16, 32, 64)
WARMUP, SAMPLES, VALIDATION_SAMPLES, INFERENCE_SAMPLES = 5, 20, 20, 20
UPDATES_PER_CASE, MAX_UPDATES = WARMUP + SAMPLES, 600
MAX_WALL_SECONDS, MAX_BYTES, MPS_CAP = 900, 4 * 1024**3, 2 * 1024**3
MODEL_ARRAYS = ("history", "history_mask", "weather", "weather_mask", "state", "state_mask", "target", "target_mask")
COORDINATES = ("site_id", "origin_time", "history_time", "target_time", "case_id", "source_cycle")
VALIDATION_START = "2023-08-25T00:00:00Z"
QUALIFICATION = "REAL_DATA_LOCAL_ENGINEERING_THROUGHPUT_NOT_SCIENTIFIC_TRAINING_OR_SKILL"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def epoch(stamp):
    return int(datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp())


def is_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def matrix():
    return [dict(id=f"{device}-{architecture}-B{batch}", architecture=architecture, device=device, batch_size=batch)
            for device in ("cpu", "mps") for architecture in ARCHITECTURES for batch in BATCHES]


def receipt_identity(receipt):
    require(receipt.get("schema") == "hydra-real-batch-fixture/v1", "Sealed real-batch fixture receipt required")
    require(receipt["gauge_id"] == "03161000" and receipt["history_hours"] == 168
            and receipt["horizons_hours"] == list(range(1, 19)), "Fixed gauge/L168/H18 fixture required")
    require(receipt["input_quality_policy"] == receipt["target_quality_policy"] == PRIMARY_APPROVED,
            "PRIMARY_APPROVED must independently govern inputs and labels")
    require(is_digest(receipt["npz"]["sha256"]) and type(receipt["npz"]["bytes"]) is int
            and 0 < receipt["npz"]["bytes"] <= 32 * 1024**2, "Bounded fixture identity required")
    require(is_digest(receipt["case_source_digest"]) and is_digest(receipt["arrays_sha256"]), "Source and array identities required")
    require(isinstance(receipt["source_receipts"], list) and receipt["source_receipts"], "Original preprocessing source receipts required")
    for source in receipt["source_receipts"]:
        require(source.get("schema") == "hydra-cached-development/v1"
                and source.get("input_quality_policy") == source.get("target_quality_policy") == PRIMARY_APPROVED
                and source.get("policy", {}).get("units") == "PHYSICAL_UNSCALED"
                and source.get("execution", {}).get("optimizer_updates") == 0
                and source.get("execution", {}).get("protected_assessment_values_decoded") == 0,
                "Source receipts must bind primary-quality physical units before fitting and exclude assessment")
    return {key: receipt[key] for key in ("schema", "gauge_id", "history_hours", "horizons_hours", "input_quality_policy",
                                         "target_quality_policy", "npz", "case_source_digest", "arrays_sha256")}


def implementation_identity():
    files = [Path(__file__), ROOT / "src/hydra_directflow/pipeline_timing.py", ROOT / "src/hydra_directflow/engineering.py",
             ROOT / "src/hydra_directflow/quality.py",
             ROOT / "src/hydra_directflow/telemetry.py", ROOT / "src/hydra_directflow/models/directflow.py",
             ROOT / "src/hydra_directflow/models/mamba2_math.py"]
    return {str(path.relative_to(ROOT)): sha(_local_bytes(path, 256 * 1024)) for path in files}


def plan(receipt_path):
    raw = _local_bytes(receipt_path, 2 * 1024**2)
    receipt = json.loads(raw)
    source = receipt_identity(receipt)
    specification = dict(matrix=matrix(), warmup_updates=WARMUP, measured_updates=SAMPLES,
                         validation_samples=VALIDATION_SAMPLES, inference_samples=INFERENCE_SAMPLES,
                         maximum_optimizer_updates=MAX_UPDATES, maximum_campaign_wall_seconds=MAX_WALL_SECONDS,
                         combined_parent_worker_rss_mps_bytes=MAX_BYTES, mps_driver_cap_bytes=MPS_CAP,
                         history_hours=168, horizons_hours=list(range(1, 19)), model_width=64,
                         dtype="float32", dropout=.1, seed=1729, optimizer="AdamW_foreach_false", learning_rate=1e-4,
                         validation_origin_utc=VALIDATION_START, split="train final label < validation origin; validation origin >= boundary",
                         expected_training_cases=558, expected_validation_cases=168, expected_purged_cases=18,
                         unit_mapping=UNIT_MAPPING, masks="original fixture masks preserved; fixed-unit zero placeholders only",
                         all_batch_cases_distinct=True, repeated_passes_for_timing=True, weights_used_for_scientific_selection=False,
                         source=source, fixture_receipt_sha256=sha(raw), source_receipts_sha256=sha(canonical(receipt["source_receipts"])),
                         implementation_sha256=implementation_identity(), qualification=QUALIFICATION)
    require(len(matrix()) * UPDATES_PER_CASE == MAX_UPDATES, "Fixed campaign update accounting differs")
    return dict(schema="hydra-real-batch-benchmark-plan/v1", status="PLANNED_NOT_EXECUTED",
                spec=specification, plan_sha256=sha(canonical(specification)), cloud_jobs=0, optimizer_updates=0)


def batch_indices(indices, batch_size, step):
    """Cycle a fixed real cohort; no duplicate case can occur within a batch."""
    require(type(batch_size) is int and 0 < batch_size <= len(indices) and type(step) is int and step >= 0,
            "A nonnegative step and batch no larger than the real cohort are required")
    require(len(set(indices)) == len(indices), "Cohort indices must be unique")
    offset = step * batch_size
    return [indices[(offset + position) % len(indices)] for position in range(batch_size)]


def split_coordinates(origins, targets):
    boundary = epoch(VALIDATION_START)
    train, validation, purged = [], [], []
    for i, (origin, labels) in enumerate(zip(origins, targets)):
        require(labels == [origin + h * 3600 for h in range(1, 19)], "H18 target clocks differ")
        if origin >= boundary:
            validation.append(i)
        elif labels[-1] < boundary:
            train.append(i)
        else:
            purged.append(i)
    require((len(train), len(validation), len(purged)) == (558, 168, 18), "Exact chronological August cohort differs")
    return dict(train=train, validation=validation, purged=purged)


def load_fixture(path, receipt):
    """Bounded numerical fixture admission; source identities stay receipt-bound."""
    import numpy as np
    from hydra_directflow.local_data import bounded_npy
    identity = receipt_identity(receipt)
    raw = _local_bytes(path, 32 * 1024**2)
    require(len(raw) == identity["npz"]["bytes"] and sha(raw) == identity["npz"]["sha256"], "Private fixture SHA/size differs")
    shapes = dict(history=(744, 168, 2), weather=(744, 18, 4), state=(744, 2), target=(744, 18),
                  site_id=(744,), origin_time=(744,), history_time=(744, 168), target_time=(744, 18),
                  case_id=(744,), source_cycle=(744,), soil_source_cycle=(744,))
    arrays = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        names = {m.filename for m in members}
        required = {name + ".npy" for name in MODEL_ARRAYS + COORDINATES}
        require(required <= names <= required | {"soil_source_cycle.npy"} and len(members) == len(names)
                and sum(m.file_size for m in members) <= 64 * 1024**2, "Fixture NPZ member/expansion bound differs")
        for member in members:
            name = member.filename[:-4]
            blob = archive.read(member.filename)
            shape = shapes[name[:-5]] if name.endswith("_mask") else shapes[name]
            dtype = "?" if name.endswith("_mask") else "<i8" if name in ("site_id", "origin_time", "history_time", "target_time") else "<f8"
            if name in ("case_id", "source_cycle", "soil_source_cycle"):
                stream = io.BytesIO(blob)
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), "Unsupported coordinate array format")
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                actual_shape, fortran, dtype = reader(stream)
                require(actual_shape == shape and not fortran and dtype.kind == "U" and 0 < dtype.itemsize <= 512,
                        "Bounded Unicode case/source coordinates required")
            arrays[name] = bounded_npy(blob, shape=shape, dtype=dtype)
    require(_array_digest(arrays) == identity["arrays_sha256"], "Fixture array/provenance digest differs")
    expected_origins = np.arange(epoch("2023-08-01T00:00:00Z"), epoch("2023-09-01T00:00:00Z"), 3600, dtype=np.int64)
    require(np.array_equal(arrays["origin_time"], expected_origins) and np.all(arrays["site_id"] == 0), "Every August origin at the selected gauge is required once")
    expected_ids = ["03161000:" + datetime.fromtimestamp(int(t), timezone.utc).isoformat().replace("+00:00", "Z") for t in expected_origins]
    require(arrays["case_id"].tolist() == expected_ids, "Case IDs differ from exact chronological gauge/origin coordinates")
    require(np.array_equal(arrays["history_time"], expected_origins[:, None] - np.arange(169, 1, -1)[None, :] * 3600),
            "History must contain L168 exact slots ending at C minus two hours")
    for name in ("history", "weather", "state", "target"):
        values, mask = arrays[name], arrays[name + "_mask"]
        require(np.isfinite(values[mask]).all(), "Admitted model values must be finite")
        # The selected gauge's retained month is complete. This restriction does
        # not replace the main study's separately recorded missingness policy.
        require(mask.all(), "This fixed throughput cohort requires complete source-supported inputs/labels")
    require((arrays["target"] >= 0).all() and (arrays["history"][..., 0] >= 0).all(), "Negative admitted discharge")
    require(np.array_equal(arrays["history"][..., 1], np.broadcast_to(np.arange(169, 1, -1), (744, 168))), "History age feature differs")
    for origin, cycle in zip(expected_origins, arrays["source_cycle"]):
        reference = int(datetime.strptime(str(cycle), "%Y%m%d_%HZ").replace(tzinfo=timezone.utc).timestamp())
        require(reference % 21600 == 0 and origin - 18 * 3600 <= reference <= origin - 2 * 3600,
                "Selected weather source cannot cover eligible single-run H18")
    if "soil_source_cycle" in arrays:
        for i, (origin, cycle) in enumerate(zip(expected_origins, arrays["soil_source_cycle"])):
            reference = int(datetime.strptime(str(cycle), "%Y%m%d_%HZ").replace(tzinfo=timezone.utc).timestamp())
            require(reference % 21600 == 0 and 2 <= (origin - reference) / 3600 <= 8
                    and arrays["state"][i, 1] == (origin - reference) / 3600, "Soil availability/age coordinates differ")
    split = split_coordinates(arrays["origin_time"].tolist(), arrays["target_time"].tolist())
    return arrays, split


def summarize(values):
    require(values and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values), "Positive finite timing samples required")
    return dict(count=len(values), total=sum(values), mean=statistics.mean(values), median=statistics.median(values),
                empirical_p95=sorted(values)[math.ceil(.95 * len(values)) - 1], percentile_method="nearest_rank")


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def claim_worker_case(path, value):
    """Durable exclusive claim precedes imports, data admission and updates."""
    path = Path(path)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def precharge(plan_value, run_directory):
    """The fixed matrix is charged once; failures/unstarted cases never refund."""
    directory = ROOT / ".cache"
    directory.mkdir(exist_ok=True)
    sync_directory(ROOT)
    ledger_path = directory / "real-batch-update-budget-v1.json"
    with (directory / "real-batch-update-budget-v1.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        require(not ledger_path.exists() and not ledger_path.is_symlink(), "This 600-update campaign allowance is already charged; no retry or increase")
        ledger = dict(schema="hydra-real-batch-update-budget/v1", maximum_updates=MAX_UPDATES,
                      precharged_updates=MAX_UPDATES, plan_sha256=plan_value["plan_sha256"],
                      run_directory=str(Path(run_directory).resolve()), created_at_utc=datetime.now(timezone.utc).isoformat(),
                      cases=[dict(id=case["id"], precharged_updates=UPDATES_PER_CASE) for case in matrix()])
        atomic_json(ledger_path, ledger)
        sync_directory(directory)
        return {key: value for key, value in ledger.items() if key != "run_directory"}


def verify_worker_allowance(run_dir, plan_value):
    run_dir = Path(run_dir).resolve()
    require(any(run_dir.is_relative_to((ROOT / name).resolve()) for name in ("runs", ".local")), "Worker run directory must be ignored")
    ledger = json.loads(_local_bytes(ROOT / ".cache/real-batch-update-budget-v1.json", 128 * 1024))
    require(ledger["plan_sha256"] == plan_value["plan_sha256"] and ledger["precharged_updates"] == ledger["maximum_updates"] == MAX_UPDATES
            and Path(ledger["run_directory"]).resolve() == run_dir
            and ledger["cases"] == [dict(id=c["id"], precharged_updates=UPDATES_PER_CASE) for c in matrix()],
            "Worker requires the exact campaign precharge and run directory")


def portable(value):
    if isinstance(value, str):
        return value.replace(str(ROOT), "[repo]").replace(str(Path.home()), "[home]")
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    return value


def worker(args):
    started = time.monotonic()
    def deny_network(event, unused):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            raise RuntimeError("Network prohibited during real-batch timing")
    sys.addaudithook(deny_network)
    run_dir = args.run_directory.resolve()
    require(any(run_dir.is_relative_to((ROOT / name).resolve()) for name in ("runs", ".local")), "Worker run directory must be ignored")
    permit = json.loads(_local_bytes(run_dir / "private-request.json", 2 * 1024**2))
    actual = plan(permit["receipt_path"])
    require(actual["plan_sha256"] == permit["plan"]["plan_sha256"], "Source/specification/implementation changed")
    verify_worker_allowance(run_dir, actual)
    case = next(case for case in matrix() if case["id"] == args.case)
    require(0 < args.wall_limit <= MAX_WALL_SECONDS, "Worker wall allowance exceeds campaign cap")
    # A worker command cannot be replayed against its old allowance.
    claim_worker_case(run_dir / (args.case + ".started.json"),
                      dict(pid=os.getpid(), precharged_updates=UPDATES_PER_CASE, plan_sha256=actual["plan_sha256"]))
    result = dict(**case, status="FAILED", optimizer_updates_precharged=UPDATES_PER_CASE,
                  optimizer_updates_entered=0, optimizer_updates_confirmed=0, warmup_updates_confirmed=0,
                  training_samples=[], warmup_samples=[], validation_samples=[], inference_samples=[],
                  plan_sha256=actual["plan_sha256"], update_count_exact=False, forecast_skill_reported=False,
                  scientific_weights_selected=False, stage_seconds={})
    result_path, status_path = run_dir / (args.case + ".json"), run_dir / (args.case + ".status.json")
    stopped, monitor_errors = threading.Event(), []
    watcher = None
    stage = result["stage_seconds"]
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
            require(torch.backends.mps.is_available(), "MPS unavailable; no CPU fallback")
            torch.mps.set_per_process_memory_fraction(min(1., MPS_CAP / torch.mps.recommended_max_memory()))
        process = psutil.Process()
        peaks = dict(worker_rss_bytes=0, mps_driver_bytes=0)
        def snapshot():
            rss = process.memory_info().rss
            driver = torch.mps.driver_allocated_memory() if device == "mps" else 0
            peaks["worker_rss_bytes"] = max(peaks["worker_rss_bytes"], rss)
            peaks["mps_driver_bytes"] = max(peaks["mps_driver_bytes"], driver)
            return dict(worker_rss_bytes=rss, mps_driver_bytes=driver,
                        optimizer_updates_entered=result["optimizer_updates_entered"], optimizer_updates_confirmed=result["optimizer_updates_confirmed"])
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
            require(not monitor_errors and time.monotonic() - started < args.wall_limit, "Worker wall/telemetry guard failed")
            current = snapshot()
            require(current["worker_rss_bytes"] + current["mps_driver_bytes"] <= MAX_BYTES, "Worker memory limit reached")
        def timed(callback):
            guard()
            sync()
            tick = time.monotonic()
            value = callback()
            sync()
            elapsed = time.monotonic() - tick
            guard()
            return value, elapsed
        receipt = json.loads(_local_bytes(permit["receipt_path"], 2 * 1024**2))
        (arrays, split), stage["fixture_read_authentication"] = timed(lambda: load_fixture(permit["fixture_path"], receipt))
        host, stage["fixed_unit_mapping_tensorization"] = timed(lambda: {name: torch.from_numpy(value) for name, value in _map_arrays(arrays).items()})
        config = ModelConfig(case["architecture"], FEATURES["history"], FEATURES["weather"], FEATURES["state"], 168, tuple(range(1, 19)), width=64, dropout=.1)
        def setup():
            model = DirectFlowModel(config).to(device)
            return model, torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
        (model, optimizer), stage["model_optimizer_setup"] = timed(setup)
        result.update(environment=environment_metadata(torch), parameter_count=sum(p.numel() for p in model.parameters()),
                      cohort_counts={key: len(value) for key, value in split.items()},
                      train_case_ids_sha256=sha(canonical(arrays["case_id"][split["train"]].tolist())),
                      validation_case_ids_sha256=sha(canonical(arrays["case_id"][split["validation"]].tolist())),
                      source_array_sha256=actual["spec"]["source"]["arrays_sha256"],
                      copy_semantics="CPU gather plus explicit CPU clone" if device == "cpu" else "CPU gather plus synchronized CPU-to-MPS transfer")
        input_names = ("history", "history_mask", "weather", "weather_mask", "state", "state_mask")
        def transfer(indices, *, labels=True):
            names = MODEL_ARRAYS if labels else input_names
            return {name: host[name][indices].to(device=device, copy=True) for name in names}
        train_started = time.monotonic()
        model.train()
        for step in range(UPDATES_PER_CASE):
            indices = batch_indices(split["train"], batch_size, step)
            batch, copy_seconds = timed(lambda: transfer(indices))
            def update():
                optimizer.zero_grad(set_to_none=True)
                prediction = model(*(batch[name] for name in input_names))
                loss = masked_mse(prediction, batch["target"], batch["target_mask"])
                require(bool(torch.isfinite(loss)), "Nonfinite loss")
                loss.backward()
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "Missing/nonfinite gradient")
                result["optimizer_updates_entered"] += 1
                optimizer.step()
                sync()
                result["optimizer_updates_confirmed"] += 1
                require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Nonfinite updated parameters")
            _, update_seconds = timed(update)
            sample = dict(step=step + 1, cases=batch_size, copy_seconds=copy_seconds, update_seconds=update_seconds)
            if step < WARMUP:
                result["warmup_samples"].append(sample)
                result["warmup_updates_confirmed"] += 1
            else:
                result["training_samples"].append(sample)
        stage["all_training_loops_including_guards"] = time.monotonic() - train_started
        model.eval()
        reference_inputs = reference_prediction = None
        validation_started = time.monotonic()
        with torch.inference_mode():
            for step in range(VALIDATION_SAMPLES + 1):
                indices = batch_indices(split["validation"], batch_size, step)
                batch, copy_seconds = timed(lambda: transfer(indices))
                def validate():
                    prediction = model(*(batch[name] for name in input_names))
                    loss = masked_mse(prediction, batch["target"], batch["target_mask"])
                    require(bool(torch.isfinite(loss)), "Nonfinite validation mechanics loss")
                _, seconds = timed(validate)
                sample = dict(cases=batch_size, copy_seconds=copy_seconds, forward_loss_seconds=seconds)
                if step == 0:
                    result["first_eval_sample"] = sample
                else:
                    result["validation_samples"].append(sample)
            stage["validation_loops_including_guards"] = time.monotonic() - validation_started
            inference_started = time.monotonic()
            for step in range(INFERENCE_SAMPLES):
                indices = batch_indices(split["validation"], batch_size, step)
                inputs, copy_seconds = timed(lambda: transfer(indices, labels=False))
                prediction, seconds = timed(lambda: model(*(inputs[name] for name in input_names)).detach().cpu())
                require(prediction.shape == (batch_size, 18) and bool(torch.isfinite(prediction).all()), "Inference output geometry/finiteness differs")
                result["inference_samples"].append(dict(cases=batch_size, copy_seconds=copy_seconds, forward_to_cpu_seconds=seconds))
                reference_inputs, reference_prediction = inputs, prediction
            stage["inference_loops_including_guards"] = time.monotonic() - inference_started
        checkpoint = run_dir / (args.case + ".private-checkpoint.pt")
        def write_checkpoint():
            value = dict(model=_cpu_tree(model.state_dict(), torch), optimizer=_cpu_tree(optimizer.state_dict(), torch),
                         cpu_rng=torch.get_rng_state(), device_rng=torch.mps.get_rng_state().cpu() if device == "mps" else None,
                         plan_sha256=actual["plan_sha256"], fixture_sha256=receipt["npz"]["sha256"], optimizer_updates=result["optimizer_updates_confirmed"])
            with checkpoint.open("xb") as stream:
                torch.save(value, stream)
                stream.flush()
                os.fsync(stream.fileno())
        _, stage["checkpoint_write"] = timed(write_checkpoint)
        checkpoint_sha = sha(_local_bytes(checkpoint, 32 * 1024**2))
        def restore_checkpoint():
            raw = _local_bytes(checkpoint, 32 * 1024**2)
            require(sha(raw) == checkpoint_sha, "Checkpoint identity differs")
            saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            require(saved["plan_sha256"] == actual["plan_sha256"] and saved["fixture_sha256"] == receipt["npz"]["sha256"], "Checkpoint provenance differs")
            restored, restored_optimizer = setup()
            restored.load_state_dict(saved["model"])
            restored_optimizer.load_state_dict(saved["optimizer"])
            torch.set_rng_state(saved["cpu_rng"])
            if device == "mps":
                torch.mps.set_rng_state(saved["device_rng"])
            return restored.eval(), restored_optimizer
        (restored, restored_optimizer), stage["checkpoint_read_restore"] = timed(restore_checkpoint)
        with torch.inference_mode():
            replay, stage["checkpoint_forward_verification"] = timed(lambda: restored(*(reference_inputs[name] for name in input_names)).detach().cpu())
        require(torch.equal(replay, reference_prediction), "Checkpoint forward replay differs")
        result.update(status="PASS", update_count_exact=True, checkpoint_forward_exact=True,
                      checkpoint=dict(bytes=checkpoint.stat().st_size, sha256=checkpoint_sha),
                      worker_observed_peaks=peaks, warmup_sample_visits=WARMUP * batch_size, timed_training_sample_visits=SAMPLES * batch_size,
                      training_unique_cases_visited=len({i for step in range(UPDATES_PER_CASE) for i in batch_indices(split["train"], batch_size, step)}),
                      validation_sample_visits=VALIDATION_SAMPLES * batch_size, inference_sample_visits=INFERENCE_SAMPLES * batch_size,
                      summary={"training_copy_seconds": summarize([s["copy_seconds"] for s in result["training_samples"]]),
                               "update_seconds": summarize([s["update_seconds"] for s in result["training_samples"]]),
                               "train_batch_total_seconds": summarize([s["copy_seconds"] + s["update_seconds"] for s in result["training_samples"]]),
                               "validation_batch_total_seconds": summarize([s["copy_seconds"] + s["forward_loss_seconds"] for s in result["validation_samples"]]),
                               "inference_batch_total_seconds": summarize([s["copy_seconds"] + s["forward_to_cpu_seconds"] for s in result["inference_samples"]])},
                      optimizer_updates_in_validation_inference=0, scaler_fits=0, protected_assessment_scores=0)
        result["examples_per_second_at_median_train_batch"] = batch_size / result["summary"]["train_batch_total_seconds"]["median"]
    except Exception as error:
        result.update(status="FAILED", error_type=type(error).__name__, error=portable(str(error))[:2000])
    finally:
        stopped.set()
        if watcher is not None:
            watcher.join(timeout=1)
            if watcher.is_alive() or monitor_errors:
                result.update(status="FAILED", error="Resource monitor did not complete", update_count_exact=False)
        result["worker_wall_seconds"] = time.monotonic() - started
        atomic_json(result_path, portable(result))
        atomic_json(status_path, {key: result[key] for key in ("optimizer_updates_entered", "optimizer_updates_confirmed")})


def supervise(case, run_dir, deadline):
    import psutil
    remaining = deadline - time.monotonic()
    require(remaining > 0, "Campaign wall allowance exhausted")
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", PYTORCH_ENABLE_MPS_FALLBACK="0")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--case", case["id"],
               "--run-directory", str(run_dir), "--wall-limit", str(remaining)]
    child, peak, last, reason, interruption = None, 0, {}, None, None
    started = time.monotonic()
    status_path = run_dir / (case["id"] + ".status.json")
    try:
        with (run_dir / (case["id"] + ".log")).open("x") as log:
            child = subprocess.Popen(command, env=environment, cwd=ROOT, stdout=log, stderr=log)
            process, supervisor = psutil.Process(child.pid), psutil.Process()
            while child.poll() is None:
                try:
                    rss = process.memory_info().rss + supervisor.memory_info().rss + sum(p.memory_info().rss for p in process.children(recursive=True))
                    if status_path.exists():
                        last = json.loads(status_path.read_text())
                    combined = rss + last.get("mps_driver_bytes", 0)
                    peak = max(peak, combined)
                    if combined > MAX_BYTES or time.monotonic() >= deadline:
                        reason = "Combined memory or campaign wall limit reached"
                        child.kill()
                        break
                except psutil.NoSuchProcess:
                    break
                time.sleep(.05)
            child.wait(timeout=5)
    except BaseException as error:
        # The case remains in the campaign receipt even if supervision itself
        # is interrupted. Its full allowance remains charged.
        interruption = error
        reason = "Supervisor interrupted: " + type(error).__name__
    finally:
        try:
            if child is not None and child.poll() is None:
                child.kill()
                child.wait(timeout=5)
        except BaseException as error:
            interruption = interruption or error
            reason = reason or "Supervisor could not confirm child reaping: " + type(error).__name__
    # Refresh after the child is reaped: a final durable snapshot may contain
    # more known updates than the last monitor poll. Never claim exactness on
    # an interrupted/failed supervisor path.
    try:
        if status_path.exists():
            last = json.loads(_local_bytes(status_path, 64 * 1024))
    except (OSError, ValueError):
        reason = reason or "Final worker status could not be read"
    result_path = run_dir / (case["id"] + ".json")
    try:
        result = json.loads(_local_bytes(result_path, 256 * 1024)) if result_path.exists() else None
    except (OSError, ValueError):
        result = None
        reason = reason or "Worker result could not be read"
    if result is None:
        result = dict(**case, status="FAILED", optimizer_updates_precharged=UPDATES_PER_CASE,
                      optimizer_updates_confirmed=0, optimizer_updates_entered=0,
                      update_count_exact=False, error="Worker result absent")
    for counter in ("optimizer_updates_confirmed", "optimizer_updates_entered"):
        known = [value for value in (last.get(counter, 0), result.get(counter, 0))
                 if type(value) is int and 0 <= value <= UPDATES_PER_CASE]
        result[counter] = max(known, default=0)
    result.update(supervised_wall_seconds=time.monotonic() - started, combined_observed_peak_bytes=peak)
    exit_code = child.returncode if child is not None else None
    if reason or exit_code:
        result.update(status="FAILED", stop_reason=reason, exit_code=exit_code, update_count_exact=False)
    if interruption is not None:
        result.update(supervisor_interrupted=True, supervisor_error_type=type(interruption).__name__)
    return result


def project_runtime(measurement, *, train_cases, validation_cases, epochs, trials, seeds, count_audit_sha256, count_qualification):
    """Use audited counts and observed common-batch rates; no cloud conversion."""
    require(measurement.get("status") == "PASS" and measurement.get("update_count_exact")
            and measurement.get("checkpoint_forward_exact"), "Completed exact timing receipt required")
    require(is_digest(count_audit_sha256) and isinstance(count_qualification, str) and count_qualification,
            "Count audit identity and its coverage qualification are required")
    for value in (train_cases, validation_cases, epochs, trials, seeds):
        require(type(value) is int and value > 0, "Positive explicit scenario counts required")
    batch = measurement["batch_size"]
    require(batch in BATCHES and len(measurement["training_samples"]) == SAMPLES
            and len(measurement["warmup_samples"]) == WARMUP
            and len(measurement["validation_samples"]) == VALIDATION_SAMPLES
            and all(sample["cases"] == batch for key in ("training_samples", "warmup_samples", "validation_samples") for sample in measurement[key]),
            "Exact measured sample counts and common batch geometry required")
    train = statistics.mean(s["copy_seconds"] + s["update_seconds"] for s in measurement["training_samples"])
    validation = statistics.mean(s["copy_seconds"] + s["forward_loss_seconds"] for s in measurement["validation_samples"])
    first_use = max(0., sum(s["copy_seconds"] + s["update_seconds"] for s in measurement["warmup_samples"]) - WARMUP * train)
    first_eval = measurement["first_eval_sample"]
    first_use += max(0., first_eval["copy_seconds"] + first_eval["forward_loss_seconds"] - validation)
    stage = measurement["stage_seconds"]
    startup = sum(stage[name] for name in ("imports", "fixture_read_authentication", "fixed_unit_mapping_tensorization", "model_optimizer_setup"))
    per_epoch = math.ceil(train_cases / batch) * train + math.ceil(validation_cases / batch) * validation + stage["checkpoint_write"]
    seconds = (startup + first_use + epochs * per_epoch) * trials * seeds
    return dict(status="CONDITIONAL_LOCAL_RUNTIME_SCENARIO", architecture=measurement["architecture"], device=measurement["device"],
                batch_size=batch, train_cases=train_cases, validation_cases=validation_cases, epochs=epochs, trials=trials, seeds=seeds,
                count_audit_sha256=count_audit_sha256, count_qualification=count_qualification,
                per_epoch_seconds=per_epoch, first_use_excess_seconds=first_use, total_hours=seconds / 3600,
                exclusions=["Full-corpus I/O and cache materialization", "Convergence", "Cloud runtime and price", "Fault/restart overhead",
                            "Difference between month-sized and full-corpus startup/cache read"],
                assumptions=["Observed batch rates remain stable", "Validation and checkpoint write each epoch", "Partial final batches charged as full batches"])


def run_campaign(*, receipt_path, fixture_path, run_directory, report_path, approved_plan_sha256):
    plan_value = plan(receipt_path)
    require(approved_plan_sha256 == plan_value["plan_sha256"], "Exact reviewed plan digest required")
    run_dir, report_path = Path(run_directory).resolve(), Path(report_path).absolute()
    require(any(run_dir.is_relative_to((ROOT / name).resolve()) for name in ("runs", ".local")), "Private run directory must be ignored")
    require(not run_dir.exists() and not report_path.exists() and not report_path.is_symlink(), "Run directory and report must be new")
    # Complete numerical admission before charging any optimizer allowance.
    load_fixture(fixture_path, json.loads(_local_bytes(receipt_path, 2 * 1024**2)))
    ledger = precharge(plan_value, run_dir)
    journal = RunJournal(run_dir, plan_value, max_wall_seconds=MAX_WALL_SECONDS, max_rss_bytes=MAX_BYTES)
    sync_directory(run_dir.parent)
    atomic_json(run_dir / "private-request.json", dict(plan=plan_value, receipt_path=str(Path(receipt_path).resolve()), fixture_path=str(Path(fixture_path).resolve())))
    started, deadline = time.monotonic(), time.monotonic() + MAX_WALL_SECONDS - 1
    report = dict(schema="hydra-real-batch-benchmark/v1", status="RUNNING", qualification=QUALIFICATION,
                  created_at_utc=datetime.now(timezone.utc).isoformat(), plan=plan_value, allowance=ledger,
                  cases=[], optimizer_updates_precharged=MAX_UPDATES, cloud_jobs=0, network_requests=0,
                  scientific_training=False, source_preprocessing_timed=False, os_page_cache_flushed=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(report_path, report)
    try:
        for case in matrix():
            with journal.stage("case_" + case["id"].lower().replace("-", "_")):
                result = supervise(case, run_dir, deadline)
                report["cases"].append(result)
                journal.event("case_finished", case_id=case["id"], status=result["status"], optimizer_updates_confirmed=result.get("optimizer_updates_confirmed", 0))
                atomic_json(report_path, portable(report))
            print(json.dumps({key: result.get(key) for key in ("id", "status", "optimizer_updates_confirmed", "examples_per_second_at_median_train_batch")}), flush=True)
            require(result["status"] == "PASS", "Campaign stopped after failed case; no automatic retry")
        report["status"] = "PASS"
        journal.finish()
    except BaseException as error:
        report.update(status="PARTIAL_OR_FAILED", error_type=type(error).__name__, error=portable(str(error))[:2000])
        if journal.status == "RUNNING":
            journal.finish("STOPPED", error_type=type(error).__name__)
    report.update(optimizer_updates_confirmed=sum(c.get("optimizer_updates_confirmed", 0) for c in report["cases"]),
                  optimizer_updates_entered=sum(c.get("optimizer_updates_entered", 0) for c in report["cases"]),
                  warmup_updates_confirmed=sum(c.get("warmup_updates_confirmed", 0) for c in report["cases"]),
                  measured_updates_confirmed=sum(len(c.get("training_samples", [])) for c in report["cases"]),
                  elapsed_seconds=time.monotonic() - started,
                  observed_combined_peak_bytes=max((c.get("combined_observed_peak_bytes", 0) for c in report["cases"]), default=0),
                  failed_and_unstarted_allowances_retained=True, journal_status=journal.status,
                  unstarted_case_count=len(matrix()) - len(report["cases"]),
                  all_reported_update_counts_exact=all(c.get("update_count_exact", False) for c in report["cases"]))
    atomic_json(report_path, portable(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=[case["id"] for case in matrix()], help=argparse.SUPPRESS)
    parser.add_argument("--wall-limit", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.receipt is None:
        parser.error("A sealed fixture receipt is required")
    if not args.execute:
        print(json.dumps(plan(args.receipt), indent=2, sort_keys=True, allow_nan=False))
        return
    if any(value is None for value in (args.fixture, args.run_directory, args.report, args.approved_plan_sha256)):
        parser.error("Execution requires fixture, new private run directory/report, and exact approved plan digest")
    result = run_campaign(receipt_path=args.receipt, fixture_path=args.fixture, run_directory=args.run_directory,
                          report_path=args.report, approved_plan_sha256=args.approved_plan_sha256)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
