"""Bounded local pipeline timing; engineering measurements are not model skill.

The default API only plans. An enabled attempt requires a fresh ignored run
directory and a digest binding inputs, implementation, and an explicit update
allowance. No network, fitted scaler, protected assessment, or cloud path exists.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
import zipfile

from .contract import ARCHITECTURES, GAUGE_IDS, HORIZONS_HOURS, require
from .engineering import FEATURES, UNIT_MAPPING, _cpu_tree, _local_bytes
from .telemetry import RunJournal, atomic_json, canonical

MAX_BYTES = 4 * 1024**3
CACHE_LIMIT = 128 * 1024**2
QUALIFICATION = "LOCAL_DEVELOPMENT_ENGINEERING_TIMING_ONLY_NO_FORECAST_SKILL"
MODEL_ARRAYS = ("history", "history_mask", "weather", "weather_mask", "state", "state_mask", "target", "target_mask")
COORDINATES = ("site_id", "origin_time", "history_time", "target_time", "case_id", "source_cycle")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def utc_epoch(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(stamp.tzinfo is not None and stamp.utcoffset() == timedelta(0)
            and stamp.microsecond == 0 and stamp.minute == stamp.second == 0,
            "An explicit whole-hour UTC timestamp is required")
    return int(stamp.timestamp())


@dataclass(frozen=True)
class PipelineTimingSpec:
    architecture: str = "LSTM"
    device: str = "cpu"
    gauge_id: str = "03161000"
    history_hours: int = 168
    batch_size: int = 32
    max_optimizer_steps: int = 0
    engineering_epochs: int = 1
    validation_origin_utc: str | None = None
    max_wall_seconds: int = 300
    max_combined_bytes: int = MAX_BYTES
    seed: int = 1729

    def __post_init__(self):
        require(self.architecture in ARCHITECTURES and self.device in ("cpu", "mps"), "Only prepared local CPU/MPS candidates allowed")
        require(self.gauge_id in GAUGE_IDS and self.history_hours in (24, 168), "Confirmed gauge and L24/L168 required")
        for value, bound, name in ((self.batch_size, 64, "batch size"), (self.max_wall_seconds, 900, "wall seconds"),
                                   (self.max_combined_bytes, MAX_BYTES, "combined bytes")):
            require(type(value) is int and 0 < value <= bound, "Invalid bounded " + name)
        require(type(self.max_optimizer_steps) is int and 0 <= self.max_optimizer_steps <= 64,
                "Explicit allowance from zero to 64 updates required")
        require(type(self.engineering_epochs) is int and 1 <= self.engineering_epochs <= 3, "One to three engineering timing epochs required")
        require(type(self.seed) is int and 0 <= self.seed < 2**32, "Uint32 seed required")
        if self.validation_origin_utc is not None:
            require(utc_epoch(self.validation_origin_utc) < utc_epoch("2024-07-31T00:00:00Z"), "Protected boundary cannot be validation")


def split_cases(site_ids, origins, targets, *, gauge_id, validation_origin_utc):
    """Select one gauge; purge training windows reaching the validation origin.

    This is a mechanical chronological split for timing, not a selected study
    split. Origin C's targets are C+1..C+18. Requiring the last train target
    <= validation C makes the first validation target strictly later.
    """
    require(gauge_id in GAUGE_IDS, "Unknown gauge")
    require(len(site_ids) == len(origins) == len(targets) and len(origins) > 0, "Case coordinate lengths differ")
    require(validation_origin_utc is not None, "Explicit chronological validation origin required")
    boundary = utc_epoch(validation_origin_utc)
    protected = utc_epoch("2024-07-31T00:00:00Z")
    train, validation, purged = [], [], []
    seen = set()
    for index, (site, origin, label_times) in enumerate(zip(site_ids, origins, targets)):
        require(type(site) is int and site in range(3) and type(origin) is int, "Integral case coordinates required")
        require((site, origin) not in seen, "Duplicate gauge/origin case")
        seen.add((site, origin))
        expected = [origin + 3600 * hour for hour in HORIZONS_HOURS]
        require(list(label_times) == expected and expected[-1] < protected, "Labels must be aligned and entirely before protected assessment")
        if site != GAUGE_IDS.index(gauge_id):
            continue
        if origin >= boundary:
            validation.append(index)
        elif expected[-1] <= boundary:
            train.append(index)
        else:
            purged.append(index)
    require(train and validation, "Timing requires nonempty chronological target-disjoint training and validation cases")
    return dict(train=train, validation=validation, purged=purged, boundary_epoch=boundary)


def sample_summary(values):
    require(values and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values),
            "Positive finite timing samples required")
    return dict(count=len(values), total=sum(values), mean=statistics.mean(values), median=statistics.median(values),
                empirical_p95=sorted(values)[math.ceil(.95 * len(values)) - 1], percentile_method="nearest_rank")


def environment_metadata(torch):
    """Portable execution geometry; never emit hostnames or filesystem paths."""
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "pyarrow", "numcodecs", "transformers")}
    return dict(python_version=platform.python_version(), operating_system=platform.system(),
                os_release=platform.release(), machine_architecture=platform.machine(),
                package_versions=versions, torch_intraop_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads(), model_dtype="float32", model_width=64,
                mps_available=bool(torch.backends.mps.is_available()), device_fallback_enabled=False)


def estimate_workflow(measurement, *, train_cases, validation_cases, epochs, trials, seeds,
                      gauges=3, validation_every=1, checkpoint_every=1, hourly_rate=None):
    """Conditional local projection at the observed batch geometry, not a quote.

    A small last batch is kept at its measured cost rather than scaled to an
    unseen larger batch. Preprocessing is reported separately because disk,
    cycle-count, partition, and cloud-read scaling have not been measured.
    """
    for value in (train_cases, validation_cases, epochs, trials, seeds, gauges, validation_every, checkpoint_every):
        require(type(value) is int and value > 0, "Positive explicit scenario counts required")
    require(measurement.get("status") == "COMPLETED_ENGINEERING_TIMING", "Completed measurement required")
    train, validation = measurement["training_batches"], measurement["validation_batches"]
    require(train and validation, "Measured update and validation samples required")
    train_sizes, validation_sizes = {x["cases"] for x in train}, {x["cases"] for x in validation}
    require(len(train_sizes) == len(validation_sizes) == 1, "Use a single measured batch size per stage for projections")
    train_size, validation_size = next(iter(train_sizes)), next(iter(validation_sizes))
    train_cost = statistics.mean(x["copy_seconds"] + x["update_seconds"] for x in train)
    validation_cost = statistics.mean(x["copy_seconds"] + x["forward_loss_seconds"] for x in validation)
    for value in (train_cost, validation_cost):
        require(math.isfinite(value) and value > 0, "Positive finite batch costs required")
    training_seconds = math.ceil(train_cases / train_size) * train_cost * epochs
    validation_seconds = math.ceil(validation_cases / validation_size) * validation_cost * math.ceil(epochs / validation_every)
    stages = measurement["stage_seconds"]
    checkpoint_seconds = stages["checkpoint_write"] * math.ceil(epochs / checkpoint_every)
    startup_seconds = sum(stages[name] for name in ("imports", "cache_read_verified", "tensorization", "model_setup"))
    fits = trials * seeds * gauges
    per_fit = startup_seconds + training_seconds + validation_seconds + checkpoint_seconds
    seconds = fits * per_fit
    result = dict(status="CONDITIONAL_LOCAL_RUNTIME_PROJECTION_NOT_CONVERGENCE_ESTIMATE",
                  architecture=measurement["spec"]["architecture"], device=measurement["spec"]["device"],
                  scenario=dict(train_cases_per_gauge=train_cases, validation_cases_per_gauge=validation_cases,
                                epochs=epochs, trials=trials, seeds=seeds, gauges=gauges,
                                validation_every=validation_every, checkpoint_every=checkpoint_every),
                  measured_train_batch_size=train_size, measured_validation_batch_size=validation_size,
                  per_fit_seconds=per_fit, total_fits=fits, total_seconds=seconds, total_hours=seconds / 3600,
                  per_fit_components_seconds=dict(startup=startup_seconds, training=training_seconds,
                                                  validation=validation_seconds, checkpoint_write=checkpoint_seconds),
                  preprocessing_seconds_observed=stages["preprocess_first_pass"],
                  exclusions=["Full-corpus source I/O and preprocessing", "Cloud hardware and transfer costs",
                              "Convergence and selected epoch count", "Hyperparameter search/selection overhead",
                              "Fault recovery and scheduling", "Unmeasured larger batch throughput"],
                  assumptions=["Observed local batch costs remain stable", "Runs execute serially",
                               "Configured epochs/trials/seeds/gauges are scenario inputs, not scientific choices",
                               "Every incomplete final batch charged as one observed batch"])
    if hourly_rate is not None:
        require(type(hourly_rate) in (int, float) and math.isfinite(hourly_rate) and hourly_rate >= 0,
                "Hourly rate must be an explicit finite nonnegative scenario input")
        result["user_supplied_hourly_rate"] = hourly_rate
        result["conditional_compute_cost"] = hourly_rate * seconds / 3600
        result["cost_qualification"] = "Arithmetic only; observed local speed cannot be assumed for unmeasured cloud hardware"
    return result


def _implementation():
    package = Path(__file__).resolve().parent
    return {name: sha(_local_bytes(package / name, 256 * 1024)) for name in
            ("pipeline_timing.py", "multicycle.py", "local_data.py", "lineage.py", "contract.py", "reference.py", "telemetry.py",
             "engineering.py", "models/directflow.py", "models/mamba2_math.py")}


def load_builder_request(path):
    """Match the preprocessing CLI's request-relative local path semantics."""
    raw = _local_bytes(path, 128 * 1024)
    request = json.loads(raw)
    required = {"archive_specs", "qualified_index_path", "weights_path", "usgs_manifest_path", "usgs_paths", "origins_utc", "history_hours"}
    require(isinstance(request, dict) and set(request) == required, "Exact documented builder request keys required")
    require(isinstance(request["archive_specs"], list) and request["archive_specs"], "Explicit archive specifications required")
    require(isinstance(request["usgs_paths"], dict) and request["usgs_paths"], "Monthly USGS local path map required")
    def local(value):
        require(isinstance(value, str) and bool(value), "Nonempty local source path required")
        candidate = Path(value)
        return str(candidate if candidate.is_absolute() else Path(path).absolute().parent / candidate)
    for key in ("qualified_index_path", "weights_path", "usgs_manifest_path"):
        request[key] = local(request[key])
    request["usgs_paths"] = {month: local(value) for month, value in request["usgs_paths"].items()}
    for source in request["archive_specs"]:
        require(isinstance(source, dict) and set(source) in ({"cycle_id", "archive_path"}, {"cycle_id", "archive_path", "commit_path"}), "Exact archive specification keys required")
        for key in ("archive_path", "commit_path"):
            if key in source:
                source[key] = local(source[key])
    return raw, request


def plan_pipeline(request_path, spec):
    require(isinstance(spec, PipelineTimingSpec), "Validated timing specification required")
    raw, request = load_builder_request(request_path)
    require(request.get("history_hours", 24) == spec.history_hours, "Builder and timing history lengths differ")
    require(spec.validation_origin_utc is not None, "Explicit chronological validation boundary required")
    require(len(request["archive_specs"]) >= 2, "Actual multi-cycle inputs required")
    origins = [utc_epoch(value) for value in request["origins_utc"]]
    require(len(set(origins)) == len(origins) and len(origins) >= 2, "Unique multiple origins required")
    require(max(origins) + 18 * 3600 < utc_epoch("2024-07-31T00:00:00Z"), "Protected development labels are prohibited")
    identity = dict(spec=asdict(spec), private_request_sha256=sha(raw), resolved_request_sha256=sha(canonical(request)), implementation_sha256=_implementation(),
                    unit_mapping=UNIT_MAPPING, qualification=QUALIFICATION)
    return dict(schema="hydra-pipeline-timing-plan/v1", status="PLANNED_NOT_EXECUTED", **identity,
                plan_sha256=sha(canonical(identity)), planned_optimizer_updates_maximum=spec.max_optimizer_steps,
                requested_origin_count=len(origins), requested_archive_count=len(request["archive_specs"]),
                source_access="LOCAL_RESIDENT_ONLY", cloud_jobs=0, scientific_campaign=False,
                cache_semantics="Fresh-process first pass and same-process warm pass; OS disk cache is not flushed")


def _validate_arrays(arrays, length):
    import numpy as np
    n = len(arrays["origin_time"])
    require(0 < n <= 4096, "Bounded positive case count required")
    shapes = dict(history=(n, length, 2), weather=(n, 18, 4), state=(n, 2), target=(n, 18))
    for name, shape in shapes.items():
        values, mask = arrays[name], arrays[name + "_mask"]
        require(values.shape == mask.shape == shape and values.dtype.kind == "f" and mask.dtype == np.dtype("?"),
                "Model array shape/type differs: " + name)
        require(np.isfinite(values[mask]).all(), "Admitted model values must be finite")
    require(arrays["site_id"].shape == arrays["origin_time"].shape == arrays["case_id"].shape == (n,), "Case identity dimensions differ")
    require(arrays["history_time"].shape == (n, length) and arrays["target_time"].shape == (n, 18), "Clock dimensions differ")
    require(np.array_equal(arrays["history_time"], arrays["origin_time"][:, None] - np.arange(length + 1, 1, -1)[None, :] * 3600),
            "History must end at C minus two hours inclusive")
    require(len(set(arrays["case_id"].tolist())) == n, "Case IDs must be unique")
    return n


def _array_digest(arrays):
    result = hashlib.sha256()
    for name in sorted(arrays):
        value = arrays[name]
        result.update(canonical(dict(name=name, dtype=value.dtype.str, shape=value.shape)))
        result.update(value.tobytes(order="C"))
    return result.hexdigest()


def _map_arrays(arrays):
    import numpy as np
    result = {}
    for name in ("history", "weather", "state", "target"):
        source, mask = arrays[name], arrays[name + "_mask"]
        mapped = np.zeros(source.shape, dtype=np.float32)
        if name == "target":
            mapped[mask] = source[mask] / UNIT_MAPPING[name][0][1]
        else:
            for column, (offset, divisor) in enumerate(UNIT_MAPPING[name]):
                admitted = mask[..., column]
                mapped[..., column][admitted] = (source[..., column][admitted] - offset) / divisor
        require(np.isfinite(mapped).all(), "Fixed unit mapping overflowed")
        result[name], result[name + "_mask"] = mapped, mask.copy()
    return result


def _worker(request_path):
    worker_started = time.monotonic()
    def deny_network(event, unused):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            raise RuntimeError("Network prohibited during local pipeline timing")
    sys.addaudithook(deny_network)
    private = json.loads(_local_bytes(request_path, 256 * 1024))
    plan = private["plan"]
    spec = PipelineTimingSpec(**plan["spec"])
    actual = plan_pipeline(private["builder_request_path"], spec)
    require(actual["plan_sha256"] == plan["plan_sha256"], "Plan/source/implementation changed")
    directory = Path(request_path).parent
    with (directory / "worker-started.json").open("x") as stream:
        json.dump(dict(pid=os.getpid(), precharged_updates=spec.max_optimizer_steps), stream)
    stages = {}
    started = time.monotonic()
    import numpy as np
    import psutil
    import torch
    from .multicycle import build_multicycle_development
    from .models.directflow import DirectFlowModel, ModelConfig, masked_mse
    stages["imports"] = time.monotonic() - started
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(spec.seed)
    if spec.device == "mps":
        require(torch.backends.mps.is_available(), "MPS unavailable; no CPU fallback")
        torch.mps.set_per_process_memory_fraction(min(1., 2 * 1024**3 / torch.mps.recommended_max_memory()))
    process = psutil.Process()
    status = dict(optimizer_updates_confirmed=0, optimizer_updates_entered=0, mps_driver_bytes=0)
    stopped, monitor_errors = threading.Event(), []

    def monitor():
        try:
            while not stopped.is_set():
                status["mps_driver_bytes"] = torch.mps.driver_allocated_memory() if spec.device == "mps" else 0
                atomic_json(directory / "worker-status.json", dict(status))
                stopped.wait(.05)
        except Exception as exc:
            monitor_errors.append(type(exc).__name__)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()

    def sync():
        if spec.device == "mps":
            torch.mps.synchronize()

    def guard():
        require(not monitor_errors, "Resource monitor failed")
        require(time.monotonic() - worker_started < spec.max_wall_seconds, "Worker wall allowance reached")
        driver = torch.mps.driver_allocated_memory() if spec.device == "mps" else 0
        require(process.memory_info().rss + driver <= spec.max_combined_bytes, "Worker memory allowance reached")

    def timed(name, callback):
        guard()
        sync()
        tick = time.monotonic()
        value = callback()
        sync()
        stages[name] = time.monotonic() - tick
        guard()
        return value

    try:
        _, builder = load_builder_request(private["builder_request_path"])
        arrays, receipt = timed("preprocess_first_pass", lambda: build_multicycle_development(**builder))
        _validate_arrays(arrays, spec.history_hours)
        source_array_digest = _array_digest(arrays)
        warm, warm_receipt = timed("preprocess_warm_pass", lambda: build_multicycle_development(**builder))
        require(_array_digest(warm) == source_array_digest, "Warm preprocessing changed arrays or provenance")
        del warm
        selected = {name: arrays[name] for name in MODEL_ARRAYS + COORDINATES}
        selected_digest = _array_digest(selected)
        cache_path = directory / "private-feature-cache.npz"

        def write_cache():
            with cache_path.open("xb") as stream:
                np.savez_compressed(stream, **selected)
                stream.flush()
                os.fsync(stream.fileno())
        timed("cache_write", write_cache)
        identity = dict(bytes=cache_path.stat().st_size, sha256=sha(_local_bytes(cache_path, CACHE_LIMIT)))

        def read_cache():
            raw = _local_bytes(cache_path, CACHE_LIMIT)
            require(len(raw) == identity["bytes"] and sha(raw) == identity["sha256"], "Feature cache changed")
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                members = archive.infolist()
                require(len(members) == len(selected) and {m.filename for m in members} == {name + ".npy" for name in selected}
                        and sum(m.file_size for m in members) <= CACHE_LIMIT, "Unexpected or unbounded feature cache inventory")
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                loaded = {name: archive[name] for name in selected}
            _validate_arrays(loaded, spec.history_hours)
            require(_array_digest(loaded) == selected_digest, "Feature cache array identity differs")
            return loaded
        loaded = timed("cache_read_verified", read_cache)
        split = split_cases(loaded["site_id"].tolist(), loaded["origin_time"].tolist(), loaded["target_time"].tolist(),
                            gauge_id=spec.gauge_id, validation_origin_utc=spec.validation_origin_utc)
        train_indices, val_indices = split["train"], split["validation"]
        steps = math.ceil(len(train_indices) / spec.batch_size)
        require(0 < steps * spec.engineering_epochs <= spec.max_optimizer_steps, "Complete engineering epoch exceeds precharged update budget")
        host = timed("tensorization", lambda: {name: torch.from_numpy(value) for name, value in _map_arrays(loaded).items()})
        config = ModelConfig(spec.architecture, FEATURES["history"], FEATURES["weather"], FEATURES["state"],
                             spec.history_hours, HORIZONS_HOURS, width=64, dropout=.1)
        def setup():
            model = DirectFlowModel(config).to(spec.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
            return model, optimizer
        model, optimizer = timed("model_setup", setup)

        def transfer(indices):
            return {name: value[indices].to(spec.device, copy=True) for name, value in host.items()}
        inputs = ("history", "history_mask", "weather", "weather_mask", "state", "state_mask")
        training, validation = [], []
        epoch_samples = []
        reference_batch = reference_prediction = None
        for epoch in range(1, spec.engineering_epochs + 1):
            epoch_started = time.monotonic()
            model.train()
            for offset in range(0, len(train_indices), spec.batch_size):
                indices = train_indices[offset:offset + spec.batch_size]
                batch = timed("batch_copy", lambda: transfer(indices))
                def update():
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(*(batch[name] for name in inputs))
                    loss = masked_mse(prediction, batch["target"], batch["target_mask"])
                    require(bool(torch.isfinite(loss)), "Nonfinite training mechanics loss")
                    loss.backward()
                    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "Invalid gradient")
                    status["optimizer_updates_entered"] += 1
                    optimizer.step()
                    sync()
                    status["optimizer_updates_confirmed"] += 1
                    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Invalid parameter")
                timed("batch_update", update)
                training.append(dict(epoch=epoch, cases=len(indices), copy_seconds=stages["batch_copy"], update_seconds=stages["batch_update"]))
            train_epoch_seconds = time.monotonic() - epoch_started
            model.eval()
            validation_started = time.monotonic()
            with torch.no_grad():
                for offset in range(0, len(val_indices), spec.batch_size):
                    indices = val_indices[offset:offset + spec.batch_size]
                    batch = timed("batch_copy", lambda: transfer(indices))
                    def validate():
                        prediction = model(*(batch[name] for name in inputs))
                        loss = masked_mse(prediction, batch["target"], batch["target_mask"])
                        require(bool(torch.isfinite(loss)), "Nonfinite validation mechanics loss")
                        return prediction.detach().cpu()
                    prediction = timed("batch_validation", validate)
                    validation.append(dict(epoch=epoch, cases=len(indices), copy_seconds=stages["batch_copy"], forward_loss_seconds=stages["batch_validation"]))
                    reference_batch, reference_prediction = batch, prediction
            epoch_samples.append(dict(epoch=epoch, training_seconds=train_epoch_seconds, validation_seconds=time.monotonic() - validation_started))
        checkpoint_path = directory / "private-checkpoint.pt"
        def save_checkpoint():
            value = dict(model=_cpu_tree(model.state_dict(), torch), optimizer=_cpu_tree(optimizer.state_dict(), torch),
                         cpu_rng=torch.get_rng_state(), device_rng=torch.mps.get_rng_state().cpu() if spec.device == "mps" else None,
                         source_array_digest=source_array_digest, plan_sha256=plan["plan_sha256"],
                         optimizer_updates=status["optimizer_updates_confirmed"])
            with checkpoint_path.open("xb") as stream:
                torch.save(value, stream)
                stream.flush()
                os.fsync(stream.fileno())
        timed("checkpoint_write", save_checkpoint)
        checkpoint_identity = dict(bytes=checkpoint_path.stat().st_size, sha256=sha(_local_bytes(checkpoint_path, CACHE_LIMIT)))
        def reload_checkpoint():
            raw = _local_bytes(checkpoint_path, CACHE_LIMIT)
            require(sha(raw) == checkpoint_identity["sha256"], "Checkpoint bytes changed")
            value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            require(value["plan_sha256"] == plan["plan_sha256"] and value["source_array_digest"] == source_array_digest,
                    "Checkpoint source/plan changed")
            restored, restored_optimizer = setup()
            restored.load_state_dict(value["model"])
            restored_optimizer.load_state_dict(value["optimizer"])
            torch.set_rng_state(value["cpu_rng"])
            if spec.device == "mps":
                torch.mps.set_rng_state(value["device_rng"])
            restored.eval()
            return restored, restored_optimizer
        restored, restored_optimizer = timed("checkpoint_read_restore", reload_checkpoint)
        with torch.no_grad():
            reloaded_prediction = timed("checkpoint_forward_verification", lambda: restored(*(reference_batch[name] for name in inputs)).detach().cpu())
        require(torch.equal(reloaded_prediction, reference_prediction), "Checkpoint forward replay differs")
        stages["worker_wall_seconds"] = time.monotonic() - worker_started
        for transient in ("batch_copy", "batch_update", "batch_validation"):
            stages.pop(transient, None)
        result = dict(schema="hydra-pipeline-timing-result/v1", status="COMPLETED_ENGINEERING_TIMING", qualification=QUALIFICATION,
                      spec=asdict(spec), plan_sha256=plan["plan_sha256"], implementation_sha256=plan["implementation_sha256"],
                      environment=environment_metadata(torch),
                      source_array_sha256=source_array_digest, source_receipt=receipt,
                      preprocessing_warm_stage_seconds=warm_receipt.get("stage_seconds", {}),
                      stage_seconds=stages, training_batches=training, validation_batches=validation, epoch_samples=epoch_samples,
                      training_cases=len(train_indices), validation_cases=len(val_indices), purged_cases=len(split["purged"]),
                      source_cases=len(loaded["origin_time"]), source_cycles=len(set(loaded["source_cycle"].tolist())),
                      optimizer_updates_precharged=spec.max_optimizer_steps, optimizer_updates_confirmed=status["optimizer_updates_confirmed"],
                      warmup_optimizer_updates=0, training_epochs=spec.engineering_epochs, validation_epochs=spec.engineering_epochs, checkpoint_forward_exact=True,
                      feature_cache=identity, checkpoint=checkpoint_identity, scaler_fits=0, protected_assessment_scores=0,
                      network_requests=0, cloud_jobs=0, forecast_skill_reported=False,
                      cache_semantics=plan["cache_semantics"],
                      timing_scope="Local source authentication through bounded engineering train/validation epochs and checkpoint replay; all timings include checks unless named otherwise")
        atomic_json(directory / "worker-result.json", result)
    finally:
        stopped.set()
        watcher.join(timeout=1)
        atomic_json(directory / "worker-status.json", status)


def run_pipeline_timing(*, request_path, spec, run_directory=None, execution_enabled=False, approved_plan_sha256=None):
    """Default plan; explicitly enabled execution consumes one fresh allowance."""
    require(type(execution_enabled) is bool, "Explicit boolean execution flag required")
    plan = plan_pipeline(request_path, spec)
    if not execution_enabled:
        return plan
    require(spec.max_optimizer_steps > 0 and approved_plan_sha256 == plan["plan_sha256"], "Positive update allowance and exact plan digest required")
    require(run_directory is not None, "New private run directory required")
    directory = Path(run_directory).absolute()
    root = Path(__file__).resolve().parents[2]
    require(any(directory.resolve().is_relative_to((root / name).resolve()) for name in ("runs", ".local")),
            "Timing artifacts must stay in this checkout's ignored runs/ or .local/ directory")
    require(not directory.exists() and not directory.is_symlink(), "Run directory must be new")
    import psutil
    journal = RunJournal(directory, plan, max_wall_seconds=spec.max_wall_seconds, max_rss_bytes=spec.max_combined_bytes)
    journal.event("optimizer_budget_precharged", updates=spec.max_optimizer_steps)
    worker_request = directory / "private-worker-request.json"
    atomic_json(worker_request, dict(plan=plan, builder_request_path=str(Path(request_path).resolve())))
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", PYTORCH_ENABLE_MPS_FALLBACK="0")
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + environment.get("PYTHONPATH", "")
    result = dict(schema="hydra-pipeline-timing-result/v1", status="FAILED", qualification=QUALIFICATION,
                  spec=asdict(spec), plan_sha256=plan["plan_sha256"], optimizer_updates_precharged=spec.max_optimizer_steps,
                  optimizer_updates_confirmed=0, optimizer_count_exact=False, cloud_jobs=0, network_requests=0)
    child, peak, started = None, 0, time.monotonic()
    try:
        with (directory / "worker.log").open("x") as log:
            child = subprocess.Popen([sys.executable, "-m", __name__, "--worker", str(worker_request)], env=environment, stdout=log, stderr=log)
            process, supervisor = psutil.Process(child.pid), psutil.Process()
            while child.poll() is None:
                try:
                    rss = supervisor.memory_info().rss + process.memory_info().rss + sum(p.memory_info().rss for p in process.children(recursive=True))
                    status_path = directory / "worker-status.json"
                    current = json.loads(status_path.read_text()) if status_path.exists() else {}
                    peak = max(peak, rss + current.get("mps_driver_bytes", 0))
                    require(peak <= spec.max_combined_bytes, "Combined supervisor/worker/MPS memory cap reached")
                    require(time.monotonic() - started < spec.max_wall_seconds, "Overall timing wall cap reached")
                except psutil.NoSuchProcess:
                    break
                time.sleep(.05)
            child.wait(timeout=5)
            require(child.returncode == 0, "Timing worker failed; inspect private worker.log")
        result.update(json.loads((directory / "worker-result.json").read_text()), optimizer_count_exact=True)
        for name in ("private-feature-cache.npz", "private-checkpoint.pt"):
            journal.artifact(directory / name)
        journal.metric("optimizer_updates_confirmed", result["optimizer_updates_confirmed"])
        journal.metric("combined_observed_peak_bytes", peak)
        journal.finish()
    except Exception as exc:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        result.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        if journal.status == "RUNNING":
            journal.finish("FAILED", error_type=type(exc).__name__)
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        status_path = directory / "worker-status.json"
        if not result["optimizer_count_exact"] and status_path.exists():
            result.update({key: value for key, value in json.loads(status_path.read_text()).items() if key.startswith("optimizer_")})
        result.update(combined_observed_peak_bytes=peak, supervised_wall_seconds=time.monotonic() - started, journal_status=journal.status,
                      artifact_created_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(directory / "summary.json", result)
    return result


if __name__ == "__main__":
    require(len(sys.argv) == 3 and sys.argv[1] == "--worker", "Use the plan-first API or scripts/benchmark_pipeline.py")
    _worker(sys.argv[2])
