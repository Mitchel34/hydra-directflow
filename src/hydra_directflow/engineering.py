"""Explicitly enabled, supervised training mechanics on one development case.

Importing this module or preparing a plan uses only the standard library. There
is no scientific-campaign or protected-assessment branch. Execution requires a
positive bounded step budget and the exact reviewed plan digest; it is disabled
by default. A repeated invocation is a new execution requiring its own decision.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import os
import stat
import subprocess
import sys
import threading
import time
import zipfile

from .contract import ARCHITECTURES, GAUGE_IDS, HORIZONS_HOURS
from .reference import digest, require
from .telemetry import RunJournal


ORIGIN = "2021-08-23T02:00:00Z"
CYCLE = "20210823_00Z"
FIXTURE_SHA256 = "0aad781c8bcb77a901e1f6c3e1a5d2efc1115d7819145ceeb1d410a4532548b8"
FIXTURE_BYTES = 30047
HRRR_SHA256 = "6f34151174311054173b03dc32bcb4080b9244537c24f9a45f7746c6be6cdf00"
USGS_SHA256 = "16f1dcbc063fed46a72ce286571c3b78fc2c199fce24e07aa76cdd03318c6d2b"
WEIGHTS_SHA256 = "9eaaf3b221360ed6faaa0376e953fd0c2d0356fb419df96daed9107404283a74"
FEATURES = {"history": ("discharge_cms", "age_hours"),
            "weather": ("precip_mm", "temp_k", "specific_humidity", "swe_kg_m2"),
            "state": ("soil_water_fraction", "soil_age_hours")}
WEATHER_GROUPS = ("precipitation", "temperature", "humidity", "snow_water_equivalent")
ORIGIN_EPOCH = int(datetime.fromisoformat(ORIGIN.replace("Z", "+00:00")).timestamp())
HISTORY_TIMES = tuple(ORIGIN_EPOCH - hour * 3600 for hour in range(25, 1, -1))
TARGET_TIMES = tuple(ORIGIN_EPOCH + hour * 3600 for hour in HORIZONS_HOURS)
UNIT_MAPPING = {
    "history": ((0., 100.), (0., 168.)),
    "weather": ((0., 10.), (273.15, 30.), (0., .01), (0., 100.)),
    "state": ((0., 1.), (0., 24.)),
    "target": ((0., 100.),),
}
QUALIFICATION = (
    "DEVELOPMENT_ENGINEERING_ONLY: repeated updates on one already observed case; "
    "fixed declared unit mapping, no fitted scalers, no forecast skill, "
    "no scientific campaign, no protected assessment, no deployment")


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _local_bytes(path, maximum):
    path = Path(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), "Only regular local files, never symlinks, are admitted")
    require(not getattr(info, "st_flags", 0) & 0x40000000, "Dataless files cannot be hydrated")
    require(0 < info.st_size <= maximum, "Local input exceeds its byte bound")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    require(len(data) == info.st_size, "Local input changed during bounded read")
    return data


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


@dataclass(frozen=True)
class ProbeSpec:
    gauge_id: str
    architecture: str = "LSTM"
    device: str = "cpu"
    max_optimizer_steps: int = 0
    max_wall_seconds: int = 300
    max_combined_bytes: int = 4 * 1024**3
    learning_rate: float = 0.0001
    seed: int = 17

    def __post_init__(self):
        require(self.gauge_id in GAUGE_IDS, "Explicit confirmed gauge required")
        require(self.architecture in ARCHITECTURES, "Unknown architecture")
        require(self.device in ("cpu", "mps"), "Only local CPU/MPS engineering is implemented")
        require(type(self.max_optimizer_steps) is int and 0 <= self.max_optimizer_steps <= 16,
                "Engineering update budget must be an integer from zero to 16")
        require(type(self.max_wall_seconds) is int and 0 < self.max_wall_seconds <= 300,
                "Engineering wall limit must be at most 300 seconds")
        require(type(self.max_combined_bytes) is int and 0 < self.max_combined_bytes <= 4 * 1024**3,
                "Combined supervisor/worker RSS plus MPS driver limit cannot exceed 4 GiB")
        require(type(self.learning_rate) in (int, float) and math.isfinite(self.learning_rate)
                and 0 < self.learning_rate <= .001, "Finite engineering learning rate in (0,.001] required")
        require(type(self.seed) is int and 0 <= self.seed < 2**32, "Explicit uint32 seed required")


def validate_fixture_receipt(receipt):
    """Admit only the pinned development smoke; never infer permission to fit."""
    require(receipt["schema"] == "hydra-directflow-local-smoke/v1"
            and receipt["status"] == "PASSED_DEVELOPMENT_SMOKE", "A passed DEVELOPMENT_SMOKE fixture is required")
    require(receipt["origin_utc"] == ORIGIN and receipt["source_cycle"] == CYCLE,
            "Only the fixed August 2021 development origin is admitted")
    require(tuple(receipt["gauge_ids"]) == GAUGE_IDS and receipt["history_hours"] == 24
            and tuple(receipt["horizons_hours"]) == HORIZONS_HOURS, "Development fixture coordinates differ")
    require(receipt["history_start_utc"] == "2021-08-22T01:00:00Z"
            and receipt["history_end_utc"] == "2021-08-23T00:00:00Z"
            and receipt["target_start_utc"] == "2021-08-23T03:00:00Z"
            and receipt["target_end_utc"] == "2021-08-23T20:00:00Z", "Development fixture clock alignment differs")
    require(receipt["sources"]["hrrr"]["sha256"] == HRRR_SHA256
            and receipt["sources"]["usgs"]["sha256"] == USGS_SHA256
            and receipt["geometry"]["weights_sha256"] == WEIGHTS_SHA256, "Pinned source identities differ")
    require(receipt["geometry"]["scientific_admission"] is False
            and receipt["geometry"]["original_candidate_status_unchanged"] is True,
            "Original development-only candidate basin status must remain explicit")
    require(receipt["availability"]["mode"] == "retrospective_simulated"
            and receipt["availability"]["lag_seconds"] == 7200
            and receipt["availability"]["retrieval_time_used_as_availability"] is False,
            "The explicit two-hour simulated availability scenario is required")
    for name, expected in FEATURES.items():
        require(tuple(receipt["features"][name]) == expected, "Feature ordering differs: " + name)
    require(receipt["features"]["target"] == "discharge_cms"
            and receipt["features"]["transform"] == "PHYSICAL_UNITS_UNSCALED"
            and receipt["features"]["quality_policy"] == "STRICT_EXACT_A_ONLY",
            "This engineering fixture requires unscaled physical units and strict A-only quality")
    require(receipt["execution"]["assessment_rows_read"] == 0
            and receipt["execution"]["training_steps"] == 0, "Fixture must precede training and exclude assessment")
    identity = receipt["private_tensor_artifact"]
    require(receipt["private_tensor_artifact_written"] is True and identity["format"] == "npz"
            and identity["sha256"] == FIXTURE_SHA256 and type(identity["bytes"]) is int
            and identity["bytes"] == FIXTURE_BYTES, "Only the pinned provenance-v2 private fixture is admitted")
    return dict(npz_sha256=identity["sha256"], npz_bytes=identity["bytes"],
                hrrr_sha256=HRRR_SHA256, usgs_sha256=USGS_SHA256, weights_sha256=WEIGHTS_SHA256,
                origin_utc=ORIGIN, cycle=CYCLE)


def plan_probe(receipt_path, spec):
    require(isinstance(spec, ProbeSpec), "A validated ProbeSpec is required")
    raw = _local_bytes(receipt_path, 64 * 1024)
    source = validate_fixture_receipt(json.loads(raw))
    source["receipt_sha256"] = _sha(raw)
    package = Path(__file__).resolve().parent
    implementation = {name: _sha(_local_bytes(package / name, 128 * 1024))
                      for name in ("engineering.py", "models/directflow.py", "models/mamba2_math.py")}
    identity = dict(spec=asdict(spec), source=source, unit_mapping=UNIT_MAPPING,
                    implementation_sha256=implementation, optimizer="AdamW_foreach_false",
                    loss="masked_mse_on_fixed_unit_targets",
                    target_role="LOSS_ONLY_NEVER_FORWARD", qualification=QUALIFICATION)
    return dict(schema="hydra-engineering-probe-plan/v1", status="PLANNED_NOT_EXECUTED",
                artifact_created_at_utc=now_utc(), spec_sha256=digest(identity),
                source_digest=digest(source), **identity)


def _verify_coordinates(site_ids, history_times, target_times):
    require(all(type(value) is int for values in (site_ids, history_times, target_times) for value in values),
            "Embedded coordinates require integer identities and epoch seconds")
    require(tuple(site_ids) == (0, 1, 2), "Embedded gauge ordering differs")
    require(tuple(history_times) == HISTORY_TIMES and tuple(target_times) == TARGET_TIMES,
            "Embedded history/target coordinates differ from the admitted development origin")


def _utc_source_epoch(value):
    """Compare equivalent explicit UTC spellings without rewriting provenance."""
    require(isinstance(value, str), "Original source clock must be an explicit UTC string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)
            and parsed.microsecond == 0, "Original source clock must be integral and timezone-aware UTC")
    return int(parsed.timestamp())


def _verify_selected_provenance(metadata):
    """Validate chosen-row original metadata separately from fixed unit mapping."""
    for name, count in (("history", 24), ("target", 18)):
        flags, ids = metadata[name + "_quality_flags"], metadata[name + "_source_ids"]
        require(len(flags) == len(ids) == count, "Observation provenance dimensions differ")
        require(all(json.loads(value) == ["A"] for value in flags), "Chosen USGS qualifiers must be exactly A-only")
        require(all(isinstance(value, str) and value.startswith("usgs-wire-sha256:")
                    and _is_digest(value[len("usgs-wire-sha256:"):]) for value in ids),
                "Original chosen USGS wire identities are required")

    def check_record(record, valid, group):
        require(record["group"] == group
                and _utc_source_epoch(record["source_reference_UTC"]) == ORIGIN_EPOCH - 7200
                and _utc_source_epoch(record["source_valid_UTC"]) == valid
                and record["lead"] == (valid - (ORIGIN_EPOCH - 7200)) // 3600,
                "Original HRRR record coordinates differ")
        require(record["group_availability"]["status"] == "AVAILABLE_VERIFIED"
                and isinstance(record["group_availability"]["source_key"], str)
                and bool(record["group_availability"]["source_key"]), "Original HRRR group support is required")
        require(all(_is_digest(record[key]) for key in ("source_values_sha256_float64_le", "missing_bits_sha256")),
                "Original record values/mask hashes required")
        source = record["source_identity"]
        require(isinstance(source["key"], str) and source["key"] and _is_digest(source["sha256"])
                and type(source["bytes"]) is int and source["bytes"] > 0,
                "Original source byte identity required")

    require(len(metadata["weather_source_metadata"]) == len(metadata["weather_source_ids"])
            == len(metadata["weather_record_indices"]) == 18, "Weather provenance horizon differs")
    units = ("kg m-2", "K", "kg kg-1", "kg m-2")
    for valid, row, ids, indices in zip(TARGET_TIMES, metadata["weather_source_metadata"],
                                       metadata["weather_source_ids"], metadata["weather_record_indices"]):
        require(len(row) == len(ids) == len(indices) == 4, "Weather provenance feature axis differs")
        for column, raw in enumerate(row):
            record = json.loads(raw)
            check_record(record, valid, WEATHER_GROUPS[column])
            require(record["field_source_id"] == ids[column] and record["record_index"] == indices[column],
                    "Weather original identity/index alignment differs")
            require(record["units"] == units[column]
                    and record["field_availability"]["historical_available_at_UTC"] is None,
                    "Weather source units/publication scenario differs")
            expected_interval = [record["lead"] - 1, record["lead"]] if column == 0 else None
            require(record["APCP_interval_hours"] == expected_interval, "Original precipitation interval differs")
    soil = json.loads(metadata["soil_source_metadata"])
    check_record(soil, ORIGIN_EPOCH - 7200, "soil_moisture")
    require(soil["source_identity"]["key"] == metadata["soil_source_ids"]
            and soil["record_index"] == metadata["soil_record_indices"]
            and soil["historical_available_at_UTC"] is None and soil["native_units"] == "Proportion",
            "Original initial-soil identity/index/publication differs")


def _load_case(path, source, gauge_id):
    """Read named arrays only; select exactly one gauge before model admission."""
    import numpy as np
    from .local_data import bounded_npy
    raw = _local_bytes(path, 2 * 1024**2)
    require(len(raw) == source["npz_bytes"] and _sha(raw) == source["npz_sha256"], "Private fixture identity changed")
    shapes = {"history": (3, 24, 2), "weather": (3, 18, 4), "state": (3, 2), "target": (3, 18)}
    row = GAUGE_IDS.index(gauge_id)
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        require(len(members) <= 64 and len({m.filename for m in members}) == len(members)
                and all(m.file_size <= 4 * 1024**2 for m in members)
                and sum(m.file_size for m in members) <= 16 * 1024**2, "NPZ inventory exceeds bounded unique contract")
        coordinates = [bounded_npy(archive.read(name + ".npy"), shape=shape, dtype="<i8").tolist()
                       for name, shape in (("site_id", (3,)), ("history_time", (24,)), ("target_time", (18,)))]
        _verify_coordinates(*coordinates)

        def text_array(name, shape):
            data = archive.read(name + ".npy")
            stream = io.BytesIO(data)
            version = np.lib.format.read_magic(stream)
            require(version in ((1, 0), (2, 0)), "Unsupported provenance array format")
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            actual_shape, fortran, dtype = reader(stream)
            require(actual_shape == shape and not fortran and dtype.kind == "U"
                    and 0 < dtype.itemsize <= 4 * 4096, "Bounded original Unicode provenance required")
            return bounded_npy(data, shape=shape, dtype=dtype)[row].tolist()

        metadata = {}
        for name, shape in (("history_quality_flags", (3, 24)), ("history_source_ids", (3, 24)),
                            ("target_quality_flags", (3, 18)), ("target_source_ids", (3, 18)),
                            ("weather_source_metadata", (3, 18, 4)), ("weather_source_ids", (3, 18, 4)),
                            ("soil_source_metadata", (3,)), ("soil_source_ids", (3,))):
            metadata[name] = text_array(name, shape)
        for name, shape in (("weather_record_indices", (3, 18, 4)), ("soil_record_indices", (3,))):
            metadata[name] = bounded_npy(archive.read(name + ".npy"), shape=shape, dtype="<i8")[row].tolist()
        _verify_selected_provenance(metadata)
        for name, shape in shapes.items():
            values = bounded_npy(archive.read(name + ".npy"), shape=shape, dtype="<f8")[row:row + 1]
            mask = bounded_npy(archive.read(name + "_mask.npy"), shape=shape, dtype="?")[row:row + 1]
            require(np.isfinite(values[mask]).all(), "Admitted values must be finite")
            if name != "target":
                require(mask.all(), "Complete input support required for this engineering fixture")
            else:
                require(mask.any() and (values[mask] >= 0).all(), "At least one valid nonnegative target is required")
            mapped = np.zeros_like(values)
            if name == "target":
                mapped[mask] = values[mask] / UNIT_MAPPING[name][0][1]
            else:
                for column, (offset, divisor) in enumerate(UNIT_MAPPING[name]):
                    admitted = mask[..., column]
                    mapped[..., column][admitted] = (values[..., column][admitted] - offset) / divisor
            result[name] = mapped.astype(np.float32)
            result[name + "_mask"] = mask
    return result


def _cpu_tree(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item, torch) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(item, torch) for item in value)
    return value


def _worker(request_path):
    """Private subprocess entry; parent owns the cumulative run allowance."""
    def deny_network(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            raise RuntimeError("Network prohibited during local engineering probe")
    sys.addaudithook(deny_network)
    request = json.loads(_local_bytes(request_path, 128 * 1024))
    plan = request["plan"]
    spec = ProbeSpec(**plan["spec"])
    actual_plan = plan_probe(request["receipt_path"], spec)
    require(actual_plan["spec_sha256"] == plan["spec_sha256"]
            and actual_plan["source_digest"] == plan["source_digest"]
            and actual_plan["source"] == plan["source"], "Reviewed plan/source receipt changed")
    plan = actual_plan
    require(spec.max_optimizer_steps > 0, "Worker requires a positive precharged budget")
    import torch
    import psutil
    from .models.directflow import DirectFlowModel, ModelConfig, masked_mse
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(spec.seed)
    if spec.device == "mps":
        require(torch.backends.mps.is_available(), "MPS unavailable; no CPU fallback allowed")
        torch.mps.set_per_process_memory_fraction(min(1., min(2 * 1024**3, spec.max_combined_bytes)
                                                       / torch.mps.recommended_max_memory()))
    case = _load_case(request["fixture_path"], plan["source"], spec.gauge_id)
    tensors = {name: torch.from_numpy(value).to(spec.device) for name, value in case.items()}
    process = psutil.Process()
    losses = []
    started = time.monotonic()
    output_dir = Path(request_path).parent

    def snapshot():
        return dict(confirmed_optimizer_steps=len(losses), host_rss_bytes=process.memory_info().rss,
                    mps_driver_bytes=torch.mps.driver_allocated_memory() if spec.device == "mps" else 0)

    stop = threading.Event()
    monitor_errors = []

    def sample():
        try:
            while not stop.is_set():
                _write_json(output_dir / "worker-status.json", snapshot())
                stop.wait(.05)
        except Exception as exc:
            monitor_errors.append(type(exc).__name__)

    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()

    def status():
        require(not monitor_errors, "Worker resource telemetry failed")
        current = snapshot()
        require(time.monotonic() - started <= spec.max_wall_seconds, "Worker time limit exceeded")
        require(current["host_rss_bytes"] + current["mps_driver_bytes"] <= spec.max_combined_bytes,
                "Worker memory limit exceeded")

    status()
    model_config = ModelConfig(spec.architecture, FEATURES["history"], FEATURES["weather"], FEATURES["state"],
                               history_hours=24, horizon_hours=HORIZONS_HOURS, width=64, dropout=.1)
    model = DirectFlowModel(model_config).to(spec.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, foreach=False)
    for _ in range(spec.max_optimizer_steps):
        status()
        optimizer.zero_grad(set_to_none=True)
        # Targets/masks are a separate loss argument and never model inputs.
        prediction = model(*(tensors[name] for name in
                             ("history", "history_mask", "weather", "weather_mask", "state", "state_mask")))
        loss = masked_mse(prediction, tensors["target"], tensors["target_mask"])
        require(bool(torch.isfinite(loss)), "Nonfinite training-only loss")
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                "Missing/nonfinite training gradient")
        optimizer.step()
        if spec.device == "mps":
            torch.mps.synchronize()
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Nonfinite updated parameter")
        losses.append(float(loss.detach().cpu()))
        status()
    checkpoint = dict(schema="hydra-engineering-checkpoint/v1", qualification=QUALIFICATION,
                      artifact_created_at_utc=now_utc(), spec_sha256=plan["spec_sha256"],
                      source_digest=plan["source_digest"], spec=asdict(spec), model_config=asdict(model_config),
                      implementation_sha256=plan["implementation_sha256"], torch_version=str(torch.__version__),
                      model=_cpu_tree(model.state_dict(), torch), optimizer=_cpu_tree(optimizer.state_dict(), torch),
                      cpu_rng=torch.get_rng_state().cpu(),
                      device_rng=torch.mps.get_rng_state().cpu() if spec.device == "mps" else None,
                      optimizer_steps=len(losses), unit_mapping=UNIT_MAPPING)
    checkpoint_path = output_dir / "engineering-checkpoint.pt"
    with checkpoint_path.open("xb") as stream:
        torch.save(checkpoint, stream)
    status()
    stop.set()
    watcher.join(timeout=1)
    require(not watcher.is_alive() and not monitor_errors, "Worker resource telemetry did not complete")
    _write_json(output_dir / "worker-status.json", snapshot())
    result = dict(status="COMPLETED_ENGINEERING_ONLY", confirmed_optimizer_steps=len(losses),
                  training_losses_fixed_units=losses, scaler_fits=0, protected_assessment_scores=0,
                  valid_target_cells=int(tensors["target_mask"].sum().cpu()),
                  checkpoint_file=checkpoint_path.name, checkpoint_bytes=checkpoint_path.stat().st_size,
                  checkpoint_sha256=_sha(checkpoint_path.read_bytes()),
                  spec_sha256=plan["spec_sha256"], source_digest=plan["source_digest"],
                  implementation_sha256=plan["implementation_sha256"], torch_version=str(torch.__version__),
                  artifact_created_at_utc=now_utc(), qualification=QUALIFICATION)
    _write_json(output_dir / "worker-result.json", result)


def run_engineering_probe(*, receipt_path, fixture_path, spec, run_directory=None,
                          execution_enabled=False, approved_spec_sha256=None):
    """Plan by default; an enabled call runs one supervised, precharged attempt.

    The approved digest is an explicit caller decision bound to this exact plan,
    not an assertion that a main study is frozen or training is scientifically
    validated. No automatic retry, resumption or allocation increase occurs.
    """
    require(type(execution_enabled) is bool, "Execution enablement must be an explicit boolean")
    plan = plan_probe(receipt_path, spec)
    if not execution_enabled:
        return plan
    require(spec.max_optimizer_steps > 0, "Choose a positive bounded update budget before execution")
    require(approved_spec_sha256 == plan["spec_sha256"], "Execution requires the exact reviewed spec digest")
    require(run_directory is not None, "Explicit new private run directory required")
    source_paths = {Path(receipt_path).resolve(), Path(fixture_path).resolve()}
    directory = Path(run_directory).resolve()
    require(directory not in source_paths and not directory.exists(), "Run directory must be new and separate from inputs")
    import psutil
    journal = RunJournal(directory, plan, max_wall_seconds=spec.max_wall_seconds,
                         max_rss_bytes=spec.max_combined_bytes)
    deadline = time.monotonic() + spec.max_wall_seconds
    journal.event("optimizer_budget_precharged", maximum_optimizer_steps=spec.max_optimizer_steps)
    request_path = directory / "private-worker-request.json"
    _write_json(request_path, dict(plan=plan, receipt_path=str(Path(receipt_path).resolve()),
                                   fixture_path=str(Path(fixture_path).resolve())))
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       HF_HUB_DISABLE_TELEMETRY="1", PYTORCH_ENABLE_MPS_FALLBACK="0")
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + environment.get("PYTHONPATH", "")
    summary = dict(schema="hydra-engineering-probe-result/v1", status="FAILED", qualification=QUALIFICATION,
                   artifact_created_at_utc=now_utc(), spec_sha256=plan["spec_sha256"], source_digest=plan["source_digest"],
                   optimizer_steps_precharged=spec.max_optimizer_steps, confirmed_optimizer_steps=0,
                   optimizer_step_count_exact=False, network_requests=0, protected_assessment_scores=0,
                   scaler_fits=0, scientific_campaign_enabled=False)
    child = None
    peak = 0
    try:
        with journal.stage("training_engineering"):
            with (directory / "worker.log").open("x") as log:
                child = subprocess.Popen([sys.executable, "-m", __name__, "--worker", str(request_path)],
                                         env=environment, stdout=log, stderr=log)
                process, supervisor = psutil.Process(child.pid), psutil.Process()
                while child.poll() is None:
                    try:
                        rss = supervisor.memory_info().rss + process.memory_info().rss
                        rss += sum(p.memory_info().rss for p in process.children(recursive=True))
                        status_path = directory / "worker-status.json"
                        current = json.loads(status_path.read_text()) if status_path.exists() else {}
                        combined = rss + current.get("mps_driver_bytes", 0)
                        peak = max(peak, combined)
                        journal.metric("combined_observed_peak_bytes", peak)
                        require(combined <= spec.max_combined_bytes, "Combined RSS/MPS memory limit exceeded")
                        require(time.monotonic() < deadline, "Overall engineering wall limit exceeded")
                    except psutil.NoSuchProcess:
                        break
                    time.sleep(.05)
                child.wait(timeout=5)
                require(child.returncode == 0, "Engineering worker failed; inspect private worker.log")
            result = json.loads((directory / "worker-result.json").read_text())
            require(result["confirmed_optimizer_steps"] == spec.max_optimizer_steps
                    and result["spec_sha256"] == plan["spec_sha256"]
                    and result["source_digest"] == plan["source_digest"], "Worker result differs from admitted plan")
            checkpoint = journal.artifact(directory / result["checkpoint_file"])
            require(checkpoint["sha256"] == result["checkpoint_sha256"]
                    and checkpoint["bytes"] == result["checkpoint_bytes"], "Checkpoint custody identity differs")
            summary.update(result, checkpoint=checkpoint, optimizer_step_count_exact=True)
        journal.finish()
    except Exception as exc:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        summary.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        if journal.status == "RUNNING":
            journal.finish("FAILED", error_type=type(exc).__name__)
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        status_path = directory / "worker-status.json"
        if not summary["optimizer_step_count_exact"] and status_path.exists():
            summary["confirmed_optimizer_steps"] = json.loads(status_path.read_text())["confirmed_optimizer_steps"]
        summary["combined_observed_peak_bytes"] = peak
        summary["journal_status"] = journal.status
        _write_json(directory / "summary.json", summary)
    return summary


if __name__ == "__main__":
    require(len(sys.argv) == 3 and sys.argv[1] == "--worker", "Use run_engineering_probe; no implicit training CLI")
    _worker(sys.argv[2])
