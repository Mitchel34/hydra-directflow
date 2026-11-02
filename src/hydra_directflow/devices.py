"""Explicit CPU/MPS/CUDA runtime primitives and a non-executing CUDA plan.

Importing this module does not import Torch, initialize an accelerator, access a
provider, or authorize spending. The qualified development trainer is unchanged;
these helpers require a separately reviewed integration and CUDA qualification.
"""
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat

from .contract import require

ARCHITECTURES = ("LSTM", "Transformer", "GRU_Transformer", "Mamba2_Transformer")
GIB = 1024**3


@dataclass(frozen=True)
class DeviceSelection:
    kind: str
    index: int | None = None

    def __post_init__(self):
        require(self.kind in ("cpu", "mps", "cuda"), "Only explicit CPU, MPS and CUDA backends are supported")
        if self.kind == "cuda":
            require(type(self.index) is int and 0 <= self.index < 128, "Explicit nonnegative CUDA index required")
        else:
            require(self.index is None, "CPU/MPS selections must not carry a CUDA index")

    @property
    def name(self):
        return self.kind if self.index is None else f"cuda:{self.index}"

    @classmethod
    def parse(cls, value):
        require(type(value) is str, "Device must be an explicit string")
        if value in ("cpu", "mps"):
            return cls(value)
        match = re.fullmatch(r"cuda:(0|[1-9][0-9]*)", value)
        require(match is not None, "Use cpu, mps, or cuda:<explicit index>; implicit CUDA/current-device aliases are refused")
        return cls("cuda", int(match.group(1)))


@dataclass(frozen=True)
class RuntimePolicy:
    device: str
    allocator_limit_bytes: int | None = None
    deterministic_algorithms: bool = False
    allow_tf32: bool = False

    def __post_init__(self):
        selected = DeviceSelection.parse(self.device)
        require(type(self.deterministic_algorithms) is bool and type(self.allow_tf32) is bool,
                "Precision/determinism choices must be explicit booleans")
        if selected.kind == "cpu":
            require(self.allocator_limit_bytes is None, "CPU memory requires a host RSS supervisor, not an accelerator allocator limit")
        elif self.allocator_limit_bytes is not None:
            require(type(self.allocator_limit_bytes) is int and 0 < self.allocator_limit_bytes <= 64 * GIB,
                    "Bounded positive accelerator allocator limit required")
        require(not self.allow_tf32 or selected.kind == "cuda", "TF32 applies only to an explicitly selected CUDA backend")


def _torch(module=None):
    if module is None:
        import torch
        return torch
    return module


def available_device(device, *, torch_module=None):
    """Fail rather than substituting another backend or CUDA index."""
    selected = DeviceSelection.parse(device)
    torch = _torch(torch_module)
    if selected.kind == "mps":
        require(torch.backends.mps.is_available(), "MPS is unavailable; CPU fallback is forbidden")
    elif selected.kind == "cuda":
        require(torch.cuda.is_available(), "CUDA is unavailable; CPU/MPS fallback is forbidden")
        require(selected.index < torch.cuda.device_count(), "Requested CUDA device index does not exist")
    return selected


def synchronize(device, *, torch_module=None):
    torch = _torch(torch_module)
    selected = available_device(device, torch_module=torch)
    if selected.kind == "mps":
        torch.mps.synchronize()
    elif selected.kind == "cuda":
        torch.cuda.synchronize(selected.index)


def seed_rng(device, seed, *, torch_module=None):
    """Seed CPU plus the selected accelerator, never every CUDA device."""
    require(type(seed) is int and 0 <= seed < 2**32, "Explicit uint32 seed required")
    torch = _torch(torch_module)
    selected = available_device(device, torch_module=torch)
    # torch.manual_seed also seeds all CUDA devices. A CPU Generator avoids that
    # hidden side effect while setting the normal default CPU generator state.
    torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
    if selected.kind == "mps":
        torch.mps.manual_seed(seed)
    elif selected.kind == "cuda":
        with torch.cuda.device(selected.index):
            # manual_seed queues a callback before CUDA initialization. Force
            # initialization while this selected index is still current.
            torch.cuda.get_rng_state(selected.index)
            torch.cuda.manual_seed(seed)


def _rng_bytes(value, torch):
    require(isinstance(value, torch.Tensor) and value.dtype == torch.uint8 and value.ndim == 1
            and 0 < value.numel() <= 65536 and value.device.type == "cpu", "Bounded CPU byte-tensor RNG state required")
    return value.detach().clone()


def capture_rng(device, *, torch_module=None):
    torch = _torch(torch_module)
    selected = available_device(device, torch_module=torch)
    synchronize(device, torch_module=torch)
    accelerator = None
    if selected.kind == "mps":
        accelerator = _rng_bytes(torch.mps.get_rng_state().cpu(), torch)
    elif selected.kind == "cuda":
        accelerator = _rng_bytes(torch.cuda.get_rng_state(selected.index).cpu(), torch)
    return dict(schema="hydra-device-rng/v1", device=selected.name,
                cpu=_rng_bytes(torch.get_rng_state().cpu(), torch), accelerator=accelerator)


def restore_rng(state, device, *, torch_module=None):
    """Validate every state before mutating either RNG; cross-device replay fails."""
    torch = _torch(torch_module)
    selected = available_device(device, torch_module=torch)
    require(isinstance(state, dict) and set(state) == {"schema", "device", "cpu", "accelerator"}
            and state["schema"] == "hydra-device-rng/v1" and state["device"] == selected.name,
            "RNG checkpoint requires the same explicitly selected backend/index")
    cpu = _rng_bytes(state["cpu"], torch)
    if selected.kind == "cpu":
        require(state["accelerator"] is None, "CPU RNG state cannot contain an accelerator state")
        accelerator = None
    else:
        accelerator = _rng_bytes(state["accelerator"], torch)
    require(cpu.numel() == torch.get_rng_state().numel(), "CPU RNG encoding length changed")
    if selected.kind == "mps":
        require(accelerator.numel() == torch.mps.get_rng_state().numel(), "MPS RNG encoding length changed")
    elif selected.kind == "cuda":
        require(accelerator.numel() == torch.cuda.get_rng_state(selected.index).numel(), "CUDA RNG encoding length changed")
    synchronize(device, torch_module=torch)
    torch.set_rng_state(cpu)
    if selected.kind == "mps":
        torch.mps.set_rng_state(accelerator)
    elif selected.kind == "cuda":
        torch.cuda.set_rng_state(accelerator, device=selected.index)


def memory_snapshot(device, *, torch_module=None):
    """Allocator telemetry is not a claim about total process or device memory."""
    torch = _torch(torch_module)
    selected = available_device(device, torch_module=torch)
    if selected.kind == "cpu":
        return dict(device="cpu", allocator_allocated_bytes=None, allocator_reserved_bytes=None,
                    allocator_peak_allocated_bytes=None, allocator_peak_reserved_bytes=None,
                    scope="CPU allocator telemetry unavailable here; measure process RSS separately")
    if selected.kind == "mps":
        return dict(device="mps", allocator_allocated_bytes=torch.mps.current_allocated_memory(),
                    driver_allocated_bytes=torch.mps.driver_allocated_memory(),
                    recommended_max_memory_bytes=torch.mps.recommended_max_memory(),
                    scope="MPS current allocator/driver values; peaks require external sampling")
    return dict(device=selected.name, allocator_allocated_bytes=torch.cuda.memory_allocated(selected.index),
                allocator_reserved_bytes=torch.cuda.memory_reserved(selected.index),
                allocator_peak_allocated_bytes=torch.cuda.max_memory_allocated(selected.index),
                allocator_peak_reserved_bytes=torch.cuda.max_memory_reserved(selected.index),
                scope="PyTorch CUDA allocator only; excludes driver/context and non-PyTorch VRAM; host RSS is separate")


def runtime_profile(device, *, torch_module=None, cuda_driver_version=None):
    """Portable numerical/runtime identity without hostnames, UUIDs or paths."""
    torch = _torch(torch_module)
    selected = available_device(device, torch_module=torch)
    result = dict(schema="hydra-device-runtime/v1", device=selected.name, torch_version=str(torch.__version__),
                  python_version=platform.python_version(), operating_system=platform.system(),
                  operating_system_release=platform.release(), machine=platform.machine(),
                  cpu_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                  default_dtype=str(torch.get_default_dtype()), requested_model_precision="float32_no_autocast",
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
    if selected.kind == "cuda":
        require(cuda_driver_version is None or type(cuda_driver_version) is str and bool(cuda_driver_version),
                "CUDA driver version must come from the worker's recorded hardware inspection")
        properties = torch.cuda.get_device_properties(selected.index)
        result.update(cuda_toolkit_version=torch.version.cuda, cuda_driver_version=cuda_driver_version,
                      driver_version_source="CALLER_HARDWARE_RECEIPT" if cuda_driver_version else "NOT_YET_COLLECTED",
                      cudnn_version=torch.backends.cudnn.version(),
                      accelerator_name=properties.name, device_total_memory_bytes=properties.total_memory,
                      compute_capability=list(torch.cuda.get_device_capability(selected.index)),
                      cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                      cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                      cudnn_benchmark=bool(torch.backends.cudnn.benchmark), cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                      float32_matmul_precision=torch.get_float32_matmul_precision(),
                      cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    return result


def configure_runtime(policy, *, torch_module=None):
    """Apply an explicit policy inside an already authorized dedicated worker.

    This does not provision hardware, supervise wall/RSS limits, authenticate
    funding, or grant a training allowance. CUDA needs an allocator cap.
    """
    require(isinstance(policy, RuntimePolicy), "Validated runtime policy required")
    torch = _torch(torch_module)
    selected = available_device(policy.device, torch_module=torch)
    fraction = None
    if selected.kind == "cuda":
        require(policy.allocator_limit_bytes is not None, "CUDA preparation requires an explicit allocator cap")
        if policy.deterministic_algorithms:
            require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"),
                    "Set CUBLAS_WORKSPACE_CONFIG before the CUDA worker starts for deterministic qualification")
        total = torch.cuda.get_device_properties(selected.index).total_memory
        require(policy.allocator_limit_bytes <= total, "CUDA allocator cap exceeds selected device memory")
        fraction = policy.allocator_limit_bytes / total
        torch.cuda.set_device(selected.index)
        torch.cuda.set_per_process_memory_fraction(fraction, device=selected.index)
        torch.backends.cuda.matmul.allow_tf32 = policy.allow_tf32
        torch.backends.cudnn.allow_tf32 = policy.allow_tf32
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = policy.deterministic_algorithms
        torch.set_float32_matmul_precision("high" if policy.allow_tf32 else "highest")
    elif selected.kind == "mps" and policy.allocator_limit_bytes is not None:
        recommended = torch.mps.recommended_max_memory()
        require(recommended > 0, "MPS recommended memory is unavailable")
        fraction = min(1., policy.allocator_limit_bytes / recommended)
        torch.mps.set_per_process_memory_fraction(fraction)
    torch.use_deterministic_algorithms(policy.deterministic_algorithms, warn_only=False)
    return dict(policy=asdict(policy), runtime=runtime_profile(policy.device, torch_module=torch),
                allocator_fraction=fraction,
                qualification="Configuration only; requires an external wall/RSS/allocator supervisor and device-specific numerical tests")


def require_compatible_runtime(saved, current):
    if isinstance(current, dict) and str(current.get("device", "")).startswith("cuda:"):
        require(isinstance(current.get("cuda_driver_version"), str) and bool(current["cuda_driver_version"]),
                "CUDA restore requires the current worker's recorded driver version")
    require(isinstance(saved, dict) and isinstance(current, dict) and saved == current,
            "Checkpoint runtime/backend/precision identity changed; requalification is required")


@dataclass(frozen=True)
class CudaQualificationLimits:
    max_optimizer_updates: int = 24
    max_wall_seconds: int = 900
    max_host_rss_bytes: int = 4 * GIB
    max_device_allocator_bytes: int = 4 * GIB

    def __post_init__(self):
        require(type(self.max_optimizer_updates) is int and self.max_optimizer_updates == 24,
                "Prepared qualification matrix requires exactly a 24-update ceiling")
        require(type(self.max_wall_seconds) is int and 1 <= self.max_wall_seconds <= 900, "CUDA pilot wall limit cannot exceed 900 seconds")
        for value in (self.max_host_rss_bytes, self.max_device_allocator_bytes):
            require(type(value) is int and 0 < value <= 4 * GIB, "Explicit host/allocator caps cannot exceed four GiB each")


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _money(value):
    require(type(value) in (str, int, float), "Explicit numeric budget value required")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("Invalid budget amount") from error
    require(amount.is_finite() and amount >= 0, "Budget amount must be finite and nonnegative")
    return amount


def validate_funding_record(record):
    """Validate a supplied approval/ledger binding; never invent or fetch one."""
    require(isinstance(record, dict) and record.get("schema") == "hydra-cuda-pilot-funding/v1"
            and record.get("status") == "APPROVED_WITHIN_COMBINED_ENVELOPE", "Explicit approved pilot funding record required")
    for key in ("approval_sha256", "budget_ledger_sha256", "price_quote_sha256"):
        require(_digest(record.get(key)), "Funding approval, current ledger and price quote identities are required")
    cap, other, pilot, estimate = (_money(record[key]) for key in
                                  ("combined_cap_usd", "other_committed_or_reserved_usd", "pilot_cap_usd", "pilot_upper_estimate_usd"))
    require(0 < cap <= Decimal("150") and 0 < pilot <= Decimal("10") and estimate <= pilot and other + pilot <= cap,
            "Pilot must fit its approved 10-dollar allocation and remaining combined 150-dollar envelope")
    return {key: record[key] for key in ("schema", "status", "approval_sha256", "budget_ledger_sha256", "price_quote_sha256",
                                         "combined_cap_usd", "other_committed_or_reserved_usd", "pilot_cap_usd", "pilot_upper_estimate_usd")}


def plan_cuda_qualification(*, funding=None, data_protocol_ready=False, local_escalation_required=False,
                            protocol_sha256=None, cohort_sha256=None, local_feasibility_sha256=None, limits=None):
    """Return a conditional plan only. This function cannot execute a pilot."""
    require(type(data_protocol_ready) is bool and type(local_escalation_required) is bool, "Explicit conditional gates required")
    limits = CudaQualificationLimits() if limits is None else limits
    require(isinstance(limits, CudaQualificationLimits), "Validated CUDA limits required")
    blockers = []
    approval = None
    if funding is None:
        blockers.append("APPROVED_PILOT_AND_CURRENT_COMBINED_BUDGET_RECORD_REQUIRED")
    else:
        approval = validate_funding_record(funding)
    if not data_protocol_ready or not _digest(protocol_sha256) or not _digest(cohort_sha256):
        blockers.append("ADMITTED_DATA_AND_LOCKED_PROTOCOL_REQUIRED")
    if not local_escalation_required or not _digest(local_feasibility_sha256):
        blockers.append("LOCAL_FEASIBILITY_MUST_JUSTIFY_CLOUD_ESCALATION")
    cases = [dict(architecture=architecture, history_hours=length, batch_size=64, optimizer_updates=3)
             for architecture in ARCHITECTURES for length in (24, 168)]
    require(sum(case["optimizer_updates"] for case in cases) == limits.max_optimizer_updates, "Qualification accounting differs")
    root = Path(__file__).resolve().parent
    code = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("devices.py", "models/directflow.py", "models/mamba2_math.py")}
    result = dict(schema="hydra-cuda-qualification-plan/v1", status="BLOCKED_PLAN_ONLY" if blockers else "GATES_RECORDED_PLAN_ONLY",
                  execution_enabled=False, trainer_cuda_enabled=False, provider_operations=0, optimizer_updates_executed=0,
                  blockers=blockers, funding_record=approval, protocol_sha256=protocol_sha256,
                  cohort_sha256=cohort_sha256, local_feasibility_sha256=local_feasibility_sha256,
                  limits=asdict(limits), cases=cases, device="cuda:0", dtype="float32", autocast=False, tf32=False,
                  required_worker_environment={"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HUB_OFFLINE": "1",
                                               "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"},
                  planned_checks=["Record selected CUDA GPU, NVIDIA driver/CUDA/cuDNN/PyTorch versions and precision flags",
                                  "Synthetic CPU-reference and CUDA forward/gradient comparisons with declared tolerances",
                                  "Continuous two updates versus fresh-process restore after update one and matching next update",
                                  "Model, AdamW state, CPU/selected-CUDA RNG, sampler and transform identity verification",
                                  "Synchronized update/validation/checkpoint timing and host RSS/CUDA allocator supervision"],
                  update_accounting="Eight cases times three synthetic updates; precharge before execution; failed/unknown attempts retained; no automatic retry",
                  numerical_qualification="Not yet executed; no CUDA equivalence or speed claim",
                  integration_status="Standalone primitives only; existing qualified CPU/MPS trainer remains unchanged",
                  implementation_sha256=code,
                  qualification="Approval hashes bind caller-supplied evidence; this plan does not authenticate external approval documents or create authorization. A separately reviewed supervisor is still required.")
    result["plan_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Emit a non-executing CUDA qualification plan; never provision or train")
    parser.add_argument("--plan-cuda", action="store_true", required=True)
    parser.add_argument("--gates-json", type=Path, help="Optional local funding/data/local-feasibility gate record")
    args = parser.parse_args()
    gates = {}
    if args.gates_json is not None:
        info = args.gates_json.lstat()
        require(stat.S_ISREG(info.st_mode) and not getattr(info, "st_flags", 0) & 0x40000000 and 0 < info.st_size <= 65536,
                "Gate record must be a bounded resident regular JSON file")
        gates = json.loads(args.gates_json.read_bytes())
        if "limits" in gates:
            gates["limits"] = CudaQualificationLimits(**gates["limits"])
    print(json.dumps(plan_cuda_qualification(**gates), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
