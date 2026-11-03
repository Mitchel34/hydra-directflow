# CUDA preparation without execution

Trainer v2 still admits only CPU and MPS. Its [fresh-process qualification](../reports/trainer-recovery-v2.json) covers CPU LSTM and MPS Mamba2–Transformer in full and Q-only modes; it does not test CUDA. `TrainerSpec` continues to reject CUDA. This CUDA preparation is separate from that trainer update and leaves existing benchmark sources and receipts unchanged. `devices.py` provides standalone runtime primitives and a plan-only qualification entry point; it does not create a VM, access provider credentials, release a budget or run a model.

Print the pending plan with:

```sh
PYTHONPATH=src .venv-offline-check/bin/python -m hydra_directflow.devices --plan-cuda
```

The command does not import Torch or initialize an accelerator. A local `--gates-json` can bind existing trusted approval/ledger/price-quote identities, protocol/cohort identities and the measured justification for cloud escalation. Even when those inputs satisfy the structural checks, the result stays `GATES_RECORDED_PLAN_ONLY` with execution disabled. Hashes bind supplied evidence; they do not authenticate an external approval or substitute for the user's authorization.

The approved combined envelope is $150, including a $10 CUDA-pilot allocation. Staging remains the active phase. A proposed pilot must fit that allocation and the remaining combined ledger after other commitments/reservations. The plan requires admitted data, a locked protocol and measured local need before escalation. Existing conditional authorization can provide the approval evidence when its conditions are met; this module does not require inventing another approval or silently release the other allocations.

The prepared qualification geometry is four existing architectures × L24/L168 at B64 and FP32. Each case would compare an uninterrupted two-update path with a fresh-process continuation after the first update: three synthetic updates per case, **24 total**. The proposed limits are 900 seconds, four GiB host RSS and four GiB of selected CUDA allocator memory. There is no real-data fit, new architecture, mixed precision or CUDA-speed claim. These limits and counts describe a future supervised qualification, not completed work.

The device helpers require `cpu`, `mps` or an explicit `cuda:<index>`. They refuse fallback or implicit current-device selection. CPU seeding uses a dedicated CPU generator state so it does not secretly seed every GPU. CUDA seeding initializes its RNG inside the selected-device context before applying the seed; synchronization, RNG capture/restore and allocator telemetry refer to that same index. Checkpoints keep CPU byte tensors for CPU and selected-device RNG state. Backend/index, tensor encoding and runtime mismatches fail before a resumed update.

CUDA configuration requires an allocator cap. FP32 qualification disables TF32 and cuDNN benchmarking and records numerical flags. Deterministic qualification requires `CUBLAS_WORKSPACE_CONFIG` in the worker environment before CUDA startup. The runtime profile records Torch/Python/OS, the selected GPU/capability, toolkit/cuDNN and precision settings. The runner must supply the actual NVIDIA driver version from its recorded hardware inspection; an unknown driver blocks exact runtime compatibility. No hostname, GPU UUID or private filesystem path is included.

CUDA allocator counters exclude context/driver allocations and non-PyTorch VRAM. MPS exposes current allocator/driver values, with peaks requiring sampling. CPU memory uses an external RSS supervisor. Allocator fractions alone are not a whole-process or container limit. A paid pilot still needs a supervisor that enforces actual wall, host/device memory, persistent update accounting, budget reservations, interruption receipts and resource cleanup.

The next integration should be a separate, reviewed trainer/checkpoint version: include `devices.py` in implementation fingerprints; replace hardcoded MPS branches with selected-device RNG/runtime helpers; preserve model, AdamW, train-only transform, sampler cursor/permutation and stopping state; restore RNG only after model/optimizer construction; and verify fresh-process next-update parity. Do not relabel old CPU/MPS checkpoints as CUDA-qualified or silently migrate across backends. Use the existing exact math Mamba implementation until a separate fused-kernel comparison qualifies an alternative.

The 13 device tests exercise real CPU RNG replay and mocked CUDA/MPS behavior, funding/limit gates and the non-executing command. They perform zero model forwards and zero optimizer updates. They do not establish real CUDA availability, numerical equivalence, throughput, cost or readiness to launch.

`reports/cloud-readonly-readiness.json` records a November2 21:31UTC API snapshot:
all22 acquisition VMs were terminated; us-central1 L4 quota and global GPU quota
were each1 with usage0. No resources, quotas or permissions were changed. Use a
single-GPU planning assumption under existing quota. This establishes neither
zonal capacity nor a successful launch, and terminated instances do not prove
zero retained-disk or other storage costs.

The [November3 read-only compute comparison](gcp-compute-options.md) refreshes
the quota snapshot and compares L4, A100, G4 and CPU alternatives. It preserves
the CUDA implementation/qualification and paid-job holds above. Hardware prices
and quota are not measured workload speedups or launch authorization.
