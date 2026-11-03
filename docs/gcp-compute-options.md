# GCP compute options for DirectFlow — November 3, 2026

This is read-only exploration. No VM, job, reservation, quota change, API enable,
IAM change, data-object transfer or model run was requested. Scientific fits,
paid cloud jobs and protected assessment remain held. The
[evidence and arithmetic](../reports/gcp-compute-options-20261103.json) retain
timestamps, source hashes, prices, scope and unresolved constraints.

## Recommendation

Prepare a versioned CUDA trainer and benchmark **one G2 `g2-standard-8` with an
L4** first. This is the most practical GPU candidate under the observed quota
and existing machine-rate ceiling. Compare larger GPUs only after measuring
completed-fit throughput and cost. The model families have approximately
68–86 thousand parameters; memory capacity alone is not evidence of speedup.

For the 180 independent fits, first consider distributing separate gauge/fold/
seed/input-arm jobs. This preserves B64 and per-fit learning semantics. Splitting
one small fit across several GPUs would add unimplemented distributed training
without an established benefit. One cloud GPU alongside separately assigned
local fits is another candidate; measure the scheduling benefit and preserve
each fit's chosen backend. Never treat CPU/MPS-to-CUDA migration as exact resume.

## Current account evidence

At 19:32 UTC, the existing project's Compute Engine metadata showed:

- Global GPU limit1, usage0; us-central1 L4 limit1, usage0.
- us-central1 A10040GB and A10080GB quotas0.
- All22 listed Compute Engine instances terminated. This does not establish
  zero storage charges or inventory other cloud services.
- Selected G2, A2, A3 and G4 shapes appear in the us-central1-b catalog.
  Catalog presence and quota do not guarantee a successful allocation.
- The separate modern quota query timed out after40seconds. RTX/H100 quota
  remains **unverified**, rather than being inferred as zero from missing legacy
  metrics. No retry, quota request or account change was made.

Existing quota supports planning for one L4. Two separate L4 VMs would need a
reviewed regional/global quota increase. None was requested.

## Price and hardware comparison

Displayed default/Iowa whole-machine on-demand prices, checked November3; GPU,
vCPU and host memory are included. Extra disks, networking and taxes are excluded.
These are research quotes, not an invoice or guaranteed capacity. Rates in other
regions need separate verification.

| Candidate | GPU memory | vCPU / host GiB | USD/hour | VM-only24hours |
|---|---:|---:|---:|---:|
| G2 `g2-standard-4`, L4 | 24GB | 4 /16 | 0.707 | 16.96 |
| **G2 `g2-standard-8`, L4** | **24GB** | **8 /32** | **0.854** | **20.49** |
| A2 `a2-highgpu-1g`, A100 | 40GB | 12 /85 | 3.673 | 88.16 |
| A2 `a2-ultragpu-1g`, A100 | 80GB | 12 /170 | 5.069 | 121.65 |
| G4 `g4-standard-6`, fractional RTX PRO6000 | 12GB; one-eighth GPU | 6 /22 | 0.647 | 15.53 |
| G4 `g4-standard-48`, full RTX PRO6000 | 96GB | 48 /180 | 4.500 | 108.00 |

Sources: [Google accelerator pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized)
and [machine specifications](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines).

At these prices, A10040GB needs **4.30 times** L4-8 throughput and full G4 needs
**5.27 times** its throughput to reduce compute cost for identical work. These
ratios are arithmetic, not predicted performance. Two L4-8 VMs for24hours would
cost40.97USD for the machines, but current quota prevents that configuration and
aggregate scaling has not been measured. A larger machine can still be justified
by a deadline even when cost per fit rises; that requires explicit budget review.

Important alternatives and constraints:

- Fractional G4 is currently restricted to us-central1-b. It needs vGPU driver
  authentication through an enabled service account/scopes; G4 boot storage
  uses Hyperdisk Balanced, not Persistent Disk. Its fraction of a Blackwell GPU
  must not be represented as a full card. This is a secondary candidate with
  additional setup and quota checks. [G4 requirements](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines),
  [locations](https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones).
- One-GPU H100 A3 High is Spot or Flex-start only. A reliable one-GPU price was
  not established from the retrieved pricing table; dividing an eight-GPU quote
  is not valid. Spot can be reclaimed at any time, so it adds recovery risk near
  the poster deadline. [A3 restrictions](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines),
  [Spot behavior](https://docs.cloud.google.com/compute/docs/instances/spot).
- A CPU-only C3 `c3-standard-8` is listed at0.403216USD/hour, with8vCPU/32GiB.
  The regional C3 quota is24vCPUs with zero usage. This is a fallback candidate
  for independent CPU fits, not a measured acceleration over the Mac. It avoids
  CUDA integration but still needs Linux runtime/recovery qualification, a
  bounded scheduler and review of the different VM scope.
  [Google general-purpose pricing](https://cloud.google.com/products/compute/pricing/general-purpose?hl=en).

## Software readiness and benchmark design

The actual scientific trainer accepts CPU/MPS only. The separate H4 pilot is
LSTM/CPU-only. CUDA helpers exist, but CUDA trainer/checkpoint/inference integration
and a real GPU qualification have not run. No hardware quote removes these gates.

Current Mamba2 runs the verified unfused math path; a GPU does not automatically
substitute fused SSD kernels. Host preparation, gathers/transfers, finite checks
and checkpoint serialization can limit end-to-end throughput. These are reasons
to measure, not evidence that CUDA will or will not help. Retain FP32 and the
declared numerical settings first; AMP, TF32, larger batches, fused Mamba or
compiled alternatives need separate qualification before changing comparisons.

The proposed first paid test, after code/readiness review and explicit release,
would use the reserved10USD CUDA allocation and at most one L4 worker within the
existing four-machine-hour ceiling. The recorded four-architecture L24/L168
synthetic recovery matrix is a starting plan; Q-only coverage required by the
primary matrix must be explicitly included. Missingness/H4 coverage, if retained,
is a separate addition. Bind the final update and model-call counts before launch.

Measure exact fresh-process checkpoint continuation, selected-best label-free
inference, complete batch preparation/transfer/update time, clean-inner and outer
inference, cold start, artifact growth, CPU/GPU utilization, memory and cost per
completed fit. Start with one fit process. Consider concurrency only if the GPU
is underused and a bounded comparison improves aggregate throughput without
breaking memory, reproducibility or rate limits. Preserve measured failures.

Reuse the small verified prepared development products; do not reacquire or
restage the entire HRRR archive. Any required cloud upload, dependency setup and
output prefix belongs in the reviewed staging/launch plan. Keep original sources
and protected assessment isolated. A VM lifetime limit, persistent spend/update
ledger, durable verified checkpoints and independent shutdown control precede
scientific cloud execution; a billing alert is not a spending cap.

## Funding and deadline

The approved incremental envelope remains150USD:35staging,10CUDA qualification,
75scientific compute and30contingency, with at most two workers and a1USD per-VM
hour limit under the recorded G2/L4 scope. Scientific cloud spending remains held.
A100/full G4 exceed that hourly limit and need an explicit revision even if a
short run fits the total envelope; fractional G4/CPU options also change VM scope.

The completed staging receipt reports a12.907585USD conservative estimate, not a
reconciled bill. Do not call150USD unused or subtract that estimate to invent an
available balance. Historical acquisition liabilities and retained storage are
separate. No budget increase is proposed until measured throughput identifies
the smallest configuration and runtime that can meet the scientific plan.

The current main local proxies are82.45hours at50epochs and329.26hours at the
proposed200 ceiling if all fits reach that cap, before stated overhead. Completing
the latter modeled work in48hours would require about6.86times aggregate speedup
before extra overhead. No GCP speedup or November6 completion guarantee has been
established. Qualify, measure and then decide whether the cloud workload fits
the remaining money and deadline.
