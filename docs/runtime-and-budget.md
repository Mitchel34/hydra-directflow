# Measured pipeline timing and conditional budget inputs

Eight local engineering runs passed on November 1, 2026 (America/Chicago):
four architectures on CPU and MPS. The entire campaign took **39.60 seconds**,
confirmed **48 optimizer updates**, and peaked at **883 MiB** of observed combined
supervisor/worker RSS plus MPS driver allocation. Every checkpoint restored an
exact matching forward prediction. No scientific forecast skill was calculated.

Each run used gauge `03161000`, L168, H18, 16 training cases, four chronologically
later development-validation cases, and three complete engineering epochs.
Training batches had eight cases; validation batches had four. The validation
origin was February 29, 2024 at 02Z. Training labels and validation labels were
disjoint. Fixed unit mappings were used; no research scaler was fitted.

First preprocessing passes took **0.92–1.39 seconds** for five authenticated
cycles, five USGS monthly partitions and 60 gauge cases. Repeated passes took
**0.76–0.91 seconds**. These first passes occurred in fresh Python processes,
but the operating system's disk cache was not flushed. The measurements include
source authentication, source decoding, basin reduction and case assembly.
The runner additionally measures cache writes/verified reads, tensorization,
device transfers, model updates, validation, and checkpoint write/reload.

## Local runtime sensitivity

The editable scenario in `configs/runtime-scenario.json` assumes 10,000 training
cases and 1,000 development-validation cases **per gauge**, 20 epochs, one
configuration and one seed. Four architectures and three gauges produce 12
separate fits. Validation and checkpoint writes occur each epoch.

| Calculation | CPU, all four models | MPS, all four models |
|---|---:|---:|
| Mean of all measured batches, including first use | 3.23 h | 21.31 h |
| Later-epoch batch means plus first-use excess once per fit | 2.32 h | 8.13 h |

These are two assumption-driven sensitivity calculations, **not confidence
bounds**. There are only six measured training batches and three validation
batches per candidate. Engineering finite-value checks and synchronization are
included. Small batches and first-use overhead strongly affect the comparison.
It does not establish a hardware winner for larger batches; the earlier B64
synthetic benchmark answers a different question.

The estimates exclude full-corpus source loading/preprocessing, convergence,
scientific evaluation, parameter search management, fault recovery and cloud
hardware. Trial and seed counts multiply the fit count; an assumed epoch count
is not evidence of how many epochs will converge. Recompute without training:

```sh
PYTHONPATH=src .venv-offline-check/bin/python scripts/project_runtime.py \
  --scenario configs/runtime-scenario.json \
  --output .local/new-runtime-scenario.json
```

See `reports/pipeline-timing.json` for measurements and
`reports/runtime-scenarios.json` for formulas, inputs and file digests.

## Cloud unit prices, with execution still on hold

Google's displayed default on-demand price for `g2-standard-4` (one L4, four
vCPUs, 16 GiB) is **$0.706832276 per machine-hour**. At that listed rate, 1/4/12/24
elapsed machine-hours cost approximately **$0.71/$2.83/$8.48/$16.96** for compute.
These are supplied-duration scenarios, not estimates of this experiment's L4
runtime. Confirm the applicable regional SKU before any approved pilot.
[Google accelerator VM pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized).

For a separate hypothetical storage allocation, 50 GiB of Standard storage in
Iowa for seven days is approximately **$0.23** at the displayed
$0.000027397/GiB-hour rate. Requests, transfer, persistent disks, existing retained
resources and taxes are excluded. Actual bucket location/class and total payload
size still need confirmation. [Google Storage pricing](https://cloud.google.com/storage/pricing).

The cloud campaign budget remains **undecided**. Local CPU/MPS timing cannot be
converted into L4 runtime. `reports/budget-scenarios.json` records these price
references and omissions; it grants no execution or spending authority.

## What the next chunk needs

The sealed index contains 53,064 source initializations before the assessment
period: 52,905 complete cycles and 159 explicit gaps. Of the six-hour
initializations, 8,773 are complete. Under the current two-hour availability
scenario, ordinary native-18-hour runs cannot supply direct H18, so fetching all
ordinary weather payloads is unnecessary for that scenario. Some incomplete
cycles retain usable weather or soil and must remain explicit in selection.

The full payload byte total is unknown from the index alone. Scaling the five
extended fixtures to the complete six-hour cycles gives roughly **47.6 GB**;
this is a sizing scenario, not an inventory. Idealized numeric basin weather,
soil and hourly discharge arrays would occupy about **35.8 MB** before metadata,
identities and indexes. Materializing every possible L168 case would instead
require about **613 MB** in float64 values plus masks. See
`reports/development-scale.json` for assumptions and exact metadata counts.

The next implementation should normalize older authenticated index formats,
cache basin reductions once per cycle and discharge once per hour, and assemble
windows from case indices. It must checkpoint chunk completion, authenticate
the index once per worker, and resolve all required candidates before claiming
globally latest selection. July 2024 also needs a separate protected-data
boundary because its USGS monthly partition overlaps the assessment start.
The November 2 continuation completed the legacy adapter, resumable caches and
an actual August 2023 timing exercise. See [month preprocessing](month-preprocessing.md):
9.22 seconds with an empty derived cache including stop/resume, 5.12 seconds
for warm full assembly, and 0.78 seconds for verification-only resume. These
observations replace the earlier single-fixture preprocessing evidence for this
month; the full corpus and cloud hardware remain unmeasured. A separately
approved, capped storage read supplied the month. No cloud compute job has started.
