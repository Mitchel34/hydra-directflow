# Bounded training engineering API

`hydra_directflow.engineering` prepares a plan by default. Importing it and
running its standard-library unit tests performs no optimizer updates, model
imports, real fixture decoding, network activity or training. Implementation of
this API is not a receipt that a real-data training probe has run.

The only supported fixture is the passed local smoke for origin
`2021-08-23T02:00:00Z`: L24 history ending at 00Z, hourly H18 targets through 20Z,
the pinned HRRR/USGS sources and unchanged candidate weights. The input receipt,
private NPZ digest and byte count, source identities, feature order, strict
A-only quality, simulated two-hour availability and declared coordinates are
checked. Choose exactly one of the three confirmed gauges per invocation.
There is no full-study, scientific-campaign, cloud or protected-assessment path.

The private provenance-v2 NPZ is pinned to SHA256
`0aad781c8bcb77a901e1f6c3e1a5d2efc1115d7819145ceeb1d410a4532548b8` and
30,047 bytes. Arbitrary replacement NPZ hashes are rejected even if a receipt
asserts the same source names. At execution, embedded gauge IDs, all history and
target epochs, chosen A-only qualifiers/wire identities, per-target weather
source clocks/units/intervals/hashes, and initial-soil identity are checked before
model admission. An explicit later code/review change is needed to admit any
newly generated fixture.

The runner uses declared fixed engineering unit mappings, not fitted scalers:
discharge and its target are divided by 100; history age by 168; precipitation by
10; temperature is `(K−273.15)/30`; specific humidity by 0.01; snow water
equivalent by 100; soil fraction is unchanged and soil age is divided by 24.
These choices make a bounded optimizer mechanics check possible. They are not a
recommended scientific transform, selected hyperparameters or evidence of skill.
Targets and target masks are used only by the training loss and never appear in
the model's forward arguments. No target filling or input imputation occurs.

## Planning and later explicit execution

```python
from hydra_directflow.engineering import ProbeSpec, run_engineering_probe

spec = ProbeSpec(gauge_id="03161000", architecture="LSTM", device="cpu",
                 max_optimizer_steps=2, max_wall_seconds=120)
plan = run_engineering_probe(receipt_path=receipt_path,
                            fixture_path=private_fixture_path, spec=spec)
# plan["status"] == "PLANNED_NOT_EXECUTED"
```

Review the concrete plan before an independently authorized tiny probe. An
enabled call must supply `execution_enabled=True`, the reviewed
`approved_spec_sha256`, and an explicit new private `run_directory`. A changed
gauge, architecture, budget, mapping, source receipt or runner/model source code
changes that digest. The
default step budget is zero and the default execution switch is false. The
worker is an implementation detail; callers use the supervised API.

Limits are at most 16 optimizer updates, 300 seconds overall and 4 GiB for
supervisor/worker RSS plus MPS driver allocation. CPU and MPS are the only
supported devices; there is no fallback, retry, resumption or automatic limit
increase. Each invocation precharges its whole allowed update budget in the run
journal before launching a worker. This is a per-attempt bound, not a cumulative
campaign authorization: callers must separately account for every later attempt.
The supervisor polls the child and records observed peaks; polling is not a
proof that no shorter peak occurred. MPS also has an allocator cap of at most
2 GiB. Imports, decoding, updates and checkpoint work share the parent deadline.

## Receipts and checkpoint custody

Training uses one initialized width-64 model, fixed seed, AdamW and dropout 0.1.
Every update checks finite loss, gradients and resulting parameters. The receipt
records **training losses on one repeated observed case**, which cannot establish
generalization, hydrologic performance or forecasting skill. No prediction export
or protected metric is generated.

The private checkpoint stores model state, optimizer state, Torch CPU RNG and
MPS RNG when applicable, model configuration, unit mappings, actual creation time,
source digest, reviewed spec digest and confirmed update count. Its exact bytes
and SHA256 are recorded through `RunJournal`. Store run directories and
checkpoints outside Git. Checkpoint creation alone does not prove fresh-process
restoration; that requires its own explicit recovery test.

Actual artifact timestamps stay separate from the historical origin. On success,
the confirmed step count is exact. On failure or forced termination, the last
confirmed count can undercount a step interrupted before telemetry was written;
the precharged maximum remains the conservative allowance consumed. Failure
logs, summary, and journal remain available. No failure silently returns a
scientifically qualified model.
