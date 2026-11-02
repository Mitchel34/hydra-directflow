# Bounded typed bundles for trainer mechanics

`trainer_bundle.py` converts explicit `corpus_dataset` selections into the
NumPy mapping expected by `DevelopmentTrainer`. It does not instantiate a
trainer, fit a scaler, run an optimizer, access a service, or write source data.
Scientific admission remains external.

```python
from hydra_directflow.trainer_bundle import assemble_bundle

bundle = assemble_bundle(
    [training_selection, validation_selection],
    feature_view="full",  # Or "q_only", with the identical selections and roles.
    role_time_bounds={
        "fit": {"start_utc": "<explicit UTC hour>", "end_utc": "<exclusive UTC hour>"},
        "inner_stop": {"start_utc": "<explicit UTC hour>", "end_utc": "<exclusive UTC hour>"},
        "outer": {"start_utc": "<explicit UTC hour>", "end_utc": "<exclusive UTC hour>"},
    },
    max_total_bytes=512 * 1024**2,
    max_batch_reads=1000,
    batch_size=128,
)
```

These example role names are supported mechanics, not recommended dates or a
selected CV design. Omit roles that the caller has not declared. An omitted
`inner_stop` raises on lookup; it never falls back to `outer`.

Selections must share one gauge, history view, immutable snapshot and source
plan. They are combined chronologically, retaining exact case IDs, labels,
masks, input values and clocks. Duplicated, overlapping or interleaved gauge
cases fail. A partial snapshot is refused unless the caller explicitly supplies
`allow_partial=True`; that remains provisional engineering evidence.

The adapter allocates once after checking the dense byte estimate and planned
batch-read count. It streams bounded per-shard batches into the typed arrays;
there is no repeated full-corpus concatenation. Process RSS is guarded with a
maximum of four GiB, including a preallocation working-memory reserve. The
returned arrays are read-only and the mapping cannot gain or replace keys.

## Explicit roles and strict target windows

Each role has an exact caller-supplied half-open UTC interval. A row belongs to
that role only when its origin **and every H18 target timestamp** lie inside the
interval. Role boundaries never come from the first admitted later-role case;
missing observations cannot move a cutoff. Overlapping, out-of-order or
protected-period role intervals fail.

Rows outside all role intervals or whose targets cross a role boundary remain
in the bundle, explicitly unassigned. The manifest reports origin-window counts,
purged counts, zero-row support, and observed-label support. Missing labels stay
missing; no quality-based readmission or target filling occurs in this adapter.

`bundle.index_ranges("fit")` returns compact half-open array-index ranges.
`bundle.indices_for("fit")` explicitly returns their bounded, reiterable tuple
of Python integer indices, compatible with the existing trainer split validator.
The same methods support declared `inner_stop` and `outer` roles. Callers must
choose which indices to pass to trainer mechanics; the adapter does not make
that scientific choice.

These are **strict target-window bounds only**. The workload planner's optional
simulated two-hour refit-label convention is not implemented or adopted here.
Any additional label publication/refit cutoff must be resolved explicitly
before scientific fitting. Source-time bounds do not reconstruct historical
USGS revision availability.

## Identity and provenance

Full and Q-only assembly from identical selections preserves one combined
comparison-cohort digest and the same role-assignment digest. Q-only supplies
only discharge history, age and known horizon tokens, with an empty state;
its original full-feature cohort remains unchanged. No normalization occurs.

Per-shard commitments, selected source-row indices, source split names and
output-index ranges live in the separate manifest, not repeated in model
feature tensors. The bundle identity binds the feature schema, raw-array
identity, provenance and explicit roles. Runtime measurements are kept in a
separate construction report, so repeat assembly of unchanged inputs does not
change the source identity merely because timing or memory usage differs.

The manifest explicitly reports `scientific_admission=False`. This component
is neither a campaign runner nor a generic storage framework. It does not
replace upstream source validation, freeze preprocessing choices, or enable
protected assessment.
