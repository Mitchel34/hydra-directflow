# Read-only development corpus dataset

`corpus_dataset.py` reads the committed corpus format without modifying the
corpus, touching raw sources, hydrating files, fitting statistics, training a
model, or contacting a service. A complete dataset snapshot is still preparation
evidence, not a frozen scientific cohort or permission to train.

## Snapshot contract

```python
from hydra_directflow.corpus_dataset import inspect_corpus, iter_batches

snapshot = inspect_corpus(
    ".local/development-corpus-L168-v3",
    expected_plan_sha256="<externally recorded corpus plan digest>",
)
report = snapshot.report()
```

The reader verifies the plan digest, source-authority identifiers, scientific
split intervals, storage/chunk order, H18 boundary purges, and candidate-source
calendar. It records the SHA256 of each immutable shard commitment observed in
that scan. Each commitment's payload hashes, NPZ member inventory, expansion
bounds, numeric schema, clocks, masks, source ages and admission accounting are
checked. Coordinate checks precede flow/weather/state value decoding.

This verifies **derived plan and shard bytes**. It does not reopen upstream
basin caches, hourly caches, source archives or their scientific evidence. Their
references remain bound in the shard records, and the report explicitly says
`upstream_sources_reaudited=False`.

Missing planned chunks produce a `PARTIAL_COMMITTED_CORPUS_SNAPSHOT`; an existing
but corrupt, incomplete, symlinked or nonresident committed chunk fails rather
than silently disappearing from the cohort. Uncommitted staging directories are
not included. The mutable corpus `RESULT.json` is not used as completion proof.
Completion means every chunk in the pinned plan was observed and verified.

Later appends do not expand an existing snapshot or selection. A fresh scan
creates a new snapshot digest when the observed commitment set changes. Replacing
a commitment already captured by a snapshot causes iteration to fail.

## Explicit history and cohort choices

```python
selection = snapshot.select(
    gauge_id="03161000",
    split="candidate_training",
    history_hours=24,
    cohort_mode="common_l168_complete",
    allow_partial=True,  # Deliberate provisional inspection, never final-cohort admission.
)
```

Gauge, split, history view and cohort mode are required. Partial selection needs
an explicit `allow_partial=True`; otherwise it raises `IncompleteCorpusError`.
No primary missingness or lookback policy is chosen by the adapter.

| Mode | Included cases |
|---|---|
| `all_candidates` | Every published candidate case for the requested gauge/split, retaining masks and admission state |
| `natural_complete` | Complete full-feature cases under the requested L24 or L168 history view |
| `common_l168_complete` | The original L168-complete full-feature cohort, used unchanged for either L24 or L168 views |

A staged L168 history can provide an L24 view by taking its final 24 exact hourly
slots. Natural L24 eligibility is recomputed from those slots, weather, soil,
target masks and calendar support. An older missing hour outside the final 24
hours does not keep that case excluded merely because its stored L168 history
was incomplete. Recent history gaps, missing weather/soil and missing labels
remain exclusions. L168 cannot be reconstructed from a shorter store.

The source's original `source_eligibility_reason` and `source_admission_mask`
remain separate from derived `view_eligibility_reason` and `view_complete_mask`.
The membership digest binds snapshot, gauge, split, immutable commitments and
selected row indices. The separate selection digest also binds the history view
and explicit candidate mode. Thus matched L24/L168 common cohorts share a
membership digest while retaining distinct view specifications.

## Full and Q-only views

```python
full_batches = iter_batches(selection, feature_view="full", batch_size=64)
q_only_batches = iter_batches(selection, feature_view="q_only", batch_size=64)
```

Both views use the same selected cases and cohort digest. Q-only does not expand
to cases that lacked weather or soil in the full-feature admission. It preserves
history, labels, masks, order and clocks, substitutes only the known raw horizon
offsets 1–18, and uses an empty initial state. Corpus admission independently
authenticates and validates the full-feature derived shards in its snapshot;
the Q-only view never supplies their weather/soil values as model predictors. No normalization is fitted or applied here.

Iteration loads one bounded shard at a time and yields read-only per-gauge
copies, with at most 128 cases per batch. It does not concatenate the full corpus,
shuffle cases, cross shard boundaries to fill batches, or run campaign logic.
Compressed NPZ is not zero-copy memory mapping. A future dense/memmap export or
trainer integration requires a separate explicit implementation and validation.
