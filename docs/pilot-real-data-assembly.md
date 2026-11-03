# Real-data assembly for the proposed CV1 gap pilot

The local assembly passed on November 3, 2026 at 16:59 UTC. It prepares inputs
for a proposed experiment; it does not authorize scientific fitting or scoring.
The strict 180-fit primary campaign and protected assessment gates are unchanged.

The [execution receipt](../reports/gap-pilot-real-assembly-v1.json) binds the
verified development snapshot, 37 original monthly discharge-cache commitments,
source code and exact role memberships. The source plan is `0dad0738…`; the
development snapshot is `b41630b7…`. The immutable prepared manifest SHA256 is
`4795efd085a4924df9b4b6b5b96933cb9c70590ebc71ec6e46484c63a259c540`.
Private paths and the full prepared manifest remain outside the public repository.

The assembly read 240 selected development shards and the approved local source
cache, without replaying the completed whole-corpus integrity allowance. It used
44,407,134 counted source bytes and wrote 57,750,943 bytes in 57.62 seconds,
with 135,217,152 bytes observed parent-plus-child peak RSS. Bounds were 900 seconds,
4 GiB combined RSS, 1 GiB source reads and 1 GiB output. Workers denied network
connections and model-runtime imports. Model fits, scaler fits, optimizer updates,
paid jobs and protected-data access were all zero.

## Source checks and stored products

Each history and target slot is joined by gauge and exact UTC hour. Original
qualifiers and source request IDs are retained verbatim; a shared source ID is
not mistaken for a unique observation key. The approved A/optional-R predicate
and numeric-valid masks must agree exactly with the existing corpus. Values must
also agree exactly on admitted slots. Missing or rejected values stay missing.
Transport absence is a hard error, never a substituted missing observation.
The two-hour publication delay is simulated; original revision delivery remains
unknown.

The prepared products contain:

- Original H2 inputs, with separate label files and immutable source-shard
  commitments. H4 indicators and fills are created later by the qualified adapter.
- Compact monthly provenance lookups that retain qualifiers, source IDs,
  row-presence and numeric-valid masks without storing target values in the
  predictor-side metadata.
- Strict common-L168 fit, inner and outer memberships for both paired arms.
- A separate all-candidate outer coverage stream with original missing-input and
  target masks. It does not enlarge training membership.
- Three unique-hour ridge source series restricted to the gradient interval,
  with eligible six-lag/two-output pair indices; no ridge coefficients fitted.
- Unnormalized gradient-only gap-shape counts for reviewing the augmentation
  recipe. Neither inner nor outer gaps determine that recipe.

| Gauge | Fit | Inner | Paired outer | All outer candidates | Eligible ridge pairs |
|---|---:|---:|---:|---:|---:|
| 03161000 | 15,717 | 2,152 | 8,556 | 8,742 | 15,809 |
| 03164000 | 13,024 | 2,152 | 8,341 | 8,742 | 15,370 |
| 03479000 | 14,621 | 2,152 | 8,516 | 8,742 | 15,604 |

All strict role case-list hashes match the published CV1 delay-two authority.
Target tails are strictly inside their role boundaries; fit and inner labels
also satisfy the declared refit-availability predicate. Coverage origins use
only the outer calendar/target-time boundary, independently of target quality.
These are engineering counts, not forecasts, skill results or independent events.

## Reproducible interfaces and limits

`scripts/assemble_gap_pilot.py` prepares a metadata-only plan by default. Explicit
execution authenticates the exact plan and current implementation and preserves
its controller receipts. It cannot overwrite earlier attempts. Source metadata
and payload reads reject symlinks, cloud placeholders and changed size/checksums.
Committed output chunks are immutable; the module's reviewed resume path also
compares existing payloads against newly recomputed intended bytes.

`load_training_bundle` returns only fit+inner rows for one gauge. Separate
`iter_outer_inputs` and `iter_coverage_inputs` interfaces never open label files;
explicit label readers serve retrospective evaluation. `load_ridge_series`
returns only the unique gradient hours. Exact prepared coordinates are checked
before feature/value members are decoded.

Twenty-eight focused synthetic tests passed, including source/mask mismatches,
qualifier preservation, protected-time rejection, refit purges, label-free
input reads, missing provenance, resource bounds and resumed-output tampering.
Independent review passed. Actual prepared-loader qualification is the next
engineering check; the assembly receipt alone is not a scientific execution
release. The bounded protocol, augmentation recipe and tolerances require review.
