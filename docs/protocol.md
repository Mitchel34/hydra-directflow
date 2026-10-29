# Direct-streamflow development protocol

This is a tested development candidate, not a frozen main-study protocol or a
scientific result. Confirmed scope is hourly leads 1–18 at gauges `03161000`,
`03164000`, and `03479000`, with LSTM, Transformer, GRU–Transformer and
Mamba2–Transformer candidates. NWM is an external comparator only; it cannot be a
predictor or a prerequisite for independent training eligibility.

## Clocks and complete-cycle selection

`C` is both the nominal hourly forecast origin and information cutoff. Targets
are exactly `C+1h` through `C+18h`. The development smoke uses explicit simulated
two-hour input lags. These are scenario assumptions, not provider publication
guarantees or a reconstruction of historical USGS revisions.

`latest_weather_run` considers only cycles available at `C`. A cycle must cover
every target and have a true original support mask for every required field:
hourly precipitation, temperature, specific humidity and snow water equivalent.
Coordinates alone do not qualify a cycle. With explicit
`allow_older_extended_cycles=True`, selection falls back to the latest complete
eligible older cycle; source leads may exceed 18. It never stitches cycles.
Without that permission, an incomplete newest eligible cycle returns no run.
Competing vintages require upstream resolution. Original source quality flags
remain attached to the selected cycle; support-mask construction still requires
the source adapter's numeric, unit, precipitation-interval and quality checks.

The selected cycle age, each source lead, and delivery-relative lead are distinct.
`weather_support` is a coordinate diagnostic; its `supported` values do not
replace feature masks. `completion` is historical replay completion, not the time
the present-day artifact was created. `ReplayClock` stores simulated cutoff and
completion separately and stamps actual artifact creation from the current wall
clock by default. Do not backdate model fits or output artifacts to the replay.

## History, soil, quality and missingness

L24 and L168 are configurable development candidates. A length `L` history has
`L` consecutive exact hourly slots ending at `C−lag`, inclusive. With a two-hour
lag, L24 spans `C−25h` through `C−2h`. A missing hour stays missing; the window
does not search farther into the past, interpolate, use another gauge or include
the unavailable most recent hours. The availability policy and independently
supplied input-quality admission must both pass. Original flags and source IDs
are retained even for rejected rows.

Soil is a separate available initial state from 00/06/12/18 UTC, with explicit
age. Eight hours is a development maximum-age candidate. It is never repeated
into an invented future soil forecast. Invalid fractions, absent states and stale
states stay masked. Full source/version/grid and quality qualification is an
adapter responsibility; a finite scalar alone cannot establish that provenance.

Labels use their own explicit quality admission. Exact target timestamps and
original flags remain visible; no target filling occurs. Missing precipitation is
not zero. Input and label quality policies must be recorded by a consuming run,
and may differ. A complete-input smoke rejects incomplete input cases; it must
not count model-boundary numerical placeholders as observed values. Existing
masked tensor encoding is only a network representation.

## Development and protected assessment

Development folds require every target to remain inside the same half-open split
and any explicitly chosen embargo. A target on or past a development boundary
purges that origin. Candidate train/validation dates are in
`configs/development.json`; choosing these dates for main fits remains separate.

The protected assessment fixes origins to
`[2024-07-31T00:00:00Z, 2025-07-31T00:00:00Z)`: 8,760 requested hourly origins.
The last origin is July 30 at 23Z and its last target is July 31 at 17Z. The
predeclared label tail preserves those origins; it adds no origins and never
becomes development data. Coordinate admission does not imply observed labels,
complete inputs or scoring permission.

Protected scoring is disabled by default. A future scoring entry point must call
`require_scoring_allowed` before loading or computing protected outcomes, with
explicit enablement and SHA256 identities for locked protocol and selection
artifacts. The helper checks structure; it does not authenticate external lock
artifacts or grant study approval. Development configurations cannot enable this
gate. Dashboards and smoke tests must not expose protected skill.

## Spatial and execution scope

The bounded local smoke may reuse the original 514 candidate weight rows as
approximate independent study basins. Preserve original weights and the prior
specific outlet/area acceptances. Do not clip polygons or impose nesting. This
does not assert final main-study spatial admission, surveyed outlets or nested
conservation. Grid/version binding and source-quality metadata must remain
inspectable in the adapter receipt.

The pure contract tests use synthetic records and standard-library `unittest`.
They perform no API calls, source-payload reads, optimizer steps, training, cloud
actions or spending. Main lookback, scientific loss/transforms, fit grouping and
budget remain development decisions. Current local smoke execution is a
separately bounded adapter task; its receipt must identify actual source bytes,
artifacts and tests. No source archive or original preparation directory is
modified by this protocol chunk.
