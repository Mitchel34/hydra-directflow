# Spatial support before scientific protocol freeze

Independent reproduction now confirms all three weight sets against their retained official USGS delineations. This supports a documented **approximate gauge-associated basin** interpretation; it does not establish surveyed, exact current-gauge watersheds or grant final scientific admission. The original candidate CSV and its `CANDIDATE_SCIENTIFIC_HOLD` / `admitted=false` fields remain unchanged.

`scripts/audit_spatial_support.py` independently reconstructs native cell polygons and their basin intersections from SHA-pinned resident evidence. The first report, `reports/spatial-support-audit-resident-v1.json`, reproduces Jefferson and Sugar Grove. Galax’s retained source body was dataless, so that report deliberately carries the prior authenticated Galax diagnostics separately from current reproduction. No source was hydrated by the audit. The original geometry producer script is also dataless; this audit implements the resident, pinned methods independently rather than executing or materializing that old script.

The user separately approved materializing the exact 334,177-byte Galax body.
Its first bounded child read timed out at 60 seconds; the original subsequently
became resident. An eight-second bounded local completion verified its allocated
resident bytes and recorded SHA before publishing the copy, with no additional
explicit hydration request or new USGS request. FileProvider wire bytes remain
unobserved. `reports/galax-owned-geometry-materialization.json` preserves that
sequence. The intermediate `spatial-support-audit-full-v2.json` is still partial;
the final `spatial-support-audit-full-v3.json` independently reproduces all three
basins and the non-nesting result. All earlier reports remain unchanged.

| Gauge | Retained delineation | Basin area, km² | Difference from retained published area | Outlet diagnostic | Positive HRRR cells |
|---|---|---:|---:|---|---:|
| 03161000, Jefferson | StreamStats, point-specific delineation at returned stream-grid snap | 531.144257 | +0.037046% | Station contained; snap displaced 22.96 m | 84 |
| 03164000, Galax | USGS GAGES-II, exact `gage_id=03164000` feature | 2,952.737446 | −0.082533% | Retained station coordinate 33.73 m outside polygon; no explicit outlet point in feature | 387 |
| 03479000, Sugar Grove | StreamStats, point-specific delineation at returned stream-grid snap | 236.074901 | −1.032542% | Station contained; snap displaced 29.15 m | 43 |

The table’s station coordinates and published drainage areas are the retained October 23 metadata snapshot, not newly retrieved observations. All three areas/outlet diagnostics now independently reproduce those retained results. The prior case-specific acceptance record accepted the measured Galax offset and Sugar Grove area difference. Those same issues are preserved as known caveats, not presented as newly discovered failures. That acceptance did not establish a generic tolerance, change the geometry, or approve all remaining spatial claims.

Boundary distances are planar EPSG:5070 diagnostics, not surveyed along-channel distances. The selected geographic datum operation reports 4 m accuracy. Retained station metadata describes Jefferson/Sugar Grove as GPS positions with ±1 arcsecond accuracy, and Galax as map-interpolated with unspecified accuracy. Source-location uncertainty is additional; close area agreement alone cannot verify an exact outlet.

The retained mixed-product comparison found **5.179115 km², or 0.975086% of Jefferson’s polygon, outside Galax**. This means the two selected polygons are not exactly nested. It is compatible with treating their predictors as separate gauge-associated inputs; it does not support basin differencing, a nested conservation calculation, or a claim that the products share an exact drainage boundary. Clipping Jefferson into Galax would change the study’s spatial support and is not performed.

## What is numerically verified

The three basins use the same frozen 29 × 33 HRRR crop (957 cells), native indices `i=1364…1392`, `j=478…510`, and row order `(j−478)×29+(i−1364)`. The native grid has 3,000 m spacing on its specified spherical Lambert projection. Cell corners lie 1,500 m from each verified center. Shared cell edges are sampled every 125 m, then transformed through native angular coordinates interpreted as WGS84 into EPSG:5070. Basin polygons are transformed into that same area coordinate system. Holes, valid empty StreamStats components, and original boundaries are preserved; no geometry is repaired.

For basin B and grid cell Cᵢ, the stored weight is:

`wᵢ = area(B ∩ Cᵢ) / area(B)`.

The denominator is the complete retained polygon area, not its published drainage-area number, the bounding rectangle, or only the currently observed cells. Zero intersections are omitted from the sparse CSV; those zeros mean no basin overlap and are not filled missing weather. All 514 rows pass exact index/formula/original-flag checks, and each basin’s weights sum to one within numerical precision.

The final audit found zero uncovered area for all three basins and exact agreement
of all 84/387/43 positive-cell IDs. Maximum absolute weight differences from the
original CSV were 1.55×10⁻¹⁴ (Jefferson), 2.91×10⁻¹⁵ (Galax) and 3.37×10⁻¹⁴
(Sugar Grove). Repeating cell-edge construction at 250 m instead of 125 m changed
each weight vector by less than 3×10⁻⁹ in L1 norm, within the original 10⁻⁶
convergence criterion. The complete numerical audit took 1.159 seconds with
PROJ networking disabled; it did not read weather/discharge values.

The Sugar Grove −1.03% discrepancy is a difference between a delineated polygon area and the published drainage-area number. It is **not 1.03% missing acquired HRRR coverage**. No weight denominator was adjusted to hide it.

## Missing cells and executed preprocessing

The executed function is `local_data.basin_reduce`. It computes coverage as `Σ wᵢ presentᵢ`, and admits the weighted mean only when **every cell with positive weight is present**. One missing positive-weight cell masks the result as NaN; it never divides by the remaining observed weight. A missing zero-weight cell has no effect. The smallest positive weights are approximately 7.40×10⁻⁷ (Jefferson), 7.37×10⁻⁸ (Galax), and 2.96×10⁻⁴ (Sugar Grove); even such a small missing contribution still fails the completeness mask. Coverage is preserved separately from the masked mean.

The generic `spatial.py` scaffold permits an explicitly selected partial-coverage renormalization option and retains an old proposal-oriented docstring. That option is not the executed development reduction. The current audit does not change either implementation. Its synthetic regression verifies the actual production reduction’s missing-positive versus missing-zero behavior.

## What the authoritative sources establish

The polygons already come from USGS services; replacing them merely to obtain an “official” source is unnecessary. USGS explains that StreamStats delineates from processed elevation, stream and watershed datasets and that users should inspect the resulting boundaries. An official automated delineation is not an accuracy guarantee. [StreamStats fundamentals](https://www.usgs.gov/streamstats/streamstats-fundamentals).

GAGES-II is a USGS dataset published in 2011 that includes basin boundary files linked to gauges. The retained Galax feature verifies its gauge ID and geometry, but lacks basin-specific outlet, delineation-method and DEM-vintage attributes. The dataset’s publication date does not supply those absent details. [USGS GAGES-II publication](https://www.usgs.gov/publications/gages-ii-geospatial-attributes-gages-evaluating-streamflow).

If a later scientific decision requires a consistent replacement delineation product, NLDI’s documented `simplified=false` and `splitCatchment=true` options are relevant: default basin requests use simplified whole catchments, while splitting uses the feature’s network reference. Such a replacement would still need its outlet and footprint checked. It is not automatically more accurate merely because it is newer or consistent across gauges. No new delineations are fetched here. [USGS NLDI basin and hydrolocation documentation](https://api.water.usgs.gov/docs/nldi/basin/).

## Remaining study decisions

1. Record whether the retained official-source approximations satisfy this study’s predictor-footprint claim, with the known outlet/method/vintage limitations and the absent provider Zarr CRS attribute disclosed.
2. Record whether Jefferson and Galax are retained as separate mixed-product footprints without a nesting or conservation claim. Their measured non-nesting is preserved.

The first decision does not require a new survey. The previous two case-specific discrepancy acceptances remain intact. Further geometry downloads would be justified by a decision to test or replace the spatial support, not by a need to repeat already verified arithmetic. If geometry or weights change, the derived basin caches and case-source identities must be rebuilt under new hashes; existing prepared cases must not be silently relabeled.

The audit has no forecasting-skill evidence, no protected data access, no weather/discharge payload reads, no geometry/weight edits, and no fits. It writes new reports only. All nine synthetic tests pass under the existing GIS runtime; three geometry tests are skipped in the smaller offline environment when Shapely is absent.
