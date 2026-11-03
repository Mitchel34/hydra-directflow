"""PROPOSED within-year calendar resampling and weighted development metrics.

These pure utilities produce no study intervals, forecasts, fits or approvals.
Draw indices ONCE and reuse them across models, gauges, seeds and all H18 values.
The declared interval is the complete H18-eligible origin calendar after the
explicit end-boundary purge; retain every INTERNAL unavailable row with a false
comparison mask. Never compact that calendar to just successful forecasts.

Noncircular seasonal moving blocks preserve segment lengths but have unequal
edge inclusion probabilities. Repeated original hours are multiplicities, not
new independent observations. These are conditional within-year sampling tools,
not interannual uncertainty or calibrated forecast prediction intervals. Joint
contrast validity, percentile level and any actual execution are separate policy.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import math

import numpy as np

from .reference import ASSESSMENT_START, digest, require
from .scoring import ScoringPolicy, ScoreIdentity, LEADS, score_physical_h18, validate_score_coordinates

HOUR = 3600
PRIMARY_BLOCK_HOURS = 336
BLOCK_HOURS = (168, 336, 672)
MAX_ORIGINS = 8784
MAX_REPLICATES = 2000


def _epoch(text):
    require(isinstance(text, str), "Explicit UTC interval string required")
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    require(value.tzinfo is not None and value.utcoffset().total_seconds() == 0
            and value.minute == value.second == value.microsecond == 0, "Exact UTC hour required")
    return int(value.timestamp())


def validate_origin_calendar(origins, *, start_utc, end_exclusive_utc):
    """Require every declared hourly origin, including unavailable forecasts."""
    start, end = _epoch(start_utc), _epoch(end_exclusive_utc)
    origin = np.asarray(origins)
    require(0 <= start < end and (end - start) % HOUR == 0 and 0 < (end - start) // HOUR <= MAX_ORIGINS,
            "Explicit bounded within-year origin interval required")
    require(end - HOUR + 18 * HOUR < ASSESSMENT_START, "Protected or crossing forecast calendar is forbidden")
    require(origin.ndim == 1 and origin.dtype.kind in "iu"
            and np.array_equal(origin, np.arange(start, end, HOUR, dtype=np.int64)),
            "Complete hourly origin calendar required; no compacted, gapped, duplicated or reordered rows")
    return origin.astype(np.int64, copy=True)


def _season(epoch):
    month = datetime.fromtimestamp(int(epoch), timezone.utc).month
    return "DJF" if month in (12, 1, 2) else "MAM" if month in (3, 4, 5) else "JJA" if month in (6, 7, 8) else "SON"


def contiguous_season_segments(origins, *, start_utc, end_exclusive_utc):
    """DJF spans Dec/Jan; separated start/end JJA fragments stay separate."""
    origin = validate_origin_calendar(origins, start_utc=start_utc, end_exclusive_utc=end_exclusive_utc)
    labels = [_season(value) for value in origin]
    boundaries = [0] + [i for i in range(1, len(origin)) if labels[i] != labels[i - 1]] + [len(origin)]
    return [dict(segment_id=i, season_utc=labels[lo], start_index=lo, end_index_exclusive=hi, hours=hi-lo,
                 start_epoch=int(origin[lo]), end_exclusive_epoch=int(origin[hi-1])+HOUR)
            for i, (lo, hi) in enumerate(zip(boundaries, boundaries[1:]))]


def draw_origin_indices(origins, *, start_utc, end_exclusive_utc, block_hours=PRIMARY_BLOCK_HOURS,
                        replicates=2000, seed=20261102):
    """One immutable index matrix for all paired outputs; never generate per model.

    Full blocks start uniformly at every valid hourly noncircular start inside
    a contiguous segment. Only the FINAL drawn block of a segment is truncated
    to fill its original length. No short segment is silently shrunk or joined.
    """
    require(type(block_hours) is int and block_hours in BLOCK_HOURS, "Proposed block length must be 168, 336 or 672 hours")
    require(type(replicates) is int and 1 <= replicates <= MAX_REPLICATES
            and type(seed) is int and 0 <= seed < 2**64, "Explicit bounded replicate count and RNG seed required")
    origin = validate_origin_calendar(origins, start_utc=start_utc, end_exclusive_utc=end_exclusive_utc)
    segments = contiguous_season_segments(origin, start_utc=start_utc, end_exclusive_utc=end_exclusive_utc)
    require(all(s["hours"] >= block_hours for s in segments), "Season segment shorter than block; no silent shrink or concatenation")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = np.empty((replicates, len(origin)), dtype=np.int32)
    for row in indices:
        for segment in segments:
            lo, hi = segment["start_index"], segment["end_index_exclusive"]
            cursor = lo
            while cursor < hi:
                start = int(rng.integers(lo, hi - block_hours + 1))
                take = min(block_hours, hi-cursor)
                row[cursor:cursor+take] = np.arange(start, start+take, dtype=np.int32)
                cursor += take
    indices.setflags(write=False)
    return indices, dict(schema="hydra-proposed-seasonal-block-draws/v1", policy_status="PROPOSED_NOT_FROZEN",
        calendar_start_utc=start_utc, calendar_end_exclusive_utc=end_exclusive_utc,
        calendar_sha256=digest(origin.tolist()), origin_count=len(origin), block_hours=block_hours,
        replicates=replicates, seed=seed, generator="NumPy.PCG64", numpy_version=np.__version__,
        segments=segments, circular=False, segment_lengths_preserved=True, final_block_only_truncated=True,
        paired_axes="REUSE_SAME_ORIGIN_INDICES_FOR_ALL_MODELS_GAUGES_SEEDS_AND_18_LEADS",
        interpretation="CONDITIONAL_WITHIN_YEAR_FIXED_MODELS_AND_AVAILABLE_SUPPORT_NOT_INTERANNUAL_CI",
        unequal_edge_inclusion_probabilities=True, study_results_computed=False)


def origin_multiplicities(indices, *, origin_count):
    """Compress one draw to integer weights on unchanged original coordinates."""
    value = np.asarray(indices)
    require(type(origin_count) is int and 0 < origin_count <= MAX_ORIGINS and value.shape == (origin_count,)
            and value.dtype.kind in "iu" and np.all((value >= 0) & (value < origin_count)), "Bounded one-origin-draw indices required")
    counts = np.bincount(value.astype(np.int64), minlength=origin_count).astype(np.int64)
    counts.setflags(write=False)
    return counts


def _weighted_lead(truth, predicted, admitted, weights, target_time, policy):
    positive = admitted & (weights > 0)
    y, p, w = truth[positive], predicted[positive], weights[positive]
    distinct = len(y)
    dates = len(set((target_time[positive] // 86400).tolist()))
    mass = int(w.sum())
    metrics = {name: None for name in ("physical_mse", "rmse", "mae", "bias", "nse", "negative_frequency")}
    undefined = {}
    if mass:
        errors = [float(pv) - float(yv) for pv, yv in zip(p, y)]
        sse = math.fsum(int(wv) * error * error for wv, error in zip(w, errors))
        mse = sse / mass
        mean = math.fsum(int(wv) * float(yv) / mass for wv, yv in zip(w, y))
        denominator = math.fsum(int(wv) * (float(yv) - mean)**2 for wv, yv in zip(w, y))
        metrics.update(physical_mse=mse, rmse=math.sqrt(mse),
            mae=math.fsum(int(wv)*abs(error)/mass for wv,error in zip(w,errors)),
            bias=math.fsum(int(wv)*error/mass for wv,error in zip(w,errors)),
            negative_frequency=sum(int(wv) for wv,pv in zip(w,p) if pv < 0)/mass)
        if all(float(value) == float(y[0]) for value in y): undefined["nse"] = "CONSTANT_REFERENCE"
        elif denominator == 0: undefined["nse"] = "REFERENCE_VARIANCE_NUMERIC_UNDERFLOW"
        else: metrics["nse"] = 1-sse/denominator
    else:
        undefined = {name: "ZERO_SUPPORT" for name in metrics}
    require(all(value is None or math.isfinite(value) for value in metrics.values()), "Weighted metric arithmetic exceeded finite range")
    reasons = []
    if distinct < policy.minimum_distinct_target_hours: reasons.append("TOO_FEW_DISTINCT_TARGET_HOURS")
    if dates < policy.minimum_distinct_utc_dates: reasons.append("TOO_FEW_DISTINCT_UTC_DATES")
    if metrics["nse"] is None: reasons.append(undefined["nse"])
    return dict(**metrics, undefined=undefined, weighted_label_count=mass, distinct_target_hours=distinct,
                distinct_utc_dates=dates, primary_eligible=not reasons, primary_ineligibility=reasons)


def weighted_physical_h18(*, case_ids, origins, forecast_time, start_utc, end_exclusive_utc,
                          labels, label_mask, predictions, prediction_mask, comparison_mask,
                          expected_comparison_mask_sha256, multiplicities, identity, input_cohort_digest, label_policy_digest, label_source_digest,
                          policy=ScoringPolicy(), prediction_units="m3/s", label_units="m3/s"):
    """Validate baseline before weighting; multiplicity cannot hide failed forecasts.

    comparison_mask is predeclared forecast eligibility, independent of labels,
    model errors or a bootstrap draw, bound to a separately supplied expected
    digest from the admission plan. The digest is an identity check, not approval
    authentication or proof of how that plan was selected. Every required prediction must be present
    and finite, even for unavailable labels or zero-weight origins. Calendar
    rows outside that mask remain present and contribute no score. Labels still
    require the original quality mask; no values are filled for scoring.

    The baseline and EACH replicate must satisfy the same original ScoringPolicy
    using distinct positive-weight original target hours/dates, never weighted
    sample count. This function returns no percentile interval or study result.
    """
    require(isinstance(identity, ScoreIdentity) and isinstance(policy, ScoringPolicy), "Explicit score identity/policy required")
    origin = validate_origin_calendar(origins, start_utc=start_utc, end_exclusive_utc=end_exclusive_utc)
    ids, _, future = validate_score_coordinates(case_ids=case_ids, origins=origin, forecast_time=forecast_time, gauge_id=identity.gauge_id)
    shape = (len(origin), 18)
    comparison = np.asarray(comparison_mask)
    require(comparison.shape == shape and comparison.dtype == np.dtype(bool), "Explicit original comparison mask required")
    comparison_sha = digest(comparison.tolist())
    require(expected_comparison_mask_sha256 == comparison_sha, "Comparison mask differs from predeclared admission identity")
    truth, predicted = np.asarray(labels), np.asarray(predictions)
    mask, present = np.asarray(label_mask), np.asarray(prediction_mask)
    require(truth.shape == predicted.shape == shape and truth.dtype.kind == predicted.dtype.kind == "f"
            and all(value.shape == shape and value.dtype == np.dtype(bool) for value in (mask, present, comparison)),
            "Original physical H18 values and explicit boolean masks required")
    require(prediction_units == label_units == "m3/s", "Physical discharge units required")
    require(np.isfinite(truth[mask]).all() and np.all(truth[mask] >= 0), "Admitted original labels must be finite/nonnegative")
    require(present[comparison].all() and np.isfinite(predicted[comparison]).all(),
            "Missing/nonfinite baseline predictions fail before weighting; no draw-dependent intersection")
    admitted = mask & comparison
    # Only unscored placeholder cells receive a zero in this validation copy;
    # original values/masks and the declared score cohort are never changed.
    baseline = score_physical_h18(case_ids=case_ids, origins=origin, forecast_time=future,
        labels=truth, label_mask=admitted, predictions=np.where(comparison, predicted, 0.), identity=identity,
        input_cohort_digest=input_cohort_digest, label_policy_digest=label_policy_digest,
        label_source_digest=label_source_digest, policy=policy, prediction_units=prediction_units, label_units=label_units)
    weights = np.asarray(multiplicities)
    require(weights.shape == origin.shape and weights.dtype.kind in "iu" and np.all(weights >= 0)
            and np.all(weights <= len(origin)) and sum(int(w) for w in weights) == len(origin),
            "Nonnegative integer origin multiplicities must retain the original calendar length")
    weights = weights.astype(np.int64)
    leads = [dict(lead_hours=lead, **_weighted_lead(truth[:, i], predicted[:, i], admitted[:, i], weights, future[:, i], policy))
             for i, lead in enumerate(LEADS)]
    qualified = baseline["primary_score"] is not None and all(row["primary_eligible"] for row in leads)
    return dict(schema="hydra-proposed-weighted-h18-score/v1", policy_status="PROPOSED_NOT_FROZEN", policy_sha256=policy.sha256,
        identity=asdict(identity), case_ids_sha256=digest(list(ids)), calendar_sha256=digest(origin.tolist()),
        input_cohort_digest=input_cohort_digest, label_policy_digest=label_policy_digest, label_source_digest=label_source_digest,
        comparison_mask_sha256=comparison_sha, multiplicities_sha256=digest(weights.tolist()),
        original_calendar_hours=len(origin), weighted_calendar_hours=int(weights.sum()),
        baseline_primary_score=baseline["primary_score"], baseline_primary_qualified=baseline["primary_score"] is not None,
        baseline_lead_support=[dict(lead_hours=row["lead_hours"],n=row["n"],distinct_target_hours=row["distinct_target_hours"],
            distinct_utc_dates=row["distinct_utc_dates"],primary_eligible=row["primary_eligible"]) for row in baseline["leads"]],
        original_label_mask_false_cells=int((~mask).sum()), original_comparison_mask_false_cells=int((~comparison).sum()),
        leads=leads, primary_metric="EQUAL_LEAD_PHYSICAL_NSE", primary_score=math.fsum(row["nse"]/18 for row in leads) if qualified else None,
        primary_unavailable_reasons=(["BASELINE_PRIMARY_UNAVAILABLE"] if baseline["primary_score"] is None else [])
            + [dict(lead_hours=row["lead_hours"],reasons=row["primary_ineligibility"]) for row in leads if not row["primary_eligible"]],
        original_coordinates_preserved=True, negative_predictions_clipped=False, undefined_leads_dropped=False,
        units={"physical_mse":"(m3/s)^2","rmse":"m3/s","mae":"m3/s","bias":"m3/s","nse":"dimensionless","negative_frequency":"fraction"},
        interpretation="CONDITIONAL_WITHIN_YEAR_FIXED_MODELS_AND_AVAILABLE_SUPPORT_NOT_INTERANNUAL_CI",
        percentile_interval_computed=False, model_calls=0, optimizer_updates=0, protected_assessment_scores=0)
