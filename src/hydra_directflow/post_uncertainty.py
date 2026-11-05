"""Pure, bounded mechanics for paired development bootstrap postprocessing.

No file readers, model calls, fitting, protected scoring or protocol adoption.
The caller authenticates the saved predictions/labels and binds the execution
to the frozen protocol. Missing calendar rows remain present with false masks.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import math
from typing import Mapping

import numpy as np

from .campaign_ledger import ARCHITECTURES, GAUGES
from .reference import digest, require
from .scoring import ScoringPolicy
from .uncertainty import (BLOCK_HOURS, MAX_ORIGINS, MAX_REPLICATES,
                          draw_origin_indices, validate_origin_calendar)

FOLDS = ('cv_1', 'cv_2', 'cv_3', 'final_development')
SEEDS = (17, 29, 43)
MASTER_SEED = 20261102


def split_seed(block_hours, fold, *, master_seed=MASTER_SEED):
    """Fixed 7/14/28-day x CV1/CV2/CV3/final stream order, chosen before results.

Each child yields one uint64 seed for the existing PCG64 block sampler. Draws
within a fold are reused across gauges, models and seeds. Different folds use
different streams; pairing does not impose cross-year block-start dependence.
"""
    require(type(block_hours) is int and block_hours in BLOCK_HOURS and fold in FOLDS,
            'Declared block length and fold required')
    require(type(master_seed) is int and 0 <= master_seed < 2**64, 'Bounded master seed required')
    index = BLOCK_HOURS.index(block_hours) * len(FOLDS) + FOLDS.index(fold)
    child = np.random.SeedSequence(master_seed).spawn(12)[index]
    seed = int(child.generate_state(1, dtype=np.uint64)[0])
    return seed, dict(master_seed=master_seed, block_order_hours=list(BLOCK_HOURS),
                      fold_order=list(FOLDS), child_index=index, spawn_key=list(child.spawn_key),
                      derived_uint64_seed=seed, method='SeedSequence.spawn12.generate_state1.uint64')


@dataclass(frozen=True)
class DrawBundle:
    origins: np.ndarray
    indices: np.ndarray
    multiplicities: np.ndarray
    block_starts: np.ndarray
    block_lengths: np.ndarray
    receipt: dict


def make_draws(origins, *, start_utc, end_exclusive_utc, fold, block_hours=336,
               replicates=2000, master_seed=MASTER_SEED):
    """Construct exact existing noncircular draws plus efficient sufficient sums."""
    seed, stream = split_seed(block_hours, fold, master_seed=master_seed)
    origin = validate_origin_calendar(origins, start_utc=start_utc, end_exclusive_utc=end_exclusive_utc)
    indices, receipt = draw_origin_indices(origin, start_utc=start_utc,
        end_exclusive_utc=end_exclusive_utc, block_hours=block_hours, replicates=replicates, seed=seed)
    columns, lengths = [], []
    for segment in receipt['segments']:
        lo, hi = segment['start_index'], segment['end_index_exclusive']
        for cursor in range(lo, hi, block_hours):
            take = min(block_hours, hi - cursor)
            starts = indices[:, cursor]
            require(np.array_equal(indices[:, cursor:cursor+take],
                                   starts[:, None] + np.arange(take)[None, :]),
                    'Noncontiguous sampled block differs from declared sampler')
            columns.append(starts)
            lengths.append(take)
    counts = np.asarray([np.bincount(row, minlength=len(origin)) for row in indices], dtype=np.uint16)
    starts = np.stack(columns, axis=1).astype(np.int32)
    lengths = np.asarray(lengths, dtype=np.int32)
    for value in (origin, counts, starts, lengths):
        value.setflags(write=False)
    receipt = dict(receipt, stream_split=stream,
        indices_array_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        multiplicities_array_sha256=hashlib.sha256(counts.tobytes()).hexdigest(),
        block_starts_array_sha256=hashlib.sha256(starts.tobytes()).hexdigest(),
        block_lengths_array_sha256=hashlib.sha256(lengths.tobytes()).hexdigest(),
        weight_dtype=counts.dtype.str, original_coordinate_integer_multiplicities=True)
    return DrawBundle(origin, indices, counts, starts, lengths, receipt)


def _weights(value, n):
    weights = np.asarray(value)
    require(weights.ndim == 2 and 1 <= len(weights) <= MAX_REPLICATES and weights.shape[1] == n
            and weights.dtype.kind in 'iu' and np.all(weights <= n)
            and np.all(weights.sum(axis=1, dtype=np.uint64) == n),
            'Bounded nonnegative integer weights must retain every original calendar length')
    # Unsigned and signed kinds are accepted, but negative signed values are not.
    require(np.all(weights >= 0), 'Negative multiplicities are forbidden')
    return weights


def _prefix_sums(values, draws):
    """O(N*K + D*blocks*K), avoiding O(D*N*K) weighted dense multiplication."""
    shape = values.shape[1:]
    values = np.asarray(values, dtype=np.float64).reshape(len(draws.origins), -1)
    prefix = np.empty((len(values) + 1, values.shape[1]), dtype=np.float64)
    prefix[0] = 0.
    np.cumsum(values, axis=0, dtype=np.float64, out=prefix[1:])
    result = np.zeros((len(draws.indices), values.shape[1]), dtype=np.float64)
    for column, take in enumerate(draws.block_lengths):
        start = draws.block_starts[:, column]
        result += prefix[start + take] - prefix[start]
    require(np.isfinite(prefix).all() and np.isfinite(result).all(), 'Bootstrap arithmetic overflowed')
    return result.reshape((len(draws.indices),) + shape)


def _support(origins, admitted, weights):
    positive = weights > 0
    d, n = positive.shape
    hours = np.empty((d, 18), dtype=np.int32)
    dates = np.empty_like(hours)
    baseline_dates = np.empty(18, dtype=np.int32)
    for lead in range(18):
        present = positive & admitted[None, :, lead]
        hours[:, lead] = present.sum(axis=1, dtype=np.int32)
        day = (origins + (lead + 1) * 3600) // 86400
        boundaries = np.r_[0, np.flatnonzero(np.diff(day)) + 1]
        dates[:, lead] = np.logical_or.reduceat(present, boundaries, axis=1).sum(axis=1)
        baseline_dates[lead] = np.logical_or.reduceat(admitted[:, lead], boundaries).sum()
    return hours, dates, baseline_dates


def _score(origins, labels, label_mask, predictions, prediction_mask, comparison_mask,
           expected_comparison_mask_sha256, weights, weighted_sum, policy):
    origin = np.asarray(origins)
    n = len(origin)
    require(isinstance(policy, ScoringPolicy), 'Explicit scoring policy required')
    require(origin.ndim == 1 and origin.dtype.kind in 'iu' and 0 < n <= MAX_ORIGINS,
            'Bounded original origin coordinates required')
    truth, prediction = np.asarray(labels), np.asarray(predictions)
    mask, present, comparison_original = map(np.asarray, (label_mask, prediction_mask, comparison_mask))
    require(comparison_original.shape in ((n,), (n, 18)) and comparison_original.dtype == np.dtype(bool)
            and digest(comparison_original.tolist()) == expected_comparison_mask_sha256,
            'Original comparison mask identity differs')
    comparison = np.broadcast_to(comparison_original[:, None], (n, 18)) if comparison_original.ndim == 1 else comparison_original
    require(truth.shape == (n, 18) and truth.dtype.kind == 'f' and mask.shape == truth.shape
            and mask.dtype == np.dtype(bool) and prediction.ndim == 3 and 1 <= len(prediction) <= 36
            and prediction.shape[1:] == truth.shape and prediction.dtype.kind == 'f'
            and present.shape == prediction.shape and present.dtype == np.dtype(bool),
            'Matched physical labels/predictions and explicit masks required')
    require(np.isfinite(truth[mask]).all() and np.all(truth[mask] >= 0),
            'Admitted labels must be finite and nonnegative')
    require(present[:, comparison].all() and np.isfinite(prediction[:, comparison]).all(),
            'Requested missing or nonfinite predictions fail before drawing')
    admitted = mask & comparison
    weights = _weights(weights, n)
    hours, dates, baseline_dates = _support(origin, admitted, weights)
    base_hours = admitted.sum(axis=0)
    truth64 = np.where(admitted, truth, 0.).astype(np.float64)
    center = np.divide(truth64.sum(axis=0), base_hours, out=np.zeros(18), where=base_hours > 0)
    z = np.where(admitted, truth64 - center, 0.)
    base_denominator = np.sum(z * z, axis=0)
    base_variation = np.array([base_hours[i] > 0 and np.ptp(truth[admitted[:, i], i]) > 0 for i in range(18)])
    baseline_eligible = ((base_hours >= policy.minimum_distinct_target_hours)
                         & (baseline_dates >= policy.minimum_distinct_utc_dates)
                         & base_variation & (base_denominator > 0))
    mass = weighted_sum(admitted.astype(np.float64))
    sum_z = weighted_sum(z)
    sum_z2 = weighted_sum(z * z)
    correction = np.divide(sum_z * sum_z, mass, out=np.zeros_like(sum_z), where=mass > 0)
    denominator = sum_z2 - correction
    # Centering greatly reduces cancellation. Degenerate/nearly constant draws
    # are recomputed by the two-pass original-coordinate weighted definition.
    unstable = denominator <= np.finfo(np.float64).eps * 64 * (abs(sum_z2) + abs(correction))
    for row, lead in zip(*np.nonzero(unstable & (mass > 0))):
        selected = admitted[:, lead] & (weights[row] > 0)
        y, w = truth[selected, lead], weights[row, selected]
        if np.all(y == y[0]):
            denominator[row, lead] = 0.
        else:
            mean_y = math.fsum(float(v) * int(m) / int(mass[row, lead]) for v, m in zip(y, w))
            denominator[row, lead] = math.fsum(int(m) * (float(v) - mean_y)**2 for v, m in zip(y, w))
    # Every scored model uses the same label/support geometry. All models are
    # validated before their unavailable calendar cells receive local zeros.
    error = np.where(admitted[None, :, :], prediction.astype(np.float64) - truth64[None, :, :], 0.)
    square = error * error
    require(np.isfinite(square).all() and np.isfinite(base_denominator).all()
            and np.isfinite(denominator).all(), 'Bootstrap arithmetic exceeded finite range')
    sse = weighted_sum(square.transpose(1, 0, 2))
    base_sse = square.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        nse = 1. - sse / denominator[:, None, :]
        baseline_nse = 1. - base_sse / base_denominator[None, :]
    lead_valid = ((hours >= policy.minimum_distinct_target_hours)
                  & (dates >= policy.minimum_distinct_utc_dates) & (mass > 0)
                  & (denominator > 0) & baseline_eligible[None, :])
    nse = np.where(lead_valid[:, None, :], nse, np.nan)
    baseline_nse = np.where(baseline_eligible[None, :], baseline_nse, np.nan)
    require(not np.isinf(nse).any() and not np.isinf(baseline_nse).any(), 'NSE overflowed')
    valid = np.broadcast_to(lead_valid.all(axis=1)[:, None], nse.shape[:2]).copy()
    primary = np.where(valid, np.sum(nse / 18., axis=2), np.nan)
    baseline_primary = np.sum(baseline_nse / 18., axis=1) if baseline_eligible.all() else np.full(len(prediction), np.nan)
    require(not np.isinf(primary).any() and not np.isinf(baseline_primary).any(), 'Primary NSE overflowed')
    return dict(primary=primary, per_lead_nse=nse, valid=valid, baseline_primary=baseline_primary,
                baseline_per_lead_nse=baseline_nse, distinct_target_hours=hours, distinct_utc_dates=dates,
                weighted_label_count=mass.astype(np.int64), baseline_lead_eligible=baseline_eligible,
                baseline_distinct_hours=base_hours, baseline_distinct_dates=baseline_dates,
                lead_valid=lead_valid, undefined_or_zero_variance=(denominator <= 0),
                baseline_unavailable=not bool(baseline_eligible.all()),
                original_coordinates_preserved=True, negative_predictions_clipped=False,
                model_calls=0, optimizer_updates=0, protected_assessment_scores=0)


def evaluate_matched(draws, *, labels, label_mask, predictions, prediction_mask,
                     comparison_mask, expected_comparison_mask_sha256, policy=ScoringPolicy()):
    """Return D×P primary and D×P×18 NSE arrays for one matched gauge/fold.

P is up to36 panels (4 models×3 arms×3 seeds); labels/masks are shared N×18.
Comparison mask is N or N×18. Outside-comparison values may be NaN and are never
treated as observations. The caller reuses the SAME DrawBundle for every gauge.
"""
    require(isinstance(draws, DrawBundle), 'Authenticated shared draw bundle required')
    validated = validate_origin_calendar(draws.origins,
        start_utc=draws.receipt['calendar_start_utc'], end_exclusive_utc=draws.receipt['calendar_end_exclusive_utc'])
    # The caller binds the receipt; this protects accidental in-memory edits.
    require(digest(validated.tolist()) == draws.receipt['calendar_sha256']
            and hashlib.sha256(draws.indices.tobytes()).hexdigest() == draws.receipt['indices_array_sha256']
            and hashlib.sha256(draws.multiplicities.tobytes()).hexdigest() == draws.receipt['multiplicities_array_sha256']
            and hashlib.sha256(draws.block_starts.tobytes()).hexdigest() == draws.receipt['block_starts_array_sha256']
            and hashlib.sha256(draws.block_lengths.tobytes()).hexdigest() == draws.receipt['block_lengths_array_sha256'],
            'Shared draw bundle changed')
    result = _score(draws.origins, labels, label_mask, predictions, prediction_mask, comparison_mask,
                    expected_comparison_mask_sha256, draws.multiplicities,
                    lambda v: _prefix_sums(v, draws), policy)
    result['draws_receipt'] = dict(draws.receipt)
    result['comparison_mask_sha256'] = expected_comparison_mask_sha256
    return result


def evaluate_weights(origins, *, start_utc, end_exclusive_utc, multiplicities,
                     labels, label_mask, predictions, prediction_mask, comparison_mask,
                     expected_comparison_mask_sha256, policy=ScoringPolicy()):
    """Independent dense-sum route for bounded synthetic/reference qualification."""
    origin = validate_origin_calendar(origins, start_utc=start_utc, end_exclusive_utc=end_exclusive_utc)
    weights = _weights(multiplicities, len(origin))
    def sum_values(v):
        value = np.asarray(v, dtype=np.float64)
        return (weights.astype(np.float64) @ value.reshape(len(origin), -1)).reshape((len(weights),) + value.shape[1:])
    return _score(origin, labels, label_mask, predictions, prediction_mask, comparison_mask,
                  expected_comparison_mask_sha256, weights, sum_values, policy)


def primary_family(draw_scores: Mapping, point_scores: Mapping, *, mode,
                   max_joint_invalid=100, percentiles=(2.5, 97.5)):
    """Exactly42 final or30 CV contrasts with one joint-valid draw pool.

Keys: (fold, gauge, architecture, input_arm, history_hours, seed).
Values: one D-vector of primary NSE (NaN means unavailable) and its baseline.
CV averages all3 fold metrics per replicate; final averages all3 seed metrics.
There is no ensemble-prediction metric, omission, redraw or favorable seed.
"""
    require(mode in ('cv', 'final') and tuple(percentiles) == (2.5, 97.5)
            and type(max_joint_invalid) is int and max_joint_invalid == 100, 'Declared family policy required')
    folds = FOLDS[:3] if mode == 'cv' else FOLDS[3:]
    seeds = (17,) if mode == 'cv' else SEEDS
    arms = (('full', 168), ('q_only', 168)) if mode == 'cv' else (('full', 168), ('q_only', 168), ('full', 24))
    expected = {(f, g, a, arm, hours, seed) for f in folds for g in GAUGES for a in ARCHITECTURES
                for arm, hours in arms for seed in seeds}
    require(set(draw_scores) == set(point_scores) == expected, 'Complete declared family panels required; no omission')
    ordered = sorted(expected)
    arrays = [np.asarray(draw_scores[key], dtype=np.float64) for key in ordered]
    d = len(arrays[0])
    require(d == MAX_REPLICATES and all(x.shape == (d,) and not np.isinf(x).any() for x in arrays),
            'Exactly2000 common draws and finite or unavailable scores required')
    points = np.asarray([np.nan if point_scores[key] is None else point_scores[key] for key in ordered], dtype=float)
    require(not np.isinf(points).any(), 'Point scores must be finite or unavailable')
    joint = np.isfinite(np.stack(arrays)).all(axis=0)
    aggregated, base = {}, {}
    for gauge in GAUGES:
        for architecture in ARCHITECTURES:
            for arm, hours in arms:
                keys = [(f, gauge, architecture, arm, hours, seed) for f in folds for seed in seeds]
                aggregated[gauge, architecture, arm, hours] = np.sum(
                    [np.asarray(draw_scores[k], dtype=float) / len(keys) for k in keys], axis=0)
                base[gauge, architecture, arm, hours] = float(math.fsum(
                    (np.nan if point_scores[k] is None else float(point_scores[k])) / len(keys) for k in keys))
    specs = []
    for gauge in GAUGES:
        for architecture in ARCHITECTURES:
            specs.append(('full_minus_q', gauge, (gauge, architecture, 'full', 168), (gauge, architecture, 'q_only', 168)))
        if mode == 'final':
            for architecture in ARCHITECTURES:
                specs.append(('L168_minus_L24', gauge, (gauge, architecture, 'full', 168), (gauge, architecture, 'full', 24)))
        for left, right in combinations(ARCHITECTURES, 2):
            specs.append(('architecture_first_minus_second', gauge, (gauge, left, 'full', 168), (gauge, right, 'full', 168)))
    effects = np.column_stack([aggregated[left] - aggregated[right] for _, _, left, right in specs])
    joint &= np.isfinite(effects).all(axis=1)
    invalid = int((~joint).sum())
    baseline_ok = bool(np.isfinite(points).all())
    available = baseline_ok and invalid <= max_joint_invalid and bool(joint.any())
    intervals = np.percentile(effects[joint], percentiles, axis=0, method='linear') if available else None
    rows = []
    for index, (kind, gauge, left, right) in enumerate(specs):
        point = base[left] - base[right]
        rows.append(dict(kind=kind, gauge_id=gauge, positive_panel=list(left), negative_panel=list(right),
                         point_difference=float(point) if math.isfinite(point) else None,
                         lower=None if intervals is None else float(intervals[0, index]),
                         upper=None if intervals is None else float(intervals[1, index])))
    return dict(schema='hydra-development-paired-primary-contrast-family/v1', mode=mode,
                status='AVAILABLE_CONDITIONAL_POINTWISE' if available else 'FAMILY_INTERVALS_UNAVAILABLE',
                contrasts=rows, contrast_count=len(rows), required_panel_count=len(expected),
                replicates=d, jointly_valid_draws=int(joint.sum()), jointly_invalid_draws=invalid,
                invalid_draws_by_panel={str(key):int((~np.isfinite(value)).sum()) for key,value in zip(ordered,arrays)},
                maximum_jointly_invalid=max_joint_invalid, baseline_primary_available=baseline_ok,
                joint_valid_mask=joint, contrast_draws=effects, percentile_method='linear',
                percentiles=list(percentiles), paired_draw_pool_shared=True, invalid_draws_redrawn=False,
                mean_metrics_not_ensemble_predictions=True, seed_dispersion_is_interval=False,
                simultaneous_familywise_claim=False, interpretation='CONDITIONAL_WITHIN_YEAR_POINTWISE_EXPLORATORY',
                model_calls=0, optimizer_updates=0, protected_assessment_scores=0)
