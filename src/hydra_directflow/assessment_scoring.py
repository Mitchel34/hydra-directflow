"""Pure assessment point metrics for a separately authenticated scoring runner.

This module performs no I/O or model calls and cannot authorize assessment
access. Its caller authenticates all 36 committed prediction artifacts and the
prediction barrier before decoding verification values or calling score_gauge.
The caller also authenticates and aligns NWM issue/valid-time identities; array
position alone cannot establish issuance eligibility. H17/H18 are structurally
unavailable under the frozen short-range comparator and cutoff.
"""
from collections.abc import Mapping
import hashlib
import math

import numpy as np

from .assessment_contract import SEEDS
from .contract import ARCHITECTURES, GAUGE_IDS
from .reference import ASSESSMENT_START, ASSESSMENT_END, HOUR, require

SCHEMA = 'hydra-assessment-point-scores/v1'
N_ORIGINS = (ASSESSMENT_END - ASSESSMENT_START) // HOUR
LEADS = tuple(range(1, 19))
NWM_LEADS = tuple(range(1, 17))
METRICS = ('nse', 'physical_mse', 'rmse', 'mae', 'bias', 'negative_frequency',
           'pbias', 'pearson_r')
MIN_HOURS, MIN_DATES = 168, 28


def _sha_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _support(future, mask):
    selected = future[mask]
    return dict(forecast_target_pairs=int(mask.sum()),
                distinct_target_hours=int(np.unique(selected).size),
                distinct_utc_dates=int(np.unique(selected // (24 * HOUR)).size))


def _metric_row(truth, prediction, target_time, selected, lead, *, supported=True):
    support = _support(target_time, selected)
    n = support['forecast_target_pairs']
    row = dict(lead_hours=lead, n=n, **support, metrics_units='PHYSICAL_Q_M3_PER_SECOND',
               support_mask_sha256=_sha_array(selected),
               scored_target_times_sha256=_sha_array(target_time[selected]),
               negative_predictions_scored=int((prediction[selected] < 0).sum()),
               **{name: None for name in METRICS})
    undefined = {}
    if not supported:
        require(n == 0, 'Unsupported NWM leads cannot have score support')
        undefined = {name: 'STRUCTURALLY_UNAVAILABLE_NWM_SHORT_RANGE_LEAD' for name in METRICS}
    elif not n:
        undefined = {name: 'NO_SCORED_PAIRS' for name in METRICS}
    else:
        y = truth[selected].astype(np.float64, copy=False)
        p = prediction[selected].astype(np.float64, copy=False)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            error = p - y
            mse = float(np.mean(error * error))
            y_centered, p_centered = y - np.mean(y), p - np.mean(p)
            y_ss, p_ss = float(np.dot(y_centered, y_centered)), float(np.dot(p_centered, p_centered))
            total_y = float(np.sum(y))
            require(all(math.isfinite(v) for v in (mse, y_ss, p_ss, total_y)),
                    'Physical score overflow; invalid intermediate values cannot be dropped')
            row.update(physical_mse=mse, rmse=math.sqrt(mse), mae=float(np.mean(np.abs(error))),
                       bias=float(np.mean(error)), negative_frequency=row['negative_predictions_scored'] / n)
            if n >= 2 and y_ss > 0:
                row['nse'] = 1.0 - float(np.dot(error, error)) / y_ss
            else:
                undefined['nse'] = 'FEWER_THAN_TWO_TARGETS' if n < 2 else 'CONSTANT_OBSERVED_TARGETS'
            if total_y > 0:
                row['pbias'] = 100.0 * float(np.sum(error)) / total_y
            else:
                undefined['pbias'] = 'ZERO_OBSERVED_SUM'
            if n >= 2 and y_ss > 0 and p_ss > 0:
                # Sequential square roots avoid overflow of y_ss * p_ss.
                row['pearson_r'] = float(np.dot(y_centered, p_centered)) / math.sqrt(y_ss) / math.sqrt(p_ss)
                row['pearson_r'] = min(1.0, max(-1.0, row['pearson_r']))
            else:
                undefined['pearson_r'] = 'FEWER_THAN_TWO_PAIRS_OR_CONSTANT_SERIES'
        require(all(value is None or math.isfinite(value) for name, value in row.items() if name in METRICS),
                'Physical score overflow; invalid values cannot be dropped')
    reasons = []
    if not supported: reasons.append('STRUCTURALLY_UNAVAILABLE_NWM_SHORT_RANGE_LEAD')
    if support['distinct_target_hours'] < MIN_HOURS: reasons.append('TOO_FEW_DISTINCT_TARGET_HOURS')
    if support['distinct_utc_dates'] < MIN_DATES: reasons.append('TOO_FEW_DISTINCT_UTC_DATES')
    if row['nse'] is None: reasons.append(undefined.get('nse', 'UNDEFINED_NSE'))
    row.update(undefined=undefined, primary_eligible=not reasons,
               primary_ineligibility=list(dict.fromkeys(reasons)))
    return row


def _panel(truth, prediction, future, mask, *, leads=LEADS, structural_nwm=False):
    rows = [_metric_row(truth[:, j], prediction[:, j], future[:, j], mask[:, j], j + 1,
                        supported=not structural_nwm or j < 16) for j in range(18)]
    selected = [rows[h - 1] for h in leads]
    qualified = all(row['primary_eligible'] for row in selected)
    score = math.fsum(row['nse'] / len(leads) for row in selected) if qualified else None
    return dict(equal_lead_nse=score, equal_lead_nse_leads=list(leads),
                status='DEFINED' if qualified else 'UNAVAILABLE',
                equal_lead_nse_unavailable=[dict(lead_hours=r['lead_hours'], reasons=r['primary_ineligibility'])
                                          for r in selected if not r['primary_eligible']],
                support=_support(future, mask), support_mask_sha256=_sha_array(mask), leads=rows)


def _descriptive(values):
    if any(value is None for value in values):
        return dict(mean=None, min=None, max=None, defined_seeds=sum(v is not None for v in values),
                    total_seeds=len(SEEDS), undefined_seeds_omitted=False)
    return dict(mean=math.fsum(value / len(SEEDS) for value in values), min=min(values), max=max(values),
                defined_seeds=len(SEEDS), total_seeds=len(SEEDS), undefined_seeds_omitted=False)


def _aggregate(panels):
    require(set(panels) == set(SEEDS), 'All three seed panels required')
    ordered = [panels[s] for s in SEEDS]
    require(all(p['support_mask_sha256'] == ordered[0]['support_mask_sha256']
                and p['equal_lead_nse_leads'] == ordered[0]['equal_lead_nse_leads'] for p in ordered),
                'Seed comparison support differs')
    return dict(seeds=list(SEEDS), equal_lead_nse=_descriptive([p['equal_lead_nse'] for p in ordered]),
                equal_lead_nse_leads=ordered[0]['equal_lead_nse_leads'], support=ordered[0]['support'],
                support_mask_sha256=ordered[0]['support_mask_sha256'],
                leads=[dict(lead_hours=h, n=ordered[0]['leads'][h - 1]['n'],
                    **{name: _descriptive([p['leads'][h - 1][name] for p in ordered]) for name in METRICS}) for h in LEADS],
                interpretation='ARITHMETIC_MEAN_OF_SEED_METRICS_WITH_DESCRIPTIVE_MIN_MAX',
                seed_selection_performed=False, ensemble_predictions_scored=False,
                formal_confidence_intervals=False)


def _validate(gauge_id, origins, target, target_mask, input_admitted, predictions, nwm, nwm_mask):
    # Check the full fixed calendar before touching numerical outcome arrays.
    require(gauge_id in GAUGE_IDS, 'Confirmed assessment gauge required')
    origin = np.asarray(origins)
    require(origin.shape == (N_ORIGINS,) and origin.dtype.kind in 'iu'
            and np.array_equal(origin, np.arange(ASSESSMENT_START, ASSESSMENT_END, HOUR)),
            'Exact full 8760-origin assessment calendar required')
    expected = {(a, s) for a in ARCHITECTURES for s in SEEDS}
    require(isinstance(predictions, Mapping) and set(predictions) == expected,
            'All four architectures and all three seeds are required')
    shape = (N_ORIGINS, 18)
    truth, mask, admitted, comparator, available = map(np.asarray, (target, target_mask, input_admitted, nwm, nwm_mask))
    require(truth.shape == mask.shape == comparator.shape == available.shape == shape
            and admitted.shape == (N_ORIGINS,) and truth.dtype.kind == comparator.dtype.kind == 'f'
            and all(x.dtype == np.dtype(bool) for x in (mask, admitted, available)),
            'Exact typed assessment values/mask geometry required')
    require(np.isfinite(truth[mask]).all() and (truth[mask] >= 0).all(),
            'Admitted evaluation targets must be finite nonnegative physical Q')
    require(not available[:, 16:].any() and np.isnan(comparator[:, 16:]).all(),
            'NWM H17/H18 are structurally unavailable, with false masks and NaN values')
    require(np.isfinite(comparator[available]).all(), 'NWM available mask cannot hide missing/nonfinite forecasts')
    checked = {}
    for key in sorted(expected):
        p = np.asarray(predictions[key])
        require(p.shape == shape and p.dtype.kind == 'f' and np.isfinite(p[admitted]).all(),
                'Missing/nonfinite predictions on any input-admitted origin fail; no cohort shrinking')
        checked[key] = p
    future = origin[:, None] + np.arange(1, 19)[None, :] * HOUR
    # Repeated target hours represent the same observation, not independent
    # outcomes. Require identical target-quality decisions and valid values.
    for j in range(1, 18):
        left, right = mask[1:, j - 1], mask[:-1, j]
        require(np.array_equal(left, right), 'Repeated target-hour quality masks disagree')
        require(np.array_equal(truth[1:, j - 1][left], truth[:-1, j][right]),
                'Repeated valid target-hour values disagree')
    return origin, future, truth, mask, admitted, checked, comparator, available


def score_gauge(gauge_id, origins, target, target_mask, input_admitted, predictions, nwm, nwm_mask):
    """Score one fixed gauge after the caller authenticates the prediction barrier.

    Primary: input-admitted origins with all 18 valid targets. Secondary: each
    lead's independently valid target, without requiring other leads. NWM:
    strict primary origin support intersected with genuinely available NWM at
    the exact same lead/valid time, and a separate equal-lead mean over H1–16.
    Forecast eligibility never depends on target or NWM availability.
    """
    origin, future, truth, valid, admitted, predicted, comparator, available = _validate(
        gauge_id, origins, target, target_mask, input_admitted, predictions, nwm, nwm_mask)
    strict = admitted & valid.all(axis=1)
    primary = np.broadcast_to(strict[:, None], valid.shape)
    secondary = admitted[:, None] & valid
    matched = primary & available
    nwm_panel = _panel(truth, comparator, future, matched, leads=NWM_LEADS, structural_nwm=True)
    models = {}
    for architecture in ARCHITECTURES:
        seeds = {}
        for seed in SEEDS:
            p = predicted[(architecture, seed)]
            seeds[str(seed)] = dict(primary=_panel(truth, p, future, primary),
                per_lead_valid_target=_panel(truth, p, future, secondary),
                nwm_matched=_panel(truth, p, future, matched, leads=NWM_LEADS, structural_nwm=True))
        aggregates = {name: _aggregate({seed: seeds[str(seed)][name] for seed in SEEDS})
                      for name in ('primary', 'per_lead_valid_target', 'nwm_matched')}
        models[architecture] = dict(seeds=seeds, seed_aggregation=aggregates)
    coverage = dict(origin_count=N_ORIGINS, input_admitted_origins=int(admitted.sum()),
        input_excluded_origins=int((~admitted).sum()), primary_complete_target_origins=int(strict.sum()),
        input_admitted_with_incomplete_targets=int((admitted & ~valid.all(axis=1)).sum()),
        forecast_availability_fraction=float(admitted.mean()),
        primary_support=_support(future, primary), independent_per_lead_support=_support(future, secondary),
        matched_nwm_support=_support(future, matched),
        per_lead=[dict(lead_hours=h, forecast_available_origins=int(admitted.sum()),
            valid_observed_target_origins=int(valid[:, h - 1].sum()),
            evaluable_origins=int(secondary[:, h - 1].sum()),
            evaluable_fraction_all_origins=float(secondary[:, h - 1].mean()),
            evaluable_fraction_valid_observed_targets=(float(secondary[:, h - 1].sum() / valid[:, h - 1].sum())
                if valid[:, h - 1].any() else None),
            strict_primary_origins=int(strict.sum()), nwm_matched_origins=int(matched[:, h - 1].sum()),
            distinct_evaluable_target_hours=int(np.unique(future[secondary[:, h - 1], h - 1]).size)) for h in LEADS])
    return dict(schema=SCHEMA, gauge_id=gauge_id, units='m3/s',
        first_origin_epoch=int(origin[0]), last_origin_epoch=int(origin[-1]),
        last_target_epoch=int(future[-1, -1]), coverage=coverage, models=models,
        nwm=dict(product='SHORT_RANGE_ONLY', supported_lead_hours=list(NWM_LEADS),
                 structurally_unavailable_lead_hours=[17, 18], matched=nwm_panel),
        policy=dict(primary='STRICT_INPUT_ADMITTED_AND_ALL_18_TARGETS_VALID',
            secondary='INDEPENDENT_PER_LEAD_VALID_TARGETS_ON_INPUT_ADMITTED_ORIGINS',
            nwm='SAME_STRICT_PRIMARY_HYDRA_ORIGINS_AND_EXACT_LEAD_NWM_AVAILABILITY',
            primary_metric='EQUAL_LEAD_PHYSICAL_NSE_H1_H18', paired_metric='EQUAL_LEAD_PHYSICAL_NSE_H1_H16',
            minimum_distinct_target_hours=MIN_HOURS, minimum_distinct_utc_dates=MIN_DATES,
            pbias_sign='100 * SUM(PREDICTION_MINUS_OBSERVATION) / SUM(OBSERVATION)',
            negative_predictions_clipped=False, targets_imputed=False,
            missing_predictions='FAIL_ALL_INPUT_ADMITTED_CASES', undefined_leads_dropped=False,
            seed_selection_performed=False, ensemble_predictions_scored=False, formal_confidence_intervals=False,
            support_qualification='Reporting threshold only, not independence or statistical power',
            overlapping_forecasts_are_independent_events=False,
            training_loss='Standardized-Q training MSE differs from reported physical-unit MSE'),
        input_admission_sha256=_sha_array(admitted), target_mask_sha256=_sha_array(valid),
        forecast_coordinates_sha256=_sha_array(future),
        authorization='Caller must authenticate committed all-36 prediction barrier before verification reads; this pure function grants no access')
