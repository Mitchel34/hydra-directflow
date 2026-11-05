#!/usr/bin/env python3
"""Descriptive, calendar-lag residual dependence for the frozen development fits.

Called only by the bounded postprocessing controller. No model calls, source
fetches, imputation, assessment, significance thresholds or block-length tuning.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np

from postprocess_primary_events import FOLDS, GAUGES, load_group
from hydra_directflow.reference import ASSESSMENT_START, require
from hydra_directflow.uncertainty import validate_origin_calendar

LAG_HOURS = (1, 6, 12, 24, 48, 72, 168, 336, 672)
LEADS = tuple(range(1, 19))


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def pairwise_lag_correlation(errors, admitted, lag_hours):
    """Pearson correlation on both valid endpoints separated by calendar hours.

    Arrays retain every original hourly position. A gap never turns two distant
    observations into adjacent residuals. Each endpoint series is centered on
    its own surviving-pair mean; the result is descriptive, with no significance
    or independent-sample interpretation.
    """
    value, mask = np.asarray(errors), np.asarray(admitted)
    require(value.ndim == 1 and value.dtype == np.dtype('<f8') and len(value) <= 8784
            and mask.shape == value.shape and mask.dtype == np.dtype(bool), 'Bounded hourly residual/mask vectors required')
    require(type(lag_hours) is int and lag_hours in LAG_HOURS, 'One declared residual lag required')
    require(np.isfinite(value[mask]).all(), 'Admitted residuals must be finite')
    if lag_hours >= len(value):
        selected = np.zeros(0, dtype=bool)
        earlier = later = np.empty(0, dtype=np.float64)
    else:
        selected = mask[:-lag_hours] & mask[lag_hours:]
        earlier, later = value[:-lag_hours][selected], value[lag_hours:][selected]
    count = int(selected.sum())
    base = dict(lag_hours=lag_hours, paired_original_hours=count, correlation=None)
    if count < 2:
        return dict(base, status='UNDEFINED', reason='ZERO_PAIRS' if count == 0 else 'FEWER_THAN_TWO_PAIRS')
    left_constant, right_constant = bool(np.all(earlier == earlier[0])), bool(np.all(later == later[0]))
    if left_constant or right_constant:
        reason = 'BOTH_ENDPOINTS_CONSTANT' if left_constant and right_constant else (
            'EARLIER_ENDPOINT_CONSTANT' if left_constant else 'LATER_ENDPOINT_CONSTANT')
        return dict(base, status='UNDEFINED', reason=reason)
    x, y = earlier-np.mean(earlier, dtype=np.float64), later-np.mean(later, dtype=np.float64)
    xx, yy, xy = (float(np.dot(a, b)) for a, b in ((x, x), (y, y), (x, y)))
    require(all(math.isfinite(x) for x in (xx, yy, xy)) and xx > 0 and yy > 0,
            'Residual covariance or variance is not finite and positive')
    # Divide by norms separately to avoid overflowing the product xx*yy.
    correlation = xy/math.sqrt(xx)/math.sqrt(yy)
    require(math.isfinite(correlation) and abs(correlation) <= 1+1e-12, 'Invalid residual correlation')
    base['correlation'] = max(-1., min(1., correlation))
    return dict(base, status='DEFINED_DESCRIPTIVE', reason=None)


def _lag_maxima(panels):
    output = []
    for lag in LAG_HOURS:
        rows = [item for panel in panels for lead in panel['leads'] for item in lead['lags']
                if item['lag_hours'] == lag]
        defined = [x for x in rows if x['correlation'] is not None]
        output.append(dict(lag_hours=lag, panel_lead_series=len(rows), defined_series=len(defined),
                           undefined_series=len(rows)-len(defined),
                           maximum_absolute_correlation=max((abs(x['correlation']) for x in defined), default=None),
                           minimum_pair_count=min((x['paired_original_hours'] for x in rows), default=0),
                           maximum_pair_count=max((x['paired_original_hours'] for x in rows), default=0)))
    return output


def diagnose_group(arrays, manifest, *, check=lambda: None):
    """All panel/lead diagnostics after authenticated loader validation."""
    require(callable(check), 'Resource check callback required')
    origins = validate_origin_calendar(arrays['origins'], start_utc=manifest['calendar_start_utc'],
                                       end_exclusive_utc=manifest['calendar_end_exclusive_utc'])
    future = np.asarray(arrays['forecast_time'])
    require(future.shape == (len(origins), 18) and future.dtype.kind in 'iu'
            and np.array_equal(future, origins[:, None]+np.arange(1,19)[None, :]*3600)
            and np.all(future < ASSESSMENT_START), 'Exact development residual target clocks required')
    # Clocks are checked before touching numerical outcomes.
    truth, mask, common, predictions = map(np.asarray, (arrays['labels'], arrays['label_mask'],
                                                       arrays['comparison_mask'], arrays['predictions']))
    panels = manifest['panels']; expected_panels = 36 if manifest['fold'] == 'final_development' else 8
    require(manifest['fold'] in FOLDS and manifest['gauge_id'] in GAUGES
            and len(panels) == expected_panels and len({x['job_id'] for x in panels}) == expected_panels
            and np.array_equal(arrays['panel_ids'], np.array([x['job_id'] for x in panels])),
            'Complete, unique residual panel identities required')
    require(truth.shape == mask.shape == future.shape and truth.dtype == np.dtype('<f8')
            and mask.dtype == np.dtype(bool) and common.shape == origins.shape and common.dtype == np.dtype(bool)
            and np.array_equal(mask, np.broadcast_to(common[:,None], truth.shape))
            and predictions.shape == (expected_panels, len(origins), 18) and predictions.dtype == np.dtype('<f8'),
            'Exact common H18 residual geometry required')
    require(np.isnan(truth[~common]).all() and np.isnan(predictions[:,~common]).all()
            and np.isfinite(truth[common]).all() and np.isfinite(predictions[:,common]).all(),
            'Residuals must preserve admitted values and unavailable calendar gaps')
    rows = []
    for index, identity in enumerate(panels):
        check()
        error = predictions[index]-truth
        require(np.isfinite(error[common]).all(), 'Physical residual overflow')
        leads = []
        for column, lead in enumerate(LEADS):
            check()
            correlations = [pairwise_lag_correlation(error[:,column], mask[:,column], lag) for lag in LAG_HOURS]
            leads.append(dict(lead_hours=lead, admitted_original_hours=int(common.sum()), lags=correlations))
        rows.append(dict(panel=dict(identity), job_id=identity['job_id'], leads=leads))
        del error
    return dict(schema='hydra-primary-development-residual-group/v1', fold=manifest['fold'],
        gauge_id=manifest['gauge_id'], group_arrays_sha256=manifest['group_arrays_sha256'],
        calendar_start_utc=manifest['calendar_start_utc'], calendar_end_exclusive_utc=manifest['calendar_end_exclusive_utc'],
        calendar_hours=len(origins), admitted_origins=int(common.sum()), panels=rows,
        lead_hours=list(LEADS), lag_hours=list(LAG_HOURS), max_absolute_by_lag=_lag_maxima(rows),
        residual_definition='RAW_PHYSICAL_PREDICTION_MINUS_ADMISSIBLE_OBSERVATION',
        correlation_definition='PAIRWISE_PEARSON_SEPARATE_CENTERING_OF_EARLIER_AND_LATER_SURVIVING_ENDPOINTS',
        residual_units='m3/s', correlation_units='dimensionless',
        complete_calendar_retained=True, both_endpoint_masks_required=True, target_imputation=False,
        conditioned_on_common_L168_admission=True, paired_hours_are_independent_samples=False,
        overlapping_forecast_windows_present=True, significance_bands_computed=False,
        block_length_selected=False, operational_reliability_established=False,
        model_calls=0, optimizer_updates=0, protected_assessment_access=False)


def run_diagnostics(plan, guard):
    require(guard.run == Path(plan['run_directory']) and plan['protected_assessment_access'] is False,
            'Bounded development residual postprocessing guard required')
    audit = json.loads(guard.read(guard.run/'AUDIT.json', 16*1024**2))
    require(audit['plan_sha256'] == plan['plan_sha256']
            and audit['status'] == 'ALL180_PREDICTIONS_INDEPENDENTLY_AUDITED'
            and audit['protected_assessment_access'] is False and audit['model_calls'] == 0
            and len(audit['groups']) == 12
            and {(r['fold'],r['gauge_id']) for r in audit['groups']} == {(f,g) for f in FOLDS for g in GAUGES},
            'All12 unique audited prediction groups required before residual analysis')
    outputs, all_panels = [], []
    for record in sorted(audit['groups'], key=lambda r:(FOLDS.index(r['fold']), GAUGES.index(r['gauge_id']))):
        guard.check()
        name = f"{record['fold']}__{record['gauge_id']}"
        require(record['group'] == name and record['plan_sha256'] == plan['plan_sha256'], 'Residual group identity differs')
        committed = json.loads(guard.read(guard.run/'groups'/name/'COMMIT.json', 2*1024**2))
        require(committed == record, 'Residual input differs from immutable group commit')
        arrays, manifest = load_group(plan, guard, record)
        result = diagnose_group(arrays, manifest, check=guard.check)
        result.update(plan_sha256=plan['plan_sha256'], original_plan_sha256=plan['original_plan_sha256'],
                      protocol_sha256=plan['protocol_sha256'], input_artifacts=record['artifacts'])
        descriptor = guard.write(f'residuals/{name}/diagnostics.json', _json(result))
        outputs.append(dict(group=name, fold=record['fold'], gauge_id=record['gauge_id'],
                            panels=len(result['panels']), report=descriptor))
        all_panels.extend(result['panels'])
        del arrays, manifest, result
        guard.check(); guard.save()
        print(json.dumps(dict(stage='RESIDUAL_GROUP_COMPLETE', group=name)), flush=True)
    require(len(all_panels) == 180 and len({x['job_id'] for x in all_panels}) == 180,
            'Exactly180 unique completed-fit residual panels required')
    result = dict(schema='hydra-primary-development-residual-stage/v1',
        status='ALL180_PANELS_CALENDAR_LAG_RESIDUAL_DIAGNOSTICS_COMPLETE',
        plan_sha256=plan['plan_sha256'], original_plan_sha256=plan['original_plan_sha256'],
        protocol_sha256=plan['protocol_sha256'], groups=outputs, panel_count=180, lead_hours=list(LEADS),
        lag_hours=list(LAG_HOURS), series_count=180*18, correlation_cells=180*18*len(LAG_HOURS),
        max_absolute_by_lag=_lag_maxima(all_panels),
        interpretation='DESCRIPTIVE_CALENDAR_LAG_CORRELATION_CONDITIONAL_ON_COMMON_ADMITTED_DEVELOPMENT_CASES',
        pairing='BOTH_ORIGINAL_HOURLY_ENDPOINTS_REQUIRED_NO_CALENDAR_COMPACTION',
        uncertainty_method_reselected=False, significance_bands_computed=False,
        independence_established=False, operational_reliability_established=False,
        model_calls=0, optimizer_updates=0, new_downloads=0, protected_assessment_access=False)
    guard.write('RESIDUAL_DIAGNOSTICS.json', _json(result));guard.save()
    return result
