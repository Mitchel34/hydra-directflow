"""Pure development event/coverage summaries from the frozen observed catalog.

The caller authenticates the coverage-v2 report bytes and saved predictions.
This module opens no source files and never selects events from model errors.
Reference coverage is taken from the full-calendar census; error summaries use
only the common admitted forecasts supplied by the caller. Overlapping forecast
cells, unique valid hours, and diagnostic event episodes remain separate units.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from collections.abc import Mapping

import numpy as np

from .reference import ASSESSMENT_START, require
from .contract import GAUGE_IDS

HOUR = 3600
LEADS = tuple(range(1, 19))
REGIMES = ('LOW', 'ORDINARY', 'ELEVATED', 'HIGH', 'UNKNOWN_REFERENCE')
POPULATION = 'UNIQUE_APPROVED_GRADIENT_CALENDAR_HOURS_INDEPENDENT_OF_INPUT_CASE_AVAILABILITY'
CATALOG_RULE = 'TOP3_OBSERVED_Q95_EXCEEDING_PEAKS_SEPARATED_168H'


def _hour(value):
    require(isinstance(value, str), 'Explicit UTC catalog timestamp required')
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0
            and stamp.minute == stamp.second == stamp.microsecond == 0,
            'Exact UTC catalog hour required')
    return int(stamp.timestamp())


def _number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _requested_by_lead(start, end, left, right):
    # Full H18-eligible origin calendar is [start,end-18h); each lead's valid
    # times shift that calendar without inventing boundary-crossing origins.
    return [max(0, (min(right, end-(18-lead)*HOUR)-max(left, start+lead*HOUR))//HOUR)
            for lead in LEADS]


def validate_catalog_row(row):
    """Validate fixed rule metadata; the parent authenticates source SHA256.

    This verifies the supplied catalog's structural rule and coverage counts,
    not the presence of every candidate peak in a raw reference series.
    """
    require(isinstance(row, dict) and row.get('gauge_id') in GAUGE_IDS
            and row.get('fold') in ('cv_1', 'cv_2', 'cv_3', 'final_development'),
            'Declared gauge/fold catalog required')
    outer, thresholds = row['outer_window'], row['thresholds']
    start, end = _hour(outer['start_utc']), _hour(outer['end_exclusive_utc'])
    require(0 <= start < end <= ASSESSMENT_START, 'Development-only outer catalog required')
    require(_hour(thresholds['fit_start_utc']) < _hour(thresholds['fit_end_exclusive_utc']) <= start
            and _hour(thresholds['refit_at_utc']) == start
            and thresholds['fit_population'] == POPULATION
            and thresholds['stopping_or_outer_values_used'] is False
            and thresholds['model_scaler'] is False and thresholds['simulated_label_delay_hours'] == 2
            and thresholds['quantile_method'] == 'numpy.linear', 'Frozen gradient-calendar thresholds required')
    quantiles = [thresholds[k] for k in ('q10_cms', 'q90_cms', 'q95_cms')]
    require(all(_number(x) and x >= 0 for x in quantiles) and quantiles == sorted(quantiles),
            'Finite ordered discharge thresholds required')
    require(row['representative_event_rule'] == CATALOG_RULE
            and row['event_selection_uses_model_errors'] is False
            and row['event_windows_clipped_to_outer'] is True
            and row['event_windows_hours'] == [-48, 72], 'Frozen observed-only event rule required')
    events = row['observed_high_flow_events']
    require(isinstance(events, list) and len(events) <= 60
            and row['observed_high_flow_event_count'] == len(events), 'Bounded complete catalog required')
    peaks, previous_rank_key = [], None
    for rank, event in enumerate(events, 1):
        peak = event['peak_epoch']; left = event['start_epoch']; right = event['end_epoch']
        require(all(type(t) is int and t % HOUR == 0 for t in (peak, left, right))
                and start <= peak < end and left == max(start, peak-48*HOUR)
                and right == min(end, peak+72*HOUR), 'Catalog window/peak clock differs')
        require(_hour(event['event_id']) == _hour(event['observed_peak_time_utc']) == peak
                and _hour(event['start_utc']) == left and _hour(event['end_exclusive_utc']) == right,
                'Catalog epoch/text timestamps differ')
        q = event['observed_peak_q_cms']
        require(_number(q) and q > quantiles[2], 'Catalog peak must exceed frozen q95')
        rank_key = (-q, peak)
        require(previous_rank_key is None or previous_rank_key < rank_key, 'Catalog ranking/tie rule changed')
        previous_rank_key = rank_key
        require(all(abs(peak-old) >= 168*HOUR for old in peaks), 'Catalog peak separation changed')
        peaks.append(peak)
        require(event['observed_peak_rank'] == rank and event['representative_top3'] is (rank <= 3)
                and event['selection_uses_predictions'] is False
                and event['selection_requires_model_admission'] is False
                and event['true_peak_during_missing_reference_known'] is False,
                'Catalog selection or representative identities changed')
        require(event['left_truncated'] is (left != peak-48*HOUR)
                and event['right_truncated'] is (right != peak+72*HOUR), 'Catalog clipping flags changed')
        reference_counts = [event[k] for k in ('reference_requested_hours', 'reference_approved_hours', 'reference_unusable_hours')]
        require(all(type(n) is int and n >= 0 for n in reference_counts)
                and reference_counts[0] == (right-left)//HOUR
                and reference_counts[1]+reference_counts[2] == reference_counts[0], 'Reference-hour counts differ')
        requested_by_lead = _requested_by_lead(start, end, left, right)
        require(type(event['all_lead_requested_cells']) is int
                and event['all_lead_requested_cells'] == sum(requested_by_lead)
                and type(event['all_lead_common_comparison_cells']) is int
                and 0 <= event['all_lead_common_comparison_cells'] <= event['all_lead_requested_cells'],
                'Aggregate catalog cell counts differ')
        supported = event['observed_peak_supported_leads']
        require(isinstance(supported, list) and supported == sorted(set(supported))
                and all(type(lead) is int and lead in LEADS for lead in supported), 'Catalog peak support differs')
        coverage = event.get('forecast_case_coverage_by_lead')
        require(coverage is not None or not event['representative_top3'], 'Representative per-lead coverage missing')
        if coverage is None:
            continue
        require(isinstance(coverage, list) and [x['lead_hours'] for x in coverage] == list(LEADS), 'All18 event-coverage leads required')
        for index, entry in enumerate(coverage):
            requested, inputs, reference, common = [entry[k] for k in
                ('requested_cells', 'input_admissible_cells', 'approved_target_cells', 'common_comparison_cells')]
            require(all(type(n) is int and 0 <= n <= reference_counts[0] for n in (requested, inputs, reference, common))
                    and common <= min(inputs, reference) and max(inputs, reference) <= requested
                    and requested == requested_by_lead[index]
                    and type(entry['observed_peak_has_common_prediction_case']) is bool,
                    'Event coverage denominators differ')
        require(sum(x['common_comparison_cells'] for x in coverage) == event['all_lead_common_comparison_cells']
                and [x['lead_hours'] for x in coverage if x['observed_peak_has_common_prediction_case']] == supported,
                'Catalog aggregate/per-lead support differs')
    return start, end


def _arrays(row, origins, forecast_time, labels, label_mask, predictions):
    start, end = validate_catalog_row(row)
    origin, future = np.asarray(origins), np.asarray(forecast_time)
    require(origin.ndim == 1 and origin.dtype.kind in 'iu' and len(origin) <= 20000
            and future.shape == (len(origin), 18) and future.dtype.kind in 'iu', 'Bounded H18 coordinates required')
    require(np.all(origin >= start) and np.all(origin < end)
            and np.all(origin % HOUR == 0) and (len(origin) < 2 or np.all(origin[1:] > origin[:-1]))
            and np.array_equal(future, origin[:, None]+np.arange(1, 19)[None, :]*HOUR)
            and np.all(future < end) and np.all(future < ASSESSMENT_START),
            'Exact purged development origin/target coordinates required')
    truth, mask = np.asarray(labels), np.asarray(label_mask)
    require(truth.shape == mask.shape == future.shape and truth.dtype == np.dtype('float64')
            and mask.dtype == np.dtype(bool) and mask.all(), 'Float64 original labels and complete common H18 mask required')
    require(np.isfinite(truth).all() and np.all(truth >= 0), 'Approved finite nonnegative discharge required')
    require(isinstance(predictions, Mapping) and 1 <= len(predictions) <= 48
            and all(isinstance(name, str) and name for name in predictions), 'Named paired model/seed predictions required')
    out = {}
    for name, values in predictions.items():
        value = np.asarray(values)
        require(value.shape == truth.shape and value.dtype.kind == 'f' and np.isfinite(value).all(),
                'Missing/nonfinite requested prediction fails; no selective support')
        out[name] = value
    if len(origin):
        _, first, inverse = np.unique(future.ravel(), return_index=True, return_inverse=True)
        require(np.array_equal(truth.ravel(), truth.ravel()[first][inverse]),
                'Repeated target valid hours have inconsistent reference values')
    return origin, future, truth, out


def _metrics(truth, prediction):
    n = len(truth)
    if not n:
        return dict(status='UNAVAILABLE_NO_MATCHED_TARGETS', n=0, physical_mse=None,
                    rmse=None, mae=None, bias=None, negative_prediction_frequency=None)
    errors = prediction.astype(np.float64)-truth
    mse = float(np.mean(errors*errors, dtype=np.float64))
    require(math.isfinite(mse), 'Event metric overflow')
    return dict(status='AVAILABLE_DESCRIPTIVE', n=n, physical_mse=mse, rmse=math.sqrt(mse),
                mae=float(np.mean(np.abs(errors), dtype=np.float64)), bias=float(np.mean(errors, dtype=np.float64)),
                negative_prediction_frequency=float(np.count_nonzero(prediction < 0)/n))


def summarize_events(row, *, origins, forecast_time, labels, label_mask, predictions):
    """Keep every frozen event, including unsupported representative peaks.

    Inputs are one gauge/fold's exact common-admitted N×18 forecast cases. Full
    calendar input/reference denominators come from the authenticated census.
    Model names may include seeds; their predictions and errors are never pooled.
    """
    origin, future, truth, predicted = _arrays(row, origins, forecast_time, labels, label_mask, predictions)
    events = []
    for event in row['observed_high_flow_events']:
        window = (future >= event['start_epoch']) & (future < event['end_epoch'])
        peak = future == event['peak_epoch']
        require(int(window.sum()) == event['all_lead_common_comparison_cells']
                and [lead for i, lead in enumerate(LEADS) if peak[:, i].any()] == event['observed_peak_supported_leads'],
                'Saved event membership differs from frozen common coverage')
        start, end = _hour(row['outer_window']['start_utc']), _hour(row['outer_window']['end_exclusive_utc'])
        requested_by_lead = _requested_by_lead(start, end, event['start_epoch'], event['end_epoch'])
        frozen_coverage = event.get('forecast_case_coverage_by_lead')
        leads = []
        for index, lead in enumerate(LEADS):
            selected = window[:, index]; at_peak = peak[:, index]
            frozen = frozen_coverage[index] if frozen_coverage is not None else None
            count = int(selected.sum())
            if frozen is not None:
                require(count == frozen['common_comparison_cells']
                        and bool(at_peak.any()) is frozen['observed_peak_has_common_prediction_case'],
                        'Saved event membership differs from frozen common coverage')
            require(int(at_peak.sum()) <= 1, 'A peak/lead must be at most one original forecast cell')
            if at_peak.any():
                require(np.allclose(truth[at_peak, index], event['observed_peak_q_cms'], rtol=1e-12, atol=1e-12),
                        'Saved peak reference differs from observed catalog')
            metrics = {}
            for name, value in predicted.items():
                m = _metrics(truth[selected, index], value[selected, index])
                m['observed_peak_forecast'] = None if not at_peak.any() else dict(
                    observed_q_cms=float(truth[at_peak, index][0]), prediction_q_cms=float(value[at_peak, index][0]),
                    error_cms=float(value[at_peak, index][0]-truth[at_peak, index][0]))
                metrics[name] = m
            requested = requested_by_lead[index]
            approved = frozen['approved_target_cells'] if frozen is not None else None
            inputs = frozen['input_admissible_cells'] if frozen is not None else None
            leads.append(dict(lead_hours=lead, calendar_requested_cells=requested,
                input_admissible_opportunities=inputs, approved_reference_cells=approved,
                matched_evaluable_cells=count, matched_coverage_of_known_reference=count/approved if approved else None,
                matched_coverage_of_calendar=count/requested if requested else None,
                input_availability_fraction=inputs/requested if inputs is not None and requested else None,
                input_reference_denominator_status='FROZEN_CENSUS' if frozen is not None else 'NOT_RETAINED_PER_LEAD_FOR_NONREPRESENTATIVE_EVENT',
                observed_peak_supported=bool(at_peak.any()), models=metrics))
        events.append(dict(event_id=event['event_id'], observed_peak_rank=event['observed_peak_rank'],
            representative_top3=event['representative_top3'], start_utc=event['start_utc'],
            end_exclusive_utc=event['end_exclusive_utc'], observed_peak_time_utc=event['observed_peak_time_utc'],
            observed_peak_q_cms=event['observed_peak_q_cms'], left_truncated=event['left_truncated'],
            right_truncated=event['right_truncated'], reference_requested_hours=event['reference_requested_hours'],
            reference_approved_hours=event['reference_approved_hours'], reference_unusable_hours=event['reference_unusable_hours'],
            matched_origin_lead_cells=int(window.sum()), unique_matched_target_hours=int(len(np.unique(future[window]))),
            unique_forecast_origins=int(np.count_nonzero(window.any(axis=1))),
            observed_peak_supported_any_lead=bool(peak.any()), observed_peak_supported_all18_leads=bool(peak.any(axis=0).all()),
            per_lead=leads))
    return dict(schema='hydra-frozen-event-postprocessing/v1', gauge_id=row['gauge_id'], fold=row['fold'],
                catalog_events=len(events), representative_event_ids=[e['event_id'] for e in events if e['representative_top3']],
                events_with_any_matched_prediction=sum(e['matched_origin_lead_cells'] > 0 for e in events),
                observed_peaks_with_any_lead=sum(e['observed_peak_supported_any_lead'] for e in events),
                observed_peaks_with_all18_leads=sum(e['observed_peak_supported_all18_leads'] for e in events),
                events=events, event_selection_uses_model_errors=False, unsupported_events_replaced=False,
                event_episodes_are_proven_independent=False, annual_NSE_support_rule_applied=False,
                metrics_interpretation='DESCRIPTIVE_ON_COMMON_ADMITTED_CASES_RAW_UNCLIPPED_PREDICTIONS',
                input_opportunities_are_not_actual_saved_forecasts=True,
                full_reference_calendar_values_reconstructed=False, protected_assessment_access=False)


def summarize_regimes(row, *, origins, forecast_time, labels, label_mask, predictions):
    """Descriptive regime metrics with separate full-calendar census coverage."""
    _, future, truth, predicted = _arrays(row, origins, forecast_time, labels, label_mask, predictions)
    t = row['thresholds']; codes = np.ones(truth.shape, dtype=np.int8)
    codes[truth <= t['q10_cms']] = 0
    codes[truth > t['q90_cms']] = 2
    codes[truth > t['q95_cms']] = 3
    frozen = row['target_cell_regime_coverage']
    require([x['regime'] for x in frozen] == list(REGIMES), 'Complete frozen regime census required')
    rows = []
    for index, name in enumerate(REGIMES):
        selected = codes == index
        require(int(selected.sum()) == frozen[index]['common_comparison_cells'],
                'Saved regime membership differs from frozen census')
        rows.append(dict(regime=name, census_coverage=dict(frozen[index]), matched_origin_lead_cells=int(selected.sum()),
                         unique_matched_target_hours=int(len(np.unique(future[selected]))),
                         per_lead=[dict(lead_hours=lead, n=int(selected[:, lead-1].sum()),
                             models={model: _metrics(truth[selected[:, lead-1], lead-1], values[selected[:, lead-1], lead-1])
                                     for model, values in predicted.items()}) for lead in LEADS]))
    return dict(schema='hydra-frozen-regime-postprocessing/v1', gauge_id=row['gauge_id'], fold=row['fold'],
                thresholds=dict(t), thresholds_recomputed=False, cases_conditioned_on_common_L168_admission=True,
                high_flow_means_official_flood_stage=False, regimes=rows, protected_assessment_access=False)


def known_target_calendar(row, *, origins, forecast_time, labels, label_mask):
    """A gapped plot series from admitted labels, never a full-reference claim."""
    _, future, truth, _ = _arrays(row, origins, forecast_time, labels, label_mask,
                                  {'coordinate_check_only': np.asarray(labels)})
    start, end = validate_catalog_row(row)
    calendar = np.arange(start, end, HOUR, dtype=np.int64)
    values = np.full(len(calendar), np.nan, dtype=np.float64)
    if len(future):
        stamps, first = np.unique(future.ravel(), return_index=True)
        values[(stamps-start)//HOUR] = truth.ravel()[first]
    return dict(times=calendar, values=values, known_from_admitted_labels=np.isfinite(values),
                scope='UNIQUE_TARGET_HOURS_FROM_COMMON_ADMITTED_CASES_NOT_FULL_REFERENCE_CALENDAR',
                gaps_filled=False)
