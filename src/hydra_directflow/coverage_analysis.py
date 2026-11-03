"""Development-only coverage diagnostics; no predictions, scaling or case edits."""
from collections import Counter
from datetime import datetime, timezone
import numpy as np

from .reference import ASSESSMENT_START, digest, require
from .quality import PRIMARY_APPROVED, admit_usgs_quality

HOUR = 3600
REGIMES = ('LOW', 'ORDINARY', 'ELEVATED', 'HIGH', 'UNKNOWN_REFERENCE')
SEASONS = ('DJF', 'MAM', 'JJA', 'SON')


def season(epoch):
    month = datetime.fromtimestamp(int(epoch), timezone.utc).month
    return SEASONS[(month % 12) // 3]


def iso(epoch):
    return datetime.fromtimestamp(int(epoch), timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def reference_status(value, flags, *, present=True):
    if not present: return 'NO_EXACT_HOUR_RECORD'
    if value is None or not np.isfinite(value) or value < 0: return 'INVALID_OR_ABSENT_NUMERIC_VALUE'
    if admit_usgs_quality(flags, policy_id=PRIMARY_APPROVED): return 'APPROVED_REFERENCE'
    if isinstance(flags, (list, tuple)) and 'e' in flags: return 'ESTIMATED_REFERENCE_REJECTED'
    if isinstance(flags, (list, tuple)) and 'P' in flags: return 'PROVISIONAL_REFERENCE_REJECTED'
    return 'UNRECOGNIZED_OR_UNAPPROVED_QUALIFIERS'


def reference_arrays(observations, *, start, end):
    require(type(start) is int and type(end) is int and start % HOUR == end % HOUR == 0
            and start < end <= ASSESSMENT_START, 'Explicit development-only reference interval required')
    times = np.arange(start, end, HOUR, dtype=np.int64)
    values = np.full(len(times), np.nan)
    statuses = np.full(len(times), 'NO_EXACT_HOUR_RECORD', dtype='<U48')
    seen = set()
    for stamp, value, flags in observations:
        if not start <= stamp < end: continue
        require(type(stamp) is int and stamp % HOUR == 0 and stamp not in seen, 'Duplicate/non-hourly reference')
        seen.add(stamp); index = (stamp-start)//HOUR
        statuses[index] = reference_status(value, flags)
        if statuses[index] == 'APPROVED_REFERENCE': values[index] = value
    return dict(times=times, values=values, statuses=statuses, admitted=statuses == 'APPROVED_REFERENCE')


def gradient_thresholds(reference, *, start, end, refit_at, delay_hours=2):
    """Diagnostic quantiles use unique approved gradient-calendar hours only.

    This deliberately does not condition on weather/history case admission and
    is not a model scaler. Stopping/outer references cannot affect thresholds.
    """
    require(start < end <= refit_at <= ASSESSMENT_START and delay_hours in (0, 2), 'Explicit gradient/refit bounds required')
    selected = (reference['times'] >= start) & (reference['times'] < end)
    selected &= reference['times'] + delay_hours*HOUR <= refit_at
    selected &= reference['admitted']
    values = reference['values'][selected]
    require(len(values) > 0 and np.isfinite(values).all() and (values >= 0).all(), 'No admissible gradient reference support')
    q10, q90, q95 = np.quantile(values, [.10, .90, .95], method='linear').tolist()
    return dict(q10_cms=q10, q90_cms=q90, q95_cms=q95, reference_hours=len(values),
                reference_times_sha256=digest(reference['times'][selected].tolist()),
                fit_start_utc=iso(start), fit_end_exclusive_utc=iso(end), refit_at_utc=iso(refit_at),
                simulated_label_delay_hours=delay_hours, quantile_method='numpy.linear',
                fit_population='UNIQUE_APPROVED_GRADIENT_CALENDAR_HOURS_INDEPENDENT_OF_INPUT_CASE_AVAILABILITY',
                model_scaler=False, stopping_or_outer_values_used=False)


def regime_codes(values, admitted, thresholds):
    require(values.shape == admitted.shape and admitted.dtype == np.dtype(bool), 'Aligned reference masks required')
    out = np.full(values.shape, 4, dtype=np.int8)
    out[admitted] = 1
    out[admitted & (values <= thresholds['q10_cms'])] = 0
    out[admitted & (values > thresholds['q90_cms'])] = 2
    out[admitted & (values > thresholds['q95_cms'])] = 3
    return out


def unusable_intervals(reference):
    bad = ~reference['admitted']; result = []; first = None
    for index in range(len(bad) + 1):
        if index < len(bad) and bad[index] and first is None: first = index
        if first is not None and (index == len(bad) or not bad[index]):
            result.append(dict(start_utc=iso(reference['times'][first]), end_exclusive_utc=iso(reference['times'][index-1]+HOUR),
                hours=index-first, reasons=dict(Counter(reference['statuses'][first:index].tolist())),
                cause='UNUSABLE_REFERENCE_INTERVAL_NOT_PROOF_OF_INSTRUMENT_DAMAGE_OR_OUTAGE',
                hidden_peak_magnitude_known=False))
            first = None
    return result


def select_observed_events(reference, thresholds, *, start, end, separation_hours=168, before_hours=48, after_hours=72):
    """Rank observed high-flow hours without model errors; keep gaps/truncation.

    Greedy observed maxima separated by >=168h define diagnostic event windows.
    This is a reproducible plotting/coverage rule, not a physical event detector
    or proof of independent events. Missing reference peaks cannot be inferred.
    """
    require(start < end <= ASSESSMENT_START and separation_hours >= before_hours + after_hours
            and before_hours >= 0 and after_hours > 0, 'Nonoverlapping bounded event windows required')
    eligible = np.flatnonzero(reference['admitted'] & (reference['times'] >= start)
                              & (reference['times'] < end) & (reference['values'] > thresholds['q95_cms']))
    ranked = sorted(eligible.tolist(), key=lambda i: (-reference['values'][i], int(reference['times'][i])))
    chosen = []
    for i in ranked:
        if all(abs(int(reference['times'][i])-int(reference['times'][j])) >= separation_hours*HOUR for j in chosen): chosen.append(i)
    result = []
    for rank, i in enumerate(chosen, 1):
        peak = int(reference['times'][i]); left = max(start, peak-before_hours*HOUR); right = min(end, peak+after_hours*HOUR)
        window = (reference['times'] >= left) & (reference['times'] < right)
        result.append(dict(event_id=iso(peak), observed_peak_time_utc=iso(peak), observed_peak_q_cms=float(reference['values'][i]),
            observed_peak_rank=rank, representative_top3=rank <= 3, start_utc=iso(left), end_exclusive_utc=iso(right),
            start_epoch=left, end_epoch=right, peak_epoch=peak, reference_requested_hours=int(window.sum()),
            reference_approved_hours=int(reference['admitted'][window].sum()),
            reference_unusable_hours=int((~reference['admitted'][window]).sum()),
            left_truncated=left != peak-before_hours*HOUR, right_truncated=right != peak+after_hours*HOUR,
            selection_uses_predictions=False, selection_requires_model_admission=False,
            true_peak_during_missing_reference_known=False))
    return result


def case_masks(batch):
    n = len(batch['origin_time'])
    weather = batch['weather_mask'].reshape(n, -1).all(axis=1)
    soil = batch['state_mask'].reshape(n, -1).all(axis=1)
    long = batch['history_mask'].reshape(n, -1).all(axis=1)
    short = batch['history_mask'][:, -24:].reshape(n, -1).all(axis=1)
    labels = batch['target_mask'].all(axis=1)
    calendar = batch['source_calendar_status'] == 'RESOLVED_WITHIN_ACQUIRED_CALENDAR'
    inputs = long & weather & soil & calendar
    primary = inputs & labels
    require(np.array_equal(primary, batch['admission_mask']), 'Primary candidate membership changed')
    return dict(input_L168=inputs, input_L24=short & weather & soil & calendar,
                common_L168=primary, natural_L24=short & weather & soil & calendar & labels,
                complete_labels=labels, missing_history_L168=~long, missing_history_L24=~short,
                missing_weather=~weather, missing_soil=~soil, incomplete_labels=~labels,
                calendar_unresolved=~calendar)


def coverage_by_regime(origins, input_admitted, primary, target_mask, reference, thresholds, *, start, end, natural_l24=None):
    if natural_l24 is not None:
        require(natural_l24.shape == primary.shape and natural_l24.dtype == np.dtype(bool)
                and not (primary & ~natural_l24).any(), 'Natural L24 must contain the common L168 cohort')
    role = (origins >= start) & (origins+18*HOUR < end)
    times = origins[role, None] + np.arange(1,19)*HOUR
    positions = (times-reference['times'][0])//HOUR
    require(((positions >= 0) & (positions < len(reference['times']))).all(), 'Target reference outside declared inventory')
    observed = reference['admitted'][positions]
    require(np.array_equal(observed, target_mask[role]), 'Target quality differs from independently loaded hourly reference')
    codes = regime_codes(reference['values'][positions], observed, thresholds)
    rows = []
    for code, name in enumerate(REGIMES):
        subset = codes == code
        rows.append(dict(regime=name, requested_target_cells=int(subset.sum()),
                         reference_supported_cells=int((subset & observed).sum()),
                         input_admissible_cells=int((subset & input_admitted[role,None]).sum()),
                         input_and_reference_supported_cells=int((subset & observed & input_admitted[role,None]).sum()),
                         common_comparison_cells=int((subset & primary[role,None]).sum()),
                         coverage_grain='GAUGE_ORIGIN_LEAD_CELLS_NOT_UNIQUE_HOURS'))
        if natural_l24 is not None:
            rows[-1]['natural_L24_comparison_cells']=int((subset & natural_l24[role,None]).sum())
            rows[-1]['natural_L24_cells_lost_to_common_L168']=rows[-1]['natural_L24_comparison_cells']-rows[-1]['common_comparison_cells']
    require(sum(row['requested_target_cells'] for row in rows) == int(role.sum())*18, 'Regime denominator did not reconcile')
    return rows


def event_coverage(event, origins, inputs, primary, target_mask, *, outer_start, outer_end):
    role = (origins >= outer_start) & (origins+18*HOUR < outer_end)
    result = []
    for lead in range(1,19):
        valid = origins + lead*HOUR
        window = role & (valid >= event['start_epoch']) & (valid < event['end_epoch'])
        peak = role & (valid == event['peak_epoch'])
        result.append(dict(lead_hours=lead, requested_cells=int(window.sum()),
            input_admissible_cells=int((window & inputs).sum()), approved_target_cells=int((window & target_mask[:,lead-1]).sum()),
            common_comparison_cells=int((window & primary).sum()),
            observed_peak_has_common_prediction_case=bool((peak & primary).any())))
    return result
