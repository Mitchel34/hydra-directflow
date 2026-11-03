"""Audit-only structural history-gap candidates; no values, fills or targets.

The caller supplies its already approved exact-hour Q mask. Source observation
clocks are checked, but source qualifiers/identities are not reconstructed here.
All availability is conditional on a simulated two-hour delay, not evidence of
historical publication or revision times. No policy is scientifically adopted.
"""
from numbers import Integral
from types import MappingProxyType

import numpy as np

from .reference import ASSESSMENT_START, HOUR, digest, require

HISTORY_HOURS = 168
SIMULATED_LAG_HOURS = 2
POLICY_IDS = MappingProxyType({
    'strict': 'HIST_Q_STRICT_L168_V1',
    'interior_1': 'HIST_Q_INTERIOR_R1_T1_L168_V1',
    'interior_2': 'HIST_Q_INTERIOR_R2_T2_L168_V1',
    'trailing_1': 'HIST_Q_TRAILING_R1_T1_L168_V1',
    'trailing_2': 'HIST_Q_TRAILING_R2_T2_L168_V1',
    'combined_1': 'HIST_Q_COMBINED_R1_T1_L168_V1',
    'combined_2': 'HIST_Q_COMBINED_R2_T2_L168_V1',
})
NATURAL_L24_ID = 'HIST_Q_STRICT_NATURAL_L24_V1'


def policy_catalog():
    """Fresh JSON-safe specifications; per-run and TOTAL missing caps are explicit."""
    result = {}
    for alias, policy_id in POLICY_IDS.items():
        mode, cap = ('strict', 0) if alias == 'strict' else (alias.rsplit('_', 1)[0], int(alias[-1]))
        result[policy_id] = dict(alias=alias, mode=mode, history_hours=HISTORY_HOURS,
            per_run_cap_hours=cap, total_missing_cap_hours=cap,
            allowed_positions=[] if mode == 'strict' else ['interior','trailing'] if mode == 'combined' else [mode],
            simulated_lag_hours=SIMULATED_LAG_HOURS, leading_gap_allowed=False,
            numerical_fill_implemented=False, policy_status='AUDIT_ONLY_PROPOSED_NOT_ADOPTED')
    return result


POLICY_CATALOG_SHA256 = digest(policy_catalog())


def _integer_epoch(value):
    require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)),
            'Explicit integer source/cutoff epoch seconds required')
    value = int(value)
    require(value >= 0 and value % HOUR == 0, 'Exact nonnegative UTC observation/cutoff hour required')
    return value


def _endpoint(index, times, approved, sources, cutoff):
    if not 0 <= index < len(times):
        return None
    require(approved[index], 'Gap endpoint must be an original approved observation')
    observation = sources[index]
    available = observation + SIMULATED_LAG_HOURS * HOUR
    return dict(index=index, observation_epoch=observation, history_slot_epoch=int(times[index]),
                approved=True, simulated_available_epoch=available, simulated_available_by_cutoff=available <= cutoff,
                age_hours=(cutoff-observation)//HOUR)


def _runs(times, approved, sources, cutoff):
    result, cursor = [], 0
    while cursor < len(times):
        if approved[cursor]:
            cursor += 1; continue
        start = cursor
        while cursor < len(times) and not approved[cursor]: cursor += 1
        end = cursor
        position = ('all_missing' if start == 0 and end == len(times) else 'leading' if start == 0
                    else 'trailing' if end == len(times) else 'interior')
        result.append(dict(position=position, start_index=start, end_index_exclusive=end, length_hours=end-start,
            first_missing_epoch=int(times[start]), last_missing_epoch=int(times[end-1]),
            nearest_missing_age_hours=(cutoff-int(times[end-1]))//HOUR,
            farthest_missing_age_hours=(cutoff-int(times[start]))//HOUR,
            touches_window_start=start == 0, touches_window_end=end == len(times),
            left_endpoint=_endpoint(start-1,times,approved,sources,cutoff),
            right_endpoint=_endpoint(end,times,approved,sources,cutoff)))
    return result


def _decision(spec, runs, missing_total, approved_total):
    reasons = []
    if missing_total:
        if spec['mode'] == 'strict':
            reasons.append('STRICT_REQUIRES_COMPLETE_HISTORY')
        else:
            if approved_total == 0: reasons.append('NO_APPROVED_HISTORY')
            if any(run['touches_window_start'] for run in runs): reasons.append('LEADING_GAP_UNSUPPORTED')
            if any(run['length_hours'] > spec['per_run_cap_hours'] for run in runs): reasons.append('RUN_LENGTH_EXCEEDS_CAP')
            if missing_total > spec['total_missing_cap_hours']: reasons.append('TOTAL_MISSING_EXCEEDS_CAP')
            for position in ('interior','trailing'):
                if position not in spec['allowed_positions'] and any(run['position'] == position for run in runs):
                    reasons.append(position.upper() + '_GAP_NOT_ALLOWED')
            for run in runs:
                if run['position'] == 'interior' and 'interior' in spec['allowed_positions']:
                    if not all(run[key] is not None and run[key]['approved'] and run[key]['simulated_available_by_cutoff']
                               for key in ('left_endpoint','right_endpoint')):
                        reasons.append('INTERIOR_APPROVED_AVAILABLE_ENDPOINTS_REQUIRED')
                elif run['position'] == 'trailing' and 'trailing' in spec['allowed_positions']:
                    left = run['left_endpoint']
                    if left is None or not left['approved'] or not left['simulated_available_by_cutoff']:
                        reasons.append('TRAILING_LAST_APPROVED_AVAILABLE_READING_REQUIRED')
    return dict(**spec, history_admissible=not reasons, additional_history_candidate=bool(missing_total and not reasons),
                missing_total=missing_total, maximum_missing_run_hours=max((run['length_hours'] for run in runs),default=0),
                reasons=list(dict.fromkeys(reasons)))


def classify_history_window(*, history_times, approved_mask, source_observation_times, cutoff_epoch,
                            simulated_lag_hours=SIMULATED_LAG_HOURS):
    """Classify exactly C−169h through C−2h; do not create repaired arrays.

    source_observation_times has168 integer UTC timestamps or None values. Every
    approved slot requires its exact source timestamp; unapproved slots may be
    None or retain the exact slot's timestamp without becoming approved. The
    caller's provenance must establish that these are source observation clocks,
    not an invented nearest-record match. Invalid clocks fail rather than turn
    into recoverable missingness. Weather/soil/targets remain separate gates.
    """
    cutoff = _integer_epoch(cutoff_epoch)
    require(HISTORY_HOURS * HOUR + HOUR <= cutoff < ASSESSMENT_START,
            'Development-only cutoff with a complete L168 clock window required')
    require(type(simulated_lag_hours) is int and simulated_lag_hours == SIMULATED_LAG_HOURS,
            'Only the explicitly simulated two-hour availability convention is classified')
    times, approved = np.asarray(history_times), np.asarray(approved_mask)
    expected = cutoff - np.arange(HISTORY_HOURS+1,1,-1,dtype=np.int64)*HOUR
    require(times.shape == (HISTORY_HOURS,) and times.dtype.kind in 'iu' and np.array_equal(times,expected),
            'Exact consecutive L168 history slots ending C minus two hours required')
    require(approved.shape == (HISTORY_HOURS,) and approved.dtype == np.dtype(bool),
            'Original one-dimensional Boolean approved Q mask required')
    try: supplied = tuple(source_observation_times)
    except TypeError as error: raise ValueError('Explicit aligned source observation timestamps required') from error
    require(len(supplied) == HISTORY_HOURS, 'Source observation timestamps must align with all168 slots')
    sources = []
    for index, value in enumerate(supplied):
        source = None if value is None else _integer_epoch(value)
        require(source is None or source == int(times[index]), 'Source observation timestamp differs from its exact history slot')
        if approved[index]:
            require(source is not None and source + SIMULATED_LAG_HOURS*HOUR <= cutoff,
                    'Approved history requires its original timestamp available by simulated cutoff')
        sources.append(source)
    runs = _runs(times,approved,sources,cutoff)
    missing_total, approved_total = int((~approved).sum()), int(approved.sum())
    catalog = policy_catalog()
    natural_missing = int((~approved[-24:]).sum())
    return dict(schema='hydra-audit-history-gap-candidates/v1', scope='HISTORY_MASK_AND_CLOCKS_ONLY_NO_REPAIR',
        cutoff_epoch=cutoff, history_hours=HISTORY_HOURS, history_start_epoch=int(times[0]), history_end_epoch=int(times[-1]),
        approved_total=approved_total, missing_total=missing_total, maximum_missing_run_hours=max((r['length_hours'] for r in runs),default=0),
        runs=runs, candidates={policy_id:_decision(spec,runs,missing_total,approved_total) for policy_id,spec in catalog.items()},
        natural_l24=dict(policy_id=NATURAL_L24_ID,history_hours=24,history_start_epoch=int(times[-24]),
            history_end_epoch=int(times[-1]),history_admissible=natural_missing == 0,missing_total=natural_missing,
            reasons=[] if natural_missing == 0 else ['STRICT_REQUIRES_COMPLETE_HISTORY'],
            relation='UNREPAIRED_EXACT_FINAL24_SLOTS_SEPARATE_FROM_L168_REPAIR_CANDIDATES'),
        availability=dict(mode='RETROSPECTIVE_SIMULATED',lag_hours=SIMULATED_LAG_HOURS,
                          exact_approved_source_clocks_checked=True,actual_delivery_or_revision_times_verified=False),
        source_quality=dict(mask_semantics='CALLER_SUPPLIED_ALREADY_APPROVED_Q_OBSERVATIONS',original_qualifiers_rechecked=False),
        policy_catalog_sha256=POLICY_CATALOG_SHA256,
        window_identity=digest(dict(cutoff_epoch=cutoff,history_times=[int(t) for t in times],approved_mask=approved.tolist(),
                                    source_observation_times=sources,simulated_lag_hours=SIMULATED_LAG_HOURS)),
        execution=dict(numerical_Q_values_read=False,history_arrays_filled=False,original_masks_changed=False,
                       targets_read_or_changed=False,model_or_scaler_fits=0,scientific_policy_adopted=False),
        limitation='Structural candidate only: no interpolation, carry-forward, model imputation, source-quality change or forecast-case admission is performed')
