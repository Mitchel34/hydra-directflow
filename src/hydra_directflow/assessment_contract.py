"""Assessment preparation primitives; no data reader or execution release.

These coordinate-only functions preserve the origin-defined assessment year,
its declared H18 target tail, and the selected NWM short-range cutoff policy.
They do not load observations, forecasts or checkpoints, fit models, authorize
protected access, or relax any development-only entry point.
"""
from dataclasses import dataclass
from types import MappingProxyType
from .contract import ARCHITECTURES, GAUGE_IDS
from .reference import (ASSESSMENT_START, ASSESSMENT_END, ASSESSMENT_LAST_TARGET,
                        HOUR, assessment_origin_admitted, require)

SEEDS = (17, 29, 43)
ARMS = (('full', 168), ('q_only', 168), ('full', 24))
NWM_REACHES = MappingProxyType({'03161000': 6892192, '03164000': 6887572, '03479000': 19743430})
NWM_DELAY_HOURS = 2
NWM_NATIVE_LEADS = tuple(range(1, 19))
NWM_MATCHED_HORIZONS = tuple(range(1, 17))


def _hour(value):
    require(type(value) is int and value % HOUR == 0,
            'An exact integer UTC hour is required')
    return value


def origins():
    """8,760 requested origin hours, not a count of admitted forecasts."""
    return tuple(range(ASSESSMENT_START, ASSESSMENT_END, HOUR))


def clock(origin, history_hours=168):
    _hour(origin)
    require(history_hours in (24, 168) and type(history_hours) is int,
            'Only the retained L24/L168 histories are in scope')
    target = tuple(origin + h * HOUR for h in range(1, 19))
    require(assessment_origin_admitted(origin_anchor=origin, target_times=target),
            'Origin lies outside the assessment year or declared target tail')
    history = tuple(origin - h * HOUR for h in range(history_hours + 1, 1, -1))
    return {'origin': origin, 'cutoff': origin, 'history': history,
            'valid_times': target, 'target_tail_limit': ASSESSMENT_LAST_TARGET}


def expected_checkpoint_jobs():
    """Candidate roster, not a frozen selection or permission to load weights."""
    return tuple(f'final_development__{g}__{a}__{arm}_L{length}__seed{seed}'
                 for g in GAUGE_IDS for a in ARCHITECTURES
                 for arm, length in ARMS for seed in SEEDS)


def expected_primary_checkpoint_jobs():
    """Shortest requested first pass: 36 saved full-input L168 models."""
    return tuple(job for job in expected_checkpoint_jobs() if '__full_L168__' in job)


def validate_primary_checkpoint_roster(job_ids):
    jobs = tuple(job_ids)
    require(len(jobs) == len(set(jobs)) == 36
            and set(jobs) == set(expected_primary_checkpoint_jobs()),
            'Exactly the 36 retained full-L168 final-development identities are required')
    return tuple(sorted(jobs))


def validate_checkpoint_roster(job_ids):
    jobs = tuple(job_ids)
    require(len(jobs) == len(set(jobs)) == 108
            and set(jobs) == set(expected_checkpoint_jobs()),
            'Exactly the 108 retained final-development checkpoint identities are required')
    return tuple(sorted(jobs))


@dataclass(frozen=True)
class NwmLeadSupport:
    origin_horizon: int
    expected_reference_time: int
    valid_time: int
    native_lead: int
    supported: bool
    reason: str


def nwm_support(origin, *, reference_time=None, native_leads=()):
    """Coordinate support for the exact C-2 issue, with no hidden fallback.

    Missingness here describes the owned snapshot, not proof that NOAA failed
    to issue a forecast. The two-hour availability is a declared simulation
    assumption because the owned table lacks historical publication times.
    Numeric validity remains a separate source-reader qualification.
    """
    c = clock(origin)
    expected = origin - NWM_DELAY_HOURS * HOUR
    leads = tuple(native_leads)
    require(len(set(leads)) == len(leads)
            and all(type(h) is int and h in NWM_NATIVE_LEADS for h in leads),
            'Unique native short-range leads 1 through 18 are required')
    if reference_time is not None:
        _hour(reference_time)
        require(reference_time == expected,
                'Use the exact C-2 issue; do not change cutoff or add an unregistered fallback')
    else:
        require(not leads, 'Absent issue cannot declare present native leads')
    result = []
    for h, valid in enumerate(c['valid_times'], 1):
        native = h + NWM_DELAY_HOURS
        if native not in NWM_NATIVE_LEADS:
            reason = 'NATIVE_HORIZON_UNAVAILABLE'
        elif reference_time is None:
            reason = 'ISSUE_ABSENT_IN_OWNED_SNAPSHOT'
        elif native not in leads:
            reason = 'NATIVE_LEAD_ABSENT_IN_OWNED_SNAPSHOT'
        else:
            reason = 'COORDINATE_SUPPORTED_NUMERIC_VALIDITY_NOT_YET_CHECKED'
        result.append(NwmLeadSupport(h, expected, valid, native,
                                    reason.startswith('COORDINATE_SUPPORTED'), reason))
    return tuple(result)


def validate_nwm_row_coordinates(row, *, gauge_id, origin_horizon, origin):
    """Read only metadata keys. Streamflow values are deliberately untouched."""
    require(gauge_id in NWM_REACHES, 'Declared gauge required')
    require(type(origin_horizon) is int and origin_horizon in NWM_MATCHED_HORIZONS,
            'NWM does not support origin H17/H18 under this cutoff contract')
    c = clock(origin)
    reference = origin - NWM_DELAY_HOURS * HOUR
    require(type(row['feature_id']) is int and row['feature_id'] == NWM_REACHES[gauge_id],
            'NWM reach differs from the retained three-gauge crosswalk')
    require(type(row['ensemble']) is int and row['ensemble'] == 0,
            'Only the owned ensemble-0 short-range product is in scope')
    require(_hour(row['reference_time']) == reference
            and _hour(row['valid_time']) == c['valid_times'][origin_horizon - 1]
            and type(row['lead_hours']) is int
            and row['lead_hours'] == origin_horizon + NWM_DELAY_HOURS
            and row['valid_time'] == row['reference_time'] + row['lead_hours'] * HOUR,
            'NWM reference, native lead and origin-relative valid time do not agree')
    return (gauge_id, origin, origin_horizon, row['valid_time'])


def _keys(values, *, nwm=False):
    keys = tuple(values)
    for key in keys:
        require(isinstance(key, tuple) and len(key) == 4, 'Explicit gauge/origin/horizon/valid key required')
        g, c, h, valid = key
        require(g in GAUGE_IDS and type(h) is int and 1 <= h <= 18,
                'Declared gauge and origin-relative H1-H18 required')
        require(_hour(valid) == clock(c)['valid_times'][h - 1], 'Comparison valid time differs')
        if nwm:
            require(h in NWM_MATCHED_HORIZONS, 'Unsupported NWM horizon cannot enter a paired comparison')
    require(len(keys) == len(set(keys)), 'Duplicate comparison keys must be resolved with lineage')
    return frozenset(keys)


def paired_keys(hydra_forecasts, nwm_forecasts, approved_targets):
    """Return matched scoring keys without changing independent HYDRA eligibility.

    This helper checks coordinates only. A consuming runner must authenticate
    its release and committed predictions before loading verification targets.
    """
    hydra = _keys(hydra_forecasts)
    nwm = _keys(nwm_forecasts, nwm=True)
    target = _keys(approved_targets)
    return {'independent_hydra_forecast_keys': hydra,
            'independent_hydra_evaluable_keys': hydra & target,
            'paired_evaluable_keys': hydra & nwm & target}
