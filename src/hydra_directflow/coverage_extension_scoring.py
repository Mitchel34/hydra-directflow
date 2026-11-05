"""Pure Phase A development scoring with independent per-lead verification.

This module opens no files and performs no model calls. The caller authenticates
the prediction-completion barrier before loading targets. Routing is never
inferred from targets, errors, or a favorable model prediction. The mixed routed
product has its own identity, rather than pretending to have one lookback.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import re
import statistics

import numpy as np

from .contract import ARCHITECTURES, GAUGE_IDS
from .post_events import validate_catalog_row
from .quality import PRIMARY_APPROVED, admit_usgs_quality
from .reference import ASSESSMENT_START, digest, require
from .trainer_bundle import _array_digest

HOUR = 3600
START = int(datetime(2023, 7, 31, tzinfo=timezone.utc).timestamp())
END = int(datetime(2024, 7, 30, 6, tzinfo=timezone.utc).timestamp())
ORIGINS = 8766
LEADS = tuple(range(1, 19))
SEEDS = (17, 29, 43)
PRODUCTS = ('PRIMARY_FROZEN', 'L168_INPUT_ONLY_V1', 'L24_INPUT_ONLY_V1', 'ROUTED_CLEAN_168_24_V1')
COHORTS = ('ALL_REQUESTED', 'ORIGINAL_COMMON', 'ADDITIONAL_L168', 'L24_FALLBACK', 'UNAVAILABLE')
METRICS = ('physical_mse', 'rmse', 'mae', 'bias', 'nse', 'negative_frequency')
SHA_FIELDS = ('phase_plan_sha256', 'route_map_sha256', 'prediction_sha256',
              'prediction_barrier_sha256', 'input_source_sha256', 'label_source_sha256')
IDENTITY_FIELDS = frozenset(('gauge_id', 'architecture', 'seed', 'product',
                            'checkpoint_sha256_by_history', *SHA_FIELDS))


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def validate_calendar(origins, forecast_time, gauge_id):
    """Full fixed final-development calendar, before any outcome inspection."""
    require(gauge_id in GAUGE_IDS, 'Confirmed gauge required')
    origin, future = np.asarray(origins), np.asarray(forecast_time)
    require(origin.shape == (ORIGINS,) and origin.dtype.kind in 'iu'
            and np.array_equal(origin, np.arange(START, END, HOUR, dtype=np.int64)),
            'Exact complete final-development origin calendar required')
    require(future.shape == (ORIGINS, 18) and future.dtype.kind in 'iu'
            and np.array_equal(future, origin[:, None] + np.arange(1, 19)[None, :] * HOUR)
            and np.all(future < ASSESSMENT_START), 'Protected/crossing or changed H18 coordinates forbidden')
    return origin, future


def _boolean(value, shape, name):
    value = np.asarray(value)
    require(value.shape == shape and value.dtype == np.dtype(bool), name + ' must be an exact boolean array')
    return value


def _route_inputs(eligible168, eligible24, route):
    long = _boolean(eligible168, (ORIGINS,), 'L168 eligibility')
    short = _boolean(eligible24, (ORIGINS,), 'L24 eligibility')
    choices = np.asarray(route)
    require(choices.shape == (ORIGINS,) and choices.dtype.kind == 'U'
            and np.isin(choices, ('L168', 'L24', 'UNAVAILABLE')).all(), 'Explicit original route choices required')
    require(np.all(~long | short), 'Complete L168 must imply complete L24 under shared non-Q rules')
    expected = np.where(long, 'L168', np.where(short, 'L24', 'UNAVAILABLE'))
    require(np.array_equal(choices, expected), 'Router must choose complete L168 before complete L24')
    return long, short, choices


def _predictions(values, emitted):
    p = np.asarray(values)
    require(p.shape == (ORIGINS, 18) and p.dtype == np.dtype('<f8'), 'Full-calendar float64 physical predictions required')
    require(np.isfinite(p[emitted]).all(), 'Missing/nonfinite requested predictions are a technical failure')
    require(np.isnan(p[~emitted]).all(), 'Unavailable forecasts must remain explicit NaN, not hidden predictions')
    return p


def assemble_routed(*, gauge_id, origins, forecast_time, eligible168, eligible24, route,
                    predictions168, predictions24):
    """Choose one existing-checkpoint H18 vector per origin without labels."""
    validate_calendar(origins, forecast_time, gauge_id)
    long, short, choices = _route_inputs(eligible168, eligible24, route)
    a, b = _predictions(predictions168, long), _predictions(predictions24, short)
    result = np.full((ORIGINS, 18), np.nan, dtype=np.float64)
    result[choices == 'L168'] = a[choices == 'L168']
    result[choices == 'L24'] = b[choices == 'L24']
    result.setflags(write=False)
    return result


def verification_masks(*, gauge_id, origins, forecast_time, labels, qualifier_json, present_mask):
    """Apply original A/A+R qualifiers per target; never alter original values.

    JSON strings preserve qualifier order and duplicates. Malformed JSON is a
    source/encoding failure; valid but unrecognized flag combinations are
    excluded. Numeric invalidity and quality exclusions are recorded separately.
    """
    _, future = validate_calendar(origins, forecast_time, gauge_id)
    values, flags = np.asarray(labels), np.asarray(qualifier_json)
    present = _boolean(present_mask, (ORIGINS, 18), 'Reference presence')
    require(values.shape == flags.shape == future.shape and values.dtype == np.dtype('<f8')
            and flags.dtype.kind == 'U', 'Original physical targets and qualifier JSON must align')
    unique, inverse = np.unique(flags, return_inverse=True)
    admitted_flags = np.empty(len(unique), bool)
    classes = []
    for i, text in enumerate(unique.tolist()):
        require(len(text) <= 4096, 'Bounded qualifier encoding required')
        try:
            sequence = json.loads(text)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError('Malformed original qualifier JSON') from error
        admitted_flags[i] = admit_usgs_quality(sequence, policy_id=PRIMARY_APPROVED)
        if admitted_flags[i]: classes.append('APPROVED_REFERENCE')
        elif isinstance(sequence, list) and 'e' in sequence: classes.append('ESTIMATED_REFERENCE_REJECTED')
        elif isinstance(sequence, list) and 'P' in sequence: classes.append('PROVISIONAL_REFERENCE_REJECTED')
        else: classes.append('UNRECOGNIZED_OR_UNAPPROVED_QUALIFIERS')
    quality = admitted_flags[inverse].reshape(flags.shape)
    numeric = np.isfinite(values) & (values >= 0)
    mask = present & quality & numeric
    reason = np.asarray(classes, dtype='<U48')[inverse].reshape(flags.shape).copy()
    reason[~numeric] = 'INVALID_OR_ABSENT_NUMERIC_VALUE'
    reason[~present] = 'NO_EXACT_HOUR_RECORD'
    mask.setflags(write=False); reason.setflags(write=False)
    return dict(label_mask=mask, exclusion_reason=reason, quality_policy=PRIMARY_APPROVED,
                original_qualifier_json_sha256=_array_digest({'qualifier_json': flags}),
                original_values_sha256=_array_digest({'labels': values}),
                presence_sha256=_array_digest({'present': present}),
                label_mask_sha256=_array_digest({'label_mask': mask}),
                exclusion_counts_overlap=dict(missing_exact_hour=int((~present).sum()),
                    unapproved_flags=int((present & ~quality).sum()),
                    invalid_numeric=int((present & ~numeric).sum())),
                original_qualifiers_rewritten=False, targets_filled=False)


def _identity(identity):
    require(isinstance(identity, dict) and set(identity) == IDENTITY_FIELDS, 'Exact Phase A scoring identity required')
    require(identity['gauge_id'] in GAUGE_IDS and identity['architecture'] in ARCHITECTURES
            and type(identity['seed']) is int and identity['seed'] in SEEDS and identity['product'] in PRODUCTS,
            'Known gauge, architecture, seed and Phase A product required')
    require(all(_sha(identity[k]) for k in SHA_FIELDS), 'Input, prediction barrier and label SHA bindings required')
    weights = identity['checkpoint_sha256_by_history']
    required = {'168','24'} if identity['product'] == 'ROUTED_CLEAN_168_24_V1' else ({'24'} if identity['product'] == 'L24_INPUT_ONLY_V1' else {'168'})
    require(isinstance(weights, dict) and set(weights) == required and all(_sha(x) for x in weights.values()),
            'Truthful checkpoint/lookback inventory required for this product')
    require(len(set(weights.values()))==len(weights), 'Distinct lookbacks require their distinct saved checkpoints')


def _prepared(identity, origins, forecast_time, labels, label_mask, predictions,
              original_common, eligible168, eligible24, route):
    _identity(identity)
    origin, future = validate_calendar(origins, forecast_time, identity['gauge_id'])
    long, short, choices = _route_inputs(eligible168, eligible24, route)
    common = _boolean(original_common, (ORIGINS,), 'Original common cohort')
    require(np.all(~common | long), 'Original common cases must remain L168 input-eligible')
    product = identity['product']
    emitted = common if product == 'PRIMARY_FROZEN' else long if product == 'L168_INPUT_ONLY_V1' else short
    # Check every requested prediction independently of target quality.
    p = _predictions(predictions, emitted)
    y = np.asarray(labels); mask = _boolean(label_mask, (ORIGINS, 18), 'Independent target quality')
    require(y.shape == future.shape and y.dtype == np.dtype('<f8')
            and np.isfinite(y[mask]).all() and np.all(y[mask] >= 0), 'Admitted references must be finite nonnegative physical discharge')
    require(np.array_equal(common, long & mask.all(axis=1)),
            'Original common cohort must equal complete-target L168 eligibility; reconcile changed membership')
    _, first, inverse = np.unique(future.ravel(), return_index=True, return_inverse=True)
    require(np.array_equal(mask.ravel(), mask.ravel()[first][inverse]), 'Repeated target hours have different quality masks')
    repeated = y.ravel()[first][inverse].reshape(y.shape)
    require(np.array_equal(y[mask], repeated[mask]), 'Repeated approved target hours have different values')
    cohorts = dict(ALL_REQUESTED=np.ones(ORIGINS, bool), ORIGINAL_COMMON=common,
                   ADDITIONAL_L168=long & ~common, L24_FALLBACK=short & ~long, UNAVAILABLE=~short)
    require(np.all(sum(cohorts[k].astype(np.int8) for k in COHORTS[1:]) == 1), 'Origin cohorts must form one disjoint partition')
    product_history = np.zeros(ORIGINS, np.int16)
    if product == 'ROUTED_CLEAN_168_24_V1':
        product_history[choices == 'L168'] = 168; product_history[choices == 'L24'] = 24
    else: product_history[emitted] = 24 if product == 'L24_INPUT_ONLY_V1' else 168
    return origin, future, y, mask, p, long, short, choices, emitted, cohorts, product_history


def _metrics(truth, predicted, target_times):
    n = len(truth)
    out = dict(n=n, physical_mse=None, rmse=None, mae=None, bias=None, nse=None,
               negative_predictions_scored=int(np.count_nonzero(predicted < 0)),
               negative_frequency=None, distinct_target_hours=int(len(np.unique(target_times))),
               distinct_utc_dates=int(len(np.unique(target_times // 86400))), undefined={})
    if not n:
        out['undefined'] = {key:'ZERO_SUPPORT' for key in METRICS}
    else:
        errors = [float(p)-float(y) for p,y in zip(predicted, truth)]
        sse = math.fsum(x*x for x in errors); center = math.fsum(float(y)/n for y in truth)
        denominator = math.fsum((float(y)-center)**2 for y in truth)
        out.update(physical_mse=sse/n, rmse=math.sqrt(sse/n), mae=math.fsum(abs(x)/n for x in errors),
                   bias=math.fsum(x/n for x in errors), negative_frequency=out['negative_predictions_scored']/n)
        if np.all(truth == truth[0]): out['undefined']['nse'] = 'CONSTANT_REFERENCE'
        elif denominator == 0: out['undefined']['nse'] = 'REFERENCE_VARIANCE_NUMERIC_UNDERFLOW'
        else: out['nse'] = 1-sse/denominator
    require(all(out[k] is None or math.isfinite(out[k]) for k in METRICS), 'Physical metric arithmetic overflow')
    reasons = []
    if out['distinct_target_hours'] < 168: reasons.append('TOO_FEW_DISTINCT_TARGET_HOURS')
    if out['distinct_utc_dates'] < 28: reasons.append('TOO_FEW_DISTINCT_UTC_DATES')
    if out['nse'] is None: reasons.append(out['undefined'].get('nse','UNDEFINED_NSE'))
    out.update(primary_eligible=not reasons, primary_ineligibility=reasons)
    return out


def _fraction(numerator, denominator):
    return numerator/denominator if denominator else None


def _cell_panel(origin, future, y, mask, p, eligible, emitted, cohort, cell_window=None, *, annual=True):
    window = np.broadcast_to(cohort[:, None], (ORIGINS, 18)).copy()
    if cell_window is not None: window &= cell_window
    leads=[]
    for h in range(18):
        requested = window[:, h]; reference = requested & mask[:, h]
        forecast = requested & emitted; scored = reference & emitted
        m = _metrics(y[scored,h], p[scored,h], future[scored,h])
        count, valid, issued = int(requested.sum()), int(reference.sum()), int(forecast.sum())
        eligible_count = int((requested & eligible).sum())
        m.update(lead_hours=h+1, n_requested=count, requested_origins=count, input_eligible_origins=eligible_count,
            emitted_forecasts=issued, valid_target_count=valid, evaluable_forecasts=m['n'],
            data_unavailable_origins=count-eligible_count, policy_not_emitted_with_eligible_inputs=eligible_count-issued,
            not_emitted_origins=count-issued, invalid_or_missing_targets=count-valid, technical_failures=0,
            forecast_availability_fraction=_fraction(issued,count), evaluable_coverage_fraction=_fraction(m['n'],count),
            coverage_among_valid_targets=_fraction(m['n'],valid),
            negative_predictions_emitted=int(np.count_nonzero(p[forecast,h] < 0)),
            negative_frequency_among_emitted=_fraction(int(np.count_nonzero(p[forecast,h] < 0)),issued),
            scored_coordinate_sha256=_array_digest({'origin':origin[scored], 'valid_time':future[scored,h]}),
            scoring_mask_sha256=_array_digest({'mask':scored}))
        if not annual:
            m.pop('nse');m.pop('primary_eligible');m.pop('primary_ineligibility')
            m['undefined'].pop('nse',None)
        leads.append(m)
    scored_cells=window & mask & emitted[:,None]
    primary = math.fsum(x['nse']/18 for x in leads) if annual and all(x['primary_eligible'] for x in leads) else None
    return dict(status=('PRIMARY_DEFINED' if primary is not None else 'PRIMARY_UNAVAILABLE') if annual else 'DESCRIPTIVE_FIXED_EVENT_OR_REGIME',
        requested_origins=int(np.count_nonzero(window.any(axis=1))), cohort_origins=int(cohort.sum()), primary_score=primary, leads=leads,
        origin_cohort_sha256=_array_digest({'cohort':cohort}),
        requested_origin_lead_cells=int(window.sum()), evaluable_origin_lead_cells=int(scored_cells.sum()),
        unique_evaluable_target_hours=int(len(np.unique(future[scored_cells]))),
        unique_forecast_origins_with_evaluable_target=int(np.count_nonzero(scored_cells.any(axis=1))),
        origin_lead_cells_are_independent_events=False, annual_primary_gate_applied=annual)


def score_product(*, identity, origins, forecast_time, labels, label_mask, predictions,
                  original_common, eligible168, eligible24, route):
    args=_prepared(identity,origins,forecast_time,labels,label_mask,predictions,original_common,eligible168,eligible24,route)
    origin,future,y,mask,p,long,short,choices,emitted,cohorts,history=args
    eligible = long if identity['product'] in ('PRIMARY_FROZEN','L168_INPUT_ONLY_V1') else short
    panels={name:_cell_panel(origin,future,y,mask,p,eligible,emitted,selected) for name,selected in cohorts.items()}
    kind='MIXED_L168_L24_BY_ORIGIN' if identity['product']=='ROUTED_CLEAN_168_24_V1' else 'FIXED_CHECKPOINT_HISTORY'
    return dict(schema='hydra-phase-a-per-lead-product-score/v1', identity=deepcopy(identity),
        history_identity=dict(mode=kind, fixed_history_hours=None if kind.startswith('MIXED') else (24 if identity['product']=='L24_INPUT_ONLY_V1' else 168),
            per_origin_history_sha256=_array_digest({'history_hours':history}), checkpoint_sha256_by_history=deepcopy(identity['checkpoint_sha256_by_history'])),
        shared_router_origin_counts={name:int(np.count_nonzero(choices==name)) for name in ('L168','L24','UNAVAILABLE')},
        product_history_usage={str(hours):int(np.count_nonzero(history==hours)) for hours in (168,24,0)},
        origin_population=ORIGINS, availability_denominator_is_single_gauge_origin=True,
        cohorts=panels, input_admission_independent_of_verification=True,
        labels_sha256=_array_digest({'labels':y}), label_mask_sha256=_array_digest({'label_mask':mask}),
        coordinates_sha256=_array_digest({'origin_time':origin,'forecast_time':future}),
        computed_router_choices_sha256=_array_digest({'route':choices,'eligible168':long,'eligible24':short}),
        predictions_sha256=_array_digest({'predictions':p}), quality_policy=PRIMARY_APPROVED,
        primary_metric='EQUAL_MEAN_OF_ALL18_QUALIFIED_LEAD_NSE',
        primary_support=dict(distinct_target_hours=168, distinct_utc_dates=28, positive_reference_variance=True),
        negative_predictions_clipped=False, targets_filled=False, technical_failure_policy='RAISE_NOT_ROW_INTERSECTION',
        new_population_score_difference_is_paired_improvement=False,
        routed_weather_soil_attribution_supported=False, protected_assessment_access=False,
        optimizer_updates=0, scaler_fits=0)


def verify_product_report(report, *, identity, origins, forecast_time, labels, label_mask, predictions,
                          original_common, eligible168, eligible24, route, tolerance=1e-10):
    """Independent NumPy calculations and exact support checks, not `_metrics`."""
    require(tolerance==1e-10,'Fixed independent numerical tolerance required')
    args=_prepared(identity,origins,forecast_time,labels,label_mask,predictions,original_common,eligible168,eligible24,route)
    origin,future,y,mask,p,long,short,choices,emitted,cohorts,history=args
    require(report['identity']==identity and set(report['cohorts'])==set(COHORTS), 'Reported identity/cohort inventory differs')
    require(report['origin_population']==ORIGINS
            and report['shared_router_origin_counts']=={k:int(np.count_nonzero(choices==k)) for k in ('L168','L24','UNAVAILABLE')}
            and report['product_history_usage']=={str(k):int(np.count_nonzero(history==k)) for k in (168,24,0)},
            'Independent product/route origin counts differ')
    mixed=identity['product']=='ROUTED_CLEAN_168_24_V1'
    expected_history=dict(mode='MIXED_L168_L24_BY_ORIGIN' if mixed else 'FIXED_CHECKPOINT_HISTORY',
        fixed_history_hours=None if mixed else (24 if identity['product']=='L24_INPUT_ONLY_V1' else 168),
        per_origin_history_sha256=_array_digest({'history_hours':history}),
        checkpoint_sha256_by_history=identity['checkpoint_sha256_by_history'])
    require(report['history_identity']==expected_history,'Independent routed history identity differs')
    largest=0.;checked=0
    eligible=long if identity['product'] in ('PRIMARY_FROZEN','L168_INPUT_ONLY_V1') else short
    for name,cohort in cohorts.items():
        panel=report['cohorts'][name];primary=[]
        require(len(panel['leads'])==18 and panel['requested_origins']==int(np.count_nonzero(cohort)), 'Reported cohort size differs')
        for h,row in enumerate(panel['leads']):
            selected=cohort & emitted & mask[:,h];reference=cohort & mask[:,h];forecast=cohort & emitted
            truth,pred=y[selected,h],p[selected,h];n=len(truth);dates=len(np.unique(future[selected,h]//86400));unique=len(np.unique(future[selected,h]))
            exact=dict(lead_hours=h+1,n=n,requested_origins=int(cohort.sum()),input_eligible_origins=int(np.count_nonzero(cohort&eligible)),
                emitted_forecasts=int(forecast.sum()),valid_target_count=int(reference.sum()),evaluable_forecasts=n,
                distinct_target_hours=unique,distinct_utc_dates=dates,negative_predictions_scored=int((pred<0).sum()),
                negative_predictions_emitted=int((p[forecast,h]<0).sum()),technical_failures=0)
            exact.update(n_requested=exact['requested_origins'],
                data_unavailable_origins=exact['requested_origins']-exact['input_eligible_origins'],
                policy_not_emitted_with_eligible_inputs=exact['input_eligible_origins']-exact['emitted_forecasts'],
                not_emitted_origins=exact['requested_origins']-exact['emitted_forecasts'],
                invalid_or_missing_targets=exact['requested_origins']-exact['valid_target_count'])
            require(all(row[k]==v for k,v in exact.items()),'Independent Phase A support/count mismatch')
            fractions=dict(forecast_availability_fraction=_fraction(exact['emitted_forecasts'],exact['requested_origins']),
                evaluable_coverage_fraction=_fraction(n,exact['requested_origins']),
                coverage_among_valid_targets=_fraction(n,exact['valid_target_count']),
                negative_frequency_among_emitted=_fraction(exact['negative_predictions_emitted'],exact['emitted_forecasts']))
            require(all(row[k]==v for k,v in fractions.items()),'Independent Phase A coverage fraction mismatch')
            wanted={k:None for k in METRICS}
            if n:
                error=pred-truth;mse=float(np.mean(error*error));den=float(np.sum((truth-np.mean(truth))**2))
                wanted.update(physical_mse=mse,rmse=float(np.sqrt(mse)),mae=float(np.mean(np.abs(error))),bias=float(np.mean(error)),
                    nse=None if np.all(truth==truth[0]) or den==0 else 1-float(np.sum(error*error))/den,
                    negative_frequency=float(np.count_nonzero(pred<0)/n))
            qualified=unique>=168 and dates>=28 and wanted['nse'] is not None
            require(row['primary_eligible']==qualified,'Independent primary support differs')
            primary.append(wanted['nse'] if qualified else None)
            for key,value in wanted.items():
                actual=row[key]
                require(value is None and actual is None or value is not None and actual is not None
                        and math.isclose(value,actual,rel_tol=tolerance,abs_tol=tolerance),'Independent Phase A metric mismatch: '+key)
                if value is not None:largest=max(largest,abs(value-actual))
            checked+=1
        wanted=None if any(x is None for x in primary) else float(np.mean(primary))
        require(wanted is None and panel['primary_score'] is None or wanted is not None and panel['primary_score'] is not None
                and math.isclose(wanted,panel['primary_score'],rel_tol=tolerance,abs_tol=tolerance),'Independent Phase A primary mismatch')
    return dict(status='INDEPENDENT_METRICS_AND_COUNTS_MATCH',lead_panels_checked=checked,
                maximum_absolute_metric_difference=largest,relative_absolute_tolerance=tolerance,model_calls=0)


def aggregate_seed_reports(reports):
    """Mean all three seed metrics, never an averaged-prediction ensemble."""
    require(isinstance(reports,dict) and set(reports)==set(SEEDS),'All three final seeds required')
    first=reports[17]
    common_identity={k:v for k,v in first['identity'].items() if k not in ('seed','checkpoint_sha256_by_history','prediction_sha256')}
    for seed,report in reports.items():
        require(report['schema']=='hydra-phase-a-per-lead-product-score/v1' and report['identity']['seed']==seed
                and {k:v for k,v in report['identity'].items() if k not in ('seed','checkpoint_sha256_by_history','prediction_sha256')}==common_identity,
                'Seed reports have different product/input/label scope')
        for key in ('coordinates_sha256','label_mask_sha256','labels_sha256','computed_router_choices_sha256','shared_router_origin_counts'):
            require(report[key]==first[key],'Seed comparison populations or targets differ')
    result={}
    for name in COHORTS:
        panels=[reports[s]['cohorts'][name] for s in SEEDS];values=[p['primary_score'] for p in panels]
        rows=[]
        for h in range(18):
            leads=[p['leads'][h] for p in panels]
            count_keys=('requested_origins','input_eligible_origins','emitted_forecasts','valid_target_count','evaluable_forecasts',
                        'distinct_target_hours','distinct_utc_dates','scored_coordinate_sha256','scoring_mask_sha256',
                        'primary_eligible','primary_ineligibility')
            require(all(all(row[k]==leads[0][k] for k in count_keys) for row in leads),'Seed support must be identical, not pooled')
            metrics={k:(math.fsum(row[k]/3 for row in leads) if all(row[k] is not None for row in leads) else None) for k in METRICS}
            rows.append(dict(lead_hours=h+1,shared_counts={k:leads[0][k] for k in count_keys},mean_metrics=metrics,
                seed_metrics={str(s):{k:reports[s]['cohorts'][name]['leads'][h][k] for k in METRICS} for s in SEEDS}))
        complete=all(x is not None for x in values)
        result[name]=dict(primary_score=math.fsum(x/3 for x in values) if complete else None,
            status='PRIMARY_DEFINED' if complete else 'PRIMARY_UNAVAILABLE',seed_primary_scores={str(s):reports[s]['cohorts'][name]['primary_score'] for s in SEEDS},
            descriptive_seed_standard_deviation=statistics.stdev(values) if complete else None,
            seed_standard_deviation_is_confidence_interval=False,leads=rows)
    return dict(schema='hydra-phase-a-three-seed-product-score/v1',identity=common_identity,seeds=list(SEEDS),cohorts=result,
        all_seeds_retained=True,counts_not_summed_across_seeds=True,mean_metrics_not_ensemble_predictions=True,
        favorable_seed_selected=False,protected_assessment_access=False)


def high_flow_coverage(*, identity, origins, forecast_time, labels, label_mask, predictions,
                       original_common, eligible168, eligible24, route, catalog_row):
    """Known high-flow cells versus unique observed hours and lead support."""
    args=_prepared(identity,origins,forecast_time,labels,label_mask,predictions,original_common,eligible168,eligible24,route)
    origin,future,y,mask,p,long,short,choices,emitted,cohorts,history=args
    _catalog(catalog_row,identity['gauge_id']);threshold=catalog_row['thresholds']['q95_cms']
    high=mask & (y>threshold);supported=high & emitted[:,None]
    # A fixed lead has at most one origin for a valid hour, so an occurrence
    # count of18 means all18 distinct leads, never duplicate samples.
    known_times,known_counts=np.unique(future[high],return_counts=True)
    any_times,supported_counts=np.unique(future[supported],return_counts=True)
    per_lead=[]
    for h in range(18):
        known=int(high[:,h].sum());score=int(supported[:,h].sum())
        per_lead.append(dict(lead_hours=h+1,known_high_target_cells=known,evaluable_high_forecasts=score,
            input_eligible_high_target_cells=int(np.count_nonzero(high[:,h]&(long if identity['product'] in ('PRIMARY_FROZEN','L168_INPUT_ONLY_V1') else short))),
            coverage_among_known_high_targets=_fraction(score,known),
            unique_known_high_target_hours=int(len(np.unique(future[high[:,h],h]))),
            unique_high_hours_with_forecasts=int(len(np.unique(future[supported[:,h],h])))))
    return dict(schema='hydra-phase-a-high-flow-coverage/v1',identity=deepcopy(identity),q95_cms=threshold,
        catalog_row_sha256=digest(catalog_row),
        threshold_population='FROZEN_UNIQUE_APPROVED_GRADIENT_CALENDAR',threshold_recomputed=False,HIGH_is_official_flood_stage=False,
        per_lead=per_lead,known_origin_lead_cells=int(high.sum()),evaluable_origin_lead_cells=int(supported.sum()),
        known_unique_high_target_hours=len(known_times),unique_high_hours_with_any_supported_lead=len(any_times),
        unique_high_hours_with_all18_supported_leads=int(np.count_nonzero(supported_counts==18)),
        unique_high_hours_with_all18_candidate_leads=int(np.count_nonzero(known_counts==18)),
        availability_denominator_multiplied_by_models_or_seeds=False,unobserved_high_flow_inferred=False,
        protected_assessment_access=False)


def _catalog(row,gauge):
    start,end=validate_catalog_row(row)
    require(row['gauge_id']==gauge and row['fold']=='final_development' and start==START and end==ASSESSMENT_START,
            'Original final-development fixed catalog required')


def fixed_events(*, identity, origins, forecast_time, labels, label_mask, predictions,
                 original_common, eligible168, eligible24, route, catalog_row):
    """All fixed observed events; no annual primary gate or new timing criterion."""
    args=_prepared(identity,origins,forecast_time,labels,label_mask,predictions,original_common,eligible168,eligible24,route)
    origin,future,y,mask,p,long,short,choices,emitted,cohorts,history=args
    _catalog(catalog_row,identity['gauge_id'])
    eligible=long if identity['product'] in ('PRIMARY_FROZEN','L168_INPUT_ONLY_V1') else short
    rows=[]
    for event in catalog_row['observed_high_flow_events']:
        window=(future>=event['start_epoch'])&(future<event['end_epoch'])
        peak=future==event['peak_epoch'];peak_supported=[]
        result=_cell_panel(origin,future,y,mask,p,eligible,emitted,cohorts['ALL_REQUESTED'],window,annual=False)
        for h,lead in enumerate(result['leads']):
            at=peak[:,h];scored=at&mask[:,h]&emitted
            require(int(at.sum())<=1,'One original forecast per peak/lead required')
            available=bool(scored.any());peak_supported.append(available)
            if np.any(at&mask[:,h]):
                require(np.all(y[at&mask[:,h],h]==event['observed_peak_q_cms']), 'Frozen observed peak reference differs')
            lead['peak_forecast_available']=bool(np.any(at&emitted))
            lead['peak_candidate_requested']=bool(at.any())
            lead['peak_reference_known_in_catalog']=True
            lead['peak_target_admissible']=bool(np.any(at&mask[:,h]))
            lead['observed_peak_forecast']=None if not available else dict(observed_q_cms=float(y[scored,h][0]),
                prediction_q_cms=float(p[scored,h][0]),error_cms=float(p[scored,h][0]-y[scored,h][0]))
        rows.append(dict(event_id=event['event_id'],rank=event['observed_peak_rank'],representative_top3=event['representative_top3'],
            start_utc=event['start_utc'],end_exclusive_utc=event['end_exclusive_utc'],observed_peak_time_utc=event['observed_peak_time_utc'],
            observed_peak_q_cms=event['observed_peak_q_cms'],reference_approved_hours=event['reference_approved_hours'],
            reference_requested_hours=event['reference_requested_hours'],reference_unusable_hours=event['reference_unusable_hours'],
            observed_peak_supported_leads=[h+1 for h,ok in enumerate(peak_supported) if ok],
            observed_peak_candidate_leads=[h+1 for h in range(18) if peak[:,h].any()],
            observed_peak_supported_any_lead=any(peak_supported),observed_peak_supported_all18_leads=all(peak_supported),
            metrics=result))
    return dict(schema='hydra-phase-a-fixed-event-scores/v1',identity=deepcopy(identity),events=rows,catalog_event_count=len(rows),
        catalog_row_sha256=digest(catalog_row),
        events_with_any_evaluable_forecast=sum(x['metrics']['evaluable_origin_lead_cells']>0 for x in rows),
        observed_peaks_with_any_forecast=sum(x['observed_peak_supported_any_lead'] for x in rows),
        observed_peaks_with_all18_forecasts=sum(x['observed_peak_supported_all18_leads'] for x in rows),
        representative_events_reselected=False,events_with_no_forecasts_retained=True,
        episodes_are_proven_independent=False,peak_timing_or_false_alarm_acceptance_rule_added=False,
        annual_NSE_gate_used_for_short_events=False,protected_assessment_access=False)
