#!/usr/bin/env python3
"""Sanitized Phase A evidence/report publisher; no model or target readers.

The parent passes the completed A5 index and a hash-verifying JSON reader. Public
outputs contain aggregate metrics/counts and SVG figures, never raw arrays,
checkpoints or private per-origin provenance. All seeds and empty cohorts remain.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import stat
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hydra_directflow.contract import ARCHITECTURES,GAUGE_IDS
from hydra_directflow.coverage_extension_scoring import PRODUCTS,COHORTS,SEEDS,METRICS,ORIGINS,START,END
from hydra_directflow.reference import digest,require,ASSESSMENT_START
from hydra_directflow.trainer_bundle import _array_digest

NAMES={'03161000':'Jefferson','03164000':'Galax','03479000':'Sugar Grove'}
MODEL_NAMES={'LSTM':'LSTM','Transformer':'Transformer','GRU_Transformer':'GRU–Transformer','Mamba2_Transformer':'Mamba-2–Transformer'}
PRODUCT_NAMES={'PRIMARY_FROZEN':'Original saved L168 common','L168_INPUT_ONLY_V1':'L168 input-only',
    'L24_INPUT_ONLY_V1':'L24 input-only','ROUTED_CLEAN_168_24_V1':'Clean L168→L24 router'}
COUNT_KEYS=('requested_origins','input_eligible_origins','emitted_forecasts','valid_target_count','evaluable_forecasts',
    'data_unavailable_origins','policy_not_emitted_with_eligible_inputs','not_emitted_origins','invalid_or_missing_targets',
    'technical_failures','distinct_target_hours','distinct_utc_dates','negative_predictions_scored','negative_predictions_emitted')
FRACTION_KEYS=('forecast_availability_fraction','evaluable_coverage_fraction','coverage_among_valid_targets','negative_frequency_among_emitted')
ACCOUNTING_KEYS=('started','deadline','entered_cases','confirmed_cases','entered_batches','confirmed_batches',
    'entered_models','confirmed_models','read_reserved','output_reserved','infrastructure_resumes','peak_combined_rss',
    'peak_mps','minimum_free','last_clock','original_campaign_deadline','elapsed_seconds','original_optimizer_updates_before','original_optimizer_updates_after')


def _json(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def _sha(raw):return hashlib.sha256(raw).hexdigest()
def _hex(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def _same(a,b):
    return a is None and b is None or a is not None and b is not None and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)


def _identity(value,expected,index):
    identity=value['identity']
    require(tuple(identity[k] for k in ('gauge_id','architecture','seed','product'))==expected
            and identity['phase_plan_sha256']==index['phase_plan_sha256']
            and identity['prediction_barrier_sha256']==index['prediction_barrier_sha256']
            and value.get('protected_assessment_access') is False,'Product evidence has a different model, phase or prediction barrier')


def _validate_panel(panel):
    require(len(panel['leads'])==18 and [r['lead_hours'] for r in panel['leads']]==list(range(1,19)),
            'Complete H18 reporting required')
    primary=[]
    for row in panel['leads']:
        require(all(type(row[k]) is int and row[k]>=0 for k in COUNT_KEYS)
                and row['technical_failures']==0
                and row['evaluable_forecasts']==row['n']<=row['emitted_forecasts']<=row['requested_origins']
                and row['evaluable_forecasts']<=row['valid_target_count']<=row['requested_origins'],
                'Complete successful product counts required; failures cannot be omitted')
        require(all(row[k] is None or type(row[k]) in (int,float) and math.isfinite(row[k]) for k in METRICS),
                'Finite metric values or explicit unavailability required')
        require(row['negative_predictions_scored']<=row['n'] and row['negative_predictions_emitted']<=row['emitted_forecasts']
                and (all(row[k] is None for k in METRICS) if row['n']==0 else all(row[k] is not None for k in METRICS if k!='nse')),
                'Missing metrics are not zero-error or omitted negative predictions')
        qualified=row['distinct_target_hours']>=168 and row['distinct_utc_dates']>=28 and row['nse'] is not None
        require(row['primary_eligible']==qualified,'Primary support floor differs')
        primary.append(row['nse'] if qualified else None)
    wanted=None if any(v is None for v in primary) else math.fsum(v/18 for v in primary)
    require(_same(wanted,panel['primary_score'])
            and panel['status']==('PRIMARY_UNAVAILABLE' if wanted is None else 'PRIMARY_DEFINED'),
            'Primary mean may not omit unsupported leads')


def _shared_coverage(score):
    panel=score['cohorts']['ALL_REQUESTED']
    return dict(candidate_origins=score['origin_population'],route_counts=score['shared_router_origin_counts'],
        cohort_origins={name:score['cohorts'][name]['cohort_origins'] for name in COHORTS},
        per_lead=[{k:row[k] for k in ('lead_hours','requested_origins','input_eligible_origins','emitted_forecasts',
            'valid_target_count','evaluable_forecasts','data_unavailable_origins','policy_not_emitted_with_eligible_inputs',
            'forecast_availability_fraction','evaluable_coverage_fraction','coverage_among_valid_targets')} for row in panel['leads']])


def _shared_high(high):
    return {k:high[k] for k in ('q95_cms','per_lead','known_origin_lead_cells','evaluable_origin_lead_cells',
        'known_unique_high_target_hours','unique_high_hours_with_any_supported_lead',
        'unique_high_hours_with_all18_supported_leads','unique_high_hours_with_all18_candidate_leads')}


def _event_counts(events):
    return dict(catalog_event_count=events['catalog_event_count'],
        events_with_any_evaluable_forecast=events['events_with_any_evaluable_forecast'],
        observed_peaks_with_any_forecast=events['observed_peaks_with_any_forecast'],
        observed_peaks_with_all18_forecasts=events['observed_peaks_with_all18_forecasts'],
        events=[dict(event_id=e['event_id'],rank=e['rank'],representative_top3=e['representative_top3'],
            start_utc=e['start_utc'],end_exclusive_utc=e['end_exclusive_utc'],
            observed_peak_supported_leads=e['observed_peak_supported_leads'],
            observed_peak_candidate_leads=e['observed_peak_candidate_leads'],
            reference_approved_hours=e['reference_approved_hours'],reference_requested_hours=e['reference_requested_hours'],
            unique_evaluable_target_hours=e['metrics']['unique_evaluable_target_hours'],
            evaluable_origin_lead_cells=e['metrics']['evaluable_origin_lead_cells'],
            per_lead=[{k:r[k] for k in ('lead_hours','requested_origins','input_eligible_origins','emitted_forecasts',
                'valid_target_count','evaluable_forecasts','peak_forecast_available','peak_candidate_requested','peak_target_admissible')}
                for r in e['metrics']['leads']]) for e in events['events']])


def _public_score(score):
    return dict(seed=score['identity']['seed'],history_identity=dict(mode=score['history_identity']['mode'],
        fixed_history_hours=score['history_identity']['fixed_history_hours']),
        cohorts={name:dict(status=p['status'],primary_score=p['primary_score'],requested_origins=p['requested_origins'],
            unique_evaluable_target_hours=p['unique_evaluable_target_hours'],
            leads=[{k:row[k] for k in ('lead_hours',*COUNT_KEYS,*FRACTION_KEYS,*METRICS,
                                      'primary_eligible','primary_ineligibility','undefined')} for row in p['leads']])
            for name,p in score['cohorts'].items()})


def _public_events(events):
    return [dict(event_id=e['event_id'],rank=e['rank'],representative_top3=e['representative_top3'],
        leads=[{k:row[k] for k in ('lead_hours','n','rmse','mae','bias','physical_mse','negative_frequency',
            'observed_peak_forecast','peak_forecast_available','peak_target_admissible','undefined')}
            for row in e['metrics']['leads']]) for e in events['events']]


def build_publication(index,read_json_descriptor,accounting):
    """Validate complete A5 evidence and return sanitized file payloads.

    The injected reader must authenticate descriptor bytes before JSON decoding.
    It is called only for score/verification/high-flow/event/aggregate reports;
    target or prediction arrays are not requested by this publication path.
    """
    require(index.get('schema')=='hydra-phase-a-evaluation-index/v1'
            and index.get('status')=='ALL144_PRODUCTS_SCORED_AND_INDEPENDENTLY_VERIFIED'
            and _hex(index.get('phase_plan_sha256')) and _hex(index.get('prediction_barrier_sha256'))
            and _hex(index.get('original_plan_sha256')) and index.get('protected_assessment_access') is False,
            'Completed development evaluation and prediction barrier required')
    require(accounting.get('plan_sha256')==index['phase_plan_sha256'], 'Accounting belongs to a different Phase A plan')
    for entered,confirmed,maximum in (('entered_cases','confirmed_cases',1280000),('entered_batches','confirmed_batches',20000),('entered_models','confirmed_models',145)):
        require(type(accounting.get(entered)) is int and type(accounting.get(confirmed)) is int
                and 0<accounting[confirmed]<=accounting[entered]<=maximum,'Finite charged/confirmed inference accounting required')
    for key,maximum in (('read_reserved',8*1024**3),('output_reserved',1024**3),('infrastructure_resumes',1),
                        ('peak_combined_rss',4*1024**3),('peak_mps',2*1024**3)):
        require(type(accounting.get(key)) is int and 0<=accounting[key]<=maximum,'Phase A resource accounting bound differs: '+key)
    require(type(accounting.get('minimum_free')) is int and accounting['minimum_free']>=8*1024**3,
            'Original free-disk floor must remain satisfied')
    for key in ('started','last_clock','deadline','original_campaign_deadline'):
        require(type(accounting.get(key)) in (int,float) and math.isfinite(accounting[key]),'Persistent execution clocks required')
    require(accounting['started']<=accounting['last_clock']<=accounting['deadline']<=accounting['original_campaign_deadline']
            and accounting['last_clock']-accounting['started']<=21600
            and accounting['deadline']-accounting['started']<=21600,'Persistent six-hour/original campaign deadline exceeded')
    require(accounting.get('original_optimizer_updates_before')==accounting.get('original_optimizer_updates_after')==1151690,
            'Original scientific optimizer counter must remain unchanged')
    common_checks=index.get('original_common_checks',[])
    required_jobs={f'final_development__{g}__{a}__full_L{h}__seed{s}' for g in GAUGE_IDS for a in ARCHITECTURES for h in (168,24) for s in SEEDS}
    require(len(common_checks)==72 and {x['job_id'] for x in common_checks}==required_jobs
            and all(x['status']=='ORIGINAL_COMMON_METRICS_REPRODUCED' for x in common_checks),
            'All72 original L168/L24 common-metric reconciliation checks are required')
    expected={(g,a,s,p) for g in GAUGE_IDS for a in ARCHITECTURES for s in SEEDS for p in PRODUCTS}
    records=index['products'];actual=[tuple(r[k] for k in ('gauge_id','architecture','seed','product')) for r in records]
    require(len(actual)==144 and set(actual)==expected,'All144 products/seeds must remain; no favorable subset')
    scores={};events_by_key={};shared={};highs={};event_counts={};source_descriptors=[]
    for key,record in zip(actual,records):
        require(set(record['artifacts'])=={'score','verification','high_flow','events'},'Full product evidence set required')
        loaded={name:read_json_descriptor(desc) for name,desc in record['artifacts'].items()}
        source_descriptors.extend(record['artifacts'].values())
        score,verification,high,events=(loaded[k] for k in ('score','verification','high_flow','events'))
        for body in (score,high,events):_identity(body,key,index)
        require(score['schema']=='hydra-phase-a-per-lead-product-score/v1' and set(score['cohorts'])==set(COHORTS)
                and score['origin_population']==ORIGINS and score['availability_denominator_is_single_gauge_origin'] is True
                and score['input_admission_independent_of_verification'] is True
                and score['routed_weather_soil_attribution_supported'] is False
                and score['negative_predictions_clipped'] is False and score['targets_filled'] is False
                and score['optimizer_updates']==score['scaler_fits']==0,'Truthful Phase A source and scoring rules required')
        require(verification['status']=='INDEPENDENT_METRICS_AND_COUNTS_MATCH'
                and verification['lead_panels_checked']==90 and verification['relative_absolute_tolerance']==1e-10,
                'Independent metric/count verification is required')
        if key[3]=='ROUTED_CLEAN_168_24_V1':
            require(score['history_identity']['mode']=='MIXED_L168_L24_BY_ORIGIN'
                    and score['history_identity']['fixed_history_hours'] is None,'Routed stream cannot claim one fixed history')
        for panel in score['cohorts'].values():_validate_panel(panel)
        require(events['representative_events_reselected'] is False and events['events_with_no_forecasts_retained'] is True
                and events['episodes_are_proven_independent'] is False and events['annual_NSE_gate_used_for_short_events'] is False,
                'Fixed events and unavailable cases must remain explicit')
        require(len(events['events'])==events['catalog_event_count']
                and len({e['event_id'] for e in events['events']})==events['catalog_event_count']
                and [e['rank'] for e in events['events']]==list(range(1,events['catalog_event_count']+1))
                and all(e['representative_top3']==(e['rank']<=3) for e in events['events']),
                'Full fixed catalog order and representative selection required')
        g,a,s,product=key;group=(g,product)
        views=(_shared_coverage(score),_shared_high(high),_event_counts(events))
        for destination,value in zip((shared,highs,event_counts),views):
            if group in destination:require(destination[group]==value,'Coverage cannot depend on model/seed or be multiplied')
            else:destination[group]=value
        scores[key]=score;events_by_key[key]=events
    coverage=[]
    for gauge in GAUGE_IDS:
        base=shared[gauge,'PRIMARY_FROZEN'];c=base['cohort_origins'];routes=base['route_counts']
        require(c['ALL_REQUESTED']==ORIGINS and sum(c[k] for k in COHORTS[1:])==ORIGINS
                and routes=={'L168':c['ORIGINAL_COMMON']+c['ADDITIONAL_L168'],'L24':c['L24_FALLBACK'],'UNAVAILABLE':c['UNAVAILABLE']},
                'Unique-origin partition and route counts differ')
        require(all(shared[gauge,p]['cohort_origins']==c and shared[gauge,p]['route_counts']==routes for p in PRODUCTS),
                'Policy reports have changed the origin population')
        require(shared[gauge,'L24_INPUT_ONLY_V1']['per_lead']==shared[gauge,'ROUTED_CLEAN_168_24_V1']['per_lead'],
                'Clean L24 and fixed router must share issuance/evaluable coverage')
        coverage.append(dict(gauge_id=gauge,candidate_origins=ORIGINS,origin_cohorts=c,route_counts=routes,
            computed_router_choices_sha256=scores[gauge,ARCHITECTURES[0],17,'PRIMARY_FROZEN']['computed_router_choices_sha256'],
            original_common_mask_sha256=scores[gauge,ARCHITECTURES[0],17,'PRIMARY_FROZEN']['cohorts']['ORIGINAL_COMMON']['origin_cohort_sha256'],
            input_reason_counts=index['coverage_by_gauge'][gauge].get('primary_reason_counts',{}),
            overlapping_input_reasons=index['coverage_by_gauge'][gauge].get('overlapping_input_reasons',{}),
            products={p:dict(coverage=shared[gauge,p],high_flow=highs[gauge,p],fixed_event_coverage=event_counts[gauge,p]) for p in PRODUCTS}))
    agg_expected={(g,a,p) for g in GAUGE_IDS for a in ARCHITECTURES for p in PRODUCTS}
    require(len(index['aggregates'])==48 and {tuple(x[k] for k in ('gauge_id','architecture','product')) for x in index['aggregates']}==agg_expected,
            'All48 three-seed product aggregates required')
    aggregates={}
    for descriptor in index['aggregates']:
        key=tuple(descriptor[k] for k in ('gauge_id','architecture','product'));g,a,p=key
        body=read_json_descriptor(descriptor['report']);source_descriptors.append(descriptor['report'])
        require(body['schema']=='hydra-phase-a-three-seed-product-score/v1' and body['seeds']==list(SEEDS)
                and all(body['identity'][k]==v for k,v in zip(('gauge_id','architecture','product'),key))
                and body['all_seeds_retained'] is True and body['favorable_seed_selected'] is False
                and body['mean_metrics_not_ensemble_predictions'] is True,'All seed metrics must be retained')
        clean={}
        for cohort in COHORTS:
            original=[scores[g,a,s,p]['cohorts'][cohort] for s in SEEDS];recorded=body['cohorts'][cohort]
            points=[x['primary_score'] for x in original];mean=None if any(v is None for v in points) else math.fsum(v/3 for v in points)
            spread=None if mean is None else statistics.stdev(points)
            require(_same(mean,recorded['primary_score']) and _same(spread,recorded['descriptive_seed_standard_deviation'])
                    and recorded['seed_primary_scores']=={str(s):scores[g,a,s,p]['cohorts'][cohort]['primary_score'] for s in SEEDS},
                    'Three-seed primary aggregation differs')
            leads=[]
            for h in range(18):
                old=[x['leads'][h] for x in original];row=recorded['leads'][h]
                require(all(all(v[k]==old[0][k] for k in ('n','requested_origins','emitted_forecasts','valid_target_count','scored_coordinate_sha256','scoring_mask_sha256')) for v in old),
                        'Seed supports must agree exactly')
                means={metric:None if any(v[metric] is None for v in old) else math.fsum(v[metric]/3 for v in old) for metric in METRICS}
                require(all(_same(means[k],row['mean_metrics'][k]) for k in METRICS),'Three-seed lead aggregation differs')
                leads.append(dict(lead_hours=h+1,n=old[0]['n'],requested_origins=old[0]['requested_origins'],
                    valid_target_count=old[0]['valid_target_count'],emitted_forecasts=old[0]['emitted_forecasts'],
                    primary_eligible=old[0]['primary_eligible'],primary_ineligibility=old[0]['primary_ineligibility'],mean_metrics=means))
            clean[cohort]=dict(status=recorded['status'],primary_score=mean,seed_primary_scores={str(s):points[i] for i,s in enumerate(SEEDS)},
                descriptive_seed_standard_deviation=spread,spread_is_confidence_interval=False,leads=leads)
        aggregates[key]=clean
    paired=[]
    for gauge in GAUGE_IDS:
        for architecture in ARCHITECTURES:
            differences={};lead_differences=[]
            for seed in SEEDS:
                old=scores[gauge,architecture,seed,'PRIMARY_FROZEN']['cohorts']['ORIGINAL_COMMON']
                routed_common=scores[gauge,architecture,seed,'ROUTED_CLEAN_168_24_V1']['cohorts']['ORIGINAL_COMMON']
                long_common=scores[gauge,architecture,seed,'L168_INPUT_ONLY_V1']['cohorts']['ORIGINAL_COMMON']
                for current in (routed_common,long_common):
                    require(_same(old['primary_score'],current['primary_score'])
                            and all(all(_same(o[m],v[m]) for m in METRICS) and o['scored_coordinate_sha256']==v['scored_coordinate_sha256']
                                    for o,v in zip(old['leads'],current['leads'])), 'Original common L168 results changed')
                left=scores[gauge,architecture,seed,'ROUTED_CLEAN_168_24_V1']['cohorts']['ALL_REQUESTED']
                right=scores[gauge,architecture,seed,'L24_INPUT_ONLY_V1']['cohorts']['ALL_REQUESTED']
                require(all(x['scored_coordinate_sha256']==y['scored_coordinate_sha256'] and x['scoring_mask_sha256']==y['scoring_mask_sha256']
                            for x,y in zip(left['leads'],right['leads'])), 'Routed/L24 accuracy contrast must use identical cases')
                differences[str(seed)]=None if left['primary_score'] is None or right['primary_score'] is None else left['primary_score']-right['primary_score']
            for h in range(18):
                left=aggregates[gauge,architecture,'ROUTED_CLEAN_168_24_V1']['ALL_REQUESTED']['leads'][h]
                right=aggregates[gauge,architecture,'L24_INPUT_ONLY_V1']['ALL_REQUESTED']['leads'][h]
                lead_differences.append(dict(lead_hours=h+1,n=left['n'],mean_metric_differences={m:None if left['mean_metrics'][m] is None or right['mean_metrics'][m] is None
                    else left['mean_metrics'][m]-right['mean_metrics'][m] for m in METRICS}))
            paired.append(dict(gauge_id=gauge,architecture=architecture,positive_product='ROUTED_CLEAN_168_24_V1',
                negative_product='L24_INPUT_ONLY_V1',population='IDENTICAL_L24_INPUT_ELIGIBLE_PER_LEAD_EVALUABLE_CASES',
                primary_difference=None if any(x is None for x in differences.values()) else math.fsum(x/3 for x in differences.values()),
                seed_primary_differences=differences,leads=lead_differences,paired_by_exact_coordinate_and_mask=True,
                uncertainty_intervals_computed=False,weather_soil_attribution=False))
    summary=dict(schema='hydra-phase-a-public-results/v1',status='COMPLETE_DESCRIPTIVE_DEVELOPMENT_SENSITIVITY',
        phase_plan_sha256=index['phase_plan_sha256'],prediction_barrier_sha256=index['prediction_barrier_sha256'],
        original_plan_sha256=index['original_plan_sha256'],products=list(PRODUCTS),cohorts=list(COHORTS),seeds=list(SEEDS),
        origin_calendar_utc=['2023-07-31T00:00:00Z','2024-07-30T06:00:00Z'],horizon_hours=list(range(1,19)),
        architectures=list(ARCHITECTURES),coverage_by_gauge=coverage,
        routed_prediction_array_sha256={g:{a:{str(seed):scores[g,a,seed,'ROUTED_CLEAN_168_24_V1']['predictions_sha256'] for seed in SEEDS} for a in ARCHITECTURES} for g in GAUGE_IDS},
        target_arrays_sha256_by_gauge={g:{k:scores[g,ARCHITECTURES[0],17,'PRIMARY_FROZEN'][k]
            for k in ('labels_sha256','label_mask_sha256')} for g in GAUGE_IDS},
        primary_scores=[dict(gauge_id=g,architecture=a,product=p,cohorts={name:{k:v for k,v in panel.items() if k!='leads'} for name,panel in value.items()})
                        for (g,a,p),value in sorted(aggregates.items())],paired_routed_vs_l24=paired,
        accounting={k:accounting[k] for k in ACCOUNTING_KEYS if k in accounting},
        accounting_snapshot_scope='REPORT_BUILD_START_FINAL_CLOSURE_LEDGER_SUPERSEDES_RESOURCE_TOTALS',
        source_report_inventory_sha256=digest(source_descriptors),all144_product_seed_reports_retained=True,
        counts_multiplied_by_models_or_seeds=False,new_population_error_difference_is_paired_improvement=False,
        routed_weather_soil_attribution_supported=False,raw_arrays_or_checkpoints_published=False,
        additional_uncertainty_run=False,Phase_B_real_execution_authorized=False,protected_assessment_access=False)
    # The overview retains catalog identities/counts; detailed lead-level event
    # counts remain in the split detail files rather than inflating the overview.
    for gauge in summary['coverage_by_gauge']:
        for product,item in gauge['products'].items():
            item['fixed_event_coverage']={**item['fixed_event_coverage'],
                'events':[{k:v for k,v in e.items() if k!='per_lead'} for e in item['fixed_event_coverage']['events']]}
    files={}
    for gauge in GAUGE_IDS:
        for product in PRODUCTS:
            for architecture in ARCHITECTURES:
                body=dict(schema='hydra-phase-a-public-gauge-product-model-details/v1',gauge_id=gauge,product=product,
                    architecture=architecture,phase_plan_sha256=index['phase_plan_sha256'],
                    coverage=shared[gauge,product],high_flow=highs[gauge,product],
                    fixed_event_coverage=event_counts[gauge,product],aggregate=aggregates[gauge,architecture,product],
                    seeds={str(seed):dict(score=_public_score(scores[gauge,architecture,seed,product]),
                        fixed_event_metrics=_public_events(events_by_key[gauge,architecture,seed,product])) for seed in SEEDS})
                name=f'details/{gauge}-{product.lower()}-{architecture.lower()}.json'
                raw=(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
                require(len(raw)<1_000_000,'Public detail file exceeds repository bound; split before publishing')
                files[name]=raw
    summary['detail_files']={name:dict(bytes=len(raw),sha256=_sha(raw)) for name,raw in files.items()}
    files['summary.json']=_json(summary);require(len(files['summary.json'])<1_000_000,'Public overview exceeds repository bound')
    files['review.md']=markdown(summary).encode()
    return dict(summary=summary,files=files)


def _value(x):return 'unavailable' if x is None else f'{x:.4f}'


def markdown(summary):
    out=['# Phase A: input-only admission and clean L168→L24 routing','',
         '**Descriptive 2023–24 development sensitivity.** Existing full-input models, saved transforms and selected-best weights are reused. No new training, short-gap repair or protected assessment is included.','',
         'At origin/cutoff C, exact Q histories end at C−2: L168 covers C−169…C−2 and L24 covers C−25…C−2. The two-hour availability delay is simulated; today\'s revised approved observations do not prove historical delivery at that delay. One eligible issued HRRR cycle and the separately timestamped soil initial state retain their frozen rules.','',
         'Forecast availability uses one denominator of 8,766 candidate origins per gauge. It is not multiplied by architectures or seeds. Verification is independent at each lead; an invalid H3 reference does not cancel an admissible H18 reference.','',
         '## Unique-origin coverage','',
         '| Gauge | Original common | Additional L168 | Clean L24 fallback | Unavailable | Routed forecast availability |',
         '|---|---:|---:|---:|---:|---:|']
    for row in summary['coverage_by_gauge']:
        c=row['origin_cohorts'];available=row['candidate_origins']-c['UNAVAILABLE']
        out.append(f"| {NAMES[row['gauge_id']]} | {c['ORIGINAL_COMMON']:,} | {c['ADDITIONAL_L168']:,} | {c['L24_FALLBACK']:,} | {c['UNAVAILABLE']:,} | {available:,}/{row['candidate_origins']:,} ({available/row['candidate_origins']:.2%}) |")
    out += ['', 'The router selects L168 for the entire H18 vector when its inputs qualify, otherwise clean L24, otherwise unavailable. Route selection never uses future observations, target-quality masks or forecast errors. Ninety percent is a research aspiration, not a pass condition.','',
            '## Known high flow and fixed peaks','',
            '| Gauge | Product | Known-high origin/lead cells covered | Unique known-high hours with any lead | Unique hours with all 18 leads | Observed catalog peaks with any / all leads |',
            '|---|---|---:|---:|---:|---:|']
    for row in summary['coverage_by_gauge']:
        for product in PRODUCTS:
            item=row['products'][product];h=item['high_flow'];e=item['fixed_event_coverage']
            out.append(f"| {NAMES[row['gauge_id']]} | {PRODUCT_NAMES[product]} | {h['evaluable_origin_lead_cells']:,}/{h['known_origin_lead_cells']:,} | {h['unique_high_hours_with_any_supported_lead']}/{h['known_unique_high_target_hours']} | {h['unique_high_hours_with_all18_supported_leads']}/{h['unique_high_hours_with_all18_candidate_leads']} | {e['observed_peaks_with_any_forecast']}/{e['catalog_event_count']} · {e['observed_peaks_with_all18_forecasts']}/{e['catalog_event_count']} |")
    out += ['', 'HIGH uses the fixed gradient-only q95 threshold, not an official flood stage. Target cells, unique valid hours and catalog peaks are different denominators; catalog episodes are not proven independent floods. No missing target is imputed and no unseen peak is inferred. All originally fixed representative events remain, including unsupported cases.','',
            '## Accuracy on explicit populations','',
            'The detail files retain every gauge, architecture, seed, product, cohort and lead, including empty/unsupported panels. Equal-lead mean NSE is unavailable unless all 18 leads meet the unchanged 168-hour/28-date/positive-variance support rules. Raw negative predictions and both emitted/evaluable negative-frequency counts remain reported.','',
            '| Gauge | Model | Routed minus L24 mean NSE on identical support |', '|---|---|---:|']
    for row in summary['paired_routed_vs_l24']:
        out.append(f"| {NAMES[row['gauge_id']]} | {MODEL_NAMES[row['architecture']]} | {_value(row['primary_difference'])} |")
    out += ['', 'Only the routed-versus-L24 contrast above is labeled paired, after exact coordinate/mask equality. Errors on newly recovered cases are reported separately; a difference from original-common scores is not a paired improvement. Original-common routed/L168 scores reproduce the frozen primary. All three seed metrics are averaged, never an averaged-prediction ensemble.','',
            'There is no Q-only L24 model. A routed-full versus unrouted-Q comparison cannot isolate weather or soil. No new policy uncertainty family is claimed; the original 42/30 contrast intervals do not automatically apply to this extension. Galax residual dependence and historical-vintage/availability limitations remain relevant.','',
            '## Execution and next decision','',
            'The accompanying accounting retains charged model/forward work, failed/replayed work, resource use and the original training counter. The complete prediction barrier precedes target loading. Raw arrays, checkpoints and detailed per-origin source/route provenance remain private.','',
            'Phase B remains a decision packet, not execution authority. If the remaining gaps justify study, predeclare bounded Q-only interior/trailing policies, route order, paired artificial gaps, clean-degradation/peak/false-alarm/negative-output/failure tolerances and measured compute costs before seeking authorization. Do not treat coverage below 90% as permission to add trials.','']
    return '\n'.join(out)


def render_aggregate_figures(summary,directory,*,png_previews=False):
    """Three SVG figures; use the existing plotting runtime, never model code."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'hydra-phase-a-v1',
        'axes.spines.top':False,'axes.spines.right':False})
    colors=('#777777','#0d6074','#bf5700','#58752d');model_colors=('#bf5700','#0d6074','#815ca4','#58752d')
    rows={r['gauge_id']:r for r in summary['coverage_by_gauge']};lead=np.arange(1,19);outputs=[]
    for kind in ('valid_target_coverage','high_flow_coverage','paired_routed_l24'):
        fig,axes=plt.subplots(1,3,figsize=(14,4.7),sharex=True,sharey=True)
        for gi,(gauge,ax) in enumerate(zip(GAUGE_IDS,axes)):
            if kind=='paired_routed_l24':
                for mi,model in enumerate(ARCHITECTURES):
                    row=next(x for x in summary['paired_routed_vs_l24'] if x['gauge_id']==gauge and x['architecture']==model)
                    value=[np.nan if x['mean_metric_differences']['nse'] is None else x['mean_metric_differences']['nse'] for x in row['leads']]
                    ax.plot(lead,value,color=model_colors[mi],lw=1.8,label=MODEL_NAMES[model])
                ax.axhline(0,color='#777777',lw=.8)
            else:
                for pi,product in enumerate(PRODUCTS):
                    data=rows[gauge]['products'][product]
                    values=[x['coverage_among_valid_targets'] for x in data['coverage']['per_lead']] if kind=='valid_target_coverage' else [x['coverage_among_known_high_targets'] for x in data['high_flow']['per_lead']]
                    ax.plot(lead,[np.nan if x is None else 100*x for x in values],color=colors[pi],lw=2 if pi==3 else 1.4,
                            ls='--' if pi in (0,2) else '-',label=PRODUCT_NAMES[product])
                ax.set_ylim(0,102)
            ax.set_title(NAMES[gauge]+' · '+gauge);ax.set_xlabel('Lead (hours)');ax.set_xticks([1,3,6,9,12,15,18]);ax.grid(alpha=.15)
        titles={'valid_target_coverage':'Evaluable forecast coverage among approved targets','high_flow_coverage':'Forecast coverage of known high-flow targets','paired_routed_l24':'Routed versus L24 accuracy on identical evaluable cases'}
        axes[0].set_ylabel('ΔNSE: routed − L24' if kind=='paired_routed_l24' else 'Coverage (%)')
        prefix='SYNTHETIC QUALIFICATION · NOT STUDY RESULTS · ' if summary.get('synthetic_qualification') else 'Phase A development · '
        fig.suptitle(prefix+titles[kind],fontsize=14,y=1.03)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,-.045),fontsize=9)
        note='Three-seed mean metric differences; no new uncertainty intervals.' if kind=='paired_routed_l24' else 'One gauge/origin denominator; not multiplied by models or seeds. L24 and router share availability and may overlap.'
        fig.text(.5,-.075,note,ha='center',fontsize=9);fig.tight_layout()
        path=directory/('phase-a-'+kind+'.svg');require(not path.exists(),'Preserve existing figure output')
        fig.savefig(path,bbox_inches='tight',metadata={'Date':None,'Creator':'HYDRA DirectFlow Phase A'})
        if png_previews:
            preview=directory/'private-png-previews';preview.mkdir(exist_ok=True)
            fig.savefig(preview/(path.stem+'.png'),dpi=160,bbox_inches='tight')
        plt.close(fig)
        raw=path.read_bytes();outputs.append(dict(path=path.name,bytes=len(raw),sha256=_sha(raw)))
    return outputs


def validate_event_plot_data(summary,gauge,value):
    """Validate a parent-authenticated, post-barrier plotting bundle."""
    import numpy as np
    require(gauge in GAUGE_IDS and value['phase_plan_sha256']==summary['phase_plan_sha256']
            and value['prediction_barrier_sha256']==summary['prediction_barrier_sha256'],
            'Event plotting requires the completed prediction barrier identity')
    origin=np.asarray(value['origin_time']);future=np.asarray(value['forecast_time'])
    require(origin.dtype==np.dtype('<i8') and np.array_equal(origin,np.arange(START,END,3600,dtype=np.int64))
            and future.dtype==np.dtype('<i8') and np.array_equal(future,origin[:,None]+np.arange(1,19)*3600)
            and np.all(future<ASSESSMENT_START),'Event plot coordinates must remain H18-safe development')
    common=np.asarray(value['original_common']);route=np.asarray(value['route'])
    require(common.shape==route.shape==(ORIGINS,) and common.dtype==np.dtype(bool) and route.dtype.kind=='U'
            and np.isin(route,('L168','L24','UNAVAILABLE')).all() and np.all(~common|(route=='L168')),
            'Event route/common population differs')
    coverage=next(x for x in summary['coverage_by_gauge'] if x['gauge_id']==gauge)
    require(int(common.sum())==coverage['origin_cohorts']['ORIGINAL_COMMON']
            and {r:int((route==r).sum()) for r in ('L168','L24','UNAVAILABLE')}==coverage['route_counts'],
            'Plot routing differs from the verified coverage report')
    require(_array_digest({'cohort':common})==coverage['original_common_mask_sha256']
            and _array_digest({'route':route,'eligible168':route=='L168','eligible24':route!='UNAVAILABLE'})==coverage['computed_router_choices_sha256'],
            'Plot route/common masks differ from their scored identities')
    predictions=np.asarray(value['routed_predictions']);available=route!='UNAVAILABLE'
    require(predictions.dtype==np.dtype('<f8') and predictions.shape==(4,3,ORIGINS,18)
            and np.isfinite(predictions[:,:,available]).all() and np.isnan(predictions[:,:,~available]).all(),
            'All four models and three seeds must retain raw finite requested forecasts')
    require(all(_array_digest({'predictions':predictions[mi,si]})==summary['routed_prediction_array_sha256'][gauge][architecture][str(seed)]
                for mi,architecture in enumerate(ARCHITECTURES) for si,seed in enumerate(SEEDS)),
            'Plotted forecasts differ from the authenticated scored predictions')
    times=np.asarray(value['reference_time']);q=np.asarray(value['reference_value']);mask=np.asarray(value['reference_mask'])
    require(times.dtype==np.dtype('<i8') and np.array_equal(times,np.arange(START,ASSESSMENT_START,3600,dtype=np.int64))
            and q.shape==mask.shape==times.shape and q.dtype==np.dtype('<f8') and mask.dtype==np.dtype(bool)
            and np.isfinite(q[mask]).all() and np.all(q[mask]>=0) and np.isnan(q[~mask]).all(),
            'Original complete reference calendar with explicit quality gaps required')
    y=np.asarray(value['verification_target_values']);m=np.asarray(value['verification_target_mask'])
    require(y.dtype==np.dtype('<f8') and m.dtype==np.dtype(bool) and y.shape==m.shape==future.shape
            and _array_digest({'labels':y})==summary['target_arrays_sha256_by_gauge'][gauge]['labels_sha256']
            and _array_digest({'label_mask':m})==summary['target_arrays_sha256_by_gauge'][gauge]['label_mask_sha256'],
            'Plot verification labels differ from the independently scored targets')
    positions=((future-START)//3600).astype(np.int64)
    require(np.array_equal(mask[positions],m) and np.array_equal(q[positions][m],y[m]),
            'Plotted observations differ from original approved verification values')
    return value


def render_event_hydrographs(summary,data_by_gauge,directory,*,png_previews=False):
    """Fixed top-three H1/H18 panels; no model-based event or route selection.

    Parent authenticates raw descriptor bytes and passes only post-barrier data.
    routed_predictions axes are ARCHITECTURES × SEEDS × origins × H18. Reference
    values are the original approved calendar, with NaN at rejected/missing hours.
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
    require(set(data_by_gauge)==set(GAUGE_IDS),'All three event plotting gauges required')
    for gauge,value in data_by_gauge.items():validate_event_plot_data(summary,gauge,value)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);colors=('#bf5700','#0d6074','#815ca4','#58752d');output=[]
    for lead in (1,18):
        fig,axes=plt.subplots(3,3,figsize=(16,11))
        for gi,gauge in enumerate(GAUGE_IDS):
            coverage=next(x for x in summary['coverage_by_gauge'] if x['gauge_id']==gauge)
            events=[x for x in coverage['products']['ROUTED_CLEAN_168_24_V1']['fixed_event_coverage']['events'] if x['representative_top3']]
            require(len(events)==min(3,coverage['products']['ROUTED_CLEAN_168_24_V1']['fixed_event_coverage']['catalog_event_count']),
                    'Keep every fixed representative event')
            data=data_by_gauge[gauge];valid_time=data['forecast_time'][:,lead-1]
            for ei,ax in enumerate(axes[gi]):
                if ei>=len(events):ax.text(.5,.5,'No further catalogued event',ha='center',transform=ax.transAxes);continue
                event=events[ei];lo=int(datetime.fromisoformat(event['start_utc'].replace('Z','+00:00')).timestamp());hi=int(datetime.fromisoformat(event['end_exclusive_utc'].replace('Z','+00:00')).timestamp())
                observed=(data['reference_time']>=lo)&(data['reference_time']<hi);requested=(valid_time>=lo)&(valid_time<hi)
                clock=valid_time[requested];route=data['route'][requested];common=data['original_common'][requested]
                for selected,color in (((route=='L168')&~common,'#94c8d8'),(route=='L24','#f0c48a')):
                    transitions=np.diff(np.r_[False,selected,False].astype(np.int8))
                    for start,end in zip(np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1)):
                        ax.axvspan(datetime.fromtimestamp(int(clock[start]),timezone.utc),datetime.fromtimestamp(int(clock[end-1])+3600,timezone.utc),color=color,alpha=.20,zorder=0)
                x=[datetime.fromtimestamp(int(t),timezone.utc) for t in clock]
                reference_clock=[datetime.fromtimestamp(int(t),timezone.utc) for t in data['reference_time'][observed]]
                ax.plot(reference_clock,data['reference_value'][observed],color='#111111',lw=1.7,label='Approved observed flow',zorder=5)
                for mi,architecture in enumerate(ARCHITECTURES):
                    p=data['routed_predictions'][mi][:,requested,lead-1];valid=np.isfinite(p).all(axis=0)
                    center=np.full(p.shape[1],np.nan);low=center.copy();high=center.copy()
                    center[valid]=p[:,valid].mean(axis=0);low[valid]=p[:,valid].min(axis=0);high[valid]=p[:,valid].max(axis=0)
                    ax.plot(x,center,color=colors[mi],lw=1.3,label=MODEL_NAMES[architecture]);ax.fill_between(x,low,high,color=colors[mi],alpha=.10)
                peak_epoch=int(datetime.fromisoformat(event['event_id'].replace('Z','+00:00')).timestamp())
                at=valid_time==peak_epoch;ref_index=(peak_epoch-START)//3600
                actual_peak=bool(np.any(at&(data['route']!='UNAVAILABLE')) and data['reference_mask'][ref_index])
                require(actual_peak==(lead in event['observed_peak_supported_leads']),'Event plotted peak support differs from verified counts')
                peak='covered' if actual_peak else 'UNAVAILABLE'
                ax.set_title(f"{NAMES[gauge]} · fixed event {event['rank']}\n{event['event_id'][:10]} · peak forecast {peak}",fontsize=10)
                ax.text(.02,.96,f"{int((route!='UNAVAILABLE').sum())}/{len(route)} candidate hours forecast; {int((route=='L24').sum())} use L24",transform=ax.transAxes,va='top',fontsize=8)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d',tz=timezone.utc));ax.grid(alpha=.15)
                if ei==0:ax.set_ylabel('Discharge (m³/s)')
                if gi==2:ax.set_xlabel('Valid time (UTC)')
        prefix='SYNTHETIC QUALIFICATION · ' if summary.get('synthetic_qualification') else 'Phase A development · '
        fig.suptitle(prefix+f'fixed top-three events · routed H{lead} forecasts',fontsize=15,y=1.01)
        handles,labels=axes[0,0].get_legend_handles_labels();handles += [Patch(color='#94c8d8',alpha=.3),Patch(color='#f0c48a',alpha=.3)]
        labels += ['Additional L168 origin','Clean L24 fallback'];fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,-.05),fontsize=9)
        fig.text(.5,-.075,'Curves = seed means; bands = seed min–max, not uncertainty intervals. Metrics remain means of individual-seed scores.',ha='center',fontsize=9)
        fig.text(.5,-.095,'Background marks newly issued populations. Fixed events are not replaced; missing forecasts and rejected references remain gaps.',ha='center',fontsize=9)
        fig.tight_layout();path=directory/f'phase-a-fixed-events-H{lead:02d}.svg';require(not path.exists(),'Preserve existing event figure')
        fig.savefig(path,bbox_inches='tight',metadata={'Date':None,'Creator':'HYDRA DirectFlow Phase A'})
        if png_previews:
            preview=directory/'private-png-previews';preview.mkdir(exist_ok=True);fig.savefig(preview/(path.stem+'.png'),dpi=160,bbox_inches='tight')
        plt.close(fig);raw=path.read_bytes();output.append(dict(path=path.name,bytes=len(raw),sha256=_sha(raw)))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase-root',type=Path,required=True);parser.add_argument('--evaluation-index',type=Path,required=True)
    parser.add_argument('--expected-index-sha256',required=True);parser.add_argument('--accounting',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--figures',action='store_true');parser.add_argument('--png-previews',action='store_true');args=parser.parse_args()
    root=args.phase_root.absolute();require(root==root.resolve(),'Stable Phase A root required')
    def resident(path,maximum):
        path=Path(path).absolute();require(path==path.resolve() and path.is_relative_to(root),'Report input escaped Phase A')
        before=path.lstat();require(stat.S_ISREG(before.st_mode) and 0<before.st_size<=maximum
                and not getattr(before,'st_flags',0)&0x40000000,'Resident bounded report input required')
        body=path.read_bytes();after=path.lstat()
        key=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
        require(key(before)==key(after) and len(body)==before.st_size,'Report input changed during read');return body
    raw=resident(args.evaluation_index,16*1024**2);require(_sha(raw)==args.expected_index_sha256,'Exact completed evaluation index required')
    def reader(desc):
        require(set(desc)=={'path','bytes','sha256'} and not Path(desc['path']).is_absolute(),'Exact private report descriptor required')
        path=root/desc['path'];require(path==path.resolve() and path.is_relative_to(root),'Report escaped phase directory')
        body=resident(path,16*1024**2);require(len(body)==desc['bytes'] and _sha(body)==desc['sha256'],'Report artifact changed')
        return json.loads(body)
    publication=build_publication(json.loads(raw),reader,json.loads(resident(args.accounting,16*1024**2)))
    require(args.output.absolute()==args.output.resolve() and args.output.absolute().is_relative_to(root),'Private phase publication output required')
    require(not args.output.exists(),'Fresh publication directory required');args.output.mkdir(parents=True)
    files={}
    for name,body in publication['files'].items():
        p=args.output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body);files[name]=dict(bytes=len(body),sha256=_sha(body))
    figures=render_aggregate_figures(publication['summary'],args.output/'figures',png_previews=args.png_previews) if args.figures else []
    (args.output/'MANIFEST.json').write_bytes(_json(dict(schema='hydra-phase-a-publication/v1',files=files,figures=figures,
        phase_plan_sha256=publication['summary']['phase_plan_sha256'],prediction_barrier_sha256=publication['summary']['prediction_barrier_sha256'],
        source_index_sha256=args.expected_index_sha256,script_sha256=_sha(Path(__file__).read_bytes()),protected_assessment_access=False)))
    print(json.dumps(dict(status='PUBLICATION_RENDERED_REQUIRES_REVIEW',files=len(files),figures=len(figures))))


if __name__=='__main__':main()
