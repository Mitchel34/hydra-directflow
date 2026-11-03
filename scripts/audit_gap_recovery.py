#!/usr/bin/env python3
"""Development-only mask recoverability census, without filling values or fitting.

Forecast availability, target-evaluable support and the unchanged complete-H18
primary cohort have separate denominators. Missing reference peaks stay unknown.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from hydra_directflow import corpus_dataset as dataset
from hydra_directflow.coverage_analysis import HOUR, season, gradient_thresholds, select_observed_events
from hydra_directflow.local_data import GAUGES
from hydra_directflow.reference import ASSESSMENT_START, digest, require
from hydra_directflow.quality import PRIMARY_APPROVED
from audit_corpus_admission import Guard, write_report
from audit_coverage_regimes import load_references, START
from plan_scientific_workload import fold_windows, epoch

ALIASES = ('strict_l168', 'natural_l24', 'perfect_q', 'interior_1', 'interior_2', 'trailing_1', 'trailing_2', 'combined_1', 'combined_2')
OPTIMISTIC = ('optimistic_interior_run2_total_uncapped', 'optimistic_combined_run2_total_uncapped')
WEATHER = ('precipitation', 'temperature', 'humidity', 'snow_water_equivalent')
CODE = ('scripts/audit_gap_recovery.py', 'src/hydra_directflow/gap_policies.py',
        'scripts/audit_coverage_regimes.py', 'scripts/audit_corpus_admission.py',
        'scripts/plan_scientific_workload.py', 'src/hydra_directflow/coverage_analysis.py',
        'src/hydra_directflow/corpus_dataset.py', 'src/hydra_directflow/cache.py',
        'src/hydra_directflow/local_data.py', 'src/hydra_directflow/quality.py', 'src/hydra_directflow/reference.py')


def sha(raw): return hashlib.sha256(raw).hexdigest()


def pinned_report(path, expected_sha, content_key):
    raw = dataset._read(path, 1024**2)
    require(sha(raw) == expected_sha, 'Pinned parent report SHA differs')
    value = json.loads(raw)
    require(value[content_key] == digest({k:v for k,v in value.items() if k not in (content_key, 'runtime')}), 'Parent content identity differs')
    return value


def fractions(numerators, denominators):
    return [None if int(d) == 0 else int(n) / int(d) for n,d in zip(numerators, denominators)]


def unique_hours(times, mask):
    return int(np.unique(times[np.broadcast_to(mask, times.shape)]).size)


def summarize_support(origins, target_mask, primary, policies):
    """All origins remain the headline denominator; valid-target fraction is extra."""
    origins = np.asarray(origins); n = len(origins)
    require(n > 0 and target_mask.shape == (n,18) and target_mask.dtype == np.dtype(bool)
            and primary.shape == (n,) and primary.dtype == np.dtype(bool), 'Aligned support census arrays required')
    require(np.all(~primary | target_mask.all(axis=1)), 'Primary cases must retain complete H18 labels')
    times = origins[:,None] + np.arange(1,19) * HOUR
    valid = target_mask.sum(axis=0)
    result = dict(requested_origins=n, valid_target_origins_by_lead=valid.tolist(),
        unknown_target_origins_by_lead=(n-valid).tolist(), unchanged_primary_complete_h18_cases=int(primary.sum()),
        unique_requested_target_hours=int(np.unique(times).size), unique_valid_target_hours=unique_hours(times,target_mask),
        unique_unknown_target_hours=unique_hours(times,~target_mask), policies={})
    for alias, available in policies.items():
        require(available.shape == (n,) and available.dtype == np.dtype(bool), 'Policy availability must be one Boolean per origin')
        evaluable = available[:,None] & target_mask
        count = evaluable.sum(axis=0); forecasts = np.full(18,available.sum(),dtype=np.int64)
        all18 = available & target_mask.all(axis=1)
        require(np.all(~primary | available), 'A sensitivity policy removed an unchanged primary case')
        result['policies'][alias] = dict(forecast_available_by_lead=forecasts.tolist(), target_evaluable_by_lead=count.tolist(),
            forecast_availability_fraction_by_lead=fractions(forecasts,np.full(18,n)),
            evaluable_fraction_all_origins_by_lead=fractions(count,np.full(18,n)),
            evaluable_fraction_valid_targets_by_lead=fractions(count,valid), all18_evaluable_cases=int(all18.sum()),
            recovered_all18_cases_vs_primary=int((all18 & ~primary).sum()),
            unique_forecast_target_hours=unique_hours(times,available[:,None]), unique_evaluable_target_hours=unique_hours(times,evaluable),
            all_lead_evaluable_fraction_all_origins=float(evaluable.sum()/(n*18)),
            all_lead_evaluable_fraction_valid_targets=None if not valid.sum() else float(evaluable.sum()/valid.sum()))
    return result


def gap_runs(mask):
    """Describe observed mask runs; policy admission is delegated separately."""
    missing = ~np.asarray(mask)
    require(missing.shape == (168,) and missing.dtype == np.dtype(bool), 'Actual L168 Q mask required')
    edges = np.diff(np.r_[0, missing.astype(np.int8), 0])
    result = []
    for left,right in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        position = 'ALL_MISSING' if left==0 and right==168 else 'LEADING' if left==0 else 'TRAILING' if right==168 else 'INTERIOR'
        result.append(dict(start_index=int(left),end_index_exclusive=int(right),length_hours=int(right-left),position=position,
                           nearest_missing_age_hours=int(170-right),farthest_missing_age_hours=int(169-left)))
    return result


def failure_combination(q_complete, weather_columns, soil, calendar):
    labels = []
    if not q_complete: labels.append('Q_HISTORY')
    labels.extend('WEATHER_'+name.upper() for name,good in zip(WEATHER,weather_columns) if not good)
    if not soil: labels.append('SOIL')
    if not calendar: labels.append('CALENDAR')
    return '|'.join(labels) or 'NONE'


def history_options(classification, ids):
    result = {'natural_l24':bool(classification['natural_l24']['history_admissible']), 'perfect_q':True}
    for alias in ('strict_l168', *ALIASES[3:]):
        key='strict' if alias=='strict_l168' else alias
        result[alias]=bool(classification['candidates'][ids[key]]['history_admissible'])
    for mode in ('interior','combined'):
        decision=classification['candidates'][ids[mode+'_2']]
        result['optimistic_'+mode+'_run2_total_uncapped'] = bool(decision['history_admissible'] or decision['reasons']==['TOTAL_MISSING_EXCEEDS_CAP'])
    return result


def event_support(event, origins, targets, primary, policies):
    """Retain every observed event, including those with zero usable inputs."""
    result = dict(event_id=event['event_id'], observed_peak_rank=event['observed_peak_rank'],
        representative_top3=event['representative_top3'], start_utc=event['start_utc'], end_exclusive_utc=event['end_exclusive_utc'],
        reference_requested_hours=event['reference_requested_hours'], reference_approved_hours=event['reference_approved_hours'],
        reference_unusable_hours=event['reference_unusable_hours'], left_truncated=event['left_truncated'], right_truncated=event['right_truncated'],
        selection_uses_model_errors=False, hidden_peak_during_unknown_reference_known=False, policies={})
    valid = origins[:,None] + np.arange(1,19)*HOUR
    inside = (valid >= event['start_epoch']) & (valid < event['end_epoch'])
    at_peak = valid == event['peak_epoch']
    result['requested_cells_by_lead'] = inside.sum(axis=0).tolist()
    result['valid_target_cells_by_lead'] = (inside & targets).sum(axis=0).tolist()
    result['observed_peak_requested_by_lead'] = at_peak.any(axis=0).tolist()
    result['strict_primary_peak_supported_leads'] = (at_peak & primary[:,None]).any(axis=0).tolist()
    for alias,available in policies.items():
        evaluable = targets & available[:,None]
        peak_leads = (at_peak & evaluable).any(axis=0)
        result['policies'][alias] = dict(forecast_available_cells_by_lead=(inside & available[:,None]).sum(axis=0).tolist(),
            target_evaluable_cells_by_lead=(inside & evaluable).sum(axis=0).tolist(),
            unique_target_evaluable_hours=unique_hours(valid,inside & evaluable), observed_peak_supported_by_lead=peak_leads.tolist(),
            observed_peak_any_lead=bool(peak_leads.any()), observed_peak_all18_leads=bool(peak_leads.all()))
    return result


def high_flow_support(origins, target_mask, primary, policies, reference, thresholds):
    times = origins[:,None] + np.arange(1,19)*HOUR
    positions = (times-reference['times'][0]) // HOUR
    require(((positions>=0)&(positions<len(reference['times']))).all(), 'High-flow target join outside reference inventory')
    observed = reference['admitted'][positions]
    require(np.array_equal(observed,target_mask), 'High-flow target mask changed')
    high = observed & (reference['values'][positions] > thresholds['q95_cms'])
    denominator = high.sum(axis=0)
    report = dict(known_high_target_cells_by_lead=denominator.tolist(), unique_known_high_target_hours=unique_hours(times,high),
        unknown_target_cells_by_lead=(~observed).sum(axis=0).tolist(), high_flow_definition='APPROVED_TARGET_Q_GT_GRADIENT_ONLY_Q95',
        missing_or_rejected_targets_classified_high=False, policies={})
    for alias,available in policies.items():
        supported = high & available[:,None]; counts = supported.sum(axis=0)
        report['policies'][alias] = dict(target_evaluable_known_high_cells_by_lead=counts.tolist(),
            known_high_coverage_fraction_by_lead=fractions(counts,denominator),
            unique_evaluable_known_high_target_hours=unique_hours(times,supported),
            unique_known_high_coverage_fraction=None if not report['unique_known_high_target_hours'] else unique_hours(times,supported)/report['unique_known_high_target_hours'])
    report['unchanged_primary_known_high_cells_by_lead'] = (high & primary[:,None]).sum(axis=0).tolist()
    return report


def build_census(*, corpus, cache, expected_plan_sha256, admission_report, expected_admission_sha256,
                 coverage_report, expected_coverage_sha256):
    from hydra_directflow.gap_policies import POLICY_IDS, policy_catalog, classify_history_window
    guard = Guard(900,4*1024**3); started = time.monotonic()
    code_before = {name:sha(dataset._read(ROOT/name,512*1024)) for name in CODE}
    admission = pinned_report(admission_report,expected_admission_sha256,'audit_content_sha256')
    coverage = pinned_report(coverage_report,expected_coverage_sha256,'coverage_content_sha256')
    snapshot = dataset.inspect_corpus(corpus,expected_plan_sha256=expected_plan_sha256)
    require(snapshot.complete and snapshot.report()==admission['snapshot'] and snapshot.snapshot_sha256==coverage['snapshot_sha256'],
            'Exact complete audited development snapshot required')
    require(admission['all_candidate_cases']==coverage['all_candidate_cases']==159084, 'Fixed development case inventory differs')
    refs, monthly = load_references(Path(corpus),cache,snapshot,guard)
    require(monthly==coverage['monthly_reference_identities'], 'Reference cache inventory changed')
    policy_mapping = {'strict_l168':POLICY_IDS['strict'], **{alias:POLICY_IDS[alias] for alias in ALIASES[3:]},
                      'natural_l24':'DIAGNOSTIC_STRICT_NATURAL_L24_V1','perfect_q':'DIAGNOSTIC_PERFECT_Q_NON_Q_CEILING_V1'}
    accum = defaultdict(lambda:dict(case_count=0,q_incomplete_cases=0,missing_history_cells=0,runs=Counter(),
        nearest_missing_age_hours=Counter(),latest_approved_age_hours=Counter(),failure_combinations=Counter(),
        history_recoverable_cases=Counter(),missing_reference_reason_case_cells=Counter(),unique_missing_history_times=set()))
    parts = {g:[] for g in GAUGES}; details = []; case_total = 0
    for shard in snapshot.shards:
        arrays,_,_ = dataset._load_shard(Path(corpus),shard.unit,shard.chunk,snapshot.plan_sha256,expected_commit_sha=shard.commit_sha256)
        n = len(arrays['origin_time']); policy_masks = {alias:np.zeros(n,bool) for alias in (*ALIASES,*OPTIMISTIC)}
        weather = arrays['weather_mask'].all(axis=1); soil = arrays['state_mask'].all(axis=1)
        calendar = arrays['source_calendar_status']=='RESOLVED_WITHIN_ACQUIRED_CALENDAR'
        nonq = weather.all(axis=1) & soil & calendar
        for index in range(n):
            gauge = GAUGES[int(arrays['site_id'][index])]; origin = int(arrays['origin_time'][index]); qmask = arrays['history_mask'][index,:,0]
            times = arrays['history_time'][index].tolist()
            classification = classify_history_window(history_times=times,approved_mask=qmask.tolist(),
                source_observation_times=[int(t) if ok else None for t,ok in zip(times,qmask)],cutoff_epoch=origin,simulated_lag_hours=2)
            history=history_options(classification,POLICY_IDS)
            for alias,history_ok in history.items():
                policy_masks[alias][index] = history_ok and bool(nonq[index])
            bucket = accum[gauge,shard.unit['split']]; bucket['case_count']+=1
            runs = gap_runs(qmask); missing = np.flatnonzero(~qmask)
            bucket['q_incomplete_cases']+=bool(len(missing)); bucket['missing_history_cells']+=len(missing)
            bucket['failure_combinations'][failure_combination(bool(qmask.all()),weather[index],bool(soil[index]),bool(calendar[index]))]+=1
            approved = np.flatnonzero(qmask)
            bucket['latest_approved_age_hours']['NONE' if not len(approved) else str((origin-times[int(approved[-1])])//HOUR)]+=1
            for alias,history_ok in history.items():
                bucket['history_recoverable_cases'][alias]+=int(history_ok)
            for run in runs:
                bucket['runs'][run['position']+':'+str(run['length_hours'])]+=1
                bucket['nearest_missing_age_hours'][str(run['nearest_missing_age_hours'])]+=1
            for slot in missing:
                stamp=times[int(slot)];bucket['unique_missing_history_times'].add(stamp)
                position=(stamp-START)//HOUR
                reason='PRE_REFERENCE_START_REASON_UNKNOWN' if position<0 else str(refs[gauge]['statuses'][position])
                bucket['missing_reference_reason_case_cells'][reason]+=1
            if runs:
                details.append([gauge,origin,shard.unit['split'],[[r['start_index'],r['end_index_exclusive'],r['position']] for r in runs],
                    [int(policy_masks[alias][index]) for alias in ALIASES]])
            if case_total % 512 == 0: guard.check()
            case_total+=1
        strict = policy_masks['strict_l168'] & arrays['target_mask'].all(axis=1)
        require(np.array_equal(strict,arrays['admission_mask']), 'Unchanged primary admission differs')
        for gi,gauge in enumerate(GAUGES):
            indices=np.flatnonzero(arrays['site_id']==gi); ref=refs[gauge]
            positions=(arrays['target_time'][indices]-START)//HOUR
            require(((positions>=0)&(positions<len(ref['times']))).all(), 'Target outside development reference')
            target_mask=arrays['target_mask'][indices]
            require(np.array_equal(ref['admitted'][positions],target_mask), 'Reference/case target masks changed')
            require(np.array_equal(ref['values'][positions][target_mask],arrays['target'][indices][target_mask]), 'Reference/case approved target values differ')
            history_positions=(arrays['history_time'][indices]-START)//HOUR
            inside=history_positions>=0
            require((history_positions[inside]<len(ref['times'])).all(),'History reference join outside development')
            observed_history=arrays['history_mask'][indices,:,0]
            require(np.array_equal(ref['admitted'][history_positions[inside]],observed_history[inside]),'Reference/corpus history masks differ')
            known=inside & observed_history
            require(np.array_equal(ref['values'][history_positions[known]],arrays['history'][indices,:,0][known]),'Reference/corpus approved history values differ')
            parts[gauge].append(dict(origin=arrays['origin_time'][indices].copy(),target_mask=target_mask.copy(),primary=strict[indices].copy(),
                split=np.array([shard.unit['split']]*len(indices)),**{alias:value[indices].copy() for alias,value in policy_masks.items()}))
        guard.check()
    require(case_total==159084,'Candidate origin census incomplete')
    series={g:{key:np.concatenate([part[key] for part in entries]) for key in entries[0]} for g,entries in parts.items()}
    split_rows=[]; seasonal_rows=[]; gap_rows=[]; optimistic_rows=[]
    for gauge,data in series.items():
        require(len(data['origin'])==53028 and np.all(np.diff(data['origin'])>0),'Chronological gauge census differs')
        for split in ('candidate_training','candidate_validation'):
            chosen=data['split']==split
            split_rows.append(dict(gauge_id=gauge,split=split,**summarize_support(data['origin'][chosen],data['target_mask'][chosen],data['primary'][chosen],{a:data[a][chosen] for a in ALIASES})))
            optimistic_rows.append(dict(gauge_id=gauge,split=split,status='NONADOPTED_OPTIMISTIC_ELIGIBILITY_ONLY',
                additional_history_cap_only_cases={mode:int(accum[gauge,split]['history_recoverable_cases']['optimistic_'+mode+'_run2_total_uncapped']-accum[gauge,split]['history_recoverable_cases'][mode+'_2']) for mode in ('interior','combined')},
                **summarize_support(data['origin'][chosen],data['target_mask'][chosen],data['primary'][chosen],{a:data[a][chosen] for a in OPTIMISTIC})))
            bucket=accum[gauge,split];unique=sorted(bucket.pop('unique_missing_history_times'))
            gap_rows.append(dict(gauge_id=gauge,split=split,**{k:dict(v) if isinstance(v,Counter) else int(v) for k,v in bucket.items()},
                unique_missing_history_hours=len(unique),unique_missing_history_times_sha256=digest(unique),
                gap_run_grain='OVERLAPPING_GAUGE_ORIGIN_L168_HISTORY_WINDOWS',missing_reference_reason_grain='HISTORY_CASE_CELLS_NOT_UNIQUE_HOURS'))
            seasons=np.array([season(t) for t in data['origin']])
            for label in ('DJF','MAM','JJA','SON'):
                selected=chosen & (seasons==label)
                if not selected.any():continue
                n=int(selected.sum()); valid=data['target_mask'][selected]
                seasonal_rows.append(dict(gauge_id=gauge,split=split,season_utc=label,requested_origins=n,valid_target_cells=int(valid.sum()),
                    unchanged_primary_complete_h18_cases=int(data['primary'][selected].sum()),
                    policies={a:dict(forecast_available_origins=int(data[a][selected].sum()),target_evaluable_cells=int((data[a][selected,None]&valid).sum()),
                        all18_evaluable_cases=int((data[a][selected]&valid.all(axis=1)).sum())) for a in ALIASES}))
    prior_folds={(row['gauge_id'],row['fold']):row for row in coverage['fold_regime_and_event_diagnostics']}
    high_rows=[]; all_events=[]
    for gauge,data in series.items():
        reference=refs[gauge]
        for fold in fold_windows(simulated_label_delay_hours=2):
            gradient,outer=fold['gradient_fit'],fold['outer_scoring']; start,end=epoch(outer['start_utc']),epoch(outer['end_exclusive_utc'])
            thresholds=gradient_thresholds(reference,start=epoch(gradient['start_utc']),end=epoch(gradient['end_exclusive_utc']),refit_at=epoch(gradient['refit_at_utc']),delay_hours=2)
            require(thresholds==prior_folds[gauge,fold['name']]['thresholds'],'Gradient-only diagnostic thresholds changed')
            selected=(data['origin']>=start)&(data['origin']+18*HOUR<end)
            origin=data['origin'][selected]; targets=data['target_mask'][selected]; primary=data['primary'][selected]
            policies={a:data[a][selected] for a in ALIASES}
            high=high_flow_support(origin,targets,primary,policies,reference,thresholds)
            events=select_observed_events(reference,thresholds,start=start,end=end)
            require([e['event_id'] for e in events]==[e['event_id'] for e in prior_folds[gauge,fold['name']]['observed_high_flow_events']], 'Observed-event census changed')
            event_rows=[event_support(event,origin,targets,primary,policies) for event in events]
            all_events.append(dict(gauge_id=gauge,fold=fold['name'],events=event_rows))
            event_totals={a:dict(observed_peak_any_lead_events=sum(e['policies'][a]['observed_peak_any_lead'] for e in event_rows),
                observed_peak_all18_lead_events=sum(e['policies'][a]['observed_peak_all18_leads'] for e in event_rows),
                observed_peak_supported_event_counts_by_lead=np.sum([e['policies'][a]['observed_peak_supported_by_lead'] for e in event_rows],axis=0,dtype=np.int64).tolist() if event_rows else [0]*18,
                all_event_window_target_evaluable_cells_by_lead=np.sum([e['policies'][a]['target_evaluable_cells_by_lead'] for e in event_rows],axis=0,dtype=np.int64).tolist() if event_rows else [0]*18) for a in ALIASES}
            high_rows.append(dict(gauge_id=gauge,fold=fold['name'],outer_window=outer,thresholds=thresholds,requested_origins=len(origin),
                **high,observed_event_count=len(event_rows),event_policy_totals=event_totals,representative_top3=[e for e in event_rows if e['representative_top3']],
                observed_event_peak_requested_counts_by_lead=np.sum([e['observed_peak_requested_by_lead'] for e in event_rows],axis=0,dtype=np.int64).tolist() if event_rows else [0]*18,
                event_definition='OBSERVED_Q95_EXCEEDING_GREEDY_MAXIMA_SEPARATED_AT_LEAST168H',event_window_hours=[-48,72],event_windows_clipped_to_outer=True))
            guard.check()
    require(code_before=={name:sha(dataset._read(ROOT/name,512*1024)) for name in CODE},'Census implementation changed')
    require(sum(r['unchanged_primary_complete_h18_cases'] for r in split_rows)==admission['snapshot']['source_admitted_cases'],'Primary total changed')
    private=dict(schema='hydra-development-gap-recovery-private/v1',case_columns=['gauge_id','origin_epoch','split','missing_runs_start_end_exclusive_position','policy_forecast_availability_bits'],
                 policy_bit_order=list(ALIASES),only_q_incomplete_cases=True,case_gap_details=details,fold_event_details=all_events)
    result=dict(schema='hydra-development-gap-recovery/v1',status='DEVELOPMENT_MASK_RECOVERABILITY_CENSUS_NOT_PROTOCOL_ADOPTION',
        created_at_utc=datetime.now(timezone.utc).isoformat(),source_plan_sha256=snapshot.plan_sha256,snapshot_sha256=snapshot.snapshot_sha256,
        admission_report_sha256=expected_admission_sha256,coverage_report_sha256=expected_coverage_sha256,monthly_reference_identities=monthly,
        policy_ids=policy_mapping,policy_catalog=policy_catalog(),quality_policy=PRIMARY_APPROVED,
        all_candidate_cases=case_total,leads_hours=list(range(1,19)),split_coverage=split_rows,seasonal_coverage=seasonal_rows,
        history_gap_census=gap_rows,fold_high_flow=high_rows,
        nonadopted_optimistic_eligibility=optimistic_rows,
        boundaries=dict(requested_calendar_origins=53064,retained_candidate_origins_per_gauge=53028,strict_h18_boundary_purged_origins_per_gauge=36,
                        original_boundary_purges_preserved=True,pre_acquisition_calendar_status_unchanged=True),
        source_timestamp_provenance='APPROVED_HISTORY_TIMES_ARE_EXACT_SOURCE_OBSERVATION_TIMES_ESTABLISHED_BY_PINNED_CORPUS_BUILDER_AND_VALIDATED_CLOCKS;FALSE_MASK_TIMESTAMPS_ARE_NOT_INVENTED',
        limitations=['Weather selection remains a single complete H18 run: all four variable masks fail jointly; raw variable-specific failure causes and alternative per-lead weather fallback are not inferred',
            'Recoverable history masks are counterfactual availability only: no numerical imputation or model skill is measured',
            'Perfect-Q is a non-Q-input ceiling, not an implemented fill or guaranteed90percent result',
            'Target masks/values and strict matched complete-H18 primary cohort remain unchanged; expanded evaluable cells are diagnostic sensitivity support only',
            'Unknown targets remain in the all-origin denominator and are never assigned high flow',
            'Known high-flow and event denominators condition on approved observed references; unobserved peaks and instrument damage cannot be inferred',
            'Early history slots preceding the shared reference calendar retain corpus availability; their missing-reference reason is explicitly unknown',
            'Unique target hours are deduplicated across overlapping origins/leads; counts are not independent statistical replicates'],
        implementation_sha256=code_before,execution=dict(network_requests=0,raw_source_reads=0,numerical_fill_operations=0,source_values_or_masks_changed=False,
            target_values_or_masks_changed=False,model_fits=0,optimizer_updates=0,protected_assessment_access=False,scientific_policy_adopted=False),
        runtime=guard.report())
    return result,private


def report_bytes(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def package_report(result, stem):
    """Keep the headline frontier readable; bind explicit per-gauge event files."""
    summary={**result};summary['fold_high_flow']=[];details={g:[] for g in GAUGES}
    for original in result['fold_high_flow']:
        row=dict(original);events=row.pop('representative_top3')
        row['representative_top3_event_ids']=[e['event_id'] for e in events]
        summary['fold_high_flow'].append(row)
        details[row['gauge_id']].append(dict(fold=row['fold'],thresholds=row['thresholds'],outer_window=row['outer_window'],
                                              representative_top3=events))
    files=[]
    for gauge,folds in details.items():
        value=dict(schema='hydra-development-gap-recovery-event-details/v1',gauge_id=gauge,source_plan_sha256=result['source_plan_sha256'],
            snapshot_sha256=result['snapshot_sha256'],coverage_report_sha256=result['coverage_report_sha256'],policy_ids=result['policy_ids'],
            folds=folds,numerical_fill_operations=0,scientific_policy_adopted=False)
        raw=report_bytes(value);require(len(raw)<=1024**2,'Per-gauge event details exceed1MiB')
        files.append((stem+'-events-'+gauge+'.json',value,dict(gauge_id=gauge,sha256=sha(raw),bytes=len(raw))))
    summary['event_detail_files']=[dict(file=name,**identity) for name,_,identity in files]
    return summary,files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','cache','admission-report','coverage-report','report','private-details'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('expected-plan-sha256','expected-admission-sha256','expected-coverage-sha256'):parser.add_argument('--'+name,required=True)
    args=vars(parser.parse_args()); public=Path(args.pop('report')); private=Path(args.pop('private_details'))
    require(private.resolve().is_relative_to((ROOT/'.local').resolve()),'Private census details must remain in ignored .local')
    require(not public.exists() and not private.exists() and not public.is_symlink() and not private.is_symlink(),'Fresh census outputs required')
    def timeout(*unused):raise TimeoutError('Gap census exceeded900seconds')
    previous=signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    try:
        result,details=build_census(**args)
        result,event_files=package_report(result,public.stem)
        require(all(not public.with_name(name).exists() and not public.with_name(name).is_symlink() for name,_,_ in event_files),
                'Event detail outputs already exist')
        # Preserve detailed observed events and sparse missing-case descriptors privately.
        write_report(details,private,source_root=args['corpus'])
        raw=dataset._read(private,16*1024**2);result['private_details']=dict(sha256=sha(raw),bytes=len(raw))
        result['census_content_sha256']=digest({k:v for k,v in result.items() if k!='runtime'})
        require(len(report_bytes(result))<=1024**2,'Public census exceeds1MiB; split explicitly before publication')
        for name,value,_ in event_files:write_report(value,public.with_name(name),source_root=args['corpus'])
        write_report(result,public,source_root=args['corpus'])
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)
    print(json.dumps(dict(status=result['status'],all_candidate_cases=result['all_candidate_cases'],census_content_sha256=result['census_content_sha256'])))


if __name__=='__main__':main()
