#!/usr/bin/env python3
"""Describe development coverage by season/reference regime; never fit or score a model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np

from hydra_directflow import corpus_dataset as dataset
from hydra_directflow.cache import DerivedCache, unpack_month
from hydra_directflow.coverage_analysis import (HOUR, SEASONS, reference_arrays, gradient_thresholds,
    unusable_intervals, select_observed_events, case_masks, coverage_by_regime, event_coverage, season, iso)
from hydra_directflow.local_data import GAUGES, USGS_MANIFEST_SHA256
from hydra_directflow.quality import PRIMARY_APPROVED
from hydra_directflow.reference import ASSESSMENT_START, digest, require
from audit_corpus_admission import Guard, write_report
from plan_scientific_workload import fold_windows, epoch

START = epoch('2018-07-12T00:00:00Z')
CODE = ('scripts/audit_coverage_regimes.py','src/hydra_directflow/coverage_analysis.py',
        'src/hydra_directflow/corpus_dataset.py','src/hydra_directflow/cache.py',
        'src/hydra_directflow/quality.py','scripts/plan_scientific_workload.py')


def reference_month(cache, month, key):
    require(month <= '2024-07' and dataset._hex(key), 'Protected/invalid monthly reference rejected before value access')
    path=cache.path/'entries'/key;dataset._directory(path)
    raw=dataset._read(path/'COMMIT.json',128*1024);commit=json.loads(raw);identity=commit['identity']
    require(commit['kind']=='usgs_month' and identity['month']==month
            and identity['source_manifest_sha256']==USGS_MANIFEST_SHA256
            and commit['key']==key and DerivedCache.key('usgs_month',identity)==key,
            'Corpus-pinned monthly cache identity/key differs')
    if month=='2024-07':
        require(identity.get('development_export',{}).get('cutoff_exclusive_epoch')==ASSESSMENT_START,
                'July requires the verified development-only export')
    arrays,metadata=cache._load(path,'usgs_month',identity,key)
    require(dataset._read(path/'COMMIT.json',128*1024)==raw,'Reference cache commitment changed')
    return unpack_month(arrays,metadata),dict(month=month,cache_key=key,commit_sha256=hashlib.sha256(raw).hexdigest())


def load_references(root, cache_directory, snapshot, guard):
    month_keys = {}
    for shard in snapshot.shards:
        directory = root/'chunks'/shard.identifier
        raw = dataset._read(directory/'COMMIT.json', 65536)
        require(hashlib.sha256(raw).hexdigest() == shard.commit_sha256, 'Snapshot commitment changed')
        commit = json.loads(raw); body = dataset._read(directory/'case-index.json',256*1024)
        require(hashlib.sha256(body).hexdigest() == commit['payloads']['case-index.json']['sha256'], 'Case index changed')
        for month,key in json.loads(body)['month_keys'].items():
            require(month not in month_keys or month_keys[month] == key, 'Competing monthly cache identities')
            month_keys[month] = key
    cache = DerivedCache(cache_directory)
    dataset._directory(cache.path);dataset._directory(cache.path/'entries')
    observations = {g:[] for g in GAUGES}; identities = []
    for month,key in sorted(month_keys.items()):
        rows,identity=reference_month(cache,month,key)
        for observation in rows:
            observations[observation.site].append((observation.valid_at,observation.value,observation.quality_flags))
        identities.append(identity);guard.check()
    return {g:reference_arrays(rows,start=START,end=ASSESSMENT_START) for g,rows in observations.items()},identities


def run(corpus,cache,expected_plan_sha256,admission_report):
    guard=Guard(900,4*1024**3)
    raw=dataset._read(admission_report,16*1024**2); prior=json.loads(raw)
    require(prior['status']=='COMPLETE_SNAPSHOT_CANDIDATE_AUDIT' and not prior['physical_domain_review_needed'],
            'Complete reviewed candidate audit required')
    require(prior['audit_content_sha256']==digest({k:v for k,v in prior.items() if k not in ('audit_content_sha256','runtime')}),
            'Admission content digest differs')
    snapshot=dataset.inspect_corpus(corpus,expected_plan_sha256=expected_plan_sha256)
    require(snapshot.complete and snapshot.report()==prior['snapshot'], 'Exact complete admission snapshot required')
    root=Path(corpus); refs,month_ids=load_references(root,cache,snapshot,guard)
    parts={g:[] for g in GAUGES}; seasonal={}; case_total=0
    for shard in snapshot.shards:
        arrays,_,_=dataset._load_shard(root,shard.unit,shard.chunk,snapshot.plan_sha256,expected_commit_sha=shard.commit_sha256)
        masks=case_masks(arrays)
        for gi,gauge in enumerate(GAUGES):
            rows=np.flatnonzero(arrays['site_id']==gi); origin=arrays['origin_time'][rows]
            ref=refs[gauge]; positions=(arrays['target_time'][rows]-START)//HOUR
            require(((positions>=0)&(positions<len(ref['times']))).all(), 'Reference join outside development')
            require(np.array_equal(ref['admitted'][positions],arrays['target_mask'][rows]), 'Reference/case target quality mismatch')
            good=arrays['target_mask'][rows]
            require(np.array_equal(ref['values'][positions][good],arrays['target'][rows][good]), 'Reference/case values disagree')
            part={'origin':origin.copy(),'target_mask':arrays['target_mask'][rows].copy()}
            part.update({k:v[rows].copy() for k,v in masks.items()}); parts[gauge].append(part)
            for j,index in enumerate(rows):
                key=(gauge,shard.unit['split'],season(origin[j])); counts=seasonal.setdefault(key,Counter())
                counts['candidate_origins']+=1
                for name,value in masks.items(): counts[name]+=int(value[index])
            case_total+=len(rows)
        guard.check()
    require(case_total==prior['all_candidate_cases']==159084,'Coverage denominator differs')
    series={g:{key:np.concatenate([p[key] for p in values]) for key in values[0]} for g,values in parts.items()}
    season_rows=[]
    for (gauge,split,s),values in sorted(seasonal.items()):
        values=dict(values);n=values['candidate_origins']
        values['excluded_from_common_L168']=n-values['common_L168']
        values['natural_L24_lost_to_common_L168']=values['natural_L24']-values['common_L168']
        season_rows.append(dict(gauge_id=gauge,split=split,season_utc=s,counts=values,
            common_L168_fraction=values['common_L168']/n,natural_L24_fraction=values['natural_L24']/n,
            exclusion_dimensions_overlap=True,grain='GAUGE_ORIGIN_CASE'))
    reference_rows=[]; folds=[]
    for gauge in GAUGES:
        reference=refs[gauge];seasons=np.array([season(t) for t in reference['times']])
        intervals=unusable_intervals(reference)
        reference_rows.append(dict(gauge_id=gauge,reference_calendar_hours=len(reference['times']),
            statuses=dict(Counter(reference['statuses'].tolist())),
            by_season={s:dict(Counter(reference['statuses'][seasons==s].tolist())) for s in SEASONS},
            unusable_reference_intervals=intervals,
            longest_unusable_interval_hours=max((i['hours'] for i in intervals),default=0),
            station_disable_or_damage_not_inferred=True,unobserved_high_flow_peaks_not_inferred=True))
        for fold in fold_windows(simulated_label_delay_hours=2):
            gradient,outer=fold['gradient_fit'],fold['outer_scoring']
            thresholds=gradient_thresholds(reference,start=epoch(gradient['start_utc']),end=epoch(gradient['end_exclusive_utc']),
                refit_at=epoch(gradient['refit_at_utc']),delay_hours=2)
            data=series[gauge];start,end=epoch(outer['start_utc']),epoch(outer['end_exclusive_utc'])
            regimes=coverage_by_regime(data['origin'],data['input_L168'],data['common_L168'],data['target_mask'],reference,thresholds,start=start,end=end,natural_l24=data['natural_L24'])
            events=select_observed_events(reference,thresholds,start=start,end=end)
            for event in events:
                support=event_coverage(event,data['origin'],data['input_L168'],data['common_L168'],data['target_mask'],outer_start=start,outer_end=end)
                event['all_lead_requested_cells']=sum(r['requested_cells'] for r in support)
                event['all_lead_common_comparison_cells']=sum(r['common_comparison_cells'] for r in support)
                event['observed_peak_supported_leads']=[r['lead_hours'] for r in support if r['observed_peak_has_common_prediction_case']]
                if event['representative_top3']: event['forecast_case_coverage_by_lead']=support
            folds.append(dict(gauge_id=gauge,fold=fold['name'],thresholds=thresholds,outer_window=outer,
                target_cell_regime_coverage=regimes,observed_high_flow_events=events,
                observed_high_flow_event_count=len(events),representative_event_rule='TOP3_OBSERVED_Q95_EXCEEDING_PEAKS_SEPARATED_168H',
                event_windows_hours=[-48,72],event_windows_clipped_to_outer=True,event_selection_uses_model_errors=False))
            guard.check()
    result=dict(schema='hydra-development-coverage-regimes/v1',status='DEVELOPMENT_DIAGNOSTICS_NOT_PROTOCOL_FREEZE',
        snapshot_sha256=snapshot.snapshot_sha256,source_plan_sha256=snapshot.plan_sha256,
        admission_report_sha256=hashlib.sha256(raw).hexdigest(),monthly_reference_identities=month_ids,
        quality_policy=PRIMARY_APPROVED,season_definition='UTC_METEOROLOGICAL_DJF_MAM_JJA_SON',
        all_candidate_cases=case_total,complete_common_L168=sum(r['counts']['common_L168'] for r in season_rows),
        natural_L24=sum(r['counts']['natural_L24'] for r in season_rows),seasonal_coverage=season_rows,
        reference_coverage=reference_rows,fold_regime_and_event_diagnostics=folds,
        diagnostic_regime_rule='q<=q10 LOW; q10<q<=q90 ORDINARY; q90<q<=q95 ELEVATED; q>q95 HIGH; unusable reference UNKNOWN',
        scientific_holds=['Event/uncertainty specifications require review before protocol freeze',
            'Reference outages include missing/rejected records and do not establish damaged/disabled instruments',
            'Flow regimes cannot be assigned to missing or rejected target references',
            'Observed high-flow selection cannot recover unobserved peaks; no official flood-stage claim',
            'Complete-case results condition on availability; exclusion rates do not measure skill during outages'],
        implementation_sha256={name:hashlib.sha256(dataset._read(ROOT/name,512*1024)).hexdigest() for name in CODE},
        execution=dict(network_requests=0,raw_source_reads=0,model_scaler_fits=0,optimizer_updates=0,
            diagnostic_reference_quantiles_computed=True,model_predictions_or_errors_used=False,
            source_masks_or_values_changed=False,protected_assessment_access=False))
    result['coverage_content_sha256']=digest(result);result['runtime']=guard.report()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','cache','admission-report','report'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-plan-sha256',required=True);a=p.parse_args()
    require(not a.report.exists() and not a.report.is_symlink(),'Fresh report required')
    def deadline(signum,frame):raise TimeoutError('Coverage audit wall deadline reached')
    previous=signal.signal(signal.SIGALRM,deadline);signal.alarm(900)
    try:
        report=run(a.corpus,a.cache,a.expected_plan_sha256,a.admission_report)
        write_report(report,a.report,source_root=a.corpus)
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)
    print(json.dumps({k:report[k] for k in ('status','all_candidate_cases','complete_common_L168','natural_L24','coverage_content_sha256')}))


if __name__=='__main__':main()
