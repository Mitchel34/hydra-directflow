#!/usr/bin/env python3
"""Separate structural short-Q-or-L24 routing census; never create forecasts."""
import argparse
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
from hydra_directflow.coverage_analysis import select_observed_events
from hydra_directflow.local_data import GAUGES
from hydra_directflow.reference import digest, require
from audit_corpus_admission import Guard, write_report
from audit_coverage_regimes import load_references
from audit_gap_recovery import summarize_support, high_flow_support, event_support
from plan_scientific_workload import epoch

CODE = ('scripts/audit_gap_route.py', 'scripts/audit_gap_recovery.py',
        'scripts/audit_coverage_regimes.py', 'src/hydra_directflow/corpus_dataset.py',
        'src/hydra_directflow/coverage_analysis.py', 'src/hydra_directflow/gap_policies.py',
        'src/hydra_directflow/cache.py')


def route_masks(qmask, nonq):
    require(qmask.ndim == 2 and qmask.shape[1] == 168 and qmask.dtype == np.dtype(bool)
            and nonq.shape == (len(qmask),) and nonq.dtype == np.dtype(bool),
            'Exact Boolean L168 history and non-Q input masks required')
    strict = nonq & qmask.all(1)
    # With a total cap of two, every run is necessarily <=2. A real first
    # slot excludes leading gaps; interior/trailing endpoints then exist.
    short = nonq & qmask[:, 0] & ((~qmask).sum(1) <= 2)
    l24 = nonq & qmask[:, -24:].all(1)
    return {'strict_l168': strict, 'combined_2': short, 'natural_l24': l24,
            'short2_or_l24': short | l24}


def run(corpus, cache, census, expected_census_sha256, expected_plan_sha256):
    guard = Guard(900, 4*1024**3)
    code_before = {p: hashlib.sha256(dataset._read(ROOT/p, 512*1024)).hexdigest() for p in CODE}
    raw = dataset._read(census, 1_000_000)
    require(hashlib.sha256(raw).hexdigest() == expected_census_sha256, 'Pinned census SHA differs')
    parent = json.loads(raw)
    require(parent['schema'] == 'hydra-development-gap-recovery/v1'
            and parent['status'] == 'DEVELOPMENT_MASK_RECOVERABILITY_CENSUS_NOT_PROTOCOL_ADOPTION',
            'Completed development-only structural census required')
    require(parent['census_content_sha256'] == digest({k:v for k,v in parent.items()
            if k not in ('census_content_sha256', 'runtime')}), 'Parent census identity differs')
    snapshot = dataset.inspect_corpus(corpus, expected_plan_sha256=expected_plan_sha256)
    require(snapshot.complete and snapshot.snapshot_sha256 == parent['snapshot_sha256'],
            'Complete parent census snapshot required')
    references, monthly = load_references(Path(corpus), cache, snapshot, guard)
    require(monthly == parent['monthly_reference_identities'], 'Monthly source identities differ')
    parts = {g: [] for g in GAUGES}
    for shard in snapshot.shards:
        if shard.unit['split'] != 'candidate_validation': continue
        arrays, _, _ = dataset._load_shard(Path(corpus), shard.unit, shard.chunk,
            snapshot.plan_sha256, expected_commit_sha=shard.commit_sha256)
        nonq = arrays['weather_mask'].all((1,2)) & arrays['state_mask'].all(1)
        nonq &= arrays['source_calendar_status'] == 'RESOLVED_WITHIN_ACQUIRED_CALENDAR'
        masks = route_masks(arrays['history_mask'][...,0], nonq)
        require(np.array_equal(masks['strict_l168'] & arrays['target_mask'].all(1),
                               arrays['admission_mask']), 'Strict primary changed')
        for gi, gauge in enumerate(GAUGES):
            rows = arrays['site_id'] == gi
            parts[gauge].append({'origin': arrays['origin_time'][rows].copy(),
                'target_mask': arrays['target_mask'][rows].copy(),
                'primary': arrays['admission_mask'][rows].copy(),
                **{k:v[rows].copy() for k,v in masks.items()}})
        guard.check()
    split_rows, high_rows = [], []
    for gauge, chunks in parts.items():
        data = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        require(len(data['origin']) == 8766 and np.all(np.diff(data['origin']) == 3600),
                'Exact development-validation origin calendar required')
        policies = {k:data[k] for k in ('strict_l168','combined_2','natural_l24','short2_or_l24')}
        support = summarize_support(data['origin'], data['target_mask'], data['primary'], policies)
        prior = next(r for r in parent['split_coverage'] if r['gauge_id'] == gauge
                     and r['split'] == 'candidate_validation')
        for key in ('strict_l168','combined_2','natural_l24'):
            require(support['policies'][key] == prior['policies'][key],
                    'Independent closed-form policy counts differ from census')
        support['route_origin_counts'] = {
            'strict_L168': int(policies['strict_l168'].sum()),
            'short_Q_repair_only': int((policies['combined_2'] & ~policies['strict_l168']).sum()),
            'L24_after_short_repair_ineligible': int((policies['natural_l24'] & ~policies['combined_2']).sum()),
            'unavailable': int((~policies['short2_or_l24']).sum())}
        split_rows.append({'gauge_id':gauge,'split':'candidate_validation',**support})
        prior_high = next(r for r in parent['fold_high_flow'] if r['gauge_id'] == gauge
                          and r['fold'] == 'final_development')
        high = high_flow_support(data['origin'],data['target_mask'],data['primary'],policies,
                                 references[gauge],prior_high['thresholds'])
        for key in ('strict_l168','combined_2','natural_l24'):
            require(high['policies'][key] == prior_high['policies'][key], 'High-flow census differs')
        outer = prior_high['outer_window']
        events = select_observed_events(references[gauge],prior_high['thresholds'],
            start=epoch(outer['start_utc']),end=epoch(outer['end_exclusive_utc']))
        require(len(events) == prior_high['observed_event_count'], 'Event count differs')
        event_rows = [event_support(e,data['origin'],data['target_mask'],data['primary'],policies) for e in events]
        high_rows.append({'gauge_id':gauge,'fold':'final_development',**high,
            'observed_event_count':len(events), 'event_details':event_rows,
            'event_policy_totals':{k:{
                'observed_peak_any_lead_events':sum(e['policies'][k]['observed_peak_any_lead'] for e in event_rows),
                'observed_peak_all18_lead_events':sum(e['policies'][k]['observed_peak_all18_leads'] for e in event_rows),
                'observed_peak_supported_event_counts_by_lead':np.sum(
                    [e['policies'][k]['observed_peak_supported_by_lead'] for e in event_rows],axis=0,dtype=np.int64).tolist()
            } for k in policies}})
        guard.check()
    require(code_before == {p:hashlib.sha256(dataset._read(ROOT/p,512*1024)).hexdigest() for p in CODE},
            'Route census code changed during read')
    result = {'schema':'hydra-development-gap-routing/v1',
        'status':'NONADOPTED_STRUCTURAL_ROUTE_CENSUS_NOT_FORECAST_SKILL',
        'parent_census_sha256':expected_census_sha256,'snapshot_sha256':snapshot.snapshot_sha256,
        'source_plan_sha256':snapshot.plan_sha256,'implementation_sha256':code_before,
        'split_coverage':split_rows,'fold_high_flow':high_rows,
        'route_precedence':['strict_L168','qualified_short_Q_repair','strict_L24_fallback','abstain'],
        'interpretation':['No model or numerical fill has been executed; short-repair model failures can lower realized coverage.',
            'This union combines distinct input/model paths and does not extend the matched primary cohort.',
            'L24 models are planned only for final development, not the CV1 sensitivity pilot.',
            'Targets and other inputs remain unchanged; observed peaks are defined episodes, not independent floods.',
            'Structural eligibility cannot establish accuracy, false-alarm rates, operational availability or unseen peak magnitudes.'],
        'execution':{'network_requests':0,'raw_source_reads':0,'model_fits':0,'numerical_fills':0,
                     'protected_assessment_access':False,'original_arrays_changed':False}}
    result['content_sha256'] = digest(result)
    result['runtime'] = guard.report()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('corpus','cache','census','report'): parser.add_argument('--'+key,type=Path,required=True)
    for key in ('expected-census-sha256','expected-plan-sha256'): parser.add_argument('--'+key,required=True)
    args=vars(parser.parse_args()); report=args.pop('report')
    require(not report.exists() and not report.is_symlink(),'Fresh routing report required')
    def timeout(*unused): raise TimeoutError('Routing census deadline')
    previous=signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    try: result=run(**args);write_report(result,report,source_root=args['corpus'])
    finally: signal.alarm(0);signal.signal(signal.SIGALRM,previous)
    print(json.dumps({'status':result['status'],'content_sha256':result['content_sha256']}))


if __name__ == '__main__': main()
