#!/usr/bin/env python3
"""Authenticate committed forecasts, then prepare verification and point metrics.

No inference, training, model selection, downloads or uncertainty fitting here.
This implementation is separately frozen before the first verification-role read.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from collections import Counter
import csv
import gzip
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import run_assessment_first_pass as first
from hydra_directflow.assessment_runtime import AssessmentRuntime
from hydra_directflow.assessment_contract import origins,NWM_REACHES,SEEDS,validate_primary_checkpoint_roster
from hydra_directflow.contract import ARCHITECTURES
from hydra_directflow.assessment_inference import validate_selected_job
from hydra_directflow.assessment_nwm import NwmSourceQualification,normalize_nwm_row,_utc_hour,DECODED_REPRESENTATION,UNITS
from hydra_directflow.assessment_data import load_assessment_usgs,assemble_assessment_targets
from hydra_directflow.local_data import GAUGES,iso
from hydra_directflow.reference import ASSESSMENT_START,ASSESSMENT_END,ASSESSMENT_LAST_TARGET,HOUR,require,digest
from hydra_directflow.training import array_identity,FULL_FEATURE_SCHEMA

def normalize_owned_rows(rows,qualification):
    """Stream fixed snapshot metadata first, retain exact source row references."""
    calendar=np.array(origins(),dtype=np.int64);n=len(calendar)
    result={g:dict(prediction_q=np.full((n,18),np.nan),source_valid=np.zeros((n,18),bool),
        source_row=np.full((n,18),-1,dtype=np.int64),numeric_status=np.full((n,18),'ISSUE_OR_LEAD_ABSENT',dtype='U64'),
        raw_streamflow=np.full((n,18),'',dtype='U64'),source_forecast_offset=np.full((n,18),'',dtype='U32')) for g in GAUGES}
    for a in result.values():a['numeric_status'][:,16:]='NATIVE_HORIZON_UNAVAILABLE'
    gauges={v:k for k,v in NWM_REACHES.items()};counts=Counter();total=0
    for number,row in enumerate(rows,1):
        total=number
        feature=int(row['feature_id']);reference=_utc_hour(row['reference_time']);lead=int(row['lead_hours'])
        origin=reference+2*HOUR
        if feature not in gauges or not ASSESSMENT_START<=origin<ASSESSMENT_END or not 3<=lead<=18:
            counts['OUTSIDE_MATCHED_ORIGIN_NATIVE_LEAD_SCOPE']+=1;continue
        g=gauges[feature];h=lead-2;value=normalize_nwm_row(row,gauge_id=g,origin=origin,origin_horizon=h,source=qualification,row_id=number)
        i=(origin-ASSESSMENT_START)//HOUR;j=h-1;a=result[g]
        require(a['source_row'][i,j]==-1,'Duplicate owned NWM key; unique snapshot must not be silently re-deduplicated')
        raw=value.raw_streamflow or '';offset='' if value.raw_source_forecast_offset is None else str(value.raw_source_forecast_offset)
        require(len(raw)<=64 and len(offset)<=32,'Original NWM strings exceed bounded preservation columns')
        a['source_row'][i,j]=number;a['numeric_status'][i,j]=value.numeric_status
        a['raw_streamflow'][i,j]=raw;a['source_forecast_offset'][i,j]=offset
        if value.numeric_valid:a['prediction_q'][i,j]=value.streamflow_m3_s;a['source_valid'][i,j]=True
        counts[value.numeric_status]+=1
    return result,dict(source_rows=total,dispositions=dict(counts),source_qualification_identity=qualification.identity,
        reference_rule='EXACT_ORIGIN_MINUS_2H',native_leads=[3,18],origin_leads=[1,16],unsupported_origin_leads=[17,18],
        units=UNITS,representation=DECODED_REPRESENTATION,raw_rows_retained_in_original_snapshot=True,
        source_row_number_convention='ONE_BASED_DATA_ROW_EXCLUDING_HEADER',no_rescaling=True)

def get_nwm(runtime,p):
    cached=runtime.reuse('NWM_MATCHED_INPUT')
    if cached:return cached
    d=p['nwm_snapshot'];raw=first.resident(Path(d['path']),16*1024**2,d,lambda n:runtime.reserve_read(n,purpose='OWNED_NWM_SNAPSHOT'))
    parity=p['nwm_parity'];first.resident(Path(parity['path']),1024**2,parity,lambda n:runtime.reserve_read(n,numeric=False))
    source=NwmSourceQualification(source_id='owned-unique-short-range-bigquery',snapshot_sha256=d['sha256'],
        parity_receipt_sha256=parity['sha256'],units=UNITS,representation=DECODED_REPRESENTATION)
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as binary:
        with io.TextIOWrapper(binary,encoding='utf-8',newline='') as text:
            arrays,audit=normalize_owned_rows(csv.DictReader(text),source)
    artifacts={g:runtime.write('nwm/'+g+'.npz',first.npz(a),kind='MATCHED_NWM_SOURCE') for g,a in arrays.items()}
    audit['arrays_sha256_by_gauge']={g:array_identity(a) for g,a in arrays.items()};audit['snapshot']=d
    artifacts['audit']=runtime.write('nwm/PROVENANCE.json',first.encoded(audit),kind='NWM_PROVENANCE')
    return runtime.commit('NWM_MATCHED_INPUT',artifacts,metadata=dict(snapshot_sha256=d['sha256']))

def get_targets(runtime,p,authority,barrier_sha):
    cached=runtime.reuse('VERIFICATION_TARGETS',expected_metadata=dict(prediction_barrier_sha256=barrier_sha))
    if cached:return cached
    a=replace(authority,verification_enabled=True,prediction_barrier_sha256=barrier_sha)
    folder=first.RUN/'source-staging/usgs';paths={x.stem:str(x) for x in folder.glob('*.parquet')}
    source=load_assessment_usgs(paths,str(folder/'source_manifest.json'),[(ASSESSMENT_START+HOUR,ASSESSMENT_LAST_TARGET)],a,
        mode='verification',read_budget=lambda n:runtime.reserve_read(n,purpose='VERIFICATION_USGS'))
    calendar=origins();parts={g:[] for g in GAUGES}
    for start in range(0,len(calendar),120):
        chunk=assemble_assessment_targets(source['observations'],tuple(calendar[start:start+120]),a)['arrays']
        for j,g in enumerate(GAUGES):parts[g].append({k:v[j::3] for k,v in chunk.items()})
        runtime.check()
    artifacts={}
    for g in GAUGES:
        arrays={k:np.concatenate([x[k] for x in parts[g]],axis=0) for k in parts[g][0]}
        artifacts[g]=runtime.write('verification/'+g+'.npz',first.npz(arrays),kind='VERIFICATION_TARGETS')
    audit={k:v for k,v in source.items() if k!='observations'}
    audit['predictions_committed_before_role_read']=True
    artifacts['audit']=runtime.write('verification/PROVENANCE.json.gz',gzip.compress(first.encoded(audit),mtime=0),kind='TARGET_PROVENANCE')
    return runtime.commit('VERIFICATION_TARGETS',artifacts,metadata=dict(prediction_barrier_sha256=barrier_sha))

def semantic_prediction_barrier(runtime,p,sources,authority,structural):
    """Authenticate all36 complete numerical products before any outcome read.

    A structural commit count is insufficient: every saved selection, input
    cohort, exact coordinate and complete prediction must agree. This function
    reads only committed inputs/predictions/receipts and writes a durable barrier
    whose identity, not the structural count alone, releases the target role.
    """
    records=sources['selection']['records']
    validate_primary_checkpoint_roster(j['job_id'] for j in records)
    for j in records:validate_selected_job(j,authority)
    structural_sha=structural['artifacts']['barrier']['sha256']
    ic=runtime.reuse('INPUTS',expected_metadata=dict(source_identity=p['input_source_identity']))
    require(ic is not None,'Committed input inventory required before semantic verification')
    inventory=json.loads(runtime.read(ic['artifacts']['inventory'],numeric=False))
    require(inventory['input_source_identity']==p['input_source_identity']
        and set(inventory['gauge_bundles'])==set(GAUGES),'Exact original gauge input inventory required')
    calendar=np.asarray(origins(),dtype=np.int64);future=calendar[:,None]+np.arange(1,19)*HOUR
    products={};gauges={}
    for si,g in enumerate(GAUGES):
        bundle=inventory['gauge_bundles'][g]
        a=first.arrays_from_bytes(runtime.read(bundle['arrays']))
        manifest=json.loads(runtime.read(bundle['manifest'],numeric=False))
        typed=array_identity(a)
        require(manifest['arrays_sha256']==typed and manifest['gauge_id']==g
            and manifest['source_identity']==p['input_source_identity']
            and manifest['binding']['execution_plan_sha256']==p['plan_sha256']
            and a['inference_source_identity'].item()==p['input_source_identity']
            and a['cohort_admission_digest'].item()==manifest['cohort_admission_digest']
            and a['feature_schema'].item()==manifest['feature_schema']==FULL_FEATURE_SCHEMA,
            'Gauge input typed/source/cohort/release identity differs')
        require(np.array_equal(a['origin_time'],calendar) and np.array_equal(a['forecast_time'],future)
            and np.array_equal(a['history_time'],calendar[:,None]-np.arange(169,1,-1)*HOUR)
            and a['site_id'].shape==(8760,) and np.all(a['site_id']==si)
            and a['case_id'].tolist()==[g+':'+iso(int(c)) for c in calendar],
            'Gauge input exact calendar/gauge/case coordinates differ')
        admitted=a['input_admitted']
        require(admitted.dtype==np.dtype(bool) and admitted.shape==(8760,),
                'Explicit original input admission required')
        complete=np.ones(8760,dtype=bool)
        for name,shape in (('history',(8760,168,2)),('weather',(8760,18,4)),('state',(8760,2))):
            values,mask=a[name],a[name+'_mask']
            require(values.shape==mask.shape==shape and values.dtype.kind=='f'
                and mask.dtype==np.dtype(bool) and np.isfinite(values[mask]).all(),
                'Committed input value/mask geometry differs')
            complete &= mask.reshape(8760,-1).all(axis=1)
        require(np.array_equal(admitted,complete) and manifest['requested_origins']==8760
            and manifest['admitted_origins']==int(admitted.sum()),'Original complete-input admission differs')
        jobs=[j for j in records if j['gauge_id']==g]
        require(len(jobs)==12 and {(j['architecture'],j['seed']) for j in jobs}==
                {(arch,seed) for arch in ARCHITECTURES for seed in SEEDS},
                'All12 family/seed forecasts required for each gauge')
        positions=np.flatnonzero(admitted)
        for j in jobs:
            c=runtime.reuse(j['job_id'])
            require(c is not None and c['job_id']==j['job_id'] and c['kind']=='model'
                and digest(c)==structural['metadata']['model_commit_sha256_by_job'][j['job_id']],
                'Selected model commit differs from structural barrier')
            z=first.arrays_from_bytes(runtime.read(c['artifacts']['predictions']))
            r=json.loads(runtime.read(c['artifacts']['receipt'],numeric=False))
            require(set(z)=={'origin_time','case_id','forecast_time','site_id','input_admitted','prediction_q'}
                and all(np.array_equal(a[k],z[k]) for k in ('origin_time','case_id','forecast_time','site_id','input_admitted')),
                'Prediction coordinates/admission changed before verification')
            y=z['prediction_q']
            require(y.dtype==np.dtype('float64') and y.shape==(8760,18)
                and np.isfinite(y[admitted]).all() and np.isnan(y[~admitted]).all(),
                'All admitted predictions must be finite and all excluded predictions NaN before verification')
            require(array_identity(z)==r['predictions_array_sha256'] and r['job']==j
                and r['checkpoint_sha256']==j['checkpoint']['sha256']==p['checkpoint_sha256_by_job'][j['job_id']]
                and r['input_arrays_sha256']==typed and r['input_artifact']==bundle['arrays']
                and r['forecast_origins']==len(positions) and r['candidate_origins']==8760
                and r['target_values_assembled'] is False and r['optimizer_updates']==r['scaler_fits']==0,
                'Prediction typed identity, selected checkpoint/job or input source differs')
            batches=r['batches']
            require(len(batches)==(len(positions)+63)//64,'Complete chronological batch receipts required')
            for offset,b in zip(range(0,len(positions),64),batches):
                take=positions[offset:offset+64]
                require(b['job_id']==j['job_id'] and b['checkpoint_sha256']==j['checkpoint']['sha256']
                    and b['gauge_id']==g and b['architecture']==j['architecture'] and b['seed']==j['seed']
                    and b['selected_state']=='best_model' and b['selected_best_epoch']==j['selected_best_epoch']
                    and b['execution_plan_sha256']==p['plan_sha256']
                    and b['selection_sha256']==authority.selection_sha256
                    and b['inference_source_identity']==p['input_source_identity']
                    and b['inference_parent_cohort_admission_digest']==manifest['cohort_admission_digest']
                    and b['input_case_ids_sha256']==digest(a['case_id'][take].tolist())
                    and b['case_count']==len(take) and b['target_values_read'] is False
                    and b['target_masks_read'] is False and b['optimizer_updates']==b['scaler_fits']==0,
                    'Saved forward receipt differs from selected chronological input batch')
            products[j['job_id']]=dict(model_commit_sha256=digest(c),artifacts=c['artifacts'],
                predictions_array_sha256=r['predictions_array_sha256'],checkpoint_sha256=j['checkpoint']['sha256'],
                input_arrays_sha256=typed,forecast_origins=len(positions))
        gauges[g]=dict(input_bundle=bundle,input_arrays_sha256=typed,admitted_origins=len(positions))
        runtime.check()
    require(len(products)==36,'All36 semantically authenticated forecast products required')
    body=dict(schema='hydra-assessment-semantic-prediction-barrier/v1',
        status='ALL36_PREDICTION_SEMANTICS_VERIFIED_BEFORE_VERIFICATION_ROLE_READ',
        plan_sha256=p['plan_sha256'],selection_sha256=authority.selection_sha256,
        structural_prediction_barrier_sha256=structural_sha,input_inventory=ic['artifacts']['inventory'],
        gauges=gauges,models=products,verification_values_read=False,nwm_values_read=False)
    previous=runtime.reuse('SEMANTIC_PREDICTION_BARRIER',expected_metadata=body)
    if previous:return previous
    d=runtime.write('SEMANTIC_PREDICTION_BARRIER.json',first.encoded(body),numeric=False,kind='SEMANTIC_PREDICTION_BARRIER')
    return runtime.commit('SEMANTIC_PREDICTION_BARRIER',{'barrier':d},metadata=body)

def release(runtime,p):
    code={str(path.relative_to(ROOT)):first.sha(first.resident(path,2*1024**2)) for path in
        (Path(__file__).resolve(),ROOT/'src/hydra_directflow/assessment_scoring.py')}
    body=dict(schema='hydra-assessment-point-scoring-release/v1',plan_sha256=p['plan_sha256'],
        protocol_sha256=p['json_sources']['protocol']['sha256'],selection_sha256=p['json_sources']['selection']['sha256'],
        implementation=code,no_inference=True,no_training=True,no_post_outcome_method_selection=True,
        formal_confidence_intervals=False)
    previous=runtime.reuse('SCORING_RELEASE',expected_metadata=body)
    if previous:return previous
    d=runtime.write('SCORING_RELEASE.json',first.encoded(body),numeric=False,kind='SCORING_RELEASE')
    return runtime.commit('SCORING_RELEASE',{'release':d},metadata=body)

def run(path,expected):
    # Only the minimal frozen envelope is read before the runtime owns and
    # precharges the bounded authentication phase.
    p=json.loads(first.resident(Path(path),16*1024**2))
    require(p['plan_sha256']==expected==digest({k:v for k,v in p.items() if k!='plan_sha256'})
        and p['run_directory']==str(first.RUN),'Frozen local scoring execution envelope required')
    from hydra_directflow.assessment_scoring import score_gauge
    with AssessmentRuntime(p).writer() as runtime:
        runtime.reserve_read(16*1024**2,numeric=False,purpose='BOUNDED_SCORING_AUTHENTICATION')
        p,sources,authority=first.authenticate(path,expected)
        release(runtime,p)
        structural=runtime.prediction_barrier()
        barrier=semantic_prediction_barrier(runtime,p,sources,authority,structural)
        barrier_sha=barrier['artifacts']['barrier']['sha256']
        nwm=get_nwm(runtime,p);targets=get_targets(runtime,p,authority,barrier_sha)
        input_commit=runtime.reuse('INPUTS');inventory=json.loads(runtime.read(input_commit['artifacts']['inventory']))
        calendar=np.array(origins(),dtype=np.int64);scores={}
        for g in GAUGES:
            key='SCORE_'+g;existing=runtime.reuse(key)
            if existing:scores[g]=existing['artifacts']['scores'];continue
            a=first.arrays_from_bytes(runtime.read(inventory['gauge_bundles'][g]['arrays']))
            require(np.array_equal(a['origin_time'],calendar),'Gauge input calendar differs')
            truth=first.arrays_from_bytes(runtime.read(targets['artifacts'][g]))
            require(np.array_equal(truth['target_time'],calendar[:,None]+np.arange(1,19)*HOUR),'Target coordinates differ')
            predictions={}
            for j in sources['selection']['records']:
                if j['gauge_id']!=g:continue
                c=runtime.reuse(j['job_id']);z=first.arrays_from_bytes(runtime.read(c['artifacts']['predictions']))
                for name in ('origin_time','case_id','forecast_time','site_id','input_admitted'):
                    require(np.array_equal(a[name],z[name]),'Prediction coordinates/admission changed')
                r=json.loads(runtime.read(c['artifacts']['receipt']))
                require(array_identity(z)==r['predictions_array_sha256'] and r['checkpoint_sha256']==j['checkpoint']['sha256'],'Prediction typed identity/checkpoint differs')
                predictions[(j['architecture'],j['seed'])]=z['prediction_q']
            nw=first.arrays_from_bytes(runtime.read(nwm['artifacts'][g]))
            result=score_gauge(gauge_id=g,origins=calendar,target=truth['target'],target_mask=truth['target_mask'],
                input_admitted=a['input_admitted'],predictions=predictions,nwm=nw['prediction_q'],nwm_mask=nw['source_valid'])
            result['provenance']=dict(plan_sha256=p['plan_sha256'],prediction_barrier_sha256=barrier_sha,
                structural_prediction_barrier_sha256=structural['artifacts']['barrier']['sha256'],
                target_artifact=targets['artifacts'][g],nwm_artifact=nwm['artifacts'][g],input_artifact=inventory['gauge_bundles'][g]['arrays'])
            d=runtime.write('scores/'+g+'.json',first.encoded(result),kind='ASSESSMENT_POINT_SCORES')
            runtime.commit(key,{'scores':d});scores[g]=d
            first.progress(runtime,'ASSESSMENT_POINT_SCORING',gauge_complete=g)
        summary=dict(schema='hydra-assessment-firstpass-completion/v1',status='POINT_METRICS_COMPLETE_PENDING_INDEPENDENT_REVIEW',
            scores=scores,models=36,origin_period=['2024-07-31','2025-07-31'],target_tail_utc='2025-07-31T17:00:00Z',
            plan_sha256=p['plan_sha256'],new_optimizer_updates=0,new_scaler_fits=0,nwm_supported_origin_leads=[1,16],
            formal_confidence_intervals=False,finished_at_utc=first.utc())
        d=runtime.write('RESULTS.json',first.encoded(summary),kind='RESULTS')
        runtime.commit('POINT_RESULTS',{'results':d});first.progress(runtime,summary['status'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',default=str(first.RUN/'EXECUTION_PLAN.json'));p.add_argument('--plan-sha256',required=True)
    a=p.parse_args();run(a.plan,a.plan_sha256)
