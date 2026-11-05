#!/usr/bin/env python3
"""Phase A supervisor: input-only routes, global clean parity, then forecasts/labels.

No training initializer is exposed. New work has its own finite inference ledger
and one extra output reservation in the unchanged original campaign ledger.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import psutil
import run_primary_campaign as primary
from report_primary_seed import _ledger_records
from hydra_directflow.campaign_ledger import CampaignLedger,digest
from hydra_directflow.corpus_dataset import _read
from hydra_directflow.coverage_extension_ledger import PhaseALedger,LIMITS,ExtensionAllowanceExhausted,validate_extension_plan
from hydra_directflow.coverage_extension_runtime import PhaseStore
from hydra_directflow.heartbeat_snapshot import read_snapshot
from hydra_directflow.local_data import bounded_npy,GAUGES
from hydra_directflow.contract import ARCHITECTURES
from hydra_directflow.reference import require,ASSESSMENT_START
from hydra_directflow.training import array_identity

ADOPTED_SHA='2a68d386dabe553bdda2e77f07c99c89e4081b8dc899296162fd203d0c81fd4a'
PROTOCOL=ROOT/'configs/coverage-extension-protocol-v1.json'
MARKER=ROOT/'.cache/coverage-extension-release-v1.json'
BOOKKEEPING=64*1024**2
AUTH_READ_RESERVE=16*1024**2
START=1690761600  # 2023-07-31T00:00:00Z
END=1722319200    # 2024-07-30T06:00:00Z


def utc():return datetime.now(timezone.utc).isoformat()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def npz(arrays):
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);return stream.getvalue()


def implementation():
    names=sorted({str(p.relative_to(ROOT)) for base in (ROOT/'src/hydra_directflow',ROOT/'scripts') for p in base.rglob('*.py')})
    return {name:sha(_read(ROOT/name,2*1024**2)) for name in names}


def prepare(original_path,adopted_path):
    adopted=_read(adopted_path,128*1024);require(sha(adopted)==ADOPTED_SHA,'Adopted private plan version differs')
    raw=_read(original_path,8*1024**2);old=json.loads(raw);old=primary.authenticated_plan(original_path,old['plan_sha256'])
    ledger=CampaignLedger(old['ledger_directory']);state=ledger.snapshot()
    require(state['complete'] and state['optimizer_updates_entered']==state['optimizer_updates_confirmed']==1151690,
            'Completed original campaign and preserved training counters required')
    require(time.time()<state['campaign_deadline'],'Original campaign deadline expired')
    protocol_raw=_read(PROTOCOL,128*1024);protocol=json.loads(protocol_raw)
    require(protocol['user_authorization']['adopted_plan_sha256']==ADOPTED_SHA and protocol['limits']==LIMITS
            and protocol['zero_new_training'] and protocol['zero_optimizer_updates'] and protocol['zero_scaler_fits']
            and protocol['protected_assessment_access'] is False,'Adopted exact inference-only protocol required')
    records=[x for x in _ledger_records(old,Path(old['ledger_directory']),43)
             if x['job']['fold']=='final_development' and x['job']['input_arm']=='full']
    models={x['job']['job_id']:x['receipt']['hashes']['checkpoint'] for x in records}
    run=Path(old['run_directory'])/'coverage-extension-v1'
    # Obtain only reference-source identities. Do not open the scientific
    # postprocessing report, which also contains event/outcome summaries.
    with ledger._reader() as connection:
        rows=connection.execute("SELECT sha256 FROM outputs WHERE label='POSTPROCESSING_V1' AND status='VERIFIED'").fetchall()
    require(len(rows)==1,'One closed original postprocessing reservation required')
    inventory_raw=_read(Path(old['run_directory'])/'postprocessing-v1/ARTIFACTS.json',2*1024**2)
    require(sha(inventory_raw)==rows[0][0],'Closed postprocessing inventory changed')
    inventory=json.loads(inventory_raw)
    matches=[x for x in inventory['artifacts'] if x['path']=='EVENTS.json']
    require(len(matches)==1,'Authenticated source-reference index missing')
    event_index=matches[0]
    reference_index=dict(path='postprocessing-v1/'+event_index['path'],bytes=event_index['bytes'],sha256=event_index['sha256'])
    result=dict(schema='hydra-coverage-extension-execution-plan/v1',created_at_utc=utc(),
        adopted_plan_sha256=ADOPTED_SHA,protocol_path=str(PROTOCOL),protocol_sha256=sha(protocol_raw),
        original_plan_path=str(original_path),original_plan_sha256=old['plan_sha256'],
        original_protocol_sha256=old['protocol_sha256'],original_run_directory=old['run_directory'],
        original_ledger_directory=old['ledger_directory'],original_campaign_deadline=state['campaign_deadline'],
        original_training_updates=1151690,original_output_reserved=state['output_bytes_reserved'],
        run_directory=str(run),ledger_directory=str(run/'ledger'),corpus_root=old['corpus_root'],
        reference_cache_root=str(Path.home()/'.codex/hydra-directflow-runtime/postprocessing-usgs-reference-v1'),
        reference_index_descriptor=reference_index,original_records=records,
        checkpoint_sha256_by_job=models,limits=LIMITS,bookkeeping_reservation_bytes=BOOKKEEPING,
        authentication_read_reserve_bytes=AUTH_READ_RESERVE,initial_metadata_read_reservation=3*AUTH_READ_RESERVE,
        code_sha256=implementation(),source_read_paths=[str(PROTOCOL),str(Path(old['protocol_path'])),str(original_path),
            str(ROOT/'reports/development-coverage-regimes-v2.json'),str(ROOT/'reports/development-corpus-residency-integrity-v1.json')],
        paid_cloud_jobs=0,protected_assessment_access=False,new_training_authorized=False)
    result['plan_sha256']=digest(result);return validate_extension_plan(result)


def authenticated(path,expected):
    path=primary.private(path);plan=json.loads(_read(path,8*1024**2));validate_extension_plan(plan)
    require(plan['plan_sha256']==expected and plan['adopted_plan_sha256']==ADOPTED_SHA,'Exact adopted Phase A plan required')
    protocol_raw=_read(PROTOCOL,128*1024);protocol=json.loads(protocol_raw)
    require(sha(protocol_raw)==plan['protocol_sha256'] and plan['code_sha256']==implementation(),
            'Phase A protocol or qualified implementation changed')
    old=primary.authenticated_plan(plan['original_plan_path'],plan['original_plan_sha256'])
    require(plan['original_plan_sha256']==protocol['primary_plan_sha256']
            and plan['original_protocol_sha256']==old['protocol_sha256']==protocol['primary_protocol_sha256']
            and old['snapshot_sha256']==protocol['snapshot_sha256'] and old['source_plan_sha256']==protocol['source_plan_sha256']
            and plan['original_run_directory']==old['run_directory']
            and plan['original_ledger_directory']==old['ledger_directory'] and plan['corpus_root']==old['corpus_root']
            and plan['run_directory']==str(Path(old['run_directory'])/'coverage-extension-v1')
            and plan['ledger_directory']==str(Path(plan['run_directory'])/'ledger')
            and plan['reference_cache_root']==str(Path.home()/'.codex/hydra-directflow-runtime/postprocessing-usgs-reference-v1')
            and plan['authentication_read_reserve_bytes']==AUTH_READ_RESERVE
            and plan['initial_metadata_read_reservation']==3*AUTH_READ_RESERVE,'Phase A bindings moved')
    state=CampaignLedger(old['ledger_directory']).snapshot()
    require(state['complete'] and state['optimizer_updates_entered']==state['optimizer_updates_confirmed']==1151690
            and state['campaign_deadline']==plan['original_campaign_deadline'],'Original training evidence changed')
    records=[x for x in _ledger_records(old,Path(old['ledger_directory']),43)
             if x['job']['fold']=='final_development' and x['job']['input_arm']=='full']
    require(plan['original_records']==records and plan['checkpoint_sha256_by_job']=={
        r['job']['job_id']:r['receipt']['hashes']['checkpoint'] for r in records},'Phase A model records differ from the completed original ledger')
    return plan,old


def counted_authentication(path,expected):
    """Honor the metadata reservation made before a worker is launched.

    Keep the original resident/identity reader intact behind a counting wrapper.
    Two MiB of the reservation covers small version/fingerprint metadata reads.
    """
    from hydra_directflow import corpus_dataset
    native=_read;count=0
    def counted(p,maximum):
        nonlocal count
        count+=Path(p).lstat().st_size
        require(count<=AUTH_READ_RESERVE-2*1024**2,'Authentication metadata exceeded its precharged bound')
        return native(p,maximum)
    with patch.object(corpus_dataset,'_read',counted),patch.dict(globals(),{'_read':counted}):
        plan,old=authenticated(path,expected)
        marker=release(plan) if Path(plan['run_directory']).exists() else None
    return plan,old,marker


def initialize(plan,old):
    run=Path(plan['run_directory']);require(not run.exists() and not MARKER.exists(),'Reconcile existing Phase A; never reset or relocate')
    require(__import__('shutil').disk_usage(run.parent).free>=LIMITS['min_free_bytes'],'Free disk floor before reservation')
    original=CampaignLedger(old['ledger_directory'])
    with original.writer():
        original.initialize(old);state=original.snapshot()
        require(state['output_bytes_reserved']==plan['original_output_reserved'],'Original artifact accounting changed')
        token=original.reserve_output(LIMITS['max_output_bytes'],kind='COVERAGE_EXTENSION_PHASE_A_V1')
    run.mkdir(mode=0o700)
    primary.atomic(run/'PLAN.json',plan,new=True)
    marker=dict(phase_plan_sha256=plan['plan_sha256'],run_directory=str(run),ledger_directory=plan['ledger_directory'],
                original_reservation_token=token,created_at_utc=utc())
    primary.atomic(MARKER,marker,new=True)
    ledger=PhaseALedger(plan['ledger_directory'])
    with ledger.writer():
        ledger.initialize(plan,token,initial_read_bytes=plan['initial_metadata_read_reservation'])
        ledger.reserve_output(BOOKKEEPING,kind='BOOKKEEPING')
    primary.atomic(run/'START.json',marker,new=True)


def release(plan):
    marker=json.loads(_read(MARKER,128*1024))
    require(marker['phase_plan_sha256']==plan['plan_sha256'] and marker['run_directory']==plan['run_directory']
            and marker['ledger_directory']==plan['ledger_directory'],'Phase A release cannot move or reset')
    require(json.loads(_read(Path(plan['run_directory'])/'PLAN.json',8*1024**2))==plan,'Original Phase A plan changed')
    return marker


def descriptors(value):
    if isinstance(value,dict):
        if set(value)=={'path','bytes','sha256'}:yield value
        else:
            for item in value.values():yield from descriptors(item)
    elif isinstance(value,list):
        for item in value:yield from descriptors(item)


def completed(store,ledger,key,*,verify_all=False):
    task=ledger.task(key);require(task is not None and task['state']=='COMPLETED','Required prior task is incomplete: '+key)
    receipt=task['receipt'];require(receipt['phase_plan_sha256']==store.plan['plan_sha256'] and digest(receipt)==task['receipt_sha'],
                                  'Committed task identity differs')
    # Every consumed artifact is authenticated by its reader. Rehashing every
    # unrelated input bundle for each of72 workers would waste the fixed read
    # allowance; the task's original commit already authenticated that inventory.
    if verify_all:
        for item in descriptors(receipt['data']):store.read(item,512*1024**2)
    return receipt['data']


def stored_result(store,value,job=None):
    receipt=json.loads(store.read(value['receipt'],2*1024**2));raw=store.read(value['arrays'],64*1024**2)
    require(receipt['phase_plan_sha256']==store.plan['plan_sha256'],'Prediction phase identity differs')
    if job is not None:
        require(all(receipt[k]==job[k] for k in ('job_id','gauge_id','architecture','seed','history_hours','device')),
                'Prediction checkpoint job identity differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        count=receipt['cases'] if 'cases' in receipt else 8766
        require(type(count) is int and 0<count<=8766,'Bounded nonempty prediction population required')
        routed=receipt.get('product')=='ROUTED_CLEAN_168_24_V1'
        specs=dict(origin_time=((count,),'<i8'),forecast_time=((count,18),'<i8'),site_id=((count,),'<i8'),case_id=((count,),'<U29'),prediction_q=((count,18),'<f8'))
        if routed:specs.update(route_history_hours=((count,),'<i2'),route_checkpoint_index=((count,),'<i1'))
        require({x.filename for x in archive.infolist()}=={k+'.npy' for k in specs}
                and len(archive.infolist())==len(specs) and sum(x.file_size for x in archive.infolist())<=64*1024**2,
                'Prediction archive inventory differs')
        arrays={name:bounded_npy(archive.read(name+'.npy'),shape=specs[name][0],dtype=np.dtype(specs[name][1]))
                for name in ('origin_time','forecast_time','site_id','case_id')}
        c=arrays['origin_time'];require(np.all((c>=START)&(c<END)) and np.all(c%3600==0) and np.all(np.diff(c)>0)
            and np.array_equal(arrays['forecast_time'],c[:,None]+np.arange(1,19)*3600)
            and np.all(arrays['forecast_time']<ASSESSMENT_START), 'Prediction clock left Phase A')
        require(np.all(arrays['site_id']==GAUGES.index(receipt['gauge_id'])) and arrays['case_id'].tolist()==[
            receipt['gauge_id']+':'+datetime.fromtimestamp(int(t),timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') for t in c], 'Prediction gauge/cases differ')
        for name in specs:
            if name not in arrays:arrays[name]=bounded_npy(archive.read(name+'.npy'),shape=specs[name][0],dtype=np.dtype(specs[name][1]))
    require(array_identity(arrays)==receipt['predictions_array_sha256'],'Prediction typed identity differs')
    if not routed:require(np.isfinite(arrays['prediction_q']).all(),'Requested prediction is missing/nonfinite')
    return dict(arrays=arrays,receipt=receipt)


def save_result(store,result):
    return dict(arrays=store.write('predictions.npz',npz(result['arrays']),kind='PREDICTIONS'),
                receipt=store.write('prediction-receipt.json',encoded(result['receipt']),kind='PREDICTION_RECEIPT'))


def bounded_arrays(store,descriptor):
    raw=store.read(descriptor,256*1024**2)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        info=archive.infolist();require(len(info)<=24 and len({i.filename for i in info})==len(info)
            and sum(i.file_size for i in info)<=256*1024**2,'Bounded committed numerical archive required')
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        arrays={name:archive[name] for name in archive.files}
    require(all(v.dtype.kind in 'fbiuUS' and v.ndim<=4 for v in arrays.values()),'Non-object figure/reference arrays required')
    return arrays


def load_reconciliation(store,ledger):
    value=completed(store,ledger,'A0_RECONCILE')
    return json.loads(store.read(value['manifest'],16*1024**2))


def input_receipt(store,ledger):
    value=completed(store,ledger,'A2_INPUTS')['inputs']
    body=json.loads(store.read(value['receipt_artifact'],16*1024**2))
    require(body=={k:v for k,v in value.items() if k!='receipt_artifact'}
            and body['phase_plan_sha256']==store.plan['plan_sha256'],'Input preparation commit differs')
    return value


def handle(plan,old,ledger,store,stage,job_id=None):
    if stage=='RECONCILE':
        from hydra_directflow.coverage_extension_manifest import reconcile_models
        result=reconcile_models(old,plan['original_records'],plan['corpus_root'],plan['original_run_directory'],read_budget=store.read_budget)
        membership={}
        for key,ids in result.pop('gradient_memberships').items():
            membership[key]=store.write('gradient-membership/'+key+'.json',encoded(dict(case_ids=list(ids),case_ids_sha256=key)),kind='GRADIENT_MEMBERSHIP')
        result['gradient_membership_artifacts']=membership
        desc=plan['reference_index_descriptor'];raw=store.read_original(desc['path'],4*1024**2,expected_sha=desc['sha256']);require(len(raw)==desc['bytes'],'Reference index size differs')
        result['reference_month_identities']=json.loads(raw)['reference_months_identities']
        return dict(manifest=store.write('MODEL_MANIFEST.json',encoded(result),kind='MODEL_MANIFEST'),checkpoints=72)
    reconciliation=load_reconciliation(store,ledger)
    if stage=='INPUTS':
        from hydra_directflow.coverage_extension_io import prepare_inputs
        result=prepare_inputs(plan,reconciliation,store,check=store.check,read_budget=store.read_budget)
        return dict(inputs=result)
    inputs=input_receipt(store,ledger)
    if stage=='CLEAN_PROFILE':
        measured={};observations=[]
        for record in plan['original_records'][:8]:
            job=record['job'];done=ledger.task('CLEAN__'+job['job_id'])
            require(done is not None and done['state']=='COMPLETED','Fixed profiling prefix is incomplete')
            data=completed(store,ledger,'CLEAN__'+job['job_id']);result=stored_result(store,data,job);stats=result['receipt']['accounting']
            require(stats['case_checkpoint_evaluations_confirmed']>0,'A real clean profiling pass is required')
            measured[job['architecture'],job['history_hours']]=stats['forward_wall_seconds']/stats['case_checkpoint_evaluations_confirmed']
            observations.append(dict(job_id=job['job_id'],forward_seconds=stats['forward_wall_seconds'],
                cases=stats['case_checkpoint_evaluations_confirmed'],task_seconds=done['receipt']['elapsed_seconds']))
        require(len(measured)==8,'All four architectures and both histories required for timing')
        remaining=0.;future_cases=0
        for record in plan['original_records']:
            job=record['job'];bundles=inputs['bundles'][job['gauge_id']][str(job['history_hours'])]
            cases=bundles['expanded']['cases']-bundles['clean']['cases']
            if ledger.task('CLEAN__'+job['job_id']) is None:cases+=bundles['clean']['cases']
            future_cases+=cases;remaining+=cases*measured[job['architecture'],job['history_hours']]+20.
        # A predeclared conservative planning allowance, not a completion promise.
        conservative=2*remaining+1200.;snapshot=ledger.snapshot()
        require(snapshot['entered_cases']+future_cases<=LIMITS['max_case_checkpoint_evaluations'], 'Projected forwards exceed adopted envelope')
        require(time.time()+conservative<snapshot['deadline'],'Measured remaining work does not fit the original phase deadline')
        result=dict(status='FIXED_REAL_CLEAN_PROFILE_PASSED',phase_plan_sha256=plan['plan_sha256'],observations=observations,
                    remaining_case_evaluations_without_replay=future_cases,estimated_remaining_seconds=conservative,
                    formula='2*(sumremainingforwardcases*observedgeometryrate+20s*72jobs)+1200s_scoring_reporting',
                    timing_is_not_forecast_skill=True,target_values_read=False,profile_calls_already_in_clean_ledger=True)
        return dict(profile=store.write('PROFILE.json',encoded(result),kind='TIMING_PREFLIGHT'))
    if stage in ('CLEAN','EXPANDED'):
        from hydra_directflow.coverage_extension_io import read_input_bundle
        from hydra_directflow.coverage_extension_inference import clean_regression,expanded_inference
        record=next(r for r in plan['original_records'] if r['job']['job_id']==job_id);job=record['job']
        model=next(r for r in reconciliation['models'] if r['job_id']==job_id)
        record={**record,'original_input_manifest_sha256':model['original_clean_input']['sha256']}
        ids=json.loads(store.read(reconciliation['gradient_membership_artifacts'][model['gradient_case_ids_sha256']],8*1024**2))
        require(digest(ids['case_ids'])==ids['case_ids_sha256']==model['gradient_case_ids_sha256'],'Gradient membership differs')
        clean=read_input_bundle(store,inputs['bundles'][job['gauge_id']][str(job['history_hours'])]['clean'])
        common=dict(phase_plan_sha256=plan['plan_sha256'],precharge_forward=store.precharge_forward,confirm_forward=store.confirm_forward,
            precharge_model=store.precharge_model,confirm_model=store.confirm_model,check=store.check,read_budget=store.read_budget)
        if stage=='CLEAN':
            result=clean_regression(old,record,plan['original_run_directory'],tuple(ids['case_ids']),clean,**common)
            require(result['receipt']['saved_predictor_receipt']['training_values_sha256']==model['training_values_sha256'],'Saved transform training values differ')
        else:
            profile=completed(store,ledger,'CLEAN_PROFILE')
            measured=json.loads(store.read(profile['profile'],256*1024))
            require(measured['status']=='FIXED_REAL_CLEAN_PROFILE_PASSED'
                    and measured['phase_plan_sha256']==plan['plan_sha256'],'Measured preflight proof changed')
            barrier_data=completed(store,ledger,'CLEAN_BARRIER');barrier=json.loads(store.read(barrier_data['barrier'],256*1024))
            clean_data=completed(store,ledger,'CLEAN__'+job_id);clean_result=stored_result(store,clean_data,job)
            expanded=read_input_bundle(store,inputs['bundles'][job['gauge_id']][str(job['history_hours'])]['expanded'])
            result=expanded_inference(old,record,plan['original_run_directory'],tuple(ids['case_ids']),clean,expanded,clean_result,
                clean_barrier=barrier,expected_barrier_sha256=barrier_data['content_sha256'],**common)
        return save_result(store,result)
    if stage=='CLEAN_BARRIER':
        hashes={};timing=[]
        for record in plan['original_records']:
            job=record['job'];item=completed(store,ledger,'CLEAN__'+job['job_id']);result=stored_result(store,item,job)
            require(result['receipt']['status']=='BYTE_EXACT_ORIGINAL_PREDICTIONS'
                    and result['receipt']['checkpoint_sha256']==plan['checkpoint_sha256_by_job'][job['job_id']]
                    and isinstance(result['receipt']['saved_transform_sha256'],str)
                    and len(result['receipt']['saved_transform_sha256'])==64,'Clean regression failed')
            hashes[job['job_id']]=digest(result['receipt']);timing.append(dict(job_id=job['job_id'],**result['receipt']['accounting']))
        require(len(plan['original_records'])==len(hashes)==72 and set(hashes)==set(plan['checkpoint_sha256_by_job']),
                'All72 exact checkpoint clean regressions required')
        barrier=dict(schema='hydra-coverage-extension-clean-barrier/v1',status='ALL72_CLEAN_REGRESSIONS_BYTE_EXACT',
            phase_plan_sha256=plan['plan_sha256'],clean_receipt_sha256=hashes)
        return dict(barrier=store.write('CLEAN_BARRIER.json',encoded(barrier),kind='CLEAN_BARRIER'),content_sha256=digest(barrier),
                    timing=store.write('CLEAN_TIMING.json',encoded(dict(jobs=timing)),kind='TIMING'))
    if stage=='ROUTED':
        from hydra_directflow.coverage_extension_io import read_route
        from hydra_directflow.coverage_extension_scoring import assemble_routed
        outputs={}
        for gauge in GAUGES:
            route=read_route(store,inputs['routes'][gauge]);a=route['arrays'];origin=a['origin_time'];future=a['forecast_time']
            for architecture in ('LSTM','Transformer','GRU_Transformer','Mamba2_Transformer'):
                for seed in (17,29,43):
                    streams={};weights={};sources={};descs={}
                    for length in (168,24):
                        job_id=f'final_development__{gauge}__{architecture}__full_L{length}__seed{seed}'
                        item=completed(store,ledger,'EXPANDED__'+job_id);job=next(r['job'] for r in plan['original_records'] if r['job']['job_id']==job_id)
                        result=stored_result(store,item,job);v=result['arrays'];pos=np.searchsorted(origin,v['origin_time'])
                        expected=a['eligible'+str(length)];require(np.array_equal(np.flatnonzero(expected),pos),'Input product membership differs from input-only route')
                        full=np.full((8766,18),np.nan,np.float64);full[pos]=v['prediction_q'];streams[length]=full
                        weights[str(length)]=result['receipt']['checkpoint_sha256'];sources[str(length)]=result['receipt']['input_source_identity'];descs[str(length)]=item['arrays']['sha256']
                    routed=assemble_routed(gauge_id=gauge,origins=origin,forecast_time=future,eligible168=a['eligible168'],eligible24=a['eligible24'],route=a['route'],
                        predictions168=streams[168],predictions24=streams[24])
                    # The pure router may return a mapping; the public contract
                    # below always contains the actual unchanged H18 vector.
                    values=routed['prediction_q'] if isinstance(routed,dict) else routed
                    arrays=dict(prediction_q=np.asarray(values,np.float64),origin_time=origin,forecast_time=future,case_id=a['case_id'],site_id=a['site_id'],
                        route_history_hours=np.where(a['route']=='L168',168,np.where(a['route']=='L24',24,0)).astype(np.int16),
                        route_checkpoint_index=np.where(a['route']=='L168',0,np.where(a['route']=='L24',1,-1)).astype(np.int8))
                    receipt=dict(schema='hydra-coverage-extension-routed-predictions/v1',status='ALL_INPUT_ELIGIBLE_ROUTED_VECTORS_COMMITTED',
                        product='ROUTED_CLEAN_168_24_V1',phase_plan_sha256=plan['plan_sha256'],original_plan_sha256=old['plan_sha256'],
                        gauge_id=gauge,architecture=architecture,seed=seed,checkpoint_sha256_by_history=weights,
                        route_map_sha256=inputs['routes'][gauge]['route_map_sha256'],
                        input_source_identity=digest(dict(candidate=inputs['routes'][gauge]['candidate_source_identity'],gauge=gauge,route=inputs['routes'][gauge]['route_map_sha256'],views=sources)),
                        source_input_view_identity_by_history=sources,source_input_prediction_artifact_sha256_by_history=descs,
                        predictions_array_sha256=array_identity(arrays),requested_origins=8766,emitted_cases=int(a['eligible24'].sum()),
                        history_hours='MIXED_PER_ORIGIN',target_values_read=False,target_masks_read=False,model_forwards=0,protected_assessment_access=False)
                    key=f'{gauge}__{architecture}__seed{seed}'
                    outputs[key]=dict(arrays=store.write('routed/'+key+'.npz',npz(arrays),kind='ROUTED_PREDICTIONS'),
                        receipt=store.write('routed/'+key+'.json',encoded(receipt),kind='ROUTED_RECEIPT'))
        return dict(products=outputs)
    if stage=='PREDICTION_BARRIER':
        from hydra_directflow.coverage_extension_io import read_route
        clean_barrier=completed(store,ledger,'CLEAN_BARRIER');routed=completed(store,ledger,'ROUTED')['products']
        clean={};expanded={};expanded_arrays={}
        for record in plan['original_records']:
            job=record['job'];key=job['job_id'];clean[key]=completed(store,ledger,'CLEAN__'+key);stored_result(store,clean[key],job)
            expanded[key]=completed(store,ledger,'EXPANDED__'+key);expanded_arrays[key]=stored_result(store,expanded[key],job)['arrays']
        expected_routes={f'{g}__{a}__seed{s}' for g in GAUGES for a in ('LSTM','Transformer','GRU_Transformer','Mamba2_Transformer') for s in (17,29,43)}
        require(len(clean)==len(expanded)==72 and set(clean)==set(expanded)==set(plan['checkpoint_sha256_by_job'])
                and set(routed)==expected_routes,'Complete prediction inventory required')
        route_views={g:read_route(store,inputs['routes'][g])['arrays'] for g in GAUGES}
        for key,stored in routed.items():
            result=stored_result(store,stored);receipt=result['receipt'];values=result['arrays']
            gauge,architecture,seed_text=key.split('__');seed=int(seed_text.removeprefix('seed'));route=route_views[gauge]
            require(receipt['gauge_id']==gauge and receipt['architecture']==architecture and receipt['seed']==seed
                    and receipt['status']=='ALL_INPUT_ELIGIBLE_ROUTED_VECTORS_COMMITTED'
                    and receipt['route_map_sha256']==inputs['routes'][gauge]['route_map_sha256'], 'Routed result identity differs')
            expected=np.full((8766,18),np.nan,np.float64)
            for length,label in ((168,'L168'),(24,'L24')):
                job=f'final_development__{gauge}__{architecture}__full_L{length}__seed{seed}'
                require(receipt['checkpoint_sha256_by_history'][str(length)]==plan['checkpoint_sha256_by_job'][job], 'Routed weight identity differs')
                source=expanded_arrays[job];positions=np.searchsorted(route['origin_time'],source['origin_time'])
                chosen=route['route'][positions]==label;expected[positions[chosen]]=source['prediction_q'][chosen]
            emitted=route['eligible24']
            require(np.isfinite(values['prediction_q'][emitted]).all() and np.isnan(values['prediction_q'][~emitted]).all()
                    and np.array_equal(values['prediction_q'][emitted],expected[emitted])
                    and np.array_equal(values['route_history_hours'],np.where(route['route']=='L168',168,np.where(route['route']=='L24',24,0)))
                    and np.array_equal(values['route_checkpoint_index'],np.where(route['route']=='L168',0,np.where(route['route']=='L24',1,-1))),
                    'Routed forecast differs from its one selected whole-H18 vector')
        body={k:v for k,v in inputs.items() if k!='receipt_artifact'}
        barrier=dict(schema='hydra-coverage-extension-prediction-barrier/v1',status='ALL_REQUESTED_PREDICTIONS_COMMITTED_BEFORE_TARGETS',
            phase_plan_sha256=plan['plan_sha256'],route_barrier_sha256=inputs['route_barrier_content_sha256'],
            clean_barrier_sha256=clean_barrier['content_sha256'],input_preparation_receipt=inputs['receipt_artifact'],
            input_preparation_content_sha256=digest(body),clean_products=clean,input_products=expanded,routed_products=routed,
            target_values_loaded=False,target_quality_loaded=False)
        return dict(barrier=store.write('PREDICTION_BARRIER.json',encoded(barrier),kind='PREDICTION_BARRIER'),content_sha256=digest(barrier))
    if stage=='EVALUATION':
        from hydra_directflow.coverage_extension_evaluate import evaluate
        entry=completed(store,ledger,'PREDICTION_BARRIER');barrier=json.loads(store.read(entry['barrier'],2*1024**2))
        result=evaluate(plan,store,inputs,barrier,entry['content_sha256'],reconciliation,check=store.check,read_budget=store.read_budget)
        return dict(index=result['receipt_artifact'],status=result['status'],product_reports=144)
    if stage=='REPORT':
        from report_coverage_extension import build_publication
        from hydra_directflow.coverage_extension_io import read_route
        evaluated=completed(store,ledger,'EVALUATION');index=json.loads(store.read(evaluated['index'],16*1024**2))
        state=CampaignLedger(old['ledger_directory']).snapshot()
        require(state['optimizer_updates_entered']==state['optimizer_updates_confirmed']==1151690,'Original training counters changed')
        accounting=ledger.snapshot();accounting.update(original_campaign_deadline=plan['original_campaign_deadline'],
            original_optimizer_updates_before=1151690,original_optimizer_updates_after=state['optimizer_updates_confirmed'])
        publication=build_publication(index,lambda d:json.loads(store.read(d,16*1024**2)),accounting)
        files={name:store.write('publication/'+name,raw,kind='PUBLIC_REPORT') for name,raw in publication['files'].items()}
        barrier_entry=completed(store,ledger,'PREDICTION_BARRIER');barrier=json.loads(store.read(barrier_entry['barrier'],2*1024**2))
        plot_data={}
        for gauge in GAUGES:
            route=read_route(store,inputs['routes'][gauge])['arrays']
            target=bounded_arrays(store,index['targets_by_gauge'][gauge]['arrays']);ref=bounded_arrays(store,index['targets_by_gauge'][gauge]['reference'])
            matrix=np.empty((4,3,8766,18),np.float64)
            for ai,arch in enumerate(ARCHITECTURES):
                for si,seed in enumerate((17,29,43)):
                    key=f'{gauge}__{arch}__seed{seed}'
                    matrix[ai,si]=stored_result(store,barrier['routed_products'][key])['arrays']['prediction_q']
            arrays=dict(origin_time=route['origin_time'],forecast_time=route['forecast_time'],route=route['route'],
                original_common=np.isin(route['case_id'],reconciliation['original_common_case_ids_by_gauge'][gauge]),
                routed_predictions=matrix,reference_time=ref['times'],reference_value=ref['values'],reference_mask=ref['admitted'],
                verification_target_values=target['labels'],verification_target_mask=target['label_mask'])
            plot_data[gauge]=store.write('plot-inputs/'+gauge+'.npz',npz(arrays),kind='PRIVATE_FIGURE_INPUTS')
            del arrays,matrix,target,ref
        plot=dict(phase_plan_sha256=plan['plan_sha256'],prediction_barrier_sha256=index['prediction_barrier_sha256'],
                  summary=files['summary.json'],gauges=plot_data)
        plot_descriptor=store.write('PLOT_INPUTS.json',encoded(plot),kind='FIGURE_SOURCE_MANIFEST')
        output=store.directory/'figures';log=store.directory/'figure-render.log'
        store.read_budget(plot_descriptor['bytes']+files['summary.json']['bytes']+sum(d['bytes'] for d in plot_data.values())+2*1024**2)
        token=ledger.reserve_output(16*1024**2,store.lease,kind='SCIENTIFIC_FIGURES')
        command=['/opt/homebrew/bin/python3.11','-I','-B',str(ROOT/'scripts/render_coverage_extension.py'),
            '--run',str(store.run),'--manifest',str(store.run/plot_descriptor['path']),'--manifest-sha256',plot_descriptor['sha256'],
            '--output',str(output)]
        child=None
        try:
            with log.open('xb') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'MPLBACKEND':'Agg','MPLCONFIGDIR':str(store.directory/'matplotlib-cache')},
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    store.check();require(log.stat().st_size<=1024**2,'Figure renderer log exceeded bound');time.sleep(.5)
                require(child.returncode==0,'Figure renderer failed; partial outputs retained')
            artifact_files=[]
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    require(path==path.resolve() and path.is_relative_to(output),'Figure output escaped its private directory')
                    store.read_budget(path.lstat().st_size);raw=_read(path,16*1024**2)
                    artifact_files.append(dict(path=str(path.relative_to(store.run)),bytes=len(raw),sha256=sha(raw)))
            actual=sum(x['bytes'] for x in artifact_files);require(actual<=16*1024**2,'Figure output reservation exceeded')
            ledger.settle_output(store.lease,token,actual_bytes=actual,sha256=digest(artifact_files))
        except BaseException:
            if child is not None and child.poll() is None:child.terminate();child.wait(timeout=15)
            actual=primary.folder_bytes(output) if output.exists() else 0
            ledger.settle_output(store.lease,token,actual_bytes=min(actual,16*1024**2),failed=True);raise
        return dict(publication_files=files,figures=artifact_files,figure_manifest=next(x for x in artifact_files if x['path'].endswith('/MANIFEST.json')),
                    evaluation_index=evaluated['index'],phase_a_descriptive_results_complete=True,visual_review_pending=True)
    raise ValueError('Unknown Phase A stage')


def task(plan,old,key,stage,*,job_id=None,resume=False,supervisor_pid=None,release_marker=None):
    marker=release_marker if release_marker is not None else release(plan);ledger=PhaseALedger(plan['ledger_directory'])
    with ledger.writer():
        ledger.initialize(plan,marker['original_reservation_token']);lease=ledger.claim_task(key,stage,resume=resume)
        device=next((r['job']['device'] for r in plan['original_records'] if r['job']['job_id']==job_id),None)
        store=PhaseStore(plan,ledger,lease,plan['original_run_directory'],supervisor_pid=supervisor_pid,job_device=device)
        started=time.time()
        try:
            store.heartbeat(stage=stage,job_id=job_id,force=True)
            data=handle(plan,old,ledger,store,stage,job_id)
            receipt=dict(**ledger.receipt_bindings(lease),status='VERIFIED_PHASE_A_TASK',data=data,elapsed_seconds=time.time()-started,
                         optimizer_updates=0,scaler_fits=0,protected_assessment_access=False)
            def validate(value,bindings):
                for item in descriptors(value['data']):store.read(item,512*1024**2)
                return digest(value)
            store.write('COMPLETE.json',encoded(receipt),kind='TASK_COMPLETION')
            ledger.complete_task(lease,receipt,validator=validate)
            return receipt
        except (KeyboardInterrupt,ExtensionAllowanceExhausted):
            ledger.interrupt(lease,'SUPERVISED_INTERRUPTION_OR_RESOURCE_BOUND');raise
        except BaseException as error:
            kind='NUMERICAL' if any(x in str(error).lower() for x in ('nonfinite','byte parity','overflow')) else 'INTEGRITY' if isinstance(error,(ValueError,AssertionError,OSError)) else 'IMPLEMENTATION'
            ledger.fail(lease,kind,type(error).__name__);raise


def progress(plan,status,*,worker_pid=None,error=None):
    snapshot=PhaseALedger(plan['ledger_directory']).snapshot()
    data=dict(schema='hydra-phase-a-progress/v1',status=status,updated_at_utc=utc(),phase_plan_sha256=plan['plan_sha256'],
        supervisor_pid=os.getpid(),worker_pid=worker_pid,deadline_utc=datetime.fromtimestamp(snapshot['deadline'],timezone.utc).isoformat(),
        completed_tasks=sum(t['state']=='COMPLETED' for t in snapshot['tasks']),
        active_tasks=[dict(task_key=t['task_key'],stage=t['stage'],state=t['state']) for t in snapshot['tasks'] if t['state']!='COMPLETED'],
        counters={k:snapshot[k] for k in ('entered_cases','confirmed_cases','entered_batches','confirmed_batches','entered_models','confirmed_models','read_reserved','output_reserved','infrastructure_resumes')},
        error=error,optimizer_updates=0,scaler_fits=0,protected_assessment_access=False,paid_cloud_jobs=0)
    run=Path(plan['run_directory']);primary.atomic(run/'progress.json',data)
    rows=''.join('<tr><th>'+html.escape(k)+'</th><td>'+html.escape(str(v))+'</td></tr>' for k,v in data['counters'].items())
    body='<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="15"><title>DirectFlow Phase A</title></head><body style="font:16px system-ui;margin:40px;max-width:1100px"><h1>DirectFlow · inference-only coverage extension</h1><p>'+html.escape(status)+'</p><p>'+html.escape(str(data['active_tasks']))+'</p><table>'+rows+'</table><p>Zero new training. Protected assessment remains sealed.</p></body></html>'
    temp=run/'progress.html.writing';temp.write_text(body);os.replace(temp,run/'progress.html')
    return data


def failure_record(plan,category,error):
    run=Path(plan['run_directory'])
    target=run/('SUPERVISION_FAILURE_RESUMED.json' if (run/'SUPERVISION_RESUME.json').exists() else 'SUPERVISION_FAILURE.json')
    value=dict(phase_plan_sha256=plan['plan_sha256'],category=category,observed_at_utc=utc(),error=error)
    if not target.exists():primary.atomic(target,value,new=True)
    return value


def filesystem_bounds(plan):
    run=Path(plan['run_directory']);bookkeeping=0;total=0
    for path in run.rglob('*'):
        require(not path.is_symlink(),'Phase A output symlink refused')
        if not path.is_file():continue
        try:size=path.stat().st_size
        except FileNotFoundError:continue  # Owned atomic status publication.
        total+=size
        relative=path.relative_to(run)
        if len(relative.parts)==1 or relative.parts[0]=='ledger' or path.name.startswith('heartbeat.json') \
                or path.name=='figure-render.log' or 'matplotlib-cache' in relative.parts:
            bookkeeping+=size
    if total>LIMITS['max_output_bytes'] or bookkeeping>BOOKKEEPING:
        raise ExtensionAllowanceExhausted('Actual phase output/bookkeeping allowance exceeded')
    return dict(actual_output_bytes=total,actual_bookkeeping_bytes=bookkeeping)


def supervise(plan_path,plan,old,*,resume=False,resume_diagnosis=None):
    run=Path(plan['run_directory']);lock=os.open(run/'supervisor.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ledger=PhaseALedger(plan['ledger_directory']);child=None;armed=False
    previous=signal.signal(signal.SIGTERM,lambda *unused:(_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        marker=release(plan)
        with ledger.writer():ledger.initialize(plan,marker['original_reservation_token'])
        initial=run/'SUPERVISION_START.json';resumed=run/'SUPERVISION_RESUME.json'
        if initial.exists():
            require(resume and resume_diagnosis is not None and not resumed.exists(),
                    'Existing supervision requires the single diagnosed infrastructure resume')
            diagnosis_raw=_read(primary.private(resume_diagnosis),128*1024);diagnosis=json.loads(diagnosis_raw)
            require(diagnosis.get('phase_plan_sha256')==plan['plan_sha256']
                    and diagnosis.get('classification')=='INFRASTRUCTURE'
                    and isinstance(diagnosis.get('evidence'),str) and diagnosis['evidence'].strip(),
                    'Concrete infrastructure diagnosis required; do not reclassify numerical/integrity failures')
            require(not any(t['state'].startswith('FAILED_') for t in ledger.snapshot()['tasks']),
                    'A failed numerical/integrity/implementation task cannot be hidden by resume')
            for path in run.glob('SUPERVISION_FAILURE*.json'):
                recorded=json.loads(_read(path,128*1024))
                require(recorded['category']=='INFRASTRUCTURE','Recorded non-infrastructure failure cannot be relabeled for resume')
            primary.atomic(resumed,dict(phase_plan_sha256=plan['plan_sha256'],created_at_utc=utc(),pid=os.getpid(),
                diagnosis_sha256=sha(diagnosis_raw),diagnosis=diagnosis),new=True)
        else:
            require(not resume,'Cannot resume supervision that never started')
            primary.atomic(initial,dict(phase_plan_sha256=plan['plan_sha256'],created_at_utc=utc(),pid=os.getpid()),new=True)
        armed=True
        schedule=[('A0_RECONCILE','RECONCILE',None),('A2_INPUTS','INPUTS',None)]
        for number,r in enumerate(plan['original_records']):
            schedule.append(('CLEAN__'+r['job']['job_id'],'CLEAN',r['job']['job_id']))
            if number==7:schedule.append(('CLEAN_PROFILE','CLEAN_PROFILE',None))
        schedule += [('CLEAN_BARRIER','CLEAN_BARRIER',None)]
        schedule += [('EXPANDED__'+r['job']['job_id'],'EXPANDED',r['job']['job_id']) for r in plan['original_records']]
        schedule += [(s,s,None) for s in ('ROUTED','PREDICTION_BARRIER','EVALUATION','REPORT')]
        for key,stage,job_id in schedule:
            filesystem_bounds(plan)
            existing=ledger.task(key)
            if existing and existing['state']=='COMPLETED':continue
            resume_task=bool(resume and existing is not None and existing['state']=='INTERRUPTED')
            if stage not in ('CLEAN','EXPANDED'):
                progress(plan,'RUNNING_'+stage);task(plan,old,key,stage,job_id=job_id,resume=resume_task,release_marker=marker)
                resume=False;continue
            ordinal=len(ledger.snapshot()['attempts'])+1;log_path=run/f'worker-{ordinal:06d}.log'
            command=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'--worker','--stage',stage,'--job-id',job_id,
                '--plan',str(plan_path),'--approved-plan-sha256',plan['plan_sha256'],'--supervisor-pid',str(os.getpid())]
            if resume_task:command.append('--resume')
            env={**os.environ,'PYTHONNOUSERSITE':'1','PYTORCH_ENABLE_MPS_FALLBACK':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
            with ledger.writer():
                ledger.initialize(plan,marker['original_reservation_token'])
                ledger.reserve_read(AUTH_READ_RESERVE,purpose='WORKER_AUTHENTICATION_METADATA')
            with log_path.open('xb') as output:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=output,start_new_session=True)
                last=0.
                while child.poll() is None:
                    state=ledger.snapshot();now=time.time();require(now<state['deadline'],'Persistent Phase A deadline reached')
                    require(log_path.stat().st_size<=8*1024**2,'Worker log limit reached')
                    heartbeat=None;active=next((a for a in state['attempts'] if a['status']=='ACTIVE'),None)
                    if active:
                        path=run/'tasks'/key/f"attempt-{active['id']:06d}"/'heartbeat.json'
                        if path.exists():heartbeat=json.loads(read_snapshot(path,64*1024))
                    driver=0 if heartbeat is None else heartbeat.get('mps_driver_bytes',0)
                    if heartbeat is not None:require(heartbeat.get('pid')==child.pid
                        and heartbeat.get('phase_plan_sha256')==plan['plan_sha256']
                        and heartbeat.get('attempt_id')==active['id'] and heartbeat.get('task_key')==key,
                        'Foreign worker heartbeat')
                    try:memory=psutil.Process().memory_info().rss+psutil.Process(child.pid).memory_info().rss+driver
                    except psutil.NoSuchProcess:break
                    if memory>LIMITS['max_rss_bytes'] or driver>LIMITS['max_mps_bytes']:
                        raise ExtensionAllowanceExhausted('Parent/worker/MPS memory limit reached')
                    if __import__('shutil').disk_usage(run).free<LIMITS['min_free_bytes']:
                        raise ExtensionAllowanceExhausted('Disk resource limit reached')
                    filesystem_bounds(plan)
                    if time.monotonic()-last>2:progress(plan,'RUNNING_'+stage,worker_pid=child.pid);last=time.monotonic()
                    time.sleep(.5)
                child.wait(timeout=15)
            if child.returncode:
                with ledger.writer():pass
                state=ledger.task(key)
                category='INFRASTRUCTURE' if child.returncode<0 or state and state['state']=='INTERRUPTED' else 'UNCLASSIFIED_TECHNICAL'
                if state and state['state'].startswith('FAILED_'):category=state['state'][7:]
                failure_record(plan,category,dict(task=key,exit_code=child.returncode))
                return progress(plan,'STOPPED_WORKER_FAILURE',error=dict(task=key,exit_code=child.returncode))
            settled=ledger.task(key)
            require(settled is not None and settled['state']=='COMPLETED','Worker exited without authenticated completion')
            child=None;resume=False
        return progress(plan,'PHASE_A_EXECUTION_COMPLETE_FINAL_REVIEW_AND_PUBLICATION_PENDING')
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        if not armed:raise
        with ledger.writer():pass
        category='INFRASTRUCTURE' if isinstance(error,(KeyboardInterrupt,ExtensionAllowanceExhausted)) else 'INTEGRITY' if isinstance(error,(ValueError,AssertionError)) else 'IMPLEMENTATION'
        failure_record(plan,category,dict(type=type(error).__name__,message=str(error)[:400]))
        progress(plan,'STOPPED_SUPERVISOR_GUARD',error=dict(type=type(error).__name__,message=str(error)[:400]));raise
    finally:
        signal.signal(signal.SIGTERM,previous);fcntl.flock(lock,fcntl.LOCK_UN);os.close(lock)


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    for name in ('prepare','execute','worker'):mode.add_argument('--'+name,action='store_true')
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--original-plan',type=Path);p.add_argument('--adopted-plan',type=Path)
    p.add_argument('--approved-plan-sha256');p.add_argument('--resume',action='store_true');p.add_argument('--resume-diagnosis',type=Path)
    p.add_argument('--stage');p.add_argument('--job-id');p.add_argument('--supervisor-pid',type=int)
    args=p.parse_args()
    if args.prepare:
        require(args.original_plan and args.adopted_plan,'Original and adopted plans required');plan=prepare(args.original_plan,args.adopted_plan)
        primary.atomic(primary.private(args.plan),plan,new=True);print(json.dumps(dict(status='PREPARED_NOT_EXECUTED',plan_sha256=plan['plan_sha256'])));return
    require(args.approved_plan_sha256,'Exact approved execution plan required');plan,old,marker=counted_authentication(args.plan,args.approved_plan_sha256)
    if args.worker:
        require(args.stage in ('CLEAN','EXPANDED') and args.job_id in plan['checkpoint_sha256_by_job'] and args.supervisor_pid,
                'Only an explicit approved model worker is supported')
        key=args.stage+'__'+args.job_id;result=task(plan,old,key,args.stage,job_id=args.job_id,resume=args.resume,
            supervisor_pid=args.supervisor_pid,release_marker=marker)
        print(json.dumps(dict(status=result['status'],task=key)));return
    if not Path(plan['run_directory']).exists():initialize(plan,old)
    else:release(plan)
    result=supervise(args.plan,plan,old,resume=args.resume,resume_diagnosis=args.resume_diagnosis);print(json.dumps(dict(status=result['status'],counters=result['counters'])))


if __name__=='__main__':main()
