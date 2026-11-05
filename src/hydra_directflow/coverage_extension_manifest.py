"""Read-only Phase A model/source reconciliation; no model or target decoding."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from . import corpus_dataset as ds
from .campaign_data import SOURCE_PLAN, SNAPSHOT, INTEGRITY_PATH, INTEGRITY_SHA256
from .contract import ARCHITECTURES, GAUGE_IDS
from .reference import digest, require

HOUR = 3600
PLAN_SHA = '2a68d386dabe553bdda2e77f07c99c89e4081b8dc899296162fd203d0c81fd4a'
SEEDS = (17, 29, 43)


def read(path, maximum, *, read_budget=lambda n: None, expected=None):
    path = Path(path); size = path.lstat().st_size
    require(0 < size <= maximum, 'Manifest input exceeds byte bound')
    read_budget(size); raw = ds._read(path, maximum)
    require(expected is None or hashlib.sha256(raw).hexdigest() == expected, 'Manifest input hash differs')
    return raw


def descriptor(path, base, raw):
    path, base = Path(path), Path(base)
    require(path == path.resolve() and path.is_relative_to(base), 'Manifest artifact escaped its base')
    return dict(path=str(path.relative_to(base)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def source_authority(corpus, *, read_budget=lambda n: None):
    corpus = Path(corpus).absolute(); ds._directory(corpus)
    # _plan consumes metadata only; charge its immutable PLAN before reuse.
    read_budget((corpus/'PLAN.json').lstat().st_size)
    plan, chunks, plan_file_sha = ds._plan(corpus, SOURCE_PLAN)
    receipt = json.loads(read(INTEGRITY_PATH, 4*1024**2, read_budget=read_budget, expected=INTEGRITY_SHA256))
    require(receipt['status']=='COMPLETE_COMMITTED_CORPUS_SNAPSHOT' and receipt['snapshot_sha256']==SNAPSHOT
            and receipt['plan_sha256']==SOURCE_PLAN and receipt['plan_file_sha256']==plan_file_sha
            and receipt['missing_chunks']==[] and receipt['numerical_payloads_checked'] is True,
            'Previously verified complete development snapshot required')
    observed = receipt['observed_commits']
    require(len(chunks)==len(observed) and digest(dict(plan_sha256=SOURCE_PLAN,
            plan_file_sha256=plan_file_sha,observed=observed,missing=[]))==SNAPSHOT, 'Snapshot commitment inventory differs')
    mapping = {}
    resident_files = resident_bytes = 0
    for (unit, chunk), item in zip(chunks, observed):
        name=unit['id']+'/'+f"{chunk['index']:03d}"
        require(item['chunk']==name, 'Snapshot chunk order differs')
        directory=corpus/'chunks'/name; ds._directory(directory)
        require({p.name for p in directory.iterdir()}=={'COMMIT.json','case-index.json','model.npz'}, 'Unexpected source shard inventory')
        for path in directory.iterdir():
            stat=path.lstat()
            require(path==path.resolve() and path.is_file() and not getattr(stat,'st_flags',0)&0x40000000,
                    'Source must be resident; no implicit hydration')
            resident_files+=1;resident_bytes+=stat.st_size
        mapping[name] = dict(first=ds._hour(chunk['first_origin_utc']),origins=chunk['origin_count'],
                             commit_sha256=item['commit_sha256'])
    return mapping, dict(snapshot_sha256=SNAPSHOT,source_plan_sha256=SOURCE_PLAN,
        plan_file_sha256=plan_file_sha,integrity_receipt_sha256=INTEGRITY_SHA256,
        resident_shard_files=resident_files,resident_shard_bytes=resident_bytes,full_integrity_audit_replayed=False)


def gradient_membership(manifest, gauge, chunks, expected_sha):
    """Recover case IDs from authenticated row metadata, without numerical reads.

    The pinned corpus contract stores three gauges in order at every exact hour.
    The recovered list must match BOTH the saved fit manifest and scaler receipt.
    """
    require(gauge in GAUGE_IDS, 'Known gauge required')
    gi = GAUGE_IDS.index(gauge); window = manifest['role_time_bounds']['fit']
    start,end=ds._hour(window['start_utc']),ds._hour(window['end_utc'])
    refit=ds._hour(manifest['refit_at_utc']); result=[]
    require(start < end <= refit == ds._hour('2023-07-31T00:00:00Z')
            and type(manifest['roles']['fit']['cases']) is int and manifest['roles']['fit']['cases']>0,
            'Exact final-development refit and positive gradient window required')
    require(len({m['source_chunk'] for m in manifest['members']})==len(manifest['members']),
            'Duplicate gradient member chunk')
    for member in manifest['members']:
        require(member['source_chunk'] in chunks, 'Unknown gradient source chunk')
        chunk=chunks[member['source_chunk']]
        require(member['commit_sha256']==chunk['commit_sha256'], 'Gradient chunk identity differs')
        rows=member['rows']
        require(isinstance(rows,list) and rows==sorted(set(rows)), 'Unique ordered source rows required')
        for row in rows:
            require(type(row) is int and 0<=row<3*chunk['origins'] and row%3==gi, 'Wrong gauge or source row')
            origin=chunk['first']+(row//3)*HOUR
            if start<=origin and origin+18*HOUR<end and origin+20*HOUR<=refit:
                result.append(gauge+':'+datetime.fromtimestamp(origin,timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
    require(result==sorted(set(result)) and len(result)==manifest['roles']['fit']['cases']
            and digest(result)==manifest['roles']['fit']['case_ids_sha256']==expected_sha,
            'Recovered gradient case identity differs from saved transform')
    return tuple(result)


def original_common_ids(manifest, gauge, chunks, expected_sha):
    """Original regression membership only; never used to admit a new route."""
    require(gauge in GAUGE_IDS and manifest['operation']=='inputs' and manifest['gauge_id']==gauge
            and manifest['fold']=='final_development' and manifest['input_arm']=='full'
            and manifest['history_hours'] in (24,168) and manifest['source_plan_sha256']==SOURCE_PLAN
            and manifest['snapshot_sha256']==SNAPSHOT and type(manifest['roles']['outer']['cases']) is int
            and manifest['roles']['outer']['cases']>0, 'Original input manifest and gauge required')
    window=manifest['role_time_bounds']['outer']
    require(window==dict(start_utc='2023-07-31T00:00:00Z',end_utc='2024-07-31T00:00:00Z'), 'Original final outer window differs')
    require(len({m['source_chunk'] for m in manifest['members']})==len(manifest['members']), 'Duplicate original common chunk')
    gi=GAUGE_IDS.index(gauge);result=[]
    for member in manifest['members']:
        require(member['source_chunk'] in chunks, 'Unknown original common chunk')
        chunk=chunks[member['source_chunk']]
        require(member['commit_sha256']==chunk['commit_sha256'], 'Original common chunk identity differs')
        require(member['rows']==sorted(set(member['rows'])), 'Original common rows repeated or unordered')
        for row in member['rows']:
            require(type(row) is int and 0<=row<3*chunk['origins'] and row%3==gi, 'Wrong original common gauge/row')
            origin=chunk['first']+row//3*HOUR
            require(ds._hour(window['start_utc'])<=origin and origin+18*HOUR<ds._hour(window['end_utc']), 'Original common clock differs')
            result.append(gauge+':'+datetime.fromtimestamp(origin,timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
    require(result==sorted(set(result)) and len(result)==manifest['roles']['outer']['cases']
            and digest(result)==manifest['roles']['outer']['case_ids_sha256']==expected_sha, 'Original common membership differs')
    return tuple(result)


def reconcile_models(original_plan, records, corpus, run, *, read_budget=lambda n:None):
    """Authenticate all72 checkpoint blobs as opaque bytes and bind fit metadata."""
    run=Path(run).absolute(); require(str(run)==original_plan['run_directory'] and run==run.resolve(), 'Exact original run required')
    chunks, source = source_authority(corpus, read_budget=read_budget)
    selected=[r for r in records if r['job']['fold']=='final_development' and r['job']['input_arm']=='full']
    expected={(g,a,h,s) for g in GAUGE_IDS for a in ARCHITECTURES for h in (168,24) for s in SEEDS}
    require(len(selected)==72 and {(r['job']['gauge_id'],r['job']['architecture'],r['job']['history_hours'],r['job']['seed']) for r in selected}==expected,
            'Exact72 final-development full-input models required')
    memberships={}; common_ids={};models=[]; cache={}
    def artifact(relative, maximum, expected_sha=None):
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'Safe original artifact path required')
        path=run/relative;require(path==path.resolve() and path.is_relative_to(run), 'Original artifact escaped run')
        if relative not in cache:
            raw=read(path,maximum,read_budget=read_budget,expected=expected_sha);cache[relative]=(raw,descriptor(path,run,raw))
        raw,desc=cache[relative]
        require(expected_sha is None or desc['sha256']==expected_sha, 'Cached original identity differs')
        return raw,desc
    for record in selected:
        job,receipt=record['job'],record['receipt']; base=Path(receipt['receipt_path']).parent
        require(type(job['order']) is int and 0<=job['order']<len(original_plan['jobs'])
                and job==original_plan['jobs'][job['order']], 'Job differs from original immutable plan')
        require(receipt['status']=='VERIFIED_TRAIN_EVAL' and receipt['protected_assessment_access'] is False
                and receipt['plan_sha256']==original_plan['plan_sha256'] and receipt['job_sha256']==digest(job)
                and all(receipt[k]==original_plan[k] for k in ('protocol_sha256','source_plan_sha256','snapshot_sha256'))
                and receipt['feature_schema']=='FULL_HRRR_H2_W4_S2_V1', 'Verified original job required')
        checkpoint=receipt['artifacts']['checkpoint']
        require(checkpoint['sha256']==receipt['hashes']['checkpoint'], 'Checkpoint descriptor differs from receipt')
        raw,cp=artifact(checkpoint['path'],16*1024**2,checkpoint['sha256']);require(cp==checkpoint,'Checkpoint size differs')
        # Free the large immutable model body; identity remains in the record.
        cache.pop(checkpoint['path'],None); del raw
        raw,fit_desc=artifact(str(base/'fit-source.json'),2*1024**2);fit=json.loads(raw)
        core={k:v for k,v in fit.items() if k not in ('source_identity','returned_arrays_sha256','construction')}
        require(digest(core)==fit['source_identity']==receipt['training_source_identity']
                and fit['cohort_admission_digest']==receipt['training_cohort_admission_digest']
                and fit['fold']=='final_development' and fit['gauge_id']==job['gauge_id']
                and fit['history_hours']==job['history_hours'] and fit['input_arm']=='full', 'Authenticated training-source manifest differs')
        raw,inf_desc=artifact(str(base/'inference.json'),256*1024);infer=json.loads(raw);last=infer['last_batch_inference_receipt']
        require(infer['checkpoint_sha256']==cp['sha256'] and last['checkpoint_sha256']==cp['sha256']
                and last['selected_state']=='best_model' and last['selected_best_epoch']==receipt['selected_best_epoch']
                and last['architecture']==job['architecture'] and last['gauge_id']==job['gauge_id']
                and last['history_hours']==job['history_hours'] and last['seed']==job['seed']
                and last['device']==job['device'] and last['optimizer_constructed'] is False
                and last['scaler_fitted'] is False and last['feature_schema']==receipt['feature_schema'],
                'Saved inference/model identity differs')
        key=last['training_case_ids_sha256']
        ids=gradient_membership(fit,job['gauge_id'],chunks,key)
        if key in memberships:require(memberships[key]==ids,'Conflicting gradient membership')
        else:memberships[key]=ids
        raw,input_desc=artifact(str(base/'outer-input-source.json'),2*1024**2);inputs=json.loads(raw)
        core={k:v for k,v in inputs.items() if k not in ('source_identity','returned_arrays_sha256','construction')}
        require(digest(core)==inputs['source_identity']==receipt['inference_source_identity']
                and inputs['target_values_decoded'] is False and inputs['numeric_value_members_decoded']==['history','weather','state'],
                'Original clean input source is not label-free')
        original_ids=original_common_ids(inputs,job['gauge_id'],chunks,receipt['outer_case_ids_sha256'])
        require(len(original_ids)==receipt['scored_cases'], 'Original common receipt count differs')
        if job['gauge_id'] in common_ids:require(common_ids[job['gauge_id']]==original_ids,'Original common model memberships differ')
        else:common_ids[job['gauge_id']]=original_ids
        pred=receipt['artifacts']['predictions']
        require(pred['sha256']==receipt['hashes']['predictions'], 'Prediction descriptor differs from receipt')
        _,pred_desc=artifact(pred['path'],64*1024**2,pred['sha256']);require(pred_desc==pred,'Original prediction bytes differ')
        cache.pop(pred['path'],None)
        models.append(dict(job_id=job['job_id'],gauge_id=job['gauge_id'],architecture=job['architecture'],
            seed=job['seed'],history_hours=job['history_hours'],device=job['device'],feature_schema=receipt['feature_schema'],
            checkpoint=cp,selected_best_epoch=receipt['selected_best_epoch'],
            gradient_case_ids_sha256=key,gradient_cases=len(ids),training_values_sha256=last['training_values_sha256'],
            saved_transform_sha256_status='CHECKPOINT_BYTES_BOUND_NOW_EXACT_TRANSFORM_DIGEST_RECORDED_ON_CLEAN_LOAD',
            original_clean_input=input_desc,original_predictions=pred_desc,fit_source=fit_desc,original_inference_receipt=inf_desc,
            training_source_identity=receipt['training_source_identity'],training_cohort_admission_digest=receipt['training_cohort_admission_digest'],
            original_common_case_ids_sha256=receipt['outer_case_ids_sha256'],original_common_cases=receipt['scored_cases'],
            original_completion_receipt_sha256=record['receipt_sha256']))
    return dict(models=models,gradient_memberships=memberships,original_common_case_ids_by_gauge=common_ids,source=source,
                model_constructions=0,model_forwards=0,checkpoint_deserializations=0,target_values_decoded=False)
