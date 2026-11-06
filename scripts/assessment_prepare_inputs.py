"""Resumable assessment input preparation under the authenticated runtime.

No command-line entry point, target-role read, model load, fit, or standalone
permission switch is provided. The controller supplies its authenticated
authority, source plan/provider, and single-writer runtime. Original numerical
cycles and their provenance are retained before provider-owned raw eviction.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import stat
import uuid
import zipfile

import numpy as np

from hydra_directflow.assessment_contract import origins
from hydra_directflow.assessment_data import (
    FIRST_INPUT, LAST_INPUT, _gate, _array_identity as input_array_identity, assessment_cycle_reference,
    load_assessment_cycle, load_assessment_usgs, assemble_assessment_inputs)
from hydra_directflow.assessment_inference import COORDINATES, INPUT_ARRAYS, METADATA
from hydra_directflow.local_data import (
    GAUGES, LocalCycle, WEIGHTS_SHA256, USGS_MANIFEST_SHA256, basin_reduce, iso)
from hydra_directflow.monthly import cycle_ids_for_origins
from hydra_directflow.reference import HOUR, digest, require
from hydra_directflow.staging import atomic_json
from hydra_directflow.training import FULL_FEATURE_SCHEMA, array_identity

SCHEMA='hydra-assessment-prepared-inputs/v1'
CYCLE_SCHEMA='hydra-assessment-cycle-cache/v1'
CHUNK_HOURS=120
COMPACT_KEYS=COORDINATES+INPUT_ARRAYS+('input_admitted','eligibility_reason')
GIB=1024**3


def _sha(value):return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None
def _encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def _npz(arrays):
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);return buffer.getvalue()


def _arrays(raw, *, maximum=256*1024**2):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries=archive.infolist()
        require(0<len(entries)<=64 and len({e.filename for e in entries})==len(entries)
            and sum(e.file_size for e in entries)<=maximum
            and all(re.fullmatch(r'[a-z][a-z0-9_]*\.npy',e.filename) for e in entries),
            'Bounded unique plain NPZ arrays required')
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        result={key:archive[key] for key in archive.files}
    require(all(v.dtype.kind in 'fbiuUS' and v.ndim<=4 for v in result.values()),'Plain typed input arrays required')
    for value in result.values():value.setflags(write=False)
    return result


def _identity(runtime,authority,source_identity,staging_plan):
    _gate(authority)
    require(authority.verification_enabled is False and authority.prediction_barrier_sha256 is None
        and 168 in authority.allowed_history_hours,'Preparation is restricted to the released input phase')
    require(_sha(source_identity) and authority.execution_plan_sha256==runtime.plan['plan_sha256']
        and source_identity==runtime.plan['input_source_identity'], 'Runtime/release/input-source identity differs')
    require(staging_plan.get('schema')=='hydra-assessment-staging-plan/v1'
        and staging_plan.get('status')=='APPROVED_BOUNDED_STAGING'
        and staging_plan.get('plan_sha256')==digest({k:v for k,v in staging_plan.items() if k!='plan_sha256'})
        and staging_plan['sources']['source_inventory']['sha256']==authority.source_inventory_sha256,
        'Exact approved staging plan and inventory binding required')
    return dict(schema=SCHEMA,execution_plan_sha256=authority.execution_plan_sha256,
        source_inventory_sha256=authority.source_inventory_sha256,input_source_identity=source_identity,
        staging_plan_sha256=staging_plan['plan_sha256'],weights_sha256=WEIGHTS_SHA256,
        source_manifest_sha256=USGS_MANIFEST_SHA256,history_hours=168,
        feature_schema=FULL_FEATURE_SCHEMA,target_values_assembled=False)


def _resident_probe_size(path,maximum):
    """Stat only: never hydrate a cloud-only candidate or follow a symlink."""
    p=Path(path)
    try:info=p.lstat()
    except FileNotFoundError:return 0
    return info.st_size if (stat.S_ISREG(info.st_mode) and not getattr(info,'st_flags',0)&0x40000000
                           and 0<info.st_size<=maximum and p.absolute()==p.resolve()) else 0


def _charge_provider_probes(runtime,provider,row):
    # The provider may test several resident candidates before finding a hit.
    # Precharge all candidates as an explicit conservative logical-read bound.
    # A newly downloaded commit is subsequently read by the lineage resolver.
    cid=row['cycle_id'];total=16384 if row.get('commit_identity') else 0
    for label,folder,suffix,maximum in (('local_commit_candidates','commits','.json',16384),
            ('local_archive_candidates','raw','.tar',16*1024**2)):
        candidates=[provider.directory/folder/(cid+suffix)]+[
            v['path'] if isinstance(v,Mapping) else v for v in row.get(label,[])]
        total+=sum(_resident_probe_size(p,maximum) for p in candidates)
    runtime.reserve_read(total,numeric=False,purpose='CONSERVATIVE_SOURCE_PROVIDER_LOCAL_PROBES')


def _evict_owned(runtime,provider,provider_result,receipt):
    ownership=provider_result['ownership']
    if not ownership['owned_archive']:return
    p=Path(ownership['archive_path'])
    try:info=p.lstat()
    except FileNotFoundError:return
    require(stat.S_ISREG(info.st_mode) and not getattr(info,'st_flags',0)&0x40000000
        and 0<info.st_size<=16*1024**2,'Owned raw eviction source must remain resident')
    runtime.reserve_read(info.st_size,numeric=False,purpose='OWNED_RAW_HASH_BEFORE_VERIFIED_CACHE_EVICTION')
    provider.on_cache_committed(receipt,provider_result)


def _cycle_binding(binding,cid,row):
    return dict(binding,cache_schema=CYCLE_SCHEMA,cycle_id=cid,sealed_row_sha256=digest(row))


def _restore_cycle(arrays,metadata,binding,weights):
    require(metadata['binding']==binding and metadata['schema']==CYCLE_SCHEMA
        and metadata['arrays_sha256']==array_identity(arrays), 'Restored cycle binding or typed payload differs')
    require(set(arrays)=={'values','present','means','masks','coverage'},'Exact full raw and reduced cache fields required')
    data=metadata['local_cycle'];n=len(data['records'])
    require(data['reference']==assessment_cycle_reference(binding['cycle_id'])
        and type(data['member_count']) is int and data['member_count']>=0
        and isinstance(data['fields'],dict) and isinstance(data['availability'],dict)
        and data['availability']['cycle']==binding['cycle_id']
        and arrays['values'].shape==arrays['present'].shape==(n,957)
        and arrays['values'].dtype==np.dtype('float64') and arrays['present'].dtype==np.dtype(bool)
        and np.isfinite(arrays['values'][arrays['present']]).all(), 'Truthful source cycle geometry required')
    expected=basin_reduce(arrays['values'],arrays['present'],weights)
    for key,value in zip(('means','masks','coverage'),expected):
        require(arrays[key].dtype==value.dtype and np.array_equal(arrays[key],value,equal_nan=True),
                'Cached basin reduction differs from frozen full-area weights')
    receipt=metadata['loader_receipt']
    require(receipt['cycle_id']==binding['cycle_id'] and receipt['sealed_row_sha256']==binding['sealed_row_sha256']
        and receipt['execution_plan_sha256']==binding['execution_plan_sha256']
        and _sha(receipt['archive_sha256']) and type(receipt['archive_bytes']) is int
        and 0<receipt['archive_bytes']<=16*1024**2 and receipt['verification_target_arrays_read'] is False,
        'Cached original source receipt differs')
    identity=dict(binding,archive_sha256=receipt['archive_sha256'],archive_bytes=receipt['archive_bytes'],
                  original_manifest_sha256=receipt['manifest_sha256'])
    require(metadata['cache_key']==digest(identity),'Cache key must bind row, original archive, manifest and weights')
    cycle=LocalCycle(data['reference'],arrays['values'],arrays['present'],tuple(data['records']),
                     data['member_count'],data['fields'],data['availability'])
    return cycle,(arrays['means'],arrays['masks'],arrays['coverage'])


def _cached_cycle(runtime,authority,provider,plan_row,index_row,weights,binding):
    cid=plan_row['cycle_id'];identity=_cycle_binding(binding,cid,index_row);key='CYCLE/'+cid
    existing=runtime.reuse(key)
    if existing is not None:
        require(existing['metadata']['binding']==identity,'Existing cycle derivation differs')
        metadata=json.loads(runtime.read(existing['artifacts']['metadata'],numeric=False))
        arrays=_arrays(runtime.read(existing['artifacts']['arrays']),maximum=16*1024**2)
        cycle,reduced=_restore_cycle(arrays,metadata,identity,weights)
        require(existing['metadata']['cache_key']==metadata['cache_key'],'Cycle commitment and payload identity differ')
        proof=dict(cache_payload_verified=True,cache_key=metadata['cache_key'],cache_commit_sha256=digest(existing),
                   archive_sha256=metadata['loader_receipt']['archive_sha256'])
        _evict_owned(runtime,provider,metadata['provider_result'],proof)
        return cycle,reduced,existing
    _charge_provider_probes(runtime,provider,plan_row)
    supplied=provider(cid,index_row)
    require(supplied['status']=='READY' and supplied['spec']['cycle_id']==cid,'Explicit pinned source delivery required')
    cycle,loader=load_assessment_cycle(supplied['spec'],index_row,authority,
        read_budget=lambda n:runtime.reserve_read(n,purpose='AUTHENTICATED_HRRR_INPUT'))
    require(loader['archive_sha256']==supplied['ownership']['archive_sha256'], 'Delivered archive differs from decoded source')
    known=plan_row.get('bundle_identity')
    if known:require(loader['archive_sha256']==known['sha256'] and loader['archive_bytes']==known['bytes'],
                     'Decoded archive differs from pinned source plan')
    means,masks,coverage=basin_reduce(cycle.values,cycle.present,weights)
    arrays=dict(values=cycle.values,present=cycle.present,means=means,masks=masks,coverage=coverage)
    cache_key=digest(dict(identity,archive_sha256=loader['archive_sha256'],archive_bytes=loader['archive_bytes'],
                          original_manifest_sha256=loader['manifest_sha256']))
    metadata=dict(schema=CYCLE_SCHEMA,binding=identity,cache_key=cache_key,arrays_sha256=array_identity(arrays),
        local_cycle=dict(reference=cycle.reference,records=cycle.records,member_count=cycle.member_count,
                         fields=cycle.fields,availability=cycle.availability),
        loader_receipt=loader,provider_result=supplied,targets_assembled=False,
        reduction='Frozen original full-area weights; missing positive-weight cells exclude the mean')
    _restore_cycle(arrays,metadata,identity,weights)
    prefix='cycle-cache/'+cid+'/attempt-'+uuid.uuid4().hex
    artifacts=dict(arrays=runtime.write(prefix+'.npz',_npz(arrays),kind='FULL_RAW_AND_BASIN_CYCLE'),
        metadata=runtime.write(prefix+'.json',_encoded(metadata),kind='ORIGINAL_CYCLE_PROVENANCE',numeric=False))
    commit=runtime.commit(key,artifacts,metadata=dict(binding=identity,cache_key=cache_key))
    proof=dict(cache_payload_verified=True,cache_key=cache_key,cache_commit_sha256=digest(commit),archive_sha256=loader['archive_sha256'])
    _evict_owned(runtime,provider,supplied,proof)
    return cycle,(means,masks,coverage),commit


def _usgs(runtime,authority,usgs_paths,manifest_path,binding):
    # Read exactly the input union once. Causal input hours can also be targets
    # for earlier origins; the reader explicitly records internal Parquet decode.
    source=load_assessment_usgs(usgs_paths,manifest_path,[(FIRST_INPUT,LAST_INPUT)],authority,mode='input',
        read_budget=lambda n:runtime.reserve_read(n,purpose='AUTHENTICATED_USGS_INPUT_ROLE'))
    receipt={key:value for key,value in source.items() if key not in ('observations','original_rows')}
    receipt.update(schema='hydra-assessment-usgs-input-receipt/v1',binding=binding,
        original_rows_returned=len(source['original_rows']),input_observation_count=len(source['observations']),
        target_values_assembled=False)
    existing=runtime.reuse('USGS_INPUT_SOURCE',expected_metadata=binding)
    if existing:
        require(json.loads(runtime.read(existing['artifacts']['receipt'],numeric=False))==json.loads(_encoded(receipt)),
                'Reopened original USGS input receipt differs')
        commit=existing
    else:
        desc=runtime.write('input-source/USGS_INPUT_RECEIPT-'+uuid.uuid4().hex+'.json',_encoded(receipt),kind='USGS_INPUT_ROLE_RECEIPT',numeric=False)
        commit=runtime.commit('USGS_INPUT_SOURCE',dict(receipt=desc),metadata=binding)
    return source['observations'],commit


def _chunk_binding(binding,number,calendar,usgs_commit,cycle_rows):
    return dict(binding,chunk_index=number,origin_count=len(calendar),origin_calendar_sha256=digest(list(calendar)),
        usgs_source_commit_sha256=digest(usgs_commit),
        candidate_cycle_rows_sha256=digest([(cid,digest(row)) for cid,row in sorted(cycle_rows.items())]))


def _check_chunk(arrays,manifest,calendar,binding):
    require(manifest['preparation_binding']==binding and manifest['arrays_sha256']==input_array_identity(arrays)
        and manifest['source_identity']==binding['input_source_identity']
        and manifest['execution_plan_sha256']==binding['execution_plan_sha256']
        and manifest['source_inventory_sha256']==binding['source_inventory_sha256']
        and manifest['target_values_read'] is False and manifest['target_masks_read'] is False,
        'Full-provenance input chunk differs from its identity')
    require(not any(k in arrays for k in ('target','target_mask','target_time','nwm')),
            'Input preparation cannot contain verification or comparator values')
    c=np.repeat(np.asarray(calendar,dtype=np.int64),3);s=np.tile(np.arange(3),len(calendar))
    require(np.array_equal(arrays['origin_time'],c) and np.array_equal(arrays['site_id'],s)
        and arrays['case_id'].tolist()==[GAUGES[int(i)]+':'+iso(int(t)) for i,t in zip(s,c)]
        and np.array_equal(arrays['history_time'],c[:,None]-np.arange(169,1,-1)*HOUR)
        and np.array_equal(arrays['forecast_time'],c[:,None]+np.arange(1,19)*HOUR),
        'Input chunk must preserve all exact three-gauge origin/history/horizon coordinates')
    n=len(c)
    for key,shape in (('history',(n,168,2)),('weather',(n,18,4)),('state',(n,2))):
        values,mask=arrays[key],arrays[key+'_mask']
        require(values.shape==mask.shape==shape and values.dtype==np.dtype('float64')
            and mask.dtype==np.dtype(bool) and np.isfinite(values[mask]).all(),'Chunk input/mask geometry differs')
    require(arrays['input_admitted'].dtype==np.dtype(bool) and arrays['input_admitted'].shape==(n,)
        and np.array_equal(arrays['input_admitted'],arrays['history_mask'].all(axis=(1,2))
            &arrays['weather_mask'].all(axis=(1,2))&arrays['state_mask'].all(axis=1)),
        'Admission must depend only on complete original input masks')
    cohort=digest(dict(source_identity=binding['input_source_identity'],case_ids=arrays['case_id'].tolist(),
                      input_admitted=arrays['input_admitted'].tolist(),history_hours=168))
    require(arrays['feature_schema'].item()==FULL_FEATURE_SCHEMA
        and arrays['inference_source_identity'].item()==binding['input_source_identity']
        and arrays['cohort_admission_digest'].item()==manifest['cohort_admission_digest']==cohort,
        'Chunk input source/cohort scalars differ')


def _progress(runtime,provider,number,total,processed):
    budget=provider.budget.snapshot()
    value=dict(status='ASSEMBLING_INPUTS',completed_input_chunks=number,total_input_chunks=total,
        processed_candidate_cycles=len(processed),source_download_observed_bytes=budget['observed_body_bytes'],
        source_download_precharged_bytes=budget['precharged_body_bytes'],source_get_attempts=budget['attempted_gets'],
        source_budget=budget,execution_plan_sha256=runtime.plan['plan_sha256'],
        verification_targets_assembled=False,model_calls=0,optimizer_updates=0,
        updated_at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(runtime.directory/'progress.json',value)
    print(json.dumps(value,sort_keys=True),flush=True)


def _prepare_chunks(runtime,authority,provider,plan_rows,index_rows,weights,observations,usgs_commit,binding,calendar):
    """Bounded core shared with synthetic small-calendar qualification."""
    pieces={g:{k:[] for k in COMPACT_KEYS} for g in GAUGES}
    commits=[];hot={};processed=set();total=(len(calendar)+CHUNK_HOURS-1)//CHUNK_HOURS
    for number,start in enumerate(range(0,len(calendar),CHUNK_HOURS),1):
        runtime.check();part=tuple(calendar[start:start+CHUNK_HOURS]);ids=cycle_ids_for_origins(list(part))
        rows={cid:index_rows[cid] for cid in ids};cb=_chunk_binding(binding,number,part,usgs_commit,rows)
        key=f'INPUT_CHUNK/{number:03d}';existing=runtime.reuse(key,expected_metadata=cb)
        if existing:
            arrays=_arrays(runtime.read(existing['artifacts']['arrays']),maximum=512*1024**2)
            manifest=json.loads(runtime.read(existing['artifacts']['manifest'],numeric=False));commit=existing
        else:
            # Only this chunk's at-most23 candidate cycles stay in RAM. Three
            # overlapping candidates survive into the next chronological chunk.
            hot={cid:value for cid,value in hot.items() if cid in ids}
            for cid in ids:
                if cid not in hot:
                    hot[cid]=_cached_cycle(runtime,authority,provider,plan_rows[cid],index_rows[cid],weights,binding)
            source=assemble_assessment_inputs({cid:hot[cid][0] for cid in ids},
                {cid:hot[cid][1] for cid in ids},observations,list(part),authority,
                source_identity=binding['input_source_identity'],history_hours=168)
            arrays=source['arrays'];manifest=dict(source['manifest'],preparation_binding=cb,
                usgs_input_receipt=usgs_commit['artifacts']['receipt'],
                cycle_cache_commits=[dict(cycle_id=cid,cache_commit_sha256=digest(hot[cid][2]),
                    artifacts=hot[cid][2]['artifacts']) for cid in ids])
            _check_chunk(arrays,manifest,part,cb)
            prefix=f'input-chunks/{number:03d}/attempt-'+uuid.uuid4().hex
            artifacts=dict(arrays=runtime.write(prefix+'.npz',_npz(arrays),kind='FULL_PROVENANCE_INPUT_CHUNK'),
                manifest=runtime.write(prefix+'.json',_encoded(manifest),kind='INPUT_CHUNK_MANIFEST',numeric=False))
            commit=runtime.commit(key,artifacts,metadata=cb)
        _check_chunk(arrays,manifest,part,cb)
        for si,gauge in enumerate(GAUGES):
            take=arrays['site_id']==si
            for field in COMPACT_KEYS:pieces[gauge][field].append(np.array(arrays[field][take],copy=True))
        commits.append(commit);processed.update(ids)
        _progress(runtime,provider,number,total,processed)
        del arrays,manifest
        if 'source' in locals():del source
    return commits,pieces


def _gauge_bundles(runtime,commits,pieces,binding,calendar):
    bundles={};references=[dict(chunk_key=c['key'],commit_sha256=digest(c),**c['artifacts']) for c in commits]
    for si,gauge in enumerate(GAUGES):
        arrays={key:np.concatenate(pieces[gauge][key],axis=0) for key in COMPACT_KEYS}
        pieces[gauge].clear()
        require(np.array_equal(arrays['origin_time'],calendar) and np.all(arrays['site_id']==si)
            and len(set(arrays['case_id'].tolist()))==len(calendar),'Compact gauge calendar differs')
        cohort=digest(dict(source_identity=binding['input_source_identity'],gauge_id=gauge,history_hours=168,
            case_ids=arrays['case_id'].tolist(),input_admitted=arrays['input_admitted'].tolist(),
            provenance_chunks=references))
        arrays.update(feature_schema=np.array(FULL_FEATURE_SCHEMA),cohort_admission_digest=np.array(cohort),
                      inference_source_identity=np.array(binding['input_source_identity']))
        manifest=dict(schema=SCHEMA,binding=binding,gauge_id=gauge,source_identity=binding['input_source_identity'],
            arrays_sha256=array_identity(arrays),cohort_admission_digest=cohort,feature_schema=FULL_FEATURE_SCHEMA,
            requested_origins=len(calendar),admitted_origins=int(arrays['input_admitted'].sum()),
            first_origin_utc=iso(calendar[0]),last_origin_utc=iso(calendar[-1]),
            full_provenance_chunks=references,target_values_read=False,target_masks_read=False,
            model_calls=0,optimizer_updates=0,scaler_fits=0,numerical_input_filling=False)
        key='GAUGE_INPUTS/'+gauge;meta=dict(binding,gauge_id=gauge,cohort_admission_digest=cohort,
                                          arrays_sha256=manifest['arrays_sha256'])
        existing=runtime.reuse(key,expected_metadata=meta)
        if existing:
            require(json.loads(runtime.read(existing['artifacts']['manifest'],numeric=False))==manifest,
                    'Existing compact gauge manifest differs');bundles[gauge]=existing['artifacts']
        else:
            prefix='gauge-inputs/'+gauge+'/attempt-'+uuid.uuid4().hex
            artifacts=dict(arrays=runtime.write(prefix+'.npz',_npz(arrays),kind='COMPACT_GAUGE_INPUTS'),
                manifest=runtime.write(prefix+'.json',_encoded(manifest),kind='GAUGE_INPUT_MANIFEST',numeric=False))
            runtime.commit(key,artifacts,metadata=meta);bundles[gauge]=artifacts
    return bundles


def prepare_inputs(runtime,authority,provider,staging_plan,index_rows,usgs_paths,manifest_path,weights_path,source_identity):
    """Return immutable73-chunk provenance and three complete8760-origin bundles."""
    binding=_identity(runtime,authority,source_identity,staging_plan)
    calendar=origins();require(len(calendar)==8760,'The frozen assessment has8760 exact origin hours')
    candidates=cycle_ids_for_origins(list(calendar));plan_rows={r['cycle_id']:r for r in staging_plan['cycles']}
    require(isinstance(index_rows,Mapping) and len(plan_rows)==len(staging_plan['cycles'])==1463
        and set(plan_rows)==set(index_rows)==set(candidates)==set(dict(authority.cycle_row_hashes)),
        'All1463 exact source candidates must be explicitly bound')
    for cid in candidates:
        require(plan_rows[cid]['sealed_row_sha256']==digest(index_rows[cid])==dict(authority.cycle_row_hashes)[cid],
                'Source plan/index/release row differs')
    require(getattr(provider,'plan',{}).get('plan_sha256')==staging_plan['plan_sha256'],
            'Provider is not attached to the exact approved source plan')
    expected=runtime.plan['weights'];p=Path(weights_path)
    require(str(p)==expected['path'] and expected['sha256']==WEIGHTS_SHA256 and expected['bytes']==62601,
            'Frozen original spatial-weight descriptor required')
    # Imported lazily: the additive reader retains original candidate flags and
    # requires the separate assessment authority instead of rewriting them.
    from hydra_directflow.assessment_data import read_assessment_weights
    weights,weights_receipt=read_assessment_weights(p,authority,
        read_budget=lambda n:runtime.reserve_read(n,purpose='FROZEN_BASIN_WEIGHTS'))
    existing=runtime.reuse('FROZEN_BASIN_WEIGHTS',expected_metadata=binding)
    if existing:
        require(json.loads(runtime.read(existing['artifacts']['receipt'],numeric=False))==weights_receipt,
                'Frozen original spatial receipt differs')
    else:
        descriptor=runtime.write('input-source/WEIGHTS_RECEIPT-'+uuid.uuid4().hex+'.json',_encoded(weights_receipt),
                                 kind='FROZEN_WEIGHTS_PROVENANCE',numeric=False)
        runtime.commit('FROZEN_BASIN_WEIGHTS',dict(receipt=descriptor),metadata=binding)
    observations,usgs_commit=_usgs(runtime,authority,usgs_paths,manifest_path,binding)
    commits,pieces=_prepare_chunks(runtime,authority,provider,plan_rows,index_rows,weights,
                                   observations,usgs_commit,binding,calendar)
    bundles=_gauge_bundles(runtime,commits,pieces,binding,calendar)
    return dict(chunk_commits=commits,gauge_bundles=bundles,input_source_identity=source_identity)
