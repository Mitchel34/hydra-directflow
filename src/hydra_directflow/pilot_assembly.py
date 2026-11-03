"""Bounded, immutable pilot inputs joined to original monthly USGS provenance.

This module prepares data only. It does not authorize fitting or reconstruct
historical publication vintages. No network, hydration, model or scaler API is
used. The complete corpus receipt is reused; only explicitly planned shards
and monthly caches are reopened. Predictor inputs and labels are separate files.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
import uuid
import zipfile

import numpy as np

from . import corpus_dataset as ds
from .cache import DerivedCache, unpack_month, _directory, _sync_directory
from .local_data import GAUGES, USGS_MANIFEST_SHA256, bounded_npy, iso
from .quality import PRIMARY_APPROVED, quality_admitter
from .reference import ASSESSMENT_START, digest, require
from .trainer_bundle import _roles, _refit_policy

SCHEMA = 'hydra-pilot-assembly/v1'
ROLES = ('fit', 'inner_stop', 'outer')
INPUT_KEYS = ('origin_time', 'history_time', 'target_time', 'site_id', 'case_id',
              'history', 'history_mask', 'weather', 'weather_mask', 'state', 'state_mask',
              'source_cycle', 'soil_source_cycle', 'weather_source_leads')
LABEL_KEYS = ('origin_time', 'target_time', 'site_id', 'case_id', 'target', 'target_mask')
MAX_BYTES = 1024**3
MAX_METADATA = 8 * 1024**2
MAX_PART = 16 * 1024**2


def code_identity():
    names = ('pilot_assembly.py', 'corpus_dataset.py', 'cache.py', 'local_data.py',
             'reference.py', 'quality.py', 'trainer_bundle.py', 'pilot_inputs.py')
    return {name: hashlib.sha256(ds._read(Path(__file__).parent / name, 512 * 1024)).hexdigest() for name in names}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _resident(path):
    try:
        info = Path(path).lstat()
        return stat.S_ISREG(info.st_mode) and not getattr(info, 'st_flags', 0) & 0x40000000
    except FileNotFoundError:
        return False


def _checked(path, identity, maximum=MAX_METADATA):
    require(isinstance(identity, dict) and ds._hex(identity['sha256']) and type(identity['bytes']) is int
            and 0 < identity['bytes'] <= maximum, 'Explicit bounded source identity required')
    raw = ds._read(path, identity['bytes'])
    require(_identity(raw) == identity, 'Pinned file identity differs')
    return raw


def _selected_role(origin, roles, refit):
    for name, (start, end) in roles.items():
        if start <= origin < end and origin + 18*3600 < end:
            if name == 'outer' or origin + 20*3600 <= refit:
                return name
    return None


def plan_metadata(*, corpus_root, integrity_receipt_path, expected_integrity_sha256,
                  expected_plan_sha256, expected_snapshot_sha256, month_sources,
                  role_time_bounds, refit_at_utc, expected_roles,
                  max_read_bytes=MAX_BYTES, max_write_bytes=MAX_BYTES):
    """Metadata-only plan; missing/dataless provenance returns BLOCKED, never opens it.

    ``expected_roles[gauge][role]`` contains ``cases`` and chronological
    ``case_ids_sha256``. ``month_sources[month]`` contains ``entry_path``,
    ``cache_key`` and externally pinned ``commit_sha256``.
    """
    require(all(type(x) is int and 0 < x <= MAX_BYTES for x in (max_read_bytes, max_write_bytes)), 'Bounded read/write limits required')
    roles = _roles(role_time_bounds)
    require(set(roles) == set(ROLES), 'All three explicit pilot roles required')
    _refit_policy(dict(refit_at_utc=refit_at_utc, simulated_label_delay_hours=2), roles)
    refit = ds._hour(refit_at_utc)
    root = Path(corpus_root).absolute(); ds._directory(root)
    receipt_raw = ds._read(integrity_receipt_path, MAX_METADATA)
    require(ds._hex(expected_integrity_sha256) and hashlib.sha256(receipt_raw).hexdigest() == expected_integrity_sha256, 'Integrity receipt SHA differs')
    receipt = json.loads(receipt_raw)
    plan, descriptors, plan_file_sha = ds._plan(root, expected_plan_sha256)
    require(receipt['status'] == 'COMPLETE_COMMITTED_CORPUS_SNAPSHOT' and receipt['missing_chunks'] == []
            and receipt['stored_history_hours'] == plan['history_hours'] == 168
            and receipt['plan_sha256'] == expected_plan_sha256 and receipt['plan_file_sha256'] == plan_file_sha
            and receipt['snapshot_sha256'] == expected_snapshot_sha256 and receipt['numerical_payloads_checked'] is True
            and receipt['protected_assessment_access'] is False, 'Complete pinned development integrity receipt required')
    observed = receipt['observed_commits']
    require(len(observed) == len(descriptors) and len({x['chunk'] for x in observed}) == len(observed), 'Duplicate/incomplete snapshot commitments')
    require([x['chunk'] for x in observed] == [u['id'] + '/' + f"{c['index']:03d}" for u,c in descriptors]
            and all(ds._hex(x['commit_sha256']) for x in observed), 'Snapshot commitment order differs')
    require(digest(dict(plan_sha256=expected_plan_sha256, plan_file_sha256=plan_file_sha,
                        observed=observed, missing=[])) == expected_snapshot_sha256, 'Snapshot digest differs')
    require(set(expected_roles) == set(GAUGES), 'Explicit expected counts for all three gauges required')
    for gauge in GAUGES:
        require(set(expected_roles[gauge]) == set(ROLES), 'Expected pilot role inventory differs')
        for item in expected_roles[gauge].values():
            require(set(item) == {'cases', 'case_ids_sha256'} and type(item['cases']) is int
                    and 0 <= item['cases'] <= 100000 and ds._hex(item['case_ids_sha256']), 'Expected role count/hash differs')
    shards, needed, unavailable = [], {}, []
    # All reads below are small commitments/summaries, never an NPZ body.
    for (unit, chunk), observed_row in zip(descriptors, observed):
        first, last = ds._hour(chunk['first_origin_utc']), ds._hour(chunk['last_origin_utc'])
        if not any(_selected_role(t, roles, refit) for t in range(first,last+1,3600)):
            continue
        relative = observed_row['chunk']; directory = root / 'chunks' / relative
        raw = ds._read(directory / 'COMMIT.json', 65536); commit = json.loads(raw)
        require(hashlib.sha256(raw).hexdigest() == observed_row['commit_sha256'] and commit['plan_sha256'] == expected_plan_sha256
                and commit['unit_id'] == unit['id'] and commit['chunk_index'] == chunk['index'], 'Source shard commit differs')
        summary_raw = _checked(directory / 'case-index.json', commit['payloads']['case-index.json'], 256*1024)
        summary = json.loads(summary_raw)
        require(summary['input_quality_policy'] == summary['target_quality_policy'] == PRIMARY_APPROVED,
                'Primary source quality policy required')
        require(set(summary['month_keys']) == set(ds._months(first-169*3600,last+18*3600)), 'Source month coverage differs')
        files = {'COMMIT.json':_identity(raw), **commit['payloads']}
        for name, identity in files.items():
            require(name in ('COMMIT.json','case-index.json','model.npz') and ds._hex(identity['sha256'])
                    and type(identity['bytes']) is int and 0 < identity['bytes'] <= MAX_PART, 'Invalid shard file bound')
            if not _resident(directory/name): unavailable.append(str(directory/name))
        selected_origins=[t for t in range(first,last+1,3600) if _selected_role(t,roles,refit)]
        required_months=set()
        for origin in selected_origins:
            required_months.update(ds._months(origin-169*3600,origin+18*3600))
        for month in sorted(required_months):
            key=summary['month_keys'][month]
            require(ds._hex(key) and (month not in needed or needed[month] == key), 'Conflicting monthly provenance keys')
            needed[month] = key
        shards.append(dict(unit=unit,chunk=chunk,relative=relative,commit_sha256=observed_row['commit_sha256'],files=files,month_keys=summary['month_keys']))
    require(shards and len(shards) <= 477, 'Bounded nonempty source shard selection required')
    months = {}
    for month,key in sorted(needed.items()):
        # Native monthly data crossing the assessment is never admitted here.
        from .cache import _month_bounds
        _month_bounds(month)
        spec = month_sources.get(month)
        if spec is None:
            unavailable.append('MISSING_MONTH_SPEC:'+month); continue
        require(set(spec) == {'entry_path','cache_key','commit_sha256'} and spec['cache_key'] == key
                and ds._hex(spec['commit_sha256']), 'Monthly provenance key/commit pin differs')
        directory = Path(spec['entry_path']).absolute()
        if not _resident(directory/'COMMIT.json'):
            unavailable.append(str(directory/'COMMIT.json')); continue
        raw = ds._read(directory/'COMMIT.json',128*1024); commit = json.loads(raw)
        identity = commit['identity']
        require(hashlib.sha256(raw).hexdigest() == spec['commit_sha256'] and commit['kind'] == 'usgs_month'
                and commit['key'] == key == DerivedCache.key('usgs_month',identity)
                and identity['month'] == month and identity['source_manifest_sha256'] == USGS_MANIFEST_SHA256,
                'Monthly source lineage differs')
        require(set(commit['payloads']) == {'arrays','metadata'}, 'Monthly payload inventory differs')
        files = {'COMMIT.json':_identity(raw)}
        for label,item in commit['payloads'].items():
            suffix = '.npz' if label == 'arrays' else '.json'
            require(ds._hex(item['sha256']) and item['name'] == item['sha256']+suffix
                    and type(item['bytes']) is int and 0 < item['bytes'] <= MAX_PART, 'Monthly content address/bound differs')
            files[item['name']] = {k:item[k] for k in ('bytes','sha256')}
            if not _resident(directory/item['name']): unavailable.append(str(directory/item['name']))
        months[month] = dict(**spec,identity=identity,files=files)
    worst_read = 2*sum(v['bytes'] for s in shards for v in s['files'].values()) + 2*sum(v['bytes'] for m in months.values() for v in m['files'].values()) + len(receipt_raw)+4*1024**2
    require(worst_read <= max_read_bytes, 'Planned bounded source reads exceed allowance')
    result = dict(schema=SCHEMA,status='BLOCKED_MISSING_RESIDENT_PROVENANCE' if unavailable else 'READY_LOCAL_ONLY',
        corpus_root=str(root),integrity_receipt_path=str(Path(integrity_receipt_path).absolute()),integrity_identity=_identity(receipt_raw),
        source_plan_sha256=expected_plan_sha256,source_plan_file_sha256=plan_file_sha,snapshot_sha256=expected_snapshot_sha256,
        role_time_bounds=role_time_bounds,refit_at_utc=refit_at_utc,simulated_label_delay_hours=2,
        expected_roles=expected_roles,cohort_mode='common_l168_complete',quality_policy=PRIMARY_APPROVED,
        shards=shards,months=months,unavailable=unavailable,implementation_sha256=code_identity(),
        limits=dict(max_read_bytes=max_read_bytes,max_write_bytes=max_write_bytes,max_part_bytes=MAX_PART,
                    max_training_bundle_bytes=MAX_BYTES,max_batch_cases=64),planned_source_read_bytes_upper_bound=worst_read,
        scientific_fits_authorized=False,physical_units='m3/s',historical_delivery_vintage_known=False)
    result['plan_sha256'] = digest(result)
    return result


class _Budget:
    def __init__(self, plan, check): self.plan,self.check,self.read,self.written=plan,check,0,0
    def charge(self, amount, write=False):
        self.check(); key='written' if write else 'read'; setattr(self,key,getattr(self,key)+amount)
        require(getattr(self,key) <= self.plan['limits']['max_write_bytes' if write else 'max_read_bytes'], 'Assembly byte allowance exhausted')
    def read_file(self,path,identity,maximum=MAX_METADATA):
        self.charge(identity['bytes']); return _checked(path,identity,maximum)
    def read_unknown(self,path,maximum):
        info=Path(path).lstat()
        require(_resident(path) and 0<info.st_size<=maximum,'Bounded resident metadata required')
        self.charge(info.st_size); return ds._read(path,info.st_size)


def _npz(arrays):
    buffer=io.BytesIO(); np.savez_compressed(buffer,**arrays); raw=buffer.getvalue()
    require(len(raw) <= MAX_PART and sum(x.nbytes for x in arrays.values()) <= MAX_PART, 'Artifact part exceeds bound')
    return raw


def _write(path,raw,budget):
    budget.charge(len(raw),True)
    with Path(path).open('xb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def _publish(root,relative,identity,payloads,budget):
    """Atomic immutable directory publication; interrupted staging is retained."""
    target=root/relative
    if target.exists():
        commit=_read_part(root,relative,budget=budget)
        require(commit['identity']==identity,'Existing artifact belongs to a different assembly')
        require(commit['payloads']=={name:_identity(raw) for name,raw in payloads.items()},
                'Existing artifact differs from recomputed source payloads')
        raw=budget.read_unknown(target/'COMMIT.json',MAX_METADATA)
        return dict(relative=relative,commit_sha256=hashlib.sha256(raw).hexdigest(),**commit)
    stage=root/'staging'/uuid.uuid4().hex; _directory(stage,create=True)
    for name,raw in payloads.items(): _write(stage/name,raw,budget)
    commit=dict(schema=SCHEMA,identity=identity,payloads={name:_identity(raw) for name,raw in payloads.items()})
    raw=_json(commit); _write(stage/'COMMIT.json',raw,budget); _sync_directory(stage)
    _directory(target.parent,create=True)
    require(not target.exists(),'Artifact target appeared during publication')
    os.rename(stage,target); _sync_directory(target.parent)
    return dict(relative=relative,commit_sha256=hashlib.sha256(raw).hexdigest(),**commit)


def _read_part(root,relative,*,expected=None,budget=None,only=None):
    require(isinstance(relative,str) and '..' not in Path(relative).parts and not Path(relative).is_absolute(), 'Unsafe artifact path')
    directory=root/relative; ds._directory(directory)
    raw=budget.read_unknown(directory/'COMMIT.json',MAX_METADATA) if budget else ds._read(directory/'COMMIT.json',MAX_METADATA)
    require(expected is None or hashlib.sha256(raw).hexdigest()==expected,'Artifact commit changed')
    commit=json.loads(raw)
    require(commit['schema']==SCHEMA and set(p.name for p in directory.iterdir())=={'COMMIT.json',*commit['payloads']},'Artifact inventory differs')
    for name,identity in commit['payloads'].items():
        require(isinstance(name,str) and name and Path(name).name==name and name not in ('.','..'),'Unsafe artifact payload')
        if only is not None and name not in only: continue
        if budget: budget.read_file(directory/name,identity,MAX_PART)
        else: _checked(directory/name,identity,MAX_PART)
    return commit


def _month_lookup(plan,budget):
    stores={}; cache=DerivedCache('.')
    for month,spec in plan['months'].items():
        directory=Path(spec['entry_path'])
        for name,identity in spec['files'].items(): budget.read_file(directory/name,identity,MAX_PART)
        budget.charge(sum(v['bytes'] for v in spec['files'].values()))
        arrays,metadata=cache._load(directory,'usgs_month',spec['identity'],spec['cache_key'])
        unpack_month(arrays,metadata)  # Original dictionary/mask/axis semantic validator.
        require(metadata['month']==month and metadata.get('development_cutoff_exclusive_epoch') is None,'Native development monthly source required')
        stores[month]=(arrays,metadata)
    return stores


def _row(stores,gauge,stamp):
    month=datetime.fromtimestamp(int(stamp),timezone.utc).strftime('%Y-%m')
    require(month in stores,'Missing exact-month provenance; never interpret transport absence as a source gap')
    arrays,meta=stores[month]; index=(int(stamp)-int(arrays['times'][0]))//3600
    require(0 <= index < len(arrays['times']) and int(arrays['times'][index])==int(stamp),'Provenance exact-hour axis differs')
    present=bool(arrays['row_present'][gauge,index]); valid=bool(arrays['numeric_valid'][gauge,index])
    flags=tuple(meta['qualities'][int(arrays['quality_index'][gauge,index])]) if present else ()
    source=meta['sources'][int(arrays['source_index'][gauge,index])] if present else None
    reason='MISSING_EXACT_HOUR' if not present else 'INVALID_DISCHARGE' if not valid else 'QUALITY_NOT_ADMITTED' if not quality_admitter(PRIMARY_APPROVED)(flags) else None
    value=float(arrays['values'][gauge,index]) if valid else np.nan
    return value,reason,flags,source,int(stamp) if present else None


def _join(arrays, rows, stores):
    reasons={'history':Counter(),'target':Counter()}
    for row in rows:
        gauge=int(arrays['site_id'][row]); origin=int(arrays['origin_time'][row])
        for field,times,values,masks in (
            ('history',arrays['history_time'][row],arrays['history'][row,:,0],arrays['history_mask'][row,:,0]),
            ('target',arrays['target_time'][row],arrays['target'][row],arrays['target_mask'][row])):
            for slot,stamp in enumerate(times):
                value,reason,_,_,_= _row(stores,gauge,stamp)
                require(field!='history' or int(stamp)+7200 <= origin,'History source exceeds simulated availability cutoff')
                require(bool(masks[slot])==(reason is None),'Corpus/source quality or numeric mask mismatch')
                require((reason is not None and np.isnan(values[slot])) or (reason is None and values[slot]==value),'Corpus/source discharge mismatch')
                reasons[field][reason or 'ADMITTED']+=1
    return {key:dict(value) for key,value in reasons.items()}


def _compact_month(arrays,metadata):
    return dict(month=metadata['month'],times=arrays['times'].tolist(),qualities=metadata['qualities'],sources=metadata['sources'],
                **{key:arrays[key].tolist() for key in ('row_present','numeric_valid','quality_index','source_index')})


def original_gap_shape(approved_mask):
    """Unnormalized, method-independent shape census; no augmentation adoption.

    Bands are inclusive *start indices*, matching pilot_inputs artificial_mask.
    A two-slot run beginning at55 is therefore an early-band shape even though
    its second slot lies at56. Carry/ridge variants are the same missing shape.
    """
    mask=np.asarray(approved_mask)
    require(mask.shape==(168,) and mask.dtype==np.dtype(bool),'Exact original L168 boolean mask required')
    missing=np.flatnonzero(~mask).tolist()
    if not missing: return 'clean'
    if missing[0]==0: return 'leading'
    if len(missing)>2: return 'long_or_many'
    if missing==[167]: return 'trailing_1'
    if missing==[166,167]: return 'trailing_2'
    bands={'early':(1,55),'middle':(56,110),'recent':(111,165)}
    band=next((name for name,(low,high) in bands.items() if low<=missing[0]<=high),None)
    if band is not None:
        if len(missing)==1: return 'interior_1_'+band
        if missing[1]==missing[0]+1: return 'interior_2_'+band
        if missing[1]==167: return 'mixed_'+band
    return 'nonrepresentable'


def _ridge_series(stores,gauge,roles,refit):
    first,end=roles['fit']; times=np.arange(first,end,3600,dtype=np.int64)
    require(0 < len(times) <= 100000 and end <= refit <= ASSESSMENT_START,'Bounded gradient-only ridge source window required')
    native=[_row(stores,gauge,int(t)) for t in times]
    approved=np.array([row[1] is None for row in native],bool)
    stops=np.array([i for i in range(7,len(times)) if approved[i-7:i+1].all() and int(times[i])+7200<=refit],np.int64)
    arrays=dict(observation_times=times,q_values=np.array([row[0] for row in native],np.float64),approved_mask=approved,eligible_pair_stop_indices=stops)
    metadata=dict(gauge_id=GAUGES[gauge],gradient_start_epoch=first,gradient_end_epoch=end,refit_epoch=refit,
        source_observation_times=[row[4] for row in native],source_flags=[list(row[2]) for row in native],source_ids=[row[3] for row in native],
        eligible_pairs=len(stops),pair_rule='SIX_ORIGINAL_CONSECUTIVE_LAGS_TWO_DIRECT_OUTPUTS_ALL_APPROVED_LAST_PLUS_2H_LE_REFIT',ridge_fits=0)
    return arrays,metadata


def execute(plan, *, output_directory, check=lambda:None):
    """Prepare only the declared resident sources; no automatic recovery or fitting."""
    require(plan['schema']==SCHEMA and plan['plan_sha256']==digest({k:v for k,v in plan.items() if k!='plan_sha256'})
            and plan['implementation_sha256']==code_identity(),'Assembly plan or implementation changed')
    require(plan['status']=='READY_LOCAL_ONLY' and not plan['unavailable'],'Resident provenance is incomplete')
    root=Path(output_directory).absolute(); source=Path(plan['corpus_root'])
    inputs=[source,Path(plan['integrity_receipt_path']),*(Path(x['entry_path']) for x in plan['months'].values())]
    require(all(root!=p and p not in root.parents and root not in p.parents for p in inputs),
            'Output must be separate from immutable inputs')
    budget=_Budget(plan,check); roles=_roles(plan['role_time_bounds']); refit=ds._hour(plan['refit_at_utc'])
    budget.read_file(plan['integrity_receipt_path'],plan['integrity_identity'])
    plan_raw=ds._read(source/'PLAN.json',4*1024**2); budget.charge(len(plan_raw))
    require(hashlib.sha256(plan_raw).hexdigest()==plan['source_plan_file_sha256'],'Source plan file changed')
    _directory(root,create=True)
    require({p.name for p in root.iterdir()} <= {'PLAN.json','staging','provenance','parts','ridge','MANIFEST.json'},
            'Refusing unrelated nonempty output directory')
    _directory(root/'staging',create=True)
    marker=root/'PLAN.json'; raw=_json(plan)
    if marker.exists(): require(ds._read(marker,MAX_METADATA)==raw,'Output already belongs to another plan')
    else: _write(marker,raw,budget); _sync_directory(root)
    stores=_month_lookup(plan,budget); month_parts=[]
    for month,(arrays,metadata) in stores.items():
        identity=dict(plan_sha256=plan['plan_sha256'],month=month,source_commit_sha256=plan['months'][month]['commit_sha256'])
        month_parts.append(_publish(root,'provenance/'+month,identity,{'lookup.json':_json(_compact_month(arrays,metadata))},budget))
    output_roles=(*ROLES,'coverage_outer')
    parts=[]; cases={g:{role:[] for role in output_roles} for g in GAUGES}; reasons={'history':Counter(),'target':Counter()}
    fit_gap_counts={g:Counter() for g in GAUGES}
    for position,shard in enumerate(plan['shards']):
        budget.check(); directory=source/'chunks'/shard['relative']
        for name,identity in shard['files'].items():
            require(_resident(directory/name) and (directory/name).lstat().st_size==identity['bytes'],
                    'Planned source shard size/residency changed before numeric read')
        budget.charge(sum(v['bytes'] for v in shard['files'].values()))
        arrays,summary,commit_sha=ds._load_shard(source,shard['unit'],shard['chunk'],plan['source_plan_sha256'],expected_commit_sha=shard['commit_sha256'])
        require(summary['month_keys']==shard['month_keys'],'Source month references changed')
        assigned=[_selected_role(int(t),roles,refit) for t in arrays['origin_time']]
        rows=[i for i,role in enumerate(assigned) if role is not None]
        joined=_join(arrays,rows,stores)
        for key,counts in joined.items(): reasons[key].update(counts)
        for i in rows:
            if assigned[i]=='fit':
                fit_gap_counts[GAUGES[int(arrays['site_id'][i])]][original_gap_shape(arrays['history_mask'][i,:,0])]+=1
        for gauge in GAUGES:
            for role in output_roles:
                requested_role='outer' if role=='coverage_outer' else role
                chosen=[i for i in rows if assigned[i]==requested_role and int(arrays['site_id'][i])==GAUGES.index(gauge)
                        and (role=='coverage_outer' or arrays['admission_mask'][i])]
                if not chosen: continue
                ids=arrays['case_id'][chosen].tolist(); cases[gauge][role].extend(ids)
                identity=dict(plan_sha256=plan['plan_sha256'],source_chunk=shard['relative'],source_commit_sha256=commit_sha,
                              gauge_id=gauge,role=role,cases=len(chosen),case_ids_sha256=digest(ids),
                              first_case_id=ids[0],last_case_id=ids[-1],source_rows=chosen,month_keys=shard['month_keys'],
                              role_window=plan['role_time_bounds'][requested_role],refit_epoch=refit,
                              cohort_mode='all_outer_candidates' if role=='coverage_outer' else 'common_l168_complete')
                inputs={name:arrays[name][chosen] for name in INPUT_KEYS}
                labels={name:arrays[name][chosen] for name in LABEL_KEYS}
                part=_publish(root,f'parts/{position:04d}-{gauge}-{role}',identity,{'inputs.npz':_npz(inputs),'labels.npz':_npz(labels)},budget)
                parts.append(part)
    for gauge in GAUGES:
        for role in ROLES:
            ids=cases[gauge][role]
            require(ids==sorted(set(ids)),'Duplicate/unordered role cases')
            require(dict(cases=len(ids),case_ids_sha256=digest(ids))==plan['expected_roles'][gauge][role],'Actual common-cohort cases differ from expected role membership')
        start,end=roles['outer']
        expected=[gauge+':'+iso(t) for t in range(start,end,3600) if t+18*3600<end]
        require(cases[gauge]['coverage_outer']==expected,'Outer coverage calendar is incomplete or unordered')
    ridge=[]
    for gauge_index,gauge in enumerate(GAUGES):
        arrays,metadata=_ridge_series(stores,gauge_index,roles,refit)
        identity=dict(plan_sha256=plan['plan_sha256'],gauge_id=gauge,role='gradient_only_unique_hours',
                      cases=len(arrays['observation_times']),eligible_pairs=metadata['eligible_pairs'],
                      gradient_start_epoch=roles['fit'][0],gradient_end_epoch=roles['fit'][1],refit_epoch=refit)
        ridge.append(_publish(root,'ridge/'+gauge,identity,{'series.npz':_npz(arrays),'metadata.json':_json(metadata)},budget))
    manifest=dict(schema=SCHEMA,status='ASSEMBLED_VERIFIED_LOCAL_INPUTS_NO_FITS',plan_sha256=plan['plan_sha256'],
        snapshot_sha256=plan['snapshot_sha256'],source_plan_sha256=plan['source_plan_sha256'],implementation_sha256=plan['implementation_sha256'],
        role_time_bounds=plan['role_time_bounds'],refit_at_utc=plan['refit_at_utc'],quality_policy=PRIMARY_APPROVED,
        cohort_mode=plan['cohort_mode'],roles=plan['expected_roles'],parts=parts,provenance=month_parts,ridge=ridge,
        coverage_outer={g:dict(cases=len(cases[g]['coverage_outer']),case_ids_sha256=digest(cases[g]['coverage_outer']),
                              cohort_mode='all_outer_candidates') for g in GAUGES},
        original_gradient_gap_shape_census={g:dict(candidate_windows=sum(fit_gap_counts[g].values()),counts=dict(fit_gap_counts[g]),
            denominator='ALL_ORIGINAL_FIT_ROLE_CANDIDATES_AFTER_EXPLICIT_TARGET_AND_REFIT_CLOCK_GATES',
            includes_inner_or_outer=False,uses_target_values=False,normalized=False,recipe_adopted=False,
            band_rule='INCLUSIVE_INTERIOR_START_INDICES_EARLY_1_55_MIDDLE_56_110_RECENT_111_165') for g in GAUGES},
        join_slot_counts={key:dict(value) for key,value in reasons.items()},
        execution=dict(scaler_fits=0,ridge_fits=0,optimizer_updates=0,network_requests=0,protected_access=False),
        historical_delivery_vintage_known=False,scientific_fit_authorization=False)
    manifest['content_sha256']=digest(manifest); raw=_json(manifest); target=root/'MANIFEST.json'
    if target.exists(): require(ds._read(target,MAX_METADATA)==raw,'Existing completed manifest differs')
    else: _write(target,raw,budget); _sync_directory(root)
    return dict(manifest=manifest,manifest_sha256=hashlib.sha256(raw).hexdigest(),read_bytes=budget.read,write_bytes=budget.written)


def _manifest(directory,expected):
    root=Path(directory).absolute(); raw=ds._read(root/'MANIFEST.json',MAX_METADATA)
    require(ds._hex(expected) and hashlib.sha256(raw).hexdigest()==expected,'Expected assembly manifest SHA differs')
    value=json.loads(raw)
    require(value['schema']==SCHEMA and value['status']=='ASSEMBLED_VERIFIED_LOCAL_INPUTS_NO_FITS'
            and value['content_sha256']==digest({k:v for k,v in value.items() if k!='content_sha256'}),'Assembly manifest commitment differs')
    require(value['implementation_sha256']==code_identity() and value['quality_policy']==PRIMARY_APPROVED,'Assembly reader/quality identity differs')
    _roles(value['role_time_bounds'])
    return root,value


def _load_npz(root,part,name):
    commit=_read_part(root,part['relative'],expected=part['commit_sha256'],only={name})
    require(commit['identity']==part['identity'] and commit['payloads']==part['payloads'],'Part manifest binding differs')
    raw=_checked(root/part['relative']/name,part['payloads'][name],MAX_PART)
    result={}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries=archive.infolist()
        require(len(entries)<=20 and len({e.filename for e in entries})==len(entries)
                and sum(e.file_size for e in entries)<=MAX_PART,'Prepared NPZ expansion/inventory exceeds bound')
        # Decode and gate coordinates before opening a numeric feature/label member.
        # allow_pickle=False plus the decompressed-size bound prevents object loading.
        with np.load(io.BytesIO(raw),allow_pickle=False) as data:
            n=part['identity']['cases']; gauge=part['identity']['gauge_id']
            require(type(n) is int and 0<n<=100000 and gauge in GAUGES,'Prepared case/gauge bound differs')
            if name=='series.npz':
                require(set(data.files)=={'observation_times','q_values','approved_mask','eligible_pair_stop_indices'},'Ridge tensor inventory differs')
                times=data['observation_times']; identity=part['identity']
                first,end,refit=(identity[key] for key in ('gradient_start_epoch','gradient_end_epoch','refit_epoch'))
                require(first<end<=refit<=ASSESSMENT_START and times.dtype==np.dtype('int64')
                        and np.array_equal(times,np.arange(first,end,3600,dtype=np.int64)) and len(times)==n,
                        'Prepared ridge clocks cross gradient/protected bounds')
            else:
                expected=INPUT_KEYS if name=='inputs.npz' else LABEL_KEYS
                require(name in ('inputs.npz','labels.npz') and set(data.files)==set(expected),'Prepared tensor inventory differs')
                clocks={key:data[key] for key in ('origin_time','target_time','site_id','case_id')}
                origins,future=clocks['origin_time'],clocks['target_time']
                window=part['identity']['role_window']; start,end=ds._hour(window['start_utc']),ds._hour(window['end_utc'])
                require(origins.dtype==np.dtype('int64') and origins.shape==(n,) and np.all(np.diff(origins)>0)
                        and np.all(origins%3600==0) and np.all(origins>=start) and end<=ASSESSMENT_START
                        and future.dtype==np.dtype('int64') and np.array_equal(future,origins[:,None]+np.arange(1,19)*3600)
                        and np.all(future<end),'Prepared target coordinates cross explicit role/protected bounds')
                require(clocks['site_id'].dtype==np.dtype('int64') and clocks['site_id'].shape==(n,)
                        and np.all(clocks['site_id']==GAUGES.index(gauge)) and clocks['case_id'].shape==(n,)
                        and clocks['case_id'].tolist()==[gauge+':'+iso(int(t)) for t in origins]
                        and digest(clocks['case_id'].tolist())==part['identity']['case_ids_sha256'],
                        'Prepared gauge/case identity differs')
                if part['identity']['role'] not in ('outer','coverage_outer'):
                    require(np.all(future[:,-1]+7200<=part['identity']['refit_epoch']),'Prepared labels unavailable by simulated refit')
                if name=='inputs.npz':
                    history=data['history_time']
                    require(history.dtype==np.dtype('int64') and np.array_equal(history,origins[:,None]-np.arange(169,1,-1)*3600),
                            'Prepared exact-hour history differs')
            for key in data.files:
                require(key.isidentifier(),'Invalid prepared tensor key'); result[key]=data[key]
                require(result[key].dtype.kind in 'fibU','Unsupported prepared tensor dtype')
            if name=='series.npz':
                mask=result['approved_mask']; values=result['q_values']; stops=result['eligible_pair_stop_indices']
                require(mask.dtype==np.dtype(bool) and mask.shape==values.shape==(n,) and values.dtype==np.dtype('float64')
                        and np.isfinite(values[mask]).all() and (values[mask]>=0).all(),'Prepared ridge values/masks differ')
                expected_stops=np.array([i for i in range(7,n) if mask[i-7:i+1].all() and times[i]+7200<=refit],np.int64)
                require(stops.dtype==np.dtype('int64') and np.array_equal(stops,expected_stops)
                        and len(stops)==identity['eligible_pairs'],'Prepared ridge pair census differs')
            else:
                shapes=ds._shapes(n,168)
                for field in (('history','weather','state') if name=='inputs.npz' else ('target',)):
                    values,mask=result[field],result[field+'_mask']
                    require(values.shape==mask.shape==shapes[field] and values.dtype==np.dtype('float64')
                            and mask.dtype==np.dtype(bool) and np.isfinite(values[mask]).all() and np.isnan(values[~mask]).all()
                            and (part['identity']['role']=='coverage_outer' or mask.all()),
                            'Prepared numeric schema/masks differ')
                if name=='inputs.npz':
                    mask=result['history_mask'][...,0]
                    require(np.array_equal(mask,result['history_mask'][...,1])
                            and np.array_equal(result['history'][...,1][mask],((origins[:,None]-result['history_time'])/3600)[mask])
                            and (result['history'][...,0][mask]>=0).all(),'Prepared history Q/age differs')
                else: require((result['target'][result['target_mask']]>=0).all(),'Prepared label discharge negative')
    return result


def _lookups(root,manifest):
    result={}
    for part in manifest['provenance']:
        commit=_read_part(root,part['relative'],expected=part['commit_sha256'],only={'lookup.json'})
        require(commit['identity']==part['identity'],'Provenance manifest binding differs')
        data=json.loads(_checked(root/part['relative']/'lookup.json',part['payloads']['lookup.json'],MAX_PART))
        require(data['month']==part['identity']['month'] and data['month'] not in result,'Provenance month duplicated')
        # Store original immutable primitives once; repeated cases share tuples/strings.
        data['qualities']=tuple(tuple(flags) for flags in data['qualities']); data['sources']=tuple(data['sources'])
        result[data['month']]=data
    return result


def _history_provenance(arrays,lookups):
    columns={key:[] for key in ('observation_times','quality_flags','source_ids')}
    for gauge,times in zip(arrays['site_id'],arrays['history_time']):
        stamps,flags,sources=[],[],[]
        for t in times:
            month=datetime.fromtimestamp(int(t),timezone.utc).strftime('%Y-%m'); data=lookups[month]
            index=(int(t)-data['times'][0])//3600; require(data['times'][index]==int(t),'Prepared history source clock differs')
            present=data['row_present'][int(gauge)][index]
            stamps.append(int(t) if present else None)
            flags.append(data['qualities'][data['quality_index'][int(gauge)][index]] if present else ())
            sources.append(data['sources'][data['source_index'][int(gauge)][index]] if present else None)
        for name,row in zip(columns,(stamps,flags,sources)): columns[name].append(tuple(row))
    return MappingProxyType({key:tuple(value) for key,value in columns.items()})


def _tag(arrays,manifest,gauge,roles):
    identity=dict(assembly_manifest_content_sha256=manifest['content_sha256'],gauge_id=gauge,roles=list(roles),
                  case_ids_sha256=digest(arrays['case_id'].tolist()))
    cohort=digest(dict(snapshot_sha256=manifest['snapshot_sha256'],gauge_id=gauge,roles=list(roles),case_ids_sha256=identity['case_ids_sha256']))
    arrays['feature_schema']=np.array(ds.FULL_FEATURE_SCHEMA); arrays['cohort_admission_digest']=np.array(cohort)
    for value in arrays.values(): value.setflags(write=False)
    return digest(identity),cohort


def load_training_bundle(directory, *, expected_manifest_sha256, gauge_id, max_bytes=MAX_BYTES):
    """Only fit+inner arrays/labels, plus original provenance; no statistics fit."""
    require(gauge_id in GAUGES and type(max_bytes) is int and 0<max_bytes<=MAX_BYTES,'Bounded gauge bundle required')
    root,manifest=_manifest(directory,expected_manifest_sha256); selected=[p for p in manifest['parts'] if p['identity']['gauge_id']==gauge_id and p['identity']['role'] in ('fit','inner_stop')]
    n=sum(p['identity']['cases'] for p in selected)
    # Dense arrays plus transient concatenation and three tuple provenance columns.
    require(n>0 and n*(168*2*9*2+18*4*9*2+168*3*16+4096)<=max_bytes,'Training bundle allocation bound exceeded')
    batches=[]; indices={'fit':[],'inner_stop':[]}; offset=0
    for part in selected:
        inputs=_load_npz(root,part,'inputs.npz'); labels=_load_npz(root,part,'labels.npz')
        require(all(np.array_equal(inputs[k],labels[k]) for k in ('origin_time','target_time','site_id','case_id')),'Separate label/input coordinates differ')
        batch={**inputs,'target':labels['target'],'target_mask':labels['target_mask']}; batches.append(batch)
        count=len(batch['case_id']); indices[part['identity']['role']].extend(range(offset,offset+count)); offset+=count
    arrays={name:np.concatenate([batch[name] for batch in batches]) for name in batches[0]}
    require(np.all(np.diff(arrays['origin_time'])>0),'Training bundle chronology differs')
    source,cohort=_tag(arrays,manifest,gauge_id,('fit','inner_stop'))
    provenance=_history_provenance(arrays,_lookups(root,manifest))
    return dict(arrays=MappingProxyType(arrays),history_provenance=provenance,train_indices=tuple(indices['fit']),
                inner_indices=tuple(indices['inner_stop']),source_identity=source,cohort_admission_digest=cohort,
                role_time_bounds=manifest['role_time_bounds'],refit_at_utc=manifest['refit_at_utc'])


def _iter_inputs(directory, *, expected_manifest_sha256, gauge_id, batch_size, role):
    """Never opens labels.npz. H18 clocks remain available as forecast coordinates."""
    require(gauge_id in GAUGES and type(batch_size) is int and 1<=batch_size<=64,'Bounded outer input batches required')
    root,manifest=_manifest(directory,expected_manifest_sha256); lookups=_lookups(root,manifest)
    for part in manifest['parts']:
        if part['identity']['gauge_id']!=gauge_id or part['identity']['role']!=role: continue
        arrays=_load_npz(root,part,'inputs.npz')
        for start in range(0,len(arrays['case_id']),batch_size):
            batch={name:value[start:start+batch_size].copy() for name,value in arrays.items()}
            source,cohort=_tag(batch,manifest,gauge_id,(role,))
            batch['forecast_time']=batch.pop('target_time'); batch['inference_source_identity']=np.array(source)
            yield dict(arrays=MappingProxyType(batch),history_provenance=_history_provenance(batch,lookups),
                       source_identity=source,cohort_admission_digest=cohort)


def iter_outer_inputs(directory, *, expected_manifest_sha256, gauge_id, batch_size=64):
    return _iter_inputs(directory,expected_manifest_sha256=expected_manifest_sha256,gauge_id=gauge_id,batch_size=batch_size,role='outer')


def iter_coverage_inputs(directory, *, expected_manifest_sha256, gauge_id, batch_size=64):
    """All outer candidate origins with original masks; no filling or admission."""
    return _iter_inputs(directory,expected_manifest_sha256=expected_manifest_sha256,gauge_id=gauge_id,batch_size=batch_size,role='coverage_outer')


def _load_labels(directory, *, expected_manifest_sha256, gauge_id, case_ids, role):
    """Explicit bounded retrospective label access, separate from input streaming."""
    require(gauge_id in GAUGES and isinstance(case_ids,(tuple,list)) and 0<len(case_ids)<=64
            and list(case_ids)==sorted(set(case_ids)),'Explicit ordered bounded label case IDs required')
    root,manifest=_manifest(directory,expected_manifest_sha256); found={}
    for part in manifest['parts']:
        if part['identity']['gauge_id']!=gauge_id or part['identity']['role']!=role: continue
        # Case IDs encode origin, so avoid opening unrelated label shards.
        if not any(part['identity']['first_case_id'] <= case <= part['identity']['last_case_id'] for case in case_ids): continue
        labels=_load_npz(root,part,'labels.npz')
        for i,case in enumerate(labels['case_id'].tolist()):
            if case in case_ids:
                require(case not in found,'Duplicate prepared label case'); found[case]={k:v[i] for k,v in labels.items()}
    require(set(found)==set(case_ids),'Requested labels absent from declared outer cohort')
    result={key:np.array([found[case][key] for case in case_ids]) for key in LABEL_KEYS}
    for value in result.values(): value.setflags(write=False)
    return MappingProxyType(result)


def load_outer_labels(directory, *, expected_manifest_sha256, gauge_id, case_ids):
    return _load_labels(directory,expected_manifest_sha256=expected_manifest_sha256,gauge_id=gauge_id,case_ids=case_ids,role='outer')


def load_coverage_labels(directory, *, expected_manifest_sha256, gauge_id, case_ids):
    return _load_labels(directory,expected_manifest_sha256=expected_manifest_sha256,gauge_id=gauge_id,case_ids=case_ids,role='coverage_outer')


def load_ridge_series(directory, *, expected_manifest_sha256, gauge_id):
    """Return only unique original gradient hours; does not fit ridge parameters."""
    require(gauge_id in GAUGES,'Known ridge gauge required'); root,manifest=_manifest(directory,expected_manifest_sha256)
    part=next(p for p in manifest['ridge'] if p['identity']['gauge_id']==gauge_id)
    arrays=_load_npz(root,part,'series.npz')
    metadata=json.loads(_checked(root/part['relative']/'metadata.json',part['payloads']['metadata.json'],MAX_PART))
    return dict(arrays=arrays,metadata=metadata,source_identity=digest(dict(assembly=manifest['content_sha256'],ridge_commit=part['commit_sha256'])))
