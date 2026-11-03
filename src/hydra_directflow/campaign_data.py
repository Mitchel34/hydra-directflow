"""Bounded primary-campaign arrays from the sealed resident development corpus.

Only exact jobs from the pinned 180-fit roster are supported. The original
complete integrity receipt is reused; no full inspect_corpus pass, source-cache
join, hydration, transform fit or model import occurs. Fit bundles omit outer
and unassigned rows. Outer input loading authenticates the compressed shard
bytes but never decodes the target-value member or returns retrospective labels.
"""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
from types import MappingProxyType
import zipfile

import numpy as np

from . import corpus_dataset as ds
from .ablations import q_only_view, Q_HISTORY_FEATURES
from .contract import HORIZONS_HOURS
from .local_data import GAUGES, iso
from .quality import PRIMARY_APPROVED
from .reference import ASSESSMENT_START, digest, require
from .trainer_bundle import _roles, _refit_policy, _geometry, _array_digest

ROOT=Path(__file__).resolve().parents[2]
INTEGRITY_PATH=ROOT/'reports/development-corpus-residency-integrity-v1.json'
ROSTER_PATH=ROOT/'reports/local-seed-sequence-fastest-v3.json'
INTEGRITY_SHA256='155f563238b8bcbdb549dc761775c2b24d4171d3f1f7785acba23b4e1c956d94'
ROSTER_SHA256='f45aa1afe114db80b194288ecb06c7757ebe3928de3702fac9ff1114238310e9'
SOURCE_PLAN='0dad073824984fd7eeba88e4c313d77b4015dcccb94e8eb27297fcf7c6b39b0f'
SNAPSHOT='b41630b7df9a216874eb4ba38a10176d4b72092bd6f1ce2ad4c4f369db56948c'
MAX_BYTES=1024**3
MAX_RSS=4*1024**3
ROLE_NAMES={'fit':'gradient_fit','inner_stop':'inner_stopping','outer':'outer_scoring'}
COORDS=('origin_time','history_time','target_time','site_id','case_id')
INPUT_FIELDS=('history','weather','state')


def implementation_identity():
    names=('campaign_data.py','corpus_dataset.py','trainer_bundle.py','ablations.py',
           'quality.py','contract.py','reference.py','local_data.py')
    return {name:hashlib.sha256(ds._read(Path(__file__).parent/name,512*1024)).hexdigest() for name in names}


def _pinned(path,expected,maximum):
    raw=ds._read(path,maximum)
    require(ds._hex(expected) and hashlib.sha256(raw).hexdigest()==expected,'Pinned campaign metadata SHA differs')
    return json.loads(raw),len(raw)


def _authority(corpus_root,job_spec,expected_integrity_hash):
    require(expected_integrity_hash==INTEGRITY_SHA256,'Exact published complete integrity receipt required')
    roster,roster_bytes=_pinned(ROSTER_PATH,ROSTER_SHA256,4*1024**2)
    require(roster['schema']=='hydra-local-seed-sequence/v3' and roster['fits']==len(roster['jobs'])==180
            and roster['source_plan_sha256']==SOURCE_PLAN and roster['snapshot_sha256']==SNAPSHOT
            and roster['protected_assessment_access'] is False,'Primary roster source/scope differs')
    require(isinstance(job_spec,dict) and isinstance(job_spec.get('job_id'),str),'Exact roster job mapping required')
    matches=[item for item in roster['jobs'] if item['job_id']==job_spec['job_id']]
    require(len(matches)==1 and matches[0]==job_spec,'Job differs from its immutable roster entry')
    require(job_spec['gauge_id'] in GAUGES and job_spec['fold'] in ('cv_1','cv_2','cv_3','final_development')
            and (job_spec['input_arm'],job_spec['history_hours']) in (('full',168),('q_only',168),('full',24))
            and (job_spec['history_hours']!=24 or job_spec['fold']=='final_development'),'Unsupported primary feature/history/fold')
    windows=roster['role_windows'][job_spec['fold']]
    roles={short:dict(start_utc=windows[long]['start_utc'],end_utc=windows[long]['end_exclusive_utc']) for short,long in ROLE_NAMES.items()}
    numeric_roles=_roles(roles); refit=windows['inner_stopping']['refit_at_utc']
    _refit_policy(dict(refit_at_utc=refit,simulated_label_delay_hours=2),numeric_roles)
    require(all(windows[long]['simulated_label_delay_hours']==2 for long in ROLE_NAMES.values())
            and windows['gradient_fit']['refit_at_utc']==refit and windows['outer_scoring']['refit_at_utc'] is None,
            'Explicit two-hour fit/inner label availability and separate outer role required')
    require(set(job_spec['actual_common_cases'])==set(ROLE_NAMES.values())
            and set(job_spec['role_case_ids_sha256'])==set(ROLE_NAMES.values()),'Exact role count/hash inventory required')
    for long in ROLE_NAMES.values():
        require(type(job_spec['actual_common_cases'][long]) is int and 0<job_spec['actual_common_cases'][long]<=100000
                and ds._hex(job_spec['role_case_ids_sha256'][long]),'Nonempty bounded role count/hash required')
    receipt,receipt_bytes=_pinned(INTEGRITY_PATH,expected_integrity_hash,4*1024**2)
    root=Path(corpus_root).absolute(); ds._directory(root)
    plan,descriptors,plan_file_sha=ds._plan(root,SOURCE_PLAN)
    require(receipt['status']=='COMPLETE_COMMITTED_CORPUS_SNAPSHOT' and receipt['missing_chunks']==[]
            and receipt['stored_history_hours']==plan['history_hours']==168 and receipt['plan_sha256']==SOURCE_PLAN
            and receipt['plan_file_sha256']==plan_file_sha and receipt['snapshot_sha256']==SNAPSHOT
            and receipt['numerical_payloads_checked'] is True and receipt['protected_assessment_access'] is False,
            'Complete pinned development source receipt required')
    observed=receipt['observed_commits']
    require(len(observed)==len(descriptors) and len({row['chunk'] for row in observed})==len(observed)
            and all(ds._hex(row['commit_sha256']) for row in observed)
            and [row['chunk'] for row in observed]==[unit['id']+'/'+f"{chunk['index']:03d}" for unit,chunk in descriptors],
            'Snapshot commitment inventory/order differs')
    require(digest(dict(plan_sha256=SOURCE_PLAN,plan_file_sha256=plan_file_sha,observed=observed,missing=[]))==SNAPSHOT,
            'Complete snapshot digest differs')
    return dict(root=root,job=job_spec,roles=roles,numeric_roles=numeric_roles,refit_at_utc=refit,
        refit_epoch=ds._hour(refit),descriptors=[(unit,chunk,row) for (unit,chunk),row in zip(descriptors,observed)],
        plan_file_sha256=plan_file_sha,metadata_bytes=roster_bytes+receipt_bytes+(root/'PLAN.json').stat().st_size)


def _role(origin,authority):
    for name,(start,end) in authority['numeric_roles'].items():
        if start<=origin and origin+18*3600<end:
            if name=='outer' or origin+20*3600<=authority['refit_epoch']: return name
    return None


class _Guard:
    def __init__(self,max_rss,max_read,check):
        import psutil
        self.process=psutil.Process(); self.max_rss=max_rss; self.max_read=max_read; self.callback=check
        self.peak=0; self.read_bytes=0; self.shards=0
    def check(self,additional_bytes=0):
        self.callback(); self.peak=max(self.peak,self.process.memory_info().rss)
        require(self.peak+additional_bytes<=self.max_rss,'Campaign loader RSS/headroom limit reached')
    def read(self,path,maximum,expected=None,expected_bytes=None):
        self.check(); info=Path(path).lstat()
        require(0<info.st_size<=maximum and (expected_bytes is None or info.st_size==expected_bytes),
                'Campaign source byte size differs before body read')
        require(self.read_bytes+info.st_size<=self.max_read,'Campaign read allowance exceeded before body read')
        self.read_bytes+=info.st_size; raw=ds._read(path,info.st_size)
        require(expected is None or hashlib.sha256(raw).hexdigest()==expected,'Campaign source payload SHA differs')
        return raw


def _read_shard(authority,unit,chunk,observed,guard,fields):
    """Authenticate full archived bytes, decode coordinates/masks then named values.

    Only the explicitly named numeric members are decoded. The previously pinned
    full integrity audit covers values omitted by the current feature/role view.
    """
    directory=authority['root']/'chunks'/observed['chunk']; ds._directory(directory)
    require({p.name for p in directory.iterdir()}=={'COMMIT.json','case-index.json','model.npz'},'Unexpected source shard members')
    commit=json.loads(guard.read(directory/'COMMIT.json',65536,observed['commit_sha256']))
    require(commit['schema']==ds.SCHEMA and commit['plan_sha256']==SOURCE_PLAN and commit['unit_id']==unit['id']
            and commit['chunk_index']==chunk['index'] and set(commit['payloads'])=={'case-index.json','model.npz'}
            and set(commit['arrays'])==ds.ALL_KEYS,'Source shard schema/plan differs')
    bodies={}
    for name,maximum in (('case-index.json',256*1024),('model.npz',ds.MAX_SHARD_BYTES)):
        item=commit['payloads'][name]
        require(ds._hex(item['sha256']) and type(item['bytes']) is int,'Explicit payload identity required')
        bodies[name]=guard.read(directory/name,maximum,item['sha256'],item['bytes'])
    summary=json.loads(bodies['case-index.json'])
    require(summary['unit_id']==unit['id'] and summary['chunk_index']==chunk['index']
            and summary['origin_count']==chunk['origin_count'] and summary['case_count']==3*chunk['origin_count']
            and summary['first_origin_utc']==chunk['first_origin_utc'] and summary['last_origin_utc']==chunk['last_origin_utc']
            and summary['input_quality_policy']==summary['target_quality_policy']==PRIMARY_APPROVED
            and summary['calendar_unavailable_cycle_ids']==chunk['calendar_unavailable_cycle_ids']
            and ds._hex(summary['case_source_digest']),'Source summary/quality differs')
    first,last=ds._hour(chunk['first_origin_utc']),ds._hour(chunk['last_origin_utc'])
    require(set(summary['cycle_keys'])==set(chunk['cycle_ids'])
            and set(summary['month_keys'])==set(ds._months(first-169*3600,last+18*3600))
            and all(ds._hex(key) for mapping in (summary['cycle_keys'],summary['month_keys']) for key in mapping.values()),
            'Source lineage references differ')
    n=summary['case_count']; shapes=ds._shapes(n,168); arrays={}
    with zipfile.ZipFile(io.BytesIO(bodies['model.npz'])) as archive:
        entries=archive.infolist()
        require(len(entries)==len(ds.ALL_KEYS) and len({e.filename for e in entries})==len(entries)
                and {e.filename for e in entries}=={name+'.npy' for name in ds.ALL_KEYS}
                and sum(e.file_size for e in entries)<=ds.MAX_SHARD_BYTES,'Source NPZ inventory/expansion differs')
        for name in ds.COORD_KEYS: arrays[name]=ds._decode(archive,commit,name,shapes[name])
        ds._validate_coordinates(arrays,unit,chunk)
        for field in (*INPUT_FIELDS,'target'):
            name=field+'_mask'; arrays[name]=ds._decode(archive,commit,name,shapes[name])
        require(np.array_equal(arrays['history_mask'][...,0],arrays['history_mask'][...,1])
                and np.array_equal(arrays['state_mask'][:,0],arrays['state_mask'][:,1]),'Original feature masks disagree')
        all_masks=np.ones(n,bool)
        for field in (*INPUT_FIELDS,'target'): all_masks &= arrays[field+'_mask'].reshape(n,-1).all(axis=1)
        require(np.array_equal(arrays['eligibility_reason']=='COMPLETE',all_masks)
                and np.array_equal(arrays['admission_mask'],ds._complete(arrays,168))
                and int(arrays['admission_mask'].sum())==summary['admitted_cases']
                and dict(Counter(arrays['eligibility_reason'].tolist()))==summary['eligibility_counts']
                and dict(Counter(arrays['source_calendar_status'].tolist()))==summary['source_calendar_counts'],
                'Common L168 source admission/mask accounting differs')
        for name in fields:
            values=ds._decode(archive,commit,name,shapes[name]); mask=arrays[name+'_mask']
            require(np.isfinite(values[mask]).all() and np.isnan(values[~mask]).all(),'Source value/mask semantics differ')
            if name=='target': require((values[mask]>=0).all(),'Negative admitted target discharge')
            if name=='history':
                hm=mask[...,0]
                require((values[...,0][hm]>=0).all() and np.array_equal(values[...,1][hm],((arrays['origin_time'][:,None]-arrays['history_time'])/3600)[hm]),
                        'Source historical discharge/age differs')
            if name=='state':
                for i in np.flatnonzero(mask[:,0]):
                    cycle=str(arrays['soil_source_cycle'][i]); require(cycle,'Observed state lacks source cycle')
                    age=(int(arrays['origin_time'][i])-ds._cycle_epoch(cycle))/3600
                    require(2<=age<=8 and values[i,1]==age and 0<=values[i,0]<=1,'Source soil value/age differs')
            arrays[name]=values
    guard.shards+=1; guard.check()
    return arrays


def _view(source,rows,job,kind,cohort_placeholder):
    length=job['history_hours']; q_only=job['input_arm']=='q_only'
    result={name:np.array(source[name][rows],copy=True) for name in ('origin_time','target_time','site_id','case_id')}
    if kind!='labels':
        result['history_time']=np.array(source['history_time'][rows,-length:],copy=True)
        result['history']=np.array(source['history'][rows,-length:],copy=True)
        result['history_mask']=np.array(source['history_mask'][rows,-length:],copy=True)
    if kind!='inputs':
        result['target']=np.array(source['target'][rows],copy=True); result['target_mask']=np.array(source['target_mask'][rows],copy=True)
    if kind!='labels' and not q_only:
        for field in ('weather','state'):
            for name in (field,field+'_mask'): result[name]=np.array(source[name][rows],copy=True)
    if q_only and kind=='fit':
        marker=object()
        batch={**result,'weather':marker,'weather_mask':marker,'state':marker,'state_mask':marker,
               'case_ids':tuple(result['case_id'].tolist()),'history_features':Q_HISTORY_FEATURES}
        view=q_only_view(batch,cohort_admission_digest=cohort_placeholder)
        for field in (*INPUT_FIELDS,'target'):
            for name in (field,field+'_mask'): result[name]=getattr(view,name)
    elif q_only and kind=='inputs':
        # The existing label-bearing ablation is parity-tested against this
        # identical known-horizon/empty-state view; no placeholder labels exist.
        n=len(rows); horizons=np.asarray(HORIZONS_HOURS,dtype=np.float64)
        result['weather']=np.broadcast_to(horizons[None,:,None],(n,18,1)).copy()
        result['weather_mask']=np.ones((n,18,1),bool)
        result['state']=np.empty((n,0),np.float64); result['state_mask']=np.empty((n,0),bool)
    return result


def _preflight(authority,guard,wanted):
    job=authority['job']; gauge=GAUGES.index(job['gauge_id'])
    expected={name:dict(cases=job['actual_common_cases'][ROLE_NAMES[name]],case_ids_sha256=job['role_case_ids_sha256'][ROLE_NAMES[name]]) for name in wanted}
    selected=[]; ids={name:[] for name in wanted}; members=[]
    for index,(unit,chunk,observed) in enumerate(authority['descriptors']):
        first,last=ds._hour(chunk['first_origin_utc']),ds._hour(chunk['last_origin_utc'])
        if not any(_role(t,authority) in wanted for t in range(first,last+1,3600)): continue
        source=_read_shard(authority,unit,chunk,observed,guard,())
        rows=[i for i in range(gauge,len(source['origin_time']),3)
              if _role(int(source['origin_time'][i]),authority) in wanted and source['admission_mask'][i]]
        if not rows: continue
        for row in rows: ids[_role(int(source['origin_time'][row]),authority)].append(str(source['case_id'][row]))
        selected.append((index,rows))
        members.append(dict(source_chunk=observed['chunk'],commit_sha256=observed['commit_sha256'],rows=rows))
    for name,values in ids.items():
        require(values==sorted(set(values)) and dict(cases=len(values),case_ids_sha256=digest(values))==expected[name],
                'Campaign exact role membership differs from roster before numeric values')
    cohort=digest(dict(source_plan_sha256=SOURCE_PLAN,snapshot_sha256=SNAPSHOT,gauge_id=job['gauge_id'],fold=job['fold'],
                       roles=list(wanted),members=members))
    ordered_ids=[case for name in wanted for case in ids[name]]
    require(ordered_ids==sorted(set(ordered_ids)),'Campaign role membership is not chronological/disjoint')
    return dict(selected=selected,members=members,cohort_admission_digest=cohort,roles=expected,
                case_ids_sha256=digest(ordered_ids),cases=len(ordered_ids))


def preflight_plan(corpus_root,job_spec,*,roles=('fit','inner_stop'),expected_integrity_hash=INTEGRITY_SHA256,
                   max_bytes=MAX_BYTES,max_rss_bytes=MAX_RSS,max_read_bytes=MAX_BYTES,check=lambda:None):
    """Selected coordinate/mask decoding only; no numeric source value members."""
    require(roles in (('fit','inner_stop'),('outer',)) and callable(check),'Explicit fit/inner or outer preflight required')
    require(all(type(x) is int and 0<x<=MAX_BYTES for x in (max_bytes,max_read_bytes))
            and type(max_rss_bytes) is int and 0<max_rss_bytes<=MAX_RSS,'Bounded preflight resources required')
    guard=_Guard(max_rss_bytes,max_read_bytes,check); guard.check()
    auth=_authority(corpus_root,job_spec,expected_integrity_hash); guard.read_bytes=auth['metadata_bytes']
    require(guard.read_bytes<=max_read_bytes,'Metadata exceeds campaign read allowance')
    count=sum(job_spec['actual_common_cases'][ROLE_NAMES[r]] for r in roles)
    geometry=_geometry(count,job_spec['history_hours'],job_spec['input_arm'])
    estimated=sum(int(np.prod(shape))*dtype.itemsize for shape,dtype in geometry.values())+2048
    require(estimated<=max_bytes,'Campaign array byte allowance exceeded before source values')
    selected=_preflight(auth,guard,roles)
    return dict(schema='hydra-primary-campaign-preflight/v1',job_id=job_spec['job_id'],
        source_plan_sha256=SOURCE_PLAN,snapshot_sha256=SNAPSHOT,roster_sha256=ROSTER_SHA256,
        cohort_admission_digest=selected['cohort_admission_digest'],case_ids_sha256=selected['case_ids_sha256'],
        roles=selected['roles'],case_count=selected['cases'],estimated_array_bytes=estimated,
        source_shards_read=guard.shards,source_bytes_read=guard.read_bytes,peak_rss_bytes=guard.peak,
        numeric_value_members_decoded=[],members=selected['members'],implementation_sha256=implementation_identity())


def _load(corpus_root,job_spec,*,kind,expected_integrity_hash,max_bytes,max_rss_bytes,max_read_bytes,check,
          expected_outer_cohort_digest=None,expected_case_ids_sha256=None):
    require(kind in ('fit','inputs','labels') and callable(check),'Explicit campaign loader operation required')
    require(type(max_bytes) is int and 0<max_bytes<=MAX_BYTES and type(max_rss_bytes) is int and 0<max_rss_bytes<=MAX_RSS
            and type(max_read_bytes) is int and 0<max_read_bytes<=MAX_BYTES,'Explicit campaign resource bounds required')
    guard=_Guard(max_rss_bytes,max_read_bytes,check); guard.check()
    auth=_authority(corpus_root,job_spec,expected_integrity_hash); guard.read_bytes=auth['metadata_bytes']
    require(guard.read_bytes<=max_read_bytes,'Metadata exceeds campaign read allowance')
    job=auth['job']; wanted=('fit','inner_stop') if kind=='fit' else ('outer',)
    expected={name:dict(cases=job['actual_common_cases'][ROLE_NAMES[name]],case_ids_sha256=job['role_case_ids_sha256'][ROLE_NAMES[name]]) for name in wanted}
    count=sum(row['cases'] for row in expected.values()); length=job['history_hours']; arm=job['input_arm']
    geometry=_geometry(count,length,arm)
    if kind=='inputs':
        for name in ('target','target_mask'): geometry.pop(name)
    if kind=='labels': geometry={name:spec for name,spec in geometry.items() if name in ('origin_time','target_time','site_id','case_id','target','target_mask')}
    predicted=sum(int(np.prod(shape))*dtype.itemsize for shape,dtype in geometry.values())+2048
    require(predicted<=max_bytes,'Campaign array byte allowance exceeded before allocation')
    preflight=_preflight(auth,guard,wanted)
    cohort=preflight['cohort_admission_digest']
    if kind=='labels':
        require(cohort==expected_outer_cohort_digest and preflight['case_ids_sha256']==expected_case_ids_sha256,
                'Labels do not match the declared outer inference cohort before target values')
    guard.check(predicted+64*1024**2)
    result={name:np.empty(shape,dtype) for name,(shape,dtype) in geometry.items()}
    fields=('target',) if kind=='labels' else (('history','target') if arm=='q_only' and kind=='fit' else
        ('history',) if arm=='q_only' else (*INPUT_FIELDS,'target') if kind=='fit' else INPUT_FIELDS)
    cursor=0; members=preflight['members']; role_indices={name:[] for name in wanted}
    for descriptor_index,rows in preflight['selected']:
        unit,chunk,observed=auth['descriptors'][descriptor_index]
        source=_read_shard(auth,unit,chunk,observed,guard,fields)
        view=_view(source,rows,job,kind,cohort); n=len(rows)
        require(cursor+n<=count and set(view)==set(result),'Source view count/schema differs from roster')
        for name,value in view.items():
            require(value.dtype==result[name].dtype and value.shape==(n,)+result[name].shape[1:],'Campaign typed view differs: '+name)
            result[name][cursor:cursor+n]=value
        for offset,row in enumerate(rows): role_indices[_role(int(source['origin_time'][row]),auth)].append(cursor+offset)
        cursor+=n; guard.check()
    require(cursor==count and np.all(np.diff(result['origin_time'])>0),'Campaign row count/chronology differs')
    require(result['case_id'].tolist()==[job['gauge_id']+':'+iso(int(t)) for t in result['origin_time']]
            and np.array_equal(result['target_time'],result['origin_time'][:,None]+np.arange(1,19)*3600)
            and np.all(result['target_time']<ASSESSMENT_START),'Campaign coordinate/assessment identity differs')
    for name,indices in role_indices.items():
        ids=result['case_id'][indices].tolist()
        require(dict(cases=len(ids),case_ids_sha256=digest(ids))==expected[name],'Campaign exact role membership differs from roster')
        start,end=auth['numeric_roles'][name]
        require(np.all(result['origin_time'][indices]>=start) and np.all(result['target_time'][indices]<end)
                and (name=='outer' or np.all(result['target_time'][indices,-1]+7200<=auth['refit_epoch'])),
                'Campaign role/refit target boundary differs')
    if kind!='labels':
        result['feature_schema']=np.array(ds.Q_ONLY_FEATURE_SCHEMA if arm=='q_only' else ds.FULL_FEATURE_SCHEMA)
        result['cohort_admission_digest']=np.array(cohort)
    if kind=='inputs': result['forecast_time']=result.pop('target_time')
    raw_identity=_array_digest(result)
    core=dict(schema='hydra-primary-campaign-data/v1',source_plan_sha256=SOURCE_PLAN,snapshot_sha256=SNAPSHOT,
        integrity_receipt_sha256=expected_integrity_hash,roster_sha256=ROSTER_SHA256,gauge_id=job['gauge_id'],fold=job['fold'],
        roles=expected,role_time_bounds=auth['roles'],refit_at_utc=auth['refit_at_utc'],simulated_label_delay_hours=2,
        cohort_admission_digest=cohort,cohort_mode='common_l168_complete',history_hours=length,input_arm=arm,
        operation=kind,raw_arrays_sha256=raw_identity,members=members,implementation_sha256=implementation_identity(),
        quality_policy=PRIMARY_APPROVED,case_count=count,unassigned_cases=0,
        numeric_value_members_decoded=list(fields),target_values_decoded=kind!='inputs',
        compressed_source_bytes_include_all_members=True,prior_complete_integrity_receipt_reused=True,
        raw_sources_reaudited=False,full_corpus_inspection_replayed=False)
    source_identity=digest(core)
    if kind=='inputs': result['inference_source_identity']=np.array(source_identity)
    for value in result.values(): value.setflags(write=False)
    guard.check(); exact_bytes=sum(x.nbytes for x in result.values()); require(exact_bytes<=max_bytes,'Actual bundle byte allowance exceeded')
    manifest={**core,'source_identity':source_identity,'returned_arrays_sha256':_array_digest(result),
        'construction':dict(dense_array_bytes=exact_bytes,estimated_array_bytes=predicted,source_shards_read=guard.shards,
                            source_bytes_read=guard.read_bytes,peak_rss_bytes=guard.peak,max_rss_bytes=max_rss_bytes,
                            max_array_bytes=max_bytes,scaler_fits=0,optimizer_updates=0,network_requests=0)}
    answer=dict(arrays=MappingProxyType(result),source_identity=source_identity,cohort_admission_digest=cohort,
                case_ids_sha256=digest(result['case_id'].tolist()),role_time_bounds=auth['roles'],refit_at_utc=auth['refit_at_utc'],manifest=manifest)
    if kind=='fit':
        answer.update(train_indices=tuple(role_indices['fit']),inner_indices=tuple(role_indices['inner_stop']),
                      gradient_case_ids=tuple(result['case_id'][role_indices['fit']].tolist()))
    return answer


def load_fit_bundle(corpus_root,job_spec,*,expected_integrity_hash=INTEGRITY_SHA256,max_bytes=MAX_BYTES,
                    max_rss_bytes=MAX_RSS,max_read_bytes=MAX_BYTES,check=lambda:None):
    return _load(corpus_root,job_spec,kind='fit',expected_integrity_hash=expected_integrity_hash,max_bytes=max_bytes,
                 max_rss_bytes=max_rss_bytes,max_read_bytes=max_read_bytes,check=check)


def load_outer_inputs(corpus_root,job_spec,*,expected_integrity_hash=INTEGRITY_SHA256,max_bytes=MAX_BYTES,
                      max_rss_bytes=MAX_RSS,max_read_bytes=MAX_BYTES,check=lambda:None):
    """All declared outer input cases; caller batches forward calls separately."""
    return _load(corpus_root,job_spec,kind='inputs',expected_integrity_hash=expected_integrity_hash,max_bytes=max_bytes,
                 max_rss_bytes=max_rss_bytes,max_read_bytes=max_read_bytes,check=check)


def load_outer_labels(corpus_root,job_spec,*,expected_outer_cohort_digest,expected_case_ids_sha256,
                      expected_integrity_hash=INTEGRITY_SHA256,max_bytes=MAX_BYTES,max_rss_bytes=MAX_RSS,
                      max_read_bytes=MAX_BYTES,check=lambda:None):
    require(ds._hex(expected_outer_cohort_digest) and ds._hex(expected_case_ids_sha256),'Explicit inference cohort/case commitment required before label access')
    result=_load(corpus_root,job_spec,kind='labels',expected_integrity_hash=expected_integrity_hash,max_bytes=max_bytes,
                 max_rss_bytes=max_rss_bytes,max_read_bytes=max_read_bytes,check=check,
                 expected_outer_cohort_digest=expected_outer_cohort_digest,expected_case_ids_sha256=expected_case_ids_sha256)
    require(result['cohort_admission_digest']==expected_outer_cohort_digest and result['case_ids_sha256']==expected_case_ids_sha256,
            'Labels do not match the declared outer inference cohort')
    return result


def iter_outer_batches(outer_bundle,*,batch_size=128):
    require(type(batch_size) is int and 1<=batch_size<=128 and outer_bundle['manifest']['operation']=='inputs',
            'Bounded authenticated input-only bundle required')
    arrays=outer_bundle['arrays']; require('target' not in arrays and 'target_mask' not in arrays,'Outer predictors cannot contain labels')
    n=len(arrays['origin_time'])
    for start in range(0,n,batch_size):
        batch={name:(value if value.ndim==0 else value[start:start+batch_size]) for name,value in arrays.items()}
        yield MappingProxyType(batch)
