"""Input-only final-development candidates and the fixed clean L168/L24 router.

The opaque source archives are authenticated, but target values, target masks,
mixed admission flags and target-derived exclusion members are never decoded.
Original USGS provenance is joined only to the exact causal history slots. A
history hour for a later origin can naturally be an earlier origin's valid time;
no origin joins a reading later than C-2. Monthly source values are not decoded.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from types import MappingProxyType
import zipfile

import numpy as np

from . import corpus_dataset as ds
from .cache import DerivedCache, _month_bounds
from .local_data import GAUGES, USGS_MANIFEST_SHA256, iso
from .quality import PRIMARY_APPROVED, admit_usgs_quality
from .reference import ASSESSMENT_START, digest, require
from .trainer_bundle import _array_digest

ROOT = Path(__file__).resolve().parents[2]
INTEGRITY_PATH = ROOT/'reports/development-corpus-residency-integrity-v1.json'
INTEGRITY_SHA256 = '155f563238b8bcbdb549dc761775c2b24d4171d3f1f7785acba23b4e1c956d94'
SOURCE_PLAN = '0dad073824984fd7eeba88e4c313d77b4015dcccb94e8eb27297fcf7c6b39b0f'
SNAPSHOT = 'b41630b7df9a216874eb4ba38a10176d4b72092bd6f1ce2ad4c4f369db56948c'
START = int(datetime(2023,7,31,tzinfo=timezone.utc).timestamp())
END = int(datetime(2024,7,30,6,tzinfo=timezone.utc).timestamp())
CANDIDATE_ORIGINS = 8766
HOUR = 3600
MAX_ARRAY_BYTES = 512*1024**2
MAX_READ_BYTES = 1024**3
INPUT_FIELDS = ('history','weather','state')
INPUT_COORDS = ('site_id','origin_time','history_time','target_time','case_id',
                'source_cycle','soil_source_cycle','weather_source_leads')
FORBIDDEN_MEMBERS = frozenset(('target','target_mask','admission_mask','eligibility_reason'))
INFERENCE_KEYS = ('history','history_mask','weather','weather_mask','state','state_mask',
                  'origin_time','history_time','forecast_time','site_id','case_id')
ROUTES = ('L168','L24','UNAVAILABLE')


def _sha(raw): return hashlib.sha256(raw).hexdigest()


class _Reader:
    def __init__(self, check, read_budget):
        self.check, self.read_budget = check, read_budget
        self.bytes = self.files = 0

    def read(self,path,maximum,expected=None,expected_bytes=None):
        self.check(); path=Path(path)
        size=path.lstat().st_size
        require(0<size<=maximum and (expected_bytes is None or size==expected_bytes), 'Input source byte bound differs')
        require(self.bytes+size<=MAX_READ_BYTES, 'Input-only preparation read allowance reached')
        self.read_budget(size); self.bytes+=size; self.files+=1
        raw=ds._read(path,maximum)
        require(expected is None or _sha(raw)==expected, 'Input source SHA differs')
        self.check(); return raw


def _authority(corpus_root, reader):
    root=Path(corpus_root).absolute();require(root==root.resolve(),'Stable corpus path without aliases required');ds._directory(root)
    receipt_raw=reader.read(INTEGRITY_PATH,4*1024**2,INTEGRITY_SHA256)
    receipt=json.loads(receipt_raw)
    # _plan authenticates metadata only. Charge its internal PLAN.json read.
    plan_size=(root/'PLAN.json').lstat().st_size
    reader.read_budget(plan_size); reader.bytes+=plan_size; reader.files+=1
    require(reader.bytes<=MAX_READ_BYTES,'Input metadata read allowance reached')
    plan,descriptors,plan_sha=ds._plan(root,SOURCE_PLAN)
    require(receipt['status']=='COMPLETE_COMMITTED_CORPUS_SNAPSHOT' and receipt['missing_chunks']==[]
            and receipt['stored_history_hours']==plan['history_hours']==168
            and receipt['plan_sha256']==SOURCE_PLAN and receipt['plan_file_sha256']==plan_sha
            and receipt['snapshot_sha256']==SNAPSHOT and receipt['numerical_payloads_checked'] is True
            and receipt['protected_assessment_access'] is False, 'Pinned complete resident corpus receipt differs')
    observed=receipt['observed_commits']
    require(len(observed)==len(descriptors)
            and [x['chunk'] for x in observed]==[u['id']+'/'+f"{c['index']:03d}" for u,c in descriptors]
            and all(ds._hex(x['commit_sha256']) for x in observed)
            and digest(dict(plan_sha256=SOURCE_PLAN,plan_file_sha256=plan_sha,observed=observed,missing=[]))==SNAPSHOT,
            'Pinned source commitment inventory differs')
    selected=[(u,c,o) for (u,c),o in zip(descriptors,observed)
              if ds._hour(c['first_origin_utc'])<END and ds._hour(c['last_origin_utc'])>=START]
    require(selected,'Final-development input descriptors are missing')
    return dict(root=root,descriptors=selected,plan_file_sha256=plan_sha)


def _read_input_shard(authority, unit, chunk, observed, reader):
    directory=authority['root']/'chunks'/observed['chunk']; ds._directory(directory)
    require({p.name for p in directory.iterdir()}=={'COMMIT.json','case-index.json','model.npz'},
            'Unexpected input shard inventory')
    commit=json.loads(reader.read(directory/'COMMIT.json',65536,observed['commit_sha256']))
    require(commit['schema']==ds.SCHEMA and commit['plan_sha256']==SOURCE_PLAN
            and commit['unit_id']==unit['id'] and commit['chunk_index']==chunk['index']
            and set(commit['payloads'])=={'case-index.json','model.npz'} and set(commit['arrays'])==ds.ALL_KEYS,
            'Input shard commitment differs')
    bodies={}
    for name,limit in (('case-index.json',256*1024),('model.npz',ds.MAX_SHARD_BYTES)):
        item=commit['payloads'][name]
        bodies[name]=reader.read(directory/name,limit,item['sha256'],item['bytes'])
    summary=json.loads(bodies['case-index.json'])
    require(summary['unit_id']==unit['id'] and summary['chunk_index']==chunk['index']
            and summary['origin_count']==chunk['origin_count'] and summary['case_count']==3*chunk['origin_count']
            and summary['first_origin_utc']==chunk['first_origin_utc'] and summary['last_origin_utc']==chunk['last_origin_utc']
            and summary['input_quality_policy']==PRIMARY_APPROVED
            and summary['calendar_unavailable_cycle_ids']==chunk['calendar_unavailable_cycle_ids']
            and ds._hex(summary['case_source_digest']), 'Input summary identity or input quality differs')
    require(set(summary['cycle_keys'])==set(chunk['cycle_ids'])
            and all(ds._hex(x) for d in (summary['cycle_keys'],summary['month_keys']) for x in d.values()),
            'Input source-key inventory differs')
    n=summary['case_count']; shapes=ds._shapes(n,168); arrays={}
    with zipfile.ZipFile(io.BytesIO(bodies['model.npz'])) as archive:
        items=archive.infolist()
        require(len(items)==len(ds.ALL_KEYS) and {x.filename for x in items}=={k+'.npy' for k in ds.ALL_KEYS}
                and sum(x.file_size for x in items)<=ds.MAX_SHARD_BYTES,'Input archive inventory/expansion differs')
        for name in INPUT_COORDS: arrays[name]=ds._decode(archive,commit,name,shapes[name])
        # Recompute this INPUT-only calendar status; never decode the stored
        # mixed admission or eligibility members, or either target member.
        arrays['source_calendar_status']=ds._calendar_status(arrays,chunk)
        ds._validate_coordinates(arrays,unit,chunk)
        require(np.all(arrays['target_time']<ASSESSMENT_START), 'Protected input coordinates forbidden')
        for field in INPUT_FIELDS:
            arrays[field+'_mask']=ds._decode(archive,commit,field+'_mask',shapes[field+'_mask'])
            arrays[field]=ds._decode(archive,commit,field,shapes[field])
    arrays['weather_cache_key']=np.array([summary['cycle_keys'].get(str(c),'') for c in arrays['source_cycle']],dtype='<U64')
    arrays['soil_cache_key']=np.array([summary['cycle_keys'].get(str(c),'') for c in arrays['soil_source_cycle']],dtype='<U64')
    selected=np.flatnonzero((arrays['origin_time']>=START)&(arrays['origin_time']<END))
    result={k:np.array(v[selected],copy=True) for k,v in arrays.items()}
    member=dict(source_chunk=observed['chunk'],commit_sha256=observed['commit_sha256'],
                model_archive_sha256=commit['payloads']['model.npz']['sha256'],
                case_source_digest=summary['case_source_digest'],rows=selected.tolist(),
                candidate_cycle_ids=chunk['cycle_ids'],calendar_unavailable_cycle_ids=chunk['calendar_unavailable_cycle_ids'])
    return result,member,summary['month_keys']


def _selected_npy(archive,name,geometry,columns):
    """Interpret only requested contiguous source-history columns of an NPY.

    ZIP decompression can buffer other bytes; only the selected bytes become
    NumPy values. The unused late-month qualifier positions are not decoded.
    """
    expected_shape=tuple(geometry['shape']); dtype=np.dtype(geometry['dtype'])
    require(dtype in (np.dtype('<i8'),np.dtype(bool)) and len(expected_shape) in (1,2)
            and (len(expected_shape)==1 or expected_shape[0]==3), 'Bounded provenance-only NPY geometry required')
    lo,hi=columns; total=expected_shape[-1]
    require(type(lo) is int and type(hi) is int and 0<=lo<hi<=total<=744,'Bounded causal source columns required')
    with archive.open(name+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        require(version in ((1,0),(2,0)), 'Supported provenance NPY format required')
        header=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        shape,fortran,found_dtype=header(stream)
        require(shape==expected_shape and not fortran and found_dtype==dtype,'Provenance NPY header differs')
        offset=stream.tell()
        require(archive.getinfo(name+'.npy').file_size==offset+int(np.prod(shape))*dtype.itemsize,
                'Provenance NPY expansion differs')
        rows=1 if len(shape)==1 else shape[0]; result=[]
        for row in range(rows):
            stream.seek(offset+(row*total+lo)*dtype.itemsize)
            raw=stream.read((hi-lo)*dtype.itemsize)
            require(len(raw)==(hi-lo)*dtype.itemsize,'Truncated selected provenance columns')
            result.append(np.frombuffer(raw,dtype=dtype).copy())
    return result[0] if len(expected_shape)==1 else np.stack(result)


def _read_history_month(cache_root,spec,reader,first,last):
    month,key=spec['month'],spec['cache_key']
    require(month in ds._months(first,last) and ds._hex(key) and ds._hex(spec['commit_sha256']),
            'Exact causal-history month commitment required')
    path=cache_root/'entries'/key; ds._directory(path)
    commit=json.loads(reader.read(path/'COMMIT.json',128*1024,spec['commit_sha256']))
    identity=commit['identity']
    require(commit['schema']=='hydra-derived-cache/v1' and commit['kind']=='usgs_month'
            and commit['key']==key==DerivedCache.key('usgs_month',identity)
            and identity['month']==month and identity['source_manifest_sha256']==USGS_MANIFEST_SHA256,
            'History provenance cache source identity differs')
    require(set(commit['payloads'])=={'arrays','metadata'}
            and set(commit['arrays'])=={'times','values','row_present','numeric_valid','quality_index','source_index'},
            'History provenance cache inventory differs')
    bodies={}
    for kind,extension,maximum in (('arrays','.npz',16*1024**2),('metadata','.json',4*1024**2)):
        descriptor=commit['payloads'][kind]
        require(descriptor['name']==descriptor['sha256']+extension and ds._hex(descriptor['sha256']),
                'Content-addressed history payload required')
        bodies[kind]=reader.read(path/descriptor['name'],maximum,descriptor['sha256'],descriptor['bytes'])
    require({p.name for p in path.iterdir()}=={'COMMIT.json',*(d['name'] for d in commit['payloads'].values())},
            'History cache inventory changed')
    metadata=json.loads(bodies['metadata']); cutoff=metadata.get('development_cutoff_exclusive_epoch')
    require(metadata['month']==month,'History metadata month differs')
    if month=='2024-07':
        require(cutoff==ASSESSMENT_START
                and identity.get('development_export',{}).get('cutoff_exclusive_epoch')==ASSESSMENT_START
                and metadata.get('source_receipt',{}).get('development_export',{}).get('cutoff_exclusive_epoch')==ASSESSMENT_START,
                'July history requires its existing development-only export')
    else: require(cutoff is None,'Unexpected history development cutoff')
    month_start,month_last=_month_bounds(month,development_cutoff=cutoff)
    lo=max(first,month_start);hi=min(last,month_last)
    require(lo<=hi and hi+2*HOUR<=END-HOUR,'Source provenance exceeds latest causal history')
    columns=((lo-month_start)//HOUR,(hi-month_start)//HOUR+1)
    with zipfile.ZipFile(io.BytesIO(bodies['arrays'])) as archive:
        names={'times','values','row_present','numeric_valid','quality_index','source_index'}
        require(len(archive.infolist())==6 and {x.filename for x in archive.infolist()}=={n+'.npy' for n in names}
                and sum(x.file_size for x in archive.infolist())<=16*1024**2,'History cache archive expansion differs')
        # Never open values.npy. It is not needed to recover input provenance.
        arrays={name:_selected_npy(archive,name,commit['arrays'][name],columns)
                for name in ('times','row_present','numeric_valid','quality_index','source_index')}
    require(np.array_equal(arrays['times'],np.arange(lo,hi+HOUR,HOUR,dtype=np.int64)), 'Exact source-history axis differs')
    present,valid=arrays['row_present'],arrays['numeric_valid']
    require(present.dtype==valid.dtype==np.dtype(bool) and present.shape==valid.shape==(3,len(arrays['times']))
            and not (valid&~present).any(),'History provenance numeric/presence masks differ')
    qualities,sources=metadata['qualities'],metadata['sources']
    require(isinstance(qualities,list) and isinstance(sources,list)
            and all(isinstance(q,list) and all(type(f) is str for f in q) for q in qualities)
            and all(type(s) is str and s for s in sources),'Original provenance dictionaries differ')
    for field,dictionary in (('quality_index',qualities),('source_index',sources)):
        values=arrays[field]
        require(values.shape==present.shape and values.dtype==np.dtype('<i8') and (values[~present]==-1).all()
                and (values[present]>=0).all() and (values[present]<len(dictionary)).all(),'History source dictionary indices differ')
    return arrays,metadata,dict(**spec,selected_start_utc=iso(lo),selected_last_utc=iso(hi),
        selected_columns=list(columns),monthly_value_member_decoded=False,unused_qualifier_positions_decoded=False)


def _gap_summary(mask):
    n,length=mask.shape
    result={name:np.zeros(n,np.int64) for name in ('missing_hours','missing_runs','longest_missing_run')}
    result.update(longest_start_index=np.full(n,-1,np.int64),longest_end_exclusive=np.full(n,-1,np.int64))
    for i,row in enumerate(mask):
        absent=~row; edges=np.diff(np.r_[False,absent,False].astype(np.int8))
        starts,ends=np.flatnonzero(edges==1),np.flatnonzero(edges==-1)
        result['missing_hours'][i]=int(absent.sum());result['missing_runs'][i]=len(starts)
        if len(starts):
            j=int(np.argmax(ends-starts));result['longest_missing_run'][i]=int(ends[j]-starts[j])
            result['longest_start_index'][i]=starts[j];result['longest_end_exclusive'][i]=ends[j]
    return result


def classify_inputs(arrays):
    """Pure input route decisions; unrelated keys may be poison objects."""
    origin=np.asarray(arrays['origin_time']);n=len(origin)
    require(origin.ndim==1 and origin.dtype==np.dtype('<i8') and n>0 and n<=3*CANDIDATE_ORIGINS
            and np.all((origin>=START)&(origin<END)) and np.all(origin%HOUR==0), 'Bounded final-development origins required')
    site,case_ids=np.asarray(arrays['site_id']),np.asarray(arrays['case_id'])
    require(site.shape==case_ids.shape==(n,) and site.dtype==np.dtype('<i8') and case_ids.dtype.kind=='U'
            and np.all((site>=0)&(site<len(GAUGES))) and len(set(zip(site.tolist(),origin.tolist())))==n,
            'Unique explicit input gauge/origin coordinates required')
    require(case_ids.tolist()==[GAUGES[int(s)]+':'+iso(int(c)) for s,c in zip(site,origin)],
            'Input gauge/case clock identity differs')
    history_time=np.asarray(arrays['history_time']);future=np.asarray(arrays['target_time'])
    require(history_time.dtype==future.dtype==np.dtype('<i8')
            and np.array_equal(history_time,origin[:,None]-np.arange(169,1,-1)*HOUR)
            and np.array_equal(future,origin[:,None]+np.arange(1,19)*HOUR)
            and np.all(future<ASSESSMENT_START),'Exact causal history and safe H18 clocks required')
    for field,shape in (('history',(n,168,2)),('weather',(n,18,4)),('state',(n,2))):
        value,mask=np.asarray(arrays[field]),np.asarray(arrays[field+'_mask'])
        require(value.shape==mask.shape==shape and value.dtype==np.dtype('<f8') and mask.dtype==np.dtype(bool)
                and np.isfinite(value[mask]).all() and np.isnan(value[~mask]).all(), 'Original input value/mask semantics differ: '+field)
    hm=np.asarray(arrays['history_mask']);sm=np.asarray(arrays['state_mask'])
    require(np.array_equal(hm[...,0],hm[...,1]) and np.array_equal(sm[:,0],sm[:,1]),'Paired input feature masks differ')
    history=np.asarray(arrays['history']);state=np.asarray(arrays['state'])
    require((history[...,0][hm[...,0]]>=0).all()
            and np.array_equal(history[...,1][hm[...,1]],((origin[:,None]-history_time)/HOUR)[hm[...,1]]),
            'Observed discharge or coordinate age differs')
    weather_epochs=np.full(n,-1,np.int64);soil_epochs=np.full(n,-1,np.int64)
    weather_valid=np.zeros(n,bool);soil_valid=np.zeros(n,bool)
    for i,c in enumerate(origin):
        cycle=str(arrays['source_cycle'][i]);soil=str(arrays['soil_source_cycle'][i])
        require(cycle or not np.asarray(arrays['weather_mask'])[i].any(), 'Observed weather lacks an original source cycle')
        require(soil or not sm[i].any(), 'Observed soil lacks an original source cycle')
        if cycle:
            t=ds._cycle_epoch(cycle);weather_epochs[i]=t
            require(t+2*HOUR<=c,'Future/unavailable weather cycle cannot enter routing')
            expected=(future[i]-t)//HOUR
            require(np.array_equal(arrays['weather_source_leads'][i],expected),'Single weather cycle/native leads differ')
            weather_valid[i]=2*HOUR<=c-t<=18*HOUR and np.all((expected>=1)&(expected<=36))
        else: require(np.all(np.asarray(arrays['weather_source_leads'][i])==-1),'Absent weather cycle has native leads')
        if soil:
            t=ds._cycle_epoch(soil);soil_epochs[i]=t
            require(t+2*HOUR<=c,'Future/unavailable soil state cannot enter routing')
            age=(int(c)-t)/HOUR;soil_valid[i]=2<=age<=8
            if sm[i].all():require(state[i,1]==age and 0<=state[i,0]<=1,'Soil state value/timestamp differs')
    reasons=dict(no_covering_weather_cycle=~weather_valid,incomplete_weather=~np.asarray(arrays['weather_mask']).all(axis=(1,2)),
                 missing_or_stale_soil=~soil_valid,incomplete_soil=~sm.all(axis=1),
                 unresolved_source_calendar=np.asarray(arrays['source_calendar_status'])!='RESOLVED_WITHIN_ACQUIRED_CALENDAR')
    nonq=~np.logical_or.reduce(list(reasons.values()))
    q168=hm[...,0].all(axis=1);q24=hm[:,-24:,0].all(axis=1)
    long=nonq&q168;short=nonq&q24
    require(not (long&~short).any(),'Complete L168 must imply clean L24 under identical non-Q rules')
    route=np.where(long,'L168',np.where(short,'L24','UNAVAILABLE'))
    reason=np.where(~nonq,'NON_Q_INPUT_UNAVAILABLE',np.where(~q24,'RECENT_Q_HISTORY_INCOMPLETE',
                    np.where(~q168,'OLDER_Q_HISTORY_GAP_USE_L24','COMPLETE_L168')))
    result=dict(eligible168=long,eligible24=short,non_q_eligible=nonq,q168_complete=q168,q24_complete=q24,
                route=route,primary_reason=reason,weather_reference_time=weather_epochs,soil_reference_time=soil_epochs,
                weather_age_hours=np.where(weather_epochs>=0,(origin-weather_epochs)//HOUR,-1),
                soil_age_hours=np.where(soil_epochs>=0,(origin-soil_epochs)//HOUR,-1),**reasons)
    for length in (168,24):
        gaps=_gap_summary(hm[:,-length:,0])
        for key,value in gaps.items():result[f'L{length}_{key}']=value
        exists=gaps['longest_missing_run']>0
        oldest=np.where(exists,length+1-gaps['longest_start_index'],-1)
        youngest=np.where(exists,length+2-gaps['longest_end_exclusive'],-1)
        result[f'L{length}_longest_gap_oldest_age_hours']=oldest
        result[f'L{length}_longest_gap_youngest_age_hours']=youngest
        result[f'L{length}_longest_gap_first_time']=np.where(exists,origin-oldest*HOUR,-1)
        result[f'L{length}_longest_gap_last_time']=np.where(exists,origin-youngest*HOUR,-1)
    return result


def _join_provenance(arrays,stores):
    n=len(arrays['origin_time']);times=arrays['history_time'];site=arrays['site_id']
    qidx=np.full((n,168),-1,np.int32);sidx=qidx.copy();present=np.zeros((n,168),bool);valid=present.copy()
    qualities=[];sources=[];qmap={};smap={}
    for month,(native,meta,_) in stores.items():
        first,last=int(native['times'][0]),int(native['times'][-1]);take=(times>=first)&(times<=last)
        rows,slots=np.where(take);columns=((times[rows,slots]-first)//HOUR).astype(np.int64);gauges=site[rows]
        observed=native['row_present'][gauges,columns];numeric=native['numeric_valid'][gauges,columns]
        present[rows,slots]=observed;valid[rows,slots]=numeric
        global_q=[];global_s=[]
        for flags in meta['qualities']:
            key=tuple(flags)
            if key not in qmap:qmap[key]=len(qualities);qualities.append(flags.copy())
            global_q.append(qmap[key])
        for source in meta['sources']:
            if source not in smap:smap[source]=len(sources);sources.append(source)
            global_s.append(smap[source])
        rr,ss=rows[observed],slots[observed]
        qidx[rr,ss]=np.asarray(global_q,np.int32)[native['quality_index'][gauges[observed],columns[observed]]]
        sidx[rr,ss]=np.asarray(global_s,np.int32)[native['source_index'][gauges[observed],columns[observed]]]
    admitted_quality=np.array([admit_usgs_quality(q,policy_id=PRIMARY_APPROVED) for q in qualities],bool)
    approved=present&valid
    approved[present] &= admitted_quality[qidx[present]]
    require(np.array_equal(approved,arrays['history_mask'][...,0]), 'Original history mask differs from exact native qualifiers/numeric provenance')
    # Causal join is explicit for every slot, including rejected/missing rows.
    require(np.all(times+2*HOUR<=arrays['origin_time'][:,None]), 'History provenance violates C-2 availability')
    last_admitted=np.max(np.where(approved,times,-1),axis=1)
    last_present=np.max(np.where(present,times,-1),axis=1)
    result=dict(history_quality_index=qidx,history_source_index=sidx,history_row_present=present,history_numeric_valid=valid,
                last_original_admissible_time=last_admitted,last_original_present_time=last_present,
                last_original_admissible_age_hours=np.where(last_admitted>=0,(arrays['origin_time']-last_admitted)//HOUR,-1),
                last_original_present_age_hours=np.where(last_present>=0,(arrays['origin_time']-last_present)//HOUR,-1))
    return result,dict(qualities=qualities,sources=sources,original_qualifier_order_and_duplicates_retained=True,
                      observation_time_rule='history_time_when_row_present_else_absent',
                      original_available_at=None,historical_delivery_vintage_known=False)


def load_candidates(corpus_root, *, reference_cache_root, reference_month_identities,
                    check=lambda:None, read_budget=lambda n:None):
    """Build the complete final-development input universe; no model or labels."""
    reader=_Reader(check,read_budget);auth=_authority(corpus_root,reader)
    parts=[];members=[];needed={};source_chunks=[];source_rows=[]
    for unit,chunk,observed in auth['descriptors']:
        arrays,member,months=_read_input_shard(auth,unit,chunk,observed,reader)
        if not len(arrays['origin_time']):continue
        parts.append(arrays);members.append(member)
        source_chunks.extend([len(members)-1]*len(arrays['origin_time']));source_rows.extend(member['rows'])
        for month in ds._months(int(arrays['history_time'].min()),int(arrays['history_time'].max())):
            require(month in months and (month not in needed or needed[month]==months[month]),'Causal monthly source identity conflict')
            needed[month]=months[month]
        check()
    require(parts,'No final-development input candidates')
    arrays={key:np.concatenate([part[key] for part in parts]) for key in parts[0]}
    expected=np.repeat(np.arange(START,END,HOUR,dtype=np.int64),3)
    require(len(expected)==3*CANDIDATE_ORIGINS and np.array_equal(arrays['origin_time'],expected)
            and np.array_equal(arrays['site_id'],np.tile(np.arange(3),CANDIDATE_ORIGINS))
            and len(set(arrays['case_id'].tolist()))==len(expected),'Exact complete final-development gauge/origin calendar required')
    routes=classify_inputs(arrays)
    specs={x['month']:x for x in reference_month_identities}
    require(len(specs)==len(reference_month_identities) and set(needed)<=set(specs),'Bound original history month identities required')
    cache=Path(reference_cache_root).absolute();require(cache==cache.resolve(),'Stable history provenance path without aliases required')
    ds._directory(cache);ds._directory(cache/'entries')
    first,last=int(arrays['history_time'].min()),int(arrays['history_time'].max())
    stores={}
    for month,key in sorted(needed.items()):
        require(specs[month]['cache_key']==key,'History cache differs from candidate source lineage')
        stores[month]=_read_history_month(cache,specs[month],reader,first,last)
    provenance,dictionaries=_join_provenance(arrays,stores)
    arrays.update(provenance,source_chunk_index=np.asarray(source_chunks,np.int32),source_row_index=np.asarray(source_rows,np.int32))
    require(sum(v.nbytes for v in arrays.values())+sum(v.nbytes for v in routes.values())<=MAX_ARRAY_BYTES,'Input candidate allocation exceeds bound')
    core=dict(schema='hydra-phase-a-input-only-candidates/v1',source_plan_sha256=SOURCE_PLAN,snapshot_sha256=SNAPSHOT,
        source_plan_file_sha256=auth['plan_file_sha256'],integrity_receipt_sha256=INTEGRITY_SHA256,
        calendar_start_utc=iso(START),calendar_end_exclusive_utc=iso(END),candidate_origins_per_gauge=CANDIDATE_ORIGINS,
        candidate_rows=len(expected),input_quality_policy=PRIMARY_APPROVED,source_members=members,
        history_month_identities=[store[2] for store in stores.values()],provenance_dictionaries=dictionaries,
        candidate_arrays_sha256=_array_digest(arrays),routing_arrays_sha256=_array_digest(routes),
        decoded_corpus_members=[*INPUT_COORDS,*[x for f in INPUT_FIELDS for x in (f,f+'_mask')]],
        target_value_members_decoded=False,target_quality_members_decoded=False,mixed_admission_members_decoded=False,
        monthly_source_values_decoded=False,source_archives_contain_target_bytes=True,
        causal_history_source_union=[iso(first),iso(last)],source_quality_indices_only_selected_history_columns=True,
        history_overlap_with_other_origins_valid_times_not_future_access=True,
        simulated_availability_delay_hours=2,route_policy='L168_ELSE_CLEAN_L24_ELSE_UNAVAILABLE_V1',
        gap_indices_relative_to_named_lookback=True,absent_time_or_age_sentinel=-1,
        target_imputation=False,historical_Q_filling=False,scaler_fits=0,model_calls=0,protected_assessment_access=False)
    identity=digest(core)
    manifest={**core,'candidate_source_identity':identity,'source_root_identity':identity,
              'construction':dict(logical_source_bytes_read=reader.bytes,source_files_read=reader.files,
                                  array_bytes=sum(v.nbytes for v in arrays.values())+sum(v.nbytes for v in routes.values()))}
    for v in (*arrays.values(),*routes.values()):v.setflags(write=False)
    check()
    return dict(arrays=MappingProxyType(arrays),routing=MappingProxyType(routes),manifest=manifest,
                candidate_source_identity=identity,source_root_identity=identity)


def validate_candidates(candidates):
    """Authenticate a restored committed candidate object once before its views.

    The caller first authenticates the saved file descriptors and decodes/gates
    coordinates before other array members. This verifies the decoded typed
    arrays against their committed manifest; it does not revisit source data.
    """
    require(isinstance(candidates,Mapping) and isinstance(candidates['arrays'],Mapping)
            and isinstance(candidates['routing'],Mapping),'Explicit restored input-only candidates required')
    arrays,routes,manifest=candidates['arrays'],candidates['routing'],candidates['manifest']
    require(not (set(arrays)&FORBIDDEN_MEMBERS),'Targets or mixed admission flags cannot enter candidates')
    origin,site=np.asarray(arrays['origin_time']),np.asarray(arrays['site_id'])
    expected=np.repeat(np.arange(START,END,HOUR,dtype=np.int64),3)
    require(origin.dtype==site.dtype==np.dtype('<i8') and np.array_equal(origin,expected)
            and np.array_equal(site,np.tile(np.arange(3),CANDIDATE_ORIGINS))
            and np.array_equal(arrays['target_time'],origin[:,None]+np.arange(1,19)*HOUR)
            and np.all(np.asarray(arrays['target_time'])<ASSESSMENT_START)
            and np.array_equal(arrays['history_time'],origin[:,None]-np.arange(169,1,-1)*HOUR),
            'Restored candidate calendar/history/H18 scope differs')
    require(np.asarray(arrays['case_id']).tolist()==[g+':'+iso(int(c)) for c in range(START,END,HOUR) for g in GAUGES],
            'Restored candidate case/gauge identity differs')
    core={k:v for k,v in manifest.items() if k not in ('candidate_source_identity','source_root_identity','construction')}
    identity=digest(core)
    require(identity==manifest['candidate_source_identity']==manifest['source_root_identity']
            ==candidates['candidate_source_identity']==candidates['source_root_identity']
            and manifest['schema']=='hydra-phase-a-input-only-candidates/v1'
            and manifest['source_plan_sha256']==SOURCE_PLAN and manifest['snapshot_sha256']==SNAPSHOT
            and manifest['input_quality_policy']==PRIMARY_APPROVED
            and manifest['candidate_rows']==len(expected) and manifest['candidate_origins_per_gauge']==CANDIDATE_ORIGINS
            and manifest['calendar_start_utc']==iso(START) and manifest['calendar_end_exclusive_utc']==iso(END)
            and manifest['target_value_members_decoded'] is False and manifest['target_quality_members_decoded'] is False
            and manifest['mixed_admission_members_decoded'] is False and manifest['protected_assessment_access'] is False,
            'Restored candidate source manifest differs')
    require(_array_digest(arrays)==manifest['candidate_arrays_sha256']
            and _array_digest(routes)==manifest['routing_arrays_sha256'],'Restored candidate input or routing bytes changed')
    for key in ('eligible168','eligible24','non_q_eligible'):
        value=np.asarray(routes[key]);require(value.shape==origin.shape and value.dtype==np.dtype(bool),'Restored routing mask geometry differs')
    require(not (routes['eligible168']&~routes['eligible24']).any()
            and np.array_equal(routes['route'],np.where(routes['eligible168'],'L168',np.where(routes['eligible24'],'L24','UNAVAILABLE'))),
            'Restored fixed route hierarchy differs')
    for value in (*arrays.values(),*routes.values()):value.setflags(write=False)
    return dict(status='VERIFIED_RESTORED_INPUT_ONLY_CANDIDATES',candidate_source_identity=identity,
                arrays_sha256=manifest['candidate_arrays_sha256'],routing_sha256=manifest['routing_arrays_sha256'],
                target_values_read=False,target_masks_read=False,source_files_reopened=0,model_calls=0)


def route_bundle(candidates,gauge_id):
    require(gauge_id in GAUGES,'Confirmed routing gauge required')
    source=candidates['arrays'];rows=np.flatnonzero(source['site_id']==GAUGES.index(gauge_id))
    result={name:np.array(source[name][rows],copy=True) for name in ('origin_time','target_time','case_id','site_id')}
    result['forecast_time']=result.pop('target_time')
    result.update({name:np.array(value[rows],copy=True) for name,value in candidates['routing'].items()})
    for name in ('last_original_admissible_time','last_original_present_time','last_original_admissible_age_hours',
                 'last_original_present_age_hours','source_cycle','soil_source_cycle','weather_source_leads',
                 'weather_cache_key','soil_cache_key','source_chunk_index','source_row_index'):
        result[name]=np.array(source[name][rows],copy=True)
    identity=digest(dict(candidate_source_identity=candidates['candidate_source_identity'],gauge_id=gauge_id,
                         routing_arrays_sha256=_array_digest(result)))
    for value in result.values():value.setflags(write=False)
    return dict(arrays=MappingProxyType(result),route_map_sha256=identity,gauge_id=gauge_id,
                source_root_identity=candidates['source_root_identity'],candidate_source_identity=candidates['candidate_source_identity'])


def input_bundle(candidates,gauge_id,history_hours,*,case_ids=None):
    require(gauge_id in GAUGES and history_hours in (24,168),'Final-development full input view required')
    source,routing=candidates['arrays'],candidates['routing']
    eligible=(source['site_id']==GAUGES.index(gauge_id))&routing['eligible'+str(history_hours)]
    rows=np.flatnonzero(eligible)
    if case_ids is not None:
        require(type(case_ids) is tuple and len(case_ids)==len(set(case_ids)) and list(case_ids)==sorted(case_ids),
                'Exact chronological original-case regression subset required')
        lookup={str(source['case_id'][i]):int(i) for i in rows}
        require(all(case in lookup for case in case_ids),'Requested regression case is not independently input eligible')
        rows=np.asarray([lookup[case] for case in case_ids],dtype=np.int64)
    arrays={name:np.array(source[name][rows],copy=True) for name in ('origin_time','site_id','case_id')}
    arrays['history_time']=np.array(source['history_time'][rows,-history_hours:],copy=True)
    arrays['forecast_time']=np.array(source['target_time'][rows],copy=True)
    for field in INPUT_FIELDS:
        for name in (field,field+'_mask'):
            view=source[name][rows]
            arrays[name]=np.array(view[:,-history_hours:] if field=='history' else view,copy=True)
    require(all(arrays[f+'_mask'].all() for f in INPUT_FIELDS),'Input-only model view contains an inadmissible input')
    ids=arrays['case_id'].tolist()
    cohort=digest(dict(schema='hydra-phase-a-inference-cohort/v1',candidate_source_identity=candidates['candidate_source_identity'],
                       gauge_id=gauge_id,history_hours=history_hours,case_ids=ids,
                       role='CLEAN_REGRESSION_SUBSET_AFTER_ROUTING' if case_ids is not None else 'INPUT_ONLY_ELIGIBLE_PRODUCT'))
    value_sha=_array_digest(arrays)
    arrays['feature_schema']=np.array(ds.FULL_FEATURE_SCHEMA);arrays['cohort_admission_digest']=np.array(cohort)
    members=[]
    for number,member in enumerate(candidates['manifest']['source_members']):
        selected=rows[source['source_chunk_index'][rows]==number]
        if len(selected):members.append(dict(source_chunk=member['source_chunk'],commit_sha256=member['commit_sha256'],
            rows=source['source_row_index'][selected].tolist()))
    core=dict(schema='hydra-phase-a-inference-input-view/v1',candidate_source_identity=candidates['candidate_source_identity'],
        source_root_identity=candidates['source_root_identity'],gauge_id=gauge_id,history_hours=history_hours,
        feature_schema=ds.FULL_FEATURE_SCHEMA,cohort_admission_digest=cohort,case_ids_sha256=digest(ids),
        case_count=len(ids),input_values_sha256=value_sha,input_arrays_sha256=_array_digest(arrays),members=members,
        target_values_read=False,target_masks_read=False,model_calls=0,scaler_fits=0,protected_assessment_access=False)
    identity=digest(core);arrays['inference_source_identity']=np.array(identity)
    for value in arrays.values():value.setflags(write=False)
    return dict(arrays=MappingProxyType(arrays),source_identity=identity,cohort_admission_digest=cohort,
                candidate_source_identity=candidates['candidate_source_identity'],source_root_identity=candidates['source_root_identity'],
                manifest={**core,'source_identity':identity},case_ids_sha256=digest(ids))
