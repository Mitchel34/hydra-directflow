"""Owned immutable original-history metadata for bounded pilot batch reuse.

No Q values, targets, transformed inputs, artificial masks, fills or forecasts
are retained. Interning reuses equal immutable primitive values, never merges
observations by timestamp or infers source/revision equivalence. All existing
per-slot qualifier/clock checks run at admission; views bind actual batch clocks,
original masks, gauge, order and source/cohort before original metadata reuse.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
import sys
from types import MappingProxyType

import numpy as np

from .local_data import GAUGES, iso
from .quality import PRIMARY_APPROVED, admit_usgs_quality
from .reference import ASSESSMENT_START, require

KEYS = ('observation_times', 'quality_flags', 'source_ids')
DEFAULT_MAX_STORAGE_BYTES = 256 * 1024**2
MAX_STORAGE_BYTES = 512 * 1024**2


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _rows(value, original_mask, times):
    """The original adapter's slot checks, yielded one row at a time."""
    require(isinstance(value,Mapping) and set(value)==set(KEYS),'Explicit per-slot original provenance required')
    n=original_mask.shape[0]
    require(all(isinstance(value[key],(list,tuple,np.ndarray)) and len(value[key])==n for key in value),'Provenance case dimension differs')
    for case in range(n):
        require(all(len(value[key][case])==168 for key in value),'Provenance history dimension differs')
        stamps,flags,ids=[],[],[]
        for slot in range(168):
            stamp=value['observation_times'][case][slot];native=value['quality_flags'][case][slot];source=value['source_ids'][case][slot]
            require(stamp is None or isinstance(stamp,Integral) and not isinstance(stamp,(bool,np.bool_)) and int(stamp)==int(times[case,slot]),
                    'Original source timestamp must equal its exact history slot')
            require(isinstance(native,(list,tuple)) and len(native)<=16 and all(type(flag) is str and 0<len(flag)<=32 for flag in native),
                    'Original qualifiers must be native ordered string sequences')
            require(source is None or isinstance(source,str) and 0<len(source)<=2048,'Original source ID malformed')
            if original_mask[case,slot]:
                require(stamp is not None and source is not None and admit_usgs_quality(native,policy_id=PRIMARY_APPROVED),
                        'Observed-mask/source-clock/approved-qualifier provenance differs')
            stamps.append(None if stamp is None else int(stamp));flags.append(tuple(native));ids.append(source)
        yield tuple(stamps),tuple(flags),tuple(ids)


def validate_native_provenance(value, original_mask, times):
    columns={key:[] for key in KEYS}
    for row in _rows(value,original_mask,times):
        for key,values in zip(KEYS,row):columns[key].append(values)
    return {key:tuple(rows) for key,rows in columns.items()}


def _native_digest(columns):
    """Stream exactly the legacy compact, sorted JSON without a full JSON tree."""
    result=hashlib.sha256();result.update(b'{')
    for position,key in enumerate(KEYS):
        if position:result.update(b',')
        result.update(json.dumps(key).encode());result.update(b':[')
        for index,row in enumerate(columns[key]):
            if index:result.update(b',')
            result.update(json.dumps(row,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode())
        result.update(b']')
    result.update(b'}');return result.hexdigest()


def _metadata_coordinates(arrays,cohort_digest):
    """Only coordinates and original masks are accessed; no Q/target values."""
    from .training import validate_coordinates, FULL_FEATURE_SCHEMA
    keys=set(iter(arrays));future='target_time' if 'target_time' in keys else 'forecast_time'
    coordinates={key:np.asarray(arrays[future if key=='target_time' else key])
                 for key in ('origin_time','history_time','target_time','site_id','case_id')}
    n=validate_coordinates(coordinates,168)
    origin=coordinates['origin_time'];site=coordinates['site_id'];ids=coordinates['case_id']
    require(np.all((origin>=169*3600)&(origin<ASSESSMENT_START)) and len(set(origin.tolist()))==n,
            'Unique development provenance origins required')
    gauge=GAUGES[int(site[0])]
    strings=tuple(value.decode() if isinstance(value,bytes) else str(value) for value in ids.tolist())
    require(strings==tuple(gauge+':'+iso(int(t)) for t in origin),'Exact provenance gauge/origin case IDs required')
    declared,cohort=np.asarray(arrays['feature_schema']),np.asarray(arrays['cohort_admission_digest'])
    require(declared.shape==cohort.shape==() and declared.item()==FULL_FEATURE_SCHEMA and cohort.item()==cohort_digest,
            'Provenance original H2 schema/cohort differs')
    mask=np.asarray(arrays['history_mask'])
    require(mask.shape==(n,168,2) and mask.dtype==np.dtype(bool) and np.array_equal(mask[...,0],mask[...,1]),
            'Original H2 Q/age masks must agree')
    return coordinates,mask[...,0],gauge,strings


@dataclass(frozen=True,init=False)
class ValidatedHistoryProvenance(Mapping):
    _columns: MappingProxyType
    _origin_bytes: bytes
    _mask_bytes: bytes
    _case_ids: tuple
    source_identity: str
    cohort_digest: str
    gauge_id: str
    provenance_sha256: str
    storage_bytes: int
    max_storage_bytes: int

    def __getitem__(self,key):return self._columns[key]
    def __iter__(self):return iter(KEYS)
    def __len__(self):return len(KEYS)
    @property
    def case_count(self):return len(self._case_ids)
    @property
    def origins(self):return np.frombuffer(self._origin_bytes,dtype=np.int64)
    @property
    def approved_mask(self):return np.frombuffer(self._mask_bytes,dtype=np.bool_).reshape(self.case_count,168)

    def select(self,indices):
        rows=tuple(indices)
        require(0<len(rows)<=128 and len(set(rows))==len(rows)
                and all(type(index) is int and 0<=index<self.case_count for index in rows),
                'Bounded unique source-row selection required')
        return ValidatedProvenanceBatch(self,rows)


@dataclass(frozen=True)
class ValidatedProvenanceBatch(Mapping):
    snapshot: ValidatedHistoryProvenance
    rows: tuple
    def __post_init__(self):
        require(type(self.snapshot) is ValidatedHistoryProvenance and type(self.rows) is tuple
                and 0<len(self.rows)<=128 and len(set(self.rows))==len(self.rows)
                and all(type(i) is int and 0<=i<self.snapshot.case_count for i in self.rows),'Exact immutable provenance selection required')
    def __getitem__(self,key):return tuple(self.snapshot[key][index] for index in self.rows)
    def __iter__(self):return iter(KEYS)
    def __len__(self):return len(KEYS)

    def checked_native(self,*,original_mask,times,origins,forecast_times,site_ids,case_ids,source_identity,cohort_digest):
        owned=self.snapshot;rows=list(self.rows);mask=np.asarray(original_mask)
        expected_origins=owned.origins[rows]
        actual_ids=tuple(value.decode() if isinstance(value,bytes) else str(value) for value in np.asarray(case_ids).tolist())
        require(source_identity==owned.source_identity and cohort_digest==owned.cohort_digest
                and np.array_equal(np.asarray(origins),expected_origins)
                and np.asarray(site_ids).shape==(len(rows),) and np.all(np.asarray(site_ids)==GAUGES.index(owned.gauge_id))
                and actual_ids==tuple(owned._case_ids[i] for i in rows),'Provenance source/cohort/gauge/case order changed')
        require(mask.shape==(len(rows),168) and mask.dtype==np.dtype(bool) and np.array_equal(mask,owned.approved_mask[rows])
                and np.array_equal(np.asarray(times),expected_origins[:,None]-np.arange(169,1,-1)*3600)
                and np.array_equal(np.asarray(forecast_times),expected_origins[:,None]+np.arange(1,19)*3600),
                'Provenance original mask or cutoff/history clocks changed')
        return {key:self[key] for key in KEYS}


def admit_history_provenance(arrays,value,*,source_identity,cohort_digest,max_storage_bytes=DEFAULT_MAX_STORAGE_BYTES):
    """Own and validate original metadata once, under an explicit memory bound.

    Normalize one source row at a time using the legacy trainer's JSON value
    contract; do not create a second full expanded provenance tree. The owned
    snapshot uses interned immutable primitives/rows and immutable byte buffers.
    This is structural validation, not source-data or scientific authorization.
    """
    require(_sha(source_identity) and _sha(cohort_digest),'Explicit source/cohort identities required')
    require(type(max_storage_bytes) is int and 0<max_storage_bytes<=MAX_STORAGE_BYTES,'Bounded provenance storage allowance required')
    coordinates,mask,gauge,case_ids=_metadata_coordinates(arrays,cohort_digest)
    if type(value) is ValidatedHistoryProvenance:
        require(value.source_identity==source_identity and value.cohort_digest==cohort_digest and value.gauge_id==gauge
                and value._case_ids==case_ids and np.array_equal(value.origins,coordinates['origin_time'])
                and np.array_equal(value.approved_mask,mask) and value.storage_bytes<=max_storage_bytes,
                'Existing immutable provenance snapshot differs from source coordinates/masks/bindings')
        return value
    n=len(case_ids);origin_bytes=np.asarray(coordinates['origin_time'],dtype=np.int64).tobytes()
    mask_bytes=np.asarray(mask,dtype=np.bool_).tobytes()
    # Rows and their pointers dominate retained storage; reject before building
    # them. Pools below account for each retained immutable primitive once.
    row_bytes=3*n*sys.getsizeof(tuple([None]*168))
    outer_bytes=3*sys.getsizeof(tuple([None]*n))
    fixed=sys.getsizeof(origin_bytes)+sys.getsizeof(mask_bytes)+row_bytes+outer_bytes
    fixed+=sys.getsizeof(case_ids)+sum(sys.getsizeof(item) for item in case_ids)+4096
    require(fixed<=max_storage_bytes,'Provenance snapshot storage allowance exceeded before row construction')
    require(isinstance(value,Mapping) and set(value)==set(KEYS),'Explicit original provenance columns required')
    require(all(len(value[key])==n for key in KEYS),'Provenance source row count differs')
    from .pilot_training import _canonical
    columns={key:[] for key in KEYS};strings={};stamps={};flags_pool={};owned_bytes=fixed
    def intern(pool,item):
        nonlocal owned_bytes
        if item is None:return None
        existing=pool.get(item)
        if existing is not None:return existing
        owned_bytes+=sys.getsizeof(item)
        require(owned_bytes<=max_storage_bytes,'Provenance snapshot interned-data storage allowance exceeded')
        pool[item]=item;return item
    owned_origins=np.frombuffer(origin_bytes,dtype=np.int64)
    owned_mask=np.frombuffer(mask_bytes,dtype=np.bool_).reshape(n,168)
    for index in range(n):
        # Bounded 168-slot normalization also retains legacy NumPy scalar/list
        # acceptance while keeping source IDs/qualifiers unchanged as values.
        one=_canonical({key:[value[key][index]] for key in KEYS})
        times=owned_origins[index]-np.arange(169,1,-1)*3600
        row=next(_rows(one,owned_mask[index:index+1],times[None,:]))
        columns['observation_times'].append(tuple(intern(stamps,stamp) for stamp in row[0]))
        flag_row=[]
        for flags in row[1]:
            original=tuple(intern(strings,flag) for flag in flags)
            flag_row.append(intern(flags_pool,original))
        columns['quality_flags'].append(tuple(flag_row))
        columns['source_ids'].append(tuple(intern(strings,source) for source in row[2]))
    frozen=MappingProxyType({key:tuple(rows) for key,rows in columns.items()})
    result=object.__new__(ValidatedHistoryProvenance)
    fields=dict(_columns=frozen,_origin_bytes=origin_bytes,_mask_bytes=mask_bytes,_case_ids=case_ids,
        source_identity=source_identity,cohort_digest=cohort_digest,gauge_id=gauge,
        provenance_sha256=_native_digest(frozen),storage_bytes=owned_bytes,max_storage_bytes=max_storage_bytes)
    for key,item in fields.items():object.__setattr__(result,key,item)
    return result
