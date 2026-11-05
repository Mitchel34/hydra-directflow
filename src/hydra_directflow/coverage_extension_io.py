"""Durable input preparation and bounded restores for Phase A.

The shared input-only route barrier is written before the original common-case
IDs are consulted for regression subsets. Stores own durable write/read charges
and verified file descriptors; this module never constructs models or labels.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import io
import json
from types import MappingProxyType
import zipfile

import numpy as np

from .coverage_extension_data import (START, END, CANDIDATE_ORIGINS, INFERENCE_KEYS,
                                    load_candidates, validate_candidates, route_bundle, input_bundle)
from .inference import validate_input_coordinates, validate_inputs
from .local_data import GAUGES, bounded_npy, iso
from .reference import ASSESSMENT_START, digest, require
from .trainer_bundle import _array_digest
from .training import FULL_FEATURE_SCHEMA

MAX_JSON = 8*1024**2
MAX_INPUT = 64*1024**2
MAX_ROUTE = 64*1024**2
MAX_CANDIDATE = 512*1024**2
SCALARS = ('feature_schema','cohort_admission_digest','inference_source_identity')
COORDINATES = ('origin_time','history_time','forecast_time','site_id','case_id')
ROUTE_BOOL = ('eligible168','eligible24','non_q_eligible','q168_complete','q24_complete',
              'no_covering_weather_cycle','incomplete_weather','missing_or_stale_soil','incomplete_soil',
              'unresolved_source_calendar')


def _json(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def _sha(raw):return hashlib.sha256(raw).hexdigest()
def _hex(value):return isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value)


def _npz(arrays):
    require(isinstance(arrays,Mapping) and all(isinstance(v,np.ndarray) and v.dtype.kind in 'fbiuUS' for v in arrays.values()),
            'Only non-object typed arrays may be persisted')
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);return stream.getvalue()


def _geometry(arrays):
    return {name:dict(shape=list(value.shape),dtype=value.dtype.str) for name,value in arrays.items()}


def _read(store,descriptor,maximum):
    require(isinstance(descriptor,dict) and set(descriptor)=={'path','bytes','sha256'}
            and type(descriptor['bytes']) is int and 0<descriptor['bytes']<=maximum
            and _hex(descriptor['sha256']),'Exact bounded stored artifact descriptor required')
    raw=store.read(descriptor,maximum)
    require(isinstance(raw,bytes) and len(raw)==descriptor['bytes'] and _sha(raw)==descriptor['sha256'],
            'Stored artifact bytes differ from descriptor')
    return raw


def _inventory(archive,names,maximum):
    items=archive.infolist()
    require(len(items)==len(names) and {x.filename for x in items}=={k+'.npy' for k in names}
            and sum(x.file_size for x in items)<=maximum,'Stored array inventory/expansion differs')


def _input_source_identity(manifest):
    core={k:v for k,v in manifest.items() if k!='source_identity'}
    require(manifest.get('schema')=='hydra-phase-a-inference-input-view/v1'
            and _hex(manifest.get('source_identity')) and digest(core)==manifest['source_identity']
            and manifest.get('target_values_read') is False and manifest.get('target_masks_read') is False
            and manifest.get('protected_assessment_access') is False,
            'Committed inference input manifest differs')


def _persist_bundle(store,phase,gauge,length,kind,bundle):
    require(kind in ('expanded','clean') and gauge in GAUGES and length in (24,168),'Declared input bundle required')
    arrays=bundle['arrays'];manifest=bundle['manifest'];_input_source_identity(manifest)
    require(manifest['gauge_id']==gauge and manifest['history_hours']==length
            and manifest['feature_schema']==FULL_FEATURE_SCHEMA
            and manifest['source_identity']==bundle['source_identity']
            and manifest['cohort_admission_digest']==bundle['cohort_admission_digest'],
            'Prepared input bundle scope differs')
    array_artifact=store.write(f'inputs/{gauge}/L{length}/{kind}.npz',_npz(arrays),kind='INPUT_BUNDLE_ARRAYS')
    wrapper=dict(schema='hydra-phase-a-input-bundle-storage/v1',phase_plan_sha256=phase,
                 gauge_id=gauge,history_hours=length,kind=kind,arrays=array_artifact,
                 inference_manifest=manifest,geometry=_geometry(arrays))
    manifest_artifact=store.write(f'inputs/{gauge}/L{length}/{kind}.json',_json(wrapper),kind='INPUT_BUNDLE_MANIFEST')
    return dict(schema=wrapper['schema'],phase_plan_sha256=phase,gauge_id=gauge,history_hours=length,
                kind=kind,arrays=array_artifact,manifest=manifest_artifact,source_identity=bundle['source_identity'],
                cohort_admission_digest=bundle['cohort_admission_digest'],
                candidate_source_identity=bundle['candidate_source_identity'],source_root_identity=bundle['source_root_identity'],
                cases=manifest['case_count'],case_ids_sha256=manifest['case_ids_sha256'])


def _persist_route(store,phase,candidates,gauge):
    route=route_bundle(candidates,gauge);arrays=route['arrays']
    artifact=store.write(f'routes/{gauge}.npz',_npz(arrays),kind='INPUT_ONLY_ROUTE_ARRAYS')
    manifest=dict(schema='hydra-phase-a-route-map-storage/v1',phase_plan_sha256=phase,gauge_id=gauge,
                  candidate_source_identity=route['candidate_source_identity'],source_root_identity=route['source_root_identity'],
                  route_map_sha256=route['route_map_sha256'],arrays_sha256=_array_digest(arrays),geometry=_geometry(arrays),
                  arrays=artifact,cases=len(arrays['origin_time']),calendar_start_utc=iso(START),
                  calendar_end_exclusive_utc=iso(END),route_policy='L168_ELSE_CLEAN_L24_ELSE_UNAVAILABLE_V1',
                  independent_of_architecture_seed_and_verification_targets=True,target_values_read=False,
                  target_masks_read=False,protected_assessment_access=False)
    manifest['content_sha256']=digest(manifest)
    stored=store.write(f'routes/{gauge}.json',_json(manifest),kind='INPUT_ONLY_ROUTE_MANIFEST')
    return dict(schema=manifest['schema'],phase_plan_sha256=phase,gauge_id=gauge,arrays=artifact,
                manifest=stored,route_map_sha256=route['route_map_sha256'],
                candidate_source_identity=route['candidate_source_identity'],source_root_identity=route['source_root_identity'],
                cases=manifest['cases'],route_counts={name:int(np.count_nonzero(arrays['route']==name)) for name in ('L168','L24','UNAVAILABLE')})


def prepare_inputs(plan,reconciliation,store,*,check=lambda:None,read_budget=lambda n:None):
    """Build and commit routes before any target-derived regression selection."""
    phase=plan['plan_sha256'];require(_hex(phase) and plan.get('protected_assessment_access') is False,
                                    'Adopted development-only Phase A plan required')
    check()
    candidates=load_candidates(plan['corpus_root'],reference_cache_root=plan['reference_cache_root'],
        reference_month_identities=reconciliation['reference_month_identities'],check=check,read_budget=read_budget)
    verification=validate_candidates(candidates)
    candidate_artifacts=dict(
        arrays=store.write('candidates/arrays.npz',_npz(candidates['arrays']),kind='INPUT_CANDIDATE_ARRAYS'),
        routing=store.write('candidates/routing.npz',_npz(candidates['routing']),kind='INPUT_CANDIDATE_ROUTES'),
        manifest=store.write('candidates/manifest.json',_json(candidates['manifest']),kind='INPUT_CANDIDATE_MANIFEST'))
    routes={g:_persist_route(store,phase,candidates,g) for g in GAUGES}
    barrier=dict(schema='hydra-phase-a-input-route-barrier/v1',status='ALL3_INPUT_ONLY_ROUTES_COMMITTED_BEFORE_REGRESSION_SELECTION',
                 phase_plan_sha256=phase,candidate_source_identity=candidates['candidate_source_identity'],
                 source_root_identity=candidates['source_root_identity'],candidate_artifacts=candidate_artifacts,
                 routes=routes,route_policy='L168_ELSE_CLEAN_L24_ELSE_UNAVAILABLE_V1',
                 candidate_rows=3*CANDIDATE_ORIGINS,original_common_case_ids_consulted=False,
                 target_values_read=False,target_masks_read=False,model_calls=0,protected_assessment_access=False)
    # This write must return durably before the selectors below are consulted.
    barrier_artifact=store.write('INPUT_ROUTE_BARRIER.json',_json(barrier),kind='INPUT_ONLY_ROUTE_BARRIER')
    check()
    common=reconciliation['original_common_case_ids_by_gauge']
    require(set(common)==set(GAUGES),'All three authenticated original common cohorts required')
    bundles={};common_summary={}
    for gauge in GAUGES:
        ids=tuple(common[gauge]);require(ids and list(ids)==sorted(set(ids)), 'Exact unique original common-case ordering required')
        common_summary[gauge]=dict(cases=len(ids),case_ids_sha256=digest(list(ids)))
        bundles[gauge]={}
        for length in (168,24):
            check();expanded=input_bundle(candidates,gauge,length)
            expanded_desc=_persist_bundle(store,phase,gauge,length,'expanded',expanded)
            del expanded
            clean=input_bundle(candidates,gauge,length,case_ids=ids)
            require(clean['manifest']['case_ids_sha256']==common_summary[gauge]['case_ids_sha256'],
                    'Regression selector case identity differs')
            clean_desc=_persist_bundle(store,phase,gauge,length,'clean',clean)
            del clean
            bundles[gauge][str(length)]=dict(expanded=expanded_desc,clean=clean_desc)
    result=dict(schema='hydra-phase-a-prepared-inputs/v1',status='INPUTS_AND_ROUTES_COMMITTED',phase_plan_sha256=phase,
                candidate=candidate_artifacts,candidate_source_identity=candidates['candidate_source_identity'],
                source_root_identity=candidates['source_root_identity'],candidate_verification=verification,
                routes=routes,route_barrier=barrier_artifact,route_barrier_content_sha256=digest(barrier),
                bundles=bundles,original_common_cases=common_summary,
                target_values_read=False,target_masks_read=False,model_calls=0,scaler_fits=0,protected_assessment_access=False)
    result['receipt_artifact']=store.write('PREPARED_INPUTS.json',_json(result),kind='PREPARED_INPUT_RECEIPT')
    check();return result


def read_input_bundle(store,bundle_descriptor):
    """Restore a single small model view; coordinates precede all input values."""
    d=bundle_descriptor
    require(d.get('schema')=='hydra-phase-a-input-bundle-storage/v1' and _hex(d.get('phase_plan_sha256'))
            and d.get('gauge_id') in GAUGES and d.get('history_hours') in (24,168)
            and d.get('kind') in ('expanded','clean'),'Declared stored input bundle required')
    wrapper=json.loads(_read(store,d['manifest'],MAX_JSON))
    require(all(wrapper.get(k)==d[k] for k in ('schema','phase_plan_sha256','gauge_id','history_hours','kind','arrays')),
            'Stored input wrapper scope differs')
    manifest=wrapper['inference_manifest'];_input_source_identity(manifest)
    require(all(manifest[k]==d[k] for k in ('gauge_id','history_hours','source_identity','cohort_admission_digest',
                'candidate_source_identity','source_root_identity','case_ids_sha256'))
            and manifest['case_count']==d['cases'] and manifest['feature_schema']==FULL_FEATURE_SCHEMA,
            'Stored input source/cohort differs')
    n,length=manifest['case_count'],manifest['history_hours']
    require(type(n) is int and 0<n<=CANDIDATE_ORIGINS,'Bounded nonempty model input view required')
    specs=dict(origin_time=((n,),'<i8'),history_time=((n,length),'<i8'),forecast_time=((n,18),'<i8'),
               site_id=((n,),'<i8'),case_id=((n,),'<U29'),history=((n,length,2),'<f8'),history_mask=((n,length,2),'|b1'),
               weather=((n,18,4),'<f8'),weather_mask=((n,18,4),'|b1'),state=((n,2),'<f8'),state_mask=((n,2),'|b1'),
               feature_schema=((),np.asarray(FULL_FEATURE_SCHEMA).dtype.str),cohort_admission_digest=((),'<U64'),
               inference_source_identity=((),'<U64'))
    require(wrapper['geometry']=={k:dict(shape=list(shape),dtype=np.dtype(dtype).str) for k,(shape,dtype) in specs.items()},
            'Stored inference geometry differs')
    raw=_read(store,d['arrays'],MAX_INPUT)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        _inventory(archive,specs,MAX_INPUT)
        arrays={k:bounded_npy(archive.read(k+'.npy'),shape=specs[k][0],dtype=np.dtype(specs[k][1])) for k in (*COORDINATES,*SCALARS)}
        require(np.all((arrays['origin_time']>=START)&(arrays['origin_time']<END))
                and arrays['feature_schema'].item()==FULL_FEATURE_SCHEMA
                and arrays['cohort_admission_digest'].item()==d['cohort_admission_digest']
                and arrays['inference_source_identity'].item()==d['source_identity'], 'Stored final-development scope or metadata differs')
        for start in range(0,n,64):
            batch={k:arrays[k] if k in SCALARS else arrays[k][start:start+64] for k in arrays}
            validate_input_coordinates(batch,history_hours=length,feature_schema=FULL_FEATURE_SCHEMA,
                expected_cohort_admission_digest=d['cohort_admission_digest'],expected_gauge_id=d['gauge_id'])
        require(digest(arrays['case_id'].tolist())==manifest['case_ids_sha256'],'Stored case membership differs')
        for name in INFERENCE_KEYS:
            if name not in arrays:arrays[name]=bounded_npy(archive.read(name+'.npy'),shape=specs[name][0],dtype=np.dtype(specs[name][1]))
    require(_array_digest({k:arrays[k] for k in INFERENCE_KEYS})==manifest['input_values_sha256']
            and _array_digest({k:v for k,v in arrays.items() if k!='inference_source_identity'})==manifest['input_arrays_sha256'],
            'Stored typed input values changed')
    for start in range(0,n,64):
        batch={k:v if k in SCALARS else v[start:start+64] for k,v in arrays.items()}
        validate_inputs(batch,history_hours=length,feature_schema=FULL_FEATURE_SCHEMA,
            expected_cohort_admission_digest=d['cohort_admission_digest'],expected_gauge_id=d['gauge_id'])
        require(all(batch[k].all() for k in ('history_mask','weather_mask','state_mask')), 'Stored clean input view contains missing values')
    for value in arrays.values():value.setflags(write=False)
    return dict(arrays=MappingProxyType(arrays),source_identity=d['source_identity'],cohort_admission_digest=d['cohort_admission_digest'],
                candidate_source_identity=d['candidate_source_identity'],source_root_identity=d['source_root_identity'],
                case_ids_sha256=d['case_ids_sha256'],manifest=manifest)


def read_route(store,route_descriptor):
    """Restore the shared target-free full-calendar routing ledger."""
    d=route_descriptor
    require(d.get('schema')=='hydra-phase-a-route-map-storage/v1' and d.get('gauge_id') in GAUGES
            and _hex(d.get('phase_plan_sha256')),'Declared input-only route descriptor required')
    manifest=json.loads(_read(store,d['manifest'],MAX_JSON))
    require(manifest['content_sha256']==digest({k:v for k,v in manifest.items() if k!='content_sha256'})
            and all(manifest.get(k)==d[k] for k in ('schema','phase_plan_sha256','gauge_id','arrays','route_map_sha256',
                'candidate_source_identity','source_root_identity','cases'))
            and manifest['cases']==CANDIDATE_ORIGINS and manifest['calendar_start_utc']==iso(START)
            and manifest['calendar_end_exclusive_utc']==iso(END)
            and manifest['independent_of_architecture_seed_and_verification_targets'] is True
            and manifest['target_values_read'] is False and manifest['target_masks_read'] is False
            and manifest['protected_assessment_access'] is False, 'Stored route source/scope differs')
    geometry=manifest['geometry'];names=set(geometry)
    required={'origin_time','forecast_time','case_id','site_id','route','primary_reason',*ROUTE_BOOL,
              'weather_reference_time','soil_reference_time','weather_age_hours','soil_age_hours',
              'last_original_admissible_time','last_original_present_time','last_original_admissible_age_hours',
              'last_original_present_age_hours','source_cycle','soil_source_cycle','weather_source_leads',
              'weather_cache_key','soil_cache_key','source_chunk_index','source_row_index'}
    required|={f'L{length}_{suffix}' for length in (168,24) for suffix in ('missing_hours','missing_runs','longest_missing_run',
        'longest_start_index','longest_end_exclusive','longest_gap_oldest_age_hours','longest_gap_youngest_age_hours',
        'longest_gap_first_time','longest_gap_last_time')}
    require(names==required,'Route arrays cannot contain target values/masks or undeclared fields')
    specs={}
    for name,g in geometry.items():
        dtype=np.dtype(g['dtype']);shape=tuple(g['shape']);expected_shape=(CANDIDATE_ORIGINS,18) if name in ('forecast_time','weather_source_leads') else (CANDIDATE_ORIGINS,)
        require(shape==expected_shape and dtype.kind in 'biuU' and dtype.itemsize<=1024,
                'Bounded route array geometry required')
        if name in ROUTE_BOOL:require(dtype==np.dtype(bool),'Boolean route masks required')
        elif name in ('case_id','route','primary_reason','source_cycle','soil_source_cycle','weather_cache_key','soil_cache_key'):
            require(dtype.kind=='U','Original route string coordinates required')
        elif name in ('source_chunk_index','source_row_index'):
            require(dtype==np.dtype('<i4'),'Original route source indices required')
        else:require(dtype==np.dtype('<i8'),'Exact integer route clocks/counts required')
        specs[name]=(shape,dtype)
    with zipfile.ZipFile(io.BytesIO(_read(store,d['arrays'],MAX_ROUTE))) as archive:
        _inventory(archive,names,MAX_ROUTE)
        coords=('origin_time','forecast_time','case_id','site_id')
        arrays={k:bounded_npy(archive.read(k+'.npy'),shape=specs[k][0],dtype=specs[k][1]) for k in coords}
        origin=arrays['origin_time'];gauge=d['gauge_id']
        require(origin.dtype==np.dtype('<i8') and np.array_equal(origin,np.arange(START,END,3600,dtype=np.int64))
                and arrays['forecast_time'].dtype==np.dtype('<i8')
                and np.array_equal(arrays['forecast_time'],origin[:,None]+np.arange(1,19)*3600)
                and np.all(arrays['forecast_time']<ASSESSMENT_START)
                and np.array_equal(arrays['site_id'],np.full(CANDIDATE_ORIGINS,GAUGES.index(gauge),np.int64))
                and arrays['case_id'].tolist()==[gauge+':'+iso(int(t)) for t in origin], 'Stored route coordinate/gauge boundary differs')
        for name in names-set(coords):arrays[name]=bounded_npy(archive.read(name+'.npy'),shape=specs[name][0],dtype=specs[name][1])
    require(_array_digest(arrays)==manifest['arrays_sha256']
            and digest(dict(candidate_source_identity=d['candidate_source_identity'],gauge_id=gauge,
                            routing_arrays_sha256=_array_digest(arrays)))==d['route_map_sha256'], 'Stored route typed identity differs')
    long,short,nonq=arrays['eligible168'],arrays['eligible24'],arrays['non_q_eligible']
    reason_flags=('no_covering_weather_cycle','incomplete_weather','missing_or_stale_soil','incomplete_soil','unresolved_source_calendar')
    require(np.array_equal(nonq,~np.logical_or.reduce([arrays[k] for k in reason_flags]))
            and np.array_equal(arrays['q168_complete'],arrays['L168_missing_hours']==0)
            and np.array_equal(arrays['q24_complete'],arrays['L24_missing_hours']==0)
            and np.array_equal(long,nonq&arrays['q168_complete']) and np.array_equal(short,nonq&arrays['q24_complete'])
            and not (long&~short).any()
            and np.array_equal(arrays['route'],np.where(long,'L168',np.where(short,'L24','UNAVAILABLE'))), 'Stored fixed route hierarchy differs')
    expected_reason=np.where(~nonq,'NON_Q_INPUT_UNAVAILABLE',np.where(~arrays['q24_complete'],'RECENT_Q_HISTORY_INCOMPLETE',
                    np.where(~arrays['q168_complete'],'OLDER_Q_HISTORY_GAP_USE_L24','COMPLETE_L168')))
    require(np.array_equal(arrays['primary_reason'],expected_reason),'Stored mutually exclusive input reason differs')
    require({name:int(np.count_nonzero(arrays['route']==name)) for name in ('L168','L24','UNAVAILABLE')}==d['route_counts'],
            'Stored route count differs')
    for value in arrays.values():value.setflags(write=False)
    return dict(arrays=MappingProxyType(arrays),manifest=manifest,route_map_sha256=d['route_map_sha256'],
                gauge_id=gauge,candidate_source_identity=d['candidate_source_identity'],source_root_identity=d['source_root_identity'])
