"""Phase A evaluation after the authenticated whole-campaign prediction barrier.

The evaluator verifies every clean/input-only/routed artifact before opening
reference values or qualifiers. It performs no inference, fitting or target
repair. All policies retain their input-only issuance populations and use
independent per-lead verification masks.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

from .cache import DerivedCache
from .contract import ARCHITECTURES, GAUGE_IDS
from .corpus_dataset import _read as resident_read
from .coverage_extension_io import read_route
from .coverage_extension_scoring import (
    START, END, ORIGINS, PRODUCTS, SEEDS, COHORTS, METRICS, validate_calendar,
    verification_masks, score_product, verify_product_report, aggregate_seed_reports,
    high_flow_coverage, fixed_events,
)
from .local_data import bounded_npy
from .reference import ASSESSMENT_START, digest, require
from .trainer_bundle import _array_digest

ROOT=Path(__file__).resolve().parents[2]
CATALOG_RELATIVE='reports/development-coverage-regimes-v2.json'
CATALOG_SHA='8e82f613165a5d452c74a30783b02e1c0ae5fb5d93dbe2e338d0010c215e8c0a'
FULL_SCHEMA='FULL_HRRR_H2_W4_S2_V1'
MAX_JSON=8*1024**2
MAX_PREDICTION=64*1024**2
FIVE=('prediction_q','case_id','origin_time','forecast_time','site_id')


def _sha(raw): return hashlib.sha256(raw).hexdigest()
def _hex(value): return isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value)
def _json(value): return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def _job(g,a,h,s): return f'final_development__{g}__{a}__full_L{h}__seed{s}'
def _routed(g,a,s): return f'{g}__{a}__seed{s}'


def _read(store,descriptor,maximum):
    require(isinstance(descriptor,dict) and set(descriptor)=={'path','bytes','sha256'}
            and isinstance(descriptor['path'],str) and type(descriptor['bytes']) is int
            and 0<descriptor['bytes']<=maximum and _hex(descriptor['sha256']), 'Exact bounded evaluation artifact descriptor required')
    raw=store.read(descriptor,maximum)
    require(isinstance(raw,bytes) and len(raw)==descriptor['bytes'] and _sha(raw)==descriptor['sha256'],
            'Evaluation artifact differs from committed bytes')
    return raw


def _coordinates(route,gauge):
    arrays=route['arrays'];origin,future=validate_calendar(arrays['origin_time'],arrays['forecast_time'],gauge)
    require(np.array_equal(arrays['site_id'],np.full(ORIGINS,GAUGE_IDS.index(gauge),np.int64)), 'Route gauge coordinates differ')
    return origin,future


def _prediction_arrays(store,entry,receipt,route,mask,gauge,*,routed=False):
    raw=_read(store,entry['arrays'],MAX_PREDICTION);n=ORIGINS if routed else int(mask.sum())
    require(0<n<=ORIGINS,'Positive bounded product population required')
    names=set(FIVE)|({'route_history_hours','route_checkpoint_index'} if routed else set())
    specs={'case_id':((n,),np.dtype('<U29')),'origin_time':((n,),np.dtype('<i8')),
           'forecast_time':((n,18),np.dtype('<i8')),'site_id':((n,),np.dtype('<i8')),
           'prediction_q':((n,18),np.dtype('<f8'))}
    if routed:specs.update(route_history_hours=((n,),np.dtype('<i2')),route_checkpoint_index=((n,),np.dtype('i1')))
    expected_positions=np.arange(ORIGINS) if routed else np.flatnonzero(mask)
    source=route['arrays'];coords={}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries=archive.infolist()
        require(len(entries)==len(names) and {x.filename for x in entries}=={k+'.npy' for k in names}
                and sum(x.file_size for x in entries)<=MAX_PREDICTION,'Prediction archive inventory/expansion differs')
        for name in ('case_id','origin_time','forecast_time','site_id'):
            coords[name]=bounded_npy(archive.read(name+'.npy'),shape=specs[name][0],dtype=specs[name][1])
            require(np.array_equal(coords[name],source[name][expected_positions]),'Product coordinates differ from precommitted route population')
        require(np.all(coords['forecast_time']<ASSESSMENT_START),'Protected product coordinates forbidden before values')
        for name in names-set(coords):coords[name]=bounded_npy(archive.read(name+'.npy'),shape=specs[name][0],dtype=specs[name][1])
    require(_array_digest(coords)==receipt['predictions_array_sha256'],'Typed product values differ from receipt')
    if routed:
        choices=source['route'];expected_h=np.where(choices=='L168',168,np.where(choices=='L24',24,0)).astype(np.int16)
        expected_i=np.where(choices=='L168',0,np.where(choices=='L24',1,-1)).astype(np.int8)
        require(np.array_equal(coords['route_history_hours'],expected_h)
                and np.array_equal(coords['route_checkpoint_index'],expected_i),'Routed per-origin checkpoint/history choices differ')
        values=np.array(coords['prediction_q'],copy=True)
        require(np.isfinite(values[mask]).all() and np.isnan(values[~mask]).all(),'Every requested routed vector must be finite; unavailable stays NaN')
    else:
        require(np.isfinite(coords['prediction_q']).all(),'Requested input-only or clean prediction is nonfinite')
        values=np.full((ORIGINS,18),np.nan,np.float64);values[mask]=coords['prediction_q']
    values.setflags(write=False)
    return values


def authenticate_prediction_barrier(plan,store,inputs_receipt,barrier,expected_sha,reconciliation,*,check=lambda:None):
    """Authenticate the entire population/forecast barrier before target reads."""
    phase=plan['plan_sha256'];require(_hex(phase) and plan['protected_assessment_access'] is False,'Development-only Phase A plan required')
    require(_hex(expected_sha) and digest(barrier)==expected_sha
            and barrier['schema']=='hydra-coverage-extension-prediction-barrier/v1'
            and barrier['status']=='ALL_REQUESTED_PREDICTIONS_COMMITTED_BEFORE_TARGETS'
            and barrier['phase_plan_sha256']==phase,'Complete authenticated prediction barrier required before targets')
    jobs={_job(g,a,h,s) for g in GAUGE_IDS for a in ARCHITECTURES for h in (168,24) for s in SEEDS}
    routed_ids={_routed(g,a,s) for g in GAUGE_IDS for a in ARCHITECTURES for s in SEEDS}
    require(set(barrier['clean_products'])==set(barrier['input_products'])==jobs
            and set(barrier['routed_products'])==routed_ids,'All72 clean/input and36 routed products required before targets')
    require(inputs_receipt['schema']=='hydra-phase-a-prepared-inputs/v1'
            and inputs_receipt['status']=='INPUTS_AND_ROUTES_COMMITTED' and inputs_receipt['phase_plan_sha256']==phase
            and inputs_receipt['target_values_read'] is False and inputs_receipt['target_masks_read'] is False
            and inputs_receipt['protected_assessment_access'] is False and set(inputs_receipt['routes'])==set(GAUGE_IDS),
            'Committed target-free input preparation required')
    expected_input={k:v for k,v in inputs_receipt.items() if k!='receipt_artifact'}
    require(barrier['input_preparation_receipt']==inputs_receipt['receipt_artifact']
            and barrier['input_preparation_content_sha256']==digest(expected_input)
            and json.loads(_read(store,barrier['input_preparation_receipt'],MAX_JSON))==expected_input,
            'Prediction barrier input-preparation binding differs')
    route_barrier=json.loads(_read(store,inputs_receipt['route_barrier'],MAX_JSON))
    require(digest(route_barrier)==inputs_receipt['route_barrier_content_sha256']==barrier['route_barrier_sha256']
            and route_barrier['phase_plan_sha256']==phase
            and route_barrier['status']=='ALL3_INPUT_ONLY_ROUTES_COMMITTED_BEFORE_REGRESSION_SELECTION'
            and route_barrier['routes']==inputs_receipt['routes']
            and route_barrier['original_common_case_ids_consulted'] is False
            and route_barrier['target_values_read'] is route_barrier['target_masks_read'] is False,
            'Input-only route barrier differs')
    require(_hex(barrier['clean_barrier_sha256']),'Clean regression barrier identity required')
    models={x['job_id']:x for x in reconciliation['models']}
    require(set(models)==jobs and len(reconciliation['models'])==72,'Authenticated72 model manifest required')
    records={r['job']['job_id']:r for r in plan['original_records']}
    require(set(records)==jobs and len(plan['original_records'])==72,'Exact original72 full-model records required')
    original_plans={r['receipt']['plan_sha256'] for r in records.values()}
    require(len(original_plans)==1 and all(_hex(x) for x in original_plans),'One original frozen campaign required')
    original_plan_sha=next(iter(original_plans))
    require(all(models[k]['checkpoint']==records[k]['receipt']['artifacts']['checkpoint'] for k in jobs),
            'Reconciled checkpoints differ from original completion records')
    routes={};common={};clean={};input_only={};routed={};proof=[]
    for gauge in GAUGE_IDS:
        routes[gauge]=read_route(store,inputs_receipt['routes'][gauge]);_coordinates(routes[gauge],gauge)
        require(routes[gauge]['manifest']['phase_plan_sha256']==phase,'Route phase identity differs')
        ids=list(reconciliation['original_common_case_ids_by_gauge'][gauge]);require(ids and ids==sorted(set(ids)),'Ordered original regression cases required')
        mask=np.isin(routes[gauge]['arrays']['case_id'],np.asarray(ids));require(int(mask.sum())==len(ids),'Original cases missing from candidate calendar')
        require(routes[gauge]['arrays']['case_id'][mask].tolist()==ids
                and np.all(routes[gauge]['arrays']['eligible168'][mask]),'Original cases are not unchanged clean L168 inputs')
        common[gauge]=mask
    for kind,table,destination in (('clean',barrier['clean_products'],clean),('input',barrier['input_products'],input_only)):
        if kind=='input':
            recovered_clean_barrier=dict(schema='hydra-coverage-extension-clean-barrier/v1',
                status='ALL72_CLEAN_REGRESSIONS_BYTE_EXACT',phase_plan_sha256=phase,
                clean_receipt_sha256={key:digest(value['receipt']) for key,value in clean.items()})
            require(set(clean)==jobs and digest(recovered_clean_barrier)==barrier['clean_barrier_sha256'],
                    'Global clean barrier does not match all72 authenticated clean receipts')
        for job_id in sorted(jobs):
            check();entry=table[job_id];require(set(entry)=={'arrays','receipt'},'Exact product artifact pair required')
            receipt=json.loads(_read(store,entry['receipt'],MAX_JSON));model=models[job_id]
            gauge,architecture,seed,length=[model[k] for k in ('gauge_id','architecture','seed','history_hours')]
            require(receipt['phase_plan_sha256']==phase and receipt['job_id']==job_id
                    and receipt['original_plan_sha256']==original_plan_sha
                    and all(receipt[k]==model[k] for k in ('gauge_id','architecture','seed','history_hours','device'))
                    and receipt['checkpoint_sha256']==model['checkpoint']['sha256']
                    and receipt['selected_best_epoch']==model['selected_best_epoch']
                    and receipt['optimizer_updates']==receipt['scaler_fits']==0
                    and receipt['target_values_read'] is receipt['target_masks_read'] is receipt['protected_assessment_access'] is False
                    and receipt['posthoc_clipping'] is False,'Committed model product identity/scope differs')
            bundle=inputs_receipt['bundles'][gauge][str(length)]['clean' if kind=='clean' else 'expanded']
            require(receipt['input_source_identity']==bundle['source_identity']
                    and receipt['input_cohort_admission_digest']==bundle['cohort_admission_digest']
                    and receipt['candidate_source_identity']==inputs_receipt['candidate_source_identity']
                    and _hex(receipt['saved_transform_sha256']),'Committed product input/transform identity differs')
            if kind=='clean':
                require(receipt['schema']=='hydra-coverage-extension-clean-regression/v1'
                        and receipt['status']=='BYTE_EXACT_ORIGINAL_PREDICTIONS'
                        and receipt['comparison']=='DTYPE_SHAPE_C_ORDER_BYTES' and receipt['tolerance'] is None
                        and receipt['training_case_ids_sha256']==model['gradient_case_ids_sha256']
                        and receipt['original_prediction_sha256']==model['original_predictions']['sha256'],
                        'Original byte-exact clean qualification required')
                selected=common[gauge]
            else:
                product='L168_INPUT_ONLY_V1' if length==168 else 'L24_INPUT_ONLY_V1'
                require(receipt['schema']=='hydra-coverage-extension-input-only-predictions/v1'
                        and receipt['status']=='ALL_REQUESTED_VECTORS_FINITE' and receipt['product']==product
                        and receipt['clean_barrier_sha256']==barrier['clean_barrier_sha256']
                        and receipt['clean_regression_receipt_sha256']==digest(clean[job_id]['receipt'])
                        and receipt['saved_transform_sha256']==clean[job_id]['receipt']['saved_transform_sha256'],
                        'Input-only product differs from its complete clean barrier')
                selected=routes[gauge]['arrays']['eligible168' if length==168 else 'eligible24']
            require(receipt['cases']==int(selected.sum()) and receipt['case_ids_sha256']==digest(routes[gauge]['arrays']['case_id'][selected].tolist()),
                    'Product population differs from precommitted eligibility')
            values=_prediction_arrays(store,entry,receipt,routes[gauge],selected,gauge)
            if kind=='input':
                require(values[common[gauge]].tobytes()==clean[job_id]['predictions'][common[gauge]].tobytes(),
                        'Expanded original-common predictions changed from byte-exact regression')
            destination[job_id]=dict(receipt=receipt,predictions=values,artifact_sha256=entry['arrays']['sha256'])
            proof.append(dict(kind=kind,job_id=job_id,prediction_artifact_sha256=entry['arrays']['sha256'],receipt_sha256=digest(receipt)))
    for gauge in GAUGE_IDS:
        arrays=routes[gauge]['arrays'];choices=arrays['route'];mask=arrays['eligible24']
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                key=_routed(gauge,architecture,seed);entry=barrier['routed_products'][key]
                require(set(entry)=={'arrays','receipt'},'Exact routed artifact pair required')
                receipt=json.loads(_read(store,entry['receipt'],MAX_JSON))
                a=input_only[_job(gauge,architecture,168,seed)];b=input_only[_job(gauge,architecture,24,seed)]
                checkpoints={'168':a['receipt']['checkpoint_sha256'],'24':b['receipt']['checkpoint_sha256']}
                require(receipt['schema']=='hydra-coverage-extension-routed-predictions/v1'
                        and receipt['status']=='ALL_INPUT_ELIGIBLE_ROUTED_VECTORS_COMMITTED'
                        and receipt['phase_plan_sha256']==phase and receipt['product']=='ROUTED_CLEAN_168_24_V1'
                        and receipt['gauge_id']==gauge and receipt['architecture']==architecture and receipt['seed']==seed
                        and receipt['route_map_sha256']==routes[gauge]['route_map_sha256']
                        and receipt['checkpoint_sha256_by_history']==checkpoints
                        and receipt['target_values_read'] is receipt['target_masks_read'] is False
                        and receipt.get('protected_assessment_access') is False and _hex(receipt['input_source_identity'])
                        and receipt.get('model_calls',0)==receipt.get('optimizer_updates',0)==receipt.get('scaler_fits',0)==0,
                        'Routed product identity/history map differs')
                require('cases' not in receipt or receipt['cases']==ORIGINS,'Routed calendar case count differs')
                values=_prediction_arrays(store,entry,receipt,routes[gauge],mask,gauge,routed=True)
                for choice,source in (('L168',a),('L24',b)):
                    selected=choices==choice
                    require(values[selected].tobytes()==source['predictions'][selected].tobytes(),
                            'Routed predictions do not copy the committed whole H18 checkpoint vectors')
                routed[key]=dict(receipt=receipt,predictions=values,artifact_sha256=entry['arrays']['sha256'])
                proof.append(dict(kind='routed',job_id=key,prediction_artifact_sha256=entry['arrays']['sha256'],receipt_sha256=digest(receipt)))
    return dict(routes=routes,common=common,clean=clean,input_only=input_only,routed=routed,models=models,
                authenticated_product_artifacts=proof,prediction_barrier_sha256=expected_sha)


def _reference_helpers():
    scripts=ROOT/'scripts';sys.path.insert(0,str(scripts))
    import postprocess_primary_events
    require(Path(postprocess_primary_events.__file__).resolve()==scripts/'postprocess_primary_events.py',
            'Reference helper must come from the bound checkout')
    return postprocess_primary_events


class _ReferenceGuard:
    def __init__(self,root,check,read_budget):self.root=Path(root).absolute();self.check=check;self.read_budget=read_budget
    def read(self,path,maximum):
        path=Path(path);require(path==path.resolve() and path.is_relative_to(self.root),'Reference read escaped approved resident cache')
        self.check();self.read_budget(path.lstat().st_size);raw=resident_read(path,maximum);self.check();return raw


def _load_reference_calendars(plan,reconciliation,*,check,read_budget):
    """Only called after every requested forecast product has authenticated."""
    helper=_reference_helpers();root=Path(plan['reference_cache_root']).absolute()
    require(root==root.resolve(),'Stable owned reference cache required')
    months=['2023-'+f'{m:02d}' for m in range(7,13)]+['2024-'+f'{m:02d}' for m in range(1,8)]
    source=[x for x in reconciliation['reference_month_identities'] if x['month'] in months]
    source=sorted(source,key=lambda x:x['month']);require([x['month'] for x in source]==months,'Exactly13 original reference months required')
    guard=_ReferenceGuard(root,check,read_budget);cache=DerivedCache(root);observations=[];receipts=[]
    for identity in source:
        rows,receipt=helper.checked_reference_month(cache,identity,guard);observations.extend(rows);receipts.append(receipt)
        require(len(observations)<=3*13*31*24,'Bounded reference population required')
    references=helper.build_reference_calendars(observations,start=START,end=ASSESSMENT_START)
    # The old plotting view masks rejected values. Keep original numerics
    # separately so rejection reasons do not confuse masking with source absence.
    for gauge,reference in references.items():
        raw=np.full(len(reference['times']),np.nan,np.float64);numeric_present=np.zeros(len(raw),bool)
        for row in observations:
            if row.site!=gauge or not START<=row.valid_at<ASSESSMENT_START:continue
            index=(row.valid_at-START)//3600
            if row.value is not None:raw[index]=row.value;numeric_present[index]=True
        raw.setflags(write=False);numeric_present.setflags(write=False)
        reference['original_numeric_values']=raw;reference['original_numeric_present']=numeric_present
    check();return references,receipts


def _load_catalog(check,read_budget):
    path=ROOT/CATALOG_RELATIVE;check();read_budget(path.lstat().st_size);raw=resident_read(path,16*1024**2)
    require(_sha(raw)==CATALOG_SHA,'Frozen event catalog differs')
    catalog=json.loads(raw);rows=[x for x in catalog['fold_regime_and_event_diagnostics'] if x['fold']=='final_development']
    require(len(rows)==3 and {x['gauge_id'] for x in rows}==set(GAUGE_IDS),'All3 fixed final-development catalogs required')
    return {x['gauge_id']:x for x in rows}


def _npz(arrays):
    require(all(isinstance(v,np.ndarray) and v.dtype.kind in 'biufUS' for v in arrays.values()),'Non-object typed evaluation arrays required')
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);return stream.getvalue()


def _targets(gauge,route,reference,source_receipts,barrier_sha,store):
    origin,future=_coordinates(route,gauge);positions=((future-START)//3600).astype(np.int32)
    require(np.array_equal(reference['times'],np.arange(START,ASSESSMENT_START,3600,dtype=np.int64))
            and np.all((positions>=0)&(positions<len(reference['times']))),'Reference target join escaped final-development calendar')
    values=np.asarray(reference['original_numeric_values'][positions],np.float64)
    flags=reference['quality_flags_json'][positions];present=reference['row_present'][positions]
    quality=verification_masks(gauge_id=gauge,origins=origin,forecast_time=future,labels=values,
                               qualifier_json=flags,present_mask=present)
    require(np.array_equal(quality['label_mask'],reference['admitted'][positions]),'Per-lead A/A+R policy differs from reference authority')
    arrays=dict(origin_time=origin,forecast_time=future,labels=values,label_mask=quality['label_mask'],reference_index=positions,
                target_exclusion_reason=quality['exclusion_reason'])
    ref_artifact=store.write(f'evaluation/targets/{gauge}/reference.npz',_npz(reference),kind='PHASE_A_ORIGINAL_REFERENCE')
    target_artifact=store.write(f'evaluation/targets/{gauge}/targets.npz',_npz(arrays),kind='PHASE_A_INDEPENDENT_TARGETS')
    core=dict(schema='hydra-phase-a-verification-targets/v1',gauge_id=gauge,prediction_barrier_sha256=barrier_sha,
        target_arrays_sha256=_array_digest(arrays),reference_arrays_sha256=_array_digest(reference),
        original_reference_months=[r['identity'] for r in source_receipts],
        quality_policy=quality['quality_policy'],original_qualifier_json_sha256=quality['original_qualifier_json_sha256'],
        label_mask_sha256=quality['label_mask_sha256'],original_values_sha256=quality['original_values_sha256'],
        exclusion_counts_overlap=quality['exclusion_counts_overlap'],original_flags_order_duplicates_retained=True,
        original_numeric_values_preserved_separately_from_plot_mask=True,targets_filled=False,protected_assessment_access=False)
    core['label_source_sha256']=digest(core)
    manifest=store.write(f'evaluation/targets/{gauge}/manifest.json',_json(core),kind='PHASE_A_TARGET_MANIFEST')
    return arrays,core,dict(reference=ref_artifact,arrays=target_artifact,manifest=manifest)


def _original_common_check(store,record,report):
    descriptor=record['receipt']['artifacts']['metrics']
    require(descriptor['sha256']==record['receipt']['hashes']['metrics'],'Original metric descriptor differs')
    raw=store.read_original(descriptor['path'],2*1024**2,expected_sha=descriptor['sha256'])
    require(len(raw)==descriptor['bytes'] and _sha(raw)==descriptor['sha256'],'Original metric artifact changed')
    old=json.loads(raw);job=record['job'];panel=old['panel'];new=report['cohorts']['ORIGINAL_COMMON']
    require(old['checkpoint_sha256']==record['receipt']['hashes']['checkpoint']
            and old['predictions_sha256']==record['receipt']['hashes']['predictions']
            and all(panel['identity'][k]==job[k] for k in ('gauge_id','architecture','seed','history_hours')),
            'Original baseline metric scope differs')
    require(len(panel['leads'])==len(new['leads'])==18,'All18 original common metric leads required')
    largest=0.
    for a,b in zip(new['leads'],panel['leads']):
        require(all(a[k]==b[k] for k in ('lead_hours','n','n_requested','distinct_target_hours','distinct_utc_dates','negative_predictions_scored')),
                'Original common support counts differ')
        for key in METRICS:
            x,y=a[key],b[key];require(x is None and y is None or x is not None and y is not None
                and math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-10),'Original common metric differs')
            if x is not None:largest=max(largest,abs(x-y))
    require(new['primary_score'] is None and panel['primary_score'] is None or new['primary_score'] is not None
            and panel['primary_score'] is not None and math.isclose(new['primary_score'],panel['primary_score'],rel_tol=1e-10,abs_tol=1e-10),
            'Original common primary differs')
    return dict(job_id=job['job_id'],status='ORIGINAL_COMMON_METRICS_REPRODUCED',maximum_absolute_difference=largest)


def _shared_high_flow(report):
    return {k:v for k,v in report.items() if k!='identity'}


def _aggregate_events(events):
    require(set(events)==set(SEEDS),'All event seed reports required');first=events[17];output=[]
    require(all([e['event_id'] for e in events[s]['events']]==[e['event_id'] for e in first['events']] for s in SEEDS),
            'Seed event catalog changed')
    for ei,anchor in enumerate(first['events']):
        rows=[events[s]['events'][ei] for s in SEEDS]
        result={k:deepcopy(v) for k,v in anchor.items() if k!='metrics'}
        counts={k:v for k,v in anchor['metrics'].items() if k not in ('leads','primary_score')}
        require(all({k:v for k,v in r['metrics'].items() if k not in ('leads','primary_score')}==counts for r in rows),
                'Event populations changed across seeds')
        leads=[]
        for h in range(18):
            src=[r['metrics']['leads'][h] for r in rows]
            numerical=('physical_mse','rmse','mae','bias','negative_frequency','negative_frequency_among_emitted')
            counts={k:v for k,v in src[0].items() if k not in (*numerical,'negative_predictions_scored','negative_predictions_emitted','observed_peak_forecast')}
            require(all({k:v for k,v in x.items() if k not in (*numerical,'negative_predictions_scored','negative_predictions_emitted','observed_peak_forecast')}==counts for x in src),
                    'Event per-lead populations differ across seeds')
            means={k:math.fsum(x[k]/3 for x in src) if all(x[k] is not None for x in src) else None for k in numerical}
            peaks=[x['observed_peak_forecast'] for x in src]
            peak=None if any(x is None for x in peaks) else dict(observed_q_cms=peaks[0]['observed_q_cms'],
                mean_prediction_q_cms=math.fsum(x['prediction_q_cms']/3 for x in peaks),
                mean_signed_error_cms=math.fsum(x['error_cms']/3 for x in peaks),
                mean_absolute_error_cms=math.fsum(abs(x['error_cms'])/3 for x in peaks))
            leads.append(dict(lead_hours=h+1,shared_counts=counts,mean_metrics=means,mean_observed_peak_forecast=peak,
                seed_metrics={str(s):{k:events[s]['events'][ei]['metrics']['leads'][h][k] for k in (*numerical,'negative_predictions_scored','negative_predictions_emitted','observed_peak_forecast')} for s in SEEDS}))
        result['metrics']=dict(shared_counts={k:v for k,v in anchor['metrics'].items() if k not in ('leads','primary_score')},leads=leads)
        output.append(result)
    return dict(schema='hydra-phase-a-three-seed-fixed-events/v1',events=output,catalog_event_count=len(output),
        metrics_are_means_of_seed_metrics=True,counts_not_summed_across_seeds=True,events_reselected=False,
        new_timing_or_false_alarm_acceptance_rule=False,protected_assessment_access=False)


def _paired(routed,l24):
    cohorts={}
    for name in COHORTS:
        a,b=routed['cohorts'][name],l24['cohorts'][name];leads=[]
        for ar,br in zip(a['leads'],b['leads']):
            require(ar['shared_counts']==br['shared_counts'],'Routed/L24 comparison does not have common emitted target support')
            leads.append(dict(lead_hours=ar['lead_hours'],shared_counts=ar['shared_counts'],
                mean_metric_differences={k:ar['mean_metrics'][k]-br['mean_metrics'][k] if ar['mean_metrics'][k] is not None and br['mean_metrics'][k] is not None else None for k in METRICS}))
        cohorts[name]=dict(primary_difference=a['primary_score']-b['primary_score'] if a['primary_score'] is not None and b['primary_score'] is not None else None,
            seed_primary_differences={str(s):a['seed_primary_scores'][str(s)]-b['seed_primary_scores'][str(s)] if a['seed_primary_scores'][str(s)] is not None and b['seed_primary_scores'][str(s)] is not None else None for s in SEEDS},leads=leads)
    return dict(gauge_id=routed['identity']['gauge_id'],architecture=routed['identity']['architecture'],
        positive_product='ROUTED_CLEAN_168_24_V1',negative_product='L24_INPUT_ONLY_V1',cohorts=cohorts,
        pairing='IDENTICAL_EMITTED_ORIGINS_AND_PER_LEAD_REFERENCE_MASKS',uncertainty_interval_computed=False,
        weather_soil_attribution_claim=False)


def evaluate(plan,store,inputs_receipt,prediction_barrier,expected_barrier_sha256,reconciliation,*,check=lambda:None,read_budget=lambda n:None):
    authenticated=authenticate_prediction_barrier(plan,store,inputs_receipt,prediction_barrier,expected_barrier_sha256,reconciliation,check=check)
    # The full prediction authentication above is deliberately before BOTH of
    # these functions, which are allowed to read verification values/qualifiers.
    catalog=_load_catalog(check,read_budget)
    references,source_receipts=_load_reference_calendars(plan,reconciliation,check=check,read_budget=read_budget)
    targets={};target_manifests={};target_descriptors={};coverage={};products=[];all_reports={};event_reports={};high_reports={};common_checks=[];checks=[]
    records={r['job']['job_id']:r for r in plan['original_records']}
    require(set(records)==set(authenticated['models']),'Original full model record inventory differs')
    for gauge in GAUGE_IDS:
        check();route=authenticated['routes'][gauge];ra=route['arrays'];common=authenticated['common'][gauge]
        target,source,stored=_targets(gauge,route,references[gauge],source_receipts,expected_barrier_sha256,store)
        targets[gauge],target_manifests[gauge],target_descriptors[gauge]=target,source,stored
        long,short=ra['eligible168'],ra['eligible24']
        coverage[gauge]=dict(candidate_origins=ORIGINS,original_common=int(common.sum()),additional_l168=int((long&~common).sum()),
            l24_fallback=int((short&~long).sum()),unavailable=int((~short).sum()),route_counts={k:int(np.count_nonzero(ra['route']==k)) for k in ('L168','L24','UNAVAILABLE')},
            primary_reason_counts={str(k):int(v) for k,v in zip(*np.unique(ra['primary_reason'],return_counts=True))},
            overlapping_input_reasons={k:int(np.count_nonzero(ra[k])) for k in ('no_covering_weather_cycle','incomplete_weather','missing_or_stale_soil','incomplete_soil','unresolved_source_calendar')},
            valid_targets_by_lead=[int(target['label_mask'][:,h].sum()) for h in range(18)],
            availability_by_product={PRODUCTS[0]:int(common.sum()),PRODUCTS[1]:int(long.sum()),PRODUCTS[2]:int(short.sum()),PRODUCTS[3]:int(short.sum())},
            availability_denominator_is_gauge_origin_not_model_seed=True)
        coverage[gauge]['overlapping_input_reasons'].update(
            incomplete_L168_Q=int(np.count_nonzero(~ra['q168_complete'])),
            incomplete_L24_Q=int(np.count_nonzero(~ra['q24_complete'])),
            non_Q_and_recent_Q_incomplete=int(np.count_nonzero(~ra['non_q_eligible'] & ~ra['q24_complete'])))
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                a_id,b_id=_job(gauge,architecture,168,seed),_job(gauge,architecture,24,seed)
                a,b=authenticated['input_only'][a_id],authenticated['input_only'][b_id]
                streams={PRODUCTS[0]:authenticated['clean'][a_id],PRODUCTS[1]:a,PRODUCTS[2]:b,
                         PRODUCTS[3]:authenticated['routed'][_routed(gauge,architecture,seed)]}
                for product,item in streams.items():
                    receipt=item['receipt'];weights={'168':a['receipt']['checkpoint_sha256'],'24':b['receipt']['checkpoint_sha256']}
                    if product!=PRODUCTS[3]:weights={('24' if product==PRODUCTS[2] else '168'):receipt['checkpoint_sha256']}
                    identity=dict(gauge_id=gauge,architecture=architecture,seed=seed,product=product,
                        checkpoint_sha256_by_history=weights,phase_plan_sha256=plan['plan_sha256'],route_map_sha256=route['route_map_sha256'],
                        prediction_sha256=item['artifact_sha256'],prediction_barrier_sha256=expected_barrier_sha256,
                        input_source_sha256=receipt['input_source_identity'],label_source_sha256=source['label_source_sha256'])
                    args=dict(identity=identity,origins=target['origin_time'],forecast_time=target['forecast_time'],labels=target['labels'],
                        label_mask=target['label_mask'],predictions=item['predictions'],original_common=common,
                        eligible168=long,eligible24=short,route=ra['route'])
                    score=score_product(**args);verified=verify_product_report(score,**args)
                    high=high_flow_coverage(**args,catalog_row=catalog[gauge]);events=fixed_events(**args,catalog_row=catalog[gauge])
                    prefix=f'evaluation/{gauge}/{architecture}/seed{seed}/{product}'
                    artifacts=dict(score=store.write(prefix+'/score.json',_json(score),kind='PHASE_A_PRODUCT_SCORES'),
                        verification=store.write(prefix+'/verification.json',_json(verified),kind='PHASE_A_INDEPENDENT_VERIFICATION'),
                        high_flow=store.write(prefix+'/high-flow.json',_json(high),kind='PHASE_A_HIGH_FLOW_COVERAGE'),
                        events=store.write(prefix+'/events.json',_json(events),kind='PHASE_A_FIXED_EVENTS'))
                    products.append(dict(gauge_id=gauge,architecture=architecture,seed=seed,product=product,artifacts=artifacts))
                    key=(gauge,architecture,product);all_reports.setdefault(key,{})[seed]=score;event_reports.setdefault(key,{})[seed]=events;high_reports.setdefault(key,{})[seed]=high
                    checks.append(verified)
                    if product in (PRODUCTS[1],PRODUCTS[2]):common_checks.append(_original_common_check(store,records[a_id if product==PRODUCTS[1] else b_id],score))
                    check()
    aggregates=[];aggregate_values={}
    for key,reports in sorted(all_reports.items()):
        gauge,architecture,product=key;aggregate=aggregate_seed_reports(reports)
        baseline=_shared_high_flow(high_reports[key][17]);require(all(_shared_high_flow(high_reports[key][s])==baseline for s in SEEDS),'Availability changed with seed')
        aggregate['high_flow_coverage']=baseline;aggregate['fixed_events']=_aggregate_events(event_reports[key])
        aggregate_values[key]=aggregate
        descriptor=store.write(f'evaluation/aggregates/{gauge}__{architecture}__{product}.json',_json(aggregate),kind='PHASE_A_THREE_SEED_AGGREGATE')
        aggregates.append(dict(gauge_id=gauge,architecture=architecture,product=product,report=descriptor))
    paired=[]
    for gauge in GAUGE_IDS:
        for architecture in ARCHITECTURES:
            paired.append(_paired(aggregate_values[gauge,architecture,PRODUCTS[3]],aggregate_values[gauge,architecture,PRODUCTS[2]]))
    require(len(products)==144 and len(aggregates)==48 and len(common_checks)==72,'Complete Phase A scoring matrix required')
    result=dict(schema='hydra-phase-a-evaluation-index/v1',status='ALL144_PRODUCTS_SCORED_AND_INDEPENDENTLY_VERIFIED',
        phase_plan_sha256=plan['plan_sha256'],prediction_barrier_sha256=expected_barrier_sha256,
        route_barrier_sha256=prediction_barrier['route_barrier_sha256'],
        original_plan_sha256=next(iter(authenticated['clean'].values()))['receipt']['original_plan_sha256'],
        products=products,aggregates=aggregates,targets_by_gauge=target_descriptors,coverage_by_gauge=coverage,
        paired_routed_vs_l24=paired,original_common_checks=common_checks,
        independent_verification=dict(product_reports=144,lead_panels=sum(x['lead_panels_checked'] for x in checks),
            maximum_absolute_metric_difference=max(x['maximum_absolute_metric_difference'] for x in checks),relative_absolute_tolerance=1e-10),
        prediction_artifacts_authenticated_before_targets=len(authenticated['authenticated_product_artifacts']),
        original_qualifiers_and_source_ids_preserved=True,per_lead_target_masks_independent=True,
        target_quality_never_changes_route=True,targets_filled=False,negative_predictions_clipped=False,
        all_architectures_and_seeds_retained=True,new_uncertainty_families_computed=False,
        routed_weather_soil_attribution_supported=False,model_calls=0,optimizer_updates=0,scaler_fits=0,
        paid_cloud_jobs=0,new_downloads=0,protected_assessment_access=False)
    result['receipt_artifact']=store.write('EVALUATION.json',_json(result),kind='PHASE_A_EVALUATION_RECEIPT')
    check();return result
