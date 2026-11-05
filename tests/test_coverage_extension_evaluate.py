"""Synthetic prediction-barrier and Phase A evaluation orchestration tests."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow import coverage_extension_evaluate as ev
from hydra_directflow.coverage_extension_scoring import START,END,ORIGINS,PRODUCTS,SEEDS
from hydra_directflow.coverage_analysis import select_observed_events,event_coverage,coverage_by_regime,iso
from hydra_directflow.post_events import POPULATION,CATALOG_RULE
from hydra_directflow.reference import ASSESSMENT_START,digest,Observation
from hydra_directflow.trainer_bundle import _array_digest


def js(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def hash_text(text):return sha(text.encode())
def npz(arrays):
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);return stream.getvalue()


class FakeStore:
    def __init__(self):
        self.bodies={};self.original={};self.reads=[];self.writes={};self.events=[];self.labels_opened=False
    def put(self,path,raw):
        self.bodies[path]=raw;return dict(path=path,bytes=len(raw),sha256=sha(raw))
    def read(self,descriptor,maximum):
        self.reads.append(descriptor['path']);raw=self.bodies[descriptor['path']]
        if len(raw)>maximum:raise ValueError('synthetic maximum')
        return raw
    def read_original(self,path,maximum,expected_sha=None):
        assert self.labels_opened,'Original target-derived metrics opened before target barrier'
        raw=self.original[path];assert len(raw)<=maximum and sha(raw)==expected_sha;return raw
    def write(self,path,raw,*,kind):
        json.loads(raw) if path.endswith('.json') else None
        target='attempt-synthetic/'+path
        if target in self.writes:assert self.writes[target]==raw
        self.writes[target]=raw;return self.put(target,raw)
    def check(self):pass
    def read_budget(self,n):assert n>=0


def baseline_panel(y,p,mask,forecast,identity):
    rows=[]
    for h in range(18):
        truth=y[mask,h];pred=p[mask,h];err=pred-truth;n=len(err);den=float(np.sum((truth-np.mean(truth))**2))
        rows.append(dict(lead_hours=h+1,n=n,n_requested=n,distinct_target_hours=n,
            distinct_utc_dates=len(np.unique(forecast[mask,h]//86400)),negative_predictions_scored=int(np.count_nonzero(pred<0)),
            physical_mse=float(np.mean(err*err)),rmse=float(np.sqrt(np.mean(err*err))),mae=float(np.mean(np.abs(err))),
            bias=float(np.mean(err)),nse=1-float(np.sum(err*err))/den,negative_frequency=float(np.count_nonzero(pred<0)/n)))
    return dict(identity=identity,leads=rows,primary_score=math.fsum(x['nse']/18 for x in rows))


class EvaluationFixture:
    """All data are manufactured once; no study source/model/checkpoint read."""
    def __init__(self):
        self.store=FakeStore();self.phase='a'*64;self.original_phase='b'*64;self.candidate='c'*64;self.route_barrier=None
        self.origins=np.arange(START,END,3600,dtype=np.int64);self.future=self.origins[:,None]+np.arange(1,19,dtype=np.int64)*3600
        self.positions=(self.future-START)//3600;self.routes={};self.refs={};self.catalog={};self.arrays_by_path={};self.source_receipts=[]
        self.inputs=dict(schema='hydra-phase-a-prepared-inputs/v1',status='INPUTS_AND_ROUTES_COMMITTED',phase_plan_sha256=self.phase,
            candidate_source_identity=self.candidate,target_values_read=False,target_masks_read=False,protected_assessment_access=False,
            routes={},bundles={})
        self.reconciliation=dict(models=[],original_common_case_ids_by_gauge={},reference_month_identities=[])
        self.records=[];self.clean={};self.input={};self.routed={};self.buffers={};self.product_paths=set()
        for gi,gauge in enumerate(ev.GAUGE_IDS):self.make_gauge(gi,gauge)
        for g in ev.GAUGE_IDS:
            for a in ev.ARCHITECTURES:
                for s in SEEDS:
                    for h in (168,24):self.make_model(g,a,h,s)
        clean_map={k:digest(json.loads(self.store.bodies[v['receipt']['path']])) for k,v in self.clean.items()}
        self.clean_barrier_sha=digest(dict(schema='hydra-coverage-extension-clean-barrier/v1',status='ALL72_CLEAN_REGRESSIONS_BYTE_EXACT',phase_plan_sha256=self.phase,clean_receipt_sha256=clean_map))
        for key,entry in self.input.items():
            body=json.loads(self.store.bodies[entry['receipt']['path']]);body['clean_barrier_sha256']=self.clean_barrier_sha
            self.input[key]['receipt']=self.store.put(entry['receipt']['path'],js(body))
        for g in ev.GAUGE_IDS:
            for a in ev.ARCHITECTURES:
                for s in SEEDS:self.make_routed(g,a,s)
        route_barrier=dict(schema='hydra-phase-a-input-route-barrier/v1',status='ALL3_INPUT_ONLY_ROUTES_COMMITTED_BEFORE_REGRESSION_SELECTION',
            phase_plan_sha256=self.phase,routes=self.inputs['routes'],original_common_case_ids_consulted=False,target_values_read=False,target_masks_read=False)
        self.inputs['route_barrier']=self.store.put('routes/barrier.json',js(route_barrier));self.inputs['route_barrier_content_sha256']=digest(route_barrier)
        input_body=deepcopy(self.inputs);self.inputs['receipt_artifact']=self.store.put('inputs/receipt.json',js(input_body))
        self.barrier=dict(schema='hydra-coverage-extension-prediction-barrier/v1',status='ALL_REQUESTED_PREDICTIONS_COMMITTED_BEFORE_TARGETS',
            phase_plan_sha256=self.phase,route_barrier_sha256=digest(route_barrier),clean_barrier_sha256=self.clean_barrier_sha,
            input_preparation_receipt=self.inputs['receipt_artifact'],input_preparation_content_sha256=digest(input_body),
            clean_products=self.clean,input_products=self.input,routed_products=self.routed)
        self.plan=dict(plan_sha256=self.phase,original_plan_path='SYNTHETIC_NOT_OPENED',original_run_directory='SYNTHETIC_NOT_OPENED',
            reference_cache_root='SYNTHETIC_NOT_OPENED',protected_assessment_access=False,original_records=self.records)

    def make_gauge(self,gi,gauge):
        time=np.arange(START,ASSESSMENT_START,3600,dtype=np.int64)
        latent=np.asarray(3.+.1*np.sin(np.arange(len(time))/24),np.float64)
        latent[100]=15.;latent[300]=20.;latent[500]=12.
        raw=latent.copy();admitted=np.ones(len(time),bool);admitted[700]=False
        flags=np.full(len(time),'["A"]',dtype='<U32');flags[700]='["P"]'
        values=np.where(admitted,raw,np.nan);present=np.ones(len(time),bool)
        reference=dict(times=time,values=values,admitted=admitted,quality_flags_json=flags,source_id=np.full(len(time),'SYNTHETIC_SOURCE'),
            row_present=present,original_numeric_values=raw,original_numeric_present=present)
        self.refs[gauge]=reference
        long=np.ones(ORIGINS,bool);short=np.ones(ORIGINS,bool);long[250:341]=False;long[480:501]=False;short[480:501]=False
        common=long&admitted[self.positions].all(axis=1);route=np.where(long,'L168',np.where(short,'L24','UNAVAILABLE'))
        ids=np.array([gauge+':'+iso(int(x)) for x in self.origins],dtype='<U29')
        arr=dict(origin_time=self.origins,forecast_time=self.future,case_id=ids,site_id=np.full(ORIGINS,gi,np.int64),
            eligible168=long,eligible24=short,q168_complete=long.copy(),q24_complete=short.copy(),non_q_eligible=np.ones(ORIGINS,bool),
            route=route,primary_reason=np.where(~short,'RECENT_Q_HISTORY_INCOMPLETE',np.where(~long,'OLDER_Q_HISTORY_GAP_USE_L24','COMPLETE_L168')))
        for key in ('no_covering_weather_cycle','incomplete_weather','missing_or_stale_soil','incomplete_soil','unresolved_source_calendar'):arr[key]=np.zeros(ORIGINS,bool)
        route_hash=hash_text(gauge+'route');self.routes[gauge]=dict(arrays=arr,manifest={'phase_plan_sha256':self.phase},route_map_sha256=route_hash)
        self.inputs['routes'][gauge]=dict(gauge_id=gauge,route_map_sha256=route_hash)
        self.inputs['bundles'][gauge]={str(h):{kind:dict(source_identity=hash_text(f'{gauge}-{h}-{kind}'),cohort_admission_digest=hash_text(f'{gauge}-{h}-{kind}-cohort')) for kind in ('clean','expanded')} for h in (168,24)}
        self.reconciliation['original_common_case_ids_by_gauge'][gauge]=ids[common].tolist()
        for h in (168,24):
            full=latent[self.positions]+(2. if h==168 else 4.)
            for kind,mask in [('clean',common),('expanded',long if h==168 else short)]:
                arrays={k:arr[k][mask].copy() for k in ('case_id','origin_time','forecast_time','site_id')};arrays['prediction_q']=full[mask].copy()
                self.buffers[gauge,h,kind]=(npz(arrays),_array_digest(arrays),arrays)
        full168=latent[self.positions]+2.;full24=latent[self.positions]+4.;p=np.full_like(full168,np.nan);p[long]=full168[long];p[short&~long]=full24[short&~long]
        arrays={k:arr[k].copy() for k in ('case_id','origin_time','forecast_time','site_id')};arrays.update(prediction_q=p,
            route_history_hours=np.where(long,168,np.where(short,24,0)).astype(np.int16),route_checkpoint_index=np.where(long,0,np.where(short,1,-1)).astype(np.int8))
        self.buffers[gauge,'routed']=(npz(arrays),_array_digest(arrays),arrays)
        thresholds=dict(q10_cms=1.,q90_cms=6.,q95_cms=10.,fit_start_utc='2018-07-12T00:00:00Z',fit_end_exclusive_utc='2023-05-01T00:00:00Z',
            refit_at_utc=iso(START),fit_population=POPULATION,stopping_or_outer_values_used=False,model_scaler=False,simulated_label_delay_hours=2,quantile_method='numpy.linear')
        events=select_observed_events(reference,thresholds,start=START,end=ASSESSMENT_START)
        for event in events:
            cov=event_coverage(event,self.origins,long,common,admitted[self.positions],outer_start=START,outer_end=ASSESSMENT_START)
            event.update(all_lead_requested_cells=sum(x['requested_cells'] for x in cov),all_lead_common_comparison_cells=sum(x['common_comparison_cells'] for x in cov),
                         observed_peak_supported_leads=[x['lead_hours'] for x in cov if x['observed_peak_has_common_prediction_case']])
            if event['representative_top3']:event['forecast_case_coverage_by_lead']=cov
        self.catalog[gauge]=dict(gauge_id=gauge,fold='final_development',thresholds=thresholds,
            outer_window=dict(start_utc=iso(START),end_exclusive_utc=iso(ASSESSMENT_START)),representative_event_rule=CATALOG_RULE,
            event_selection_uses_model_errors=False,event_windows_clipped_to_outer=True,event_windows_hours=[-48,72],
            observed_high_flow_event_count=len(events),observed_high_flow_events=events,
            target_cell_regime_coverage=coverage_by_regime(self.origins,long,common,admitted[self.positions],reference,thresholds,start=START,end=ASSESSMENT_START))

    def make_model(self,g,a,h,s):
        job_id=ev._job(g,a,h,s);cp=dict(path=job_id+'/checkpoint.pt',bytes=128,sha256=hash_text(job_id+'checkpoint'))
        pred=dict(path=job_id+'/original.npz',bytes=128,sha256=hash_text(job_id+'original'))
        common=np.isin(self.routes[g]['arrays']['case_id'],np.asarray(self.reconciliation['original_common_case_ids_by_gauge'][g]))
        model=dict(job_id=job_id,gauge_id=g,architecture=a,seed=s,history_hours=h,device='cpu' if a in ('LSTM','GRU_Transformer') else 'mps',
                   checkpoint=cp,selected_best_epoch=2,gradient_case_ids_sha256=hash_text(g+'gradient'),original_predictions=pred)
        self.reconciliation['models'].append(model)
        metrics_path=job_id+'/metrics.json';latent=self.refs[g]['original_numeric_values'][self.positions];full=latent+(2 if h==168 else 4)
        panel=baseline_panel(latent,full,common,self.future,dict(gauge_id=g,architecture=a,seed=s,history_hours=h))
        metrics=js(dict(panel=panel,checkpoint_sha256=cp['sha256'],predictions_sha256=pred['sha256']));self.store.original[metrics_path]=metrics
        old_receipt=dict(plan_sha256=self.original_phase,artifacts={'checkpoint':cp,'metrics':dict(path=metrics_path,bytes=len(metrics),sha256=sha(metrics))},
                         hashes={'checkpoint':cp['sha256'],'predictions':pred['sha256'],'metrics':sha(metrics)})
        self.records.append(dict(job=dict(job_id=job_id,gauge_id=g,architecture=a,seed=s,history_hours=h),receipt=old_receipt))
        for kind,table in [('clean',self.clean),('input',self.input)]:
            bkind='clean' if kind=='clean' else 'expanded';raw,arrsha,arrays=self.buffers[g,h,bkind];bundle=self.inputs['bundles'][g][str(h)][bkind]
            array_path=f'{kind}/{job_id}.npz';array_desc=self.store.put(array_path,raw);self.product_paths.add(array_path)
            body=dict(schema='hydra-coverage-extension-clean-regression/v1' if kind=='clean' else 'hydra-coverage-extension-input-only-predictions/v1',
                status='BYTE_EXACT_ORIGINAL_PREDICTIONS' if kind=='clean' else 'ALL_REQUESTED_VECTORS_FINITE',phase_plan_sha256=self.phase,
                original_plan_sha256=self.original_phase,job_id=job_id,gauge_id=g,architecture=a,seed=s,history_hours=h,device=model['device'],
                checkpoint_sha256=cp['sha256'],selected_best_epoch=2,optimizer_updates=0,scaler_fits=0,target_values_read=False,
                target_masks_read=False,protected_assessment_access=False,posthoc_clipping=False,
                input_source_identity=bundle['source_identity'],input_cohort_admission_digest=bundle['cohort_admission_digest'],
                candidate_source_identity=self.candidate,saved_transform_sha256=hash_text(job_id+'transform'),
                predictions_array_sha256=arrsha,cases=len(arrays['origin_time']),case_ids_sha256=digest(arrays['case_id'].tolist()))
            if kind=='clean':body.update(comparison='DTYPE_SHAPE_C_ORDER_BYTES',tolerance=None,training_case_ids_sha256=model['gradient_case_ids_sha256'],original_prediction_sha256=pred['sha256'])
            else:body.update(product='L168_INPUT_ONLY_V1' if h==168 else 'L24_INPUT_ONLY_V1',clean_barrier_sha256='0'*64,
                              clean_regression_receipt_sha256=digest(json.loads(self.store.bodies[self.clean[job_id]['receipt']['path']])))
            table[job_id]=dict(arrays=array_desc,receipt=self.store.put(f'{kind}/{job_id}.json',js(body)))

    def make_routed(self,g,a,s):
        key=ev._routed(g,a,s);raw,arrsha,arrays=self.buffers[g,'routed'];path='routed/'+key+'.npz';self.product_paths.add(path)
        weights={str(h):hash_text(ev._job(g,a,h,s)+'checkpoint') for h in (168,24)}
        body=dict(schema='hydra-coverage-extension-routed-predictions/v1',status='ALL_INPUT_ELIGIBLE_ROUTED_VECTORS_COMMITTED',
            phase_plan_sha256=self.phase,product='ROUTED_CLEAN_168_24_V1',gauge_id=g,architecture=a,seed=s,
            route_map_sha256=self.routes[g]['route_map_sha256'],checkpoint_sha256_by_history=weights,
            predictions_array_sha256=arrsha,input_source_identity=hash_text(g+'mixed'),cases=ORIGINS,
            target_values_read=False,target_masks_read=False,protected_assessment_access=False,model_calls=0,optimizer_updates=0,scaler_fits=0)
        self.routed[key]=dict(arrays=self.store.put(path,raw),receipt=self.store.put('routed/'+key+'.json',js(body)))

    def labels(self,*args,**kw):
        assert self.product_paths<=set(self.store.reads),'Target loader called before complete product authentication'
        self.store.labels_opened=True
        return self.refs,self.source_receipts

    def route(self,store,descriptor):return self.routes[descriptor['gauge_id']]


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base=EvaluationFixture()

    def setUp(self):self.f=deepcopy(self.base)

    def call(self):
        f=self.f
        with patch.object(ev,'read_route',side_effect=f.route),patch.object(ev,'_load_reference_calendars',side_effect=f.labels) as loader, \
             patch.object(ev,'_load_catalog',return_value=f.catalog):
            result=ev.evaluate(f.plan,f.store,f.inputs,f.barrier,digest(f.barrier),f.reconciliation)
        return result,loader

    def preflight(self):
        f=self.f
        with patch.object(ev,'read_route',side_effect=f.route):
            return ev.authenticate_prediction_barrier(f.plan,f.store,f.inputs,f.barrier,digest(f.barrier),f.reconciliation)

    def test_missing_or_changed_barrier_never_opens_targets_or_catalog(self):
        for change in ('missing','status','phase'):
            f=deepcopy(self.base)
            if change=='missing':f.barrier['routed_products'].pop(next(iter(f.barrier['routed_products'])))
            elif change=='status':f.barrier['status']='PENDING'
            else:f.barrier['phase_plan_sha256']='0'*64
            with patch.object(ev,'_load_reference_calendars',side_effect=AssertionError('Targets opened')), \
                 patch.object(ev,'_load_catalog',side_effect=AssertionError('Catalog opened')),self.assertRaises(ValueError):
                ev.evaluate(f.plan,f.store,f.inputs,f.barrier,digest(f.barrier),f.reconciliation)

    def test_all180_products_authenticated_without_reference_values(self):
        out=self.preflight();self.assertEqual(len(out['authenticated_product_artifacts']),180)
        self.assertFalse(self.f.store.labels_opened);self.assertEqual(len(out['clean']),72);self.assertEqual(len(out['routed']),36)

    def test_changed_input_preparation_body_cannot_pass_barrier(self):
        self.f.store.bodies[self.f.inputs['receipt_artifact']['path']]=b'{}'
        with self.assertRaisesRegex(ValueError,'committed bytes'):self.preflight()

    def test_global_clean_barrier_is_reconstructed_not_trusted_from_matching_claims(self):
        bad='f'*64;self.f.barrier['clean_barrier_sha256']=bad
        for entry in self.f.input.values():
            body=json.loads(self.f.store.bodies[entry['receipt']['path']]);body['clean_barrier_sha256']=bad
            entry['receipt']=self.f.store.put(entry['receipt']['path'],js(body))
        with self.assertRaisesRegex(ValueError,'Global clean barrier'):
            self.preflight()
        self.assertFalse(any(path.startswith('input/') for path in self.f.store.reads))
        self.assertFalse(self.f.store.labels_opened)

    def test_changed_product_bytes_or_wrong_checkpoint_fails_before_targets(self):
        key=next(iter(self.f.input));entry=self.f.input[key];self.f.store.bodies[entry['arrays']['path']]=b'changed'
        with self.assertRaisesRegex(ValueError,'committed bytes'):self.preflight()
        self.f=deepcopy(self.base);entry=self.f.input[key];body=json.loads(self.f.store.bodies[entry['receipt']['path']]);body['checkpoint_sha256']='0'*64
        entry['receipt']=self.f.store.put(entry['receipt']['path'],js(body))
        with self.assertRaisesRegex(ValueError,'identity/scope'):self.preflight()
        self.assertFalse(self.f.store.labels_opened)

    def test_resigned_routed_wrong_model_vector_is_detected(self):
        key=next(iter(self.f.routed));entry=self.f.routed[key];g=key.split('__')[0]
        arrays=deepcopy(self.f.buffers[g,'routed'][2]);arrays['prediction_q'][0,0]+=1.
        raw=npz(arrays);entry['arrays']=self.f.store.put(entry['arrays']['path'],raw)
        body=json.loads(self.f.store.bodies[entry['receipt']['path']]);body['predictions_array_sha256']=_array_digest(arrays)
        entry['receipt']=self.f.store.put(entry['receipt']['path'],js(body))
        with self.assertRaisesRegex(ValueError,'whole H18 checkpoint vectors'):self.preflight()

    def test_routed_history_index_and_unavailable_values_cannot_be_forged(self):
        for change in ('index','unavailable'):
            self.f=deepcopy(self.base);key=next(iter(self.f.routed));entry=self.f.routed[key];g=key.split('__')[0];arrays=deepcopy(self.f.buffers[g,'routed'][2])
            if change=='index':arrays['route_checkpoint_index'][0]=1
            else:arrays['prediction_q'][480]=0.
            entry['arrays']=self.f.store.put(entry['arrays']['path'],npz(arrays));body=json.loads(self.f.store.bodies[entry['receipt']['path']]);body['predictions_array_sha256']=_array_digest(arrays);entry['receipt']=self.f.store.put(entry['receipt']['path'],js(body))
            with self.subTest(change=change),self.assertRaises(ValueError):self.preflight()

    def test_complete_synthetic_matrix_scores_targets_after_barrier_and_keeps_all_seeds(self):
        result,loader=self.call();loader.assert_called_once()
        self.assertEqual(result['status'],'ALL144_PRODUCTS_SCORED_AND_INDEPENDENTLY_VERIFIED')
        self.assertEqual(len(result['products']),144);self.assertEqual(len(result['aggregates']),48)
        self.assertEqual(result['prediction_artifacts_authenticated_before_targets'],180)
        self.assertEqual(result['independent_verification']['lead_panels'],144*90)
        self.assertEqual(len(result['original_common_checks']),72);self.assertEqual(len(result['paired_routed_vs_l24']),12)
        self.assertFalse(result['routed_weather_soil_attribution_supported']);self.assertEqual(result['model_calls'],0)
        json.dumps(result,allow_nan=False)
        for row in result['aggregates']:
            data=json.loads(self.f.store.bodies[row['report']['path']]);self.assertEqual(data['seeds'],[17,29,43])
            self.assertIsNone(data['cohorts']['UNAVAILABLE']['primary_score'])
        sample=result['coverage_by_gauge']['03161000'];self.assertEqual(sum(sample['route_counts'].values()),ORIGINS)
        self.assertEqual(sample['availability_by_product']['L24_INPUT_ONLY_V1'],sample['availability_by_product']['ROUTED_CLEAN_168_24_V1'])
        self.assertGreater(sample['additional_l168'],0)
        self.assertEqual(sample['overlapping_input_reasons']['incomplete_L24_Q'],21)
        self.assertEqual(sample['overlapping_input_reasons']['non_Q_and_recent_Q_incomplete'],0)

    def test_original_common_metric_mismatch_is_not_hidden(self):
        record=self.f.records[0];desc=record['receipt']['artifacts']['metrics'];body=json.loads(self.f.store.original[desc['path']]);body['panel']['leads'][0]['rmse']+=1.
        raw=js(body);self.f.store.original[desc['path']]=raw;desc.update(bytes=len(raw),sha256=sha(raw));record['receipt']['hashes']['metrics']=sha(raw)
        with self.assertRaisesRegex(ValueError,'Original common metric'):self.call()


class ReferenceBoundaryTests(unittest.TestCase):
    def test_exact13_month_scope_and_original_rejected_numeric_are_retained(self):
        months=['2023-'+f'{x:02d}' for x in range(7,13)]+['2024-'+f'{x:02d}' for x in range(1,8)]
        identities=[dict(month=x,cache_key=hash_text(x),commit_sha256=hash_text(x+'commit')) for x in months]
        actual_helper=ev._reference_helpers();called=[]
        observation=Observation(site='03161000',valid_at=START+3600,available_at=START+10800,
            value=7.,quality_flags=('P',),source_id='SYNTHETIC_REFERENCE')
        def checked(cache,identity,guard):
            called.append(identity['month'])
            return ([observation] if identity['month']=='2023-07' else []),dict(identity=identity)
        fake=SimpleNamespace(checked_reference_month=checked,build_reference_calendars=actual_helper.build_reference_calendars)
        with tempfile.TemporaryDirectory() as directory,patch.object(ev,'_reference_helpers',return_value=fake):
            refs,receipt=ev._load_reference_calendars({'reference_cache_root':str(Path(directory).resolve())},
                {'reference_month_identities':identities},check=lambda:None,read_budget=lambda n:None)
        self.assertEqual(called,months);self.assertEqual(len(receipt),13)
        ref=refs['03161000'];self.assertTrue(np.isnan(ref['values'][1]));self.assertEqual(ref['original_numeric_values'][1],7.)
        self.assertEqual(ref['quality_flags_json'][1],'["P"]');self.assertFalse(ref['admitted'][1])
        self.assertTrue(ref['row_present'][1]);self.assertTrue(ref['original_numeric_present'][1])

    def test_missing_reference_month_fails_before_any_value_helper(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(ev,'_reference_helpers') as helper:
            with self.assertRaisesRegex(ValueError,'Exactly13'):
                ev._load_reference_calendars({'reference_cache_root':str(Path(directory).resolve())},
                    {'reference_month_identities':[]},check=lambda:None,read_budget=lambda n:None)
            helper.return_value.checked_reference_month.assert_not_called()

    def test_reference_read_shim_refuses_outside_owned_root_before_read_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();owned=root/'owned';owned.mkdir();outside=root/'outside.json';outside.write_bytes(b'{}');charges=[]
            guard=ev._ReferenceGuard(owned,lambda:None,charges.append)
            with self.assertRaisesRegex(ValueError,'escaped approved'):
                guard.read(outside,1024)
            self.assertEqual(charges,[])


if __name__=='__main__':unittest.main()
