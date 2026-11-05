"""Synthetic Phase A controller plumbing/gates; no real models or source values."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow.campaign_ledger import digest
from hydra_directflow.coverage_extension_ledger import LIMITS,CHECKPOINT_IDS,ExtensionAllowanceExhausted
from hydra_directflow.local_data import iso,GAUGES
from hydra_directflow.training import array_identity

PATH=Path(__file__).resolve().parents[1]/'scripts/run_coverage_extension.py'
SPEC=importlib.util.spec_from_file_location('phase_a_controller_under_test',PATH)
controller=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(controller)
MODELS=('LSTM','Transformer','GRU_Transformer','Mamba2_Transformer')


class Store:
    def __init__(self,plan):
        self.plan=plan;self.run=Path(plan['run_directory']);self.original_run=Path(plan['original_run_directory'])
        self.blobs={};self.reads=[];self.writes=[];self.fail_read=False
    def write(self,name,raw,*,kind):
        path='synthetic/'+name
        if path in self.blobs:raise ValueError('Synthetic test attempted duplicate write')
        self.blobs[path]=raw;self.writes.append((name,kind))
        return dict(path=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    def read(self,d,maximum):
        if self.fail_read:raise AssertionError('Unrelated artifact was re-read')
        self.reads.append(d['path']);raw=self.blobs[d['path']]
        assert len(raw)==d['bytes'] and hashlib.sha256(raw).hexdigest()==d['sha256'] and len(raw)<=maximum
        return raw
    def check(self):pass
    def read_budget(self,n):pass
    def precharge_forward(self,**kw):raise AssertionError('No model forward in controller tests')
    def confirm_forward(self,*args):raise AssertionError('No model forward in controller tests')
    def precharge_model(self,**kw):raise AssertionError('No model construction in controller tests')
    def confirm_model(self,*args):raise AssertionError('No model construction in controller tests')


class Ledger:
    def __init__(self,plan):self.plan=plan;self.tasks={};self.events=[];self.all_complete=False
    def put(self,key,data,*,state='COMPLETED',elapsed=1.):
        receipt=dict(phase_plan_sha256=self.plan['plan_sha256'],task_key=key,status='VERIFIED_PHASE_A_TASK',data=data,elapsed_seconds=elapsed)
        self.tasks[key]=dict(task_key=key,stage=key,state=state,receipt=receipt,receipt_sha=digest(receipt))
    def task(self,key):
        if key in self.tasks:return self.tasks[key]
        return {'state':'COMPLETED','task_key':key} if self.all_complete else None
    @contextmanager
    def writer(self):yield self
    def initialize(self,*args,**kw):self.events.append(('initialize',))
    def reserve_read(self,*args,**kw):self.events.append(('read_reserve',args,kw));return 1
    def snapshot(self):
        return dict(deadline=100000.,attempts=[],tasks=list(self.tasks.values()),entered_cases=0,confirmed_cases=0,
                    entered_batches=0,confirmed_batches=0,entered_models=0,confirmed_models=0,
                    read_reserved=0,output_reserved=0,infrastructure_resumes=0)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        (controller.ROOT/'.local').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=controller.ROOT/'.local');self.root=Path(self.temp.name).resolve()
        self.original=self.root/'original';self.original.mkdir();self.run=self.original/'coverage-extension-v1'
        records=[]
        for seed in (17,29,43):
            for gauge in GAUGES:
                for model in MODELS:
                    for history in (168,24):
                        job=dict(job_id=f'final_development__{gauge}__{model}__full_L{history}__seed{seed}',gauge_id=gauge,
                                 architecture=model,seed=seed,history_hours=history,device='cpu' if model in ('LSTM','GRU_Transformer') else 'mps',
                                 fold='final_development',input_arm='full',order=len(records))
                        cp=hashlib.sha256(job['job_id'].encode()).hexdigest()
                        records.append(dict(job=job,receipt=dict(hashes={'checkpoint':cp}),receipt_sha256='b'*64))
        self.old=dict(plan_sha256='a'*64,protocol_sha256='b'*64,run_directory=str(self.original),ledger_directory=str(self.original/'ledger'),
                      corpus_root=str(self.root/'unopened-corpus'),protocol_path=str(self.root/'old-protocol.json'),
                      snapshot_sha256='e'*64,source_plan_sha256='f'*64,limits={'max_output_bytes':8*1024**3})
        self.protocol=self.root/'protocol.json';self.protocol.write_bytes(controller.encoded(dict(synthetic=True,
            primary_plan_sha256=self.old['plan_sha256'],primary_protocol_sha256=self.old['protocol_sha256'],
            snapshot_sha256=self.old['snapshot_sha256'],source_plan_sha256=self.old['source_plan_sha256'])))
        self.plan=dict(schema='SYNTHETIC_PHASE_A',adopted_plan_sha256=controller.ADOPTED_SHA,protocol_path=str(self.protocol),
            protocol_sha256=controller.sha(self.protocol.read_bytes()),original_plan_path=str(self.root/'old-plan.json'),
            original_plan_sha256=self.old['plan_sha256'],original_protocol_sha256=self.old['protocol_sha256'],
            original_run_directory=str(self.original),original_ledger_directory=self.old['ledger_directory'],
            original_campaign_deadline=100000.,original_training_updates=1151690,original_output_reserved=4000000000,
            run_directory=str(self.run),ledger_directory=str(self.run/'ledger'),corpus_root=self.old['corpus_root'],
            reference_cache_root=str(Path.home()/'.codex/hydra-directflow-runtime/postprocessing-usgs-reference-v1'),
            authentication_read_reserve_bytes=controller.AUTH_READ_RESERVE,initial_metadata_read_reservation=3*controller.AUTH_READ_RESERVE,
            original_records=records,checkpoint_sha256_by_job={r['job']['job_id']:r['receipt']['hashes']['checkpoint'] for r in records},
            limits=deepcopy(LIMITS),code_sha256={'synthetic.py':'c'*64},paid_cloud_jobs=0,protected_assessment_access=False,new_training_authorized=False)
        self.sign(self.plan);self.store=Store(self.plan);self.ledger=Ledger(self.plan)
        self.ids=tuple('03161000:'+iso(controller.START-3600*i) for i in (4,3,2));gsha=digest(list(self.ids))
        membership=self.store.write('gradient.json',controller.encoded(dict(case_ids=list(self.ids),case_ids_sha256=gsha)),kind='SYNTHETIC')
        self.reconciliation=dict(models=[dict(job_id=r['job']['job_id'],gradient_case_ids_sha256=gsha,training_values_sha256='8'*64,
            original_clean_input={'sha256':'9'*64}) for r in records],gradient_membership_artifacts={gsha:membership})
        self.inputs=dict(phase_plan_sha256=self.plan['plan_sha256'],route_barrier_content_sha256='d'*64,
            bundles={g:{str(h):dict(clean={'kind':'clean','cases':2},expanded={'kind':'expanded','cases':3}) for h in (168,24)} for g in GAUGES},
            routes={g:dict(route_map_sha256='e'*64,candidate_source_identity='f'*64) for g in GAUGES})
        self.inputs['receipt_artifact']=self.store.write('PREPARED_INPUTS.json',controller.encoded(self.inputs),kind='SYNTHETIC')
    def tearDown(self):self.temp.cleanup()
    @staticmethod
    def sign(p):p['plan_sha256']=digest({k:v for k,v in p.items() if k!='plan_sha256'});return p
    def result(self,job,*,clean=True,half_hour=False):
        times=controller.START+np.arange(2)*3600+(1800 if half_hour else 0)
        arrays=dict(origin_time=times.astype('<i8'),forecast_time=(times[:,None]+np.arange(1,19)*3600).astype('<i8'),
            site_id=np.full(2,GAUGES.index(job['gauge_id']),dtype='<i8'),case_id=np.array([job['gauge_id']+':'+iso(int(t)) for t in times],dtype='<U29'),
            prediction_q=np.ones((2,18),np.float64))
        receipt={k:job[k] for k in ('job_id','gauge_id','architecture','seed','history_hours','device')}
        receipt.update(schema='SYNTHETIC',status='BYTE_EXACT_ORIGINAL_PREDICTIONS' if clean else 'ALL_REQUESTED_VECTORS_FINITE',
            phase_plan_sha256=self.plan['plan_sha256'],cases=2,predictions_array_sha256=array_identity(arrays),
            checkpoint_sha256=self.plan['checkpoint_sha256_by_job'][job['job_id']],saved_transform_sha256='7'*64,
            original_batch_size=64,tolerance=None,optimizer_updates=0,scaler_fits=0,target_values_read=False,target_masks_read=False,
            protected_assessment_access=False,case_ids_sha256=digest(arrays['case_id'].tolist()),
            accounting=dict(case_checkpoint_evaluations_confirmed=2,forward_wall_seconds=.01),
            saved_predictor_receipt={'training_values_sha256':'8'*64})
        return dict(arrays=arrays,receipt=receipt)
    def persist(self,name,result):
        return dict(arrays=self.store.write(name+'.npz',controller.npz(result['arrays']),kind='SYNTHETIC'),
                    receipt=self.store.write(name+'.json',controller.encoded(result['receipt']),kind='SYNTHETIC'))
    @contextmanager
    def prior(self):
        with patch.object(controller,'load_reconciliation',return_value=self.reconciliation), \
             patch.object(controller,'input_receipt',return_value=self.inputs):yield
    def all_clean(self):
        for record in self.plan['original_records']:
            job=record['job'];self.ledger.put('CLEAN__'+job['job_id'],self.persist(job['job_id'],self.result(job)))
    def profile(self):
        d=self.store.write('PROFILE.json',controller.encoded(dict(status='FIXED_REAL_CLEAN_PROFILE_PASSED',phase_plan_sha256=self.plan['plan_sha256'])),kind='SYNTHETIC')
        self.ledger.put('CLEAN_PROFILE',{'profile':d});return d

    def test_phase_calendar_constants_are_exact_h18_safe_final_year(self):
        self.assertEqual(datetime.fromtimestamp(controller.START,timezone.utc).isoformat(),'2023-07-31T00:00:00+00:00')
        self.assertEqual(datetime.fromtimestamp(controller.END,timezone.utc).isoformat(),'2024-07-30T06:00:00+00:00')
        self.assertEqual((controller.END-controller.START)//3600,8766)

    def test_completed_task_lookup_does_not_reread_unrelated_input_payloads(self):
        item=self.store.write('input.bin',b'synthetic',kind='SYNTHETIC');self.ledger.put('A2_INPUTS',dict(artifact=item))
        self.store.fail_read=True;self.assertEqual(controller.completed(self.store,self.ledger,'A2_INPUTS'),{'artifact':item})
        self.store.fail_read=False;controller.completed(self.store,self.ledger,'A2_INPUTS',verify_all=True)
        self.assertEqual(self.store.reads,[item['path']])
        self.ledger.tasks['A2_INPUTS']['receipt_sha']='0'*64
        with self.assertRaises(ValueError):controller.completed(self.store,self.ledger,'A2_INPUTS')

    def test_clean_branch_plumbs_record_gradient_ids_callbacks_and_serialized_result(self):
        job=self.plan['original_records'][0]['job'];result=self.result(job)
        with self.prior(),patch('hydra_directflow.coverage_extension_io.read_input_bundle',return_value={'synthetic_bundle':True}), \
             patch('hydra_directflow.coverage_extension_inference.clean_regression',return_value=result) as infer:
            output=controller.handle(self.plan,self.old,self.ledger,self.store,'CLEAN',job['job_id'])
        args=infer.call_args.args;self.assertEqual(args[3],self.ids);self.assertEqual(args[1]['original_input_manifest_sha256'],'9'*64)
        self.assertEqual(infer.call_args.kwargs['phase_plan_sha256'],self.plan['plan_sha256'])
        restored=controller.stored_result(self.store,output,job);np.testing.assert_array_equal(restored['arrays']['prediction_q'],result['arrays']['prediction_q'])

    def test_clean_barrier_requires_every_original_job_complete(self):
        self.all_clean();missing=self.plan['original_records'][-1]['job']['job_id'];self.ledger.tasks['CLEAN__'+missing]['state']='INTERRUPTED'
        with self.prior(),self.assertRaisesRegex(ValueError,'incomplete'):
            controller.handle(self.plan,self.old,self.ledger,self.store,'CLEAN_BARRIER')
        self.assertNotIn(('CLEAN_BARRIER.json','CLEAN_BARRIER'),self.store.writes)

    def test_clean_barrier_checks_inner_status_and_complete72_inventory(self):
        self.all_clean();first=self.plan['original_records'][0]['job'];value=self.result(first);value['receipt']['status']='NOT_QUALIFIED'
        self.ledger.put('CLEAN__'+first['job_id'],self.persist('bad-clean',value))
        with self.prior(),self.assertRaisesRegex(ValueError,'Clean regression'):
            controller.handle(self.plan,self.old,self.ledger,self.store,'CLEAN_BARRIER')
        self.ledger.put('CLEAN__'+first['job_id'],self.persist('fixed-clean',self.result(first)))
        shortened={**self.plan,'original_records':self.plan['original_records'][:-1]}
        with self.prior(),self.assertRaises(ValueError):controller.handle(shortened,self.old,self.ledger,self.store,'CLEAN_BARRIER')

    def test_valid_clean_barrier_binds_all72_exact_receipt_digests(self):
        self.all_clean()
        with self.prior():out=controller.handle(self.plan,self.old,self.ledger,self.store,'CLEAN_BARRIER')
        b=json.loads(self.store.read(out['barrier'],256*1024))
        self.assertEqual(set(b['clean_receipt_sha256']),CHECKPOINT_IDS);self.assertEqual(digest(b),out['content_sha256'])
        self.assertEqual(b['status'],'ALL72_CLEAN_REGRESSIONS_BYTE_EXACT')

    def test_expanded_cannot_call_inference_without_both_profile_and_global_barrier(self):
        job=self.plan['original_records'][0]['job']
        with self.prior(),patch('hydra_directflow.coverage_extension_io.read_input_bundle',return_value={}), \
             patch('hydra_directflow.coverage_extension_inference.expanded_inference') as infer:
            with self.assertRaises(ValueError):controller.handle(self.plan,self.old,self.ledger,self.store,'EXPANDED',job['job_id'])
            self.profile()
            with self.assertRaises(ValueError):controller.handle(self.plan,self.old,self.ledger,self.store,'EXPANDED',job['job_id'])
            infer.assert_not_called()

    def test_expanded_authenticates_consumed_profile_artifact(self):
        self.all_clean();profile=self.profile()
        with self.prior():barrier=controller.handle(self.plan,self.old,self.ledger,self.store,'CLEAN_BARRIER')
        self.ledger.put('CLEAN_BARRIER',barrier);self.store.blobs[profile['path']]=b'corrupted profile'
        job=self.plan['original_records'][0]['job']
        with self.prior(),patch('hydra_directflow.coverage_extension_io.read_input_bundle',return_value={}), \
             patch('hydra_directflow.coverage_extension_inference.expanded_inference') as infer:
            with self.assertRaises((ValueError,AssertionError)):controller.handle(self.plan,self.old,self.ledger,self.store,'EXPANDED',job['job_id'])
            infer.assert_not_called()

    def test_expanded_passes_authenticated_clean_result_and_barrier_to_executor(self):
        self.all_clean();self.profile()
        with self.prior():barrier=controller.handle(self.plan,self.old,self.ledger,self.store,'CLEAN_BARRIER')
        self.ledger.put('CLEAN_BARRIER',barrier);job=self.plan['original_records'][0]['job'];result=self.result(job,clean=False)
        with self.prior(),patch('hydra_directflow.coverage_extension_io.read_input_bundle',side_effect=[{'clean':True},{'expanded':True}]), \
             patch('hydra_directflow.coverage_extension_inference.expanded_inference',return_value=result) as infer:
            out=controller.handle(self.plan,self.old,self.ledger,self.store,'EXPANDED',job['job_id'])
        self.assertEqual(infer.call_args.kwargs['expected_barrier_sha256'],barrier['content_sha256'])
        self.assertEqual(infer.call_args.args[3],self.ids);self.assertEqual(infer.call_args.args[4],{'clean':True})
        self.assertEqual(controller.stored_result(self.store,out,job)['receipt']['status'],'ALL_REQUESTED_VECTORS_FINITE')

    def test_stored_predictions_reject_nonhourly_coordinates_before_value_use(self):
        job=self.plan['original_records'][0]['job'];result=self.result(job,half_hour=True);entry=self.persist('half-hour',result)
        with self.assertRaisesRegex(ValueError,'clock'):controller.stored_result(self.store,entry,job)

    def test_prediction_barrier_requires72clean72input36routed(self):
        self.all_clean();self.ledger.put('CLEAN_BARRIER',dict(content_sha256='a'*64))
        for record in self.plan['original_records']:
            j=record['job'];self.ledger.put('EXPANDED__'+j['job_id'],self.persist('expanded-'+j['job_id'],self.result(j,clean=False)))
        self.ledger.put('ROUTED',{'products':{}})
        with self.prior(),self.assertRaisesRegex(ValueError,'prediction inventory'):
            controller.handle(self.plan,self.old,self.ledger,self.store,'PREDICTION_BARRIER')

    def test_complete_prediction_barrier_binds_input_map_and_whole_routed_streams(self):
        origins=np.arange(controller.START,controller.END,3600,dtype=np.int64);future=origins[:,None]+np.arange(1,19)*3600
        routes={};products={};results={}
        for gauge in GAUGES:
            long=np.zeros(8766,bool);short=long.copy();long[:3]=True;short[:4]=True
            route=np.where(long,'L168',np.where(short,'L24','UNAVAILABLE'))
            routes[gauge]=dict(origin_time=origins,forecast_time=future,eligible168=long,eligible24=short,route=route)
            for model in MODELS:
                for seed in (17,29,43):
                    key=f'{gauge}__{model}__seed{seed}';routed=np.full((8766,18),np.nan,np.float64);routed[:3]=168.;routed[3]=24.
                    results[key]=dict(arrays=dict(prediction_q=routed,route_history_hours=np.where(long,168,np.where(short,24,0)),
                        route_checkpoint_index=np.where(long,0,np.where(short,1,-1))),receipt=dict(gauge_id=gauge,architecture=model,seed=seed,
                        status='ALL_INPUT_ELIGIBLE_ROUTED_VECTORS_COMMITTED',route_map_sha256='e'*64,
                        checkpoint_sha256_by_history={str(h):self.plan['checkpoint_sha256_by_job'][f'final_development__{gauge}__{model}__full_L{h}__seed{seed}'] for h in (168,24)}))
                    products[key]={'synthetic_key':key}
        for record in self.plan['original_records']:
            job=record['job'];key=job['job_id'];n=3 if job['history_hours']==168 else 4
            results['clean-'+key]=self.result(job)
            results['expanded-'+key]=dict(arrays=dict(origin_time=origins[:n],prediction_q=np.full((n,18),float(job['history_hours']))),receipt={})
            self.ledger.put('CLEAN__'+key,{'synthetic_key':'clean-'+key});self.ledger.put('EXPANDED__'+key,{'synthetic_key':'expanded-'+key})
        self.ledger.put('CLEAN_BARRIER',{'content_sha256':'a'*64});self.ledger.put('ROUTED',{'products':products})
        def read_route(_,descriptor):
            gauge=next(g for g in GAUGES if self.inputs['routes'][g] is descriptor);return {'arrays':routes[gauge]}
        with self.prior(),patch.object(controller,'stored_result',side_effect=lambda store,item,job=None:results[item['synthetic_key']]), \
             patch('hydra_directflow.coverage_extension_io.read_route',side_effect=read_route):
            output=controller.handle(self.plan,self.old,self.ledger,self.store,'PREDICTION_BARRIER')
        barrier=json.loads(self.store.read(output['barrier'],2*1024**2))
        self.assertEqual(barrier['input_preparation_receipt'],self.inputs['receipt_artifact'])
        self.assertEqual(barrier['input_preparation_content_sha256'],digest({k:v for k,v in self.inputs.items() if k!='receipt_artifact'}))
        self.assertEqual(barrier['route_barrier_sha256'],self.inputs['route_barrier_content_sha256'])
        self.assertEqual((len(barrier['clean_products']),len(barrier['input_products']),len(barrier['routed_products'])),(72,72,36))
        self.assertFalse(barrier['target_values_loaded']);self.assertFalse(barrier['target_quality_loaded'])

    def test_evaluation_does_not_open_target_reader_before_prediction_barrier(self):
        with self.prior(),patch('hydra_directflow.coverage_extension_evaluate.evaluate') as evaluate:
            with self.assertRaises(ValueError):controller.handle(self.plan,self.old,self.ledger,self.store,'EVALUATION')
            evaluate.assert_not_called()
        b=dict(schema='SYNTHETIC_BARRIER',phase_plan_sha256=self.plan['plan_sha256'])
        desc=self.store.write('PREDICTION_BARRIER.json',controller.encoded(b),kind='SYNTHETIC')
        self.ledger.put('PREDICTION_BARRIER',dict(barrier=desc,content_sha256=digest(b)))
        evaluated=self.store.write('EVALUATION_INDEX.json',b'{"synthetic":true}',kind='SYNTHETIC')
        with self.prior(),patch('hydra_directflow.coverage_extension_evaluate.evaluate',return_value={
                'receipt_artifact':evaluated,'status':'SYNTHETIC_SCORED'}) as evaluate:
            out=controller.handle(self.plan,self.old,self.ledger,self.store,'EVALUATION')
        self.assertEqual(evaluate.call_args.args[3],b);self.assertEqual(evaluate.call_args.args[4],digest(b));self.assertIn('index',out)

    def test_routed_branch_uses_one_whole_existing_vector_per_origin_without_new_forward(self):
        origins=np.arange(controller.START,controller.END,3600,dtype=np.int64);future=origins[:,None]+np.arange(1,19)*3600
        def route(_,d):
            gauge=next(g for g in GAUGES if self.inputs['routes'][g] is d);long=np.zeros(8766,bool);short=long.copy();long[:2]=True;short[:3]=True
            return dict(arrays=dict(origin_time=origins,forecast_time=future,case_id=np.array([gauge+':'+iso(int(t)) for t in origins],dtype='<U29'),
                site_id=np.full(8766,GAUGES.index(gauge),np.int64),eligible168=long,eligible24=short,route=np.where(long,'L168',np.where(short,'L24','UNAVAILABLE'))))
        def product(_,entry,job):
            n=2 if job['history_hours']==168 else 3
            return dict(arrays=dict(origin_time=origins[:n],prediction_q=np.full((n,18),float(job['history_hours']))),
                        receipt=dict(checkpoint_sha256=self.plan['checkpoint_sha256_by_job'][job['job_id']],input_source_identity='a'*64))
        for record in self.plan['original_records']:self.ledger.put('EXPANDED__'+record['job']['job_id'],dict(arrays={'sha256':'a'*64}))
        with self.prior(),patch('hydra_directflow.coverage_extension_io.read_route',side_effect=route), \
             patch.object(controller,'stored_result',side_effect=product):
            result=controller.handle(self.plan,self.old,self.ledger,self.store,'ROUTED')
        self.assertEqual(len(result['products']),36)
        first=controller.stored_result(self.store,result['products']['03161000__LSTM__seed17'])
        np.testing.assert_array_equal(first['arrays']['prediction_q'][:2],168.)
        np.testing.assert_array_equal(first['arrays']['prediction_q'][2],24.)
        self.assertTrue(np.isnan(first['arrays']['prediction_q'][3:]).all())
        self.assertEqual(first['receipt']['history_hours'],'MIXED_PER_ORIGIN')

    def test_authentication_rejects_resigned_moved_source_changed_training_or_code(self):
        base={**self.plan};path=self.root/'phase.json';state=dict(complete=True,optimizer_updates_entered=1151690,
            optimizer_updates_confirmed=1151690,campaign_deadline=100000.)
        fake=SimpleNamespace(snapshot=lambda:state)
        for mutation in ('moved','training','code'):
            value=deepcopy(base);current_code=base['code_sha256'];state['optimizer_updates_confirmed']=1151690
            if mutation=='moved':value['corpus_root']=str(self.root/'other-corpus')
            elif mutation=='training':state['optimizer_updates_confirmed']+=1
            else:current_code={'synthetic.py':'d'*64}
            self.sign(value);path.write_text(json.dumps(value))
            with self.subTest(mutation=mutation),patch.object(controller,'PROTOCOL',self.protocol), \
                 patch.object(controller,'implementation',return_value=current_code), \
                 patch.object(controller.primary,'authenticated_plan',return_value=self.old), \
                 patch.object(controller,'CampaignLedger',return_value=fake), \
                 patch.object(controller,'_ledger_records',return_value=base['original_records']):
                with self.assertRaises(ValueError):controller.authenticated(path,value['plan_sha256'])

    def test_resigned_model_records_must_still_equal_original_authoritative_ledger(self):
        value=deepcopy(self.plan);value['original_records']=value['original_records'][:-1];self.sign(value)
        path=self.root/'phase.json';path.write_text(json.dumps(value));state=dict(complete=True,
            optimizer_updates_entered=1151690,optimizer_updates_confirmed=1151690,campaign_deadline=100000.)
        with patch.object(controller,'PROTOCOL',self.protocol),patch.object(controller,'implementation',return_value=self.plan['code_sha256']), \
             patch.object(controller.primary,'authenticated_plan',return_value=self.old), \
             patch.object(controller,'CampaignLedger',return_value=SimpleNamespace(snapshot=lambda:state)), \
             patch.object(controller,'_ledger_records',return_value=self.plan['original_records']):
            with self.assertRaisesRegex(ValueError,'model records differ'):controller.authenticated(path,value['plan_sha256'])

    def test_counted_authentication_bounds_metadata_reads_before_excess_body(self):
        path=self.root/'oversize-metadata';path.write_bytes(b'x'*100)
        def auth(*args):controller._read(path,1000);raise AssertionError('Should have stopped before body')
        with patch.object(controller,'AUTH_READ_RESERVE',2*1024**2+50),patch.object(controller,'authenticated',side_effect=auth):
            with self.assertRaisesRegex(ValueError,'precharged bound'):controller.counted_authentication(self.root/'phase.json','a'*64)

    def test_initialize_reserves_once_before_new_namespace_and_initial_read_charge(self):
        self.plan['initial_metadata_read_reservation']=3*controller.AUTH_READ_RESERVE
        marker=self.root/'release.json';events=[];state={'output_bytes_reserved':self.plan['original_output_reserved']}
        class Original:
            @contextmanager
            def writer(inner):yield inner
            def initialize(inner,old):events.append('original_initialize')
            def snapshot(inner):return dict(state)
            def reserve_output(inner,n,*,kind):
                self.assertFalse(self.run.exists());self.assertFalse(marker.exists())
                events.append(('original_reserve',n,kind));state['output_bytes_reserved']+=n;return 77
        class Phase:
            @contextmanager
            def writer(inner):yield inner
            def initialize(inner,p,token,**kw):events.append(('phase_initialize',token,kw))
            def reserve_output(inner,n,*,kind):events.append(('bookkeeping',n,kind));return 1
        with patch.object(controller,'MARKER',marker),patch.object(controller,'CampaignLedger',return_value=Original()), \
             patch.object(controller,'PhaseALedger',return_value=Phase()), \
             patch('shutil.disk_usage',return_value=SimpleNamespace(free=20*1024**3)):
            controller.initialize(self.plan,self.old)
            with self.assertRaisesRegex(ValueError,'existing Phase A'):controller.initialize(self.plan,self.old)
        self.assertEqual(sum(isinstance(e,tuple) and e[0]=='original_reserve' for e in events),1)
        initialization=next(e for e in events if isinstance(e,tuple) and e[0]=='phase_initialize')
        self.assertEqual(initialization[2]['initial_read_bytes'],3*controller.AUTH_READ_RESERVE)
        self.assertEqual(json.loads(marker.read_text())['original_reservation_token'],77)

    def test_original_budget_failure_creates_no_phase_or_release(self):
        marker=self.root/'release.json';state={'output_bytes_reserved':self.plan['original_output_reserved']}
        class Original:
            @contextmanager
            def writer(inner):yield inner
            def initialize(inner,old):pass
            def snapshot(inner):return state
            def reserve_output(inner,*args,**kw):raise ExtensionAllowanceExhausted('synthetic global output cap')
        with patch.object(controller,'MARKER',marker),patch.object(controller,'CampaignLedger',return_value=Original()), \
             patch.object(controller,'PhaseALedger') as phase,patch('shutil.disk_usage',return_value=SimpleNamespace(free=20*1024**3)):
            with self.assertRaises(ExtensionAllowanceExhausted):controller.initialize(self.plan,self.old)
            phase.assert_not_called()
        self.assertFalse(marker.exists());self.assertFalse(self.run.exists())

    def test_expired_original_deadline_stops_preparation_before_checkpoint_inventory(self):
        adopted=self.root/'adopted.md';adopted.write_bytes(b'SYNTHETIC ADOPTED TEXT')
        old_path=self.root/'old-plan.json';old_path.write_text(json.dumps(self.old))
        fake=SimpleNamespace(snapshot=lambda:dict(complete=True,optimizer_updates_entered=1151690,
            optimizer_updates_confirmed=1151690,campaign_deadline=999.))
        with patch.object(controller,'ADOPTED_SHA',hashlib.sha256(adopted.read_bytes()).hexdigest()), \
             patch.object(controller.primary,'authenticated_plan',return_value=self.old), \
             patch.object(controller,'CampaignLedger',return_value=fake),patch.object(controller.time,'time',return_value=1000.), \
             patch.object(controller,'_ledger_records') as records:
            with self.assertRaisesRegex(ValueError,'deadline expired'):controller.prepare(old_path,adopted)
            records.assert_not_called()

    def supervisor_mocks(self,*,running=False,exit_code=1,large_memory=False):
        self.run.mkdir(exist_ok=True);self.ledger.put('A0_RECONCILE',{});self.ledger.put('A2_INPUTS',{})
        events=[]
        class Child:
            pid=123456
            returncode=None if running else exit_code
            def poll(inner):return inner.returncode
            def wait(inner,timeout):events.append(('wait',timeout));return inner.returncode
            def terminate(inner):events.append(('terminate',));inner.returncode=-15
            def kill(inner):events.append(('kill',));inner.returncode=-9
        child=Child();process=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=3*1024**3 if large_memory else 1024))
        return child,process,events

    @contextmanager
    def supervisor_context(self,child,process):
        with patch.object(controller,'release',return_value={'original_reservation_token':77}), \
             patch.object(controller,'PhaseALedger',return_value=self.ledger), \
             patch.object(controller.subprocess,'Popen',return_value=child) as launched, \
             patch.object(controller.psutil,'Process',return_value=process), \
             patch.object(controller.time,'time',return_value=1000.), \
             patch.object(controller,'progress',side_effect=lambda p,status,**kw:{'status':status,**kw}):
            yield launched

    def test_worker_failure_stops_without_launching_next_checkpoint(self):
        child,process,events=self.supervisor_mocks(exit_code=7)
        with self.supervisor_context(child,process) as launch:
            result=controller.supervise(self.root/'plan.json',self.plan,self.old)
        self.assertEqual(result['status'],'STOPPED_WORKER_FAILURE');launch.assert_called_once()
        self.assertIn('--worker',launch.call_args.args[0]);self.assertEqual(launch.call_args.kwargs['env']['PYTORCH_ENABLE_MPS_FALLBACK'],'0')

    def test_worker_authentication_metadata_is_precharged_before_process_launch(self):
        child,process,events=self.supervisor_mocks(exit_code=7)
        with self.supervisor_context(child,process) as launch:
            def start(*args,**kwargs):
                reservations=[x for x in self.ledger.events if x[0]=='read_reserve']
                self.assertEqual(len(reservations),1)
                self.assertEqual(reservations[0][1][0],controller.AUTH_READ_RESERVE)
                self.assertEqual(reservations[0][2]['purpose'],'WORKER_AUTHENTICATION_METADATA')
                return child
            launch.side_effect=start
            controller.supervise(self.root/'plan.json',self.plan,self.old)
        launch.assert_called_once()

    def test_parent_memory_guard_terminates_and_reaps_live_worker(self):
        child,process,events=self.supervisor_mocks(running=True,large_memory=True)
        with self.supervisor_context(child,process) as launch:
            with self.assertRaisesRegex(ValueError,'memory limit'):controller.supervise(self.root/'plan.json',self.plan,self.old)
        self.assertEqual(events.count(('terminate',)),1);self.assertTrue(any(e[0]=='wait' for e in events));launch.assert_called_once()

    def test_successful_process_exit_without_committed_task_cannot_advance(self):
        child,process,events=self.supervisor_mocks(exit_code=0)
        with self.supervisor_context(child,process) as launch:
            with self.assertRaisesRegex(ValueError,'authenticated completion'):controller.supervise(self.root/'plan.json',self.plan,self.old)
        launch.assert_called_once()

    def test_existing_supervision_requires_one_matching_concrete_resume_diagnosis(self):
        self.run.mkdir();controller.primary.atomic(self.run/'SUPERVISION_START.json',{'synthetic':True},new=True)
        self.ledger.all_complete=True;child,process,_=self.supervisor_mocks()
        with self.supervisor_context(child,process) as launch:
            with self.assertRaises(ValueError):controller.supervise(self.root/'plan.json',self.plan,self.old)
            diagnosis=self.root/'diagnosis.json';diagnosis.write_text(json.dumps(dict(phase_plan_sha256='wrong',classification='INFRASTRUCTURE',evidence='synthetic only')))
            with self.assertRaises(ValueError):controller.supervise(self.root/'plan.json',self.plan,self.old,resume=True,resume_diagnosis=diagnosis)
            diagnosis.write_text(json.dumps(dict(phase_plan_sha256=self.plan['plan_sha256'],classification='INFRASTRUCTURE',evidence='Synthetic parent loss; no model execution.')))
            controller.supervise(self.root/'plan.json',self.plan,self.old,resume=True,resume_diagnosis=diagnosis)
            self.assertTrue((self.run/'SUPERVISION_RESUME.json').exists())
            with self.assertRaises(ValueError):controller.supervise(self.root/'plan.json',self.plan,self.old,resume=True,resume_diagnosis=diagnosis)
            launch.assert_not_called()

    def test_failed_integrity_task_cannot_be_reclassified_by_resume_file(self):
        self.run.mkdir();controller.primary.atomic(self.run/'SUPERVISION_START.json',{'synthetic':True},new=True)
        self.ledger.put('bad',{},state='FAILED_INTEGRITY');child,process,_=self.supervisor_mocks()
        diagnosis=self.root/'diagnosis.json';diagnosis.write_text(json.dumps(dict(phase_plan_sha256=self.plan['plan_sha256'],classification='INFRASTRUCTURE',evidence='Must not override failure.')))
        with self.supervisor_context(child,process) as launch:
            with self.assertRaisesRegex(ValueError,'failed numerical'):controller.supervise(self.root/'plan.json',self.plan,self.old,resume=True,resume_diagnosis=diagnosis)
            launch.assert_not_called()

    def test_recorded_supervisor_integrity_failure_cannot_be_relabeled(self):
        self.run.mkdir();controller.primary.atomic(self.run/'SUPERVISION_START.json',{'synthetic':True},new=True)
        controller.failure_record(self.plan,'INTEGRITY',dict(message='Foreign heartbeat'))
        self.ledger.all_complete=True;child,process,_=self.supervisor_mocks()
        diagnosis=self.root/'diagnosis.json';diagnosis.write_text(json.dumps(dict(phase_plan_sha256=self.plan['plan_sha256'],
            classification='INFRASTRUCTURE',evidence='Must not override trusted integrity failure.')))
        with self.supervisor_context(child,process) as launch:
            with self.assertRaisesRegex(ValueError,'non-infrastructure'):controller.supervise(self.root/'phase.json',self.plan,self.old,resume=True,resume_diagnosis=diagnosis)
            launch.assert_not_called()

    def test_actual_output_and_bookkeeping_sizes_have_distinct_finite_bounds(self):
        self.run.mkdir();payload=self.run/'tasks/A2_INPUTS/attempt-000001/payload.bin';payload.parent.mkdir(parents=True)
        with payload.open('wb') as f:f.truncate(controller.BOOKKEEPING+1)
        observed=controller.filesystem_bounds(self.plan)
        self.assertEqual(observed['actual_output_bytes'],controller.BOOKKEEPING+1)
        self.assertEqual(observed['actual_bookkeeping_bytes'],0)
        log=self.run/'worker-000001.log'
        with log.open('wb') as f:f.truncate(controller.BOOKKEEPING+1)
        with self.assertRaises(ExtensionAllowanceExhausted):controller.filesystem_bounds(self.plan)
        log.unlink()
        with payload.open('wb') as f:f.truncate(LIMITS['max_output_bytes']+1)
        with self.assertRaises(ExtensionAllowanceExhausted):controller.filesystem_bounds(self.plan)

    def test_output_symlink_is_rejected_without_following_external_content(self):
        self.run.mkdir();outside=self.root/'outside.txt';outside.write_bytes(b'not read')
        (self.run/'link').symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'symlink'):controller.filesystem_bounds(self.plan)


if __name__=='__main__':unittest.main()
