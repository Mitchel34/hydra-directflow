"""Synthetic target-poison/accounting tests; no real models or scaler fitting."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow import coverage_extension_inference as ext
from hydra_directflow.local_data import GAUGES, iso
from hydra_directflow.reference import digest
from hydra_directflow.training import TrainerSpec, TrainOnlyTransform, array_identity, FULL_FEATURE_SCHEMA


class PoisonLabels(dict):
    def __getitem__(self, key):
        if key in ('target','target_mask','admission_mask','eligibility_reason'):
            raise AssertionError('Target-derived member was accessed: ' + key)
        return super().__getitem__(key)


def fake_torch():
    class Optimizer:
        def __init__(self): pass
        def step(self): pass
    class AdamW(Optimizer):
        def step(self): pass
    class Adam(Optimizer):
        def step(self): pass
    class SGD(Optimizer):
        def step(self): pass
    return SimpleNamespace(optim=SimpleNamespace(Optimizer=Optimizer,AdamW=AdamW,Adam=Adam,SGD=SGD),
                           mps=SimpleNamespace(synchronize=lambda:None,empty_cache=lambda:None))


def bundle(origins, *, history=24, source='c'*64, cohort='d'*64, base='e'*64):
    t=np.asarray(origins,np.int64);n=len(t)
    h=np.empty((n,history,2),np.float64)
    h[:,:,0]=(t[:,None]-ext.START)/3600/100+1
    h[:,:,1]=np.arange(history+1,1,-1)
    arrays=dict(origin_time=t,history_time=t[:,None]-np.arange(history+1,1,-1)*3600,
                forecast_time=t[:,None]+np.arange(1,19)*3600,site_id=np.zeros(n,np.int64),
                case_id=np.array(['03161000:'+iso(int(x)) for x in t],dtype='<U29'),
                history=h,history_mask=np.ones(h.shape,bool),weather=np.full((n,18,4),.1,np.float64),
                weather_mask=np.ones((n,18,4),bool),state=np.full((n,2),.2,np.float64),state_mask=np.ones((n,2),bool),
                feature_schema=np.array(FULL_FEATURE_SCHEMA),cohort_admission_digest=np.array(cohort),
                inference_source_identity=np.array(source))
    core=dict(schema='hydra-phase-a-inference-input-view/v1',candidate_source_identity=base,source_root_identity=base,
              gauge_id='03161000',history_hours=history,feature_schema=FULL_FEATURE_SCHEMA,
              cohort_admission_digest=cohort,case_count=n,case_ids_sha256=digest(arrays['case_id'].tolist()),
              input_values_sha256=array_identity({k:arrays[k] for k in ext.INPUT_ARRAYS}),
              input_arrays_sha256=array_identity({k:arrays[k] for k in ext.INPUT_ARRAYS+('feature_schema','cohort_admission_digest')}),
              target_values_read=False,target_masks_read=False,protected_assessment_access=False,fixture_marker=source)
    identity=digest(core);arrays['inference_source_identity']=np.array(identity)
    return dict(arrays=PoisonLabels(arrays),source_identity=identity,cohort_admission_digest=cohort,
                manifest={**core,'source_identity':identity})


def values_for(arrays):
    # Depend on batch geometry to prove original partition preservation and
    # reuse. Includes valid negative predictions; no labels are consulted.
    n=len(arrays['origin_time'])
    return arrays['history'][:,-1,0,None]-np.arange(18)[None,:]*.5+n*1e-9


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.run=Path(self.temp.name).resolve()
        self.origins=ext.START+np.arange(70)*3600
        self.clean_positions=np.r_[np.arange(64),66,67,68]
        self.clean=bundle(self.origins[self.clean_positions]);self.expanded=bundle(self.origins,source='f'*64,cohort='1'*64)
        self.gradient_ids=tuple('03161000:'+iso(ext.START-100000+i*3600) for i in range(3))
        # Saved membership format is checked by the real loader in production;
        # this mocked loader still binds exactly the manifest-supplied sequence.
        self.spec=TrainerSpec('LSTM','cpu',24,64,.0001,17,True,10000,feature_schema=FULL_FEATURE_SCHEMA)
        stats=lambda n:[dict(mean=0.,scale=1.,constant=False,cells=10) for _ in range(n)]
        self.transform=TrainOnlyTransform(dict(history=stats(2),weather=stats(4),state=stats(2),target=stats(1)),
                                         digest(list(self.gradient_ids)),'9'*64)
        self.job=dict(order=0,job_id='final_development__03161000__LSTM__full_L24__seed17',
                      fold='final_development',input_arm='full',architecture='LSTM',gauge_id='03161000',
                      seed=17,history_hours=24,device='cpu',batch_size=64,precision='FP32',
                      data_job=dict(actual_common_cases={'outer_scoring':len(self.clean_positions)}))
        self.plan=dict(plan_sha256='a'*64,protocol_sha256='b'*64,source_plan_sha256='2'*64,
                       snapshot_sha256='3'*64,implementation_sha256={'synthetic.py':'4'*64},jobs=[self.job])
        self.directory=self.run/'jobs'/self.job['job_id']/'attempt-000001';self.directory.mkdir(parents=True)
        checkpoint=self.directory/'checkpoint.pt';checkpoint.write_bytes(b'SYNTHETIC CHECKPOINT NOT TORCH')
        descriptor=dict(path=str(checkpoint.relative_to(self.run)),bytes=checkpoint.stat().st_size,
                        sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        self.receipt=dict(plan_sha256=self.plan['plan_sha256'],protocol_sha256=self.plan['protocol_sha256'],
            source_plan_sha256=self.plan['source_plan_sha256'],snapshot_sha256=self.plan['snapshot_sha256'],
            implementation_sha256=self.plan['implementation_sha256'],job_id=self.job['job_id'],job_sha256=digest(self.job),
            status='VERIFIED_TRAIN_EVAL',protected_assessment_access=False,feature_schema=FULL_FEATURE_SCHEMA,
            selected_best_epoch=2,completed_epochs=10,trainer_spec=asdict(self.spec),
            trainer_runtime=dict(torch_threads=1,torch_interop_threads=1,default_dtype='torch.float32',deterministic_algorithms=True),
            worker_execution_runtime={'deterministic_algorithms':True},training_source_identity='5'*64,
            training_cohort_admission_digest='6'*64,inference_cohort_admission_digest='7'*64,
            receipt_path=str((self.directory/'completion.json').relative_to(self.run)),
            artifacts={'checkpoint':descriptor},hashes={'checkpoint':descriptor['sha256']},
            outer_case_ids_sha256=digest(self.clean['arrays']['case_id'].tolist()))
        projection={k:self.clean['arrays'][k] for k in ext.INPUT_ARRAYS}
        projection.update(feature_schema=np.array(FULL_FEATURE_SCHEMA),cohort_admission_digest=np.array('7'*64))
        core=dict(operation='inputs',target_values_decoded=False,fold='final_development',input_arm='full',
                  history_hours=24,gauge_id='03161000',cohort_admission_digest='7'*64,raw_arrays_sha256=array_identity(projection))
        original_input={**core,'source_identity':digest(core),'returned_arrays_sha256':'8'*64,'construction':{}}
        self.input_path=self.directory/'outer-input-source.json';self.input_path.write_text(json.dumps(original_input))
        self.receipt['inference_source_identity']=original_input['source_identity']
        self.original_prediction=np.concatenate([values_for({k:v[i:i+64] if getattr(v,'ndim',0)>0 else v
             for k,v in self.clean['arrays'].items()}) for i in range(0,len(self.clean_positions),64)])
        self.record=dict(job=self.job,receipt=self.receipt)
        self.write_original_predictions()
        self.torch=fake_torch();self.events=[];self.predictor=None;self.fail_predict=None;self.bad_spec=False;self.wrong_transform=False

    def tearDown(self):self.temp.cleanup()

    def write_original_predictions(self):
        meta={k:self.receipt[k] for k in ('plan_sha256','protocol_sha256','source_plan_sha256','snapshot_sha256',
              'implementation_sha256','job_id','job_sha256','training_source_identity','training_cohort_admission_digest',
              'inference_source_identity','inference_cohort_admission_digest','selected_best_epoch')}
        meta.update(checkpoint_sha256=self.receipt['hashes']['checkpoint'],case_ids_sha256=self.receipt['outer_case_ids_sha256'],
                    case_count=len(self.clean_positions),prediction_units='m3/s',label_values_opened=False,
                    posthoc_clipping=False,best_only=True,optimizer_constructed=False,scaler_fitted=False)
        arrays={k:self.clean['arrays'][k] for k in ('case_id','origin_time','forecast_time','site_id')}
        stream=io.BytesIO();np.savez_compressed(stream,prediction_q=self.original_prediction,**arrays,metadata_json=np.array(json.dumps(meta)))
        path=self.directory/'predictions.npz';path.write_bytes(stream.getvalue());sha=hashlib.sha256(path.read_bytes()).hexdigest()
        self.receipt['artifacts']['predictions']=dict(path=str(path.relative_to(self.run)),bytes=path.stat().st_size,sha256=sha)
        self.receipt['hashes']['predictions']=sha

    def precharge_forward(self,**kw):self.events.append(('forward_enter',kw));return len(self.events)
    def confirm_forward(self,token):self.events.append(('forward_confirm',token))
    def precharge_model(self,**kw):self.events.append(('model_enter',kw));return len(self.events)
    def confirm_model(self,token):self.events.append(('model_confirm',token))
    def read_budget(self,size):self.events.append(('read',size))
    def kwargs(self):return dict(phase_plan_sha256='0'*64,precharge_forward=self.precharge_forward,
        confirm_forward=self.confirm_forward,precharge_model=self.precharge_model,confirm_model=self.confirm_model,
        read_budget=self.read_budget)

    def loader(self,path,**kw):
        self.events.append(('load',kw))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==kw['expected_sha256']
        assert kw['expected_source_identity']==self.receipt['training_source_identity']
        assert kw['gradient_fit_case_ids']==self.gradient_ids
        owner=self
        class Predictor:
            spec=deepcopy(owner.spec)
            gauge='03161000'
            transform=deepcopy(owner.transform)
            torch=owner.torch
            receipt=dict(selected_best_epoch=2,selected_state='best_model',checkpoint_sha256=owner.receipt['hashes']['checkpoint'],
                         optimizer_constructed=False,scaler_fitted=False)
            def predict(self,arrays,**kw):
                owner.events.append(('predict',len(arrays['case_id'])))
                assert owner.events[-2][0]=='forward_enter'
                if owner.fail_predict=='raise':raise RuntimeError('synthetic model failure')
                if owner.fail_predict=='learning':TrainOnlyTransform.fit({},())
                value=values_for(arrays)
                if owner.fail_predict=='nan':value[0,0]=np.nan
                out={k:arrays[k] for k in ext.COORDINATES};out['prediction_q']=value
                return out
        predictor=Predictor()
        if self.bad_spec:predictor.spec=TrainerSpec('LSTM','cpu',24,64,.0001,29,True,10000,feature_schema=FULL_FEATURE_SCHEMA)
        if self.wrong_transform:predictor.transform.parameters['history'][0]['scale']=2.
        self.predictor=predictor;return predictor

    @contextmanager
    def runtime(self):
        with patch.object(ext,'configure_runtime',return_value=self.torch), \
             patch.object(ext.BestCheckpointPredictor,'load',side_effect=self.loader):yield

    def clean_result(self):
        with self.runtime():return ext.clean_regression(self.plan,self.record,self.run,self.gradient_ids,self.clean,**self.kwargs())

    def barrier(self,result):
        ids={f'final_development__{g}__{a}__full_L{h}__seed{s}':'a'*64
             for g in ext.GAUGE_IDS for a in ext.ARCHITECTURES for h in (168,24) for s in ext.SEEDS}
        ids[self.job['job_id']]=digest(result['receipt'])
        b=dict(schema='hydra-coverage-extension-clean-barrier/v1',status='ALL72_CLEAN_REGRESSIONS_BYTE_EXACT',
               phase_plan_sha256='0'*64,clean_receipt_sha256=ids)
        return b,digest(b)

    def expand(self,clean_result,bundle=None,barrier=None):
        b,sha=self.barrier(clean_result) if barrier is None else barrier
        with self.runtime():return ext.expanded_inference(self.plan,self.record,self.run,self.gradient_ids,self.clean,
            self.expanded if bundle is None else bundle,clean_result,clean_barrier=b,expected_barrier_sha256=sha,**self.kwargs())

    def test_clean_byte_parity_original_batch_partition_and_no_target_reads(self):
        self.clean['arrays'].update(target='POISON',target_mask='POISON',admission_mask='POISON')
        result=self.clean_result();self.assertEqual(result['receipt']['status'],'BYTE_EXACT_ORIGINAL_PREDICTIONS')
        self.assertEqual(result['arrays']['prediction_q'].tobytes(),self.original_prediction.tobytes())
        self.assertEqual([n for e,n in self.events if e=='predict'],[64,3])
        self.assertEqual(result['receipt']['accounting']['case_checkpoint_evaluations_confirmed'],67)
        self.assertEqual(result['receipt']['accounting']['model_constructions_confirmed'],1)
        self.assertLess(float(result['arrays']['prediction_q'].min()),0.)
        self.assertEqual(self.clean['arrays']['cohort_admission_digest'].item(),'d'*64)
        self.assertFalse(result['receipt']['input_verification']['original_cohort_used_for_forward'])

    def test_whole72_barrier_precedes_every_expanded_model_call(self):
        result=self.clean_result();b,sha=self.barrier(result);del b['clean_receipt_sha256'][next(iter(b['clean_receipt_sha256']))]
        self.events.clear()
        with self.assertRaisesRegex(ValueError,'All72'):self.expand(result,barrier=(b,digest(b)))
        self.assertEqual(self.events,[])

    def test_expanded_only_calls_new_cases_and_reuses_exact_clean_forecasts(self):
        clean=self.clean_result();self.events.clear();result=self.expand(clean)
        self.assertEqual(result['receipt']['new_forward_cases'],3);self.assertEqual(result['receipt']['reused_clean_cases'],67)
        self.assertEqual([n for e,n in self.events if e=='predict'],[3])
        self.assertEqual(result['arrays']['prediction_q'][self.clean_positions].tobytes(),self.original_prediction.tobytes())
        self.assertEqual(result['receipt']['input_cohort_admission_digest'],'1'*64)
        self.assertNotEqual(result['receipt']['input_source_identity'],clean['receipt']['input_source_identity'])
        self.assertEqual(result['receipt']['saved_transform_sha256'],clean['receipt']['saved_transform_sha256'])

    def test_all_reused_product_constructs_no_model_and_makes_no_forward(self):
        clean=self.clean_result();self.events.clear();result=self.expand(clean,bundle=self.clean)
        self.assertEqual(result['receipt']['new_forward_cases'],0)
        self.assertEqual(result['receipt']['accounting']['model_constructions_confirmed'],0)
        self.assertFalse(any(x[0] in ('load','predict','model_enter','forward_enter') for x in self.events))

    def test_changed_reused_input_or_candidate_source_fails_before_model_load(self):
        clean=self.clean_result()
        for mutation in ('input','source'):
            expanded=deepcopy(self.expanded)
            if mutation=='input':expanded['arrays']['history'][0,0,0]+=1
            else:expanded['manifest']['candidate_source_identity']='8'*64
            self.events.clear()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.expand(clean,bundle=expanded)
            self.assertFalse(any(x[0] in ('load','predict') for x in self.events))

    def test_changed_clean_prediction_receipt_or_array_fails(self):
        clean=self.clean_result()
        for mutate in ('receipt','array'):
            changed=deepcopy(clean);b,sha=self.barrier(clean)
            if mutate=='receipt':changed['receipt']['saved_transform_sha256']='8'*64
            else:changed['arrays']['prediction_q']=changed['arrays']['prediction_q'].copy();changed['arrays']['prediction_q'][0,0]+=1
            self.events.clear()
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):self.expand(changed,barrier=(b,sha))
            self.assertFalse(any(x[0] in ('load','predict') for x in self.events))

    def test_original_input_commitment_rejects_changed_clean_inputs_before_model(self):
        self.clean['arrays']['weather'][0,0,0]+=.01
        with self.assertRaisesRegex(ValueError,'content hash'):self.clean_result()
        self.assertFalse(any(x[0]=='load' for x in self.events))

    def test_truthfully_rebound_new_input_still_cannot_fake_original_clean_commitment(self):
        self.clean['arrays']['weather'][0,0,0]+=.01
        manifest=self.clean['manifest'];manifest['input_values_sha256']=ext.input_identity(self.clean)
        manifest['input_arrays_sha256']=array_identity({k:self.clean['arrays'][k] for k in ext.INPUT_ARRAYS+('feature_schema','cohort_admission_digest')})
        identity=digest({k:v for k,v in manifest.items() if k!='source_identity'});manifest['source_identity']=identity
        self.clean['source_identity']=identity;self.clean['arrays']['inference_source_identity']=np.array(identity)
        with self.assertRaisesRegex(ValueError,'Clean input values'):self.clean_result()
        self.assertFalse(any(x[0]=='load' for x in self.events))

    def test_clean_parity_is_bytes_not_tolerance(self):
        self.original_prediction=self.original_prediction.copy();self.original_prediction[0,0]+=1e-14;self.write_original_predictions()
        with self.assertRaisesRegex(ValueError,'byte parity failed'):self.clean_result()
        self.assertEqual([n for e,n in self.events if e=='predict'],[64,3])

    def test_nonfinite_or_model_exception_is_not_an_unavailable_route(self):
        for failure in ('nan','raise'):
            self.fail_predict=failure;self.events.clear()
            with self.subTest(failure=failure),self.assertRaises((ValueError,RuntimeError)):self.clean_result()
            self.assertEqual(sum(x[0]=='forward_enter' for x in self.events),1)
            self.assertEqual(sum(x[0]=='forward_confirm' for x in self.events),0)

    def test_learning_paths_are_poisoned_during_inference(self):
        self.fail_predict='learning'
        with self.assertRaisesRegex(AssertionError,'forbids training'):self.clean_result()
        with ext.no_learning(self.torch):
            for function in (lambda:ext.DevelopmentTrainer(),lambda:TrainOnlyTransform.fit({},()),
                             lambda:self.torch.optim.Optimizer(),lambda:self.torch.optim.AdamW.step(None)):
                with self.assertRaisesRegex(AssertionError,'forbids training'):function()

    def test_wrong_loaded_spec_or_transform_stops_before_forward(self):
        self.bad_spec=True
        with self.assertRaisesRegex(ValueError,'exact selected'):self.clean_result()
        self.assertFalse(any(x[0]=='predict' for x in self.events))
        self.bad_spec=False;clean=self.clean_result();self.wrong_transform=True;self.events.clear()
        with self.assertRaisesRegex(ValueError,'transform identity'):self.expand(clean)
        self.assertFalse(any(x[0]=='predict' for x in self.events))

    def test_wrong_record_scope_and_backend_are_rejected_before_source_reads(self):
        for key,value in [('fold','cv_1'),('input_arm','q_only'),('device','mps'),('seed',7),('history_hours',48),('gauge_id','bad')]:
            job=deepcopy(self.job);job[key]=value;plan=deepcopy(self.plan);plan['jobs']=[job]
            record=deepcopy(self.record);record['job']=job;record['receipt']['job_sha256']=digest(job)
            self.events.clear()
            with self.subTest(key=key),self.assertRaises(ValueError):
                ext.clean_regression(plan,record,self.run,self.gradient_ids,self.clean,**self.kwargs())
            self.assertEqual(self.events,[])

    def test_wrong_original_order_fails_before_model(self):
        arrays=self.clean['arrays'];arrays['case_id']=arrays['case_id'][::-1].copy()
        with self.assertRaises(ValueError):self.clean_result()
        self.assertFalse(any(x[0]=='load' for x in self.events))

    def test_protected_forecast_clock_rejects_before_input_values(self):
        arrays=deepcopy(self.clean['arrays']);arrays['forecast_time'][0,-1]=ext.ASSESSMENT_START
        class PoisonInputs(PoisonLabels):
            def __getitem__(self,key):
                if key in ('history','weather','state'):raise AssertionError('Numerical input reached')
                return super().__getitem__(key)
        bad={**self.clean,'arrays':PoisonInputs(arrays)}
        with self.assertRaises(ValueError):ext.validate_bundle(bad,self.job)

    def test_model_and_forward_budget_failures_happen_before_calls(self):
        kwargs=self.kwargs();kwargs['precharge_model']=lambda **kw:(_ for _ in ()).throw(RuntimeError('model quota'))
        with self.runtime(),self.assertRaisesRegex(RuntimeError,'model quota'):
            ext.clean_regression(self.plan,self.record,self.run,self.gradient_ids,self.clean,**kwargs)
        self.assertFalse(any(x[0]=='load' for x in self.events));self.events.clear()
        kwargs=self.kwargs();kwargs['precharge_forward']=lambda **kw:(_ for _ in ()).throw(RuntimeError('forward quota'))
        with self.runtime(),self.assertRaisesRegex(RuntimeError,'forward quota'):
            ext.clean_regression(self.plan,self.record,self.run,self.gradient_ids,self.clean,**kwargs)
        self.assertFalse(any(x[0]=='predict' for x in self.events))

    def test_checkpoint_residency_and_exact_size_before_model_load(self):
        path=self.run/self.receipt['artifacts']['checkpoint']['path'];path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Resident original'):self.clean_result()
        self.assertFalse(any(x[0]=='load' for x in self.events))

    def test_new_only_input_mutation_cannot_hide_behind_clean_input_equality(self):
        clean=self.clean_result();bad=deepcopy(self.expanded);bad['arrays']['weather'][64,0,0]+=1
        self.events.clear()
        with self.assertRaisesRegex(ValueError,'content hash'):self.expand(clean,bundle=bad)
        self.assertFalse(any(x[0]=='load' for x in self.events))

    def test_all_reuse_still_authenticates_preserved_checkpoint_bytes(self):
        clean=self.clean_result();path=self.run/self.receipt['artifacts']['checkpoint']['path']
        raw=path.read_bytes();path.write_bytes(b'X'+raw[1:]);self.events.clear()
        with self.assertRaisesRegex(ValueError,'SHA differs'):self.expand(clean,bundle=self.clean)
        self.assertFalse(any(x[0] in ('load','predict') for x in self.events))

    def test_offline_configuration_is_required_before_torch_import(self):
        with patch.dict(ext.os.environ,{},clear=True),patch.object(ext.importlib,'import_module') as imported:
            with self.assertRaisesRegex(ValueError,'before Torch import'):ext.configure_runtime('cpu')
            imported.assert_not_called()

    def test_runtime_preserves_fp32_determinism_threads_and_2gib_mps_cap(self):
        state=dict(threads=4,interop=4,dtype='other',deterministic=False,fraction=None)
        fake=SimpleNamespace(float32='fp32',set_num_threads=lambda n:state.update(threads=n),
            get_num_threads=lambda:state['threads'],set_num_interop_threads=lambda n:state.update(interop=n),
            get_num_interop_threads=lambda:state['interop'],set_default_dtype=lambda v:state.update(dtype=v),
            get_default_dtype=lambda:state['dtype'],use_deterministic_algorithms=lambda v:state.update(deterministic=v),
            are_deterministic_algorithms_enabled=lambda:state['deterministic'],
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:True)),
            mps=SimpleNamespace(recommended_max_memory=lambda:8*1024**3,
                set_per_process_memory_fraction=lambda v:state.update(fraction=v)))
        with patch.dict(ext.os.environ,{'PYTORCH_ENABLE_MPS_FALLBACK':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}), \
                patch.object(ext.importlib,'import_module',return_value=fake):
            self.assertIs(ext.configure_runtime('mps'),fake)
        self.assertEqual(state,dict(threads=1,interop=1,dtype='fp32',deterministic=True,fraction=.25))


if __name__=='__main__':unittest.main()
