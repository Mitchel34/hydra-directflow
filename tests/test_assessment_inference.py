"""Synthetic assessment-clock forwards: no real weights/data, fits, or network."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hydra_directflow.assessment_data import AssessmentReadAuthority
from hydra_directflow.assessment_inference import (
    DEVICES, assessment_input_only_view, predict_assessment_batch,
    validate_assessment_coordinates, validate_assessment_inputs,
    validate_selected_job, validate_selected_predictor)
from hydra_directflow.contract import ARCHITECTURES, GAUGE_IDS
from hydra_directflow.inference import BestCheckpointPredictor, transform_inputs
from hydra_directflow.local_data import iso
from hydra_directflow.models.directflow import DirectFlowModel, ModelConfig
from hydra_directflow.reference import (
    ASSESSMENT_START, ASSESSMENT_END, ASSESSMENT_LAST_TARGET, HOUR, digest)
from hydra_directflow.training import (
    CHECKPOINT_SCHEMA, FULL_FEATURE_SCHEMA, INPUTS, EarlyStopping,
    TrainOnlyTransform, TrainerSpec, array_identity,
    feature_schema_fields, implementation_fingerprint)


SOURCE='c'*64
COHORT='d'*64
GAUGE=GAUGE_IDS[0]
GRADIENT_IDS=(GAUGE+':2021-01-01T00:00:00Z', GAUGE+':2021-01-01T01:00:00Z')


def authority(**kwargs):
    return AssessmentReadAuthority(**dict(dict(execution_plan_sha256='1'*64,
        protocol_sha256='2'*64, selection_sha256='3'*64,
        source_inventory_sha256='4'*64, inputs_enabled=True), **kwargs))


def synthetic_inputs(n=3, *, first=ASSESSMENT_START, gauge_id=GAUGE):
    origins=first+np.arange(n, dtype=np.int64)*HOUR
    history_time=origins[:,None]-np.arange(169,1,-1)*HOUR
    history=np.empty((n,168,2), dtype=np.float64)
    history[...,0]=20+np.arange(n*168).reshape(n,168)/1000
    history[...,1]=(origins[:,None]-history_time)/HOUR
    weather=np.broadcast_to(np.array([1.,280.,.01,0.]),(n,18,4)).copy()
    state=np.broadcast_to(np.array([.25,6.]),(n,2)).copy()
    result=dict(history=history, weather=weather, state=state,
        origin_time=origins, history_time=history_time,
        forecast_time=origins[:,None]+np.arange(1,19)*HOUR,
        site_id=np.full(n, GAUGE_IDS.index(gauge_id), dtype=np.int64),
        case_id=np.array([gauge_id+':'+iso(int(c)) for c in origins], dtype='U'),
        input_admitted=np.ones(n, dtype=bool),
        feature_schema=np.array(FULL_FEATURE_SCHEMA),
        inference_source_identity=np.array(SOURCE),
        cohort_admission_digest=np.array(COHORT))
    for name in INPUTS:result[name+'_mask']=np.ones(result[name].shape, dtype=bool)
    return result


class PoisonLabels(dict):
    def __getitem__(self,key):
        if key in ('target','target_mask','target_time','labels','admission_mask','nwm'):
            raise AssertionError('Inference fetched forbidden label/mixed support: '+key)
        return super().__getitem__(key)


class PoisonInputs(PoisonLabels):
    def __getitem__(self,key):
        if key in INPUTS or key in tuple(x+'_mask' for x in INPUTS):
            raise AssertionError('Numerical inputs read before gate/metadata/clock checks')
        return super().__getitem__(key)


class Accounting:
    def __init__(self):self.events=[]
    def reserve(self,**kwargs):
        self.events.append(('reserve',kwargs));return 'reservation-1'
    def confirm(self,token):self.events.append(('confirm',token))
    def check(self):self.events.append(('check',))


def synthetic_checkpoint():
    """Manually declared fake saved statistics: never call a fitting method."""
    fields=feature_schema_fields(FULL_FEATURE_SCHEMA)
    parameters={name:[dict(mean=0.,scale=1.,constant=False,cells=2)
                      for _ in names] for name,names in fields.items()}
    parameters['target']=[dict(mean=10.,scale=3.,constant=False,cells=2)]
    transform=TrainOnlyTransform(parameters,digest(list(GRADIENT_IDS)),'e'*64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model=DirectFlowModel(ModelConfig('LSTM',fields['history'],fields['weather'],fields['state'],168,tuple(range(1,19))))
    best={key:value.detach().clone() for key,value in model.state_dict().items()}
    last={key:value.detach().clone() for key,value in best.items()};last['flow_head.bias']+=100
    stopping=EarlyStopping(2,2,1,0.);stopping.observe(1.);stopping.observe(2.)
    spec=TrainerSpec('LSTM','cpu',168,64,.0001,17,True,0,feature_schema=FULL_FEATURE_SCHEMA)
    return dict(schema=CHECKPOINT_SCHEMA,spec=asdict(spec),source_identity='a'*64,
        feature_schema=FULL_FEATURE_SCHEMA,cohort_admission_digest='b'*64,
        implementation=implementation_fingerprint(),updates=0,transform=asdict(transform),
        stopping=asdict(stopping),best_model=best,model=last,optimizer={'unused':True},
        epoch_records=[dict(epoch=1,inner_stopping_loss=1.,updates=0,improved=True),
                       dict(epoch=2,inner_stopping_loss=2.,updates=0,improved=False)])


class AssessmentInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)
        with (patch.object(TrainOnlyTransform,'fit',side_effect=AssertionError('Fit')),
                patch('torch.optim.AdamW',side_effect=AssertionError('Optimizer'))):
            cls.base_checkpoint=synthetic_checkpoint()

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();path=Path(self.temp.name)/'synthetic.pt'
        torch.save(self.base_checkpoint,path)
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        with (patch.object(TrainOnlyTransform,'fit',side_effect=AssertionError('Fit')),
                patch('torch.optim.AdamW',side_effect=AssertionError('Optimizer'))):
            self.predictor=BestCheckpointPredictor.load(path,expected_sha256=sha,
                expected_source_identity='a'*64,expected_feature_schema=FULL_FEATURE_SCHEMA,
                expected_training_cohort_admission_digest='b'*64,
                expected_gauge_id=GAUGE,gradient_fit_case_ids=GRADIENT_IDS)
        self.job=dict(job_id=f'final_development__{GAUGE}__LSTM__full_L168__seed17',
            gauge_id=GAUGE,architecture='LSTM',seed=17,input_arm='full',history_hours=168,device='cpu',
            checkpoint=dict(path='jobs/synthetic/checkpoint.pt',bytes=path.stat().st_size,sha256=sha),
            selected_weights='best_model',selected_best_epoch=1,feature_schema=FULL_FEATURE_SCHEMA,
            training_source_identity='a'*64,training_cohort_admission_digest='b'*64,
            training_arrays_sha256='9'*64,gradient_case_ids_sha256=digest(list(GRADIENT_IDS)),
            original_training_receipt_sha256='8'*64,
            transform_binding='Included in exact checkpoint byte hash; no refit')
        self.account=Accounting()

    def tearDown(self):self.temp.cleanup()

    def predict(self,arrays=None,**kwargs):
        params=dict(selected_job=self.job,expected_inference_source_identity=SOURCE,
            expected_inference_cohort_admission_digest=COHORT,
            precharge_forward=self.account.reserve,confirm_forward=self.account.confirm,
            check=self.account.check)
        gate=kwargs.pop('authority',authority());params.update(kwargs)
        with (patch.object(TrainOnlyTransform,'fit',side_effect=AssertionError('Fit')),
                patch('torch.optim.AdamW',side_effect=AssertionError('Optimizer'))):
            return predict_assessment_batch(self.predictor,
                synthetic_inputs() if arrays is None else arrays,gate,**params)

    def view(self,arrays=None,**kwargs):
        params=dict(gauge_id=GAUGE,expected_inference_source_identity=SOURCE,
                    expected_inference_cohort_admission_digest=COHORT)
        gate=kwargs.pop('authority',authority());params.update(kwargs)
        return assessment_input_only_view(synthetic_inputs() if arrays is None else arrays,gate,**params)

    def test_actual_saved_best_forward_matches_direct_saved_transform_and_stays_label_free(self):
        inputs=synthetic_inputs();inputs.update(target=object(),target_mask=object(),nwm=object(),admission_mask=object())
        view=self.view(PoisonLabels(inputs));result=self.predict(PoisonLabels(view))
        self.assertNotIn('target',view);self.assertNotIn('target_mask',view);self.assertNotIn('input_admitted',view)
        mapped=transform_inputs(view,self.predictor.transform)
        with torch.inference_mode():
            direct=self.predictor.model(*(torch.from_numpy(mapped[k]) for name in INPUTS for k in (name,name+'_mask')))
        expected=self.predictor.transform.inverse_target(direct.numpy().astype(np.float64))
        np.testing.assert_array_equal(result['prediction_q'],expected)
        self.assertEqual(result['prediction_q'].dtype,np.dtype('float64'))
        np.testing.assert_array_equal(result['origin_time'],inputs['origin_time'])
        np.testing.assert_array_equal(result['forecast_time'],inputs['forecast_time'])
        for value in view.values():self.assertFalse(value.flags.writeable)
        for name in ('prediction_q','origin_time','forecast_time','site_id','case_id'):
            self.assertFalse(result[name].flags.writeable)
        self.assertTrue(torch.equal(self.predictor.model.flow_head.bias,self.base_checkpoint['best_model']['flow_head.bias']))
        self.assertFalse(torch.equal(self.predictor.model.flow_head.bias,self.base_checkpoint['model']['flow_head.bias']))
        receipt=result['receipt']
        self.assertEqual(receipt['selected_best_epoch'],1)
        self.assertFalse(receipt['last_state_used']);self.assertFalse(receipt['target_values_read'])
        self.assertFalse(receipt['target_masks_read']);self.assertFalse(receipt['scientific_scoring_performed'])
        self.assertEqual(receipt['optimizer_updates'],0);self.assertEqual(receipt['scaler_fits'],0)
        self.assertEqual(receipt['model_forward_calls'],1)
        self.assertEqual(receipt['input_case_ids_sha256'],digest(inputs['case_id'].tolist()))
        self.assertEqual(receipt['input_arrays_sha256'],array_identity(dict(view)))
        self.assertEqual(receipt['inference_parent_cohort_admission_digest'],COHORT)
        self.assertEqual(receipt['inference_source_identity'],SOURCE)
        self.assertEqual(receipt['training_source_identity'],'a'*64)
        self.assertEqual(receipt['training_cohort_admission_digest'],'b'*64)
        self.assertEqual(receipt['selection_sha256'],authority().selection_sha256)
        self.assertEqual(receipt['status'],'VERIFIED_FORWARD_AWAITING_RUNNER_ARTIFACT_COMMIT')
        self.assertEqual(receipt['accounting']['model_constructions_this_call'],0)
        self.assertEqual(receipt['accounting']['existing_predictors_used'],1)
        self.assertEqual(receipt['accounting']['case_checkpoint_evaluations_confirmed'],3)
        self.assertEqual([e[0] for e in self.account.events],['check','reserve','confirm','check'])
        self.assertEqual(self.account.events[1][1],dict(case_count=3,batch_count=1,
            stage='ASSESSMENT_FULL_L168',case_ids_sha256=digest(inputs['case_id'].tolist())))

    def test_target_values_masks_and_extra_keys_do_not_change_forecasts(self):
        inputs=synthetic_inputs();one=self.predict(inputs)
        inputs.update(target=np.full((3,18),1e99),target_mask=np.zeros((3,18),dtype=bool),
                      labels=object(),target_time=object(),admission_mask=object(),nwm=object())
        two=self.predict(PoisonLabels(inputs))
        np.testing.assert_array_equal(one['prediction_q'],two['prediction_q'])
        self.assertEqual(one['receipt']['input_arrays_sha256'],two['receipt']['input_arrays_sha256'])

    def test_old_development_predictor_still_rejects_assessment_dates(self):
        with self.assertRaisesRegex(ValueError,'Protected'):
            self.predictor.predict(PoisonInputs(synthetic_inputs()),
                expected_inference_source_identity=SOURCE,expected_inference_cohort_admission_digest=COHORT)

    def test_first_history_and_last_h18_tail_have_no_date_shift_or_development_purge(self):
        first=self.predict(synthetic_inputs(1))
        self.assertEqual(synthetic_inputs(1)['history_time'][0,0],ASSESSMENT_START-169*HOUR)
        last=self.predict(synthetic_inputs(1,first=ASSESSMENT_END-HOUR))
        self.assertEqual(first['origin_time'][0],ASSESSMENT_START)
        self.assertEqual(last['forecast_time'][0,-1],ASSESSMENT_LAST_TARGET)
        self.assertEqual(last['receipt']['last_target_utc'],iso(ASSESSMENT_LAST_TARGET))

    def test_disabled_absent_wrong_length_or_verification_phase_authority_rejects_before_inputs(self):
        gates=(None,object(),authority(inputs_enabled=False),authority(allowed_history_hours=(24,)),
               authority(verification_enabled=True,prediction_barrier_sha256='f'*64),
               authority(prediction_barrier_sha256='f'*64))
        for gate in gates:
            with self.subTest(gate=gate),self.assertRaises(ValueError):
                self.predict(PoisonInputs(synthetic_inputs()),authority=gate)
            with self.assertRaises(ValueError):self.view(PoisonInputs(synthetic_inputs()),authority=gate)
        self.assertEqual(self.account.events,[])

    def test_source_cohort_and_schema_identity_fail_before_numerical_inputs(self):
        for key in ('feature_schema','inference_source_identity','cohort_admission_digest'):
            values=synthetic_inputs();values[key]=np.array('wrong')
            with self.subTest(key=key),self.assertRaises(ValueError):self.predict(PoisonInputs(values))
        for kwargs in ({'expected_inference_source_identity':'a'*64},
                       {'expected_inference_cohort_admission_digest':'b'*64},
                       {'expected_inference_source_identity':'invalid'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                self.predict(PoisonInputs(synthetic_inputs()),**kwargs)
        self.assertEqual(self.account.events,[])

    def test_malformed_clock_gauge_case_and_admission_reject_before_inputs(self):
        for change in ('before','after','not_hour','out_of_order','duplicate','unsigned_reverse',
                       'history','future','float_clock','gauge','case','admission','admission_dtype',
                       'empty','more_than64'):
            a=synthetic_inputs()
            if change=='before':a=synthetic_inputs(first=ASSESSMENT_START-HOUR)
            elif change=='after':a=synthetic_inputs(first=ASSESSMENT_END)
            elif change=='not_hour':a['origin_time'][0]+=1
            elif change=='out_of_order':a['origin_time']=a['origin_time'][::-1]
            elif change=='unsigned_reverse':a['origin_time']=a['origin_time'][::-1].astype(np.uint64)
            elif change=='duplicate':a['origin_time'][1]=a['origin_time'][0]
            elif change=='history':a['history_time'][0,0]-=HOUR
            elif change=='future':a['forecast_time'][0,-1]+=HOUR
            elif change=='float_clock':a['origin_time']=a['origin_time'].astype(float)
            elif change=='gauge':a['site_id'][0]=1
            elif change=='case':a['case_id'][0]='invalid'
            elif change=='admission':a['input_admitted'][0]=False
            elif change=='admission_dtype':a['input_admitted']=a['input_admitted'].astype(int)
            elif change=='empty':a=synthetic_inputs(0)
            else:a=synthetic_inputs(65)
            with self.subTest(change=change),self.assertRaises(ValueError):self.predict(PoisonInputs(a))
        self.assertEqual(self.account.events,[])

    def test_complete_masks_required_even_without_admission_array(self):
        for input_name in INPUTS:
            for admitted in ('present','absent'):
                a=synthetic_inputs();a[input_name+'_mask'].flat[0]=False
                if admitted=='absent':a.pop('input_admitted')
                with self.subTest(input=input_name,admitted=admitted),self.assertRaisesRegex(ValueError,'complete'):
                    self.predict(a)
        self.assertEqual(self.account.events,[])

    def test_input_shapes_masks_nonfinite_history_age_and_soil_age_reject_before_forward(self):
        for change in ('history_nan','weather_inf','state_nan','shape','int_values','int_mask',
                       'history_age','soil_future','soil_stale','soil_fractional_age'):
            a=synthetic_inputs()
            if change=='history_nan':a['history'][0,0,0]=np.nan
            elif change=='weather_inf':a['weather'][0,0,0]=np.inf
            elif change=='state_nan':a['state'][0,0]=np.nan
            elif change=='shape':a['history']=a['history'][:,:24]
            elif change=='int_values':a['history']=a['history'].astype(int)
            elif change=='int_mask':a['history_mask']=a['history_mask'].astype(int)
            elif change=='history_age':a['history'][0,0,1]-=1
            elif change=='soil_future':a['state'][0,1]=0
            elif change=='soil_stale':a['state'][0,1]=9
            else:a['state'][0,1]=2.5
            with (self.subTest(change=change),patch.object(self.predictor.model,'forward',side_effect=AssertionError('Forward')),
                    self.assertRaises(ValueError)):self.predict(a)
        self.assertEqual(self.account.events,[])

    def test_view_accepts128_filtered_rows_but_forward_is_bounded64(self):
        view=self.view(synthetic_inputs(128));self.assertEqual(len(view['case_id']),128)
        with self.assertRaises(ValueError):self.predict(PoisonInputs(view))
        with self.assertRaises(ValueError):self.view(PoisonInputs(synthetic_inputs(129)))
        result=self.predict(synthetic_inputs(64));self.assertEqual(result['prediction_q'].shape,(64,18))
        self.assertEqual(result['receipt']['accounting']['case_checkpoint_evaluations_confirmed'],64)

    def test_copied_coordinates_are_revalidated_before_any_model_forward(self):
        class ChangesOnSecondRead(dict):
            def __init__(self,data):super().__init__(data);self.reads=0
            def __getitem__(self,key):
                if key=='forecast_time':
                    self.reads+=1
                    if self.reads>1:return super().__getitem__(key)+HOUR
                return super().__getitem__(key)
        for action in (self.view,self.predict):
            with self.subTest(action=action.__name__),self.assertRaisesRegex(ValueError,'clocks'):
                action(ChangesOnSecondRead(synthetic_inputs()))
        self.assertEqual(self.account.events,[])

    def test_all36_selected_record_combinations_are_retained_without_favorable_seed_selection(self):
        count=0
        for gauge in GAUGE_IDS:
            for architecture in ARCHITECTURES:
                for seed in (17,29,43):
                    job=deepcopy(self.job);job.update(gauge_id=gauge,architecture=architecture,
                        seed=seed,device=DEVICES[architecture],
                        job_id=f'final_development__{gauge}__{architecture}__full_L168__seed{seed}')
                    self.assertIs(validate_selected_job(job,authority()),job);count+=1
        self.assertEqual(count,36)

    def test_selection_identity_mode_best_state_and_device_mismatches_reject_before_input_read(self):
        changes=({'seed':29},{'architecture':'Transformer'},{'device':'mps'},
                 {'gauge_id':GAUGE_IDS[1]},{'input_arm':'q_only'},{'history_hours':24},
                 {'feature_schema':'wrong'},{'selected_weights':'model'},
                 {'selected_best_epoch':2},{'selected_best_epoch':101},
                 {'training_source_identity':'f'*64},{'training_cohort_admission_digest':'f'*64},
                 {'gradient_case_ids_sha256':'f'*64},{'job_id':'other'},
                 {'saved_transform_sha256':'f'*64})
        for changeset in changes:
            job=deepcopy(self.job);job.update(changeset)
            with self.subTest(changes=changeset),self.assertRaises(ValueError):
                self.predict(PoisonInputs(synthetic_inputs()),selected_job=job)
        for change in ('sha','absolute','traversal','bytes'):
            job=deepcopy(self.job)
            if change=='sha':job['checkpoint']['sha256']='f'*64
            elif change=='absolute':job['checkpoint']['path']='/tmp/bad.pt'
            elif change=='traversal':job['checkpoint']['path']='../bad.pt'
            else:job['checkpoint']['bytes']=17*1024**2
            with self.subTest(change=change),self.assertRaises(ValueError):
                self.predict(PoisonInputs(synthetic_inputs()),selected_job=job)
        self.assertEqual(self.account.events,[])

    def test_loaded_spec_receipt_and_transform_provenance_are_bound(self):
        for key,value in (('selected_state','model'),('selected_best_epoch',2),
                          ('optimizer_constructed',True),('scaler_fitted',True),
                          ('last_state_used',True),('completed_epochs',101),
                          ('training_values_sha256','f'*64)):
            with self.subTest(key=key),patch.dict(self.predictor.receipt,{key:value}),self.assertRaises(ValueError):
                self.predict(PoisonInputs(synthetic_inputs()))
        with patch.object(self.predictor,'spec',replace(self.predictor.spec,batch_size=32)),self.assertRaises(ValueError):
            self.predict(PoisonInputs(synthetic_inputs()))
        bad=replace(self.predictor.transform,training_case_ids_sha256='f'*64)
        with patch.object(self.predictor,'transform',bad),self.assertRaises(ValueError):
            self.predict(PoisonInputs(synthetic_inputs()))
        bad=deepcopy(self.predictor.transform);bad.parameters['history'][0]['scale']=0
        with patch.object(self.predictor,'transform',bad),self.assertRaises(ValueError):
            self.predict(PoisonInputs(synthetic_inputs()))
        self.assertEqual(self.account.events,[])

    def test_mismatched_model_geometry_and_backend_are_rejected_before_inputs(self):
        config=replace(self.predictor.model.config,history_hours=24)
        with patch.object(self.predictor.model,'config',config),self.assertRaises(ValueError):
            self.predict(PoisonInputs(synthetic_inputs()))
        self.predictor.model.double()
        with self.assertRaises(ValueError):self.predict(PoisonInputs(synthetic_inputs()))
        self.assertEqual(self.account.events,[])

    def test_missing_callbacks_or_failed_reservation_prevents_forward(self):
        for kwargs in ({'precharge_forward':None},{'confirm_forward':None},{'check':None}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):self.predict(PoisonInputs(synthetic_inputs()),**kwargs)
        with patch.object(self.predictor.model,'forward',side_effect=AssertionError('Forward')):
            with self.assertRaisesRegex(RuntimeError,'budget exhausted'):
                self.predict(precharge_forward=lambda **kw: (_ for _ in ()).throw(RuntimeError('budget exhausted')))
            with self.assertRaisesRegex(ValueError,'reservation'):
                self.predict(precharge_forward=lambda **kw:None)
        self.assertFalse(any(e[0]=='confirm' for e in self.account.events))

    def test_reservation_precedes_forward_and_success_confirmation_follows(self):
        forward=self.predictor.model.forward
        def checked(*args):
            self.assertEqual([x[0] for x in self.account.events],['check','reserve'])
            self.account.events.append(('model',));return forward(*args)
        with patch.object(self.predictor.model,'forward',side_effect=checked):self.predict()
        self.assertEqual([x[0] for x in self.account.events],['check','reserve','model','confirm','check'])

    def test_forward_failure_and_bad_outputs_are_charged_once_and_never_confirmed(self):
        for bad in (torch.full((3,18),float('nan')),torch.zeros((3,17)),
                    torch.zeros((3,18),dtype=torch.float64),np.zeros((3,18))):
            self.account=Accounting()
            with patch.object(self.predictor.model,'forward',return_value=bad) as forward,self.assertRaises(ValueError):
                self.predict()
            self.assertEqual(forward.call_count,1)
            self.assertEqual([x[0] for x in self.account.events],['check','reserve'])
        self.account=Accounting()
        with (patch.object(self.predictor.model,'forward',side_effect=RuntimeError('failed')) as forward,
                self.assertRaisesRegex(RuntimeError,'failed')):self.predict()
        self.assertEqual(forward.call_count,1)
        self.assertEqual([x[0] for x in self.account.events],['check','reserve'])

    def test_inverse_nonfinite_or_wrong_shape_is_not_confirmed(self):
        for bad in (np.full((3,18),np.inf),np.zeros((3,17)),np.zeros((3,18),dtype=np.float32)):
            self.account=Accounting()
            with patch.object(TrainOnlyTransform,'inverse_target',return_value=bad),self.assertRaises(ValueError):self.predict()
            self.assertEqual([x[0] for x in self.account.events],['check','reserve'])

    def test_negative_predictions_remain_raw_without_clipping(self):
        with patch.object(self.predictor.model,'forward',return_value=torch.full((3,18),-10.,dtype=torch.float32)):
            result=self.predict()
        np.testing.assert_array_equal(result['prediction_q'],np.full((3,18),-20.))
        self.assertFalse(result['receipt']['posthoc_clipping'])

    def test_inference_never_updates_weights_or_creates_gradients(self):
        before={k:v.clone() for k,v in self.predictor.model.state_dict().items()}
        self.predictor.model.train();self.predict()
        self.assertFalse(self.predictor.model.training)
        for key,value in self.predictor.model.state_dict().items():self.assertTrue(torch.equal(before[key],value))
        self.assertTrue(all(p.grad is None for p in self.predictor.model.parameters()))


if __name__=='__main__':unittest.main()
