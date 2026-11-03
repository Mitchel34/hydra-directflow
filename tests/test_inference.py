"""Label-free synthetic CPU inference; zero optimizer construction/updates."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from hydra_directflow.inference import (BestCheckpointPredictor,input_only_view,validate_inputs,transform_inputs,selected_best_epoch)
from hydra_directflow.local_data import iso
from hydra_directflow.training import (TrainOnlyTransform,TrainerSpec,EarlyStopping,implementation_fingerprint,
    CHECKPOINT_SCHEMA,FULL_FEATURE_SCHEMA,Q_ONLY_FEATURE_SCHEMA,feature_schema_fields,q_only_training_bundle)
from hydra_directflow.models.directflow import DirectFlowModel,ModelConfig
from test_training import fixture


class PoisonLabels(dict):
    def __getitem__(self,key):
        if key in ('target','target_mask'): raise AssertionError('Inference fetched labels: '+key)
        return super().__getitem__(key)


def arrays_for(schema):
    arrays=fixture();arrays['case_id']=np.array(['03161000:'+iso(int(t)) for t in arrays['origin_time']])
    arrays['feature_schema']=np.array(FULL_FEATURE_SCHEMA);arrays['cohort_admission_digest']=np.array('b'*64)
    if schema==Q_ONLY_FEATURE_SCHEMA:arrays=q_only_training_bundle(arrays,cohort_admission_digest='b'*64)
    return arrays


def checkpoint_for(arrays,schema):
    transform=TrainOnlyTransform.fit(arrays,(0,1),feature_schema=schema)
    fields=feature_schema_fields(schema)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model=DirectFlowModel(ModelConfig('LSTM',fields['history'],fields['weather'],fields['state'],24,tuple(range(1,19))))
    best={key:value.detach().clone() for key,value in model.state_dict().items()}
    last={key:value.detach().clone() for key,value in best.items()};last['flow_head.bias']+=100
    stopping=EarlyStopping(2,2,1,0.);stopping.observe(1.);stopping.observe(2.)
    spec=TrainerSpec('LSTM','cpu',24,4,.0001,17,True,0,feature_schema=schema)
    return dict(schema=CHECKPOINT_SCHEMA,spec=asdict(spec),source_identity='a'*64,feature_schema=schema,
        cohort_admission_digest='b'*64,implementation=implementation_fingerprint(),updates=0,
        transform=asdict(transform),stopping=asdict(stopping),best_model=best,model=last,optimizer={'unused':True},
        epoch_records=[dict(epoch=1,inner_stopping_loss=1.,updates=0,improved=True),dict(epoch=2,inner_stopping_loss=2.,updates=0,improved=False)])


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.old_threads)
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'best.pt'
    def tearDown(self):self.temp.cleanup()

    def load(self,checkpoint,arrays,**kwargs):
        torch.save(checkpoint,self.path);sha=hashlib.sha256(self.path.read_bytes()).hexdigest()
        params=dict(expected_sha256=sha,expected_source_identity='a'*64,expected_feature_schema=checkpoint['feature_schema'],
            expected_training_cohort_admission_digest='b'*64,expected_gauge_id='03161000',gradient_fit_case_ids=tuple(arrays['case_id'][:2].tolist()))
        params.update(kwargs)
        with patch('torch.optim.AdamW',side_effect=AssertionError('Optimizer created')),patch.object(TrainOnlyTransform,'fit',side_effect=AssertionError('Scaler fitted')):
            return BestCheckpointPredictor.load(self.path,**params)

    def test_full_and_q_only_label_free_best_state_predictions_equal_explicit_best_forward(self):
        for schema in (FULL_FEATURE_SCHEMA,Q_ONLY_FEATURE_SCHEMA):
            with self.subTest(schema=schema):
                arrays=arrays_for(schema);checkpoint=checkpoint_for(arrays,schema);predictor=self.load(checkpoint,arrays)
                inputs=input_only_view(PoisonLabels(arrays),inference_source_identity='c'*64)
                self.assertNotIn('target',inputs);self.assertNotIn('target_mask',inputs)
                result=predictor.predict(PoisonLabels(inputs),expected_inference_source_identity='c'*64,expected_inference_cohort_admission_digest='b'*64)
                mapped=transform_inputs(inputs,predictor.transform)
                with torch.inference_mode():
                    direct=predictor.model(*(torch.from_numpy(mapped[name]) for field in ('history','weather','state') for name in (field,field+'_mask')))
                expected=predictor.transform.inverse_target(direct.numpy().astype(np.float64))
                np.testing.assert_array_equal(result['prediction_q'],expected)
                self.assertEqual(result['receipt']['selected_best_epoch'],1)
                self.assertFalse(result['receipt']['last_state_used']);self.assertFalse(result['receipt']['target_values_read'])
                self.assertTrue(torch.equal(predictor.model.state_dict()['flow_head.bias'],checkpoint['best_model']['flow_head.bias']))
                self.assertFalse(torch.equal(predictor.model.state_dict()['flow_head.bias'],checkpoint['model']['flow_head.bias']))
                self.assertFalse(result['prediction_q'].flags.writeable)

    def test_inference_cohort_is_independent_from_training_parent_but_explicitly_bound(self):
        arrays=arrays_for(FULL_FEATURE_SCHEMA);predictor=self.load(checkpoint_for(arrays,FULL_FEATURE_SCHEMA),arrays)
        future={key:value.copy() for key,value in arrays.items()};future['cohort_admission_digest']=np.array('d'*64)
        future['origin_time']+=7*86400;future['history_time']+=7*86400;future['target_time']+=7*86400
        future['case_id']=np.array(['03161000:'+iso(int(t)) for t in future['origin_time']])
        inputs=input_only_view(PoisonLabels(future),inference_source_identity='e'*64)
        result=predictor.predict(inputs,expected_inference_source_identity='e'*64,expected_inference_cohort_admission_digest='d'*64)
        self.assertEqual(result['receipt']['training_cohort_admission_digest'],'b'*64)
        self.assertEqual(result['receipt']['inference_cohort_admission_digest'],'d'*64)
        for source,cohort in (('c'*64,'d'*64),('e'*64,'b'*64)):
            with self.assertRaises(ValueError):predictor.predict(inputs,expected_inference_source_identity=source,expected_inference_cohort_admission_digest=cohort)

    def test_protected_view_and_prediction_reject_before_numerical_input_access(self):
        arrays=arrays_for(FULL_FEATURE_SCHEMA)
        from hydra_directflow.local_data import utc_epoch
        shift=utc_epoch('2024-07-31T00:00:00Z')-int(arrays['origin_time'][0])
        for name in ('origin_time','history_time','target_time'):arrays[name]+=shift
        class PoisonInputs(PoisonLabels):
            def __getitem__(self,key):
                if key in ('history','weather','state'):raise AssertionError('Read numerical input before protected gate')
                return super().__getitem__(key)
        with self.assertRaisesRegex(ValueError,'Protected'):
            input_only_view(PoisonInputs(arrays),inference_source_identity='c'*64)
        protected={**arrays,'forecast_time':arrays['target_time']}
        with self.assertRaisesRegex(ValueError,'Protected'):
            validate_inputs(PoisonInputs(protected),history_hours=24,feature_schema=FULL_FEATURE_SCHEMA,
                expected_cohort_admission_digest='b'*64,expected_gauge_id='03161000')

    def test_saved_input_transform_matches_existing_mechanics_without_target_access(self):
        arrays=arrays_for(FULL_FEATURE_SCHEMA);arrays['history'][0,0]=np.nan;arrays['history_mask'][0,0]=False
        transform=TrainOnlyTransform.fit(arrays,(0,1));old=transform.transform(arrays)
        inputs=input_only_view(PoisonLabels(arrays),inference_source_identity='c'*64)
        new=transform_inputs(PoisonLabels(inputs),transform)
        self.assertNotIn('target',new)
        for key in new:np.testing.assert_array_equal(new[key],old[key])
        self.assertTrue((new['history'][0,0]==0).all())

    def test_missing_or_inconsistent_best_selection_never_falls_back_to_last(self):
        arrays=arrays_for(FULL_FEATURE_SCHEMA);base=checkpoint_for(arrays,FULL_FEATURE_SCHEMA)
        for mutation in ('absent','no_stopping','wrong_flag','wrong_loss','bad_updates'):
            value=deepcopy(base)
            if mutation=='absent':value['best_model']=None
            elif mutation=='no_stopping':value['stopping']=None
            elif mutation=='wrong_flag':value['epoch_records'][1]['improved']=True
            elif mutation=='wrong_loss':value['stopping']['best_loss']=2.
            else:value['epoch_records'][1]['updates']=1
            with self.subTest(mutation=mutation),patch('hydra_directflow.models.directflow.DirectFlowModel',side_effect=AssertionError('Model created before selection check')):
                with self.assertRaises(ValueError):self.load(value,arrays)

    def test_bad_best_state_keys_shape_dtype_and_nonfinite_values_reject(self):
        arrays=arrays_for(FULL_FEATURE_SCHEMA);base=checkpoint_for(arrays,FULL_FEATURE_SCHEMA)
        for mutation in ('missing_key','shape','dtype','nan','positions'):
            value=deepcopy(base)
            if mutation=='missing_key':del value['best_model']['flow_head.bias']
            elif mutation=='shape':value['best_model']['flow_head.bias']=torch.ones(2)
            elif mutation=='dtype':value['best_model']['flow_head.bias']=value['best_model']['flow_head.bias'].double()
            elif mutation=='nan':value['best_model']['flow_head.bias'][0]=float('nan')
            else:value['best_model']['history_position']+=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.load(value,arrays)

    def test_checkpoint_source_gauge_fit_membership_and_hash_binding_reject(self):
        arrays=arrays_for(FULL_FEATURE_SCHEMA);checkpoint=checkpoint_for(arrays,FULL_FEATURE_SCHEMA)
        for changes in ({'expected_sha256':'f'*64},{'expected_source_identity':'f'*64},
                        {'expected_training_cohort_admission_digest':'f'*64},{'expected_gauge_id':'03164000'},
                        {'gradient_fit_case_ids':tuple(arrays['case_id'][1:3].tolist())}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.load(checkpoint,arrays,**changes)

    def test_coordinates_schema_tokens_and_batch_bound_reject(self):
        arrays=arrays_for(Q_ONLY_FEATURE_SCHEMA);inputs=dict(input_only_view(arrays,inference_source_identity='c'*64))
        for mutation in ('order','case_id','horizon','schema','token','mask','batch_bound'):
            data={key:value.copy() for key,value in inputs.items()}
            if mutation=='order':data['origin_time'][1]=data['origin_time'][0]
            elif mutation=='case_id':data['case_id'][0]='wrong'
            elif mutation=='horizon':data['forecast_time'][0,-1]+=3600
            elif mutation=='schema':data['feature_schema']=np.array(FULL_FEATURE_SCHEMA)
            elif mutation=='token':data['weather'][0,0,0]=2
            elif mutation=='batch_bound':data['origin_time']=np.zeros(129,dtype=np.int64)
            else:data['weather_mask'][0,0,0]=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                validate_inputs(data,history_hours=24,feature_schema=Q_ONLY_FEATURE_SCHEMA,
                    expected_cohort_admission_digest='b'*64,expected_gauge_id='03161000')


if __name__=='__main__':unittest.main()
