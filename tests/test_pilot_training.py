"""Synthetic H4 mechanics, zero optimizer updates; no scientific execution."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from gap_pilot_fixture import training_fixture,inference_fixture,ROLE_BOUNDS,REFIT,PoisonLabels
from hydra_directflow.reference import digest,ASSESSMENT_START
from hydra_directflow.pilot_inputs import make_fill_policy,make_augmentation_recipe,prepare_h4_inputs
from hydra_directflow.training import TrainOnlyTransform,EarlyStopping,EpochSampler
from hydra_directflow.pilot_training import PilotSpec,PilotTrainer,PilotBestPredictor,validate_pilot_roles,_canonical


def setup_args(mode='clean',max_updates=0):
    arrays,provenance,source,cohort=training_fixture()
    train,inner=tuple(range(128)),tuple(range(128,192))
    policy=make_fill_policy()
    recipe=make_augmentation_recipe(mixture={'interior_1_middle':.5,'trailing_2_carry':.5},
        gradient_case_ids_sha256=digest(arrays['case_id'][:128].tolist()),source_identity=source,evidence_sha256='e'*64)
    return arrays,dict(train_indices=train,inner_indices=inner,role_time_bounds=ROLE_BOUNDS,refit_at_utc=REFIT,
        spec=PilotSpec(mode=mode,max_updates=max_updates),source_identity=source,cohort_digest=cohort,
        history_provenance=provenance,fill_policy=policy,augmentation_recipe=recipe)


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'pilot.pt'
    def tearDown(self):self.temp.cleanup()

    def trainer(self,mode='clean',max_updates=0):
        arrays,args=setup_args(mode,max_updates)
        with patch('torch.optim.AdamW',side_effect=AssertionError('Unexpected optimizer construction')):
            trainer=PilotTrainer(arrays,**args)
        return trainer,arrays,args

    def restore(self,arrays,args,sha,**changes):
        kwargs=dict(history_provenance=args['history_provenance'],expected_sha256=sha,
            expected_source_identity=args['source_identity'],expected_cohort_digest=args['cohort_digest'],
            expected_mode=args['spec'].mode,expected_fill_policy=args['fill_policy'],expected_augmentation_recipe=args['augmentation_recipe'])
        kwargs.update(changes)
        with (patch.object(TrainOnlyTransform,'fit',side_effect=AssertionError('Restore refitted scaler')),
              patch('torch.optim.AdamW',side_effect=AssertionError('Zero-update restore built optimizer'))):
            return PilotTrainer.restore(self.path,arrays,**kwargs)

    def handbuilt_best_checkpoint(self):
        """Artificial state/ledger fixture only: no actual optimizer was executed."""
        trainer,arrays,args=self.trainer(max_updates=4)
        trainer.checkpoint(self.path)
        value=torch.load(self.path,weights_only=True)
        value['best_model']={k:v.clone() for k,v in value['model'].items()}
        value['model']['flow_head.bias']+=100.
        value['updates']=value['updates_entered']=4
        value['optimizer']={'synthetic_placeholder_not_executed':True}
        sampler=EpochSampler(tuple(range(128)),seed=17,shuffle=True,epoch=1,cursor=128)
        value['sampler']=sampler.state()
        value['update_records']=[dict(epoch=(i-1)//2,updates=i,updates_entered=i,loss=1.,
            **{k:'a'*64 for k in ('mask_digest','adapter_receipt_sha256','provenance_sha256','batch_identity')}) for i in range(1,5)]
        stopping=EarlyStopping(2,1,2,0.);stopping.observe(1.);stopping.observe(2.)
        value['stopping']=asdict(stopping)
        value['epoch_records']=[dict(epoch=1,inner_stopping_loss=1.,updates=2,improved=True),
                                dict(epoch=2,inner_stopping_loss=2.,updates=4,improved=False)]
        torch.save(value,self.path)
        return value,arrays,args,hashlib.sha256(self.path.read_bytes()).hexdigest()

    def load_best(self,arrays,args,sha,**changes):
        kwargs=dict(expected_sha256=sha,expected_source_identity=args['source_identity'],expected_cohort_digest=args['cohort_digest'],
            expected_mode=args['spec'].mode,expected_fill_policy=args['fill_policy'],expected_augmentation_recipe=args['augmentation_recipe'],
            gradient_fit_case_ids=tuple(arrays['case_id'][:128]),expected_gauge_id='03161000')
        kwargs.update(changes)
        with (patch.object(TrainOnlyTransform,'fit',side_effect=AssertionError('Inference refitted scaler')),
              patch('torch.optim.AdamW',side_effect=AssertionError('Inference created optimizer'))):
            return PilotBestPredictor.load(self.path,**kwargs)

    def test_fixed_pilot_spec_and_zero_allowance_do_not_construct_or_step_optimizer(self):
        trainer,_,_=self.trainer()
        initial=trainer.sampler.state()
        with patch('torch.optim.AdamW',side_effect=AssertionError('Optimizer')):
            with self.assertRaisesRegex(ValueError,'allowance'):trainer.update()
        self.assertEqual(trainer.sampler.state(),initial);self.assertEqual(trainer.updates,0)
        for change in ({'device':'mps'},{'history_hours':24},{'architecture':'Transformer'},{'batch_size':65},{'max_updates':-1}):
            with self.subTest(change=change),self.assertRaises(ValueError):PilotSpec(mode='clean',**change)

    def test_rejected_adapter_batch_does_not_skip_sampler_cases_or_construct_optimizer(self):
        trainer,_,_=self.trainer('augmented',max_updates=1)
        initial=trainer.sampler.state()
        with (patch.object(trainer,'preview_batch',side_effect=ValueError('Unsupported synthetic fill')),
              patch('torch.optim.AdamW',side_effect=AssertionError('Optimizer'))):
            with self.assertRaisesRegex(ValueError,'Unsupported'):trainer.update()
        self.assertEqual(trainer.sampler.state(),initial)
        self.assertEqual(trainer.updates_entered,0);self.assertEqual(trainer.updates,0)

    def test_policy_transform_and_source_provenance_cannot_drift_between_batches(self):
        trainer,_,_=self.trainer('augmented')
        with self.assertRaises(TypeError):trainer.history_provenance['source_ids'][0][0]='changed'
        trainer.fill_policy['trailing_method']='ridge'
        with self.assertRaisesRegex(ValueError,'mutated'):trainer.preview_next_batch()
        trainer.fill_policy['trailing_method']='carry';trainer.scaler.parameters['history'][0]['mean']+=1
        with self.assertRaisesRegex(ValueError,'mutated'):trainer.preview_next_batch()
        with self.assertRaisesRegex(ValueError,'mutated'):trainer.checkpoint(self.path)
        other,_,_=self.trainer()
        with self.assertRaisesRegex(ValueError,'at most50'):other.run_epoch(EarlyStopping(51,1,1,0.))

    def test_role_and_refit_guards_precede_all_numeric_reads(self):
        class PoisonValues(dict):
            def __getitem__(self,key):
                if key in ('history','weather','state','target','target_mask'):raise AssertionError('Numeric read before clock guard')
                return super().__getitem__(key)
        for mutation in ('protected','fit_boundary','late_inner','wrong_case','wrong_order'):
            arrays,args=setup_args()
            if mutation=='protected':arrays['target_time'][0,-1]=ASSESSMENT_START
            elif mutation=='fit_boundary':args['role_time_bounds']=deepcopy(ROLE_BOUNDS);args['role_time_bounds']['fit']['end_utc']='2019-01-03T00:00:00Z'
            elif mutation=='late_inner':args['refit_at_utc']='2019-01-19T00:00:00Z'
            elif mutation=='wrong_case':arrays['case_id'][0]='03164000:2019-01-02T00:00:00Z'
            else:arrays['origin_time'][0]=arrays['origin_time'][1]
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):PilotTrainer(PoisonValues(arrays),**args)

    def test_complete_original_roles_provenance_and_recipe_are_required(self):
        for mutation in ('label_mask','history_mask','source_time','qualifier','recipe_source','recipe_fit_ids'):
            arrays,args=setup_args('augmented')
            if mutation=='label_mask':arrays['target_mask'][0,0]=False
            elif mutation=='history_mask':arrays['history_mask'][0,10]=False
            elif mutation=='source_time':args['history_provenance']['observation_times'][0][0]+=3600
            elif mutation=='qualifier':args['history_provenance']['quality_flags'][0][0]=['e']
            elif mutation=='recipe_source':args['augmentation_recipe']['evidence']['source_identity']='b'*64
            else:args['augmentation_recipe']['evidence']['gradient_case_ids_sha256']='b'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):PilotTrainer(arrays,**args)

    def test_clean_and_augmented_share_original_scaler_and_case_epoch_masks_are_replayable(self):
        clean,arrays,_=self.trainer('clean');aug,_,_=self.trainer('augmented')
        self.assertEqual(asdict(clean.scaler),asdict(aug.scaler))
        indices=(17,2,45,11,100,9,8,55)
        batch,a=aug.preview_batch(indices,epoch=1,augment=True)
        again,b=aug.preview_batch(indices,epoch=1,augment=True)
        reversed_batch,c=aug.preview_batch(tuple(reversed(indices)),epoch=1,augment=True)
        self.assertEqual(a,b)
        self.assertTrue(batch['history_mask'].all());self.assertGreater(int(batch['history'][...,2].sum()),0)
        for name in batch:
            np.testing.assert_array_equal(batch[name],again[name]);np.testing.assert_array_equal(batch[name][::-1],reversed_batch[name])
        _,other=aug.preview_batch(indices,epoch=2,augment=True)
        self.assertNotEqual(a['mask_digest'],other['mask_digest'])
        self.assertEqual(clean.preview_batch(indices,epoch=1)[1]['augmented'],False)
        self.assertFalse(aug.arrays['history'].flags.writeable)
        saved=aug.arrays['history'].copy();arrays['history'][:]=0;np.testing.assert_array_equal(aug.arrays['history'],saved)

    def test_clean_inner_monitor_never_uses_augmented_scenarios(self):
        trainer,_,_=self.trainer('augmented')
        import hydra_directflow.pilot_training as module
        original=module._prepare;seen=[]
        def traced(*args,**kwargs):seen.append(kwargs['scenario_id']);return original(*args,**kwargs)
        with patch.object(module,'_prepare',side_effect=traced):loss=trainer.stopping_loss()
        self.assertTrue(np.isfinite(loss));self.assertGreaterEqual(loss,0.);self.assertEqual(set(seen),{'clean'})
        self.assertEqual(trainer.updates,0);self.assertIsNone(trainer.optimizer)

    def test_checkpoint_zero_update_restore_preserves_rng_sampler_masks_and_saved_transform(self):
        trainer,arrays,args=self.trainer('augmented')
        expected=trainer.preview_next_batch()[1];rng=torch.get_rng_state().clone()
        sha=trainer.checkpoint(self.path);restored=self.restore(arrays,args,sha)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(torch.equal(trainer.cpu_rng,restored.cpu_rng))
        self.assertEqual(restored.sampler.state(),trainer.sampler.state());self.assertEqual(restored.preview_next_batch()[1],expected)
        self.assertEqual(asdict(restored.scaler),asdict(trainer.scaler))
        for key,value in trainer.model.state_dict().items():self.assertTrue(torch.equal(value,restored.model.state_dict()[key]))
        with self.assertRaises(ValueError):restored.checkpoint(self.path)

    def test_checkpoint_mode_policy_source_dataset_and_provenance_substitutions_reject(self):
        trainer,arrays,args=self.trainer('augmented');sha=trainer.checkpoint(self.path)
        for changes in ({'expected_mode':'clean'},{'expected_source_identity':'f'*64},{'expected_cohort_digest':'f'*64},
                        {'expected_sha256':'f'*64},{'expected_fill_policy':make_fill_policy(trailing_method='ridge')}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.restore(arrays,args,sha,**changes)
        changed={k:v.copy() for k,v in arrays.items()};changed['history'][0,0,0]+=1
        with self.assertRaises(ValueError):self.restore(changed,args,sha)
        provenance=deepcopy(args['history_provenance']);provenance['source_ids'][0][0]+='/different'
        with self.assertRaises(ValueError):self.restore(arrays,args,sha,history_provenance=provenance)

    def test_best_predictor_is_label_free_uses_best_not_last_and_does_not_fit(self):
        checkpoint,arrays,args,sha=self.handbuilt_best_checkpoint();predictor=self.load_best(arrays,args,sha)
        inputs,provenance,source,cohort=inference_fixture()
        result=predictor.predict(PoisonLabels(inputs),history_provenance=provenance,
            expected_inference_source_identity=source,expected_inference_cohort_digest=cohort)
        prepared=prepare_h4_inputs(PoisonLabels(inputs),history_provenance=provenance,saved_h2_transform=predictor.scaler,
            inference_source_identity=source,fill_policy=args['fill_policy'])
        with torch.inference_mode():
            direct=predictor.model(*(torch.from_numpy(np.array(prepared['transformed_inputs'][key],copy=True))
                for key in ('history','history_mask','weather','weather_mask','state','state_mask')))
        np.testing.assert_array_equal(result['prediction_q'],predictor.scaler.inverse_target(direct.numpy().astype(np.float64)))
        self.assertEqual(result['receipt']['selected_best_epoch'],1);self.assertFalse(result['receipt']['last_state_used'])
        self.assertTrue(torch.equal(predictor.model.state_dict()['flow_head.bias'],checkpoint['best_model']['flow_head.bias']))
        self.assertFalse(result['receipt']['target_values_read']);self.assertFalse(result['receipt']['optimizer_constructed'])

    def test_best_prediction_rejects_cross_gauge_and_wrong_outer_clocks_before_values(self):
        _,arrays,args,sha=self.handbuilt_best_checkpoint();predictor=self.load_best(arrays,args,sha)
        class PoisonInputs(PoisonLabels):
            def __getitem__(self,key):
                if key in ('history','weather','state'):raise AssertionError('Read values before identity gate')
                return super().__getitem__(key)
        for mutation in ('gauge','protected','outer'):
            inputs,provenance,source,cohort=inference_fixture()
            if mutation=='gauge':inputs['site_id'][:]=1;inputs['case_id']=np.array([x.replace('03161000','03164000') for x in inputs['case_id']])
            else:
                shift=ASSESSMENT_START-int(inputs['origin_time'][0]) if mutation=='protected' else -21*86400
                for name in ('origin_time','history_time','forecast_time'):inputs[name]+=shift
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                predictor.predict(PoisonInputs(inputs),history_provenance=provenance,expected_inference_source_identity=source,expected_inference_cohort_digest=cohort)
        with self.assertRaises(ValueError):self.load_best(arrays,args,sha,expected_gauge_id='03164000')

    def test_absent_best_state_or_inconsistent_update_journal_rejects_without_last_fallback(self):
        checkpoint,arrays,args,_=self.handbuilt_best_checkpoint()
        for mutation in ('best','stopping','journal','sampler','transform'):
            value=deepcopy(checkpoint)
            if mutation=='best':value['best_model']=None
            elif mutation=='stopping':value['epoch_records'][1]['improved']=True
            elif mutation=='journal':value['update_records'][0]['updates']=2
            elif mutation=='sampler':value['sampler']['cursor']=64
            else:value['transform']['parameters']['history'][0]['mean']+=1
            torch.save(value,self.path);sha=hashlib.sha256(self.path.read_bytes()).hexdigest()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.load_best(arrays,args,sha)

    def test_canonical_provenance_digest_accepts_readonly_mappings_and_numpy_masks(self):
        from types import MappingProxyType
        value=MappingProxyType({'a':np.array([True,False]),'b':(np.int64(2),('A',))})
        self.assertEqual(_canonical(value),{'a':[True,False],'b':[2,['A']]})


if __name__=='__main__':unittest.main()
