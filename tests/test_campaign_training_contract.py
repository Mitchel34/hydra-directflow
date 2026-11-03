"""Higher-cap/source-binding changes; no models, optimizer or actual data."""
from dataclasses import asdict
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from hydra_directflow.training import TrainOnlyTransform,TrainerSpec,EarlyStopping,MAX_UPDATES_PER_FIT,DevelopmentTrainer
from test_training import fixture


class CampaignTrainingContractTests(unittest.TestCase):
    def test200_epoch_update_bound_is_finite_and_exact_job_can_fit(self):
        args=dict(architecture='LSTM',device='cpu',history_hours=168,batch_size=64,learning_rate=.0001,
                  seed=17,shuffle_training=True)
        self.assertEqual(TrainerSpec(**args,max_updates=126800).max_updates,126800)
        self.assertEqual(MAX_UPDATES_PER_FIT,200000)
        for limit in (200001,True,-1):
            with self.assertRaises(ValueError):TrainerSpec(**args,max_updates=limit)

    def test_saved_transform_checks_gradient_identity_without_refitting(self):
        arrays=fixture();saved=TrainOnlyTransform.fit(arrays,[0,1])
        with patch.object(TrainOnlyTransform,'fit',side_effect=AssertionError('Refit forbidden')):
            loaded=TrainOnlyTransform.from_dict(asdict(saved))
            loaded.validate_training_source(arrays,[0,1])
            arrays['target'][2:]=1e12
            loaded.validate_training_source(arrays,[0,1])
            arrays['target'][0,0]+=1
            with self.assertRaisesRegex(ValueError,'source differs'):loaded.validate_training_source(arrays,[0,1])

    def test_changed_gradient_membership_and_saved_schema_reject(self):
        arrays=fixture();saved=TrainOnlyTransform.fit(arrays,[0,1])
        for indices in ([1,0],[0,2],[0,0],[-1,1]):
            with self.subTest(indices=indices),self.assertRaises(ValueError):saved.validate_training_source(arrays,indices)

    def test_high_ceiling_does_not_force200_epochs_or_replace_ties(self):
        stop=EarlyStopping(200,10,5,0);best=[]
        for i,loss in enumerate([1/(j+1) for j in range(10)]+[.1]*5,1):
            if stop.observe(loss):best.append(i)
        self.assertTrue(stop.finished);self.assertEqual(stop.completed_epochs,15);self.assertEqual(best[-1],10)

    def test_epoch_observer_sees_completed_stopping_and_best_state_without_extra_updates(self):
        class Tensor: pass
        fake=SimpleNamespace(stopping=None,sampler=SimpleNamespace(epoch=0,cursor=0),train=(0,),updates=0,
                             epoch_records=[],torch=SimpleNamespace(Tensor=Tensor),best_model_state=None)
        def update():
            if fake.sampler.cursor==1:fake.sampler.epoch+=1;fake.sampler.cursor=0
            fake.sampler.cursor+=1;fake.updates+=1
        losses=iter((1.,2.));fake.update=update;fake.stopping_loss=lambda:next(losses)
        fake.model=SimpleNamespace(state_dict=lambda:{'synthetic_update':fake.updates})
        seen=[]
        DevelopmentTrainer.fit_epochs(fake,EarlyStopping(2,2,5,0),after_epoch=lambda t:
            seen.append((t.stopping.completed_epochs,t.updates,t.best_model_state['synthetic_update'],len(t.epoch_records))))
        self.assertEqual(seen,[(1,1,1,1),(2,2,1,2)])
        self.assertEqual(fake.updates,2)


if __name__=='__main__':unittest.main()
