"""Qualification scheduling/provenance tests; no optimizers or model fits."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import qualify_gap_pilot as q
from gap_pilot_fixture import training_fixture,inference_fixture,PoisonLabels,synthetic_q
from hydra_directflow.reference import ASSESSMENT_START
from hydra_directflow.training import validate_coordinates


class QualificationTests(unittest.TestCase):
    def test_fixed_schedule_is_exactly_sixteen_updates_and_one_separate_ridge_fit(self):
        workers=q.workers()
        self.assertEqual(len(workers),9);self.assertEqual(len({w['id'] for w in workers}),9)
        self.assertEqual(sum(w['updates'] for w in workers),16)
        self.assertEqual(sum(w['ridge_fits'] for w in workers),1)
        for mode in ('clean','augmented'):
            self.assertEqual([w['updates'] for w in workers if w['mode']==mode],[4,2,2,0])

    def test_plan_only_binds_scope_and_performs_no_execute(self):
        with patch.object(q,'implementation',return_value={'code':{},'dependencies':{}}),patch.object(q,'execute',side_effect=AssertionError):
            plan=q.plan();self.assertEqual(plan,q.plan())
        self.assertEqual(plan['plan_sha256'],q.sha(q.canonical(plan['spec'])))
        self.assertFalse(plan['spec']['scientific_fits_authorized'])
        self.assertFalse(plan['spec']['protected_assessment_authorized'])

    def test_source_fixture_is_consistent_and_separate_label_free_inputs_exist(self):
        arrays,provenance,source,cohort=training_fixture()
        self.assertEqual(validate_coordinates(arrays,168),192)
        self.assertTrue(np.all(arrays['target_time']<ASSESSMENT_START))
        np.testing.assert_array_equal(arrays['history'][...,0],synthetic_q(arrays['history_time']))
        self.assertEqual(provenance['observation_times'],arrays['history_time'].tolist())
        inputs,_,_,_=inference_fixture();self.assertNotIn('target',inputs);self.assertNotIn('target_mask',inputs)
        with self.assertRaises(AssertionError):PoisonLabels(inputs)['target']
        self.assertNotEqual(source,inference_fixture()[2]);self.assertNotEqual(cohort,inference_fixture()[3])

    def test_counter_validation_never_treats_uncertainty_as_confirmed_updates(self):
        self.assertEqual(q.checked_counts({'optimizer_updates_entered':2,'optimizer_updates_confirmed':1},2),(2,1))
        for value in ({'optimizer_updates_entered':True},{'optimizer_updates_entered':3},
                      {'optimizer_updates_entered':1,'optimizer_updates_confirmed':2},
                      {'optimizer_updates_entered':-1}):
            with self.assertRaises(ValueError):q.checked_counts(value,2)

    def test_precharge_and_worker_claim_are_not_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run'
            with patch.object(q,'LEDGER',root/'ledger.json'),patch.object(q,'LOCK',root/'ledger.lock'),\
                 patch.object(q,'implementation',return_value={'code':{},'dependencies':{}}):
                prepared=q.plan();q.precharge(prepared,run)
                with self.assertRaises(ValueError):q.precharge(prepared,run)
                w=q.workers()[0];q.claim(run,w,prepared)
                with self.assertRaises(ValueError):q.claim(run,w,prepared)
                saved=json.loads((root/'ledger.json').read_text())
                self.assertEqual(saved['precharged_optimizer_updates'],16)
                self.assertEqual(saved['precharged_synthetic_ridge_fits'],1)
                self.assertEqual(saved['workers']['ridge']['status'],'CLAIMED')

    def test_wrong_plan_rejects_before_precharge_or_subprocess(self):
        with patch.object(q,'implementation',return_value={'code':{},'dependencies':{}}),\
             patch.object(q,'precharge',side_effect=AssertionError('Must not charge')):
            with self.assertRaises(ValueError):q.execute(q.plan(),Path('unused'),Path('unused-report'),'0'*64)

    def test_unsupported_scenario_cannot_hide_an_unrelated_prediction_failure(self):
        error=ValueError('Pilot inference has failed/unsupported fills; no silent filtering or forwarding')
        self.assertTrue(q.expected_inadmissibility(error,[True,False]))
        self.assertFalse(q.expected_inadmissibility(error,[True,True]))
        self.assertFalse(q.expected_inadmissibility(ValueError('wrong source'),[False]))
        self.assertFalse(q.expected_inadmissibility(RuntimeError(str(error)),[False]))


if __name__=='__main__':unittest.main()
