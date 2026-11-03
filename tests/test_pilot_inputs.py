"""Synthetic H4 adapter/scenario checks; no fits, optimizer or real data."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from hydra_directflow import pilot_inputs as pilot
from hydra_directflow.local_data import iso, utc_epoch
from hydra_directflow.reference import digest
from hydra_directflow.training import TrainOnlyTransform, FULL_FEATURE_SCHEMA

SOURCE='b'*64


class PoisonLabels(dict):
    def __getitem__(self,key):
        if key in ('target','target_mask','future_q'):raise AssertionError('Label/future flow accessed: '+key)
        return super().__getitem__(key)


def fixture(n=2):
    origins=utc_epoch('2023-08-10T00:00:00Z')+np.arange(n,dtype=np.int64)*3600
    times=origins[:,None]-np.arange(169,1,-1)*3600
    arrays=dict(origin_time=origins,history_time=times,target_time=origins[:,None]+np.arange(1,19)*3600,
        site_id=np.zeros(n,np.int64),case_id=np.array(['03161000:'+iso(int(t)) for t in origins]),
        feature_schema=np.array(FULL_FEATURE_SCHEMA),cohort_admission_digest=np.array('c'*64),
        history=np.stack((np.tile(np.arange(168,dtype=float)+10,(n,1)),np.tile(np.arange(169,1,-1,dtype=float),(n,1))),axis=-1),
        weather=np.ones((n,18,4),np.float64),state=np.tile(np.array([.25,4.]),(n,1)))
    for name in ('history','weather','state'):arrays[name+'_mask']=np.ones(arrays[name].shape,bool)
    provenance=dict(observation_times=tuple(tuple(map(int,row)) for row in times),
        quality_flags=tuple(tuple(('R','A','A') for _ in range(168)) for _ in range(n)),
        source_ids=tuple(tuple('synthetic-source:'+str(int(t)) for t in row) for row in times))
    parameters={name:[dict(mean=0.,scale=1.,constant=False,cells=100) for _ in range(count)]
                for name,count in [('history',2),('weather',4),('state',2),('target',1)]}
    parameters['history']=[dict(mean=100.,scale=2.,constant=False,cells=100),dict(mean=50.,scale=5.,constant=False,cells=100)]
    saved=TrainOnlyTransform(parameters,digest(arrays['case_id'].tolist()),'d'*64)
    return arrays,provenance,saved


def fake_filler(**kwargs):
    """Synthetic no-fill refusal used to verify adapter boundaries only."""
    q=np.array(kwargs['q_values'],copy=True);mask=np.array(kwargs['approved_mask'],copy=True)
    return dict(usable_q=q,usable_mask=mask,supplied_approved_mask=mask.copy(),slot_provenance=tuple({} for _ in range(168)),
                success=bool(mask.all()),failure_reasons=[] if mask.all() else ['SYNTHETIC_REFUSAL'],policy_fingerprint='e'*64)


class PilotInputTests(unittest.TestCase):
    def prepare(self,arrays,provenance,saved,**kwargs):
        return pilot.prepare_h4_inputs(PoisonLabels(arrays),history_provenance=provenance,saved_h2_transform=saved,
            inference_source_identity=SOURCE,fill_policy=pilot.make_fill_policy(),**kwargs)

    def test_manifest_has17_bounded_paired_scenarios_and_no_leading_holes(self):
        manifest=pilot.scenario_manifest();self.assertEqual(manifest['scenario_count'],17)
        self.assertEqual(len({r['id'] for r in manifest['scenarios']}),17)
        self.assertEqual(manifest['layout_policy_status'],'PROPOSED_NOT_FROZEN')
        case='03161000:2023-08-10T00:00:00Z'
        for row in manifest['scenarios']:
            mask=pilot.artificial_mask(case_id=case,scenario_id=row['id'],epoch=2,mask_seed=17)
            self.assertFalse(mask[0]);self.assertEqual(mask.sum(),row['interior_hours']+row['trailing_hours'])
            if row['trailing_method']=='carry':
                paired=pilot.artificial_mask(case_id=case,scenario_id=row['id'].replace('carry','ridge'),epoch=2,mask_seed=17)
                np.testing.assert_array_equal(mask,paired)

    def test_mask_recipe_replays_across_batch_order_without_global_rng_changes(self):
        before=np.random.get_state();cases=['03161000:2023-08-10T00:00:00Z','03161000:2023-08-10T01:00:00Z']
        first={c:pilot.artificial_mask(case_id=c,scenario_id='mixed_recent_carry',epoch=7,mask_seed=17) for c in cases}
        for case in reversed(cases):np.testing.assert_array_equal(first[case],pilot.artificial_mask(case_id=case,scenario_id='mixed_recent_carry',epoch=7,mask_seed=17))
        after=np.random.get_state();self.assertEqual(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[2:],after[2:])

    def test_shuffled_training_batch_preserves_order_and_per_case_masks(self):
        arrays,provenance,saved=fixture(3);order=[2,0,1]
        shuffled={key:value[order].copy() if value.ndim and len(value)==3 else value.copy() for key,value in arrays.items()}
        shuffled_provenance={key:tuple(value[i] for i in order) for key,value in provenance.items()}
        with patch.dict(sys.modules,{'hydra_directflow.gap_filling':SimpleNamespace(fill_history=fake_filler)}):
            original=self.prepare(arrays,provenance,saved,scenario_id='mixed_middle_carry',epoch=3,mask_seed=17)
            changed=self.prepare(shuffled,shuffled_provenance,saved,scenario_id='mixed_middle_carry',epoch=3,mask_seed=17)
        np.testing.assert_array_equal(changed['raw_inputs']['case_id'],arrays['case_id'][order])
        np.testing.assert_array_equal(changed['provenance']['artificial_hidden_mask'],original['provenance']['artificial_hidden_mask'][order])

    def test_augmentation_requires_gradient_evidence_half_probability_and_normalized_mixture(self):
        arrays,_,saved=fixture();recipe=pilot.make_augmentation_recipe(mixture={'interior_1_early':.3,'mixed_recent_carry':.7},
            gradient_case_ids_sha256=saved.training_case_ids_sha256,source_identity=SOURCE,evidence_sha256='f'*64)
        first=[pilot.augmentation_scenario(case_id=str(arrays['case_id'][0]),epoch=e,mask_seed=17,recipe=recipe) for e in range(64)]
        again=[pilot.augmentation_scenario(case_id=str(arrays['case_id'][0]),epoch=e,mask_seed=17,recipe=deepcopy(recipe)) for e in range(64)]
        self.assertEqual(first,again);self.assertIn('clean',first);self.assertTrue(set(first)-{'clean'})
        for mutation in ({'probability':1.},{'mixture':{'interior_1_early':.4}},{'evidence':{**recipe['evidence'],'scope':'VALIDATION'}}):
            with self.assertRaises(ValueError):pilot.validate_augmentation_recipe({**recipe,**mutation})

    def test_hiding_precedes_filler_and_hidden_true_values_are_not_returned(self):
        arrays,provenance,saved=fixture(1);before=deepcopy(arrays);scenario='interior_2_middle'
        hidden=pilot.artificial_mask(case_id=str(arrays['case_id'][0]),scenario_id=scenario)
        def inspect(**kwargs):
            self.assertTrue(np.isnan(kwargs['q_values'][hidden]).all());self.assertFalse(kwargs['approved_mask'][hidden].any())
            self.assertEqual(kwargs['source_flags'][np.flatnonzero(hidden)[0]],('R','A','A'))
            return fake_filler(**kwargs)
        with patch.dict(sys.modules,{'hydra_directflow.gap_filling':SimpleNamespace(fill_history=inspect)}),\
             patch.object(TrainOnlyTransform,'fit',side_effect=AssertionError('Adapter fitted statistics')):
            out=self.prepare(arrays,provenance,saved,scenario_id=scenario)
        self.assertTrue(np.isnan(out['raw_inputs']['history'][0,hidden,0]).all())
        self.assertFalse(out['case_admissible'][0]);self.assertIn('SYNTHETIC_REFUSAL',out['provenance']['failure_reasons'][0])
        self.assertTrue(out['provenance']['original_observation_mask'].all());self.assertFalse(out['provenance']['effective_observation_mask'][0,hidden].any())
        for key in arrays:np.testing.assert_array_equal(arrays[key],before[key])
        self.assertNotIn('target',out['raw_inputs']);self.assertNotIn('target_mask',out['transformed_inputs'])

    def test_clean_h4_reuses_saved_h2_stats_and_fixed_new_channels(self):
        arrays,provenance,saved=fixture()
        with patch.dict(sys.modules,{'hydra_directflow.gap_filling':SimpleNamespace(fill_history=fake_filler)}):out=self.prepare(arrays,provenance,asdict(saved))
        self.assertEqual(out['raw_inputs']['history'].shape,(2,168,4));self.assertTrue(out['case_admissible'].all())
        np.testing.assert_array_equal(out['transformed_inputs']['history'][...,0],(arrays['history'][...,0]-100)/2)
        np.testing.assert_allclose(out['transformed_inputs']['history'][...,1],(arrays['history'][...,1]-50)/5)
        self.assertTrue((out['raw_inputs']['history'][...,2:]==0).all())
        self.assertFalse(out['raw_inputs']['history'].flags.writeable);self.assertFalse(out['provenance']['original_observation_mask'].flags.writeable)
        self.assertEqual(out['provenance']['quality_flags'],provenance['quality_flags'])
        self.assertEqual(out['provenance']['source_ids'],provenance['source_ids'])

    def test_bound_ridge_artifact_is_forwarded_only_to_explicit_ridge_cases(self):
        arrays,provenance,saved=fixture(1);sentinel=object();calls=[]
        def inspect(**kwargs):
            calls.append((kwargs['trailing_method'],kwargs['ridge_model'],kwargs['expected_ridge_identity']))
            return fake_filler(**kwargs)
        with patch.dict(sys.modules,{'hydra_directflow.gap_filling':SimpleNamespace(fill_history=inspect)}):
            self.prepare(arrays,provenance,saved,scenario_id='trailing_1_carry',ridge_model=sentinel,expected_ridge_identity='a'*64)
            self.prepare(arrays,provenance,saved,scenario_id='trailing_1_ridge',ridge_model=sentinel,expected_ridge_identity='a'*64)
        self.assertEqual(calls[0],('carry',None,None));self.assertEqual(calls[1],('ridge',sentinel,'a'*64))

    def test_protected_clock_rejected_before_numerical_values_or_provenance(self):
        arrays,_,saved=fixture(1);shift=utc_epoch('2024-07-31T00:00:00Z')-int(arrays['origin_time'][0])
        for key in ('origin_time','history_time','target_time'):arrays[key]+=shift
        class PoisonInputs(PoisonLabels):
            def __getitem__(self,key):
                if key in ('history','history_mask','weather','state'):raise AssertionError('Numerical input read before clock gate')
                return super().__getitem__(key)
        with self.assertRaisesRegex(ValueError,'Protected'):
            pilot.prepare_h4_inputs(PoisonInputs(arrays),history_provenance={},saved_h2_transform=saved,inference_source_identity=SOURCE,fill_policy=pilot.make_fill_policy())

    def test_original_approved_qualifiers_cannot_be_faked_or_bypassed_by_hiding(self):
        arrays,provenance,saved=fixture(1);hidden=pilot.artificial_mask(case_id=str(arrays['case_id'][0]),scenario_id='interior_1_early')
        changed=deepcopy(provenance);flags=[list(row) for row in changed['quality_flags']];flags[0][np.flatnonzero(hidden)[0]]=('P',);changed['quality_flags']=flags
        with self.assertRaisesRegex(ValueError,'qualifier'):self.prepare(arrays,changed,saved,scenario_id='interior_1_early')
        with self.assertRaisesRegex(ValueError,'per-slot'):self.prepare(arrays,{},saved)

    def test_artificial_scenarios_cannot_hide_on_top_of_preexisting_missingness(self):
        arrays,provenance,saved=fixture(1);arrays['history_mask'][0,5]=False;arrays['history'][0,5]=np.nan
        with patch.dict(sys.modules,{'hydra_directflow.gap_filling':SimpleNamespace(fill_history=fake_filler)}):
            with self.assertRaisesRegex(ValueError,'originally complete'):self.prepare(arrays,provenance,saved,scenario_id='trailing_1_carry')

    def test_mismatched_schema_masks_age_or_augmentation_source_rejected(self):
        arrays,provenance,saved=fixture(1)
        changed=deepcopy(arrays);changed['history_mask'][0,0,1]=False
        with self.assertRaisesRegex(ValueError,'Q/age masks'):self.prepare(changed,provenance,saved)
        changed=deepcopy(arrays);changed['history'][0,0,1]+=1
        with self.assertRaisesRegex(ValueError,'coordinate-age'):self.prepare(changed,provenance,saved)
        changed=deepcopy(arrays);changed['feature_schema']=np.array(pilot.H4_FEATURE_SCHEMA)
        with self.assertRaises(ValueError):self.prepare(changed,provenance,saved)
        recipe=pilot.make_augmentation_recipe(mixture={'interior_1_early':1.},gradient_case_ids_sha256=saved.training_case_ids_sha256,source_identity='0'*64,evidence_sha256='f'*64)
        with self.assertRaisesRegex(ValueError,'evidence differs'):self.prepare(arrays,provenance,saved,scenario_id='augmentation',augmentation_recipe=recipe)

    def test_original_elapsed_clock_ignores_filled_slots_and_failed_case_is_not_admitted(self):
        arrays,provenance,saved=fixture(1)
        def successful_fill(**kwargs):
            result=fake_filler(**kwargs);q=result['usable_q'];mask=result['usable_mask']
            for i in np.flatnonzero(~mask):q[i]=q[i-1];mask[i]=True
            return {**result,'usable_q':q,'usable_mask':mask,'success':True,'failure_reasons':[]}
        with patch.dict(sys.modules,{'hydra_directflow.gap_filling':SimpleNamespace(fill_history=successful_fill)}):
            out=self.prepare(arrays,provenance,saved,scenario_id='trailing_2_carry')
        self.assertTrue(out['case_admissible'][0]);np.testing.assert_array_equal(out['raw_inputs']['history'][0,-2:,3],[1.,2.])
        np.testing.assert_allclose(out['transformed_inputs']['history'][0,-2:,3],[1/168,2/168])
        self.assertTrue(out['raw_inputs']['history_mask'].all());self.assertFalse(out['provenance']['effective_observation_mask'][0,-2:].any())
        self.assertTrue(out['provenance']['usable_q_mask'].all())

    def test_actual_linear_filler_is_invariant_to_hidden_truth_and_preserves_other_inputs(self):
        arrays,provenance,saved=fixture(1);scenario='interior_2_middle'
        hidden=pilot.artificial_mask(case_id=str(arrays['case_id'][0]),scenario_id=scenario)
        changed=deepcopy(arrays);changed['history'][0,hidden,0]=1e12
        with patch.object(TrainOnlyTransform,'fit',side_effect=AssertionError('Adapter fitted statistics')):
            first=self.prepare(arrays,provenance,saved,scenario_id=scenario)
            second=self.prepare(changed,provenance,saved,scenario_id=scenario)
        self.assertTrue(first['case_admissible'][0]);self.assertTrue(second['case_admissible'][0])
        np.testing.assert_array_equal(first['transformed_inputs']['history'],second['transformed_inputs']['history'])
        for name in ('weather','state'):
            np.testing.assert_array_equal(first['raw_inputs'][name],arrays[name]);np.testing.assert_array_equal(first['raw_inputs'][name+'_mask'],arrays[name+'_mask'])
        self.assertTrue(first['provenance']['usable_q_mask'][0,hidden].all())
        self.assertTrue((first['raw_inputs']['history'][0,hidden,2]==1).all())

    def test_actual_carry_fills_two_trailing_slots_without_resetting_real_elapsed_clock(self):
        arrays,provenance,saved=fixture(1);out=self.prepare(arrays,provenance,saved,scenario_id='trailing_2_carry')
        self.assertTrue(out['case_admissible'][0])
        np.testing.assert_array_equal(out['raw_inputs']['history'][0,-2:,0],[175.,175.])
        np.testing.assert_array_equal(out['raw_inputs']['history'][0,-2:,3],[1.,2.])
        self.assertEqual(out['provenance']['quality_flags'],provenance['quality_flags'])


if __name__=='__main__':unittest.main()
