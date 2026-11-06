"""Synthetic parameters only: no real checkpoint, model or training execution."""
from copy import deepcopy
from dataclasses import asdict,replace
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hydra_directflow.output_bound_diagnostic import (
    ARCHITECTURES,BUFFER_KEYS,EXPECTED_PARAMETER_COUNTS,ParameterReadAuthority,
    expected_jobs,expected_state_shapes,layernorm_linear_bound,bound_from_checkpoint,
    extract_saved_bound,check_saved_prediction_envelope)
from hydra_directflow.reference import digest
from hydra_directflow.training import (CHECKPOINT_SCHEMA,FULL_FEATURE_SCHEMA,
    TrainerSpec,TrainOnlyTransform,EarlyStopping,feature_schema_fields,implementation_fingerprint)


def fixture(architecture='LSTM'):
    state={k:torch.zeros(shape,dtype=torch.float32) for k,shape in expected_state_shapes(architecture).items()}
    state['final_norm.weight']=torch.linspace(.7,1.3,64)
    state['final_norm.bias']=torch.linspace(-.2,.1,64)
    state['flow_head.weight']=torch.linspace(-.25,.35,64).reshape(1,64)
    state['flow_head.bias']=torch.tensor([.15],dtype=torch.float32)
    fields=feature_schema_fields();parameters={k:[dict(mean=0.,scale=1.,constant=False,cells=2) for _ in v] for k,v in fields.items()}
    parameters['target']=[dict(mean=10.,scale=3.,constant=False,cells=2)]
    scaler=TrainOnlyTransform(parameters,'a'*64,'b'*64)
    spec=TrainerSpec(architecture,'cpu' if architecture in ('LSTM','GRU_Transformer') else 'mps',168,64,.0001,17,True,0)
    stopping=EarlyStopping(2,2,1,0.);stopping.observe(1.);stopping.observe(2.)
    checkpoint=dict(schema=CHECKPOINT_SCHEMA,spec=asdict(spec),source_identity='c'*64,feature_schema=FULL_FEATURE_SCHEMA,
        cohort_admission_digest='d'*64,implementation=implementation_fingerprint(),array_sha256='e'*64,
        updates=0,transform=asdict(scaler),stopping=asdict(stopping),best_model=state,
        model={'flow_head.bias':torch.tensor([1000.])},optimizer={'unused_original_tensors':torch.tensor([7.])},
        epoch_records=[dict(epoch=1,inner_stopping_loss=1.,updates=0,improved=True),
                       dict(epoch=2,inner_stopping_loss=2.,updates=0,improved=False)])
    job=dict(job_id=f'final_development__03161000__{architecture}__full_L168__seed17',gauge_id='03161000',
        architecture=architecture,seed=17,history_hours=168,input_arm='full',selected_weights='best_model',
        selected_best_epoch=1,device=spec.device,feature_schema=FULL_FEATURE_SCHEMA,training_source_identity='c'*64,
        training_cohort_admission_digest='d'*64,training_arrays_sha256='e'*64,gradient_case_ids_sha256='a'*64,
        checkpoint=dict(path='unused.pt',bytes=1,sha256='1'*64))
    return checkpoint,job


def authority(job,enabled=True):
    rows=tuple((j,job['checkpoint']['sha256'] if j==job['job_id'] else '1'*64) for j in sorted(expected_jobs()))
    return ParameterReadAuthority('2'*64,'3'*64,rows,enabled)


class BoundMathTests(unittest.TestCase):
    def test_random_numpy_layernorm_values_inside_centered_bound(self):
        rng=np.random.default_rng(46)
        for d in (2,4,64):
            w,g,beta=(rng.normal(size=d) for _ in range(3));b=.7
            bound=layernorm_linear_bound(w,g,beta,b,target_mean=12.,target_scale=4.)
            for scale in (1e-4,1.,1e4):
                x=rng.normal(size=(100,d))*scale+10
                u=(x-x.mean(axis=-1,keepdims=True))/np.sqrt(x.var(axis=-1,keepdims=True)+1e-5)
                q=12+4*((u*g+beta)@w+b)
                self.assertGreaterEqual(q.min(),bound['physical_lower']-1e-9)
                self.assertLessEqual(q.max(),bound['physical_upper']+1e-9)
                self.assertTrue(np.all(np.linalg.norm(u,axis=-1)<=math.sqrt(d)+1e-12))

    def test_limit_approaches_centered_bound_from_below(self):
        w=np.array([1.,-2.,3.,.1]);g=np.array([.2,2.,1.,.3]);beta=np.zeros(4)
        b=layernorm_linear_bound(w,g,beta,0.,target_mean=0.,target_scale=1.)
        a=w*g;a-=a.mean();x=a*1e5;u=(x-x.mean())/np.sqrt(x.var()+1e-5)
        self.assertLessEqual(float(np.dot(w*g,u)),b['standardized_upper']+1e-12)
        self.assertAlmostEqual(float(np.dot(w*g,u)),b['standardized_upper'],places=10)

    def test_zero_centered_coefficients_give_constant_output(self):
        b=layernorm_linear_bound(np.ones(4),np.ones(4),np.array([1.,2.,3.,4.]),2.,target_mean=10.,target_scale=3.)
        self.assertEqual(b['standardized_radius'],0)
        self.assertEqual(b['physical_lower'],46.);self.assertEqual(b['physical_upper'],46.)
        self.assertGreater(b['uncentered_cauchy_radius'],0)

    def test_positive_epsilon_only_tightens_finite_input_range(self):
        w=np.array([1.,-1.]);g=np.ones(2);beta=np.zeros(2)
        a=layernorm_linear_bound(w,g,beta,0.,target_mean=0.,target_scale=1.,epsilon=.1)
        b=layernorm_linear_bound(w,g,beta,0.,target_mean=0.,target_scale=1.,epsilon=1e-5)
        self.assertEqual(a['physical_upper'],b['physical_upper'])
        x=np.array([1.,-1.]);z=np.dot(w,x/np.sqrt(x.var()+.1))
        self.assertLess(z,a['physical_upper'])

    def test_affine_inverse_and_separate_numerical_allowance(self):
        w=np.array([1.,-2.]);g=np.array([3.,4.]);beta=np.array([2.,1.])
        a=layernorm_linear_bound(w,g,beta,1.,target_mean=0.,target_scale=1.)
        b=layernorm_linear_bound(w,g,beta,1.,target_mean=20.,target_scale=5.)
        self.assertEqual(b['physical_upper'],20+5*a['physical_upper'])
        self.assertFalse(b['fp32_formally_certified']);self.assertFalse(b['training_maximum_inferred'])
        self.assertGreater(b['comparison_allowance_cms'],0)

    def test_inputs_unchanged(self):
        v=np.arange(4,dtype=np.float32);old=v.copy()
        layernorm_linear_bound(v,v,v,0.,target_mean=0.,target_scale=1.)
        np.testing.assert_array_equal(old,v)

    def test_reject_invalid_parameters(self):
        one=np.ones(4)
        for bad in (np.ones(3),np.ones((1,4)),np.full(4,np.nan),np.full(4,np.inf),np.ones(4,dtype=complex)):
            with self.assertRaises(ValueError):layernorm_linear_bound(bad,one,one,0.,target_mean=0.,target_scale=1.)
        for kwargs in (dict(target_scale=0.),dict(epsilon=0.),dict(target_mean=float('nan')),dict(bias=True)):
            args=dict(bias=0.,target_mean=0.,target_scale=1.);args.update(kwargs)
            with self.assertRaises(ValueError):layernorm_linear_bound(one,one,one,**args)

    def test_overflow_rejected(self):
        with self.assertRaises(ValueError):layernorm_linear_bound(np.full(2,1e300),np.full(2,1e300),np.zeros(2),0.,target_mean=0.,target_scale=1.)


class CheckpointTests(unittest.TestCase):
    def test_static_shapes_match_independent_benchmark_counts_all_families(self):
        for a,count in EXPECTED_PARAMETER_COUNTS.items():
            shapes=expected_state_shapes(a)
            self.assertEqual(sum(math.prod(s) for k,s in shapes.items() if k not in BUFFER_KEYS),count)

    def test_all_families_extract_best_only_without_models_or_fit(self):
        with patch('hydra_directflow.models.directflow.DirectFlowModel.__init__',side_effect=AssertionError('Model constructed')),\
             patch('hydra_directflow.training.TrainOnlyTransform.fit',side_effect=AssertionError('Fit')),\
             patch('torch.optim.Optimizer.__init__',side_effect=AssertionError('Optimizer')):
            for a in ARCHITECTURES:
                cp,j=fixture(a);before=cp['best_model']['flow_head.weight'].clone();r=bound_from_checkpoint(cp,j)
                self.assertEqual(r['model_parameter_count'],EXPECTED_PARAMETER_COUNTS[a]);self.assertEqual(r['selected_best_epoch'],1)
                self.assertEqual(r['model_constructions'],r['model_forwards']);self.assertEqual(r['model_forwards'],0)
                self.assertLess(r['physical_upper'],1000)
                self.assertTrue(torch.equal(before,cp['best_model']['flow_head.weight']))

    def test_state_and_fingerprint_substitutions_rejected(self):
        cp,j=fixture()
        for mutate in (lambda x:x['best_model'].pop('history_projection.weight'),
                       lambda x:x['best_model'].update({'flow_head.weight':torch.zeros(1,63)}),
                       lambda x:x['best_model'].update({'final_norm.weight':torch.ones(64,dtype=torch.float64)}),
                       lambda x:x['implementation']['code'].update({'models/directflow.py':'0'*64}),
                       lambda x:x['transform'].update({'training_case_ids_sha256':'0'*64})):
            value=deepcopy(cp);mutate(value)
            with self.assertRaises(ValueError):bound_from_checkpoint(value,j)

    def test_gate_precedes_file_access(self):
        cp,j=fixture()
        with patch.object(Path,'read_bytes',side_effect=AssertionError('I/O')):
            with self.assertRaises(ValueError):extract_saved_bound('/not-opened',j,authority(j,False))

    def test_authority_requires_exact24_roster(self):
        cp,j=fixture();a=authority(j)
        with self.assertRaises(ValueError):replace(a,checkpoint_sha256_by_job=a.checkpoint_sha256_by_job[:-1])
        rows=list(a.checkpoint_sha256_by_job);rows[-1]=rows[0]
        with self.assertRaises(ValueError):replace(a,checkpoint_sha256_by_job=tuple(rows))

    def test_hash_gate_precedes_deserialization(self):
        cp,j=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp).resolve()/'fake.pt';p.write_bytes(b'not a checkpoint')
            j['checkpoint']['bytes']=p.stat().st_size
            with patch('torch.load',side_effect=AssertionError('Deserialization')):
                with self.assertRaises(ValueError):extract_saved_bound(p,j,authority(j))

    def test_verified_synthetic_saved_dictionary_roundtrip(self):
        cp,j=fixture('Mamba2_Transformer')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp).resolve()/'fake.pt';torch.save(cp,p);raw=p.read_bytes()
            j['checkpoint'].update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());charges=[]
            with patch('hydra_directflow.models.directflow.DirectFlowModel.__init__',side_effect=AssertionError('Model constructed')),\
                 patch('hydra_directflow.training.TrainOnlyTransform.fit',side_effect=AssertionError('Fit')),\
                 patch('torch.optim.Optimizer.__init__',side_effect=AssertionError('Optimizer')):
                r=extract_saved_bound(p,j,authority(j),read_budget=charges.append)
            self.assertEqual(r['checkpoint_deserializations'],1);self.assertEqual(charges[-1],len(raw))
            self.assertFalse(r['checkpoint_modified']);self.assertEqual(p.read_bytes(),raw)

    def test_symlink_is_not_hydrated_or_read(self):
        cp,j=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp).resolve()/'real';p.write_bytes(b'x');j['checkpoint'].update(bytes=1,sha256=hashlib.sha256(b'x').hexdigest())
            alias=p.with_name('alias');alias.symlink_to(p)
            with self.assertRaises(ValueError):extract_saved_bound(alias,j,authority(j))


class SavedForecastChecks(unittest.TestCase):
    def setUp(self):self.bound=layernorm_linear_bound(np.array([1.,-1.]),np.ones(2),np.zeros(2),0.,target_mean=10.,target_scale=2.)
    def test_saved_predicate_checks_raw_values_and_unavailable_rows(self):
        p=np.full((3,18),10.);p[1]=np.nan
        r=check_saved_prediction_envelope(p,np.array([True,False,True]),self.bound)
        self.assertEqual(r['finite_forecast_cells'],36);self.assertEqual(r['saved_prediction_max_cms'],10.)
    def test_saved_forecast_violation_fails(self):
        p=np.full((1,18),self.bound['physical_upper']+1.)
        with self.assertRaises(ValueError):check_saved_prediction_envelope(p,np.array([True]),self.bound)
    def test_finite_unavailable_or_nonfinite_admitted_fail(self):
        for p,a in ((np.ones((1,18)),np.array([False])),(np.full((1,18),np.nan),np.array([True]))):
            with self.assertRaises(ValueError):check_saved_prediction_envelope(p,a,self.bound)
    def test_empty_cohort_is_explicit(self):
        r=check_saved_prediction_envelope(np.full((2,18),np.nan),np.array([False,False]),self.bound)
        self.assertEqual(r['finite_forecast_cells'],0);self.assertIsNone(r['saved_prediction_max_cms'])

if __name__=='__main__':unittest.main()
