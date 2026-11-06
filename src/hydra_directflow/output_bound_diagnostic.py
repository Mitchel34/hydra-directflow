"""Read-only structural bounds from frozen terminal LayerNorm/head parameters.

No model is constructed or called. No optimizer, scaler fit, training or parameter
modification occurs. The caller authenticates the separate post-assessment
diagnostic protocol and selection before enabling ParameterReadAuthority. These
structural bindings are not an authorization mechanism by themselves.

For u=(x-mean(x))/sqrt(var(x)+epsilon), sum(u)=0 and ||u||<=sqrt(D)
in exact arithmetic, using the biased variance in the pinned LayerNorm. Thus
z=w·(gamma*u+beta)+b has c=b+w·beta, a=w*gamma and
|z-c|<=sqrt(D)||a-mean(a)||. Fixed positive affine target inverse preserves
this finite envelope. It is a global representational bound, not a prediction
interval, a training-range estimate, or proof that any decoder reaches it.

The formulas use float64 and report a separate, predeclared FP32 comparison
allowance. This is NOT a formal machine-arithmetic certificate. The caller must
check the saved predictions against that allowance before interpreting a bound.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
import re
import stat

import numpy as np

from .reference import digest, require
from .training import (FULL_FEATURE_SCHEMA, TrainerSpec, TrainOnlyTransform,
                       implementation_fingerprint, validate_checkpoint_mode)
from .inference import selected_best_epoch

ARCHITECTURES=('LSTM','Transformer','GRU_Transformer','Mamba2_Transformer')
GAUGES=('03161000','03164000')
SEEDS=(17,29,43)
WIDTH=64
EPSILON=1e-5
DIRECTFLOW_SHA256='ad767e49021f144402de669ddd50080c86d3b76cb5724afdfb9bec01a29c23a5'
SCHEMA='hydra-saved-output-bound-diagnostic/v1'
EXPECTED_PARAMETER_COUNTS=dict(LSTM=67905,Transformer=85057,GRU_Transformer=76545,Mamba2_Transformer=86093)
BUFFER_KEYS=('history_position','horizon_position')
COMPARISON_ATOL=1e-6
COMPARISON_RTOL=1e-5


def _sha(x):return type(x) is str and re.fullmatch(r'[0-9a-f]{64}',x) is not None
def _finite(x):return type(x) in (float,int) and math.isfinite(x)


def expected_jobs():
    return {f'final_development__{g}__{a}__full_L168__seed{s}' for g in GAUGES for a in ARCHITECTURES for s in SEEDS}


@dataclass(frozen=True)
class ParameterReadAuthority:
    protocol_sha256:str
    selection_sha256:str
    checkpoint_sha256_by_job:tuple[tuple[str,str],...]
    reads_enabled:bool=False

    def __post_init__(self):
        require(_sha(self.protocol_sha256) and _sha(self.selection_sha256)
            and type(self.reads_enabled) is bool and type(self.checkpoint_sha256_by_job) is tuple,
            'Explicit immutable diagnostic release bindings required')
        rows=self.checkpoint_sha256_by_job
        require(len(rows)==24 and all(type(x) is tuple and len(x)==2 and _sha(x[1]) for x in rows)
            and len({x[0] for x in rows})==24 and {x[0] for x in rows}==expected_jobs(),
            'Exactly24 original Jefferson/Galax FULL L168 checkpoint identities required')


def expected_state_shapes(architecture):
    """Static shapes from the pinned source; no nn.Module construction."""
    require(architecture in ARCHITECTURES,'Declared four-family architecture required')
    d=WIDTH
    shapes={'history_projection.weight':(d,4),'history_projection.bias':(d,),
            'weather_projection.weight':(d,12),'weather_projection.bias':(d,),
            'history_position':(168,d),'horizon_position':(18,d),
            'final_norm.weight':(d,),'final_norm.bias':(d,),
            'flow_head.weight':(1,d),'flow_head.bias':(1,)}
    def recurrent(prefix,gates):
        shapes.update({prefix+'.weight_ih_l0':(gates*d,d),prefix+'.weight_hh_l0':(gates*d,d),
                       prefix+'.bias_ih_l0':(gates*d,),prefix+'.bias_hh_l0':(gates*d,)})
    def attention(prefix):
        shapes.update({prefix+'.in_proj_weight':(3*d,d),prefix+'.in_proj_bias':(3*d,),
                       prefix+'.out_proj.weight':(d,d),prefix+'.out_proj.bias':(d,)})
    def transformer(prefix,decoder):
        attention(prefix+'.self_attn')
        if decoder:attention(prefix+'.multihead_attn')
        shapes.update({prefix+'.linear1.weight':(2*d,d),prefix+'.linear1.bias':(2*d,),
                       prefix+'.linear2.weight':(d,2*d),prefix+'.linear2.bias':(d,)})
        for n in range(1,4 if decoder else 3):
            shapes[prefix+f'.norm{n}.weight']=(d,);shapes[prefix+f'.norm{n}.bias']=(d,)
    if architecture=='LSTM':recurrent('history_encoder',4);recurrent('horizon_decoder',4)
    else:
        transformer('horizon_decoder.layers.0',True)
        if architecture=='Transformer':transformer('history_encoder.layers.0',False)
        elif architecture=='GRU_Transformer':recurrent('history_encoder',3)
        else:
            shapes.update({'mamba_norm.weight':(d,),'mamba.dt_bias':(4,),'mamba.A_log':(4,),'mamba.D':(4,),
                'mamba.conv1d.weight':(256,1,4),'mamba.conv1d.bias':(256,),
                'mamba.in_proj.weight':(388,d),'mamba.norm.weight':(128,), 'mamba.out_proj.weight':(d,128)})
    count=sum(math.prod(v) for k,v in shapes.items() if k not in BUFFER_KEYS)
    require(count==EXPECTED_PARAMETER_COUNTS[architecture],'Static parameter-count transcription differs')
    return shapes


def layernorm_linear_bound(weight,gamma,beta,bias,*,target_mean,target_scale,epsilon=EPSILON):
    """Pure parameter formula; never inspect observations or refit statistics."""
    vectors=tuple(np.asarray(x) for x in (weight,gamma,beta))
    require(all(x.dtype.kind=='f' and x.ndim==1 and np.isfinite(x).all() for x in vectors)
        and vectors[0].shape==vectors[1].shape==vectors[2].shape and 2<=len(vectors[0])<=1024,
        'Equal finite floating parameter vectors required')
    require(all(_finite(x) for x in (bias,target_mean,target_scale,epsilon)) and target_scale>0 and epsilon>0,
            'Finite scalar bias/mean and positive saved scale/epsilon required')
    w,g,beta=(x.astype(np.float64,copy=False) for x in vectors);d=len(w)
    a=[float(x)*float(y) for x,y in zip(w,g)]
    require(all(math.isfinite(x) for x in a),'Parameter multiplication overflow')
    mean=math.fsum(a)/d;centered=[x-mean for x in a]
    center=float(bias)+math.fsum(float(x)*float(y) for x,y in zip(w,beta))
    radius=math.sqrt(d)*math.hypot(*centered)
    loose_radius=math.sqrt(d)*math.hypot(*a)
    qcenter=target_mean+target_scale*center;qradius=target_scale*radius
    lower=qcenter-qradius;upper=qcenter+qradius
    require(all(math.isfinite(x) for x in (center,radius,loose_radius,qcenter,qradius,lower,upper)),
            'Bound arithmetic overflow')
    allowance=COMPARISON_ATOL+COMPARISON_RTOL*max(1.,abs(lower),abs(upper))
    return dict(schema=SCHEMA,width=d,layernorm_epsilon=epsilon,
        standardized_center=center,standardized_radius=radius,standardized_lower=center-radius,standardized_upper=center+radius,
        uncentered_cauchy_radius=loose_radius,target_mean=float(target_mean),target_scale=float(target_scale),
        physical_center=qcenter,physical_radius=qradius,physical_lower=lower,physical_upper=upper,
        units='m3/s',interpretation='EXACT_REAL_ARITHMETIC_GLOBAL_REPRESENTATIONAL_ENVELOPE',
        fp32_formally_certified=False,comparison_allowance_cms=allowance,
        comparison_allowance_rule=dict(atol=COMPARISON_ATOL,rtol=COMPARISON_RTOL,scale='MAX(1,ABS(LOWER),ABS(UPPER))'),
        finite_epsilon_can_only_tighten_exact_bound=True,attainability_by_trained_decoder_established=False,
        training_maximum_inferred=False,prediction_interval=False,
        formula='Q=target_mean+target_scale*(b+w.beta+(w*gamma).u); sum(u)=0, norm(u)<=sqrt(D)')


def bound_from_checkpoint(checkpoint,selected_job):
    """Extract a bound from an externally byte-authenticated saved dictionary.

    This never constructs a model. torch.load may deserialize original optimizer
    *tensors* in the full saved dictionary; no optimizer object is constructed,
    restored or stepped, and those tensors are not used here.
    """
    import torch
    j=selected_job
    require(j['job_id'] in expected_jobs() and j['gauge_id'] in GAUGES and j['architecture'] in ARCHITECTURES
        and j['seed'] in SEEDS and j['history_hours']==168 and j['input_arm']=='full'
        and j['selected_weights']=='best_model','Only declared saved-best FULL L168 jobs required')
    require(j['job_id']==f"final_development__{j['gauge_id']}__{j['architecture']}__full_L168__seed{j['seed']}",
            'Selected job coordinates disagree')
    validate_checkpoint_mode(checkpoint,expected_source_identity=j['training_source_identity'],
        expected_feature_schema=FULL_FEATURE_SCHEMA,expected_cohort_admission_digest=j['training_cohort_admission_digest'])
    require(checkpoint['implementation']==implementation_fingerprint()
        and checkpoint['implementation']['code']['models/directflow.py']==DIRECTFLOW_SHA256,
        'Original model topology, trainer or dependency fingerprint changed')
    spec=TrainerSpec(**checkpoint['spec'])
    require(all(getattr(spec,k)==j[k] for k in ('architecture','seed','history_hours','device','feature_schema'))
        and spec.batch_size==64 and checkpoint['array_sha256']==j['training_arrays_sha256'],
        'Original fit geometry or training array identity differs')
    epoch=selected_best_epoch(checkpoint)
    require(epoch==j['selected_best_epoch'],'Saved best epoch differs from original selection')
    transform=TrainOnlyTransform.from_dict(checkpoint['transform'])
    require(transform.training_case_ids_sha256==j['gradient_case_ids_sha256']
        and transform.feature_schema==FULL_FEATURE_SCHEMA,'Original gradient-only saved transform differs')
    state=checkpoint['best_model'];shapes=expected_state_shapes(j['architecture'])
    require(isinstance(state,Mapping) and set(state)==set(shapes),'Saved-best complete state-key inventory differs')
    signature={}
    for k,shape in shapes.items():
        v=state[k]
        require(isinstance(v,torch.Tensor) and tuple(v.shape)==shape and v.dtype==torch.float32
            and v.device.type=='cpu' and bool(torch.isfinite(v).all()),'Saved-best tensor shape/dtype/finiteness differs: '+k)
        signature[k]=dict(shape=list(shape),dtype='float32',sha256=hashlib.sha256(v.detach().contiguous().numpy().tobytes()).hexdigest())
    target=transform.parameters['target'][0]
    result=layernorm_linear_bound(state['flow_head.weight'][0].numpy(),state['final_norm.weight'].numpy(),
        state['final_norm.bias'].numpy(),float(state['flow_head.bias'][0]),
        target_mean=target['mean'],target_scale=target['scale'])
    result.update(job_id=j['job_id'],gauge_id=j['gauge_id'],architecture=j['architecture'],seed=j['seed'],
        checkpoint_sha256=j['checkpoint']['sha256'],selected_state='best_model',selected_best_epoch=epoch,
        model_parameter_count=EXPECTED_PARAMETER_COUNTS[j['architecture']],position_buffer_elements=168*64+18*64,
        terminal_parameter_count=3*64+1,best_state_typed_signature_sha256=digest(signature),
        terminal_parameter_signatures={k:signature[k] for k in ('final_norm.weight','final_norm.bias','flow_head.weight','flow_head.bias')},
        source_code_sha256=DIRECTFLOW_SHA256,target_transform_sha256=digest(checkpoint['transform']),
        training_case_ids_sha256=transform.training_case_ids_sha256,training_values_sha256=transform.training_values_sha256,
        model_constructions=0,model_forwards=0,optimizer_constructions=0,optimizer_updates=0,scaler_fits=0,
        assessments_used_to_choose_parameters=False)
    return result


def extract_saved_bound(path,selected_job,authority,*,read_budget=lambda n:None):
    require(isinstance(authority,ParameterReadAuthority) and authority.reads_enabled,
            'A separately authenticated enabled diagnostic protocol is required before checkpoint I/O')
    j=selected_job;allowed=dict(authority.checkpoint_sha256_by_job)
    require(j.get('job_id') in allowed and j['checkpoint']['sha256']==allowed[j['job_id']],
            'Checkpoint is not in the exact diagnostic release')
    code=Path(__file__).resolve().parent/'models/directflow.py'
    read_budget(code.stat().st_size);require(hashlib.sha256(code.read_bytes()).hexdigest()==DIRECTFLOW_SHA256,
                                          'Terminal model source changed')
    p=Path(path);s=p.lstat();descriptor=j['checkpoint']
    require(p.is_absolute() and p==p.resolve() and stat.S_ISREG(s.st_mode)
        and not getattr(s,'st_flags',0)&0x40000000 and s.st_size==descriptor['bytes'] and 0<s.st_size<=16*1024**2,
        'Exact resident regular checkpoint required; no source hydration')
    read_budget(s.st_size);raw=p.read_bytes();after=p.lstat()
    require((s.st_ino,s.st_size,s.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
        and len(raw)==descriptor['bytes'] and hashlib.sha256(raw).hexdigest()==descriptor['sha256'],
        'Checkpoint changed or failed its original byte hash')
    import torch
    checkpoint=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    result=bound_from_checkpoint(checkpoint,j)
    result.update(diagnostic_protocol_sha256=authority.protocol_sha256,selection_sha256=authority.selection_sha256,
                  checkpoint_bytes=len(raw),checkpoint_deserializations=1,checkpoint_modified=False)
    return result


def check_saved_prediction_envelope(predictions,input_admitted,bound):
    """Read saved forecast values only; never run a counterfactual forward."""
    p=np.asarray(predictions);a=np.asarray(input_admitted)
    require(p.ndim==2 and p.shape[1]==18 and p.dtype.kind=='f' and a.shape==(len(p),)
        and a.dtype==np.dtype(bool) and np.isfinite(p[a]).all() and np.isnan(p[~a]).all(),
        'Exact saved predictions/admission required')
    require(bound.get('schema')==SCHEMA and bound.get('fp32_formally_certified') is False,
            'Qualified exact-real bound with separate numerical allowance required')
    values=p[a];lower,upper,tol=(bound[k] for k in ('physical_lower','physical_upper','comparison_allowance_cms'))
    require(all(_finite(x) for x in (lower,upper,tol)) and lower<=upper and tol>=0,'Finite declared envelope required')
    below=int(np.count_nonzero(values<lower-tol));above=int(np.count_nonzero(values>upper+tol))
    require(below==above==0,'Saved predictions violate the structural envelope beyond its numerical allowance')
    return dict(admitted_origins=int(a.sum()),finite_forecast_cells=int(values.size),
        saved_prediction_min_cms=float(values.min()) if values.size else None,
        saved_prediction_max_cms=float(values.max()) if values.size else None,
        below_bound_beyond_allowance=below,above_bound_beyond_allowance=above,
        comparison_allowance_cms=tol,formal_fp32_certificate=False,new_model_forwards=0)
