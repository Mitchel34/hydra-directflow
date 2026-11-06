"""Separately gated assessment forwards from authenticated saved best weights.

The runner authenticates release/selection artifacts and precharges model loads
before calling the unchanged BestCheckpointPredictor.load. This module never
loads checkpoints, fits transforms, constructs optimizers, changes dates, or
relaxes the development inference path. Each successful batch records its actual
forward count and the existing predictor it used.
"""
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
import hashlib
import re
import time
from types import MappingProxyType

import numpy as np

from .assessment_data import AssessmentReadAuthority, _gate
from .assessment_contract import SEEDS
from .contract import ARCHITECTURES, GAUGE_IDS
from .inference import BestCheckpointPredictor, transform_inputs
from .local_data import iso
from .reference import (ASSESSMENT_START, ASSESSMENT_END, ASSESSMENT_LAST_TARGET,
                        HOUR, digest, require)
from .training import (INPUTS, FULL_FEATURE_SCHEMA, TrainOnlyTransform, TrainerSpec,
                       feature_schema_fields, array_identity)

COORDINATES=('origin_time','history_time','forecast_time','site_id','case_id')
METADATA=('feature_schema','cohort_admission_digest','inference_source_identity')
INPUT_ARRAYS=tuple(k for name in INPUTS for k in (name,name+'_mask'))
OUTPUT_COORDINATES=('origin_time','forecast_time','site_id','case_id')
MAX_VIEW_CASES=128
MAX_BATCH=64
DEVICES={'LSTM':'cpu','GRU_Transformer':'cpu','Transformer':'mps','Mamba2_Transformer':'mps'}


def _sha(value):
    return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None


def _prediction_gate(authority):
    _gate(authority)
    require(168 in authority.allowed_history_hours,
            'The assessment release does not admit L168 inputs')
    require(authority.verification_enabled is False and authority.prediction_barrier_sha256 is None,
            'Fresh assessment forwards require the pre-verification prediction phase')


def _scalar(arrays,name):
    value=np.asarray(arrays[name])
    require(value.shape==() and value.dtype.kind in 'US','Explicit scalar inference metadata required: '+name)
    result=value.item()
    return result.decode('utf-8') if isinstance(result,bytes) else result


def validate_assessment_coordinates(arrays,authority,*,gauge_id,expected_inference_source_identity,
                                    expected_inference_cohort_admission_digest,max_cases=MAX_VIEW_CASES):
    """Authority, metadata and clocks precede every numerical-input read."""
    _prediction_gate(authority)
    require(isinstance(arrays,Mapping) and gauge_id in GAUGE_IDS
            and _sha(expected_inference_source_identity) and _sha(expected_inference_cohort_admission_digest)
            and type(max_cases) is int and 1<=max_cases<=MAX_VIEW_CASES,
            'Explicit bounded assessment source/cohort/gauge required')
    require(_scalar(arrays,'feature_schema')==FULL_FEATURE_SCHEMA
            and _scalar(arrays,'inference_source_identity')==expected_inference_source_identity
            and _scalar(arrays,'cohort_admission_digest')==expected_inference_cohort_admission_digest,
            'Assessment inference source/cohort/schema differs')
    origin,history,future=(np.asarray(arrays[k]) for k in ('origin_time','history_time','forecast_time'))
    require(origin.ndim==1 and 0<len(origin)<=max_cases,'Bounded nonempty assessment batch required')
    n=len(origin)
    require(all(v.dtype.kind in 'iu' for v in (origin,history,future))
            and history.shape==(n,168) and future.shape==(n,18)
            and np.all(origin%HOUR==0) and np.all(origin[1:]>origin[:-1])
            and np.all((origin>=ASSESSMENT_START)&(origin<ASSESSMENT_END)),
            'Exact ordered assessment origin coordinates required')
    require(np.array_equal(history,origin[:,None]-np.arange(169,1,-1)*HOUR)
            and np.array_equal(future,origin[:,None]+np.arange(1,19)*HOUR)
            and np.all(future<=ASSESSMENT_LAST_TARGET),
            'Assessment history/H18 clocks or target tail differ')
    sites,ids=np.asarray(arrays['site_id']),np.asarray(arrays['case_id'])
    require(sites.shape==ids.shape==(n,) and sites.dtype.kind in 'iu' and ids.dtype.kind=='U'
            and np.all(sites==GAUGE_IDS.index(gauge_id)),'Assessment gauge/case geometry differs')
    names=[x.decode('utf-8') if isinstance(x,bytes) else str(x) for x in ids.tolist()]
    require(names==[gauge_id+':'+iso(int(c)) for c in origin],'Assessment case IDs differ from exact gauge/origin clocks')
    if 'input_admitted' in arrays:
        admitted=np.asarray(arrays['input_admitted'])
        require(admitted.shape==(n,) and admitted.dtype==np.dtype(bool) and admitted.all(),
                'Only input-admitted rows may enter a forecast batch')
    return n


def validate_assessment_inputs(arrays,authority,*,gauge_id,expected_inference_source_identity,
                               expected_inference_cohort_admission_digest,max_cases=MAX_VIEW_CASES):
    n=validate_assessment_coordinates(arrays,authority,gauge_id=gauge_id,
        expected_inference_source_identity=expected_inference_source_identity,
        expected_inference_cohort_admission_digest=expected_inference_cohort_admission_digest,max_cases=max_cases)
    fields=feature_schema_fields(FULL_FEATURE_SCHEMA)
    shapes={'history':(n,168,len(fields['history'])),'weather':(n,18,len(fields['weather'])),'state':(n,len(fields['state']))}
    for name,shape in shapes.items():
        value,mask=np.asarray(arrays[name]),np.asarray(arrays[name+'_mask'])
        require(value.shape==mask.shape==shape and value.dtype.kind=='f' and mask.dtype==np.dtype(bool)
                and mask.all() and np.isfinite(value).all(),
                'Assessment full L168 requires complete finite original inputs: '+name)
    expected_age=(np.asarray(arrays['origin_time'])[:,None]-np.asarray(arrays['history_time']))/HOUR
    require(np.array_equal(np.asarray(arrays['history'])[...,1],expected_age),
            'History age must retain the original slot timestamp')
    age=np.asarray(arrays['state'])[:,1]
    require(np.all((age>=2)&(age<=8)) and np.array_equal(age,np.floor(age)),
            'Soil initial-state age violates the fixed availability rule')
    return n


def assessment_input_only_view(arrays,authority,*,gauge_id,expected_inference_source_identity,
                               expected_inference_cohort_admission_digest):
    """Copy only known inputs/coordinates, never target or mixed-admission keys.

    Up to128 already-filtered one-gauge rows can be prepared here. The forward
    function accepts at most64; runners preserve chronological B64 partitions.
    The supplied cohort identity remains the explicit parent chunk identity.
    """
    validate_assessment_coordinates(arrays,authority,gauge_id=gauge_id,
        expected_inference_source_identity=expected_inference_source_identity,
        expected_inference_cohort_admission_digest=expected_inference_cohort_admission_digest)
    result={k:np.array(arrays[k],copy=True) for k in COORDINATES+METADATA+INPUT_ARRAYS}
    validate_assessment_inputs(result,authority,gauge_id=gauge_id,
        expected_inference_source_identity=expected_inference_source_identity,
        expected_inference_cohort_admission_digest=expected_inference_cohort_admission_digest)
    for value in result.values():value.setflags(write=False)
    return MappingProxyType(result)


def validate_selected_job(selected_job,authority):
    """Validate one externally authenticated PRIMARY_SELECTION record.

    The runner binds the entire selection artifact to authority.selection_sha256
    before extracting this record. This structural check is not self-approval.
    """
    _prediction_gate(authority)
    require(isinstance(selected_job,Mapping),'Explicit locked selected-job metadata required')
    j=selected_job
    require(j.get('gauge_id') in GAUGE_IDS and j.get('architecture') in ARCHITECTURES
            and type(j.get('seed')) is int and j['seed'] in SEEDS
            and j.get('input_arm')=='full' and j.get('history_hours')==168
            and type(j['history_hours']) is int and j.get('feature_schema')==FULL_FEATURE_SCHEMA
            and j.get('device')==DEVICES[j['architecture']]
            and j.get('selected_weights')=='best_model',
            'Only the36 saved full-L168 family/gauge/seed/backend selections are released')
    expected=f"final_development__{j['gauge_id']}__{j['architecture']}__full_L168__seed{j['seed']}"
    require(j.get('job_id')==expected,'Selected job ID differs from the declared saved fit')
    cp=j.get('checkpoint')
    require(isinstance(cp,dict) and set(cp)=={'path','bytes','sha256'} and _sha(cp['sha256'])
            and isinstance(cp['path'],str) and not Path(cp['path']).is_absolute() and '..' not in Path(cp['path']).parts
            and type(cp['bytes']) is int and 0<cp['bytes']<=16*1024**2,
            'Exact retained checkpoint descriptor required')
    require(type(j.get('selected_best_epoch')) is int and 1<=j['selected_best_epoch']<=100
            and all(_sha(j.get(k)) for k in ('training_source_identity','training_cohort_admission_digest',
                                           'gradient_case_ids_sha256','original_training_receipt_sha256')),
            'Selected checkpoint training provenance or best epoch differs')
    return j


def validate_selected_predictor(predictor,selected_job,authority):
    j=validate_selected_job(selected_job,authority)
    require(isinstance(predictor,BestCheckpointPredictor) and isinstance(predictor.spec,TrainerSpec)
            and isinstance(predictor.transform,TrainOnlyTransform),'Authenticated saved-best predictor required')
    receipt=predictor.receipt;spec=predictor.spec
    require(predictor.gauge==j['gauge_id'] and predictor.training_cohort==j['training_cohort_admission_digest']
            and spec.batch_size==64 and spec.feature_schema==FULL_FEATURE_SCHEMA
            and all(getattr(spec,k)==j[k] for k in ('architecture','device','seed','history_hours')),
            'Loaded predictor specification differs from the locked selected job')
    require(receipt.get('checkpoint_sha256')==j['checkpoint']['sha256']
            and receipt.get('training_source_identity')==j['training_source_identity']
            and receipt.get('training_cohort_admission_digest')==j['training_cohort_admission_digest']
            and receipt.get('training_case_ids_sha256')==j['gradient_case_ids_sha256']
            and receipt.get('selected_best_epoch')==j['selected_best_epoch']
            and receipt.get('selected_state')=='best_model'
            and type(receipt.get('completed_epochs')) is int
            and j['selected_best_epoch']<=receipt['completed_epochs']<=100
            and all(receipt.get(k)==j[k] for k in ('gauge_id','architecture','seed','history_hours','device','feature_schema'))
            and all(receipt.get(k) is False for k in ('optimizer_constructed','scaler_fitted','last_state_used')),
            'Loaded predictor receipt differs from the exact saved best checkpoint')
    transform=predictor.transform
    TrainOnlyTransform.from_dict(asdict(transform))  # Validate saved statistics; never fit.
    require(transform.feature_schema==FULL_FEATURE_SCHEMA
            and transform.training_case_ids_sha256==j['gradient_case_ids_sha256']
            and _sha(transform.training_values_sha256)
            and transform.training_values_sha256==receipt.get('training_values_sha256'),
            'Saved train-only transform provenance differs')
    transform_sha=digest(asdict(transform))
    if j.get('saved_transform_sha256') is not None:
        require(j['saved_transform_sha256']==transform_sha,'Saved transform differs from the locked digest')
    config=predictor.model.config;fields=feature_schema_fields(FULL_FEATURE_SCHEMA)
    require(config.architecture==spec.architecture and config.history_hours==168
            and config.horizon_hours==tuple(range(1,19))
            and config.width==64 and config.dropout==.1 and config.site_mode=='per_gauge'
            and all(getattr(config,k+'_features')==fields[k] for k in INPUTS),
            'Loaded model geometry or input features differ')
    parameters=tuple(predictor.model.parameters())
    require(parameters and all(v.dtype==predictor.torch.float32 and v.device.type==spec.device for v in parameters),
            'Saved model must retain its selected FP32 backend without fallback')
    if spec.device=='mps':require(predictor.torch.backends.mps.is_available(),'Selected MPS backend is unavailable')
    return j,transform_sha


def predict_assessment_batch(predictor,arrays,authority,*,selected_job,
                             expected_inference_source_identity,expected_inference_cohort_admission_digest,
                             precharge_forward,confirm_forward,check=lambda:None):
    """One charged, label-free B64 forward with the actual assessment dates.

    Model construction/read accounting belongs to the runner's separate load.
    Callbacks must durably reserve/confirm work or raise; no retry occurs here.
    Empty/unavailable batches belong in the runner's coverage ledger, not here.
    """
    _prediction_gate(authority)
    require(callable(precharge_forward) and callable(confirm_forward) and callable(check),
            'Durable forward accounting and resource callbacks are required')
    job,transform_sha=validate_selected_predictor(predictor,selected_job,authority)
    validate_assessment_coordinates(arrays,authority,gauge_id=job['gauge_id'],
        expected_inference_source_identity=expected_inference_source_identity,
        expected_inference_cohort_admission_digest=expected_inference_cohort_admission_digest,max_cases=MAX_BATCH)
    view={k:np.array(arrays[k],copy=True) for k in COORDINATES+METADATA+INPUT_ARRAYS}
    n=validate_assessment_inputs(view,authority,gauge_id=job['gauge_id'],
        expected_inference_source_identity=expected_inference_source_identity,
        expected_inference_cohort_admission_digest=expected_inference_cohort_admission_digest,max_cases=MAX_BATCH)
    raw_identity=array_identity(view)
    case_sha=digest(view['case_id'].tolist())
    check()
    mapped=transform_inputs(view,predictor.transform)
    token=precharge_forward(case_count=n,batch_count=1,stage='ASSESSMENT_FULL_L168',case_ids_sha256=case_sha)
    require(token is not None,'A durable forward reservation token is required')
    started=time.monotonic()
    tensors={name:predictor.torch.from_numpy(value).to(predictor.spec.device) for name,value in mapped.items()}
    with predictor.torch.inference_mode():
        predictor.model.eval()
        standardized=predictor.model(*(tensors[k] for name in INPUTS for k in (name,name+'_mask')))
    require(isinstance(standardized,predictor.torch.Tensor) and tuple(standardized.shape)==(n,18)
            and standardized.dtype==predictor.torch.float32 and standardized.device.type==predictor.spec.device
            and bool(predictor.torch.isfinite(standardized).all()),
            'Assessment model returned malformed or nonfinite predictions')
    if predictor.spec.device=='mps':predictor.torch.mps.synchronize()
    predictions=predictor.transform.inverse_target(standardized.detach().cpu().numpy().astype(np.float64))
    require(predictions.dtype==np.dtype('float64') and predictions.shape==(n,18) and np.isfinite(predictions).all(),
            'Assessment physical-Q inverse returned malformed or nonfinite predictions')
    confirm_forward(token);elapsed=time.monotonic()-started;check()
    output={k:np.array(view[k],copy=True) for k in OUTPUT_COORDINATES}
    output['prediction_q']=np.array(predictions,copy=True)
    for value in output.values():value.setflags(write=False)
    receipt=dict(schema='hydra-assessment-full-l168-inference/v1',status='VERIFIED_FORWARD_AWAITING_RUNNER_ARTIFACT_COMMIT',
        execution_plan_sha256=authority.execution_plan_sha256,protocol_sha256=authority.protocol_sha256,
        selection_sha256=authority.selection_sha256,source_inventory_sha256=authority.source_inventory_sha256,
        job_id=job['job_id'],gauge_id=job['gauge_id'],architecture=job['architecture'],seed=job['seed'],
        history_hours=168,feature_schema=FULL_FEATURE_SCHEMA,device=job['device'],
        selected_state='best_model',selected_best_epoch=job['selected_best_epoch'],checkpoint_sha256=job['checkpoint']['sha256'],
        training_source_identity=job['training_source_identity'],training_cohort_admission_digest=job['training_cohort_admission_digest'],
        training_case_ids_sha256=job['gradient_case_ids_sha256'],training_values_sha256=predictor.transform.training_values_sha256,
        saved_transform_sha256=transform_sha,inference_source_identity=expected_inference_source_identity,
        inference_parent_cohort_admission_digest=expected_inference_cohort_admission_digest,
        input_case_ids_sha256=case_sha,input_arrays_sha256=raw_identity,predictions_array_sha256=array_identity(output),
        first_origin_utc=iso(int(view['origin_time'][0])),last_origin_utc=iso(int(view['origin_time'][-1])),
        last_target_utc=iso(int(view['forecast_time'][-1,-1])),case_count=n,prediction_units='m3/s',
        input_admission='COMPLETE_ORIGINAL_FULL_L168_INPUTS',target_values_read=False,target_masks_read=False,
        scientific_scoring_performed=False,protected_assessment_inputs_read=True,posthoc_clipping=False,
        optimizer_updates=0,scaler_fits=0,optimizer_constructed=False,scaler_fitted=False,last_state_used=False,model_forward_calls=1,
        accounting=dict(existing_predictors_used=1,model_constructions_this_call=0,model_load_accounting='RUNNER_OWNED_SEPARATE_PRECHARGED_LOAD',
            case_checkpoint_evaluations_entered=n,case_checkpoint_evaluations_confirmed=n,
            forward_batches_entered=1,forward_batches_confirmed=1,forward_wall_seconds=elapsed),
        inference_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return dict(**output,receipt=receipt)
