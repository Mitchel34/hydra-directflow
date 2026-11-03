"""Bounded development inference using explicitly selected best checkpoint weights.

No labels, fitted statistics, optimizers, checkpoint selection or raw data access
are needed at prediction time. This component grants no scientific execution or
assessment permission. Forecast coordinates remain development-only.
"""
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
import hashlib
import io
import re
from types import MappingProxyType

import numpy as np

from .local_data import GAUGES, iso
from .reference import ASSESSMENT_START, digest, require
from .training import (INPUTS, FULL_FEATURE_SCHEMA, Q_ONLY_FEATURE_SCHEMA, TrainOnlyTransform,
                       TrainerSpec, EarlyStopping, feature_schema_fields, validate_checkpoint_mode,
                       implementation_fingerprint, _validate_bundle_schema, array_identity)

COORDINATES = ('origin_time', 'history_time', 'forecast_time', 'site_id', 'case_id')
MAX_CASES = 128


def input_only_view(arrays, *, inference_source_identity):
    """Copy only inputs/coordinates/metadata; gate clocks before numerical input access."""
    require(isinstance(arrays, Mapping), 'Explicit input mapping required')
    keys = set(iter(arrays))
    clock_name = 'forecast_time' if 'forecast_time' in keys else 'target_time'
    result = {name: np.asarray(arrays[clock_name if name == 'forecast_time' else name]) for name in COORDINATES}
    require(isinstance(inference_source_identity,str) and re.fullmatch(r'[0-9a-f]{64}',inference_source_identity),
            'Explicit inference-source identity required')
    result['inference_source_identity'] = np.array(inference_source_identity)
    for name in ('feature_schema','cohort_admission_digest'):
        require(name in keys, 'Explicit inference schema/cohort metadata required')
        result[name] = np.array(arrays[name],copy=True)
        require(result[name].shape == () and result[name].dtype.kind == 'U', 'Scalar Unicode inference metadata required')
    clock, site = result['history_time'], result['site_id']
    require(clock.ndim == 2 and site.ndim == 1 and len(site) > 0 and site.dtype.kind in 'iu'
            and int(site[0]) in range(3), 'Bounded history/gauge coordinates required')
    validate_input_coordinates(result, history_hours=clock.shape[1], feature_schema=result['feature_schema'].item(),
        expected_cohort_admission_digest=result['cohort_admission_digest'].item(), expected_gauge_id=GAUGES[int(site[0])])
    for field in INPUTS:
        for name in (field,field+'_mask'): result[name] = np.asarray(arrays[name])
    validate_inputs(result,history_hours=clock.shape[1],feature_schema=result['feature_schema'].item(),
        expected_cohort_admission_digest=result['cohort_admission_digest'].item(),expected_gauge_id=GAUGES[int(site[0])])
    result = {name:np.array(value,copy=True) for name,value in result.items()}
    for value in result.values(): value.setflags(write=False)
    return MappingProxyType(result)


def validate_input_coordinates(arrays, *, history_hours, feature_schema, expected_cohort_admission_digest, expected_gauge_id):
    """All coordinate/identity gates precede any input value access; labels are absent."""
    require(isinstance(arrays, Mapping) and history_hours in (24,168) and expected_gauge_id in GAUGES,
            'Explicit bounded development input configuration required')
    origin, history, future = (np.asarray(arrays[key]) for key in ('origin_time','history_time','forecast_time'))
    require(origin.ndim == 1 and 0 < len(origin) <= MAX_CASES, 'Inference requires 1..128 explicit cases')
    n = len(origin)
    require(all(value.dtype.kind in 'iu' for value in (origin,history,future))
            and history.shape == (n,history_hours) and future.shape == (n,18)
            and np.all(origin % 3600 == 0) and np.all(np.diff(origin) > 0), 'Exact chronological integer UTC coordinates required')
    require(np.array_equal(history, origin[:,None] - np.arange(history_hours+1,1,-1)*3600)
            and np.array_equal(future, origin[:,None] + np.arange(1,19)*3600), 'Inference history/H18 clocks differ')
    require(np.all(future < ASSESSMENT_START), 'Protected assessment or crossing inference coordinates rejected')
    site, ids = np.asarray(arrays['site_id']), np.asarray(arrays['case_id'])
    require(site.shape == ids.shape == (n,) and site.dtype.kind in 'iu' and ids.dtype.kind in 'US'
            and np.all(site == GAUGES.index(expected_gauge_id)), 'Inference gauge/case geometry differs')
    values = [value.decode('utf-8') if isinstance(value, bytes) else str(value) for value in ids.tolist()]
    require(values == [expected_gauge_id + ':' + iso(int(t)) for t in origin], 'Inference case IDs differ from gauge/origin clocks')
    require(isinstance(expected_cohort_admission_digest,str) and re.fullmatch(r'[0-9a-f]{64}',expected_cohort_admission_digest),
            'Explicit original comparison cohort required')
    declared = np.asarray(arrays['feature_schema'])
    require(declared.shape == () and declared.dtype.kind == 'U' and declared.item() == feature_schema,
            'Explicit inference feature schema differs')
    cohort = _validate_bundle_schema(arrays, feature_schema)
    require(cohort == expected_cohort_admission_digest, 'Inference comparison cohort differs')
    return n


def validate_inputs(arrays, *, history_hours, feature_schema, expected_cohort_admission_digest, expected_gauge_id):
    n = validate_input_coordinates(arrays,history_hours=history_hours,feature_schema=feature_schema,
            expected_cohort_admission_digest=expected_cohort_admission_digest,expected_gauge_id=expected_gauge_id)
    fields = feature_schema_fields(feature_schema)
    shapes = {'history':(n,history_hours,len(fields['history'])), 'weather':(n,18,len(fields['weather'])), 'state':(n,len(fields['state']))}
    for name, shape in shapes.items():
        value, mask = np.asarray(arrays[name]), np.asarray(arrays[name + '_mask'])
        require(value.shape == mask.shape == shape and value.dtype.kind == 'f' and mask.dtype == np.dtype(bool)
                and np.isfinite(value[mask]).all(), 'Inference input schema/masks differ: ' + name)
    if feature_schema == Q_ONLY_FEATURE_SCHEMA:
        require(np.asarray(arrays['weather_mask']).all()
                and np.array_equal(arrays['weather'], np.broadcast_to(np.arange(1,19)[None,:,None], (n,18,1))),
                'Q-only issued inputs must contain only declared H1-H18 coordinates')
    return n


def transform_inputs(arrays, transform):
    """Apply saved gradient-fit input statistics only; never fit or fetch labels."""
    require(isinstance(transform, TrainOnlyTransform), 'Previously fitted, validated transform required')
    mapped = {}
    for name in INPUTS:
        value, mask = np.asarray(arrays[name]), np.asarray(arrays[name + '_mask'])
        output = np.zeros(value.shape, dtype=np.float32)
        for column, stats in enumerate(transform.parameters[name]):
            observed = mask[...,column]
            output[...,column][observed] = (value[...,column][observed] - stats['mean']) / stats['scale']
        require(np.isfinite(output).all(), 'Saved input transform overflowed')
        mapped[name], mapped[name + '_mask'] = output, mask.copy()
    return mapped


def selected_best_epoch(checkpoint):
    """Require a consistent recorded best checkpoint; never fall back to last state."""
    stopping = checkpoint.get('stopping')
    require(isinstance(stopping, dict), 'Best-state inference requires recorded stopping selection')
    saved = EarlyStopping(**stopping)
    require(saved.completed_epochs > 0 and saved.best_loss is not None and saved.best_loss >= 0,
            'No selected best checkpoint epoch exists')
    records = checkpoint.get('epoch_records')
    require(isinstance(records,list) and len(records) == saved.completed_epochs, 'Best selection epoch records differ')
    replay = EarlyStopping(saved.max_epochs,saved.min_epochs,saved.patience,saved.min_improvement)
    best, previous_updates = None, 0
    for epoch, record in enumerate(records, 1):
        require(record['epoch'] == epoch and type(record['updates']) is int and previous_updates <= record['updates'] <= checkpoint['updates']
                and type(record['improved']) is bool, 'Best selection chronology/update counters differ')
        improved = replay.observe(record['inner_stopping_loss'])
        require(record['improved'] == improved, 'Recorded best-state decision differs from stopping policy')
        if improved: best = epoch
        previous_updates = record['updates']
    require(asdict(replay) == asdict(saved) and best is not None, 'Recorded best stopping state differs')
    require(isinstance(checkpoint.get('best_model'), Mapping) and checkpoint['best_model'], 'Selected best model state is absent')
    return best


class BestCheckpointPredictor:
    """A saved best model and saved train-only transform, with no trainer/optimizer."""
    @classmethod
    def load(cls, path, *, expected_sha256, expected_source_identity, expected_feature_schema,
             expected_training_cohort_admission_digest, expected_gauge_id, gradient_fit_case_ids):
        import torch
        from .engineering import _local_bytes
        from .models.directflow import DirectFlowModel, ModelConfig
        require(all(isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value)
                    for value in (expected_sha256,expected_source_identity,expected_training_cohort_admission_digest))
                and expected_gauge_id in GAUGES, 'Explicit checkpoint/source/cohort/gauge identities required')
        raw = _local_bytes(path,16*1024**2)
        require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'Checkpoint byte identity differs')
        checkpoint = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        validate_checkpoint_mode(checkpoint,expected_source_identity=expected_source_identity,
            expected_feature_schema=expected_feature_schema,expected_cohort_admission_digest=expected_training_cohort_admission_digest)
        require(checkpoint['implementation'] == implementation_fingerprint(), 'Trainer/model code or dependencies changed')
        spec = TrainerSpec(**checkpoint['spec'])
        require(type(checkpoint['updates']) is int and 0 <= checkpoint['updates'] <= spec.max_updates, 'Checkpoint update allowance differs')
        best_epoch = selected_best_epoch(checkpoint)
        transform = TrainOnlyTransform.from_dict(checkpoint['transform'])
        require(transform.feature_schema == expected_feature_schema and all(re.fullmatch(r'[0-9a-f]{64}', value)
                for value in (transform.training_case_ids_sha256,transform.training_values_sha256)), 'Saved transform mode/training identity differs')
        require(isinstance(gradient_fit_case_ids,tuple) and 0 < len(gradient_fit_case_ids) <= 200000
                and len(set(gradient_fit_case_ids)) == len(gradient_fit_case_ids)
                and all(isinstance(case,str) and re.fullmatch(expected_gauge_id + r':\d{4}-\d{2}-\d{2}T\d{2}:00:00Z',case)
                        for case in gradient_fit_case_ids)
                and digest(list(gradient_fit_case_ids)) == transform.training_case_ids_sha256,
                'Expected gauge/gradient-fit case identity differs from saved scaler')
        fields = feature_schema_fields(spec.feature_schema)
        if spec.device == 'mps': require(torch.backends.mps.is_available(), 'MPS unavailable; no fallback')
        with torch.random.fork_rng(devices=[]):
            model = DirectFlowModel(ModelConfig(spec.architecture,fields['history'],fields['weather'],fields['state'],
                                                spec.history_hours,tuple(range(1,19))))
        expected_state, best_state = model.state_dict(), checkpoint['best_model']
        require(set(best_state) == set(expected_state), 'Best model state keys differ')
        for key, template in expected_state.items():
            value = best_state[key]
            require(isinstance(value,torch.Tensor) and value.shape == template.shape and value.dtype == template.dtype
                    and bool(torch.isfinite(value).all()), 'Malformed best model tensor: ' + key)
            if key in ('history_position','horizon_position'):
                require(torch.equal(value,template), 'Best model coordinate encoding differs')
        model.load_state_dict(best_state, strict=True); model.to(spec.device); model.eval()
        result = cls()
        result.model, result.spec, result.transform, result.torch = model,spec,transform,torch
        result.training_cohort, result.gauge = expected_training_cohort_admission_digest, expected_gauge_id
        result.receipt = dict(checkpoint_sha256=expected_sha256, training_source_identity=expected_source_identity,
            training_cohort_admission_digest=expected_training_cohort_admission_digest, gauge_id=expected_gauge_id,
            feature_schema=expected_feature_schema, architecture=spec.architecture, seed=spec.seed,
            history_hours=spec.history_hours, device=spec.device, prediction_units='m3/s',
            selected_state='best_model', selected_best_epoch=best_epoch,
            completed_epochs=checkpoint['stopping']['completed_epochs'], training_case_ids_sha256=transform.training_case_ids_sha256,
            training_values_sha256=transform.training_values_sha256, model_code_and_dependencies=checkpoint['implementation'],
            optimizer_constructed=False, scaler_fitted=False, last_state_used=False,
            inference_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result

    def predict(self, arrays, *, expected_inference_source_identity, expected_inference_cohort_admission_digest):
        require(isinstance(expected_inference_source_identity,str) and re.fullmatch(r'[0-9a-f]{64}',expected_inference_source_identity),
                'Explicit current inference-source identity required')
        source = np.asarray(arrays['inference_source_identity'])
        require(source.shape == () and source.dtype.kind == 'U' and source.item() == expected_inference_source_identity,
                'Current inference-source identity differs')
        n = validate_inputs(arrays,history_hours=self.spec.history_hours,feature_schema=self.spec.feature_schema,
                            expected_cohort_admission_digest=expected_inference_cohort_admission_digest,expected_gauge_id=self.gauge)
        mapped = transform_inputs(arrays,self.transform)
        tensors = {name:self.torch.from_numpy(value).to(self.spec.device) for name,value in mapped.items()}
        with self.torch.inference_mode():
            self.model.eval()
            standardized = self.model(*(tensors[name] for field in INPUTS for name in (field,field+'_mask')))
        require(tuple(standardized.shape) == (n,18) and bool(self.torch.isfinite(standardized).all()), 'Best-model prediction geometry/values differ')
        predictions = self.transform.inverse_target(standardized.detach().cpu().numpy().astype(np.float64))
        require(np.isfinite(predictions).all(), 'Saved target inverse overflowed')
        predictions.setflags(write=False)
        coordinates = {name:np.array(arrays[name],copy=True) for name in COORDINATES}
        for value in coordinates.values(): value.setflags(write=False)
        return dict(prediction_q=predictions, **coordinates,
                    receipt={**self.receipt, 'inference_source_identity':expected_inference_source_identity,
                             'inference_cohort_admission_digest':expected_inference_cohort_admission_digest,
                             'input_case_ids_sha256':digest(coordinates['case_id'].tolist()),
                             'input_identity':array_identity({**mapped,**coordinates}), 'case_count':n,
                             'target_values_read':False, 'target_masks_read':False, 'optimizer_updates':0,
                             'posthoc_clipping':False, 'scientific_scoring_performed':False})
