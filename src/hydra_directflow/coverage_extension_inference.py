"""Counted, target-free saved-checkpoint inference for the Phase A extension.

Clean regression and expanded inference are deliberately separate calls. Every
expanded call requires the complete 72-checkpoint clean-regression barrier. The
original common rows can then reuse the newly reproduced clean forecasts only
after exact input/coordinate equality is verified. No target reader or scorer is
called by this module, and no training state is constructed.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import os
from pathlib import Path
import re
import stat
import time
from unittest.mock import patch

import numpy as np

from .contract import ARCHITECTURES, GAUGE_IDS
from .inference import BestCheckpointPredictor, COORDINATES, validate_input_coordinates, validate_inputs
from .post_data import _Reader, _predictions
from .reference import ASSESSMENT_START, digest, require
from .training import DevelopmentTrainer, FULL_FEATURE_SCHEMA, INPUTS, TrainOnlyTransform, array_identity

START = int(datetime(2023, 7, 31, tzinfo=timezone.utc).timestamp())
END = int(datetime(2024, 7, 30, 6, tzinfo=timezone.utc).timestamp())
INPUT_ARRAYS = COORDINATES + tuple(k for f in INPUTS for k in (f, f + '_mask'))
META_ARRAYS = ('feature_schema', 'cohort_admission_digest', 'inference_source_identity')
OUTPUT_ARRAYS = ('prediction_q', 'case_id', 'origin_time', 'forecast_time', 'site_id')
MAX_BATCH = 64
MPS_CAP = 2 * 1024**3
SEEDS = (17, 29, 43)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def validate_record(original_plan, record):
    """Reject a wrong fold/model/history/backend before any model or input read."""
    job, receipt = record['job'], record['receipt']
    require(job == original_plan['jobs'][job['order']], 'Checkpoint record differs from original plan')
    require(job['fold'] == 'final_development' and job['input_arm'] == 'full'
            and job['architecture'] in ARCHITECTURES and job['gauge_id'] in GAUGE_IDS
            and job['seed'] in SEEDS and job['history_hours'] in (24, 168)
            and job['batch_size'] == 64 and job['precision'] == 'FP32',
            'Only final-development full-input L168/L24 checkpoints are allowed')
    device = 'cpu' if job['architecture'] in ('LSTM', 'GRU_Transformer') else 'mps'
    require(job['device'] == device, 'Original qualified backend is required')
    bindings = dict(plan_sha256=original_plan['plan_sha256'], protocol_sha256=original_plan['protocol_sha256'],
                    source_plan_sha256=original_plan['source_plan_sha256'], snapshot_sha256=original_plan['snapshot_sha256'],
                    implementation_sha256=original_plan['implementation_sha256'], job_id=job['job_id'], job_sha256=digest(job))
    require(all(receipt.get(k) == v for k, v in bindings.items())
            and receipt['status'] == 'VERIFIED_TRAIN_EVAL' and receipt['protected_assessment_access'] is False,
            'Original verified completion bindings differ')
    require(receipt['feature_schema'] == FULL_FEATURE_SCHEMA and type(receipt['selected_best_epoch']) is int
            and 1 <= receipt['selected_best_epoch'] <= receipt['completed_epochs'] <= 100,
            'A recorded selected-best full-input checkpoint is required')
    spec = receipt['trainer_spec']
    require(all(spec[k] == job[k] for k in ('architecture', 'device', 'seed', 'history_hours', 'batch_size'))
            and spec['feature_schema'] == FULL_FEATURE_SCHEMA, 'Checkpoint trainer specification differs')
    runtime = dict(torch_threads=1, torch_interop_threads=1, default_dtype='torch.float32', deterministic_algorithms=True)
    require(receipt['trainer_runtime'] == runtime and receipt['worker_execution_runtime'] == {'deterministic_algorithms': True},
            'Recorded deterministic FP32 runtime differs')
    for key in ('checkpoint', 'predictions'):
        descriptor = receipt['artifacts'][key]
        require(set(descriptor) == {'path', 'bytes', 'sha256'} and _sha(descriptor['sha256'])
                and descriptor['sha256'] == receipt['hashes'][key]
                and type(descriptor['bytes']) is int and 0 < descriptor['bytes'] <= 64 * 1024**2,
                'Original artifact descriptor differs')
    return job, receipt


def _batch(bundle, indices, *, history_hours, gauge_id):
    arrays = bundle['arrays']
    result = {name: np.array(arrays[name][indices], copy=True) for name in COORDINATES}
    result.update({name: np.array(arrays[name], copy=True) for name in META_ARRAYS})
    validate_input_coordinates(result, history_hours=history_hours, feature_schema=FULL_FEATURE_SCHEMA,
        expected_cohort_admission_digest=bundle['cohort_admission_digest'], expected_gauge_id=gauge_id)
    for field in INPUTS:
        for name in (field, field + '_mask'): result[name] = np.array(arrays[name][indices], copy=True)
    for value in result.values(): value.setflags(write=False)
    return result


def validate_bundle(bundle, job):
    """Whitelist reads only; poison target members cannot influence inference."""
    require(isinstance(bundle, Mapping) and isinstance(bundle['arrays'], Mapping), 'Explicit input-only bundle required')
    arrays = bundle['arrays']; manifest = bundle['manifest']
    require(_sha(bundle['source_identity']) and _sha(bundle['cohort_admission_digest'])
            and _sha(manifest['candidate_source_identity']), 'New inference/source identities required')
    require(manifest['schema']=='hydra-phase-a-inference-input-view/v1'
            and manifest['source_identity']==bundle['source_identity']
            and digest({k:v for k,v in manifest.items() if k!='source_identity'})==bundle['source_identity']
            and manifest['gauge_id']==job['gauge_id'] and manifest['history_hours']==job['history_hours']
            and manifest['feature_schema']==FULL_FEATURE_SCHEMA
            and manifest['cohort_admission_digest']==bundle['cohort_admission_digest']
            and manifest['target_values_read'] is False and manifest['target_masks_read'] is False
            and manifest['protected_assessment_access'] is False,
            'Input view source identity or declared scope differs')
    require(np.asarray(arrays['inference_source_identity']).item() == bundle['source_identity']
            and np.asarray(arrays['cohort_admission_digest']).item() == bundle['cohort_admission_digest']
            and np.asarray(arrays['feature_schema']).item() == FULL_FEATURE_SCHEMA,
            'Bundle inference metadata differs')
    origin = np.asarray(arrays['origin_time'])
    require(origin.ndim == 1 and 0 < len(origin) <= 8766 and origin.dtype.kind in 'iu'
            and np.all((origin >= START) & (origin < END)) and np.all(origin % 3600 == 0)
            and np.all(np.diff(origin) > 0), 'Exact final-development input-only origins required')
    # Clock validation precedes numerical values in each immutable batch.
    for start in range(0, len(origin), MAX_BATCH):
        data = _batch(bundle, slice(start, start + MAX_BATCH), history_hours=job['history_hours'], gauge_id=job['gauge_id'])
        validate_inputs(data, history_hours=job['history_hours'], feature_schema=FULL_FEATURE_SCHEMA,
                        expected_cohort_admission_digest=bundle['cohort_admission_digest'], expected_gauge_id=job['gauge_id'])
        require(all(data[name + '_mask'].all() for name in INPUTS), 'Input-only products require complete clean inputs')
    require(manifest['case_count']==len(origin)
            and manifest['case_ids_sha256']==digest(np.asarray(arrays['case_id']).tolist())
            and manifest['input_values_sha256']==input_identity(bundle)
            and manifest['input_arrays_sha256']==array_identity({name:np.asarray(arrays[name])
                for name in INPUT_ARRAYS+('feature_schema','cohort_admission_digest')}),
            'Input view content hash or membership differs')
    return len(origin)


def input_identity(bundle, indices=None):
    arrays = bundle['arrays']
    return array_identity({name: np.asarray(arrays[name]) if indices is None else np.asarray(arrays[name])[indices]
                           for name in INPUT_ARRAYS})


def verify_original_inputs(reader, record, clean_bundle):
    """Reproduce an old input commitment without using its identity for inference.

The original cohort scalar is included only in this temporary verification hash.
The passed bundle retains its new truthful inference/cohort identity throughout.
"""
    receipt = record['receipt']; job = record['job']
    relative = str(Path(receipt['receipt_path']).parent / 'outer-input-source.json')
    raw = reader.read(relative, 2 * 1024**2, expected_sha=record.get('original_input_manifest_sha256'))
    value = __import__('json').loads(raw)
    core = {k:v for k,v in value.items() if k not in ('source_identity', 'returned_arrays_sha256', 'construction')}
    require(value['source_identity'] == receipt['inference_source_identity'] == digest(core)
            and value['operation'] == 'inputs' and value['target_values_decoded'] is False
            and value['fold'] == 'final_development' and value['input_arm'] == 'full'
            and value['history_hours'] == job['history_hours'] and value['gauge_id'] == job['gauge_id'],
            'Original input source commitment differs')
    arrays = clean_bundle['arrays']
    projection = {name: np.asarray(arrays[name]) for name in INPUT_ARRAYS}
    projection['feature_schema'] = np.asarray(arrays['feature_schema'])
    projection['cohort_admission_digest'] = np.array(value['cohort_admission_digest'])
    require(array_identity(projection) == value['raw_arrays_sha256'], 'Clean input values differ from original committed inputs')
    return dict(original_input_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                original_input_raw_arrays_sha256=value['raw_arrays_sha256'],
                clean_input_values_sha256=input_identity(clean_bundle),
                verification_only_original_cohort_metadata=True, original_cohort_used_for_forward=False)


def configure_runtime(device):
    require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
            and os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1',
            'Offline and disabled MPS fallback must be set before Torch import')
    torch = importlib.import_module('torch')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32); torch.use_deterministic_algorithms(True)
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 1
            and torch.get_default_dtype() == torch.float32 and torch.are_deterministic_algorithms_enabled(),
            'Qualified deterministic FP32 runtime required')
    if device == 'mps':
        require(torch.backends.mps.is_available(), 'Qualified MPS backend unavailable; no fallback')
        torch.mps.set_per_process_memory_fraction(min(1., MPS_CAP / torch.mps.recommended_max_memory()))
    return torch


@contextmanager
def no_learning(torch):
    """Even accidental entry into a learning path is a hard technical failure."""
    def forbidden(*unused, **unused_kwargs): raise AssertionError('Phase A forbids training, optimizer updates and scaler fitting')
    with ExitStack() as stack:
        for cls, name in ((DevelopmentTrainer, '__init__'), (DevelopmentTrainer, 'update'),
                          (TrainOnlyTransform, 'fit'), (torch.optim.Optimizer, '__init__'),
                          (torch.optim.Optimizer, 'step'), (torch.optim.AdamW, 'step'),
                          (torch.optim.Adam, 'step'), (torch.optim.SGD, 'step')):
            stack.enter_context(patch.object(cls, name, side_effect=forbidden))
        yield


def _load_model(run, record, gradient_ids, torch, precharge_model, confirm_model, read_budget, check,
                expected_transform_sha256=None):
    job, receipt = record['job'], record['receipt']; descriptor = receipt['artifacts']['checkpoint']
    relative = descriptor['path']; require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                                         'Safe original checkpoint path required')
    path = run / relative; info = path.lstat()
    require(path == path.resolve() and path.is_relative_to(run) and stat.S_ISREG(info.st_mode)
            and not getattr(info, 'st_flags', 0) & 0x40000000 and info.st_size == descriptor['bytes']
            and info.st_size <= 16 * 1024**2, 'Resident original checkpoint with exact size required')
    check(); read_budget(info.st_size)
    token = precharge_model(job_id=job['job_id'], checkpoint_sha256=descriptor['sha256'])
    predictor = BestCheckpointPredictor.load(path, expected_sha256=descriptor['sha256'],
        expected_source_identity=receipt['training_source_identity'], expected_feature_schema=FULL_FEATURE_SCHEMA,
        expected_training_cohort_admission_digest=receipt['training_cohort_admission_digest'],
        expected_gauge_id=job['gauge_id'], gradient_fit_case_ids=gradient_ids)
    require(asdict(predictor.spec) == receipt['trainer_spec'] and predictor.gauge == job['gauge_id']
            and predictor.receipt['selected_best_epoch'] == receipt['selected_best_epoch']
            and predictor.receipt['checkpoint_sha256'] == descriptor['sha256']
            and predictor.receipt['selected_state'] == 'best_model'
            and predictor.receipt['optimizer_constructed'] is False and predictor.receipt['scaler_fitted'] is False,
            'Loaded model does not match the exact selected checkpoint')
    transform_sha = digest(asdict(predictor.transform))
    require(expected_transform_sha256 is None or transform_sha == expected_transform_sha256,
            'Saved transform identity changed')
    require(predictor.transform.training_case_ids_sha256 == digest(list(gradient_ids)),
            'Saved scaler gradient membership differs')
    confirm_model(token);check()
    return predictor, transform_sha


def _forward(predictor, bundle, positions, stage, precharge_forward, confirm_forward, check, stats):
    data = _batch(bundle, positions, history_hours=predictor.spec.history_hours, gauge_id=predictor.gauge)
    n = len(data['case_id']); require(0 < n <= 64, 'Forward batch bound exceeded')
    check(); token = precharge_forward(case_count=n, batch_count=1, stage=stage,
                                      case_ids_sha256=digest(data['case_id'].tolist()))
    stats['case_checkpoint_evaluations_entered'] += n; stats['forward_batches_entered'] += 1
    started = time.monotonic()
    output = predictor.predict(data, expected_inference_source_identity=bundle['source_identity'],
                               expected_inference_cohort_admission_digest=bundle['cohort_admission_digest'])
    if predictor.spec.device == 'mps': predictor.torch.mps.synchronize()
    value = np.asarray(output['prediction_q'])
    require(value.dtype == np.dtype('<f8') and value.shape == (n, 18) and np.isfinite(value).all(),
            'Missing/nonfinite or malformed requested model predictions')
    require(all(np.array_equal(output[k], data[k]) for k in COORDINATES), 'Model returned different case coordinates')
    confirm_forward(token)
    stats['case_checkpoint_evaluations_confirmed'] += n;stats['forward_batches_confirmed'] += 1
    stats['forward_wall_seconds'] += time.monotonic() - started
    check(); return np.array(value, copy=True)


def _output(bundle, prediction):
    result = {name: np.array(bundle['arrays'][name], copy=True) for name in OUTPUT_ARRAYS if name != 'prediction_q'}
    result['prediction_q'] = np.array(prediction, copy=True)
    for value in result.values(): value.setflags(write=False)
    return result


def _stats():
    return dict(case_checkpoint_evaluations_entered=0, case_checkpoint_evaluations_confirmed=0,
                forward_batches_entered=0, forward_batches_confirmed=0, forward_wall_seconds=0.,
                model_load_attempts_entered=0, model_constructions_confirmed=0,
                forward_wall_scope='SAVED_TRANSFORM_MODEL_FORWARD_DEVICE_SYNC_AND_OUTPUT_VALIDATION')


def clean_regression(original_plan, record, run_directory, gradient_fit_case_ids, clean_bundle, *,
                     phase_plan_sha256, precharge_forward, confirm_forward, precharge_model, confirm_model,
                     check=lambda:None, read_budget=lambda n:None, expected_transform_sha256=None):
    require(_sha(phase_plan_sha256), 'Phase A plan identity required')
    job, old = validate_record(original_plan, record); n = validate_bundle(clean_bundle, job)
    run = Path(run_directory).absolute(); reader = _Reader(run, check, read_budget)
    original_inputs = verify_original_inputs(reader, record, clean_bundle)
    coords, original_prediction = _predictions(reader, original_plan, record, START, END)
    require(n == len(original_prediction) and all(np.array_equal(coords[k], clean_bundle['arrays'][k]) for k in coords),
            'Clean regression must preserve the exact original case order')
    stats = _stats(); torch = configure_runtime(job['device']); predictor = None
    try:
        with no_learning(torch):
            stats['model_load_attempts_entered'] += 1
            predictor, transform_sha = _load_model(run, record, gradient_fit_case_ids, torch,
                precharge_model, confirm_model, read_budget, check, expected_transform_sha256)
            stats['model_constructions_confirmed'] += 1
            result = np.empty((n, 18), np.float64)
            for start in range(0, n, 64):
                result[start:start+64] = _forward(predictor, clean_bundle, slice(start,start+64), 'clean_regression',
                    precharge_forward, confirm_forward, check, stats)
            require(result.dtype == original_prediction.dtype and result.shape == original_prediction.shape
                    and result.tobytes(order='C') == original_prediction.tobytes(order='C'),
                    'Clean prediction byte parity failed; tolerance relaxation is forbidden')
            arrays = _output(clean_bundle, result)
            receipt = dict(schema='hydra-coverage-extension-clean-regression/v1', status='BYTE_EXACT_ORIGINAL_PREDICTIONS',
                phase_plan_sha256=phase_plan_sha256, original_plan_sha256=original_plan['plan_sha256'],
                job_id=job['job_id'], checkpoint_sha256=old['hashes']['checkpoint'],
                original_prediction_sha256=old['hashes']['predictions'], gauge_id=job['gauge_id'],
                architecture=job['architecture'], seed=job['seed'], history_hours=job['history_hours'], device=job['device'],
                saved_transform_sha256=transform_sha, training_case_ids_sha256=digest(list(gradient_fit_case_ids)),
                selected_best_epoch=old['selected_best_epoch'], input_source_identity=clean_bundle['source_identity'],
                input_cohort_admission_digest=clean_bundle['cohort_admission_digest'],
                candidate_source_identity=clean_bundle['manifest']['candidate_source_identity'],
                clean_input_values_sha256=input_identity(clean_bundle), input_verification=original_inputs,
                predictions_array_sha256=array_identity(arrays), case_ids_sha256=digest(arrays['case_id'].tolist()),
                cases=n, original_batch_size=64, last_partial_batch_retained=True, comparison='DTYPE_SHAPE_C_ORDER_BYTES',
                tolerance=None, saved_predictor_receipt=dict(predictor.receipt), accounting=stats,
                optimizer_updates=0, scaler_fits=0, target_values_read=False, target_masks_read=False,
                protected_assessment_access=False, posthoc_clipping=False)
            return dict(arrays=arrays, receipt=receipt)
    finally:
        del predictor; gc.collect()
        if job['device'] == 'mps': torch.mps.empty_cache()


def validate_clean_barrier(barrier, *, phase_plan_sha256, expected_barrier_sha256):
    expected = {f'final_development__{gauge}__{architecture}__full_L{length}__seed{seed}'
                for gauge in GAUGE_IDS for architecture in ARCHITECTURES for length in (168,24) for seed in SEEDS}
    require(_sha(expected_barrier_sha256) and digest(barrier) == expected_barrier_sha256
            and barrier['schema'] == 'hydra-coverage-extension-clean-barrier/v1'
            and barrier['status'] == 'ALL72_CLEAN_REGRESSIONS_BYTE_EXACT'
            and barrier['phase_plan_sha256'] == phase_plan_sha256
            and set(barrier['clean_receipt_sha256']) == expected
            and all(_sha(value) for value in barrier['clean_receipt_sha256'].values()),
            'All72 authenticated clean regressions must precede expanded inference')


def expanded_inference(original_plan, record, run_directory, gradient_fit_case_ids, clean_bundle, expanded_bundle,
                       clean_result, *, phase_plan_sha256, clean_barrier, expected_barrier_sha256,
                       precharge_forward, confirm_forward, precharge_model, confirm_model,
                       check=lambda:None, read_budget=lambda n:None):
    validate_clean_barrier(clean_barrier, phase_plan_sha256=phase_plan_sha256, expected_barrier_sha256=expected_barrier_sha256)
    job, old = validate_record(original_plan, record)
    validate_bundle(clean_bundle, job); n = validate_bundle(expanded_bundle, job)
    receipt = clean_result['receipt']; saved = clean_result['arrays']
    require(digest(receipt) == clean_barrier['clean_receipt_sha256'][job['job_id']]
            and receipt['status'] == 'BYTE_EXACT_ORIGINAL_PREDICTIONS' and receipt['phase_plan_sha256'] == phase_plan_sha256
            and receipt['checkpoint_sha256'] == old['hashes']['checkpoint']
            and _sha(receipt['saved_transform_sha256'])
            and receipt['training_case_ids_sha256'] == digest(list(gradient_fit_case_ids))
            and receipt['job_id'] == job['job_id'] and receipt['predictions_array_sha256'] == array_identity(saved),
            'Committed clean prediction/transform/checkpoint evidence differs')
    require(receipt['input_source_identity'] == clean_bundle['source_identity']
            and receipt['input_cohort_admission_digest'] == clean_bundle['cohort_admission_digest']
            and receipt['clean_input_values_sha256'] == input_identity(clean_bundle)
            and receipt['candidate_source_identity'] == clean_bundle['manifest']['candidate_source_identity']
                == expanded_bundle['manifest']['candidate_source_identity'], 'Clean and expanded source identities differ')
    a, b = clean_bundle['arrays'], expanded_bundle['arrays']
    positions = np.searchsorted(b['origin_time'], a['origin_time'])
    require(np.all(positions < n) and all(np.array_equal(a[k], np.asarray(b[k])[positions]) for k in INPUT_ARRAYS)
            and input_identity(expanded_bundle, positions) == receipt['clean_input_values_sha256']
            and all(np.array_equal(saved[k], a[k]) for k in OUTPUT_ARRAYS if k != 'prediction_q'),
            'Clean predictions cannot be reused for different typed inputs or coordinates')
    require(saved['prediction_q'].dtype == np.dtype('<f8') and saved['prediction_q'].shape == (len(positions),18)
            and np.isfinite(saved['prediction_q']).all(), 'Committed clean predictions are incomplete')
    result = np.empty((n,18),np.float64);result[positions] = saved['prediction_q']
    reused = np.zeros(n,bool);reused[positions] = True; new = np.flatnonzero(~reused)
    stats = _stats(); predictor = None; torch = None
    try:
        if len(new):
            torch = configure_runtime(job['device'])
            with no_learning(torch):
                stats['model_load_attempts_entered'] += 1
                predictor, transform_sha = _load_model(Path(run_directory).absolute(), record, gradient_fit_case_ids, torch,
                    precharge_model, confirm_model, read_budget, check, receipt['saved_transform_sha256'])
                stats['model_constructions_confirmed'] += 1
                for start in range(0,len(new),64):
                    indices = new[start:start+64]
                    result[indices] = _forward(predictor, expanded_bundle, indices, 'expanded_new_origins',
                        precharge_forward, confirm_forward, check, stats)
        else:
            # Reusing predictions does not permit unnoticed alteration of the
            # preserved checkpoint artifact, and requires no model construction.
            _Reader(Path(run_directory).absolute(),check,read_budget).descriptor(old['artifacts']['checkpoint'],16*1024**2)
            transform_sha = receipt['saved_transform_sha256']
        require(np.isfinite(result).all(), 'Every requested input-only vector must be finite')
        arrays = _output(expanded_bundle, result);reused.setflags(write=False)
        product = 'L168_INPUT_ONLY_V1' if job['history_hours']==168 else 'L24_INPUT_ONLY_V1'
        output = dict(schema='hydra-coverage-extension-input-only-predictions/v1', status='ALL_REQUESTED_VECTORS_FINITE',
            phase_plan_sha256=phase_plan_sha256, original_plan_sha256=original_plan['plan_sha256'], product=product,
            job_id=job['job_id'], checkpoint_sha256=old['hashes']['checkpoint'], saved_transform_sha256=transform_sha,
            selected_best_epoch=old['selected_best_epoch'], architecture=job['architecture'], gauge_id=job['gauge_id'],
            seed=job['seed'], history_hours=job['history_hours'], device=job['device'], cases=n,
            new_forward_cases=len(new), reused_clean_cases=len(positions),
            clean_barrier_sha256=expected_barrier_sha256, clean_regression_receipt_sha256=digest(receipt),
            input_source_identity=expanded_bundle['source_identity'], input_cohort_admission_digest=expanded_bundle['cohort_admission_digest'],
            candidate_source_identity=expanded_bundle['manifest']['candidate_source_identity'],
            input_values_sha256=input_identity(expanded_bundle), predictions_array_sha256=array_identity(arrays),
            case_ids_sha256=digest(arrays['case_id'].tolist()), reused_case_mask_sha256=digest(reused.tolist()),
            accounting=stats, optimizer_updates=0, scaler_fits=0, target_values_read=False, target_masks_read=False,
            protected_assessment_access=False, posthoc_clipping=False)
        return dict(arrays=arrays, reused_clean_mask=reused, receipt=output)
    finally:
        del predictor;gc.collect()
        if torch is not None and job['device']=='mps':torch.mps.empty_cache()
