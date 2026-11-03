"""One immutable, ledger-bound local primary fit and development evaluation.

The campaign controller supplies a frozen authorized plan. This worker never
selects another configuration, retries a failed job, changes a backend, accesses
assessment data, or launches a cloud job. Importing it performs no model work.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time

from .campaign_ledger import CampaignLedger, AllowanceExhausted, digest, validate_plan
from .reference import ASSESSMENT_START, require

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3
MPS_CAP = 2 * GIB
HEARTBEAT_RESERVE = 64 * 1024
FIXED_TRAINING = dict(min_epochs=10, patience=5, min_improvement=0., learning_rate=.0001, shuffle_training=True)
REQUIRED_CODE = ('src/hydra_directflow/campaign_worker.py', 'src/hydra_directflow/campaign_ledger.py',
    'src/hydra_directflow/campaign_data.py', 'src/hydra_directflow/training.py',
    'src/hydra_directflow/inference.py', 'src/hydra_directflow/scoring.py',
    'src/hydra_directflow/models/directflow.py', 'src/hydra_directflow/models/mamba2_math.py')


class CampaignInterrupted(KeyboardInterrupt):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def _sync(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory(path):
    path = Path(path)
    require(path.absolute() == path.resolve(), 'Worker directories cannot traverse symlinks or aliases')
    if path.exists():
        require(path.is_dir() and not path.is_symlink(), 'Real private output directory required')
        return
    _directory(path.parent)
    path.mkdir(mode=0o700); _sync(path.parent)


def _private_root(value):
    path = Path(value).absolute()
    require(path == path.resolve() and path.is_relative_to(ROOT / '.local'), 'Private worker paths must stay in the stable ignored local directory')
    return path


def _file(run, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
            and all(part not in ('.', '..') for part in Path(relative).parts), 'Relative immutable artifact path required')
    path = run / relative
    require(path.resolve() == path and path.is_relative_to(run), 'Artifact escaped its private run')
    return path


def _read(path, maximum):
    from .engineering import _local_bytes
    return _local_bytes(path, maximum)


def _validate_implementation(plan):
    # Recheck at publication as well as process start. Code edited during a long
    # fit must never relabel old in-memory execution with a new disk fingerprint.
    for name, expected in plan['implementation_sha256'].items():
        require(isinstance(name, str) and not Path(name).is_absolute() and '..' not in Path(name).parts,
                'Repository-relative implementation path required')
        path = ROOT / name
        require(path.resolve().is_relative_to(ROOT), 'Implementation escaped stable checkout')
        require(hashlib.sha256(_read(path, 2 * 1024**2)).hexdigest() == expected, 'Frozen campaign implementation changed')


def validate_execution_plan(plan):
    plan = validate_plan(plan)
    require(plan.get('protocol_frozen') is True and plan.get('scientific_training_authorized') is True,
            'Frozen protocol and explicit scientific training authorization required')
    training = plan.get('training')
    require(isinstance(training, dict) and set(training) == {'max_epochs', *FIXED_TRAINING}
            and type(training['max_epochs']) is int and training['max_epochs'] in (100, 200)
            and all(training[key] == value for key, value in FIXED_TRAINING.items()), 'Frozen training policy differs')
    implementations = plan['implementation_sha256']
    require(set(REQUIRED_CODE) <= set(implementations), 'Worker and model implementation bindings required')
    _validate_implementation(plan)
    for job in plan['jobs']:
        require(re.fullmatch(r'[A-Za-z0-9_.-]{1,220}', job['job_id']) and job['job_id'] not in ('.', '..'), 'Safe job identity required')
        data = job.get('data_job')
        require(isinstance(data, dict) and all(job[key] == data[key] for key in
                ('job_id', 'fold', 'seed', 'gauge_id', 'architecture', 'input_arm', 'history_hours', 'device',
                 'batch_size', 'precision', 'role_case_ids_sha256')), 'Exact source roster identity differs')
        counts = data.get('actual_common_cases', {})
        require(set(counts) == {'gradient_fit', 'inner_stopping', 'outer_scoring'}
                and all(type(value) is int and value > 0 for value in counts.values()), 'Exact source role counts required')
        batches = math.ceil(counts['gradient_fit'] / 64)
        require(job['gradient_batches_per_epoch'] == batches
                and job['logical_update_limit'] == training['max_epochs'] * batches
                and job['max_updates'] == job['logical_update_limit'] + 30 * batches,
                'Logical training and physical replay allowances differ')
        expected_device = 'cpu' if job['architecture'] in ('LSTM', 'GRU_Transformer') else 'mps'
        require(job['device'] == expected_device, 'Frozen measured backend policy differs')
    return plan


def _offline():
    require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
            and os.environ.get('HF_HUB_OFFLINE') == '1' and os.environ.get('TRANSFORMERS_OFFLINE') == '1',
            'Offline/no-fallback environment must be set before importing Torch')
    def audit(event, unused):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
            raise RuntimeError('Campaign network access prohibited')
    sys.addaudithook(audit)


def _torch_runtime(device):
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32); torch.use_deterministic_algorithms(True)
    if device == 'mps':
        require(torch.backends.mps.is_available(), 'Required MPS unavailable; no fallback')
        torch.mps.set_per_process_memory_fraction(min(1., MPS_CAP / torch.mps.recommended_max_memory()))
    return torch


def _sync_device(torch, device):
    if device == 'mps':
        torch.mps.synchronize()


def _runtime_memory(torch, device):
    import psutil
    process = psutil.Process()
    parent = process.parent()
    parent_rss = parent.memory_info().rss if parent is not None else 0
    rss = process.memory_info().rss
    driver = torch.mps.driver_allocated_memory() if device == 'mps' else 0
    return dict(worker_rss_bytes=rss, parent_rss_bytes=parent_rss, mps_driver_bytes=driver,
                combined_rss_mps_bytes=rss + parent_rss + driver)


class ArtifactStore:
    """Exact persistent reservations before immutable publication; no refunds."""
    def __init__(self, run, directory, ledger, lease):
        self.run, self.directory, self.ledger, self.lease = run, directory, ledger, lease

    def descriptor(self, path, raw):
        return dict(path=str(path.relative_to(self.run)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    def write(self, name, raw, *, kind):
        require(isinstance(raw, bytes) and 0 < len(raw) <= 64 * 1024**2, 'Bounded nonempty output bytes required')
        path = self.directory / name
        require(path.parent == self.directory and not path.exists() and not path.is_symlink(), 'New private artifact required')
        stage = path.with_name(path.name + '.writing')
        require(not stage.exists() and not stage.is_symlink(), 'Unsettled artifact preserved; no overwrite')
        token = self.ledger.reserve_output(len(raw), self.lease, kind=kind)
        try:
            with stage.open('xb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            stage.rename(path); _sync(path.parent)
            actual = _read(path, len(raw))
            require(actual == raw, 'Published artifact byte identity differs')
            item = self.descriptor(path, actual)
            self.ledger.settle_output(self.lease, token, actual_bytes=len(actual), sha256=item['sha256'])
            return item
        except BaseException:
            observed = path.stat().st_size if path.exists() else stage.stat().st_size if stage.exists() else 0
            try:
                self.ledger.settle_output(self.lease, token, actual_bytes=min(observed, len(raw)), failed=True)
            except BaseException:
                pass  # Keep the original failure and every unresolved reservation.
            raise

    def json(self, name, value, *, kind='JSON'):
        return self.write(name, canonical_bytes(value), kind=kind)

    def checkpoint(self, trainer, *, name):
        path = self.directory / name
        pending = {}
        def before_write(size, identity):
            require(not pending and 0 < size <= 16 * 1024**2, 'One bounded checkpoint reservation required')
            pending.update(token=self.ledger.reserve_output(size, self.lease, kind='CHECKPOINT'), bytes=size, sha256=identity)
        try:
            identity = trainer.checkpoint(path, before_write=before_write)
            require(pending and identity == pending['sha256'], 'Checkpoint reservation identity differs')
            raw = _read(path, pending['bytes'])
            item = self.descriptor(path, raw)
            require(item['bytes'] == pending['bytes'] and item['sha256'] == identity, 'Checkpoint publication differs')
            self.ledger.settle_output(self.lease, pending['token'], actual_bytes=len(raw), sha256=identity)
            return item
        except BaseException:
            if pending:
                stage = path.with_name(path.name + '.writing')
                size = path.stat().st_size if path.exists() else stage.stat().st_size if stage.exists() else 0
                try:
                    self.ledger.settle_output(self.lease, pending['token'], actual_bytes=min(size, pending['bytes']), failed=True)
                except BaseException:
                    pass  # Keep the original failure and every unresolved reservation.
            raise


class ResourceGuard:
    def __init__(self, ledger, lease, run, directory, torch, job):
        self.ledger, self.lease, self.run, self.directory = ledger, lease, run, directory
        self.torch, self.job = torch, job
        self.stage = 'INITIALIZING'; self.logical_updates = self.completed_epochs = 0
        self.last_write = 0.; self.last = None
        self.token = ledger.reserve_output(HEARTBEAT_RESERVE, lease, kind='HEARTBEAT')
        self.path = directory / 'heartbeat.json'

    def check(self, *, force=False):
        memory = _runtime_memory(self.torch, self.job['device'])
        free = shutil.disk_usage(self.run).free
        require(memory['mps_driver_bytes'] <= MPS_CAP, 'MPS driver cap reached')
        self.ledger.check_resources(self.lease, free_bytes=int(free), combined_rss_bytes=int(memory['combined_rss_mps_bytes']))
        self.last = dict(schema='hydra-campaign-worker-heartbeat/v1', plan_sha256=self.ledger.plan['plan_sha256'],
            job_id=self.job['job_id'], attempt_id=self.lease.attempt_id, pid=os.getpid(), stage=self.stage,
            logical_updates=self.logical_updates, completed_epochs=self.completed_epochs,
            observed_at_utc=utc(), free_disk_bytes=int(free), **memory)
        now = time.monotonic()
        if force or now - self.last_write >= 2:
            raw = canonical_bytes(self.last)
            require(len(raw) <= HEARTBEAT_RESERVE, 'Heartbeat exceeds reserved slot')
            temporary = self.path.with_name('heartbeat.json.writing')
            require(not temporary.exists(), 'Unsettled heartbeat preserved')
            with temporary.open('xb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self.path); _sync(self.directory); self.last_write = now

    def settle(self):
        if self.path.exists():
            raw = _read(self.path, HEARTBEAT_RESERVE)
            self.ledger.settle_output(self.lease, self.token, actual_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        else:
            self.ledger.settle_output(self.lease, self.token, actual_bytes=0, failed=True)


def _read_descriptor(run, descriptor, maximum):
    require(isinstance(descriptor, dict) and set(descriptor) == {'path', 'bytes', 'sha256'}
            and type(descriptor['bytes']) is int and 0 < descriptor['bytes'] <= maximum, 'Bounded artifact descriptor required')
    path = _file(run, descriptor['path'])
    raw = _read(path, descriptor['bytes'])
    require(len(raw) == descriptor['bytes'] and hashlib.sha256(raw).hexdigest() == descriptor['sha256'], 'Immutable artifact bytes changed')
    return raw


def _validate_checkpoint_policy(saved, receipt, plan, job):
    """Bind every saved training/stopping setting to the frozen campaign."""
    from .training import TrainerSpec, EarlyStopping, FULL_FEATURE_SCHEMA, Q_ONLY_FEATURE_SCHEMA
    schema = Q_ONLY_FEATURE_SCHEMA if job['input_arm'] == 'q_only' else FULL_FEATURE_SCHEMA
    expected = asdict(TrainerSpec(job['architecture'], job['device'], job['history_hours'], job['batch_size'],
        plan['training']['learning_rate'], job['seed'], plan['training']['shuffle_training'],
        job['logical_update_limit'], feature_schema=schema))
    require(canonical_bytes(saved['spec']) == canonical_bytes(expected)
            and canonical_bytes(receipt['trainer_spec']) == canonical_bytes(expected)
            and receipt['feature_schema'] == schema, 'Checkpoint exact frozen trainer policy differs')
    stopping = saved['stopping']
    if stopping is None:
        require(saved['updates'] == 0 and receipt['completed_epochs'] == 0,
                'Only the initial zero-update checkpoint may omit stopping policy')
    else:
        EarlyStopping(**stopping)
        require(all(stopping[key] == plan['training'][key]
                    for key in ('max_epochs', 'min_epochs', 'patience', 'min_improvement')),
                'Checkpoint exact frozen stopping policy differs')
    runtime = dict(torch_threads=1, torch_interop_threads=1, default_dtype='torch.float32', deterministic_algorithms=True)
    require(saved['runtime'] == runtime and receipt['trainer_runtime'] == runtime,
            'Checkpoint deterministic FP32 runtime differs')
    require(receipt['worker_execution_runtime'] == {'deterministic_algorithms': True},
            'Worker execution determinism was not explicitly observed')


def artifact_validator(run, plan, job):
    """Authenticate real bytes and checkpoint/prediction/metric bindings."""
    def validate(receipt, bindings):
        _validate_implementation(plan)
        require(all(receipt.get(key) == value for key, value in bindings.items()), 'Receipt bindings changed')
        require(receipt['job_id'] == job['job_id'] and receipt['plan_sha256'] == plan['plan_sha256'], 'Receipt job/plan differs')
        receipt_raw = _read(_file(run, receipt['receipt_path']), 2 * 1024**2)
        require(receipt_raw == canonical_bytes(receipt), 'Published receipt bytes differ from authenticated content')
        artifacts = receipt.get('artifacts', {})
        expected = {'checkpoint'} if receipt['status'] == 'VERIFIED_CHECKPOINT' else {'checkpoint', 'predictions', 'metrics'}
        require(set(artifacts) == expected and set(receipt['hashes']) == expected
                and all(artifacts[key]['sha256'] == receipt['hashes'][key] for key in expected), 'Artifact/hash inventory differs')
        import torch
        from .training import implementation_fingerprint, validate_checkpoint_mode
        from .inference import selected_best_epoch
        raw = _read_descriptor(run, artifacts['checkpoint'], 16 * 1024**2)
        saved = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        validate_checkpoint_mode(saved, expected_source_identity=receipt['training_source_identity'],
            expected_feature_schema=receipt['feature_schema'], expected_cohort_admission_digest=receipt['training_cohort_admission_digest'])
        require(saved['implementation'] == implementation_fingerprint() and saved['updates'] == receipt['logical_updates']
                and saved['spec'] == receipt['trainer_spec'] and saved['transform']['training_case_ids_sha256'] == job['role_case_ids_sha256']['gradient_fit']
                and saved['array_sha256'] == receipt['training_arrays_sha256'], 'Checkpoint implementation/spec/source prefix differs')
        _validate_checkpoint_policy(saved, receipt, plan, job)
        completed = saved['stopping']['completed_epochs'] if saved['stopping'] is not None else 0
        require(completed == receipt['completed_epochs'], 'Checkpoint epoch differs')
        if completed:
            require(selected_best_epoch(saved) == receipt['selected_best_epoch'], 'Checkpoint selected best state differs')
        else:
            require(receipt['logical_updates'] == 0 and saved['best_model'] is None, 'Initial checkpoint state differs')
        if expected != {'checkpoint'}:
            import numpy as np
            from .scoring import validate_score_coordinates
            from .training import EarlyStopping
            require(EarlyStopping(**saved['stopping']).finished, 'Completion checkpoint has not finished its stopping policy')
            prediction_raw = _read_descriptor(run, artifacts['predictions'], 64 * 1024**2)
            with np.load(io.BytesIO(prediction_raw), allow_pickle=False) as archive:
                require(set(archive.files) == {'prediction_q', 'case_id', 'origin_time', 'forecast_time', 'site_id', 'metadata_json'},
                        'Prediction artifact member inventory differs')
                meta = json.loads(archive['metadata_json'].item())
                for key in ('job_id', 'plan_sha256', 'training_source_identity', 'training_cohort_admission_digest',
                            'inference_source_identity', 'inference_cohort_admission_digest', 'selected_best_epoch'):
                    require(meta[key] == receipt[key], 'Prediction provenance differs')
                ids, origins, future = validate_score_coordinates(case_ids=tuple(archive['case_id'].tolist()),
                    origins=archive['origin_time'], forecast_time=archive['forecast_time'], gauge_id=job['gauge_id'])
                require(digest(list(ids)) == job['role_case_ids_sha256']['outer_scoring']
                        and len(ids) == job['data_job']['actual_common_cases']['outer_scoring'], 'Outer prediction cohort differs')
                values = archive['prediction_q']
                require(values.shape == (len(ids), 18) and values.dtype == np.dtype('float64') and np.isfinite(values).all(),
                        'Missing/nonfinite physical predictions')
            metrics = json.loads(_read_descriptor(run, artifacts['metrics'], 2 * 1024**2))
            require(metrics['plan_sha256'] == plan['plan_sha256'] and metrics['job_id'] == job['job_id']
                    and metrics['checkpoint_sha256'] == receipt['hashes']['checkpoint']
                    and metrics['predictions_sha256'] == receipt['hashes']['predictions']
                    and metrics['outer_case_ids_sha256'] == job['role_case_ids_sha256']['outer_scoring']
                    and metrics['panel']['input_cohort_digest'] == receipt['inference_cohort_admission_digest']
                    and metrics['panel']['label_source_digest'] == receipt['label_source_identity']
                    and metrics['panel']['case_ids_sha256'] == job['role_case_ids_sha256']['outer_scoring']
                    and len(metrics['panel']['leads']) == 18, 'Metrics/prediction/source bindings differ')
            identity = metrics['panel']['identity']
            require(all(identity[key] == expected for key, expected in dict(architecture=job['architecture'], gauge_id=job['gauge_id'],
                feature_schema=receipt['feature_schema'], history_hours=job['history_hours'], fold_id=job['fold'], seed=job['seed']).items())
                and [row['lead_hours'] for row in metrics['panel']['leads']] == list(range(1, 19))
                and all(row['n'] == row['n_requested'] == job['data_job']['actual_common_cases']['outer_scoring']
                        for row in metrics['panel']['leads']), 'Full-support score geometry differs')
        return digest(receipt)
    return validate


def _checkpoint_receipt(trainer, descriptor, ledger, lease):
    from .inference import selected_best_epoch
    from .training import implementation_fingerprint
    completed = trainer.stopping.completed_epochs if trainer.stopping is not None else 0
    deterministic = trainer.torch.are_deterministic_algorithms_enabled()
    require(deterministic is True, 'Worker checkpoint requires deterministic execution')
    worker_runtime = {'deterministic_algorithms': deterministic}
    # Replay the same public best-state selection metadata without reloading values.
    selected = selected_best_epoch(dict(stopping=asdict(trainer.stopping), epoch_records=trainer.epoch_records,
        updates=trainer.updates, best_model=trainer.best_model_state)) if completed else None
    return dict(**ledger.receipt_bindings(lease), status='VERIFIED_CHECKPOINT', logical_updates=trainer.updates,
        hashes={'checkpoint': descriptor['sha256']}, artifacts={'checkpoint': descriptor},
        training_source_identity=trainer.source_identity, training_cohort_admission_digest=trainer.cohort_admission_digest,
        training_arrays_sha256=trainer.array_sha256, feature_schema=trainer.spec.feature_schema,
        trainer_spec=asdict(trainer.spec), trainer_runtime=dict(trainer.runtime), worker_execution_runtime=worker_runtime,
        completed_epochs=completed, selected_best_epoch=selected,
        training_implementation=implementation_fingerprint())


def bind_trainer_updates(trainer, ledger, lease, guard):
    original_update, original_batch = trainer.update, trainer._batch
    def batch(indices):
        guard.check(); value = original_batch(indices); guard.check(); return value
    def update():
        guard.stage = 'GRADIENT_UPDATE'; guard.check()
        token = ledger.enter_update(lease)
        result = original_update()
        _sync_device(trainer.torch, trainer.spec.device)
        ledger.confirm_update(lease, token)
        guard.logical_updates = trainer.updates; guard.check()
        return result
    trainer._batch, trainer.update = batch, update
    return original_update, original_batch


def _train(ledger, lease, job, plan, corpus_root, store, guard, resume_receipt):
    from .campaign_data import load_fit_bundle
    from .training import DevelopmentTrainer, TrainerSpec, EarlyStopping
    guard.stage = 'LOAD_FIT_AND_INNER'; guard.check(force=True)
    bundle = load_fit_bundle(corpus_root, job['data_job'], check=guard.check)
    spec = TrainerSpec(job['architecture'], job['device'], job['history_hours'], 64,
        plan['training']['learning_rate'], job['seed'], True, job['logical_update_limit'],
        feature_schema=bundle['arrays']['feature_schema'].item())
    train_source = dict(source_identity=bundle['source_identity'], cohort_admission_digest=bundle['cohort_admission_digest'],
                        gradient_case_ids=bundle['gradient_case_ids'], manifest=bundle['manifest'])
    if resume_receipt is None:
        trainer = DevelopmentTrainer(bundle['arrays'], train_indices=bundle['train_indices'], validation_indices=bundle['inner_indices'],
            boundary_epoch=int(datetime.fromisoformat(bundle['role_time_bounds']['inner_stop']['start_utc'].replace('Z', '+00:00')).timestamp()),
            spec=spec, source_identity=bundle['source_identity'])
    else:
        require(resume_receipt['training_source_identity'] == bundle['source_identity']
                and resume_receipt['training_cohort_admission_digest'] == bundle['cohort_admission_digest'], 'Resume source/cohort differs')
        trainer = DevelopmentTrainer.restore(_file(store.run, resume_receipt['artifacts']['checkpoint']['path']), bundle['arrays'],
            expected_sha256=resume_receipt['hashes']['checkpoint'], expected_source_identity=bundle['source_identity'],
            expected_feature_schema=spec.feature_schema, expected_cohort_admission_digest=bundle['cohort_admission_digest'])
        require(asdict(trainer.spec) == asdict(spec), 'Restored training policy differs')
    guard.logical_updates = trainer.updates
    guard.completed_epochs = trainer.stopping.completed_epochs if trainer.stopping is not None else 0
    guard.check(force=True)
    store.json('fit-source.json', bundle['manifest'], kind='SOURCE_RECEIPT')
    original_update, original_batch = bind_trainer_updates(trainer, ledger, lease, guard)
    validator = artifact_validator(store.run, plan, job)
    latest = resume_receipt
    def checkpoint():
        nonlocal latest
        guard.stage = 'CHECKPOINT'; guard.check(force=True)
        item = store.checkpoint(trainer, name=f'checkpoint-epoch-{trainer.stopping.completed_epochs if trainer.stopping else 0:04d}.pt')
        latest = _checkpoint_receipt(trainer, item, ledger, lease)
        latest['receipt_path'] = str((store.directory / f'checkpoint-epoch-{latest["completed_epochs"]:04d}.json').relative_to(store.run))
        store.json(f'checkpoint-epoch-{latest["completed_epochs"]:04d}.json', latest, kind='CHECKPOINT_RECEIPT')
        ledger.checkpoint(lease, latest, validator=validator)
        guard.check(force=True)
    if resume_receipt is None:
        checkpoint()
    def after_epoch(current):
        require(current is trainer, 'Epoch callback trainer identity changed')
        guard.completed_epochs, guard.logical_updates = current.stopping.completed_epochs, current.updates
        guard.stage = 'INNER_STOPPING_COMPLETE'; guard.check(force=True)
        store.json(f'epoch-{current.stopping.completed_epochs:04d}.json', dict(**ledger.receipt_bindings(lease),
            epoch=current.epoch_records[-1], stopping=asdict(current.stopping)), kind='EPOCH_RECEIPT')
        if current.stopping.completed_epochs % job['checkpoint_interval_epochs'] == 0 or current.stopping.finished:
            checkpoint()
    stopping = trainer.stopping or EarlyStopping(**{key: plan['training'][key] for key in ('max_epochs', 'min_epochs', 'patience', 'min_improvement')})
    trainer.fit_epochs(stopping, after_epoch=after_epoch)
    require(latest is not None and latest['logical_updates'] == trainer.updates
            and latest['completed_epochs'] == trainer.stopping.completed_epochs and trainer.stopping.finished,
            'Terminal authenticated checkpoint required before inference')
    # Remove closures holding the trainer before releasing dense source/optimizer memory.
    trainer.update, trainer._batch = original_update, original_batch
    del original_update, original_batch, trainer, bundle
    gc.collect()
    if job['device'] == 'mps':
        guard.torch.mps.empty_cache()
    guard.stage = 'TRAINER_RELEASED'; guard.check(force=True)
    return latest, train_source


def _evaluate(ledger, lease, job, plan, corpus_root, store, guard, checkpoint, training_source):
    import numpy as np
    from unittest.mock import patch
    from .campaign_data import load_outer_inputs, iter_outer_batches, load_outer_labels
    from .inference import BestCheckpointPredictor
    from .training import DevelopmentTrainer, TrainOnlyTransform
    from .scoring import ScoreIdentity, score_physical_h18
    from .quality import PRIMARY_APPROVED
    guard.stage = 'LOAD_OUTER_INPUTS'; guard.check(force=True)
    outer = load_outer_inputs(corpus_root, job['data_job'], check=guard.check)
    count = job['data_job']['actual_common_cases']['outer_scoring']
    predictions = np.empty((count, 18), np.float64)
    seen = 0
    with (patch.object(guard.torch.optim.Optimizer, '__init__', side_effect=AssertionError('Inference cannot construct optimizer')),
          patch.object(TrainOnlyTransform, 'fit', side_effect=AssertionError('Inference cannot fit scaler')),
          patch.object(DevelopmentTrainer, '__init__', side_effect=AssertionError('Inference cannot construct trainer'))):
        predictor = BestCheckpointPredictor.load(_file(store.run, checkpoint['artifacts']['checkpoint']['path']),
            expected_sha256=checkpoint['hashes']['checkpoint'], expected_source_identity=checkpoint['training_source_identity'],
            expected_feature_schema=checkpoint['feature_schema'], expected_training_cohort_admission_digest=checkpoint['training_cohort_admission_digest'],
            expected_gauge_id=job['gauge_id'], gradient_fit_case_ids=training_source['gradient_case_ids'])
        for batch in iter_outer_batches(outer, batch_size=64):
            guard.stage = 'LABEL_FREE_OUTER_INFERENCE'; guard.check()
            output = predictor.predict(batch, expected_inference_source_identity=outer['source_identity'],
                expected_inference_cohort_admission_digest=outer['cohort_admission_digest'])
            n = len(output['case_id'])
            require(np.array_equal(output['case_id'], outer['arrays']['case_id'][seen:seen+n]), 'Prediction case order differs')
            predictions[seen:seen+n] = output['prediction_q']; seen += n
            _sync_device(guard.torch, job['device']); guard.check()
    require(seen == count and np.isfinite(predictions).all(), 'Complete finite predictions required before labels')
    inference_receipt = output['receipt']
    metadata = dict(**ledger.receipt_bindings(lease), training_source_identity=checkpoint['training_source_identity'],
        training_cohort_admission_digest=checkpoint['training_cohort_admission_digest'],
        inference_source_identity=outer['source_identity'], inference_cohort_admission_digest=outer['cohort_admission_digest'],
        selected_best_epoch=checkpoint['selected_best_epoch'], checkpoint_sha256=checkpoint['hashes']['checkpoint'],
        case_ids_sha256=outer['case_ids_sha256'], case_count=count, prediction_units='m3/s', label_values_opened=False,
        posthoc_clipping=False, best_only=True, optimizer_constructed=False, scaler_fitted=False)
    coordinates = {key: np.array(outer['arrays'][key], copy=True) for key in ('case_id', 'origin_time', 'forecast_time', 'site_id')}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, prediction_q=predictions, **coordinates, metadata_json=np.array(canonical_bytes(metadata).decode()))
    prediction_artifact = store.write('predictions.npz', buffer.getvalue(), kind='PREDICTIONS')
    store.json('outer-input-source.json', outer['manifest'], kind='SOURCE_RECEIPT')
    store.json('inference.json', dict(**metadata, last_batch_inference_receipt=inference_receipt), kind='INFERENCE_RECEIPT')
    outer_cohort, outer_case_ids = outer['cohort_admission_digest'], outer['case_ids_sha256']
    del predictor, outer, output, batch, buffer
    gc.collect()
    if job['device'] == 'mps':
        guard.torch.mps.empty_cache()
    guard.stage = 'ALL_PREDICTIONS_COMMITTED_LOAD_LABELS'; guard.check(force=True)
    labels = load_outer_labels(corpus_root, job['data_job'], expected_outer_cohort_digest=outer_cohort,
                              expected_case_ids_sha256=outer_case_ids, check=guard.check)
    require(np.array_equal(labels['arrays']['case_id'], coordinates['case_id'])
            and np.array_equal(labels['arrays']['target_time'], coordinates['forecast_time']), 'Prediction/label coordinate join differs')
    identity = ScoreIdentity(job['architecture'], f'{job["architecture"]}-{job["input_arm"]}-L{job["history_hours"]}',
        job['gauge_id'], checkpoint['feature_schema'], job['history_hours'], job['fold'], job['seed'])
    panel = score_physical_h18(case_ids=tuple(coordinates['case_id'].tolist()), origins=coordinates['origin_time'],
        forecast_time=coordinates['forecast_time'], labels=labels['arrays']['target'], label_mask=labels['arrays']['target_mask'],
        predictions=predictions, identity=identity, input_cohort_digest=outer_cohort,
        label_policy_digest=digest({'policy': PRIMARY_APPROVED}), label_source_digest=labels['source_identity'])
    metrics = dict(**ledger.receipt_bindings(lease), checkpoint_sha256=checkpoint['hashes']['checkpoint'],
        predictions_sha256=prediction_artifact['sha256'], outer_case_ids_sha256=outer_case_ids,
        campaign_protocol_adopted=True, component_policy_status_preserved=True, panel=panel,
        target_imputation=False, prediction_driven_case_removal=False, protected_assessment_access=False)
    metric_artifact = store.json('metrics.json', metrics, kind='METRICS')
    store.json('outer-label-source.json', labels['manifest'], kind='SOURCE_RECEIPT')
    support = dict(**ledger.receipt_bindings(lease), requested_cases=count, predicted_cases=seen,
        score_status=panel['status'], primary_score=panel['primary_score'],
        leads=[{key: row[key] for key in ('lead_hours', 'n', 'n_requested', 'distinct_target_hours', 'distinct_utc_dates',
                                         'primary_eligible', 'primary_ineligibility')} for row in panel['leads']],
        no_support_panels_omitted=True, outer_case_ids_sha256=outer_case_ids)
    store.json('support.json', support, kind='SUPPORT')
    guard.stage = 'SCORING_COMPLETE'; guard.check(force=True)
    return {**checkpoint, 'status': 'VERIFIED_TRAIN_EVAL',
        'hashes': {**checkpoint['hashes'], 'predictions': prediction_artifact['sha256'], 'metrics': metric_artifact['sha256']},
        'artifacts': {**checkpoint['artifacts'], 'predictions': prediction_artifact, 'metrics': metric_artifact},
        'inference_source_identity': metadata['inference_source_identity'], 'inference_cohort_admission_digest': outer_cohort,
        'label_source_identity': labels['source_identity'], 'outer_case_ids_sha256': outer_case_ids,
        'scored_cases': count, 'score_status': panel['status'], 'protected_assessment_access': False}


def run_job(plan, corpus_root, run_directory, ledger_path, resume=False):
    """Run exactly one next job, or explicitly resume its latest recorded prefix."""
    require(type(resume) is bool, 'Explicit resume boolean required')
    plan = validate_execution_plan(plan)
    run = _private_root(run_directory); ledger_directory = _private_root(ledger_path)
    require(run != ledger_directory and not ledger_directory.is_relative_to(run / 'jobs'), 'Separate persistent ledger directory required')
    require(plan.get('run_directory') == str(run) and plan.get('ledger_directory') == str(ledger_directory),
            'Immutable campaign run/ledger paths differ; moving a campaign cannot reset its allowance')
    _offline(); _directory(run)
    ledger = CampaignLedger(ledger_directory)
    lease = guard = None; old_alarm = old_term = None
    with ledger.writer():
        ledger.initialize(plan)
        require(ledger.snapshot().get('bookkeeping_bytes_reserved', 0) >= 2 * GIB, 'Global2GiB bookkeeping reservation required before a fit')
        next_job = ledger.next_job()
        if next_job is None:
            return dict(status='CAMPAIGN_ALREADY_COMPLETE', plan_sha256=plan['plan_sha256'])
        job = next_job['job']; validator = artifact_validator(run, plan, job)
        resume_receipt = ledger.latest_checkpoint(job['job_id']) if resume else None
        if resume:
            require(resume_receipt is not None, 'Explicit resume requires a previously authenticated checkpoint')
            lease = ledger.resume(job['job_id'], resume_receipt, validator=validator)
        else:
            lease = ledger.claim_next()
        try:
            directory = run / 'jobs' / job['job_id'] / f'attempt-{lease.attempt_id:06d}'
            require(not directory.exists(), 'Attempt output path already exists; no overwrite')
            _directory(directory)
            active = ledger.next_job(); snapshot = ledger.snapshot()
            deadline = min(active['deadline'], snapshot['campaign_deadline'])
            def interrupted(*unused):
                raise CampaignInterrupted('SIGNAL_OR_DEADLINE')
            old_alarm = signal.signal(signal.SIGALRM, interrupted); old_term = signal.signal(signal.SIGTERM, interrupted)
            signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.time()))
            torch = _torch_runtime(job['device'])
            guard = ResourceGuard(ledger, lease, run, directory, torch, job)
            guard.check(force=True)
            store = ArtifactStore(run, directory, ledger, lease)
            checkpoint, training_source = _train(ledger, lease, job, plan, corpus_root, store, guard, resume_receipt)
            receipt = _evaluate(ledger, lease, job, plan, corpus_root, store, guard, checkpoint, training_source)
            receipt['receipt_path'] = str((directory / 'completion.json').relative_to(run))
            store.json('completion.json', receipt, kind='COMPLETION_RECEIPT')
            guard.stage = 'ARTIFACTS_COMMITTED_AWAITING_LEDGER_VERIFICATION'; guard.check(force=True); guard.settle(); guard = None
            ledger.complete(lease, receipt, validator=validator)
            return receipt
        except BaseException as failure:
            if guard is not None:
                try:
                    guard.settle()
                except BaseException:
                    pass  # Preserve any unresolved reservation; never refund it.
            if isinstance(failure, (KeyboardInterrupt, CampaignInterrupted, AllowanceExhausted)):
                ledger.interrupt(lease, 'INTERRUPTED_OR_RESOURCE_GUARD')
            else:
                ledger.fail(lease, 'WORKER_' + re.sub('[^A-Z0-9_]', '_', type(failure).__name__.upper())[:80])
            raise
        finally:
            if old_alarm is not None:
                signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old_alarm)
            if old_term is not None:
                signal.signal(signal.SIGTERM, old_term)
