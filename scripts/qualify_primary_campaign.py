#!/usr/bin/env python3
"""Plan-first synthetic primary geometry/restart/best-inference qualification.

A new once-only 36-update allowance. No study data, network, cloud, old
checkpoint migration, automatic retry, or scientific campaign launch.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from qualify_best_inference import atomic_json, mkdir_durable, read_bytes, read_json, canonical, sha

MAX_UPDATES, MAX_SECONDS, MAX_BYTES, MPS_BYTES = 36, 900, 4 * 1024**3, 2 * 1024**3
MAX_SCALER_FITS = 12
LEDGER = ROOT / '.cache/primary-campaign-qualification-budget-v1.json'
LOCK = ROOT / '.cache/primary-campaign-qualification-budget-v1.lock'
RUN = ROOT / '.local/primary-campaign-qualification-v1'
REPORT = ROOT / 'reports/primary-campaign-qualification-v1.json'
FULL, Q_ONLY = 'FULL_HRRR_H2_W4_S2_V1', 'Q_ONLY_H2_HORIZON1_S0_V1'
GAUGE = '03161000'
ROLE_BOUNDS = {
    'fit': {'start_utc': '2019-01-01T00:00:00Z', 'end_utc': '2019-01-10T00:00:00Z'},
    'inner_stop': {'start_utc': '2019-01-10T00:00:00Z', 'end_utc': '2019-01-20T00:00:00Z'},
    'outer': {'start_utc': '2019-01-20T00:00:00Z', 'end_utc': '2019-02-01T00:00:00Z'}}
REFIT = '2019-01-20T00:00:00Z'
CODE = ('scripts/qualify_primary_campaign.py', 'scripts/qualify_best_inference.py',
        'src/hydra_directflow/training.py', 'src/hydra_directflow/inference.py',
        'src/hydra_directflow/scoring.py', 'src/hydra_directflow/evaluation.py',
        'src/hydra_directflow/trainer_bundle.py', 'src/hydra_directflow/engineering.py',
        'src/hydra_directflow/ablations.py', 'src/hydra_directflow/contract.py',
        'src/hydra_directflow/reference.py', 'src/hydra_directflow/local_data.py',
        'src/hydra_directflow/models/directflow.py', 'src/hydra_directflow/models/mamba2_math.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def epoch(value):
    return int(datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp())


def cases():
    return [dict(id=f'{architecture}-{arm}', architecture=architecture, device=device,
                 arm=arm, feature_schema=schema, history_hours=hours, batch_size=64)
            for architecture, device in (('LSTM', 'cpu'), ('Transformer', 'mps'),
                                        ('GRU_Transformer', 'cpu'), ('Mamba2_Transformer', 'mps'))
            for arm, schema, hours in (('full168', FULL, 168), ('q168', Q_ONLY, 168), ('full24', FULL, 24))]


def workers():
    return [dict(id=case['id'] + '-' + phase, case_id=case['id'], phase=phase,
                 updates=updates, scaler_fits=1 if phase == 'baseline' else 0)
            for case in cases() for phase, updates in (('baseline', 2), ('resume', 1), ('infer', 0))]


def implementation():
    return dict(code={name: sha(read_bytes(ROOT / name, 512 * 1024)) for name in CODE},
                dependencies={name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'transformers', 'psutil')})


def plan():
    spec = dict(source='SYNTHETIC_GENERATOR_ONLY_NO_EXTERNAL_ARRAY_ARGUMENT', cases=cases(), workers=workers(),
                maximum_optimizer_updates=MAX_UPDATES, maximum_synthetic_scaler_fits=MAX_SCALER_FITS,
                maximum_wall_seconds=MAX_SECONDS, maximum_parent_child_rss_plus_mps_bytes=MAX_BYTES,
                maximum_mps_driver_bytes=MPS_BYTES, mps_allocator_fraction_cap_bytes=MPS_BYTES,
                fit_cases=64, inner_cases=64, outer_cases=64, training_gauge_id=GAUGE,
                role_time_bounds=ROLE_BOUNDS, refit_at_utc=REFIT, simulated_label_delay_hours=2,
                width=64, dtype='float32', torch_threads=1, seed=17, batch_size=64,
                learning_rate=.0001, epochs=2, min_epochs=2, patience=2, min_improvement=1.e9,
                stopping_policy_scope='SYNTHETIC_ONLY_LARGE_IMPROVEMENT_RETAINS_FIRST_BEST_NOT_A_SCIENCE_POLICY',
                original_gradient_transform_fits=12, resume_transform_fits=0,
                fresh_process_restart=True, independent_label_free_best_inference=True,
                exact_loss_weights_optimizer_sampler_cpu_mps_rng=True,
                synthetic_scoring_only=True, primary_support_expected='UNAVAILABLE_64_CASES',
                scientific_fits=0, protected_assessment_access=False, cloud_jobs=0,
                mps_fallback=False, automatic_retry=False, implementation=implementation())
    require(sum(w['updates'] for w in workers()) == MAX_UPDATES, 'Fixed 36-update scope differs')
    return dict(schema='hydra-primary-campaign-qualification-plan/v1', status='PREPARED_NOT_EXECUTED',
                spec=spec, plan_sha256=sha(canonical(spec)), optimizer_updates=0)


def locked():
    require(LOCK.parent.resolve().is_relative_to(ROOT.resolve()), 'Ledger outside stable checkout')
    mkdir_durable(LOCK.parent)
    stream = LOCK.open('a+')
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
    return stream


def validate_paths(run, report=None):
    require(Path(run).absolute() == RUN and RUN.resolve() == RUN and not RUN.is_symlink(),
            'Only the fixed new stable qualification run is permitted')
    if report is not None:
        require(Path(report).absolute() == REPORT and REPORT.resolve() == REPORT and not REPORT.is_symlink(),
                'Only the fixed new qualification receipt is permitted')
    require(ROOT.resolve() == ROOT and LEDGER.resolve().is_relative_to(ROOT), 'Stable local checkout required')


def precharge(prepared, run):
    validate_paths(run)
    require(prepared == plan(), 'Prepared implementation changed')
    with locked():
        require(not LEDGER.exists() and not LEDGER.is_symlink(), 'Qualification allowance consumed; no retry/reset')
        record = dict(schema='hydra-primary-campaign-consumed-allowance/v1', run_directory=str(run),
                      plan_sha256=prepared['plan_sha256'], precharged_optimizer_updates=MAX_UPDATES,
                      precharged_synthetic_scaler_fits=MAX_SCALER_FITS, deadline_epoch=time.time() + MAX_SECONDS,
                      created_at_utc=utc(), workers={w['id']: {'status': 'PENDING'} for w in workers()})
        atomic_json(LEDGER, record)
    return record


def prior_result(run, identifier):
    ledger = read_json(LEDGER)
    record = ledger['workers'][identifier]
    require(record['status'] == 'PASSED', 'Prerequisite qualification worker has not passed')
    raw = read_bytes(run / identifier / 'result.json', 2 * 1024**2)
    require(sha(raw) == record['result_sha256'], 'Prerequisite result bytes changed')
    return json.loads(raw)


def claim(run, worker, prepared):
    validate_paths(run)
    require(worker in workers(), 'Exact declared worker required')
    with locked():
        record = read_json(LEDGER)
        require(record['run_directory'] == str(run) and record['plan_sha256'] == prepared['plan_sha256']
                and record['precharged_optimizer_updates'] == MAX_UPDATES
                and record['precharged_synthetic_scaler_fits'] == MAX_SCALER_FITS,
                'Exact precharged worker allowance required')
        require(time.time() < record['deadline_epoch'], 'Qualification deadline expired')
        require(record['workers'][worker['id']]['status'] == 'PENDING', 'Worker replay forbidden')
        if worker['phase'] != 'baseline':
            prior_result(run, worker['case_id'] + '-baseline')
        if worker['phase'] == 'infer':
            prior_result(run, worker['case_id'] + '-resume')
        record['workers'][worker['id']] = dict(status='CLAIMED', pid=os.getpid(), claimed_at_utc=utc())
        atomic_json(LEDGER, record)
    return record['deadline_epoch']


def checked_counts(value, worker):
    entered, confirmed = (value.get(key, 0) for key in ('optimizer_updates_entered', 'optimizer_updates_confirmed'))
    fit_entered, fits = (value.get(key, 0) for key in ('scaler_fits_entered', 'scaler_fits_confirmed'))
    require(all(type(x) is int for x in (entered, confirmed, fit_entered, fits))
            and 0 <= confirmed <= entered <= worker['updates']
            and 0 <= fits <= fit_entered <= worker['scaler_fits'], 'Worker counters exceed exact allowance')
    return entered, confirmed, fit_entered, fits


def fixture(case, *, outer=False):
    """Generate only synthetic values; the outer input mapping has no label keys."""
    import numpy as np
    from hydra_directflow.local_data import iso
    from hydra_directflow.training import array_identity, q_only_training_bundle
    starts = ('2019-01-21T00:00:00Z',) if outer else ('2019-01-02T00:00:00Z', '2019-01-11T00:00:00Z')
    origins = np.concatenate([epoch(start) + np.arange(64, dtype=np.int64) * 3600 for start in starts])
    n, hours = len(origins), case['history_hours']
    history = origins[:, None] - np.arange(hours + 1, 1, -1) * 3600
    future = origins[:, None] + np.arange(1, 19) * 3600
    values = np.stack((synthetic_q(history), np.broadcast_to(np.arange(hours + 1, 1, -1), (n, hours))), axis=-1)
    weather = np.stack((.1 + .03 * np.sin(future / 3600 / 13), 285 + 4 * np.sin(future / 3600 / 24),
                        .005 + .0001 * np.cos(future / 3600 / 7), np.full(future.shape, .02)), axis=-1)
    state = np.tile([.25, 4.], (n, 1))
    ids = np.array([GAUGE + ':' + iso(int(t)) for t in origins])
    cohort = sha(canonical(dict(source='SYNTHETIC_PRIMARY_QUALIFICATION_V1', case_ids=ids.tolist())))
    data = dict(origin_time=origins, history_time=history, target_time=future, site_id=np.zeros(n, np.int64),
                case_id=ids, history=values, history_mask=np.ones(values.shape, bool), weather=weather,
                weather_mask=np.ones(weather.shape, bool), state=state, state_mask=np.ones(state.shape, bool),
                feature_schema=np.array(FULL), cohort_admission_digest=np.array(cohort))
    # Q-only adapter expects labels; for outer use explicit horizon-only geometry
    # instead, so synthetic outer label values are not constructed before inference.
    if not outer:
        data.update(target=synthetic_q(future), target_mask=np.ones((n, 18), bool))
    if case['feature_schema'] == Q_ONLY:
        if outer:
            data.update(weather=np.broadcast_to(np.arange(1, 19, dtype=float)[None, :, None], (n, 18, 1)).copy(),
                        weather_mask=np.ones((n, 18, 1), bool), state=np.empty((n, 0)),
                        state_mask=np.empty((n, 0), bool), feature_schema=np.array(Q_ONLY))
        else:
            data = q_only_training_bundle(data, cohort_admission_digest=cohort)
    if outer:
        data['forecast_time'] = data.pop('target_time')
    source = sha(canonical(dict(synthetic_source='PRIMARY_QUALIFICATION_V1', array_sha256=array_identity(data))))
    return data, source, cohort


def synthetic_q(times):
    import numpy as np
    hours = (np.asarray(times, dtype=np.float64) - epoch('2018-12-01T00:00:00Z')) / 3600
    return 25. + 3. * np.sin(hours / 24.) + .3 * np.cos(hours / 7.)


def validate_roles(arrays):
    import numpy as np
    from hydra_directflow.trainer_bundle import refit_label_mask
    for name, rows in (('fit', slice(0, 64)), ('inner_stop', slice(64, 128))):
        bounds = ROLE_BOUNDS[name]
        require(np.all(arrays['origin_time'][rows] >= epoch(bounds['start_utc']))
                and np.all(arrays['target_time'][rows] < epoch(bounds['end_utc'])), 'Synthetic role window differs')
    require(refit_label_mask(arrays['target_time'], refit_at_utc=REFIT, simulated_label_delay_hours=2).all(),
            'Synthetic label availability exceeds refit')
    require(np.asarray(arrays['target_mask']).all(), 'Complete synthetic H18 labels required')


def tree_digest(value):
    import torch
    result = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            result.update(canonical(['tensor', str(tensor.dtype), list(tensor.shape)]))
            result.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            result.update(canonical(['dict', len(item)]))
            for key in sorted(item, key=lambda k: canonical([type(k).__name__, k])):
                result.update(canonical([type(key).__name__, key])); visit(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(canonical([type(item).__name__, len(item)]))
            for child in item: visit(child)
        else:
            result.update(canonical([type(item).__name__, item]))
    visit(value)
    return result.hexdigest()


def sync(torch, device):
    if device == 'mps':
        torch.mps.synchronize()


def checkpoint_value(path, expected):
    import torch
    raw = read_bytes(path, 16 * 1024**2)
    require(sha(raw) == expected, 'Qualification checkpoint SHA differs')
    return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)


def train_phase(run, worker, case, directory):
    import numpy as np
    import torch
    from hydra_directflow.training import DevelopmentTrainer, TrainerSpec, EarlyStopping, EpochSampler, array_identity
    arrays, source, cohort = fixture(case)
    validate_roles(arrays)
    if worker['phase'] == 'baseline':
        spec = TrainerSpec(case['architecture'], case['device'], case['history_hours'], 64, .0001, 17, True, 2,
                           feature_schema=case['feature_schema'])
        trainer = DevelopmentTrainer(arrays, train_indices=tuple(range(64)), validation_indices=tuple(range(64, 128)),
                                     boundary_epoch=epoch(ROLE_BOUNDS['inner_stop']['start_utc']), spec=spec, source_identity=source)
    else:
        base = prior_result(run, case['id'] + '-baseline')
        trainer = DevelopmentTrainer.restore(run / (case['id'] + '-baseline') / 'epochs/epoch-0001-update-00000001.pt',
                    arrays, expected_sha256=base['prefix_checkpoint_sha256'], expected_source_identity=source,
                    expected_feature_schema=case['feature_schema'], expected_cohort_admission_digest=cohort)
        next_batch = EpochSampler.restore(trainer.sampler.state()).next(64)
        require(next_batch == base['trace'][1]['indices'], 'Restored next-sample order differs before update')
    trace, timings = [], []
    original_update = trainer.update
    def traced_update():
        sync(torch, case['device']); start = time.monotonic()
        value = original_update(); sync(torch, case['device'])
        timings.append(time.monotonic() - start); trace.append(value)
        return value
    trainer.update = traced_update
    stop = trainer.stopping if trainer.stopping is not None else EarlyStopping(2, 2, 2, 1.e9)
    trainer.fit_epochs(stop, checkpoint_directory=directory / 'epochs')
    checkpoint_sha = trainer.checkpoint(directory / 'checkpoint.pt')
    saved = checkpoint_value(directory / 'checkpoint.pt', checkpoint_sha)
    result = dict(trace=trace, update_seconds=timings, checkpoint_sha256=checkpoint_sha,
                  source_identity=source, cohort_digest=cohort, gradient_case_ids=arrays['case_id'][:64].tolist(),
                  array_sha256=array_identity(arrays), parameter_count=sum(p.numel() for p in trainer.model.parameters()),
                  sampler=trainer.sampler.state(), scaler=asdict(trainer.scaler), stopping=asdict(trainer.stopping),
                  epoch_records=trainer.epoch_records, current_updates=trainer.updates,
                  model_sha256=tree_digest(saved['model']), optimizer_sha256=tree_digest(saved['optimizer']),
                  best_model_sha256=tree_digest(saved['best_model']), cpu_rng_sha256=tree_digest(saved['cpu_rng']),
                  mps_rng_sha256=tree_digest(saved['mps_rng']))
    if worker['phase'] == 'baseline':
        prefix = directory / 'epochs/epoch-0001-update-00000001.pt'
        result['prefix_checkpoint_sha256'] = sha(read_bytes(prefix, 16 * 1024**2))
        from hydra_directflow.inference import input_only_view, transform_inputs, selected_best_epoch
        outer, outer_source, outer_cohort = fixture(case, outer=True)
        inputs = input_only_view(outer, inference_source_identity=outer_source)
        mapped = transform_inputs(inputs, trainer.scaler)
        tensors = {key: torch.from_numpy(value).to(case['device']) for key, value in mapped.items()}
        arguments = [tensors[key] for key in ('history', 'history_mask', 'weather', 'weather_mask', 'state', 'state_mask')]
        trainer.model.eval()
        with torch.inference_mode():
            last = trainer.scaler.inverse_target(trainer.model(*arguments).detach().cpu().numpy().astype(np.float64))
            trainer.model.load_state_dict(trainer.best_model_state)
            best = trainer.scaler.inverse_target(trainer.model(*arguments).detach().cpu().numpy().astype(np.float64))
        require(selected_best_epoch(saved) == 1 and saved['stopping']['completed_epochs'] == 2,
                'Synthetic policy did not select a nonlast best epoch')
        require(not np.array_equal(best, last), 'Best-versus-last prediction distinction not established')
        with (directory / 'best-reference.npy').open('xb') as stream:
            np.save(stream, best, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
        result.update(best_reference_sha256=sha(read_bytes(directory / 'best-reference.npy', 128 * 1024)),
                      best_prediction_sha256=array_identity({'prediction_q': best}),
                      last_prediction_sha256=array_identity({'prediction_q': last}),
                      inference_source_identity=outer_source, inference_cohort_digest=outer_cohort,
                      selected_best_epoch=1, completed_epochs=2, best_differs_from_last=True)
    return result


def infer_phase(run, worker, case, directory):
    import numpy as np
    import torch
    from unittest.mock import patch
    from hydra_directflow.training import DevelopmentTrainer, TrainOnlyTransform, array_identity
    from hydra_directflow.inference import BestCheckpointPredictor, input_only_view
    from hydra_directflow.scoring import ScoreIdentity, score_physical_h18
    base = prior_result(run, case['id'] + '-baseline')
    arrays, source, cohort = fixture(case, outer=True)
    require((source, cohort) == (base['inference_source_identity'], base['inference_cohort_digest']),
            'Separate inference source/cohort differs')
    class PoisonLabels(dict):
        def __getitem__(self, key):
            if key in ('target', 'target_mask'):
                raise AssertionError('Inference attempted verification-label access')
            return super().__getitem__(key)
    with (patch.object(torch.optim.Optimizer, '__init__', side_effect=AssertionError('Inference optimizer creation')),
          patch.object(TrainOnlyTransform, 'fit', side_effect=AssertionError('Inference scaler fitting')),
          patch.object(DevelopmentTrainer, '__init__', side_effect=AssertionError('Inference trainer creation'))):
        inputs = input_only_view(PoisonLabels(arrays), inference_source_identity=source)
        predictor = BestCheckpointPredictor.load(run / (case['id'] + '-baseline') / 'checkpoint.pt',
            expected_sha256=base['checkpoint_sha256'], expected_source_identity=base['source_identity'],
            expected_feature_schema=case['feature_schema'], expected_training_cohort_admission_digest=base['cohort_digest'],
            expected_gauge_id=GAUGE, gradient_fit_case_ids=tuple(base['gradient_case_ids']))
        output = predictor.predict(inputs, expected_inference_source_identity=source,
                                   expected_inference_cohort_admission_digest=cohort)
    raw = read_bytes(run / (case['id'] + '-baseline') / 'best-reference.npy', 128 * 1024)
    require(sha(raw) == base['best_reference_sha256'], 'Reference prediction bytes changed')
    reference = np.load(io.BytesIO(raw), allow_pickle=False)
    require(np.array_equal(output['prediction_q'], reference), 'Fresh best-only physical predictions differ')
    require(output['receipt']['selected_best_epoch'] == 1 and not output['receipt']['last_state_used'], 'Best selection differs')
    # Verification labels are created only after label-free inference is complete.
    labels = synthetic_q(output['forecast_time'])
    panel = score_physical_h18(case_ids=tuple(output['case_id'].tolist()), origins=output['origin_time'],
        forecast_time=output['forecast_time'], labels=labels, label_mask=np.ones(labels.shape, bool),
        predictions=output['prediction_q'], identity=ScoreIdentity(case['architecture'], 'synthetic-qualification', GAUGE,
            case['feature_schema'], case['history_hours'], 'cv_1', 17), input_cohort_digest=cohort,
        label_policy_digest=sha(b'SYNTHETIC_COMPLETE_LABELS'), label_source_digest=sha(b'SYNTHETIC_PRIMARY_QUALIFICATION_V1'))
    require(panel['status'] == 'PRIMARY_UNAVAILABLE' and panel['primary_score'] is None
            and all(row['n'] == 64 and row['physical_mse'] is not None for row in panel['leads']),
            'Synthetic scoring/support behavior differs')
    atomic_json(directory / 'synthetic-score.json', panel)
    return dict(exact_fresh_best_predictions=True, saved_transform_reused_without_fit=True,
                best_prediction_sha256=array_identity({'prediction_q': output['prediction_q']}),
                inference_receipt=output['receipt'], synthetic_score_sha256=sha(read_bytes(directory / 'synthetic-score.json', 256 * 1024)),
                synthetic_score_status=panel['status'], scored_cases=64, leads=18,
                scientific_scoring_performed=False, optimizer_constructed=False, scaler_fitted=False)


def worker(run, identifier):
    run = Path(run).absolute(); validate_paths(run)
    prepared = read_json(run / 'plan.json')
    require(prepared == plan(), 'Worker implementation/plan changed')
    declaration = next(w for w in workers() if w['id'] == identifier)
    case = next(c for c in cases() if c['id'] == declaration['case_id'])
    deadline = claim(run, declaration, prepared)
    def deny_network(event, unused):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
            raise RuntimeError('Synthetic qualification network forbidden')
    sys.addaudithook(deny_network)
    def timeout(*unused):
        raise TimeoutError('Primary qualification worker deadline')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.time()))
    require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0', 'Explicit no-MPS-fallback environment required')
    import torch
    import psutil
    from unittest.mock import patch
    from hydra_directflow.training import TrainOnlyTransform
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    if case['device'] == 'mps':
        require(torch.backends.mps.is_available(), 'MPS unavailable; no CPU fallback')
        torch.mps.set_per_process_memory_fraction(min(1., MPS_BYTES / torch.mps.recommended_max_memory()))
    directory = run / identifier; process = psutil.Process()
    state = dict(pid=os.getpid(), worker_id=identifier, plan_sha256=prepared['plan_sha256'],
                 optimizer_updates_entered=0, optimizer_updates_confirmed=0,
                 scaler_fits_entered=0, scaler_fits_confirmed=0, peak_rss_bytes=0, mps_driver_bytes=0)
    def progress():
        state['peak_rss_bytes'] = max(state['peak_rss_bytes'], process.memory_info().rss)
        driver = torch.mps.driver_allocated_memory() if case['device'] == 'mps' else 0
        state['mps_driver_bytes'] = max(state['mps_driver_bytes'], driver)
        atomic_json(directory / 'status.json', state)
        require(time.time() < deadline and state['peak_rss_bytes'] + driver <= MAX_BYTES and driver <= MPS_BYTES,
                'Worker resource cap exceeded')
    original_step = torch.optim.AdamW.step
    original_fit = TrainOnlyTransform.fit.__func__
    def step(optimizer, *args, **kwargs):
        require(state['optimizer_updates_entered'] < declaration['updates'], 'Worker update allowance exhausted')
        state['optimizer_updates_entered'] += 1; progress()
        result = original_step(optimizer, *args, **kwargs)
        sync(torch, case['device']); state['optimizer_updates_confirmed'] += 1; progress()
        return result
    def fit(cls, *args, **kwargs):
        require(state['scaler_fits_entered'] < declaration['scaler_fits'], 'Worker scaler-fit allowance exhausted')
        state['scaler_fits_entered'] += 1; progress()
        result = original_fit(cls, *args, **kwargs)
        state['scaler_fits_confirmed'] += 1; progress()
        return result
    progress()
    try:
        with patch.object(torch.optim.AdamW, 'step', step), patch.object(TrainOnlyTransform, 'fit', classmethod(fit)):
            result = infer_phase(run, declaration, case, directory) if declaration['phase'] == 'infer' else train_phase(run, declaration, case, directory)
        require(checked_counts(state, declaration) == (declaration['updates'], declaration['updates'],
                declaration['scaler_fits'], declaration['scaler_fits']), 'Worker counts incomplete')
        progress(); atomic_json(directory / 'result.json', {**result, **state, 'status': 'PASSED'})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def supervise(run, declaration, deadline, env, plan_sha):
    import psutil
    directory = run / declaration['id']; mkdir_durable(directory)
    peak = 0; child = result = error = None; counts = [0, 0, 0, 0]
    def observe(value):
        nonlocal counts
        counts = [max(a, b) for a, b in zip(counts, checked_counts(value, declaration))]
    try:
        with (directory / 'worker.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--worker', declaration['id']],
                                     cwd=ROOT, env=env, stdout=log, stderr=log)
            process, parent = psutil.Process(child.pid), psutil.Process()
            while child.poll() is None:
                status = read_json(directory / 'status.json') if (directory / 'status.json').exists() else {}
                observe(status)
                driver = status.get('mps_driver_bytes', 0)
                require(type(driver) is int and 0 <= driver <= MPS_BYTES, 'MPS driver cap exceeded')
                try:
                    peak = max(peak, process.memory_info().rss + parent.memory_info().rss + driver)
                except psutil.NoSuchProcess:
                    pass
                require(peak <= MAX_BYTES and time.time() < deadline, 'Qualification supervisor resource cap exceeded')
                time.sleep(.05)
            if (directory / 'result.json').exists():
                result = read_json(directory / 'result.json', 2 * 1024**2)
    except BaseException as failure:
        error = type(failure).__name__
    finally:
        if child is not None and child.poll() is None:
            child.kill(); child.wait(timeout=5)
        for name in ('status.json', 'result.json'):
            if (directory / name).exists():
                try:
                    value = read_json(directory / name, 2 * 1024**2)
                    observe(value)
                    driver, worker_peak = value.get('mps_driver_bytes', 0), value.get('peak_rss_bytes', 0)
                    require(type(driver) is int and 0 <= driver <= MPS_BYTES
                            and type(worker_peak) is int and 0 <= worker_peak <= MAX_BYTES, 'Final worker memory counters invalid')
                    peak = max(peak, worker_peak + driver + psutil.Process().memory_info().rss)
                    require(peak <= MAX_BYTES and time.time() < deadline, 'Final supervisor resource cap exceeded')
                except BaseException as failure:
                    error = error or type(failure).__name__
    expected = (declaration['updates'], declaration['updates'], declaration['scaler_fits'], declaration['scaler_fits'])
    passed = bool(child is not None and child.returncode == 0 and result is not None and result.get('status') == 'PASSED'
                  and result.get('pid') == child.pid and result.get('worker_id') == declaration['id']
                  and result.get('plan_sha256') == plan_sha and error is None and tuple(counts) == expected)
    return dict(worker_id=declaration['id'], passed=passed, pid=child.pid if child else None,
                exit_code=child.returncode if child else None, error_type=error,
                optimizer_updates_entered=counts[0], optimizer_updates_confirmed=counts[1],
                scaler_fits_entered=counts[2], scaler_fits_confirmed=counts[3], count_exact=passed,
                peak_parent_child_rss_plus_mps_bytes=peak,
                result_sha256=sha(read_bytes(directory / 'result.json', 2 * 1024**2)) if passed else None)


def compare_case(base, resumed, inferred):
    require(len(base['trace']) == 2 and resumed['trace'] == base['trace'][1:], 'Exact next-update loss/sample order differs')
    keys = ('model_sha256', 'optimizer_sha256', 'sampler', 'scaler', 'stopping', 'best_model_sha256',
            'epoch_records', 'cpu_rng_sha256', 'mps_rng_sha256', 'array_sha256', 'source_identity', 'cohort_digest')
    for key in keys:
        require(base[key] == resumed[key], 'Exact restored state differs: ' + key)
    require(inferred['exact_fresh_best_predictions'] and inferred['best_prediction_sha256'] == base['best_prediction_sha256']
            and base['best_differs_from_last'], 'Fresh best-only inference differs')
    require(len({base['pid'], resumed['pid'], inferred['pid']}) == 3, 'Separate worker processes required')
    return dict(exact_next_loss_samples_weights_optimizer_rng=True, exact_fresh_best_prediction=True,
                selected_best_epoch=base['selected_best_epoch'], completed_epochs=base['completed_epochs'],
                best_differs_from_last=True, parameter_count=base['parameter_count'],
                baseline_update_seconds=base['update_seconds'], synthetic_score_status=inferred['synthetic_score_status'])


def execute(prepared, approved_plan_sha256):
    require(prepared == plan() and prepared['plan_sha256'] == approved_plan_sha256, 'Exact reviewed plan required')
    validate_paths(RUN, REPORT)
    require(not RUN.exists() and not REPORT.exists(), 'Fresh qualification run and receipt required')
    consumed = precharge(prepared, RUN)
    mkdir_durable(RUN); atomic_json(RUN / 'plan.json', prepared); mkdir_durable(REPORT.parent)
    receipt = dict(schema='hydra-primary-campaign-qualification/v1', status='RUNNING', plan=prepared,
                   created_at_utc=utc(), precharged_optimizer_updates=MAX_UPDATES,
                   precharged_synthetic_scaler_fits=MAX_SCALER_FITS, optimizer_updates_entered=0,
                   optimizer_updates_confirmed=0, scaler_fits_entered=0, scaler_fits_confirmed=0,
                   count_exact=False, attempts=[], checks=[], scientific_fits=0,
                   study_source_reads=0, network_requests=0, protected_assessment_access=False)
    atomic_json(REPORT, receipt); started = time.monotonic()
    env = {**os.environ, 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(ROOT / 'src'),
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1', 'PYTORCH_ENABLE_MPS_FALLBACK': '0'}
    try:
        for declaration in workers():
            attempt = supervise(RUN, declaration, consumed['deadline_epoch'], env, prepared['plan_sha256'])
            receipt['attempts'].append(attempt)
            for key in ('optimizer_updates_entered', 'optimizer_updates_confirmed', 'scaler_fits_entered', 'scaler_fits_confirmed'):
                receipt[key] += attempt[key]
            atomic_json(REPORT, receipt)
            with locked():
                ledger = read_json(LEDGER)
                ledger['workers'][declaration['id']] = {**ledger['workers'][declaration['id']], **attempt,
                                                      'status': 'PASSED' if attempt['passed'] else 'FAILED'}
                atomic_json(LEDGER, ledger)
            require(attempt['passed'], 'Qualification failed; allowance and evidence preserved without retries')
            if declaration['phase'] == 'infer':
                result = [prior_result(RUN, declaration['case_id'] + '-' + phase) for phase in ('baseline', 'resume', 'infer')]
                receipt['checks'].append(dict(case_id=declaration['case_id'], **compare_case(*result)))
                atomic_json(REPORT, receipt)
        require(receipt['optimizer_updates_entered'] == receipt['optimizer_updates_confirmed'] == MAX_UPDATES
                and receipt['scaler_fits_entered'] == receipt['scaler_fits_confirmed'] == MAX_SCALER_FITS, 'Campaign counters differ')
        receipt.update(status='PASSED_SYNTHETIC_PRIMARY_CAMPAIGN_QUALIFICATION', count_exact=True)
    except BaseException as failure:
        receipt.update(status='FAILED_WITH_CONSUMED_ALLOWANCE', error_type=type(failure).__name__)
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - started
        receipt['finished_at_utc'] = utc()
        atomic_json(REPORT, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--approved-plan-sha256')
    parser.add_argument('--worker', choices=[w['id'] for w in workers()]); args = parser.parse_args()
    if args.worker:
        worker(RUN, args.worker); return
    prepared = plan()
    if not args.execute:
        print(json.dumps(prepared, indent=2)); return
    result = execute(prepared, args.approved_plan_sha256)
    print(json.dumps({key: result.get(key) for key in ('status', 'optimizer_updates_entered', 'optimizer_updates_confirmed', 'elapsed_seconds')}))
    require(result['status'] == 'PASSED_SYNTHETIC_PRIMARY_CAMPAIGN_QUALIFICATION', 'Qualification incomplete')


if __name__ == '__main__':
    main()
