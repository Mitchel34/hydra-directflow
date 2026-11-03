"""Pure mocked training/inference orchestration and durable-output guard tests."""
from contextlib import contextmanager
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from hydra_directflow import campaign_worker as w
from hydra_directflow.campaign_ledger import digest


class FakeLedger:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.plan = {'plan_sha256': 'a' * 64}
        self.reserved = {}; self.counter = 0; self.receipts = []
    def reserve_output(self, size, lease=None, *, kind):
        self.events.append(('reserve', kind, size)); self.counter += 1
        self.reserved[self.counter] = [size, kind, 'PRECHARGED']; return self.counter
    def settle_output(self, lease, token, *, actual_bytes, sha256=None, failed=False):
        self.events.append(('settle', self.reserved[token][1], actual_bytes, failed))
        assert self.reserved[token][2] == 'PRECHARGED'
        self.reserved[token][2] = 'FAILED' if failed else 'VERIFIED'
    def enter_update(self, lease):
        self.events.append(('enter',)); return 7
    def confirm_update(self, lease, token): self.events.append(('confirm', token))
    def check_resources(self, lease, **kwargs): self.events.append(('resources',))
    def receipt_bindings(self, lease): return {'plan_sha256': 'a' * 64, 'job_id': 'test-job'}
    def checkpoint(self, lease, receipt, *, validator):
        self.receipts.append(receipt); self.events.append(('checkpoint', receipt['completed_epochs']))


class FakeGuard:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.stage = ''; self.logical_updates = self.completed_epochs = 0
        self.torch = SimpleNamespace(optim=SimpleNamespace(Optimizer=type('FakeOptimizer', (), {'__init__': lambda self: None})))
    def check(self, **kwargs): self.events.append(('check', self.stage))
    def settle(self): self.events.append(('heartbeat-settle',))


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.run = Path(self.temp.name).resolve()
        self.directory = self.run / 'attempt'; self.directory.mkdir()
        self.events = []; self.ledger = FakeLedger(self.events); self.lease = SimpleNamespace(attempt_id=1)
        self.store = w.ArtifactStore(self.run, self.directory, self.ledger, self.lease)
    def tearDown(self): self.temp.cleanup()

    def test_exact_output_is_precharged_before_file_creation_and_settled_after_hash(self):
        original = self.ledger.reserve_output
        def reserve(*args, **kwargs):
            self.assertFalse((self.directory / 'data.json').exists())
            self.assertFalse((self.directory / 'data.json.writing').exists())
            return original(*args, **kwargs)
        self.ledger.reserve_output = reserve
        descriptor = self.store.write('data.json', b'{"test":true}\n', kind='JSON')
        self.assertEqual(descriptor['bytes'], 14)
        self.assertEqual(w._read_descriptor(self.run, descriptor, 100), b'{"test":true}\n')
        self.assertEqual(self.events[0], ('reserve', 'JSON', 14))
        self.assertEqual(self.events[-1], ('settle', 'JSON', 14, False))
        with self.assertRaisesRegex(ValueError, 'New private'):
            self.store.write('data.json', b'x', kind='JSON')

    def test_failed_write_keeps_partial_and_consumed_reservation(self):
        with patch.object(w.os, 'fsync', side_effect=OSError('synthetic fsync failure')):
            with self.assertRaises(OSError): self.store.write('data', b'payload', kind='JSON')
        self.assertTrue((self.directory / 'data.writing').exists())
        self.assertEqual(self.ledger.reserved[1], [7, 'JSON', 'FAILED'])
        with self.assertRaisesRegex(ValueError, 'Unsettled'):
            self.store.write('data', b'payload', kind='JSON')
        self.assertEqual(len(self.ledger.reserved), 1)

    def test_checkpoint_requires_exact_before_write_reservation(self):
        payload = b'synthetic-checkpoint'
        class Trainer:
            def checkpoint(inner, path, *, before_write):
                before_write(len(payload), w.hashlib.sha256(payload).hexdigest())
                self.assertEqual(self.events[0], ('reserve', 'CHECKPOINT', len(payload)))
                path.write_bytes(payload)
                return w.hashlib.sha256(payload).hexdigest()
        descriptor = self.store.checkpoint(Trainer(), name='checkpoint.pt')
        self.assertEqual(descriptor['bytes'], len(payload))
        self.assertEqual(self.ledger.reserved[1][2], 'VERIFIED')
        class Unreserved:
            def checkpoint(inner, path, *, before_write): return 'b' * 64
        with self.assertRaisesRegex(ValueError, 'reservation identity'):
            self.store.checkpoint(Unreserved(), name='bad.pt')

    def _saved_policy_fixture(self):
        from hydra_directflow.training import TrainerSpec, EarlyStopping, FULL_FEATURE_SCHEMA
        plan = {'training': {**w.FIXED_TRAINING, 'max_epochs':100}}
        job = dict(architecture='LSTM', device='cpu', history_hours=168, batch_size=64, seed=17,
                   logical_update_limit=200, input_arm='full')
        spec = asdict(TrainerSpec('LSTM','cpu',168,64,.0001,17,True,200,feature_schema=FULL_FEATURE_SCHEMA))
        runtime = dict(torch_threads=1,torch_interop_threads=1,default_dtype='torch.float32',deterministic_algorithms=True)
        saved = dict(spec=spec,stopping=asdict(EarlyStopping(100,10,5,0.)),updates=2,runtime=runtime)
        receipt = dict(trainer_spec=dict(spec),feature_schema=FULL_FEATURE_SCHEMA,completed_epochs=0,trainer_runtime=dict(runtime),
                       worker_execution_runtime={'deterministic_algorithms':True})
        return saved,receipt,plan,job

    def test_checkpoint_binds_all_frozen_trainer_and_stopping_policy_fields(self):
        saved,receipt,plan,job = self._saved_policy_fixture()
        w._validate_checkpoint_policy(saved,receipt,plan,job)
        for field,value in (('learning_rate',.001),('batch_size',32),('shuffle_training',False),
                            ('feature_schema','Q_ONLY_H2_HORIZON1_S0_V1')):
            with self.subTest(trainer_field=field):
                changed = {**saved, 'spec':{**saved['spec'],field:value}}
                changed_receipt = {**receipt,'trainer_spec':dict(changed['spec'])}
                with self.assertRaisesRegex(ValueError,'exact frozen trainer policy'):
                    w._validate_checkpoint_policy(changed,changed_receipt,plan,job)
        for field,value in (('max_epochs',200),('min_epochs',5),('patience',2),('min_improvement',.01)):
            with self.subTest(stopping_field=field):
                changed = {**saved,'stopping':{**saved['stopping'],field:value}}
                with self.assertRaisesRegex(ValueError,'exact frozen stopping policy'):
                    w._validate_checkpoint_policy(changed,receipt,plan,job)
        initial = {**saved,'stopping':None,'updates':0}
        w._validate_checkpoint_policy(initial,receipt,plan,job)
        with self.assertRaisesRegex(ValueError,'initial zero-update'):
            w._validate_checkpoint_policy({**initial,'updates':1},receipt,plan,job)

    def test_checkpoint_records_and_binds_deterministic_runtime(self):
        saved,receipt,plan,job = self._saved_policy_fixture()
        wrong_runtime = {**saved['runtime'],'deterministic_algorithms':False}
        with self.assertRaisesRegex(ValueError,'deterministic FP32 runtime'):
            w._validate_checkpoint_policy({**saved,'runtime':wrong_runtime},
                {**receipt,'trainer_runtime':wrong_runtime},plan,job)

    def test_runtime_fixture_keys_match_actual_core_constructor_without_model_calls(self):
        import ast
        import hydra_directflow.training as training
        tree = ast.parse(Path(training.__file__).read_text())
        trainer = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='DevelopmentTrainer')
        constructor = next(node for node in trainer.body if isinstance(node,ast.FunctionDef) and node.name=='__init__')
        assignments = [node for node in ast.walk(constructor) if isinstance(node,ast.Assign)
            and any(isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name)
                    and target.value.id=='self' and target.attr=='runtime' for target in node.targets)]
        self.assertEqual(len(assignments),1)
        value = assignments[0].value
        self.assertIsInstance(value,ast.Call); self.assertEqual(value.func.id,'dict')
        actual_fields = {item.arg for item in value.keywords}
        saved,receipt,plan,job = self._saved_policy_fixture()
        self.assertEqual(set(saved['runtime']),actual_fields)
        self.assertEqual(set(receipt['trainer_runtime']),actual_fields)

    def test_checkpoint_worker_determinism_is_observed_separately_from_saved_runtime(self):
        from hydra_directflow.training import TrainerSpec, FULL_FEATURE_SCHEMA
        observer = Mock(return_value=True)
        trainer = SimpleNamespace(spec=TrainerSpec('LSTM','cpu',168,64,.0001,17,True,200,feature_schema=FULL_FEATURE_SCHEMA),
            runtime={'torch_threads':1,'torch_interop_threads':1,'default_dtype':'torch.float32'},
            torch=SimpleNamespace(are_deterministic_algorithms_enabled=observer),stopping=None,updates=0,
            source_identity='a'*64,cohort_admission_digest='b'*64,array_sha256='c'*64)
        with patch('hydra_directflow.training.implementation_fingerprint', return_value={}):
            receipt = w._checkpoint_receipt(trainer, {'sha256':'d'*64}, self.ledger, self.lease)
            self.assertEqual(receipt['trainer_runtime'], trainer.runtime)
            self.assertEqual(receipt['worker_execution_runtime'], {'deterministic_algorithms':True})
            observer.assert_called_once()
            observer.return_value = False
            with self.assertRaisesRegex(ValueError,'requires deterministic execution'):
                w._checkpoint_receipt(trainer, {'sha256':'d'*64}, self.ledger, self.lease)
        saved,receipt,plan,job = self._saved_policy_fixture()
        with self.assertRaisesRegex(ValueError,'not explicitly observed'):
            w._validate_checkpoint_policy(saved,{**receipt,'worker_execution_runtime':{'deterministic_algorithms':False}},plan,job)

    def test_midjob_code_drift_rejects_artifact_before_checkpoint_or_model_read(self):
        code = self.run / 'bound.py'; code.write_bytes(b'original code')
        plan = {'implementation_sha256': {'bound.py': w.hashlib.sha256(code.read_bytes()).hexdigest()}}
        with patch.object(w, 'ROOT', self.run):
            w._validate_implementation(plan)
            callback = w.artifact_validator(self.run, plan, {})
            code.write_bytes(b'changed code')
            with self.assertRaisesRegex(ValueError, 'Frozen campaign implementation changed'):
                callback({}, {})

    def test_artifact_hash_and_relative_paths_cannot_be_substituted(self):
        descriptor = self.store.write('data', b'original', kind='JSON')
        (self.directory / 'data').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            w._read_descriptor(self.run, descriptor, 100)
        for path in ('../elsewhere', '/tmp/elsewhere'):
            with self.assertRaises(ValueError): w._file(self.run, path)
        outside = self.run / 'elsewhere'; outside.write_text('x')
        (self.directory / 'link').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escaped'):
            w._file(self.run, 'attempt/link')

    def test_update_enter_precedes_work_confirm_follows_and_failure_not_refunded(self):
        guard = FakeGuard(self.events)
        trainer = SimpleNamespace(updates=0, spec=SimpleNamespace(device='cpu'), torch=object())
        trainer._batch = lambda indices: self.events.append(('batch',))
        def update():
            self.events.append(('work',)); trainer.updates += 1; trainer._batch([0]); return {'updates': trainer.updates}
        trainer.update = update
        w.bind_trainer_updates(trainer, self.ledger, self.lease, guard)
        trainer.update()
        self.assertLess(self.events.index(('enter',)), self.events.index(('work',)))
        self.assertLess(self.events.index(('work',)), self.events.index(('confirm', 7)))
        self.assertEqual(guard.logical_updates, 1)
        failing = SimpleNamespace(updates=0, spec=SimpleNamespace(device='cpu'), torch=object(), _batch=lambda ids: None,
                                  update=Mock(side_effect=RuntimeError('synthetic update failure')))
        self.events.clear(); w.bind_trainer_updates(failing, self.ledger, self.lease, guard)
        with self.assertRaises(RuntimeError): failing.update()
        self.assertIn(('enter',), self.events); self.assertNotIn(('confirm', 7), self.events)

    def test_initial_and_terminal_cadence_checkpoint_not_duplicated(self):
        from hydra_directflow.training import EarlyStopping
        guard = FakeGuard(self.events)
        arrays = {'feature_schema': np.array('FULL_HRRR_H2_W4_S2_V1')}
        bundle = dict(arrays=arrays, source_identity='b' * 64, cohort_admission_digest='c' * 64,
            gradient_case_ids=('synthetic',), manifest={'operation': 'fit'}, train_indices=(0,), inner_indices=(1,),
            role_time_bounds={'inner_stop': {'start_utc': '2019-01-10T00:00:00Z'}})
        job = dict(architecture='LSTM', device='cpu', history_hours=168, seed=17, logical_update_limit=100,
                   data_job={}, checkpoint_interval_epochs=10)
        policy = {'training': {**w.FIXED_TRAINING, 'max_epochs': 100}}
        class Trainer:
            def __init__(inner, arrays, **kwargs):
                inner.spec = kwargs['spec']; inner.stopping = None; inner.updates = 0; inner.epoch_records = []
                inner.torch = object(); inner._batch = lambda ids: None
                inner.update = lambda: setattr(inner, 'updates', inner.updates + 1)
            def fit_epochs(inner, stopping, *, after_epoch):
                inner.stopping = stopping
                while not stopping.finished:
                    inner.update(); improved = stopping.observe(1.)
                    inner.epoch_records.append({'epoch': stopping.completed_epochs, 'updates': inner.updates, 'improved': improved})
                    after_epoch(inner)
        def cp_receipt(trainer, descriptor, ledger, lease):
            return dict(logical_updates=trainer.updates, completed_epochs=trainer.stopping.completed_epochs if trainer.stopping else 0)
        store = SimpleNamespace(run=self.run, directory=self.directory, json=Mock(), checkpoint=Mock(return_value={'sha256': 'd' * 64}))
        with patch('hydra_directflow.campaign_data.load_fit_bundle', return_value=bundle), \
             patch('hydra_directflow.training.DevelopmentTrainer', Trainer), \
             patch.object(w, '_checkpoint_receipt', side_effect=cp_receipt), patch.object(w, 'artifact_validator', return_value=Mock()):
            checkpoint, source = w._train(self.ledger, self.lease, job, policy, self.run, store, guard, None)
        self.assertEqual([receipt['completed_epochs'] for receipt in self.ledger.receipts], [0, 10])
        self.assertEqual(checkpoint['logical_updates'], 10)
        self.assertEqual(store.checkpoint.call_count, 2)
        self.assertEqual(source['source_identity'], 'b' * 64)
        self.assertEqual(guard.stage, 'TRAINER_RELEASED')

    def test_every_prediction_is_saved_before_label_reader_is_called(self):
        from hydra_directflow.training import FULL_FEATURE_SCHEMA
        n = 64; origin = 1548028800 + np.arange(n, dtype=np.int64) * 3600
        future = origin[:, None] + np.arange(1, 19) * 3600
        ids = np.array(['03161000:' + w.datetime.fromtimestamp(int(t), w.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') for t in origin])
        outer = {'arrays': {'case_id': ids, 'origin_time': origin, 'forecast_time': future, 'site_id': np.zeros(n, np.int64)},
                 'manifest': {'operation': 'inputs'}, 'source_identity': 'c' * 64,
                 'cohort_admission_digest': 'd' * 64, 'case_ids_sha256': digest(ids.tolist())}
        job = dict(job_id='test-job', architecture='LSTM', device='cpu', input_arm='full', history_hours=168, gauge_id='03161000',
            fold='cv_1', seed=17, data_job={'actual_common_cases': {'outer_scoring': n}})
        checkpoint = dict(logical_updates=10, artifacts={'checkpoint': {'path': 'attempt/checkpoint.pt'}},
            hashes={'checkpoint': 'e' * 64}, training_source_identity='f' * 64,
            training_cohort_admission_digest='a' * 64, selected_best_epoch=1, feature_schema=FULL_FEATURE_SCHEMA)
        guard = FakeGuard(self.events)
        class Predictor:
            def predict(inner, batch, **kwargs):
                self.events.append(('predict', len(batch['case_id'])))
                self.assertNotIn('target', batch); self.assertNotIn('target_mask', batch)
                return {**batch, 'prediction_q': np.ones((len(batch['case_id']), 18), np.float64), 'receipt': {'target_values_read': False}}
        def labels(*args, **kwargs):
            self.events.append(('labels',))
            self.assertTrue((self.directory / 'predictions.npz').exists())
            self.assertEqual(sum(row[1] for row in self.events if row[0] == 'predict'), n)
            return dict(arrays={'case_id': ids, 'target_time': future, 'target': np.tile(np.arange(1, n+1)[:, None], (1, 18)).astype(float),
                               'target_mask': np.ones((n, 18), bool)}, source_identity='9' * 64, manifest={'operation': 'labels'})
        with patch('hydra_directflow.campaign_data.load_outer_inputs', return_value=outer), \
             patch('hydra_directflow.campaign_data.load_outer_labels', side_effect=labels), \
             patch('hydra_directflow.inference.BestCheckpointPredictor.load', return_value=Predictor()):
            result = w._evaluate(self.ledger, self.lease, job, {'plan_sha256': 'a' * 64}, self.run,
                                 self.store, guard, checkpoint, {'gradient_case_ids': ('source',)})
        self.assertEqual(result['status'], 'VERIFIED_TRAIN_EVAL'); self.assertEqual(result['scored_cases'], n)
        self.assertEqual(result['score_status'], 'PRIMARY_UNAVAILABLE')
        self.assertEqual(set(result['hashes']), {'checkpoint', 'predictions', 'metrics'})
        self.assertTrue(all(value[2] == 'VERIFIED' for value in self.ledger.reserved.values()))

    def test_input_prediction_failure_never_opens_labels(self):
        outer = {'arrays': {}, 'manifest': {'operation': 'inputs'}, 'source_identity': 'a' * 64,
                 'cohort_admission_digest': 'b' * 64}
        job = {'data_job': {'actual_common_cases': {'outer_scoring': 64}}, 'device': 'cpu'}
        checkpoint = {'artifacts': {'checkpoint': {'path': 'attempt/last.pt'}}, 'hashes': {'checkpoint': 'c' * 64},
                      'training_source_identity': 'd' * 64, 'feature_schema': 'full', 'training_cohort_admission_digest': 'e' * 64}
        job['gauge_id'] = '03161000'
        with patch('hydra_directflow.campaign_data.load_outer_inputs', return_value=outer), \
             patch('hydra_directflow.inference.BestCheckpointPredictor.load', side_effect=ValueError('bad checkpoint')), \
             patch('hydra_directflow.campaign_data.load_outer_labels') as labels:
            with self.assertRaisesRegex(ValueError, 'bad checkpoint'):
                w._evaluate(self.ledger, self.lease, job, {}, self.run, self.store, FakeGuard(), checkpoint, {'gradient_case_ids': ()})
            labels.assert_not_called()

    def _orchestration(self, *, resume=False, failure=None):
        root = self.run; run = root / '.local/run'; ledger_path = root / '.local/ledger'
        plan = {'plan_sha256': 'a' * 64, 'run_directory': str(run), 'ledger_directory': str(ledger_path)}
        job = {'job_id': 'test-job', 'device': 'cpu'}
        lease = SimpleNamespace(attempt_id=2 if resume else 1, job_id='test-job')
        ledger = FakeLedger(self.events)
        @contextmanager
        def writer(): yield ledger
        ledger.writer = writer
        ledger.initialize = lambda value: setattr(ledger, 'plan', value)
        ledger.snapshot = lambda: {'bookkeeping_bytes_reserved': 2*w.GIB, 'campaign_deadline': time.time()+100}
        ledger.next_job = lambda: {'job': job, 'deadline': time.time()+100}
        ledger.latest_checkpoint = Mock(return_value={'checkpoint': 'latest'})
        ledger.claim_next = Mock(return_value=lease); ledger.resume = Mock(return_value=lease)
        ledger.complete = Mock(); ledger.fail = Mock(); ledger.interrupt = Mock()
        guard = FakeGuard(self.events)
        train = Mock(return_value=({'checkpoint': 'terminal'}, {'source': 'training'}), side_effect=failure)
        evaluate = Mock(return_value={'status': 'VERIFIED_TRAIN_EVAL', 'logical_updates': 10})
        with patch.object(w, 'ROOT', root), patch.object(w, 'validate_execution_plan', return_value=plan), \
             patch.object(w, 'CampaignLedger', return_value=ledger), patch.object(w, '_offline'), \
             patch.object(w, '_torch_runtime', return_value=object()), patch.object(w, 'ResourceGuard', return_value=guard), \
             patch.object(w, 'artifact_validator', return_value=Mock()), patch.object(w, '_train', train), \
             patch.object(w, '_evaluate', evaluate), patch.object(w.signal, 'signal', return_value=0), \
             patch.object(w.signal, 'setitimer'):
            if failure is None:
                result = w.run_job(plan, root/'corpus', run, ledger_path, resume=resume)
                self.assertEqual(result['status'], 'VERIFIED_TRAIN_EVAL')
            else:
                with self.assertRaises(type(failure)):
                    w.run_job(plan, root/'corpus', run, ledger_path, resume=resume)
        return ledger, train, evaluate, guard

    def test_one_job_completes_only_after_train_evaluate_and_heartbeat_settlement(self):
        ledger, train, evaluate, guard = self._orchestration()
        ledger.claim_next.assert_called_once(); ledger.resume.assert_not_called()
        train.assert_called_once(); evaluate.assert_called_once(); ledger.complete.assert_called_once()
        ledger.fail.assert_not_called(); ledger.interrupt.assert_not_called()
        self.assertIn(('heartbeat-settle',), self.events)
        self.assertEqual(guard.stage, 'ARTIFACTS_COMMITTED_AWAITING_LEDGER_VERIFICATION')

    def test_explicit_resume_uses_latest_checkpoint_and_never_claims_new_job(self):
        ledger, train, evaluate, guard = self._orchestration(resume=True)
        ledger.latest_checkpoint.assert_called_once_with('test-job')
        ledger.claim_next.assert_not_called(); ledger.resume.assert_called_once()
        self.assertEqual(train.call_args.args[-1], {'checkpoint': 'latest'})

    def test_integrity_failure_blocks_evaluation_completion_and_automatic_retry(self):
        ledger, train, evaluate, guard = self._orchestration(failure=ValueError('synthetic integrity failure'))
        ledger.fail.assert_called_once(); ledger.complete.assert_not_called(); evaluate.assert_not_called()
        ledger.resume.assert_not_called(); self.assertEqual(ledger.claim_next.call_count, 1)

    def test_interruption_is_preserved_for_explicit_same_job_resume(self):
        ledger, train, evaluate, guard = self._orchestration(failure=w.CampaignInterrupted('synthetic interruption'))
        ledger.interrupt.assert_called_once(); ledger.fail.assert_not_called(); ledger.complete.assert_not_called()
        evaluate.assert_not_called(); ledger.resume.assert_not_called()

    def test_authorization_policy_and_logical_physical_limits_gate_before_any_source(self):
        from hydra_directflow.campaign_ledger import PHASES, GAUGES, ARCHITECTURES, ARMS, LIMITS
        jobs = []
        for fold, seed in PHASES:
            for gauge in GAUGES:
                for architecture in ARCHITECTURES:
                    for arm, hours in (ARMS if fold == 'final_development' else ARMS[:2]):
                        data = dict(job_id=f'{fold}-{gauge}-{architecture}-{arm}-{hours}-{seed}', fold=fold, seed=seed,
                            gauge_id=gauge, architecture=architecture, input_arm=arm, history_hours=hours,
                            device='cpu' if architecture in ('LSTM', 'GRU_Transformer') else 'mps', batch_size=64, precision='FP32',
                            role_case_ids_sha256={role:'b'*64 for role in ('gradient_fit','inner_stopping','outer_scoring')},
                            actual_common_cases={role:64 for role in ('gradient_fit','inner_stopping','outer_scoring')})
                        jobs.append({**data, 'data_job':data, 'order':len(jobs), 'logical_update_limit':100,
                            'max_updates':130, 'gradient_batches_per_epoch':1, 'checkpoint_interval_epochs':10, 'max_resumes':3})
        plan = dict(protocol_sha256='a'*64, source_plan_sha256='b'*64, snapshot_sha256='c'*64,
            implementation_sha256={name:'d'*64 for name in w.REQUIRED_CODE}, limits=LIMITS, jobs=jobs,
            protocol_frozen=True, scientific_training_authorized=True, protected_assessment_access=False, paid_cloud_jobs_started=0,
            training={**w.FIXED_TRAINING, 'max_epochs':100})
        def repin(value):
            value.pop('plan_sha256', None); value['plan_sha256'] = digest(value); return value
        with patch.object(w, '_validate_implementation'):
            self.assertEqual(w.validate_execution_plan(repin(plan))['training']['max_epochs'], 100)
            denied = {**plan, 'scientific_training_authorized':False}
            with self.assertRaisesRegex(ValueError, 'authorization'):
                w.validate_execution_plan(repin(denied))
            expanded = {**plan, 'training':{**plan['training'], 'max_epochs':200}}
            with self.assertRaisesRegex(ValueError, 'Logical training'):
                w.validate_execution_plan(repin(expanded))
            wrong = {**plan, 'training':{**plan['training'], 'learning_rate':.001}}
            with self.assertRaisesRegex(ValueError, 'training policy'):
                w.validate_execution_plan(repin(wrong))

    def test_offline_environment_required_before_runtime_import(self):
        with patch.dict(w.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'before importing Torch'):
                w._offline()

    def test_relocated_ledger_rejects_before_offline_or_model_setup(self):
        root = self.run; run = root / '.local/run'; ledger = root / '.local/ledger'
        plan = {'run_directory': str(run), 'ledger_directory': str(root / '.local/original-ledger')}
        with patch.object(w, 'ROOT', root), patch.object(w, 'validate_execution_plan', return_value=plan), \
             patch.object(w, '_offline') as offline, patch.object(w, '_torch_runtime') as runtime:
            with self.assertRaisesRegex(ValueError, 'reset'):
                w.run_job(plan, root / 'corpus', run, ledger)
            offline.assert_not_called(); runtime.assert_not_called()

    def test_resource_heartbeat_reserves_once_and_keeps_small_bounded_slot(self):
        guard = w.ResourceGuard(self.ledger, self.lease, self.run, self.directory, object(), {'job_id': 'x', 'device': 'cpu'})
        observation = dict(worker_rss_bytes=100, parent_rss_bytes=200, mps_driver_bytes=0, combined_rss_mps_bytes=300)
        with patch.object(w, '_runtime_memory', return_value=observation), patch.object(w.shutil, 'disk_usage', return_value=SimpleNamespace(free=9*w.GIB)):
            guard.check(force=True); guard.check(force=True); guard.settle()
        self.assertEqual(len(self.ledger.reserved), 1)
        self.assertEqual(self.ledger.reserved[1][0], w.HEARTBEAT_RESERVE)
        self.assertEqual(json.loads((self.directory / 'heartbeat.json').read_text())['combined_rss_mps_bytes'], 300)


if __name__ == '__main__': unittest.main()
