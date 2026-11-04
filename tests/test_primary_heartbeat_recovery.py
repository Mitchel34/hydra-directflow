"""No-model qualification of the explicit, scoped heartbeat recovery overlay."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / 'scripts/recover_primary_campaign_heartbeat.py'
SPEC = importlib.util.spec_from_file_location('primary_heartbeat_recovery_tested', PATH)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=recovery.ROOT / '.local')
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / 'run'
        self.run.mkdir()
        self.plan = dict(plan_sha256='a' * 64, protocol_sha256='b' * 64,
                         run_directory=str(self.run), ledger_directory=str(self.run / 'ledger'),
                         jobs=[{'job_id': f'job-{i}'} for i in range(3)],
                         limits={'min_free_bytes': 1024, 'max_output_bytes': 8 * 1024**3})
        self.code = {'synthetic-code.py': 'c' * 64}
        self.qualification = {'path': str(recovery.ROOT / 'reports/synthetic-qualification.json'),
                              'sha256': 'd' * 64}
        self.state = {'synthetic_state': True}
        self.auth = dict(source='DIRECT_USER_MESSAGE', scope=recovery.SCOPE,
                         quote='SYNTHETIC_TEST_ONLY_NOT_EXECUTION_APPROVAL',
                         recorded_at_utc='2026-11-04T00:00:00Z')
        self.amendment = dict(schema='hydra-primary-heartbeat-recovery-amendment/v1',
                              scope=recovery.SCOPE, authorization=self.auth,
                              original_plan_path=str(self.root / 'plan.json'),
                              original_plan_sha256=self.plan['plan_sha256'],
                              original_protocol_sha256=self.plan['protocol_sha256'],
                              implementation_sha256=self.code, qualification=self.qualification,
                              original_state=self.state, scientific_method_changes=False,
                              resource_limit_changes=False, paid_cloud_jobs=0,
                              protected_assessment_access=False)
        self.sign(self.amendment)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def sign(value):
        value['amendment_sha256'] = recovery.digest({k: v for k, v in value.items() if k != 'amendment_sha256'})
        return value

    def heartbeat(self, job='job-0', attempt='attempt-000001'):
        path = self.run / 'jobs' / job / attempt / 'heartbeat.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{"synthetic":true}')
        return path

    @contextmanager
    def mocked_validation_dependencies(self):
        with patch.object(recovery, 'implementation', return_value=self.code), \
                patch.object(recovery, 'qualified_receipt', return_value=self.qualification), \
                patch.object(recovery.controller, 'authenticated_plan', return_value=self.plan), \
                patch.object(recovery, 'recovery_state', return_value=self.state):
            yield

    def test_only_exact_planned_heartbeat_paths_use_snapshot_reader_including_later_jobs(self):
        first = self.heartbeat()
        later = self.heartbeat('job-2', 'attempt-000099')
        with patch.object(recovery.controller, 'read', return_value=b'immutable') as strict, \
                patch.object(recovery, 'read_snapshot', return_value=b'snapshot') as snapshot:
            with recovery.heartbeat_reads(self.plan):
                for path in (first, later):
                    self.assertEqual(recovery.controller.read(path, 65536), b'snapshot')
            strict.assert_not_called()
            self.assertEqual(snapshot.call_count, 2)
            self.assertIs(recovery.controller.read, strict)

    def test_immutable_and_outside_paths_always_delegate_to_strict_reader(self):
        paths = [self.run / 'PLAN.json', self.run / 'jobs/job-0/attempt-000001/checkpoint.pt',
                 self.root / 'heartbeat.json', self.root / 'source/array.npz']
        with patch.object(recovery.controller, 'read', return_value=b'immutable') as strict, \
                patch.object(recovery, 'read_snapshot', side_effect=AssertionError('Snapshot scope widened')):
            with recovery.heartbeat_reads(self.plan):
                for path in paths:
                    self.assertEqual(recovery.controller.read(path, 1234), b'immutable')
            self.assertEqual(strict.call_count, len(paths))
            for call, path in zip(strict.call_args_list, paths):
                self.assertEqual(call.args, (path, 1234))

    def test_unplanned_job_bad_attempt_layout_and_byte_bound_are_rejected(self):
        cases = [(self.heartbeat('unplanned'), 65536),
                 (self.heartbeat(attempt='attempt-000000'), 65536),
                 (self.heartbeat(attempt='attempt-1'), 65536),
                 (self.heartbeat(attempt='attempt-000001-extra'), 65536),
                 (self.heartbeat(), 65535),
                 (self.run / 'heartbeat.json', 65536)]
        with patch.object(recovery, 'read_snapshot', side_effect=AssertionError('Invalid path reached reader')):
            with recovery.heartbeat_reads(self.plan):
                for path, maximum in cases:
                    with self.subTest(path=str(path), maximum=maximum):
                        with self.assertRaisesRegex(ValueError, 'Unexpected heartbeat'):
                            recovery.controller.read(path, maximum)

    def test_symlink_and_parent_path_alias_are_rejected(self):
        path = self.heartbeat()
        target = path.with_name('target.json')
        target.write_bytes(b'{}')
        path.unlink()
        path.symlink_to(target)
        alias = self.run / 'jobs/job-0/../job-0/attempt-000001/heartbeat.json'
        with recovery.heartbeat_reads(self.plan):
            with self.assertRaisesRegex(ValueError, 'alias or symlink'):
                recovery.controller.read(path, 65536)
            with self.assertRaisesRegex(ValueError, 'Unexpected heartbeat'):
                recovery.controller.read(alias, 65536)

    def test_malformed_json_is_not_hidden_or_replaced_by_previous_heartbeat(self):
        path = self.heartbeat()
        path.write_bytes(b'not json')
        with recovery.heartbeat_reads(self.plan):
            raw = recovery.controller.read(path, 65536)
            self.assertEqual(raw, b'not json')
            with self.assertRaises(json.JSONDecodeError):
                json.loads(raw)

    def test_read_patch_is_restored_even_when_supervisor_body_raises(self):
        original = recovery.controller.read
        with self.assertRaisesRegex(RuntimeError, 'synthetic supervisor error'):
            with recovery.heartbeat_reads(self.plan):
                self.assertIsNot(recovery.controller.read, original)
                raise RuntimeError('synthetic supervisor error')
        self.assertIs(recovery.controller.read, original)

    def test_prepare_binds_existing_state_without_starting_or_initializing(self):
        with self.mocked_validation_dependencies(), \
                patch.object(recovery.controller, 'initialize', side_effect=AssertionError('Initialization forbidden')), \
                patch.object(recovery.controller, 'supervise', side_effect=AssertionError('Training forbidden')):
            result = recovery.prepare(self.root / 'plan.json', self.plan['plan_sha256'],
                                      self.qualification['path'], self.auth)
        self.assertEqual(result['original_state'], self.state)
        self.assertEqual(result['original_plan_sha256'], self.plan['plan_sha256'])
        self.assertEqual(result['implementation_sha256'], self.code)
        self.assertEqual(result['amendment_sha256'], recovery.digest({k: v for k, v in result.items() if k != 'amendment_sha256'}))

    def test_prepare_requires_explicit_user_recovery_record(self):
        auth = {**self.auth, 'quote': ''}
        with self.mocked_validation_dependencies(), \
                patch.object(recovery, 'recovery_state', side_effect=AssertionError('State read after invalid approval')):
            with self.assertRaisesRegex(ValueError, 'Explicit user recovery instruction'):
                recovery.prepare(self.root / 'plan.json', self.plan['plan_sha256'], self.qualification['path'], auth)

    def test_valid_amendment_returns_original_authenticated_plan(self):
        with self.mocked_validation_dependencies():
            result = recovery.validate_amendment(self.amendment, self.amendment['amendment_sha256'])
        self.assertIs(result, self.plan)

    def test_tampered_amendment_or_wrong_requested_hash_refuses_before_dependencies(self):
        altered = deepcopy(self.amendment)
        altered['resource_limit_changes'] = True
        with patch.object(recovery, 'implementation', side_effect=AssertionError('Tamper reached dependencies')):
            for amendment, expected in [(altered, self.amendment['amendment_sha256']), (self.amendment, '0' * 64)]:
                with self.subTest(expected=expected):
                    with self.assertRaisesRegex(ValueError, 'Exact recovery amendment SHA'):
                        recovery.validate_amendment(amendment, expected)

    def test_even_resigned_scope_expansions_are_rejected(self):
        for key, value in [('scientific_method_changes', True), ('resource_limit_changes', True),
                           ('paid_cloud_jobs', 1), ('protected_assessment_access', True), ('scope', 'NEW_CAMPAIGN')]:
            changed = deepcopy(self.amendment)
            changed[key] = value
            self.sign(changed)
            with self.subTest(key=key), self.mocked_validation_dependencies():
                with self.assertRaisesRegex(ValueError, 'bounded operational amendment'):
                    recovery.validate_amendment(changed, changed['amendment_sha256'])

    def test_changed_implementation_qualification_protocol_or_recovery_state_is_rejected(self):
        cases = [('implementation', {'changed': 'x'}, 'implementation changed'),
                 ('qualified_receipt', {'path': 'changed'}, 'qualification changed'),
                 ('recovery_state', {'synthetic_state': False}, 'recovery state changed')]
        for function, value, message in cases:
            with self.subTest(function=function), self.mocked_validation_dependencies():
                with patch.object(recovery, function, return_value=value):
                    with self.assertRaisesRegex(ValueError, message):
                        recovery.validate_amendment(self.amendment, self.amendment['amendment_sha256'])
        with self.mocked_validation_dependencies(), \
                patch.object(recovery.controller, 'authenticated_plan', return_value={**self.plan, 'protocol_sha256': 'changed'}):
            with self.assertRaisesRegex(ValueError, 'Frozen protocol identity'):
                recovery.validate_amendment(self.amendment, self.amendment['amendment_sha256'])

    def test_resigned_amendment_without_explicit_recovery_authorization_is_rejected(self):
        changed = deepcopy(self.amendment)
        changed['authorization']['source'] = 'INFERRED'
        self.sign(changed)
        with self.mocked_validation_dependencies():
            with self.assertRaisesRegex(ValueError, 'Explicit recovery authorization missing'):
                recovery.validate_amendment(changed, changed['amendment_sha256'])

    def make_ledger_fixture(self):
        checkpoint_file = self.run / 'checkpoint.pt'
        checkpoint_file.write_bytes(b'SYNTHETIC OPAQUE CHECKPOINT NOT TORCH')
        checkpoint = dict(receipt_path='checkpoint.json', logical_updates=3800,
                          artifacts={'checkpoint': {'path': checkpoint_file.name,
                                                   'bytes': checkpoint_file.stat().st_size,
                                                   'sha256': recovery.controller.sha(checkpoint_file.read_bytes())}})
        (self.run / 'checkpoint.json').write_text(json.dumps(checkpoint))
        first = dict(position=2, job_id='job-2', state='INTERRUPTED', deadline=2000)
        snapshot = dict(plan_sha256=self.plan['plan_sha256'],
                        attempts=[{'status': 'INTERRUPTED'}],
                        jobs=[{'state': 'VERIFIED_TRAIN_EVAL'}, {'state': 'VERIFIED_TRAIN_EVAL'}, first],
                        campaign_deadline=3000, optimizer_updates_entered=4572,
                        optimizer_updates_confirmed=4572, output_bytes_reserved=1234,
                        campaign_started_at=1)
        completed = []
        for i in range(2):
            receipt = {'job_id': f'job-{i}', 'synthetic': True}
            completed.append((i, recovery.digest(receipt), json.dumps(receipt)))

        @contextmanager
        def reader():
            yield SimpleNamespace(execute=lambda *args: SimpleNamespace(fetchall=lambda: completed))

        ledger = SimpleNamespace(snapshot=lambda: snapshot,
                                 latest_checkpoint=lambda job: checkpoint,
                                 _reader=reader)
        return snapshot, checkpoint, completed, ledger

    def test_recovery_state_authenticates_opaque_checkpoint_and_completed_prefix(self):
        snapshot, checkpoint, completed, ledger = self.make_ledger_fixture()
        with patch.object(recovery.controller, 'verify_release'), \
                patch.object(recovery, 'CampaignLedger', return_value=ledger), \
                patch.object(recovery.time, 'time', return_value=1000):
            result = recovery.recovery_state(self.plan)
        self.assertEqual(result['checkpoint_logical_updates'], 3800)
        self.assertEqual(result['latest_checkpoint_receipt_sha256'], recovery.digest(checkpoint))
        self.assertEqual(result['completed_receipt_sha256'], {f'job-{i}': row[1] for i, row in enumerate(completed)})
        self.assertEqual(result['optimizer_updates_confirmed'], 4572)
        self.assertEqual(result['campaign_deadline'], 3000)

    def test_recovery_state_rejects_active_wrong_state_or_expired_job(self):
        for change, message in [('active', 'active attempt'), ('pending', 'must be interrupted'), ('expired', 'deadline expired')]:
            snapshot, checkpoint, completed, ledger = self.make_ledger_fixture()
            if change == 'active':
                snapshot['attempts'][0]['status'] = 'ACTIVE'
            elif change == 'pending':
                snapshot['jobs'][2]['state'] = 'PENDING'
            else:
                snapshot['jobs'][2]['deadline'] = 999
            with self.subTest(change=change), patch.object(recovery.controller, 'verify_release'), \
                    patch.object(recovery, 'CampaignLedger', return_value=ledger), \
                    patch.object(recovery.time, 'time', return_value=1000):
                with self.assertRaisesRegex(ValueError, message):
                    recovery.recovery_state(self.plan)

    def test_recovery_state_rejects_changed_checkpoint_bytes_or_completed_receipt(self):
        for change, message in [('checkpoint', 'artifact bytes changed'), ('completed', 'receipt identity differs')]:
            snapshot, checkpoint, completed, ledger = self.make_ledger_fixture()
            if change == 'checkpoint':
                (self.run / 'checkpoint.pt').write_bytes(b'altered synthetic bytes')
            else:
                completed[0] = (0, '0' * 64, completed[0][2])
            with self.subTest(change=change), patch.object(recovery.controller, 'verify_release'), \
                    patch.object(recovery, 'CampaignLedger', return_value=ledger), \
                    patch.object(recovery.time, 'time', return_value=1000):
                with self.assertRaisesRegex(ValueError, message):
                    recovery.recovery_state(self.plan)

    def test_execute_calls_only_original_supervisor_with_explicit_resume_and_restores_reader(self):
        amendment_path = self.root / 'amendment.json'
        amendment_path.write_text(json.dumps(self.amendment))
        original_read = recovery.controller.read
        result = {'status': 'SYNTHETIC_RETURN_NO_FITS'}

        def supervisor(plan_path, plan, *, resume):
            self.assertEqual(plan_path, Path(self.amendment['original_plan_path']))
            self.assertIs(plan, self.plan)
            self.assertTrue(resume)
            self.assertIsNot(recovery.controller.read, original_read)
            return result

        with patch.object(recovery, 'validate_amendment', return_value=self.plan), \
                patch.object(recovery.controller.shutil, 'disk_usage', return_value=SimpleNamespace(free=9999999)), \
                patch.object(recovery.controller, 'folder_bytes', return_value=0), \
                patch.object(recovery.controller, 'supervise', side_effect=supervisor) as supervise, \
                patch.object(recovery.controller, 'initialize', side_effect=AssertionError('Cannot create new campaign')):
            self.assertIs(recovery.execute(amendment_path, self.amendment['amendment_sha256']), result)
        supervise.assert_called_once()
        self.assertIs(recovery.controller.read, original_read)
        receipts = list((self.run / 'operational-recovery').glob('*.json'))
        self.assertEqual(len(receipts), 1)
        receipt = json.loads(receipts[0].read_text())
        self.assertEqual(receipt['amendment'], self.amendment)
        self.assertTrue(receipt['resume_requested'])

    def test_execute_preserves_failure_receipt_and_restores_reader_after_supervisor_error(self):
        amendment_path = self.root / 'amendment.json'
        amendment_path.write_text(json.dumps(self.amendment))
        original_read = recovery.controller.read
        with patch.object(recovery, 'validate_amendment', return_value=self.plan), \
                patch.object(recovery.controller.shutil, 'disk_usage', return_value=SimpleNamespace(free=9999999)), \
                patch.object(recovery.controller, 'folder_bytes', return_value=0), \
                patch.object(recovery.controller, 'supervise', side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                recovery.execute(amendment_path, self.amendment['amendment_sha256'])
        self.assertIs(recovery.controller.read, original_read)
        self.assertEqual(len(list((self.run / 'operational-recovery').glob('*.json'))), 1)

    def test_execute_changed_state_and_expired_deadline_refuse_before_start_receipt(self):
        amendment_path = self.root / 'amendment.json'
        amendment_path.write_text(json.dumps(self.amendment))
        for issue in ('state', 'deadline'):
            state_result = (patch.object(recovery, 'recovery_state', return_value={'changed': True})
                            if issue == 'state' else
                            patch.object(recovery, 'recovery_state', side_effect=ValueError('Original recovery deadline expired')))
            with self.subTest(issue=issue), self.mocked_validation_dependencies(), state_result, \
                    patch.object(recovery.controller, 'atomic', side_effect=AssertionError('Receipt must not be written')), \
                    patch.object(recovery.controller, 'supervise', side_effect=AssertionError('Must not resume')):
                with self.assertRaisesRegex(ValueError, 'state changed|deadline expired'):
                    recovery.execute(amendment_path, self.amendment['amendment_sha256'])
        self.assertFalse((self.run / 'operational-recovery').exists())

    def test_original_supervisor_still_rejects_foreign_and_malformed_heartbeats(self):
        for index, issue in enumerate(('pid', 'plan_sha256', 'job_id', 'malformed')):
            with self.subTest(issue=issue):
                run = self.root / f'supervised-{index}'
                heartbeat_path = run / 'jobs/job-0/attempt-000001/heartbeat.json'
                heartbeat_path.parent.mkdir(parents=True)
                plan = deepcopy(self.plan)
                plan.update(run_directory=str(run), ledger_directory=str(run / 'ledger'),
                            per_worker_log_limit_bytes=1024**2)
                plan['limits'].update(max_job_seconds=60, max_combined_rss_bytes=1024**3)
                pulse = dict(pid=123456, plan_sha256=plan['plan_sha256'], job_id='job-0', mps_driver_bytes=0)
                if issue != 'malformed':
                    pulse[issue] = 'FOREIGN'
                heartbeat_path.write_text('not-json' if issue == 'malformed' else json.dumps(pulse))
                events = []

                class Ledger:
                    @contextmanager
                    def writer(self):
                        yield self

                    def initialize(self, ignored):
                        events.append('existing-ledger-check')

                    def next_job(self):
                        return dict(state='INTERRUPTED', deadline=None,
                                    job={'job_id': 'job-0', 'order': 0})

                    def snapshot(self):
                        return dict(attempts=[{'status': 'ACTIVE', 'id': 1}],
                                    campaign_deadline=None, jobs=[{'state': 'ACTIVE'}])

                class Child:
                    pid = 123456
                    returncode = None

                    def poll(self):
                        return self.returncode

                    def wait(self, timeout):
                        return self.returncode

                    def terminate(self):
                        events.append('terminate')
                        self.returncode = -15

                    def kill(self):
                        events.append('kill')
                        self.returncode = -9

                child = Child()
                process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1024))
                original_read = recovery.controller.read
                with patch.object(recovery.controller, 'verify_release'), \
                        patch.object(recovery.controller, 'CampaignLedger', return_value=Ledger()), \
                        patch.object(recovery.controller, 'publish_ready_seeds'), \
                        patch.object(recovery.controller, 'validate_execution_plan'), \
                        patch.object(recovery.controller.subprocess, 'Popen', return_value=child) as launch, \
                        patch('psutil.Process', return_value=process), \
                        patch.object(recovery.controller, 'progress') as progress:
                    with recovery.heartbeat_reads(plan):
                        expected_error = json.JSONDecodeError if issue == 'malformed' else ValueError
                        with self.assertRaises(expected_error):
                            recovery.controller.supervise(self.root / 'plan.json', plan, resume=True)
                self.assertIs(recovery.controller.read, original_read)
                self.assertEqual(events.count('terminate'), 1)
                launch.assert_called_once()
                self.assertIn('--resume', launch.call_args.args[0])
                self.assertEqual(progress.call_args.kwargs['status'], 'STOPPED_CONTROLLER_GUARD')


if __name__ == '__main__':
    unittest.main()
