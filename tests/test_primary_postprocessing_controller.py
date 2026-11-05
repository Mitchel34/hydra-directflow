"""Synthetic metadata/guards only; no scientific values, fits or real ledger writes."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1]/'scripts/run_primary_postprocessing.py'
SPEC = importlib.util.spec_from_file_location('primary_postprocessing_tested', PATH)
post = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(post)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=post.ROOT/'.local')
        self.root = Path(self.temp.name).resolve()
        self.campaign = self.root/'campaign'
        self.campaign.mkdir()
        self.original = dict(plan_sha256='a'*64, protocol_sha256='b'*64,
                             corpus_root=str(self.root/'never-read-source'),
                             run_directory=str(self.campaign), ledger_directory=str(self.campaign/'ledger'),
                             limits={'max_output_bytes': 8*post.GIB, 'min_free_bytes': 8*post.GIB})
        self.original_path = self.root/'original.json'
        self.original_path.write_text(json.dumps(self.original))
        self.state = dict(complete=True, jobs=[{'state': 'VERIFIED_TRAIN_EVAL'}]*180,
                          output_bytes_reserved=2*post.GIB, campaign_deadline=2000000000.)
        self.guards = []

    def tearDown(self):
        for guard in self.guards:
            guard.close()
        self.temp.cleanup()

    @contextmanager
    def source_mocks(self):
        ledger = SimpleNamespace(snapshot=lambda: deepcopy(self.state))
        with patch.object(post.driver, 'authenticated_plan', return_value=deepcopy(self.original)) as authenticate, \
                patch.object(post, 'CampaignLedger', return_value=ledger), \
                patch.object(post, 'CODE', ('scripts/run_primary_postprocessing.py',)), \
                patch.object(post.driver, 'initialize', side_effect=AssertionError('Fresh training forbidden')):
            yield authenticate

    def plan(self):
        with self.source_mocks():
            return post.prepare(self.original_path)

    def put_plan(self, plan):
        plan['plan_sha256'] = post.digest({k: v for k, v in plan.items() if k != 'plan_sha256'})
        path = self.root/'post-plan.json'
        path.write_text(json.dumps(plan))
        return path

    @contextmanager
    def resource_mocks(self, *, now=1000., rss=1024, child_rss=0, free=20*post.GIB, output_bytes=0):
        child = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=child_rss))
        process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=rss),
                                  children=lambda recursive: [child] if child_rss else [])
        with patch.object(post.time, 'time', return_value=now), \
                patch.object(post.psutil, 'Process', return_value=process), \
                patch.object(post.shutil, 'disk_usage', return_value=SimpleNamespace(free=free)), \
                patch.object(post.driver, 'folder_bytes', return_value=output_bytes):
            yield

    def guard(self, plan=None):
        plan = plan or self.plan()
        run = Path(plan['run_directory']);run.mkdir(exist_ok=True)
        (run/'PLAN.json').write_text(json.dumps(plan))
        (run/'RESOURCE.json').write_text(json.dumps(dict(plan_sha256=plan['plan_sha256'],
            started_epoch=1000., started_at_utc='SYNTHETIC', read_bytes_reserved=0, peak_rss_bytes=0,
            original_ledger_output_reservation_token='synthetic-token', original_ledger_output_reservation_bytes=post.GIB)))
        guard = post.Guard(plan);self.guards.append(guard)
        return guard

    def test_prepare_only_binds_complete_original_study_and_fixed_scope(self):
        with self.source_mocks() as auth:
            plan = post.prepare(self.original_path)
        auth.assert_called_once_with(self.original_path, self.original['plan_sha256'])
        self.assertEqual(plan['original_plan_sha256'], self.original['plan_sha256'])
        self.assertEqual(plan['original_campaign_deadline'], self.state['campaign_deadline'])
        self.assertEqual(plan['limits']['max_new_output_bytes'], post.GIB)
        self.assertEqual(plan['limits']['max_rss_bytes'], 3*post.GIB)
        self.assertEqual(plan['uncertainty']['draws'], 2000)
        self.assertEqual(plan['uncertainty']['block_hours'], [168, 336, 672])
        self.assertFalse(plan['protected_assessment_access'])
        self.assertEqual(plan['model_calls'], 0)
        self.assertFalse(Path(plan['run_directory']).exists())

    def test_prepare_refuses_unfinished_campaign_or_insufficient_output_allowance(self):
        for condition, message in [('unfinished', 'completed fits'), ('capacity', 'output allowance')]:
            original = deepcopy(self.state)
            if condition == 'unfinished': self.state['complete'] = False
            else: self.state['output_bytes_reserved'] = 8*post.GIB
            with self.subTest(condition=condition), self.source_mocks():
                with self.assertRaisesRegex(ValueError, message):
                    post.prepare(self.original_path)
            self.state = original

    def test_checked_plan_rejects_wrong_sha_and_changed_code(self):
        plan = self.plan();path = self.put_plan(plan)
        with self.source_mocks():
            with self.assertRaisesRegex(ValueError, 'plan SHA'):
                post.checked_plan(path, '0'*64)
        plan['implementation_sha256'][post.CODE[0]] = '0'*64
        path = self.put_plan(plan)
        with self.source_mocks():
            with self.assertRaisesRegex(ValueError, 'implementation changed'):
                post.checked_plan(path, plan['plan_sha256'])

    def test_resigned_source_scope_limits_and_gate_changes_are_rejected(self):
        variants = [('corpus_root', str(self.root/'wrong-source')), ('campaign_directory', str(self.root/'wrong-run')),
                    ('protocol_sha256', '0'*64), ('protected_assessment_access', True),
                    ('paid_cloud_jobs', 1), ('model_calls', 1)]
        for key, value in variants:
            plan = self.plan();plan[key] = value;path = self.put_plan(plan)
            with self.subTest(key=key), self.source_mocks(), self.assertRaises(ValueError):
                post.checked_plan(path, plan['plan_sha256'])
        plan = self.plan();plan['limits']['max_seconds'] += 1;path = self.put_plan(plan)
        with self.source_mocks(), self.assertRaises(ValueError):
            post.checked_plan(path, plan['plan_sha256'])

    def test_resigned_uncertainty_deadline_and_reservation_changes_are_rejected(self):
        for field in ('uncertainty', 'deadline', 'reservation'):
            plan = deepcopy(self.plan())
            if field == 'uncertainty':plan['uncertainty']['draws'] = 1999
            elif field == 'deadline':plan['original_campaign_deadline'] += 1
            else:plan['original_output_reserved'] -= 1
            path = self.put_plan(plan)
            with self.subTest(field=field), self.source_mocks(), self.assertRaises(ValueError):
                post.checked_plan(path, plan['plan_sha256'])

    def test_checked_resume_requires_exact_once_only_output_reservation(self):
        plan = self.plan();path = self.put_plan(plan)
        with self.source_mocks():
            got, original = post.checked_plan(path, plan['plan_sha256'])
        self.assertEqual(got, plan)
        self.assertEqual(original, self.original)
        Path(plan['run_directory']).mkdir()
        with self.source_mocks(), self.assertRaisesRegex(ValueError, 'reservation accounting'):
            post.checked_plan(path, plan['plan_sha256'])
        self.state['output_bytes_reserved'] += post.GIB
        with self.source_mocks():
            self.assertEqual(post.checked_plan(path, plan['plan_sha256'])[0], plan)
        self.state['output_bytes_reserved'] += post.GIB
        with self.source_mocks(), self.assertRaisesRegex(ValueError, 'reservation accounting'):
            post.checked_plan(path, plan['plan_sha256'])

    def test_start_reserves_exact_gib_before_output_writes(self):
        plan = self.plan();run = Path(plan['run_directory']);events = []
        @contextmanager
        def writer():
            events.append('writer');yield
        def reserve(size, **kwargs):
            self.assertFalse(run.exists())
            self.assertEqual(size, post.GIB)
            self.assertEqual(kwargs, {'kind': 'POSTPROCESSING_V1'})
            events.append('reserved');return 'synthetic-token'
        ledger = SimpleNamespace(writer=writer, initialize=lambda original: self.assertEqual(original, self.original),
                                 snapshot=lambda: deepcopy(self.state), reserve_output=reserve)
        with self.resource_mocks(), patch.object(post, 'CampaignLedger', return_value=ledger), \
                patch.object(post.driver, 'initialize', side_effect=AssertionError('Fresh training forbidden')):
            guard = post.start(plan);self.guards.append(guard)
        self.assertEqual(events, ['writer', 'reserved'])
        self.assertEqual(guard.record['original_ledger_output_reservation_bytes'], post.GIB)
        self.assertEqual(guard.record['original_ledger_output_reservation_token'], 'synthetic-token')

    def test_existing_run_reuses_original_clock_and_reservation_without_ledger_mutation(self):
        guard = self.guard();plan = guard.plan;record = deepcopy(guard.record);guard.close()
        with patch.object(post, 'CampaignLedger', side_effect=AssertionError('Reservation cannot repeat')):
            resumed = post.start(plan);self.guards.append(resumed)
        self.assertEqual(resumed.record, record)
        with self.resource_mocks(now=record['started_epoch']+plan['limits']['max_seconds']):
            with self.assertRaisesRegex(ValueError, 'deadline'):
                resumed.check()

    def test_exclusive_writer_lock_prevents_overlapping_postprocessing(self):
        guard = self.guard()
        with self.assertRaises(BlockingIOError):
            post.Guard(guard.plan)
        guard.close()
        second = post.Guard(guard.plan);self.guards.append(second)

    def test_read_precharge_is_durable_even_if_body_read_fails(self):
        guard = self.guard();source = self.root/'synthetic-input';source.write_bytes(b'bytes')
        with self.resource_mocks(), patch.object(post, '_read', side_effect=RuntimeError('synthetic body failure')):
            with self.assertRaisesRegex(RuntimeError, 'body failure'):
                guard.read(source, 100)
        stored = json.loads(guard.record_path.read_text())
        self.assertEqual(stored['read_bytes_reserved'], 5)

    def test_read_limit_exhaustion_prevents_body_open_and_has_no_refund(self):
        guard = self.guard();guard.record['read_bytes_reserved'] = guard.plan['limits']['max_read_bytes']-2
        source = self.root/'synthetic-input';source.write_bytes(b'bytes')
        with self.resource_mocks(), patch.object(post, '_read', side_effect=AssertionError('Body opened')):
            with self.assertRaisesRegex(ValueError, 'read allowance'):
                guard.read(source, 100)
        self.assertEqual(guard.record['read_bytes_reserved'], guard.plan['limits']['max_read_bytes']-2)

    def test_memory_free_space_output_and_original_campaign_deadline_guards(self):
        guard = self.guard()
        cases = [({'rss': 2*post.GIB, 'child_rss': 2*post.GIB}, 'memory'),
                 ({'free': 7*post.GIB}, 'free-space'),
                 ({'output_bytes': post.GIB+1}, 'output'),
                 ({'now': guard.plan['original_campaign_deadline']}, 'deadline')]
        for values, message in cases:
            with self.subTest(bound=message), self.resource_mocks(**values):
                with self.assertRaisesRegex(ValueError, message):guard.check()

    def test_torch_import_is_refused_without_running_any_model(self):
        guard = self.guard()
        with patch.dict(post.sys.modules, {'torch': object()}), self.resource_mocks():
            with self.assertRaisesRegex(ValueError, 'Torch or models'):guard.check()

    def test_writes_preserve_existing_artifacts_and_reject_escaping_or_absolute_paths(self):
        guard = self.guard()
        with self.resource_mocks():
            first = guard.write('groups/synthetic/audit.json', b'{"synthetic":true}')
            self.assertEqual(first, guard.write('groups/synthetic/audit.json', b'{"synthetic":true}'))
            with self.assertRaises(ValueError):guard.write('groups/synthetic/audit.json', b'changed')
            for bad in ('../escape', str(guard.run/'absolute.json')):
                with self.subTest(path=bad), self.assertRaises(ValueError):guard.write(bad, b'x')
        self.assertEqual((guard.run/'groups/synthetic/audit.json').read_bytes(), b'{"synthetic":true}')

    def test_failed_reservation_leaves_no_postprocessing_directory(self):
        plan = self.plan()
        @contextmanager
        def writer():yield
        ledger = SimpleNamespace(writer=writer, initialize=lambda original: None,
            snapshot=lambda: deepcopy(self.state), reserve_output=lambda *a, **k: (_ for _ in ()).throw(ValueError('synthetic exhausted')))
        with self.resource_mocks(), patch.object(post, 'CampaignLedger', return_value=ledger):
            with self.assertRaisesRegex(ValueError, 'synthetic exhausted'):post.start(plan)
        self.assertFalse(Path(plan['run_directory']).exists())

    def test_changed_existing_plan_cannot_reuse_old_outputs(self):
        guard = self.guard();plan = guard.plan;guard.close()
        changed = deepcopy(plan);changed['plan_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'Existing postprocessing plan differs'):post.start(changed)

    def group_fixture(self):
        plan = self.plan();run = Path(plan['run_directory']);run.mkdir(exist_ok=True)
        summary = self.campaign/'reports/seed-43/summary.json';summary.parent.mkdir(parents=True)
        summary.write_text(json.dumps({'jobs': []}))
        records = []
        for fold in post.FOLDS:
            for gauge in post.GAUGES:
                name = f'{fold}__{gauge}';directory = run/'groups'/name;directory.mkdir(parents=True)
                artifacts = {}
                for kind in ('arrays', 'manifest', 'audit'):
                    suffix = 'npz' if kind == 'arrays' else 'json'
                    relative = f'groups/{name}/{kind}.{suffix}'
                    raw = f'SYNTHETIC_METADATA_ONLY_{name}_{kind}'.encode()
                    (run/relative).write_bytes(raw)
                    artifacts[kind] = {'path': relative, 'bytes': len(raw), 'sha256': post.sha(raw)}
                record = dict(group=name, fold=fold, gauge_id=gauge, plan_sha256=plan['plan_sha256'], artifacts=artifacts)
                (directory/'COMMIT.json').write_text(json.dumps(record));records.append(record)
        reads, writes = [], []
        def read(path, maximum):
            reads.append(str(path));return Path(path).read_bytes()
        guard = SimpleNamespace(run=run, read=read, check=lambda: None, read_budget=lambda n: None,
                                write=lambda path, raw: writes.append((path, raw)), save=lambda: None,
                                record={'read_bytes_reserved': 0})
        return plan, records, guard, reads, writes

    def test_audit_resume_rechecks_all_committed_group_artifacts_without_rebuilding(self):
        plan, records, guard, reads, writes = self.group_fixture()
        with patch('report_primary_seed._ledger_records', return_value=[]), \
                patch('hydra_directflow.post_data.build_group', side_effect=AssertionError('Completed group rebuilt')):
            result = post.audit(plan, self.original, guard)
        self.assertEqual(result['groups_total'], 12)
        self.assertEqual(result['groups'], records)
        self.assertEqual(len(reads), 1+12*4)
        self.assertEqual([x[0] for x in writes], ['AUDIT.json'])

    def test_audit_resume_refuses_wrong_group_plan_or_escaped_artifact_before_read(self):
        for condition in ('group', 'plan', 'path', 'sha'):
            # A new fixture directory avoids overwriting any scientific output.
            child = self.root/f'resume-{condition}';child.mkdir()
            old_campaign = self.campaign;old_original = deepcopy(self.original)
            self.campaign = child
            self.original['run_directory'] = str(child)
            self.original_path.write_text(json.dumps(self.original))
            plan, records, guard, reads, writes = self.group_fixture()
            first = deepcopy(records[0]);path = guard.run/'groups'/first['group']/'COMMIT.json'
            if condition == 'group':first['gauge_id'] = '03479000'
            elif condition == 'plan':first['plan_sha256'] = '0'*64
            elif condition == 'path':first['artifacts']['arrays']['path'] = '../../never-read-protected.npz'
            else:first['artifacts']['arrays']['sha256'] = '0'*64
            path.write_text(json.dumps(first))
            with self.subTest(condition=condition), patch('report_primary_seed._ledger_records', return_value=[]), \
                    patch('hydra_directflow.post_data.build_group', side_effect=AssertionError('Group rebuilt')):
                with self.assertRaises(ValueError):post.audit(plan, self.original, guard)
            self.assertFalse(any('never-read-protected' in x for x in reads))
            self.assertEqual(writes, [])
            self.campaign = old_campaign;self.original = old_original
            self.original_path.write_text(json.dumps(self.original))


if __name__ == '__main__':unittest.main()
