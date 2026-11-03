"""Controller planning/barrier tests; no workers, models or scientific values."""
from copy import deepcopy
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().parents[1]/'scripts/run_primary_campaign.py'
SPEC=importlib.util.spec_from_file_location('primary_controller_tests',PATH)
controller=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(controller)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=controller.ROOT/'.local');self.root=Path(self.temp.name).resolve()
    def tearDown(self):self.temp.cleanup()

    def test_draft_protocol_refuses_before_any_qualification_or_worker(self):
        p=self.root/'draft.json';p.write_text(json.dumps({'protocol_frozen':False,'scientific_training_authorized':False}))
        with patch.object(controller,'qualification_evidence',side_effect=AssertionError('Reached qualification')):
            with self.assertRaisesRegex(ValueError,'User-reviewed'):controller.prepare(p,self.root/'run')

    def test_boolean_flags_without_human_method_record_do_not_release(self):
        p=self.root/'draft.json';p.write_text(json.dumps({'protocol_frozen':True,'scientific_training_authorized':True,'human_confirmation':{'received':False}}))
        with self.assertRaisesRegex(ValueError,'human method confirmation'):controller.prepare(p,self.root/'run')

    def test_executed_qualification_source_hashes_are_required(self):
        with patch.object(controller,'read',return_value=b'{}'):
            with self.assertRaisesRegex(ValueError,'receipt hashes'):controller.qualification_evidence()

    def test_actual_qualification_reports_match_current_core_without_model_import(self):
        evidence=controller.qualification_evidence()
        self.assertEqual(evidence['model_report_sha256'],controller.MODEL_QUAL_SHA)
        self.assertEqual(evidence['data_report_sha256'],controller.DATA_QUAL_SHA)

    def test_prepared_private_plan_preserves180_order_and_exact_replay_budget(self):
        protocol=json.loads((controller.ROOT/'configs/primary-training-protocol-v1.json').read_text())
        protocol.update(protocol_frozen=True,scientific_training_authorized=True)
        protocol['training']['max_epochs']=200;protocol['limits']['max_optimizer_updates']=20930920
        protocol['human_confirmation']={'received':True,'message_reference':'SYNTHETIC_TEST_ONLY','confirmed_at_utc':'2026-11-03T00:00:00Z'}
        p=self.root/'synthetic-review.json';p.write_text(json.dumps(protocol))
        # This prepares metadata only. It does not initialize a campaign or
        # invoke the worker even though this fixture exercises reviewed flags.
        with patch.object(controller,'controller_evidence',return_value={'status':'SYNTHETIC_TEST_ONLY'}):
            result=controller.prepare(p,self.root/'never-executed')
        self.assertEqual(len(result['jobs']),180)
        self.assertEqual(sum(j['logical_update_limit'] for j in result['jobs']),18200800)
        self.assertEqual(sum(j['max_updates'] for j in result['jobs']),20930920)
        self.assertTrue(all(j['max_updates']==j['logical_update_limit']+30*j['gradient_batches_per_epoch'] for j in result['jobs']))
        self.assertEqual([j['seed'] for j in result['jobs']],[17]*108+[29]*36+[43]*36)
        self.assertFalse((self.root/'never-executed').exists())

    def test_private_paths_and_atomic_receipts_do_not_overwrite(self):
        with self.assertRaisesRegex(ValueError,'private path'):controller.private(controller.ROOT/'reports'/'bad.json')
        p=self.root/'receipt.json';controller.atomic(p,{'ok':1},new=True)
        original=p.read_bytes()
        with self.assertRaises(ValueError):controller.atomic(p,{'ok':2},new=True)
        self.assertEqual(p.read_bytes(),original)

    def test_global_release_prevents_relocated_initialization_before_output_creation(self):
        release=self.root/'release.json';release.write_text('{}')
        with patch.object(controller,'GLOBAL_RELEASE',release):
            with self.assertRaisesRegex(ValueError,'already claimed'):
                controller.initialize({'run_directory':str(self.root/'another')})
        self.assertFalse((self.root/'another').exists())

    def test_missing_controller_qualification_does_not_become_execution_authority(self):
        with self.assertRaisesRegex(ValueError,'controller receipt'):controller.controller_evidence({})

    def test_mutable_progress_recovers_while_preserving_abandoned_temporary(self):
        target=self.root/'progress.json';abandoned=self.root/'progress.json.writing'
        abandoned.write_bytes(b'preserve interrupted progress')
        controller.atomic(target,{'status':'RESUMED'})
        self.assertEqual(json.loads(target.read_text()),{'status':'RESUMED'})
        self.assertEqual(abandoned.read_bytes(),b'preserve interrupted progress')

    def test_existing_seed_report_is_hashed_before_allowing_advance(self):
        run=self.root/'run';directory=run/'reports'/'seed-17';directory.mkdir(parents=True)
        jobs=[{'job_id':f'job-{i}'} for i in range(180)]
        plan={'plan_sha256':'a'*64,'run_directory':str(run),'ledger_directory':str(run/'ledger'),
              'jobs':jobs,'maximum_reports_output_bytes':64*1024**2}
        summary={'synthetic':True,'plan_sha256':'a'*64,'seed':17};summary['content_sha256']=controller.digest(summary)
        payloads={'summary.json':json.dumps(summary).encode(),'index.html':b'<p>synthetic</p>'}
        for name,raw in payloads.items():(directory/name).write_bytes(raw)
        commit={'schema':'hydra-primary-seed-report-commit/v1','seed':17,'plan_sha256':'a'*64,
                'completion_receipts':{j['job_id']:'b'*64 for j in jobs[:108]},
                'payloads':{name:{'bytes':len(raw),'sha256':controller.sha(raw)} for name,raw in payloads.items()}}
        (directory/'COMMIT.json').write_text(json.dumps(commit))
        ledger=SimpleNamespace(snapshot=lambda:{'seed_reports_ready':[17]})
        records=[{'job':j,'receipt_sha256':'b'*64} for j in jobs[:108]]
        with patch('report_primary_seed._ledger_records',return_value=records):
            controller.publish_ready_seeds(plan,ledger)
            commit['completion_receipts']['job-0']='c'*64
            (directory/'COMMIT.json').write_text(json.dumps(commit))
            with self.assertRaisesRegex(ValueError,'verified ledger'):controller.publish_ready_seeds(plan,ledger)
            commit['completion_receipts']['job-0']='b'*64
            (directory/'COMMIT.json').write_text(json.dumps(commit))
            (directory/'index.html').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'bytes changed'):controller.publish_ready_seeds(plan,ledger)

    def test_execution_cannot_change_epoch_policy_under_a_matching_protocol_file_hash(self):
        protocol={'protocol_frozen':True,'scientific_training_authorized':True,'protected_assessment_access':False,
            'paid_cloud_jobs_started':0,'human_confirmation':{'received':True,'message_reference':'SYNTHETIC','confirmed_at_utc':'2026-11-03T00:00:00Z'},
            'training':{'max_epochs':100},'limits':{},'source_plan_sha256':'b'*64,'snapshot_sha256':'c'*64}
        pp=self.root/'reviewed.json';pp.write_text(json.dumps(protocol))
        plan={'plan_sha256':'a'*64,'protocol_path':str(pp),'protocol_sha256':controller.sha(pp.read_bytes()),
            'training':{'max_epochs':200},'limits':{},'source_plan_sha256':'b'*64,'snapshot_sha256':'c'*64}
        path=self.root/'plan.json';path.write_text(json.dumps(plan))
        with patch.object(controller,'validate_execution_plan'):
            with self.assertRaisesRegex(ValueError,'reviewed protocol'):controller.authenticated_plan(path,'a'*64)

    def _supervisor_fixture(self, exit_code, *, verified=False, running=False, oversized_memory=False):
        run=self.root/'supervised';run.mkdir()
        plan={'plan_sha256':'a'*64,'protocol_sha256':'b'*64,'run_directory':str(run),'ledger_directory':str(run/'ledger'),
              'limits':{'max_job_seconds':60,'max_output_bytes':1024**2,'min_free_bytes':1,'max_combined_rss_bytes':1024**3},
              'per_worker_log_limit_bytes':1024**2}
        events=[]
        class Ledger:
            @contextmanager
            def writer(self):yield self
            def initialize(self,p):events.append('initialize')
            def next_job(self):
                return {'state':'PENDING','deadline':None,'job':{'job_id':'synthetic','order':0}}
            def snapshot(self):
                return {'attempts':[],'campaign_deadline':None,'jobs':[{'state':'VERIFIED_TRAIN_EVAL' if verified else 'PENDING'}]}
        class Child:
            pid=123456
            returncode=None if running else exit_code
            def poll(self):return self.returncode
            def wait(self,timeout):return self.returncode
            def terminate(self):events.append('terminate');self.returncode=-15
            def kill(self):events.append('kill');self.returncode=-9
        child=Child();ledger=Ledger()
        memory=2*1024**3 if oversized_memory else 1024
        process=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=memory))
        return plan,ledger,child,process,events

    def test_successful_process_exit_without_metrics_cannot_advance(self):
        plan,ledger,child,process,events=self._supervisor_fixture(0)
        with patch.object(controller,'verify_release'),patch.object(controller,'CampaignLedger',return_value=ledger),\
             patch.object(controller,'publish_ready_seeds'),patch.object(controller,'validate_execution_plan'),\
             patch.object(controller.subprocess,'Popen',return_value=child) as launch,\
             patch('psutil.Process',return_value=process),patch.object(controller,'progress') as progress:
            with self.assertRaisesRegex(ValueError,'authenticated training/evaluation'):
                controller.supervise(self.root/'plan.json',plan)
            launch.assert_called_once()
            self.assertEqual(progress.call_args.kwargs['status'],'STOPPED_CONTROLLER_GUARD')
            self.assertIn('--worker',launch.call_args.args[0])
            self.assertEqual(launch.call_args.kwargs['env']['PYTORCH_ENABLE_MPS_FALLBACK'],'0')

    def test_worker_failure_stops_without_a_replacement_fit(self):
        plan,ledger,child,process,events=self._supervisor_fixture(7)
        with patch.object(controller,'verify_release'),patch.object(controller,'CampaignLedger',return_value=ledger),\
             patch.object(controller,'publish_ready_seeds'),patch.object(controller,'validate_execution_plan'),\
             patch.object(controller.subprocess,'Popen',return_value=child) as launch,\
             patch('psutil.Process',return_value=process),patch.object(controller,'progress',side_effect=lambda p,l,**kw:kw):
            result=controller.supervise(self.root/'plan.json',plan)
            self.assertEqual(result['status'],'STOPPED_WORKER_FAILURE');launch.assert_called_once()

    def test_parent_memory_guard_terminates_and_reaps_worker(self):
        plan,ledger,child,process,events=self._supervisor_fixture(0,running=True,oversized_memory=True)
        with patch.object(controller,'verify_release'),patch.object(controller,'CampaignLedger',return_value=ledger),\
             patch.object(controller,'publish_ready_seeds'),patch.object(controller,'validate_execution_plan'),\
             patch.object(controller.subprocess,'Popen',return_value=child) as launch,\
             patch('psutil.Process',return_value=process),patch.object(controller,'progress'):
            with self.assertRaisesRegex(ValueError,'memory guard'):controller.supervise(self.root/'plan.json',plan)
            self.assertEqual(events.count('terminate'),1);launch.assert_called_once()


if __name__=='__main__':unittest.main()
