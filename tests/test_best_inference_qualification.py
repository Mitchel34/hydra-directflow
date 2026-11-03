"""Zero-update qualification plan, replay, budget, and supervisor tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SPEC=importlib.util.spec_from_file_location('best_inference_qualification',Path(__file__).resolve().parents[1]/'scripts/qualify_best_inference.py')
qual=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(qual)


class BestInferenceQualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.run=self.root/'.local'/'qualification'
        self.prepared=qual.plan();self.old_root=qual.ROOT;qual.ROOT=self.root
    def tearDown(self):qual.ROOT=self.old_root;self.temp.cleanup()

    def prepare(self):
        record=qual.precharge(self.prepared,self.run)
        qual.mkdir_durable(self.run);qual.atomic_json(self.run/'plan.json',self.prepared)
        for case in qual.matrix():qual.mkdir_durable(self.run/case['id'])
        return record

    def test_exact_plan_is_four_cases_eight_actual_updates_and_no_execution(self):
        self.assertEqual(len(qual.matrix()),4)
        self.assertEqual(sum(c['training_updates'] for c in qual.matrix()),8)
        self.assertTrue(all(c['batch_size']==c['fit_cases']==8 and c['epochs']==2 for c in qual.matrix()))
        self.assertEqual({(c['architecture'],c['device']) for c in qual.matrix()},{('LSTM','cpu'),('Mamba2_Transformer','mps')})
        with patch.object(qual.sys,'argv',['qualification']),patch.object(qual,'plan',return_value=self.prepared),\
             patch.object(qual,'execute') as execute,patch.object(qual.subprocess,'Popen') as spawn,patch('builtins.print'):
            qual.main();execute.assert_not_called();spawn.assert_not_called()
        self.assertFalse(qual.ledger_path().exists())

    def test_consumed_budget_cannot_reset_relocate_or_use_prior_qualification(self):
        self.prepare();before=qual.ledger_path().read_bytes()
        for run in (self.run,self.root/'.local'/'other'):
            with self.assertRaisesRegex(ValueError,'already consumed'):qual.precharge(self.prepared,run)
        self.assertEqual(before,qual.ledger_path().read_bytes())
        self.assertFalse((self.root/'.cache'/'timing-controls-update-budget-v1.json').exists())
        self.assertFalse((self.root/'.cache'/'real-batch-update-budget-v1.json').exists())
        with self.assertRaisesRegex(ValueError,'exact consumed'):
            qual.claim_worker(self.root/'.local'/'other',qual.matrix()[0]['id'],'train',self.prepared)

    def test_claim_is_durable_once_only_and_inference_needs_settled_training(self):
        self.prepare();case=qual.matrix()[0]
        with self.assertRaisesRegex(ValueError,'successful two-update'):
            qual.claim_worker(self.run,case['id'],'infer',self.prepared)
        qual.claim_worker(self.run,case['id'],'train',self.prepared)
        with self.assertRaisesRegex(ValueError,'replay forbidden'):
            qual.claim_worker(self.run,case['id'],'train',self.prepared)
        with self.assertRaisesRegex(ValueError,'successful two-update'):
            qual.claim_worker(self.run,case['id'],'infer',self.prepared)
        result=self.run/case['id']/'train-result.json';qual.atomic_json(result,{'synthetic':True})
        attempt=dict(pid=qual.os.getpid(),passed=True,optimizer_updates_entered=2,optimizer_updates_confirmed=2,
                     count_exact=True,result_sha256=hashlib.sha256(result.read_bytes()).hexdigest())
        qual.settle_worker(self.run,case['id'],'train',attempt)
        qual.claim_worker(self.run,case['id'],'infer',self.prepared)
        with self.assertRaisesRegex(ValueError,'replay forbidden'):
            qual.claim_worker(self.run,case['id'],'infer',self.prepared)

    def test_unreserved_replayed_expired_or_wrong_plan_worker_cannot_reach_training(self):
        self.prepare();case=qual.matrix()[0];qual.claim_worker(self.run,case['id'],'train',self.prepared)
        with patch.object(qual,'plan',return_value=self.prepared),patch.object(qual,'train_case') as train:
            with self.assertRaisesRegex(ValueError,'replay forbidden'):qual.worker(self.run,case['id'],'train')
            train.assert_not_called()
        ledger=qual.read_json(qual.ledger_path());ledger['deadline_epoch']=time.time()-1;qual.atomic_json(qual.ledger_path(),ledger)
        with self.assertRaisesRegex(ValueError,'deadline expired'):
            qual.claim_worker(self.run,qual.matrix()[1]['id'],'train',self.prepared)
        with self.assertRaisesRegex(ValueError,'exact consumed'):
            qual.claim_worker(self.run,qual.matrix()[1]['id'],'train',{**self.prepared,'plan_sha256':'f'*64})

    def test_changed_training_result_blocks_fresh_inference_claim(self):
        self.prepare();case=qual.matrix()[0];qual.claim_worker(self.run,case['id'],'train',self.prepared)
        path=self.run/case['id']/'train-result.json';qual.atomic_json(path,{'original':True})
        qual.settle_worker(self.run,case['id'],'train',dict(pid=qual.os.getpid(),passed=True,
            optimizer_updates_entered=2,optimizer_updates_confirmed=2,count_exact=True,result_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        qual.atomic_json(path,{'modified':True})
        with self.assertRaisesRegex(ValueError,'result changed'):qual.claim_worker(self.run,case['id'],'infer',self.prepared)

    def test_directory_and_claim_durability_precede_worker_work(self):
        events=[];real_sync=qual.sync_directory
        def sync(path):events.append(('sync',str(path)));real_sync(path)
        with patch.object(qual,'sync_directory',side_effect=sync):
            self.prepare();qual.claim_worker(self.run,qual.matrix()[0]['id'],'train',self.prepared)
        self.assertIn(('sync',str(self.root)),events)
        self.assertEqual(events[-1],('sync',str(self.root/'.cache')))
        ledger=qual.read_json(qual.ledger_path())
        self.assertEqual(ledger['cases'][qual.matrix()[0]['id']]['train']['status'],'CLAIMED')
        self.assertEqual(ledger['precharged_updates'],8)

    def test_failed_durable_charge_never_spawns(self):
        with patch.object(qual,'plan',return_value=self.prepared),patch.object(qual.os,'fsync',side_effect=OSError('synthetic sync failure')),\
             patch.object(qual.subprocess,'Popen') as spawn:
            with self.assertRaises(OSError):
                qual.execute(self.prepared,run_directory=self.run,report_path=self.root/'report.json',approved_plan_sha256=self.prepared['plan_sha256'])
            spawn.assert_not_called()

    def test_fixture_ids_are_canonical_training_inference_are_separate_and_label_free(self):
        from hydra_directflow.training import validate_coordinates
        from hydra_directflow.inference import validate_inputs
        for schema in (qual.FULL,qual.Q_ONLY):
            arrays,training_cohort=qual.training_fixture(schema);inputs,source,cohort=qual.inference_fixture(schema)
            self.assertEqual(validate_coordinates(arrays,24),12)
            self.assertEqual(validate_inputs(inputs,history_hours=24,feature_schema=schema,
                expected_cohort_admission_digest=cohort,expected_gauge_id='03161000'),4)
            self.assertNotEqual(training_cohort,cohort)
            self.assertFalse(set(arrays['case_id'].tolist())&set(inputs['case_id'].tolist()))
            self.assertNotIn('target',inputs);self.assertNotIn('target_mask',inputs)
            self.assertTrue(all(str(case).startswith('03161000:2021-') for case in arrays['case_id']))
            again,again_source,again_cohort=qual.inference_fixture(schema)
            self.assertEqual((source,cohort),(again_source,again_cohort))
            np.testing.assert_array_equal(inputs['history'],again['history'])

    def test_supervisor_interrupt_retains_counts_kills_child_and_never_claims_exactness(self):
        self.prepare();case=qual.matrix()[0];directory=self.run/case['id']
        qual.atomic_json(directory/'train-status.json',dict(optimizer_updates_entered=2,optimizer_updates_confirmed=1,mps_driver_bytes=0))
        class Child:
            pid=123;returncode=None
            def poll(self):return self.returncode
            def kill(self):self.returncode=-9
            def wait(self,timeout=None):return self.returncode
        child=Child();process=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=100))
        with patch.object(qual.subprocess,'Popen',return_value=child),patch('psutil.Process',return_value=process),\
             patch.object(qual.time,'sleep',side_effect=KeyboardInterrupt):
            attempt=qual.supervise(['unused'],run=self.run,case=case,phase='train',deadline=time.time()+60,env={},plan_sha256=self.prepared['plan_sha256'])
        self.assertEqual(child.returncode,-9);self.assertFalse(attempt['passed']);self.assertFalse(attempt['count_exact'])
        self.assertEqual((attempt['optimizer_updates_entered'],attempt['optimizer_updates_confirmed']),(2,1))
        self.assertEqual(attempt['error_type'],'KeyboardInterrupt');self.assertEqual(qual.read_json(qual.ledger_path())['precharged_updates'],8)

    def test_supervisor_rejects_wrong_result_phase_or_plan_even_with_success_exit(self):
        self.prepare();case=qual.matrix()[0];directory=self.run/case['id']
        class Child:
            pid=123;returncode=0
            def poll(self):return 0
            def wait(self,timeout=None):return 0
        for mutation in ('phase','plan','case'):
            result=dict(status='PASSED',pid=123,phase='train',case_id=case['id'],plan_sha256=self.prepared['plan_sha256'],
                        optimizer_updates_entered=2,optimizer_updates_confirmed=2)
            if mutation=='phase':result['phase']='infer'
            elif mutation=='plan':result['plan_sha256']='f'*64
            else:result['case_id']='other'
            qual.atomic_json(directory/'train-result.json',result)
            log=directory/'train.log'
            if log.exists():log.unlink()
            with patch.object(qual.subprocess,'Popen',return_value=Child()),patch('psutil.Process'):
                attempt=qual.supervise(['unused'],run=self.run,case=case,phase='train',deadline=time.time()+60,env={},plan_sha256=self.prepared['plan_sha256'])
            self.assertFalse(attempt['passed'])


if __name__=='__main__':unittest.main()
