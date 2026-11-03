"""Synthetic27-view controller checks; no study values/Torch/fitting/network."""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hydra_directflow import campaign_data as data,corpus_dataset as ds
import test_campaign_data as fixtures
from isolated_test_support import rerun_without_model_imports_if_needed

PATH=Path(__file__).resolve().parents[1]/'scripts/qualify_campaign_data.py'
SPEC=importlib.util.spec_from_file_location('campaign_data_qualifier_tests',PATH)
q=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(q)


class CampaignQualificationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.CampaignDataTests('test_fit_inner_only_exact_roles_values_and_readonly'); self.fixture.setUp()
        self.stack=ExitStack(); root=self.fixture.root.resolve()
        for sub in ('.local','.cache','reports'): (root/sub).mkdir(exist_ok=True)
        for name,value in {'ROOT':root,'CORPUS':self.fixture.corpus.resolve(),'RUN':root/'.local/run','REPORT':root/'reports/report.json',
                           'LEDGER':root/'.cache/allowance.json','CLAIM':root/'.cache/claim.json'}.items():
            self.stack.enter_context(patch.object(q,name,value))
        self.stack.enter_context(patch.object(q,'implementation',return_value={'test':'fixed'}))
    def tearDown(self): self.stack.close(); self.fixture.tearDown()

    def test_exact27_views_do_not_repeat_architecture_or_later_seeds(self):
        jobs=q.unique_jobs(self.fixture.roster)
        self.assertEqual(len(jobs),27); self.assertTrue(all(j['architecture']=='LSTM' and j['seed']==17 for j in jobs))
        self.assertEqual(sum(j['history_hours']==24 for j in jobs),3)
        wrong=deepcopy(self.fixture.roster); wrong['jobs'][0]['input_arm']='unknown'
        with self.assertRaisesRegex(ValueError,'27'): q.unique_jobs(wrong)

    def test_plan_opens_no_numeric_shard_members_and_limits_are_distinct(self):
        with patch.object(ds,'_decode',side_effect=AssertionError('numeric/mask decode')),patch.object(ds,'inspect_corpus',side_effect=AssertionError('inspection')):
            plan=q.plan()
        self.assertEqual(plan['spec']['maximum_cumulative_corpus_logical_read_bytes'],8*1024**3)
        self.assertEqual(plan['spec']['maximum_parent_child_rss_bytes'],4*1024**3)
        self.assertEqual(plan['spec']['maximum_bundle_estimated_bytes'],1024**3)
        self.assertEqual(plan['spec']['maximum_wall_seconds'],900)
        self.assertEqual(plan['spec']['model_calls'],0)

    def test_fixed_once_precharge_and_worker_replay_refuse(self):
        prepared=q.plan(); allowance=q.precharge(prepared)
        self.assertEqual(allowance['maximum_cumulative_corpus_logical_read_bytes'],8*1024**3)
        self.assertEqual(q.claim(prepared)['plan_sha256'],prepared['plan_sha256'])
        with self.assertRaises(FileExistsError): q.claim(prepared)
        with self.assertRaisesRegex(ValueError,'Fresh once-only'): q.precharge(prepared)

    def test_expired_or_changed_allowance_blocks_worker(self):
        prepared=q.plan(); q.precharge(prepared)
        grant=json.loads(q.LEDGER.read_text()); grant['deadline_epoch']=time.time()-1; q.LEDGER.write_text(json.dumps(grant))
        with self.assertRaisesRegex(ValueError,'live'): q.claim(prepared)
        self.assertFalse(q.CLAIM.exists())

    def test_exact_scope_cannot_expand_by_rehash(self):
        prepared=q.plan(); changed=deepcopy(prepared); changed['spec']['maximum_cumulative_corpus_logical_read_bytes']+=1
        changed['plan_sha256']=q.sha(q.canonical(changed['spec']))
        with self.assertRaisesRegex(ValueError,'source/code/scope'): q.precharge(changed)
        self.assertFalse(q.LEDGER.exists())

    def test_all27_synthetic_views_check_labels_pairs_and_do_not_import_torch(self):
        if rerun_without_model_imports_if_needed(self): return
        prepared=q.plan(); guard=q.ReadGuard(q.CORPUS,maximum=q.MAX_READ); original=Path.open
        def guarded(path,mode='r',*args,**kwargs):
            guard.audit('open',(str(path),mode,0)); return original(path,mode,*args,**kwargs)
        with patch.object(Path,'open',guarded): result=q.qualify(prepared,guard,lambda:None)
        self.assertEqual(len(result['views']),27); self.assertEqual(result['paired_views']['fold_gauge_pairs'],12)
        self.assertTrue(result['paired_views']['matched_original_targets'])
        self.assertTrue(all(row['fit_inner_cases']==11 and row['outer_cases']==6 for row in result['views']))
        self.assertTrue(all(row['predictor_batch_cases']==6 and not row['outer_target_values_decoded'] for row in result['views']))
        self.assertGreater(result['read_telemetry']['logical_bytes_reserved_at_open'],0)
        for key in ('torch_imports','model_calls','scaler_fits','optimizer_updates','scientific_scores','network_requests'):
            self.assertEqual(result[key],0)
        changed=deepcopy(result['views']); changed[0]['fit_cohort']='different'
        with self.assertRaisesRegex(ValueError,'identity differs'): q.verify_pairs(changed)

    def test_source_identity_does_not_change_for_architecture_or_seed_only(self):
        a=self.fixture.load(self.fixture.job(fold='final_development',architecture='LSTM',seed=17))
        b=self.fixture.load(self.fixture.job(fold='final_development',architecture='Transformer',seed=29))
        self.assertEqual(a['source_identity'],b['source_identity']); self.assertEqual(a['cohort_admission_digest'],b['cohort_admission_digest'])

    def test_cumulative_read_guard_rejects_before_extra_open(self):
        source=q.CORPUS/'PLAN.json'; size=source.stat().st_size; guard=q.ReadGuard(q.CORPUS,maximum=size)
        guard.audit('open',(str(source),'rb',0)); self.assertEqual(guard.read_bytes,size)
        with self.assertRaisesRegex(ValueError,'before open'): guard.audit('open',(str(source),'rb',0))
        self.assertEqual(guard.opens,1)

    def test_supervisor_interrupt_kills_child_and_preserves_allowance(self):
        prepared=q.plan()
        class Child:
            pid=123; returncode=None
            def poll(self): return self.returncode
            def kill(self): self.returncode=-9
            def wait(self,timeout=None): return self.returncode
        child=Child(); process=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=100))
        with patch.object(q.subprocess,'Popen',return_value=child),patch('psutil.Process',return_value=process),\
                patch.object(q.time,'sleep',side_effect=KeyboardInterrupt):
            result=q.execute(prepared,prepared['plan_sha256'])
        self.assertEqual(result['status'],'FAILED_PRESERVED_ATTEMPT'); self.assertEqual(result['failure_reason'],'KeyboardInterrupt')
        self.assertEqual(child.returncode,-9); self.assertTrue(q.LEDGER.is_file()); self.assertTrue(q.REPORT.is_file())

    def test_fast_success_exit_with_final_memory_overage_preserves_failed_report(self):
        prepared=q.plan(); process=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=100))
        child=SimpleNamespace(pid=123,returncode=0,poll=lambda:0,wait=lambda timeout=None:0)
        def spawn(*args,**kwargs):
            q.write_new(q.RUN/'worker-result.json',dict(status='PASSED',plan_sha256=prepared['plan_sha256'],worker_pid=123,
                views=[{}]*27,read_telemetry={'logical_bytes_reserved_at_open':0},observed_worker_peak_rss_bytes=q.MAX_RSS))
            return child
        with patch.object(q.subprocess,'Popen',side_effect=spawn),patch('psutil.Process',return_value=process):
            result=q.execute(prepared,prepared['plan_sha256'])
        self.assertEqual(result['status'],'FAILED_PRESERVED_ATTEMPT'); self.assertEqual(result['failure_reason'],'ValueError')
        self.assertTrue(q.REPORT.is_file()); self.assertTrue(q.LEDGER.is_file())


if __name__=='__main__': unittest.main()
