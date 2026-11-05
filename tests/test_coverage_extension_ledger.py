"""Synthetic durable Phase A accounting; no models or real study ledger."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from hydra_directflow.coverage_extension_ledger import (
    PhaseALedger, LIMITS, CHECKPOINT_IDS, ExtensionAllowanceExhausted, validate_extension_plan)
from hydra_directflow.campaign_ledger import digest


def plan():
    value=dict(schema='SYNTHETIC_PHASE_A_NO_REAL_EXECUTION',limits=deepcopy(LIMITS),
        checkpoint_sha256_by_job={key:hashlib.sha256(key.encode()).hexdigest() for key in sorted(CHECKPOINT_IDS)},
        original_campaign_deadline=100000.,protected_assessment_access=False,paid_cloud_jobs=0)
    value['plan_sha256']=digest(value);return value


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.directory=Path(self.temp.name).resolve()/'ledger'
        self.now=[1000.];self.plan=plan();self.ledger=PhaseALedger(self.directory,clock=lambda:self.now[0])
        self.job=sorted(CHECKPOINT_IDS)[0];self.checkpoint=self.plan['checkpoint_sha256_by_job'][self.job]
    def tearDown(self):self.temp.cleanup()

    def init(self):self.ledger.initialize(self.plan,77)
    def model(self,lease):
        token=self.ledger.enter_model(lease,job_id=self.job,checkpoint_sha256=self.checkpoint)
        self.ledger.confirm_model(lease,token);return token
    def forward(self,lease,n=64):
        token=self.ledger.enter_forward(lease,case_count=n,batch_count=1,stage='clean_regression',case_ids_sha256='d'*64)
        self.ledger.confirm_forward(lease,token);return token
    def complete(self,lease):
        receipt={**self.ledger.receipt_bindings(lease),'status':'SYNTHETIC_VERIFIED_TASK','opaque_artifact_sha256':'f'*64}
        self.ledger.complete_task(lease,receipt,validator=lambda r,b:digest(r));return receipt

    def test_finite_plan_cannot_change_limits_checkpoint_inventory_or_protected_scope(self):
        for change in ('limit','boolean_limit','checkpoint','protected','hash'):
            p=deepcopy(self.plan)
            if change=='limit':p['limits']['max_case_checkpoint_evaluations']+=1
            elif change=='boolean_limit':p['limits']['max_infrastructure_resumes']=True
            elif change=='checkpoint':p['checkpoint_sha256_by_job'].pop(self.job)
            elif change=='protected':p['protected_assessment_access']=True
            else:p['plan_sha256']='e'*64
            if change!='hash':p['plan_sha256']=digest({k:v for k,v in p.items() if k!='plan_sha256'})
            with self.subTest(change=change),self.assertRaises(ValueError):validate_extension_plan(p)

    def test_initialization_persists_original_clock_reservation_and_read_charge(self):
        with self.ledger.writer():
            self.ledger.initialize(self.plan,77,initial_read_bytes=123)
        self.now[0]+=99
        with self.ledger.writer():self.init()
        s=self.ledger.snapshot();self.assertEqual(s['started'],1000.)
        self.assertEqual(s['deadline'],1000.+21600);self.assertEqual(s['read_reserved'],123)
        self.assertEqual(s['reservation_token'],77)

    def test_original_campaign_deadline_can_only_shorten_phase_window(self):
        p=deepcopy(self.plan);p['original_campaign_deadline']=1005.;p['plan_sha256']=digest({k:v for k,v in p.items() if k!='plan_sha256'})
        with self.ledger.writer():self.ledger.initialize(p,77)
        self.assertEqual(self.ledger.snapshot()['deadline'],1005.)
        self.now[0]=1005.
        with self.ledger.writer(),self.assertRaises(ExtensionAllowanceExhausted):self.ledger.initialize(p,77)

    def test_changed_plan_or_reservation_and_repeated_initial_charge_are_rejected(self):
        with self.ledger.writer():
            self.init()
            for changed,token,initial in (({**self.plan,'different':'x'},77,0),(self.plan,78,0),(self.plan,77,1)):
                p=deepcopy(changed);p['plan_sha256']=digest({k:v for k,v in p.items() if k!='plan_sha256'})
                with self.assertRaises(ValueError):self.ledger.initialize(p,token,initial_read_bytes=initial)

    def test_exclusive_writer_blocks_concurrent_owner(self):
        other=PhaseALedger(self.directory,clock=lambda:self.now[0])
        with self.ledger.writer():
            self.init()
            with self.assertRaises(BlockingIOError):
                with other.writer():pass

    def test_model_forward_charges_are_visible_before_calls_and_confirm_once(self):
        with self.ledger.writer():
            self.init();lease=self.ledger.claim_task('clean/job','clean')
            model=self.ledger.enter_model(lease,job_id=self.job,checkpoint_sha256=self.checkpoint)
            snap=self.ledger.snapshot();self.assertEqual((snap['entered_models'],snap['confirmed_models']),(1,0))
            with self.assertRaisesRegex(ValueError,'confirmed checkpoint'):self.ledger.enter_forward(lease,case_count=2,batch_count=1,stage='clean',case_ids_sha256='d'*64)
            self.ledger.confirm_model(lease,model)
            token=self.ledger.enter_forward(lease,case_count=3,batch_count=1,stage='clean',case_ids_sha256='d'*64)
            snap=self.ledger.snapshot();self.assertEqual((snap['entered_cases'],snap['confirmed_cases']),(3,0))
            with self.assertRaises(ValueError):self.ledger.confirm_model(lease,token)
            self.ledger.confirm_forward(lease,token)
            with self.assertRaises(ValueError):self.ledger.confirm_forward(lease,token)
            self.complete(lease)
        snap=self.ledger.snapshot();self.assertEqual(snap['confirmed_cases'],3)
        self.assertEqual(snap['distinct_checkpoint_jobs'],1);self.assertEqual(snap['attempts'][0]['confirmed_batches'],1)

    def test_quota_failures_precede_any_new_call_token(self):
        with self.ledger.writer():
            self.init();lease=self.ledger.claim_task('clean/job','clean');self.model(lease)
            self.ledger.connection.execute('UPDATE phase SET entered_cases=? WHERE id=1',(LIMITS['max_case_checkpoint_evaluations']-2,))
            with self.assertRaises(ExtensionAllowanceExhausted):self.forward(lease,n=3)
            self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM calls WHERE kind='FORWARD'").fetchone()[0],0)
            self.ledger.connection.execute('UPDATE phase SET entered_cases=0,entered_batches=? WHERE id=1',(LIMITS['max_forward_batches'],))
            with self.assertRaises(ExtensionAllowanceExhausted):self.forward(lease,n=1)
            with self.assertRaises(ValueError):self.ledger.enter_forward(lease,case_count=65,batch_count=1,stage='clean',case_ids_sha256='d'*64)
            self.ledger.interrupt(lease,'SYNTHETIC_RESOURCE_STOP')

    def test_unknown_checkpoint_and_model_switch_fail_without_charge(self):
        with self.ledger.writer():
            self.init();lease=self.ledger.claim_task('clean/job','clean')
            with self.assertRaises(ValueError):self.ledger.enter_model(lease,job_id=self.job,checkpoint_sha256='f'*64)
            self.assertEqual(self.ledger.snapshot()['entered_models'],0);self.model(lease)
            other=sorted(CHECKPOINT_IDS)[1]
            with self.assertRaises(ValueError):self.ledger.enter_model(lease,job_id=other,checkpoint_sha256=self.plan['checkpoint_sha256_by_job'][other])
            self.complete(lease)

    def test_failed_unconfirmed_charge_is_retained_and_blocks_success(self):
        with self.ledger.writer():
            self.init();lease=self.ledger.claim_task('clean/job','clean');self.model(lease)
            self.ledger.enter_forward(lease,case_count=64,batch_count=1,stage='clean',case_ids_sha256='d'*64)
            with self.assertRaisesRegex(ValueError,'Unconfirmed'):self.complete(lease)
            self.ledger.interrupt(lease,'SYNTHETIC_INTERRUPTION')
        s=self.ledger.snapshot();self.assertEqual((s['entered_cases'],s['confirmed_cases']),(64,0))

    def test_single_explicit_resume_keeps_charges_and_deadline_and_blocks_skipping(self):
        with self.ledger.writer():
            self.init();lease=self.ledger.claim_task('clean/job','clean');self.model(lease);self.forward(lease)
            self.ledger.interrupt(lease,'SYNTHETIC_INTERRUPTION')
        deadline=self.ledger.snapshot()['deadline'];self.now[0]+=100
        with self.ledger.writer():
            self.init()
            with self.assertRaises(ValueError):self.ledger.claim_task('other','clean')
            with self.assertRaises(ValueError):self.ledger.claim_task('clean/job','clean')
            lease=self.ledger.claim_task('clean/job','clean',resume=True);self.model(lease);self.forward(lease,n=4)
            self.ledger.interrupt(lease,'SECOND_SYNTHETIC_INTERRUPTION')
            with self.assertRaises(ExtensionAllowanceExhausted):self.ledger.claim_task('clean/job','clean',resume=True)
        s=self.ledger.snapshot();self.assertEqual(s['deadline'],deadline);self.assertEqual(s['infrastructure_resumes'],1)
        self.assertEqual((s['entered_cases'],s['confirmed_cases']),(68,68))

    def test_numerical_or_integrity_failure_is_not_resumeable_or_skippable(self):
        for category in ('NUMERICAL','INTEGRITY','IMPLEMENTATION'):
            ledger=PhaseALedger(self.directory/category,clock=lambda:self.now[0])
            with ledger.writer():
                ledger.initialize(self.plan,77);lease=ledger.claim_task('bad','clean');ledger.fail(lease,category,'SYNTHETIC_FAILURE')
                with self.assertRaises(ValueError):ledger.claim_task('bad','clean',resume=True)
                with self.assertRaises(ValueError):ledger.claim_task('next','clean')

    def test_completed_task_cannot_be_repeated_and_receipt_digest_is_checked(self):
        with self.ledger.writer():
            self.init();lease=self.ledger.claim_task('inputs','inputs');receipt=self.complete(lease)
            for resume in (False,True):
                with self.assertRaises(ValueError):self.ledger.claim_task('inputs','inputs',resume=resume)
        self.assertEqual(self.ledger.task('inputs')['receipt'],receipt)
        con=sqlite3.connect(self.ledger.database);con.execute("UPDATE tasks SET receipt_sha=? WHERE task_key='inputs'",('e'*64,));con.commit();con.close()
        with self.assertRaisesRegex(ValueError,'receipt digest'):self.ledger.snapshot()

    def test_completed_state_requires_receipt_and_exact_attempt_bindings(self):
        for mutate in ('missing','wrong_phase','wrong_task','wrong_stage','wrong_attempt'):
            ledger=PhaseALedger(self.directory/mutate,clock=lambda:self.now[0])
            with ledger.writer():
                ledger.initialize(self.plan,77);lease=ledger.claim_task('inputs','inputs')
                receipt={**ledger.receipt_bindings(lease),'status':'SYNTHETIC_VERIFIED_TASK'}
                ledger.complete_task(lease,receipt,validator=lambda r,b:digest(r))
            con=sqlite3.connect(ledger.database)
            if mutate=='missing':con.execute("UPDATE tasks SET receipt_json=NULL WHERE task_key='inputs'")
            else:
                key,value={'wrong_phase':('phase_plan_sha256','0'*64),'wrong_task':('task_key','other'),
                           'wrong_stage':('task_stage','other'),'wrong_attempt':('attempt_id',999)}[mutate]
                receipt[key]=value
                con.execute("UPDATE tasks SET receipt_json=?,receipt_sha=? WHERE task_key='inputs'",(json.dumps(receipt),digest(receipt)))
            con.commit();con.close()
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):ledger.snapshot()

    def test_output_precharge_settlement_failure_and_read_no_refunds(self):
        with self.ledger.writer():
            self.init();lease=self.ledger.claim_task('inputs','inputs')
            token=self.ledger.reserve_output(100,lease,kind='ARRAY')
            self.assertEqual(self.ledger.snapshot()['output_reserved'],100)
            self.ledger.settle_output(lease,token,actual_bytes=9,failed=True)
            self.ledger.reserve_read(321,lease)
            with self.assertRaisesRegex(ValueError,'failed output'):self.complete(lease)
            with self.assertRaises(ValueError):self.ledger.settle_output(lease,token,actual_bytes=9,sha256='d'*64)
            self.ledger.interrupt(lease,'WRITE_INTERRUPTION')
        s=self.ledger.snapshot();self.assertEqual(s['output_reserved'],100);self.assertEqual(s['read_reserved'],321)

    def test_read_output_and_model_limits_fail_before_new_reservation(self):
        with self.ledger.writer():
            self.init();lease=self.ledger.claim_task('inputs','inputs')
            for column,value,operation in (
                ('read_reserved',LIMITS['max_read_bytes'],lambda:self.ledger.reserve_read(1,lease)),
                ('output_reserved',LIMITS['max_output_bytes'],lambda:self.ledger.reserve_output(1,lease,kind='ARRAY')),
                ('entered_models',LIMITS['max_model_constructions'],lambda:self.ledger.enter_model(lease,job_id=self.job,checkpoint_sha256=self.checkpoint))):
                self.ledger.connection.execute(f'UPDATE phase SET {column}=? WHERE id=1',(value,))
                with self.subTest(column=column),self.assertRaises(ExtensionAllowanceExhausted):operation()
            self.ledger.interrupt(lease,'SYNTHETIC_ALLOWANCE_STOP')

    def test_resource_and_clock_limits_are_persistent(self):
        with self.ledger.writer():
            self.init();lease=self.ledger.claim_task('inputs','inputs')
            self.ledger.check_resources(combined_rss_bytes=100,mps_bytes=0,free_bytes=LIMITS['min_free_bytes'])
            for kw in (dict(combined_rss_bytes=LIMITS['max_rss_bytes']+1,mps_bytes=0,free_bytes=LIMITS['min_free_bytes']),
                       dict(combined_rss_bytes=100,mps_bytes=LIMITS['max_mps_bytes']+1,free_bytes=LIMITS['min_free_bytes']),
                       dict(combined_rss_bytes=100,mps_bytes=0,free_bytes=LIMITS['min_free_bytes']-1)):
                with self.assertRaises(ExtensionAllowanceExhausted):self.ledger.check_resources(**kw)
            self.now[0]=999.
            with self.assertRaisesRegex(ValueError,'backward'):self.ledger.reserve_read(1,lease)
            self.now[0]=1000.+21600
            with self.assertRaisesRegex(ExtensionAllowanceExhausted,'deadline'):self.ledger.reserve_read(1,lease)
            self.ledger.interrupt(lease,'EXPIRED')
        self.now[0]+=1
        with self.ledger.writer(),self.assertRaises(ExtensionAllowanceExhausted):self.init()

    def test_writer_loss_is_detected_without_restarting_or_refunding(self):
        self.directory.mkdir()
        pp=Path(self.temp.name)/'plan.json';pp.write_text(json.dumps(self.plan))
        code="""import json,os,sys
from hydra_directflow.coverage_extension_ledger import PhaseALedger
ledger=PhaseALedger(sys.argv[1],clock=lambda:1000.)
with ledger.writer():
 ledger.initialize(json.load(open(sys.argv[2])),77)
 ledger.claim_task('orphan','clean')
 ledger.reserve_read(50)
 os._exit(0)
"""
        result=subprocess.run([sys.executable,'-B','-c',code,str(self.directory),str(pp)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        with self.ledger.writer():
            self.init();s=self.ledger.snapshot();self.assertEqual(s['tasks'][0]['state'],'INTERRUPTED')
            self.assertEqual(s['read_reserved'],50);self.assertEqual(s['infrastructure_resumes'],0)
            with self.assertRaises(ValueError):self.ledger.claim_task('next','clean')

    def test_unclassified_exception_is_not_silently_infrastructure_resume(self):
        with self.assertRaisesRegex(RuntimeError,'synthetic implementation error'):
            with self.ledger.writer():
                self.init();self.ledger.claim_task('broken','clean')
                raise RuntimeError('synthetic implementation error')
        self.assertEqual(self.ledger.snapshot()['tasks'][0]['state'],'FAILED_IMPLEMENTATION')

    def test_unclassified_source_oserror_cannot_become_automatic_infrastructure_resume(self):
        with self.assertRaises(FileNotFoundError):
            with self.ledger.writer():
                self.init();self.ledger.claim_task('missing_source','inputs')
                raise FileNotFoundError('synthetic missing source')
        self.assertEqual(self.ledger.snapshot()['tasks'][0]['state'],'FAILED_IMPLEMENTATION')


if __name__=='__main__':unittest.main()
