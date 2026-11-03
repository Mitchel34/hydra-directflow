"""Synthetic SQLite/OS-lease accounting. No Torch, study values or fitting."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hydra_directflow.campaign_ledger import (CampaignLedger, AttemptLease, LedgerError,
    LedgerBusy, AllowanceExhausted, LIMITS, PHASES, GAUGES, ARCHITECTURES, ARMS,
    MAX_UPDATES, MAX_JOB_UPDATES, digest, validate_plan)


def plan(*,logical=2,physical=4,global_updates=720,output_bytes=100):
    jobs=[]
    for fold,seed in PHASES:
        for gauge in GAUGES:
            for architecture in ARCHITECTURES:
                for arm,length in (ARMS if fold=='final_development' else ARMS[:2]):
                    jobs.append(dict(job_id=f'{fold}__{gauge}__{architecture}__{arm}_L{length}__seed{seed}',
                        order=len(jobs),fold=fold,seed=seed,gauge_id=gauge,architecture=architecture,
                        input_arm=arm,history_hours=length,device='cpu' if architecture in ('LSTM','GRU_Transformer') else 'mps',
                        batch_size=64,precision='FP32',logical_update_limit=logical,max_updates=physical,
                        gradient_batches_per_epoch=1,checkpoint_interval_epochs=10,max_resumes=3,
                        role_case_ids_sha256={k:digest([fold,gauge,k]) for k in ('gradient_fit','inner_stopping','outer_scoring')}))
    value=dict(schema='synthetic-campaign/v1',protocol_sha256='a'*64,source_plan_sha256='b'*64,
        snapshot_sha256='c'*64,implementation_sha256={'fixture':'d'*64},jobs=jobs,
        limits={**LIMITS,'max_optimizer_updates':global_updates,'max_job_seconds':100,
                'max_campaign_seconds':1000,'max_output_bytes':output_bytes},protected_assessment_access=False)
    value['plan_sha256']=digest(value); return value


def repin(value):
    value.pop('plan_sha256',None); value['plan_sha256']=digest(value); return value


def evidence(ledger,lease,updates,checkpoint=False):
    return dict(**ledger.receipt_bindings(lease),status='VERIFIED_CHECKPOINT' if checkpoint else 'VERIFIED_TRAIN_EVAL',
        logical_updates=updates,hashes={k:digest([lease.job_id,k,updates]) for k in
            (('checkpoint',) if checkpoint else ('checkpoint','predictions','metrics'))})


def authenticated_fixture(receipt,bindings):
    # Synthetic integration boundary only. Real caller verifies artifact files,
    # checkpoint/prediction/metric content and these bindings before returning.
    assert all(receipt[k]==v for k,v in bindings.items())
    return digest(receipt)


class CampaignLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name).resolve(); self.path=self.root/'ledger'
        self.now=1000.; self.ledger=CampaignLedger(self.path,clock=lambda:self.now)
        self.value=plan(); self.ledger.initialize(self.value)

    def tearDown(self): self.temp.cleanup()

    def update(self,lease):
        token=self.ledger.enter_update(lease); self.ledger.confirm_update(lease,token); return token

    def test_initialization_is_immutable_and_clock_starts_only_on_claim(self):
        self.assertIsNone(self.ledger.snapshot()['campaign_started_at'])
        self.ledger.initialize(deepcopy(self.value))
        changed=deepcopy(self.value); changed['limits']['max_job_seconds']+=1
        with self.assertRaisesRegex(LedgerError,'cannot be reset'): self.ledger.initialize(repin(changed))
        with self.ledger.writer():
            lease=self.ledger.claim_next(); self.assertEqual(lease.job_id,self.value['jobs'][0]['job_id'])
            self.assertEqual(self.ledger.snapshot()['campaign_deadline'],2000.)

    def test_sqlite_is_wal_full_and_precharge_visible_before_optimizer(self):
        with self.ledger.writer():
            self.assertEqual(self.ledger.connection.execute('PRAGMA journal_mode').fetchone()[0],'wal')
            self.assertEqual(self.ledger.connection.execute('PRAGMA synchronous').fetchone()[0],2)
            lease=self.ledger.claim_next(); token=self.ledger.enter_update(lease)
            observer=CampaignLedger(self.path).snapshot()
            self.assertEqual((observer['optimizer_updates_entered'],observer['optimizer_updates_confirmed']),(1,0))
            self.ledger.confirm_update(lease,token)
            with self.assertRaisesRegex(LedgerError,'already settled'): self.ledger.confirm_update(lease,token)
            self.assertEqual(self.ledger.snapshot()['optimizer_updates_confirmed'],1)

    def test_unconfirmed_entry_survives_resume_without_refund_or_new_fit(self):
        with self.ledger.writer():
            first=self.ledger.claim_next(); self.update(first)
            checkpoint=evidence(self.ledger,first,1,True); self.ledger.checkpoint(first,checkpoint,validator=authenticated_fixture)
            self.ledger.enter_update(first)
            with self.assertRaisesRegex(LedgerError,'Unconfirmed'): self.ledger.enter_update(first)
            self.ledger.interrupt(first,'CONNECTION_INTERRUPTED')
        snap=self.ledger.snapshot(); self.assertEqual((snap['optimizer_updates_entered'],snap['optimizer_updates_confirmed']),(2,1))
        started=snap['jobs'][0]['started']; deadline=snap['jobs'][0]['deadline']
        with self.ledger.writer():
            with self.assertRaisesRegex(LedgerError,'explicit same-job'): self.ledger.claim_next()
            second=self.ledger.resume(first.job_id,checkpoint,validator=authenticated_fixture)
            self.assertNotEqual(first.attempt_id,second.attempt_id)
            with self.assertRaisesRegex(LedgerError,'Stale'): self.ledger.confirm_update(first,1)
            self.update(second); self.ledger.complete(second,evidence(self.ledger,second,2),validator=authenticated_fixture)
            snap=self.ledger.snapshot()
            self.assertEqual((snap['optimizer_updates_entered'],snap['optimizer_updates_confirmed']),(3,2))
            self.assertEqual((snap['jobs'][0]['started'],snap['jobs'][0]['deadline']),(started,deadline))
            self.assertEqual(snap['attempts'][0]['error_type'],'CONNECTION_INTERRUPTED')
            self.assertEqual(self.ledger.claim_next().job_id,self.value['jobs'][1]['job_id'])

    def test_context_release_interrupts_and_stale_lease_cannot_act(self):
        with self.ledger.writer(): lease=self.ledger.claim_next()
        self.assertEqual(self.ledger.snapshot()['attempts'][0]['error_type'],'LEASE_RELEASED')
        with self.assertRaisesRegex(LedgerError,'writer lease'): self.ledger.enter_update(lease)
        with self.ledger.writer():
            with self.assertRaisesRegex(LedgerError,'Stale'): self.ledger.enter_update(lease)

    def test_abrupt_worker_exit_retains_precharge_and_is_not_automatically_retried(self):
        # Separate process dies without context cleanup, simulating a lost lease.
        code='''import os,sys
from hydra_directflow.campaign_ledger import CampaignLedger
with CampaignLedger(sys.argv[1]).writer() as ledger:
 lease=ledger.claim_next(); ledger.enter_update(lease); os._exit(0)
'''
        result=subprocess.run([sys.executable,'-B','-c',code,str(self.path)],capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr.decode())
        self.assertEqual(self.ledger.snapshot()['optimizer_updates_entered'],1)
        with self.ledger.writer():
            snap=self.ledger.snapshot(); self.assertEqual(snap['jobs'][0]['state'],'INTERRUPTED')
            self.assertEqual(snap['attempts'][0]['error_type'],'LOST_LEASE')
            with self.assertRaisesRegex(LedgerError,'explicit same-job'): self.ledger.claim_next()

    def test_single_writer_and_symlinked_database_are_rejected(self):
        with self.ledger.writer():
            other=CampaignLedger(self.path)
            with self.assertRaises(LedgerBusy):
                with other.writer(): pass
        target=self.root/'target'; target.write_text('unchanged')
        linked=self.root/'bad'; linked.mkdir(); (linked/'campaign.sqlite3').symlink_to(target)
        with self.assertRaisesRegex(LedgerError,'never links'): CampaignLedger(linked).initialize(self.value)
        self.assertEqual(target.read_text(),'unchanged')

    def test_complete_requires_artifact_validator_and_every_identity_hash(self):
        with self.ledger.writer():
            lease=self.ledger.claim_next(); self.update(lease); receipt=evidence(self.ledger,lease,1)
            for name in ('protocol_sha256','snapshot_sha256','source_plan_sha256','job_sha256','job_id'):
                wrong=deepcopy(receipt); wrong[name]='wrong'
                with self.subTest(field=name),self.assertRaisesRegex(LedgerError,'identity differs'):
                    self.ledger.complete(lease,wrong,validator=authenticated_fixture)
            wrong=deepcopy(receipt); wrong['hashes'].pop('metrics')
            with self.assertRaisesRegex(LedgerError,'artifact hashes'): self.ledger.complete(lease,wrong,validator=authenticated_fixture)
            with self.assertRaisesRegex(LedgerError,'mandatory'): self.ledger.complete(lease,receipt,validator=None)
            with self.assertRaisesRegex(LedgerError,'authenticate'): self.ledger.complete(lease,receipt,validator=lambda *x:True)
            self.assertEqual(self.ledger.snapshot()['jobs'][0]['state'],'ACTIVE')
            self.ledger.complete(lease,receipt,validator=authenticated_fixture)
            with self.assertRaisesRegex(LedgerError,'not active'): self.ledger.complete(lease,receipt,validator=authenticated_fixture)

    def test_resume_requires_recorded_checkpoint_and_failure_never_autoretries(self):
        with self.ledger.writer():
            lease=self.ledger.claim_next(); self.update(lease); cp=evidence(self.ledger,lease,1,True)
            self.ledger.interrupt(lease)
            with self.assertRaisesRegex(LedgerError,'never authenticated'): self.ledger.resume(lease.job_id,cp,validator=authenticated_fixture)
        other=CampaignLedger(self.root/'failure',clock=lambda:self.now); other.initialize(self.value)
        with other.writer():
            lease=other.claim_next(); other.fail(lease,'NONFINITE')
            with self.assertRaisesRegex(LedgerError,'explicit same-job'): other.claim_next()
            with self.assertRaisesRegex(LedgerError,'failed/integrity'): other.resume(lease.job_id,cp,validator=authenticated_fixture)
        self.assertEqual(other.snapshot()['attempts'][0]['error_type'],'NONFINITE')

    def test_initial_checkpoint_and_exact_three_explicit_resumes(self):
        with self.ledger.writer():
            lease=self.ledger.claim_next(); cp=evidence(self.ledger,lease,0,True)
            self.ledger.checkpoint(lease,cp,validator=authenticated_fixture)
            for _ in range(3):
                self.ledger.interrupt(lease)
                lease=self.ledger.resume(lease.job_id,cp,validator=authenticated_fixture)
            self.ledger.interrupt(lease)
            with self.assertRaisesRegex(LedgerError,'resume allowance'): self.ledger.resume(lease.job_id,cp,validator=authenticated_fixture)
        self.assertEqual(len(self.ledger.snapshot()['attempts']),4)

    def test_logical_and_physical_allowances_are_distinct(self):
        with self.ledger.writer():
            lease=self.ledger.claim_next(); self.update(lease); self.update(lease)
            with self.assertRaisesRegex(LedgerError,'Logical'): self.ledger.enter_update(lease)
            self.assertEqual(self.ledger.snapshot()['optimizer_updates_entered'],2)
        data=plan(logical=2,physical=2,global_updates=1)
        other=CampaignLedger(self.root/'global',clock=lambda:self.now); other.initialize(data)
        with other.writer():
            lease=other.claim_next(); token=other.enter_update(lease); other.confirm_update(lease,token)
            with self.assertRaises(AllowanceExhausted): other.enter_update(lease)
            other.complete(lease,evidence(other,lease,1),validator=authenticated_fixture)
            with self.assertRaises(AllowanceExhausted): other.claim_next()

    def test_time_and_resource_bounds_do_not_reset(self):
        with self.ledger.writer():
            lease=self.ledger.claim_next(); cp=evidence(self.ledger,lease,0,True)
            self.ledger.checkpoint(lease,cp,validator=authenticated_fixture)
            self.ledger.check_resources(lease,free_bytes=8*1024**3,combined_rss_bytes=4*1024**3)
            for free,rss in ((8*1024**3-1,0),(8*1024**3,4*1024**3+1)):
                with self.assertRaises(AllowanceExhausted): self.ledger.check_resources(lease,free_bytes=free,combined_rss_bytes=rss)
            self.now=999
            with self.assertRaisesRegex(LedgerError,'backwards'): self.ledger.enter_update(lease)
            self.now=1100
            with self.assertRaises(AllowanceExhausted): self.ledger.enter_update(lease)
            self.ledger.interrupt(lease)
            with self.assertRaises(AllowanceExhausted): self.ledger.resume(lease.job_id,cp,validator=authenticated_fixture)
        self.assertEqual(self.ledger.snapshot()['jobs'][0]['deadline'],1100.)

    def test_output_reservations_include_global_bookkeeping_and_failed_partials(self):
        with self.ledger.writer():
            global_token=self.ledger.reserve_output(20,kind='BOOKKEEPING')
            self.assertEqual(self.ledger.snapshot()['bookkeeping_bytes_reserved'],20)
            lease=self.ledger.claim_next(); token=self.ledger.reserve_output(70,lease,kind='CHECKPOINT')
            self.ledger.settle_output(lease,token,actual_bytes=3,failed=True)
            self.assertEqual(self.ledger.snapshot()['output_bytes_reserved'],90)
            with self.assertRaises(AllowanceExhausted): self.ledger.reserve_output(11,lease,kind='PREDICTIONS')
            last=self.ledger.reserve_output(10,lease,kind='METRICS'); self.update(lease)
            with self.assertRaisesRegex(LedgerError,'Unsettled'): self.ledger.complete(lease,evidence(self.ledger,lease,1),validator=authenticated_fixture)
            self.ledger.settle_output(lease,last,actual_bytes=10,sha256='a'*64)
            self.ledger.complete(lease,evidence(self.ledger,lease,1),validator=authenticated_fixture)
            self.ledger.settle_output(None,global_token,actual_bytes=1,sha256='b'*64)
            self.assertEqual(self.ledger.snapshot()['output_bytes_reserved'],100)

    def test_seed_barriers_require_every_exact_job_training_and_evaluation(self):
        with self.ledger.writer():
            for i in range(180):
                lease=self.ledger.claim_next(); self.assertEqual(lease.job_id,self.value['jobs'][i]['job_id'])
                self.update(lease)
                if i in (107,143,179):
                    expected=[] if i==107 else [17] if i==143 else [17,29]
                    self.assertEqual(self.ledger.snapshot()['seed_reports_ready'],expected)
                    with self.assertRaisesRegex(LedgerError,'explicit same-job'): self.ledger.claim_next()
                self.ledger.complete(lease,evidence(self.ledger,lease,1),validator=authenticated_fixture)
            self.assertIsNone(self.ledger.claim_next())
            self.assertEqual(self.ledger.snapshot()['seed_reports_ready'],[17,29,43])
        self.assertTrue(self.ledger.snapshot()['complete'])

    def test_plan_caps_geometry_and_replay_fields_are_pinned(self):
        self.assertEqual((MAX_UPDATES,MAX_JOB_UPDATES),(20_930_920,145_820))
        for field,value in (('max_resumes',4),('checkpoint_interval_epochs',11),('logical_update_limit',126801),('max_updates',145821)):
            wrong=deepcopy(self.value); wrong['jobs'][0][field]=value
            with self.subTest(field=field),self.assertRaises(LedgerError): validate_plan(repin(wrong))
        wrong=deepcopy(self.value); wrong['jobs'][0]['max_updates']=33
        with self.assertRaisesRegex(LedgerError,'replay reserve'): validate_plan(repin(wrong))
        wrong['jobs'][0]['max_updates']=32; validate_plan(repin(wrong))
        wrong=deepcopy(self.value); wrong['jobs'][0]['device']='cuda:0'
        with self.assertRaisesRegex(LedgerError,'device'): validate_plan(repin(wrong))

    def test_snapshot_does_not_settle_an_active_attempt_or_precharge(self):
        with self.ledger.writer():
            lease=self.ledger.claim_next(); self.ledger.enter_update(lease)
            first=CampaignLedger(self.path).snapshot(); second=CampaignLedger(self.path).snapshot()
            self.assertEqual(first,second); self.assertEqual(first['jobs'][0]['state'],'ACTIVE')
            self.assertEqual(first['optimizer_updates_confirmed'],0)

    def test_public_checkpoint_lookup_is_readonly_and_resume_refuses_older_state(self):
        expected=self.value['jobs'][0]['job_id']
        self.assertEqual(self.ledger.next_job()['job']['job_id'],expected)
        self.assertIsNone(self.ledger.latest_checkpoint(expected))
        with self.ledger.writer():
            lease=self.ledger.claim_next(); zero=evidence(self.ledger,lease,0,True)
            self.ledger.checkpoint(lease,zero,validator=authenticated_fixture); self.update(lease)
            one=evidence(self.ledger,lease,1,True); self.ledger.checkpoint(lease,one,validator=authenticated_fixture)
            self.assertEqual(CampaignLedger(self.path).latest_checkpoint(expected),one)
            self.assertEqual(CampaignLedger(self.path).next_job()['state'],'ACTIVE')
            self.ledger.interrupt(lease)
            with self.assertRaisesRegex(LedgerError,'no older fallback'): self.ledger.resume(expected,zero,validator=authenticated_fixture)
            lease=self.ledger.resume(expected,one,validator=authenticated_fixture)
            self.assertEqual(self.ledger.checkpoints(expected),[zero,one])

    def test_failed_checkpoint_writes_count_against_retained_artifact_limit(self):
        with self.ledger.writer():
            lease=self.ledger.claim_next()
            for _ in range(21):
                token=self.ledger.reserve_output(1,lease,kind='CHECKPOINT')
                self.ledger.settle_output(lease,token,actual_bytes=0,failed=True)
            with self.assertRaisesRegex(LedgerError,'At most21'): self.ledger.reserve_output(1,lease,kind='CHECKPOINT')
            self.assertEqual(self.ledger.snapshot()['output_bytes_reserved'],21)


if __name__=='__main__': unittest.main()
