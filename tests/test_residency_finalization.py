"""Synthetic finalization guards only; no real corpus or model work."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().parents[1]/'scripts/finalize_development_residency.py'
SPEC=importlib.util.spec_from_file_location('finalization_tests',PATH)
final=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(final)
r=final.recovery


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.old=self.root/'old-control';self.new=self.old/'finalization-v1'
        self.source=self.root/'source';self.target=self.root/'target'
        for path in (self.old,self.new,self.new/'partials',self.source,self.target):r.directory(path,True)
        self.jobs=[]
        for number in range(2):
            body=('synthetic opaque '+str(number)).encode();source=self.source/f'{number}.npz';source.write_bytes(body)
            self.jobs.append(dict(relative=f'{number}.npz',source=str(source),source_root=str(self.source),target=str(self.target/f'{number}.npz'),
                                 expected_bytes=len(body),expected_sha256=r.sha(body),kind='DEVELOPMENT_MODEL_NPZ'))
        self.context=dict(schema='synthetic',finalizer_sha256=final.self_sha(),copier_sha256=r.sha(Path(r.__file__).read_bytes()),
                          target_root=str(self.target),expected_files=2,body_cap=final.MAX_BODY,deadline_epoch=time.time()+60)
        self.plan=dict(original_control=str(self.old),remaining_files=self.jobs,target_root=str(self.target))
        self.db=r.connect(self.new);r.initialize(self.db,self.context,self.jobs)
    def tearDown(self):self.db.close();self.temp.cleanup()

    def test_resident_confirmation_precedes_any_plan_or_source_access(self):
        with patch.object(final,'load_authority',side_effect=AssertionError('Read before source-readiness confirmation')):
            with self.assertRaisesRegex(ValueError,'resident source readiness'):final.execute('unused','unused','a'*64,'unused',False)

    def test_both_sources_must_be_resident_before_creating_allowance(self):
        with patch.object(final,'load_authority',return_value=(self.plan,self.jobs,100)),\
             patch.object(final,'resident_sources',return_value=[True,False]),\
             patch.object(r,'lease',side_effect=AssertionError('Created allowance before residency')):
            with self.assertRaisesRegex(ValueError,'Both sources'):
                final.execute('unused','unused',final.self_sha(),self.root/'report.json',True)

    def test_resident_copy_has_its_own_reservation_and_preserves_source(self):
        job=self.jobs[0];body=Path(job['source']).read_bytes();token=r.reserve(self.db,job['relative'])
        with r.lease(self.old) as old_fd,r.lease(self.new) as new_fd,\
             patch.object(final,'context_authority',return_value=(self.context,self.plan,self.jobs)):
            final.copy_worker(self.new,job['relative'],token,new_fd,old_fd)
        self.assertEqual(Path(job['target']).read_bytes(),body);self.assertEqual(Path(job['source']).read_bytes(),body)
        self.assertEqual(self.db.execute('SELECT charged FROM context').fetchone()[0],len(body)+1)
        self.assertEqual(self.db.execute('SELECT status FROM files WHERE relative=?',(job['relative'],)).fetchone()[0],'VERIFIED')

    def test_changed_residency_rejects_before_worker_claim_or_source_read(self):
        token=r.reserve(self.db,self.jobs[0]['relative'])
        with r.lease(self.old) as old_fd,r.lease(self.new) as new_fd,\
             patch.object(final,'context_authority',return_value=(self.context,self.plan,self.jobs)),\
             patch.object(final,'resident_sources',return_value=[False,True]),\
             patch.object(r,'claim',side_effect=AssertionError('Claimed unavailable source')):
            with self.assertRaisesRegex(ValueError,'Both sources'):
                final.copy_worker(self.new,self.jobs[0]['relative'],token,new_fd,old_fd)
        self.assertEqual(self.db.execute('SELECT status FROM files WHERE relative=?',(self.jobs[0]['relative'],)).fetchone()[0],'PRECHARGED')

    def test_copy_replay_and_existing_target_are_not_overwritten(self):
        job=self.jobs[0];token=r.reserve(self.db,job['relative'])
        with r.lease(self.old) as old_fd,r.lease(self.new) as new_fd,patch.object(final,'context_authority',return_value=(self.context,self.plan,self.jobs)):
            final.copy_worker(self.new,job['relative'],token,new_fd,old_fd)
            with self.assertRaisesRegex(ValueError,'replayed'):final.copy_worker(self.new,job['relative'],token,new_fd,old_fd)
        job=self.jobs[1];Path(job['target']).write_bytes(b'preserve old target');token=r.reserve(self.db,job['relative'])
        with r.lease(self.old) as old_fd,r.lease(self.new) as new_fd,patch.object(final,'context_authority',return_value=(self.context,self.plan,self.jobs)):
            with self.assertRaisesRegex(ValueError,'overwrite'):final.copy_worker(self.new,job['relative'],token,new_fd,old_fd)
        self.assertEqual(Path(job['target']).read_bytes(),b'preserve old target')

    def test_integrity_pass_cannot_enter_before_every_target_is_resident(self):
        with r.lease(self.old) as old_fd,r.lease(self.new) as new_fd,\
             patch.object(final,'context_authority',return_value=(self.context,self.plan,self.jobs)),\
             patch.object(final,'old_evidence',return_value={}),patch.object(r,'verify_worker') as verify:
            with self.assertRaises(FileNotFoundError):final.verify_worker(self.new,new_fd,old_fd)
            verify.assert_not_called()

    def test_integrity_delegates_only_after_local_target_and_old_history_gates(self):
        for job in self.jobs:Path(job['target']).write_bytes(Path(job['source']).read_bytes())
        with r.lease(self.old) as old_fd,r.lease(self.new) as new_fd,\
             patch.object(final,'context_authority',return_value=(self.context,self.plan,self.jobs)),\
             patch.object(final,'old_evidence',return_value={}) as old_check,patch.object(r,'verify_worker') as verify:
            final.verify_worker(self.new,new_fd,old_fd)
            old_check.assert_called_once();verify.assert_called_once_with(self.new,new_fd)

    def test_old_history_is_read_only_and_pending_integrity_is_mandatory(self):
        old_db=r.connect(self.old);context=dict(deadline_epoch=time.time()-300,plan_file_sha256='a'*64,copier_sha256='b'*64)
        jobs=[dict(relative=f'old-{i}',expected_sha256='c'*64) for i in range(1430)]+[dict(relative=relative,expected_sha256='d'*64) for relative in final.REMAINING]
        r.initialize(old_db,context,jobs)
        with old_db:
            old_db.execute("UPDATE files SET status='VERIFIED'")
            old_db.executemany("UPDATE files SET status='FAILED',error_type='ValueError' WHERE relative=?",[(x,) for x in final.REMAINING])
            old_db.execute('UPDATE context SET charged=123')
        rows=list(old_db.execute('SELECT * FROM files ORDER BY relative'));logical=[]
        for row in rows:
            job=json.loads(row['data']);logical.append(dict(relative=row['relative'],status=row['status'],expected_sha256=job['expected_sha256'],reserved=0,observed=0,error_type=row['error_type']))
        plan=dict(original_control=str(self.old),original_deadline_epoch=context['deadline_epoch'],original_recovery_plan_file_sha256='a'*64,original_copier_sha256='b'*64,
                  original_logical_ledger_state_sha256=r.sha(r.canonical(logical)),remaining_files=jobs[-2:],budget={'first_wave_reserved':123})
        before=old_db.execute('SELECT data,charged,verification FROM context').fetchone()
        self.assertEqual(final.old_evidence(plan)['charged'],123)
        self.assertEqual(tuple(before),tuple(old_db.execute('SELECT data,charged,verification FROM context').fetchone()))
        with old_db:old_db.execute("UPDATE context SET verification='CLAIMED'")
        with self.assertRaisesRegex(ValueError,'integrity'):final.old_evidence(plan)
        old_db.close()

    def test_scope_constants_distinguish_source_copy_cap_from_integrity_reads(self):
        self.assertEqual(final.MAX_BODY,2*1024**2);self.assertEqual(final.MAX_SECONDS,120)
        self.assertEqual(final.MAX_COMBINED_SOURCE_BYTES,1024**3);self.assertEqual(final.MAX_RSS,4*1024**3)
        self.assertEqual(len(final.REMAINING),2);self.assertNotEqual(final.MAX_BODY,75805012)
        self.assertEqual(final.PLAN_CONTENT_SHA,'2b024be358b4047cc52676c731db9f34730352b2b2554120657538c386864a53')

    def test_exact_combined_source_byte_boundary_and_new_source_cap(self):
        new=206470;maximum=1024**3
        self.assertEqual(final.combined_source_reservation(maximum-new,new),maximum)
        with self.assertRaisesRegex(ValueError,'1GiB'):final.combined_source_reservation(maximum-new+1,new)
        with self.assertRaises(ValueError):final.combined_source_reservation(0,2*1024**2+1)
        with self.assertRaises(ValueError):final.combined_source_reservation(-1,new)


if __name__=='__main__':unittest.main()
