"""Synthetic opaque-file/ledger tests; no corpus values, network or models."""
from copy import deepcopy
import importlib.util
import json
import os
import signal
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().parents[1]/'scripts/recover_development_residency.py'
SPEC=importlib.util.spec_from_file_location('residency_recovery_tests',PATH)
recovery=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(recovery)


def manifest():
    source=Path('/synthetic-owned-cloud/development-corpus-L168-v4')
    local=Path.home()/'.codex/hydra-directflow-runtime'
    target=local/'synthetic/development-corpus-L168-v4';cache=local/'synthetic/metadata'
    rows=[]
    def row(relative,size,kind):
        rows.append(dict(relative_path=relative,source=str(source/relative),destination=str(target/relative),bytes=size,sha256='a'*64,
                         kind=kind,verified_metadata_cache=None if kind=='DEVELOPMENT_MODEL_NPZ' else str(cache/relative)))
    row('PLAN.json',2,'PLAN_METADATA')
    size,remainder=divmod(75805012-955*2,477)
    for index in range(477):
        base=f'chunks/synthetic/{index:03d}/'
        row(base+'COMMIT.json',2,'COMMIT_METADATA');row(base+'case-index.json',2,'CASE_INDEX_METADATA')
        row(base+'model.npz',size+(remainder if index==476 else 0),'DEVELOPMENT_MODEL_NPZ')
    result=dict(schema='hydra-qualified-development-corpus-residency-recovery-plan/v1',status='PLAN_ONLY_MODEL_PAYLOAD_COPY_NOT_EXECUTED',
        source_plan_sha256=recovery.SOURCE_PLAN,snapshot_sha256=recovery.SNAPSHOT,source_root=str(source),target_root=str(target),files=rows,
        totals={'exact_source_bytes':75805012},protection=dict(all_H18_targets_strictly_before_utc='2024-07-31T00:00:00Z',
            original36_boundary_purges_preserved=True,raw_source_archives_included=False,monthly_parquet_included=False,
            derived_cache_values_included=False,assessment_files_included=False))
    result['plan_content_sha256']=recovery.sha(recovery.canonical(result));return result


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.control=self.root/'control';self.source=self.root/'source';self.target=self.root/'target'
        for path in (self.control,self.control/'partials',self.source,self.target):recovery.directory(path,True)
        self.body=b'synthetic opaque archive bytes';self.input=self.source/'model.npz';self.input.write_bytes(self.body)
        self.job=dict(relative='model.npz',source=str(self.input),source_root=str(self.source),target=str(self.target/'model.npz'),
                      expected_bytes=len(self.body),expected_sha256=recovery.sha(self.body),kind='DEVELOPMENT_MODEL_NPZ')
        self.db=recovery.connect(self.control)
        self.context=dict(schema=recovery.SCHEMA,copier_sha256=recovery.sha(PATH.read_bytes()),target_root=str(self.target),
                          expected_files=1,body_cap=1024,deadline_epoch=time.time()+60)
        recovery.initialize(self.db,self.context,[self.job])
    def tearDown(self):self.db.close();self.temp.cleanup()

    def test_manifest_exact_whitelist_bytes_and_outside_cloud_target(self):
        value=manifest();jobs=recovery.validate_manifest(value)
        self.assertEqual(len(jobs),1432);self.assertEqual(sum(j['expected_bytes'] for j in jobs),75805012)
        self.assertEqual(sum(j['kind']=='DEVELOPMENT_MODEL_NPZ' for j in jobs),477)
        self.assertTrue(all('metadata' in j['source'] for j in jobs if j['kind']!='DEVELOPMENT_MODEL_NPZ'))

    def test_manifest_protected_paths_duplicates_and_changed_authority_rejected(self):
        for mutation in ('snapshot','assessment','path','duplicate'):
            value=manifest()
            if mutation=='snapshot':value['snapshot_sha256']='f'*64
            elif mutation=='assessment':value['protection']['assessment_files_included']=True
            elif mutation=='path':value['files'][0]['relative_path']='../assessment.parquet'
            else:value['files'][1]=deepcopy(value['files'][0])
            value['plan_content_sha256']=recovery.sha(recovery.canonical({k:v for k,v in value.items() if k!='plan_content_sha256'}))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):recovery.validate_manifest(value)

    def test_durable_precharge_and_claim_precede_source_open_and_publication(self):
        token=recovery.reserve(self.db,'model.npz');real_open=recovery.os.open;observed=[]
        def checked(path,*args,**kwargs):
            if Path(path)==self.input:
                row=self.db.execute('SELECT status,reserved FROM files').fetchone();observed.append(row['status'])
                self.assertEqual(row['status'],'CLAIMED');self.assertEqual(row['reserved'],len(self.body)+1)
            return real_open(path,*args,**kwargs)
        with recovery.lease(self.control) as fd,patch.object(recovery.os,'open',side_effect=checked):
            recovery.copy_worker(self.control,'model.npz',token,fd)
        self.assertEqual(observed,['CLAIMED']);self.assertEqual((self.target/'model.npz').read_bytes(),self.body)
        self.assertEqual(self.db.execute('SELECT status FROM files').fetchone()[0],'VERIFIED')
        self.assertEqual(self.input.read_bytes(),self.body)

    def test_budget_failure_does_not_claim_or_open_source(self):
        context={**self.context,'body_cap':len(self.body)}
        with self.db:self.db.execute('UPDATE context SET data=?',(recovery.canonical(context).decode(),))
        with self.assertRaisesRegex(ValueError,'budget'):recovery.reserve(self.db,'model.npz')
        self.assertEqual(self.db.execute('SELECT charged FROM context').fetchone()[0],0)
        self.assertEqual(self.db.execute('SELECT status FROM files').fetchone()[0],'PENDING')

    def test_bad_sha_retains_partial_charge_and_forbids_second_attempt(self):
        job={**self.job,'expected_sha256':'f'*64}
        with self.db:self.db.execute('UPDATE files SET data=?',(recovery.canonical(job).decode(),))
        token=recovery.reserve(self.db,'model.npz')
        with recovery.lease(self.control) as fd,self.assertRaisesRegex(ValueError,'SHA'):
            recovery.copy_worker(self.control,'model.npz',token,fd)
        self.assertFalse((self.target/'model.npz').exists())
        self.assertEqual(len(list((self.control/'partials').iterdir())),1)
        self.assertEqual(self.db.execute('SELECT charged FROM context').fetchone()[0],len(self.body)+1)
        self.assertIsNone(recovery.reserve(self.db,'model.npz'))

    def test_unreserved_replayed_worker_and_missing_lease_reject_before_source_reads(self):
        with recovery.lease(self.control) as fd:
            with self.assertRaisesRegex(ValueError,'Unreserved'):recovery.copy_worker(self.control,'model.npz','wrong',fd)
            token=recovery.reserve(self.db,'model.npz');recovery.copy_worker(self.control,'model.npz',token,fd)
            with self.assertRaisesRegex(ValueError,'replayed'):recovery.copy_worker(self.control,'model.npz',token,fd)
        with self.assertRaises((OSError,ValueError)):recovery.verify_lease(self.control,999999)

    def test_existing_destination_never_overwritten(self):
        target=self.target/'model.npz';target.write_bytes(b'original user file')
        token=recovery.reserve(self.db,'model.npz')
        with recovery.lease(self.control) as fd,self.assertRaisesRegex(ValueError,'Fresh'):
            recovery.copy_worker(self.control,'model.npz',token,fd)
        self.assertEqual(target.read_bytes(),b'original user file')

    def test_symlink_source_and_database_rejected(self):
        alternate=self.source/'other';alternate.write_bytes(self.body);self.input.unlink();self.input.symlink_to(alternate)
        token=recovery.reserve(self.db,'model.npz')
        with recovery.lease(self.control) as fd,self.assertRaisesRegex(ValueError,'stat'):
            recovery.copy_worker(self.control,'model.npz',token,fd)
        other=self.root/'badcontrol';other.mkdir();(other/'transfer.sqlite3').symlink_to(self.control/'transfer.sqlite3')
        with self.assertRaisesRegex(ValueError,'regular transfer'):recovery.connect(other)

    def test_reconcile_crash_after_publication_without_reopening_source(self):
        token=recovery.reserve(self.db,'model.npz');recovery.claim(self.db,'model.npz',token)
        (self.target/'model.npz').write_bytes(self.body);self.input.unlink()
        self.assertTrue(recovery.reconcile_published(self.db,self.job))
        self.assertEqual(self.db.execute('SELECT status FROM files').fetchone()[0],'VERIFIED')
        self.assertEqual(self.db.execute('SELECT charged FROM context').fetchone()[0],len(self.body)+1)

    def test_integrity_pass_requires_all_verified_files_and_has_one_atomic_claim(self):
        with self.db:self.db.execute("UPDATE context SET verification='PRECHARGED'")
        with self.assertRaisesRegex(ValueError,'Incomplete'):recovery.claim_verification(self.db)
        with self.db:self.db.execute("UPDATE files SET status='VERIFIED'")
        self.assertEqual(recovery.claim_verification(self.db)['expected_files'],1)
        with self.assertRaisesRegex(ValueError,'Single integrity'):recovery.claim_verification(self.db)

    def test_resume_cannot_reset_source_budget_or_change_context(self):
        recovery.reserve(self.db,'model.npz')
        result=recovery.initialize(self.db,{**self.context,'deadline_epoch':time.time()+900},[self.job])
        self.assertEqual(result['deadline_epoch'],self.context['deadline_epoch'])
        self.assertEqual(self.db.execute('SELECT charged FROM context').fetchone()[0],len(self.body)+1)
        with self.assertRaisesRegex(ValueError,'context'):recovery.initialize(self.db,{**self.context,'body_cap':999999},[self.job])

    def test_public_summary_contains_no_private_paths_and_failed_decode_is_unknown(self):
        with self.db:self.db.execute("UPDATE context SET verification='FAILED'")
        out=recovery.summarize(self.db,self.context,'a'*64);text=json.dumps(out)
        self.assertNotIn(str(self.root),text);self.assertIsNone(out['postcopy_integrity_phase']['development_arrays_decoded'])
        self.assertFalse(out['opaque_copy_phase']['numerical_values_decoded'])

    def test_worker_local_timeout_precedes_read_and_preserves_charge_and_partial(self):
        token=recovery.reserve(self.db,'model.npz');real_open=recovery.os.open;original_handler=signal.getsignal(signal.SIGALRM)
        def timed_out(path,*args,**kwargs):
            if Path(path)==self.input:
                handler=signal.getsignal(signal.SIGALRM);self.assertTrue(callable(handler));self.assertNotEqual(handler,original_handler)
                handler(signal.SIGALRM,None)
            return real_open(path,*args,**kwargs)
        with recovery.lease(self.control) as fd,patch.object(recovery.os,'open',side_effect=timed_out),self.assertRaises(TimeoutError):
            recovery.copy_worker(self.control,'model.npz',token,fd)
        self.assertEqual(signal.getsignal(signal.SIGALRM),original_handler)
        self.assertEqual(self.db.execute('SELECT status,error_type FROM files').fetchone()['error_type'],'TimeoutError')
        self.assertEqual(self.db.execute('SELECT charged FROM context').fetchone()[0],len(self.body)+1)
        self.assertFalse((self.target/'model.npz').exists());self.assertEqual(len(list((self.control/'partials').iterdir())),1)

    def test_integrity_worker_has_its_own_deadline_before_inspection(self):
        with self.db:
            self.db.execute("UPDATE context SET verification='PRECHARGED'");self.db.execute("UPDATE files SET status='VERIFIED'")
        original_handler=signal.getsignal(signal.SIGALRM)
        def timed_out(*args,**kwargs):
            handler=signal.getsignal(signal.SIGALRM);self.assertTrue(callable(handler));self.assertNotEqual(handler,original_handler)
            handler(signal.SIGALRM,None)
        with recovery.lease(self.control) as fd,patch('hydra_directflow.corpus_dataset.inspect_corpus',side_effect=timed_out),self.assertRaises(TimeoutError):
            recovery.verify_worker(self.control,fd)
        self.assertEqual(self.db.execute('SELECT verification FROM context').fetchone()[0],'FAILED')
        self.assertEqual(signal.getsignal(signal.SIGALRM),original_handler)

    def test_private_worker_logs_are_retained_and_never_overwritten(self):
        log=self.control/'worker.log'
        with patch.object(recovery.subprocess,'Popen',side_effect=OSError('synthetic launch failure')):
            result=recovery.supervise(['unused'],fd=0,deadline=time.time()+1,max_rss=1024,log_path=log)
        self.assertEqual(result['error_type'],'OSError');self.assertGreater(result['log_bytes'],0)
        self.assertEqual(result['log_sha256'],recovery.sha(log.read_bytes()))
        with patch.object(recovery.subprocess,'Popen') as spawn,self.assertRaises(FileExistsError):
            recovery.supervise(['unused'],fd=0,deadline=time.time()+1,max_rss=1024,log_path=log)
        spawn.assert_not_called()


if __name__=='__main__':unittest.main()
