"""Synthetic Phase A storage/resource glue; no real study or model execution."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from hydra_directflow import coverage_extension_runtime as runtime
from hydra_directflow.coverage_extension_ledger import LIMITS,CHECKPOINT_IDS,ExtensionAllowanceExhausted
from hydra_directflow.campaign_ledger import digest
from hydra_directflow.heartbeat_snapshot import read_snapshot


class FakeLedger:
    def __init__(self,plan):
        self.plan=plan;self.events=[];self.output={};self.counter=0;self.bookkeeping=64*1024**2
        self.read_reserved=0;self.failed_resource=False;self.failed_output=False
    def receipt_bindings(self,lease):return dict(phase_plan_sha256=self.plan['plan_sha256'])
    def bookkeeping_reserved_bytes(self):return self.bookkeeping
    def telemetry(self):return dict(entered_cases=1,confirmed_cases=1,entered_batches=1,confirmed_batches=1,
        entered_models=1,confirmed_models=1,read_reserved=self.read_reserved,output_reserved=sum(x['reserved'] for x in self.output.values()))
    def check_resources(self,**kw):
        self.events.append(('resources',kw))
        if self.failed_resource or kw['combined_rss_bytes']>LIMITS['max_rss_bytes'] or kw['mps_bytes']>LIMITS['max_mps_bytes'] or kw['free_bytes']<LIMITS['min_free_bytes']:
            raise ExtensionAllowanceExhausted('synthetic resource guard')
    def reserve_read(self,n,lease):self.events.append(('read',n));self.read_reserved+=n;return 1
    def reserve_output(self,n,lease,*,kind):
        self.events.append(('reserve',n,kind))
        if self.failed_output:raise ExtensionAllowanceExhausted('synthetic output guard')
        self.counter+=1;self.output[self.counter]=dict(reserved=n,status='PRECHARGED');return self.counter
    def settle_output(self,lease,token,*,actual_bytes,sha256=None,failed=False):
        self.events.append(('settle',token,failed));self.output[token].update(actual=actual_bytes,sha256=sha256,status='FAILED' if failed else 'VERIFIED')
    def enter_forward(self,lease,**kw):self.events.append(('forward_enter',kw));return 5
    def confirm_forward(self,lease,token):self.events.append(('forward_confirm',token))
    def enter_model(self,lease,**kw):self.events.append(('model_enter',kw));return 6
    def confirm_model(self,lease,token):self.events.append(('model_confirm',token))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        (runtime.ROOT/'.local').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=runtime.ROOT/'.local');self.root=Path(self.temp.name).resolve()
        self.phase=self.root/'phase';self.original=self.root/'original';self.source=self.root/'source'
        self.original.mkdir();self.source.mkdir()
        self.plan=dict(schema='SYNTHETIC_NO_REAL_EXECUTION',run_directory=str(self.phase),original_run_directory=str(self.original),
            limits=deepcopy(LIMITS),original_campaign_deadline=9999999999.,protected_assessment_access=False,paid_cloud_jobs=0,
            checkpoint_sha256_by_job={j:hashlib.sha256(j.encode()).hexdigest() for j in sorted(CHECKPOINT_IDS)},
            corpus_root=str(self.source),source_read_paths=[str(self.root/'approved.json')])
        self.plan['plan_sha256']=digest(self.plan);self.ledger=FakeLedger(self.plan)
        self.lease=SimpleNamespace(task_key='CLEAN__synthetic',stage='clean',attempt_id=1)
        self.store=runtime.PhaseStore(self.plan,self.ledger,self.lease,self.original)
        self.store.last_heartbeat=time.monotonic()
    def tearDown(self):self.temp.cleanup()

    @contextmanager
    def resources(self,*,own=100,children=(),supervisor=500,free=20*1024**3):
        processes={pid:SimpleNamespace(pid=pid,memory_info=lambda n=rss:SimpleNamespace(rss=n)) for pid,rss in children}
        proc=SimpleNamespace(pid=11,memory_info=lambda:SimpleNamespace(rss=own),children=lambda recursive:list(processes.values()))
        def lookup(pid=None):
            if pid is None:return proc
            if pid==22:return SimpleNamespace(pid=22,memory_info=lambda:SimpleNamespace(rss=supervisor))
            raise runtime.psutil.NoSuchProcess(pid)
        with patch.object(runtime.psutil,'Process',side_effect=lookup), \
             patch.object(runtime.shutil,'disk_usage',return_value=SimpleNamespace(free=free)):
            yield

    def test_bookkeeping_and_safe_task_namespace_required_before_creation(self):
        self.ledger.bookkeeping=0
        lease=SimpleNamespace(task_key='NEW',stage='inputs',attempt_id=2)
        with self.assertRaisesRegex(ValueError,'bookkeeping'):runtime.PhaseStore(self.plan,self.ledger,lease,self.original)
        self.assertFalse((self.phase/'tasks/NEW').exists());self.ledger.bookkeeping=64*1024**2
        for key in ('../bad','a/b','x:y','..','a..b'):
            lease.task_key=key
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Safe single-component'):
                runtime.PhaseStore(self.plan,self.ledger,lease,self.original)

    def test_write_precharges_before_file_and_readback_is_charged(self):
        path=self.store.directory/'nested/value.json';reserve=self.ledger.reserve_output
        def before(n,lease,*,kind):
            self.assertFalse(path.exists());self.assertFalse(path.parent.exists())
            return reserve(n,lease,kind=kind)
        with self.resources(),patch.object(self.ledger,'reserve_output',side_effect=before):
            d=self.store.write('nested/value.json',b'{"a":1}',kind='JSON')
            self.assertEqual(self.store.read(d,100),b'{"a":1}')
        self.assertEqual(d['path'],'tasks/CLEAN__synthetic/attempt-000001/nested/value.json')
        self.assertEqual(self.ledger.output[1]['status'],'VERIFIED')
        self.assertEqual(self.ledger.read_reserved,2*len(b'{"a":1}'))
        self.assertEqual(d['sha256'],hashlib.sha256(b'{"a":1}').hexdigest())

    def test_output_quota_stops_before_directory_or_payload(self):
        self.ledger.failed_output=True
        with self.resources(),self.assertRaises(ExtensionAllowanceExhausted):
            self.store.write('newdir/value',b'payload',kind='ARRAY')
        self.assertFalse((self.store.directory/'newdir').exists())

    def test_partial_write_preserved_and_failed_charge_not_refunded(self):
        with self.resources(),patch.object(runtime.os,'fsync',side_effect=OSError('synthetic fsync failure')):
            with self.assertRaises(OSError):self.store.write('partial.bin',b'payload',kind='ARRAY')
        stage=self.store.directory/'partial.bin.writing';self.assertTrue(stage.exists())
        self.assertEqual(self.ledger.output[1]['reserved'],7);self.assertEqual(self.ledger.output[1]['status'],'FAILED')
        with self.resources(),self.assertRaisesRegex(ValueError,'partial immutable'):
            self.store.write('partial.bin',b'payload',kind='ARRAY')
        self.assertEqual(len(self.ledger.output),1)

    def test_existing_output_and_unsafe_paths_never_overwrite(self):
        with self.resources():
            d=self.store.write('result',b'old',kind='JSON')
            for path in ('result','../escape','nested/../../escape',str(self.store.directory/'absolute')):
                with self.subTest(path=path),self.assertRaises(ValueError):self.store.write(path,b'new',kind='JSON')
        self.assertEqual((self.phase/d['path']).read_bytes(),b'old')

    def test_descriptor_reads_refuse_escape_symlink_corruption_and_implicit_original_redirect(self):
        with self.resources():d=self.store.write('result',b'old',kind='JSON')
        for mutation in ('escape','sha','size','symlink','original'):
            bad=deepcopy(d)
            if mutation=='escape':bad['path']='../outside'
            elif mutation=='sha':bad['sha256']='0'*64
            elif mutation=='size':bad['bytes']+=1
            elif mutation=='original':
                (self.original/'original.bin').write_bytes(b'old');bad['path']='original.bin'
            else:
                link=self.phase/'link';link.symlink_to(self.phase/d['path']);bad['path']='link'
            with self.subTest(mutation=mutation),self.resources(),self.assertRaises((ValueError,FileNotFoundError)):
                self.store.read(bad,100)

    def test_original_and_explicit_source_reads_remain_separate_and_count_repeats(self):
        (self.original/'old.json').write_bytes(b'old');(self.source/'source.json').write_bytes(b'source');(self.root/'approved.json').write_bytes(b'approved')
        with self.resources():
            self.assertEqual(self.store.read_original('old.json',100,hashlib.sha256(b'old').hexdigest()),b'old')
            self.assertEqual(self.store.read_source(self.source/'source.json',100),b'source')
            self.assertEqual(self.store.read_source(self.source/'source.json',100),b'source')
            self.assertEqual(self.store.read_source(self.root/'approved.json',100),b'approved')
            with self.assertRaisesRegex(ValueError,'outside explicit'):self.store.read_source(self.original/'old.json',100)
        self.assertEqual(self.ledger.read_reserved,3+6+6+8)

    def test_cloud_only_file_rejected_before_body_or_read_charge(self):
        path=self.original/'offloaded';path.write_bytes(b'synthetic')
        actual=Path.lstat
        def info(p):
            s=actual(p)
            return SimpleNamespace(st_size=s.st_size,st_mode=s.st_mode,st_flags=0x40000000) if p==path else s
        with self.resources(),patch.object(Path,'lstat',info),patch.object(runtime,'_read',side_effect=AssertionError('Body read')):
            with self.assertRaisesRegex(ValueError,'implicit hydration'):self.store.read_original('offloaded',100)
        self.assertEqual(self.ledger.read_reserved,0)

    def test_memory_counts_worker_children_and_explicit_supervisor_without_duplicates(self):
        self.store.supervisor_pid=22
        with self.resources(own=100,children=((33,200),(22,500)),supervisor=500):self.store.check()
        self.assertEqual(self.store.last_measurement['combined_rss_mps_bytes'],800)
        with self.resources(own=100,children=((33,200),),supervisor=500):self.store.check()
        self.assertEqual(self.store.last_measurement['combined_rss_mps_bytes'],800)

    def test_cpu_path_does_not_touch_torch_or_mps(self):
        class PoisonTorch:
            def __getattr__(self,key):raise AssertionError('CPU monitor accessed Torch')
        with self.resources(),patch.dict(runtime.sys.modules,{'torch':PoisonTorch()}):self.store.check()
        self.assertEqual(self.store.last_measurement['mps_driver_bytes'],0)

    def test_mps_driver_and_allocator_caps_and_backend_identity(self):
        job=next(j for j in CHECKPOINT_IDS if '__Transformer__' in j)
        self.store.set_job_device('mps')
        fake=SimpleNamespace(mps=SimpleNamespace(driver_allocated_memory=lambda:300,current_allocated_memory=lambda:200))
        with self.resources(own=100),patch.dict(runtime.sys.modules,{'torch':fake}):
            self.store.heartbeat(job_id=job,force=True)
            self.assertEqual(self.store.last_measurement['combined_rss_mps_bytes'],400)
            fake.mps.current_allocated_memory=lambda:LIMITS['max_mps_bytes']+1
            with self.assertRaises(ExtensionAllowanceExhausted):self.store.check()
        self.store.set_job_device('cpu')
        with self.resources(),self.assertRaisesRegex(ValueError,'backend differs'):self.store.check()

    def test_mutable_heartbeat_is_atomic_bounded_and_rate_limited(self):
        self.store.last_heartbeat=-float('inf')
        with self.resources(),patch.object(runtime.time,'monotonic',return_value=1000):
            self.store.heartbeat(stage='PREPARING',force=True)
            first=self.store.heartbeat_path.read_bytes();self.store.heartbeat(stage='UPDATED')
            self.assertEqual(self.store.heartbeat_path.read_bytes(),first)
        with self.resources(),patch.object(runtime.time,'monotonic',return_value=1002.1):self.store.heartbeat(stage='UPDATED')
        value=json.loads(read_snapshot(self.store.heartbeat_path))
        self.assertEqual(value['stage'],'UPDATED');self.assertEqual(value['phase_plan_sha256'],self.plan['plan_sha256'])
        self.assertEqual(value['attempt_id'],1);self.assertLess(self.store.heartbeat_path.stat().st_size,65536)
        self.assertEqual(self.ledger.output,{}) # Already-covered bookkeeping, no duplicate per-heartbeat charges.

    def test_resource_failure_prevents_artifact_or_forward_operation(self):
        self.ledger.failed_resource=True
        with self.resources():
            with self.assertRaises(ExtensionAllowanceExhausted):self.store.write('never',b'value',kind='JSON')
            with self.assertRaises(ExtensionAllowanceExhausted):self.store.precharge_forward(case_count=1,batch_count=1,stage='clean',case_ids_sha256='a'*64)
        self.assertFalse((self.store.directory/'never').exists());self.assertFalse(any(x[0]=='forward_enter' for x in self.ledger.events))

    def test_model_forward_callbacks_keep_lease_and_backend(self):
        job=next(j for j in CHECKPOINT_IDS if '__LSTM__' in j)
        with self.resources():
            token=self.store.precharge_model(job_id=job,checkpoint_sha256=self.plan['checkpoint_sha256_by_job'][job]);self.store.confirm_model(token)
            token=self.store.precharge_forward(case_count=2,batch_count=1,stage='clean',case_ids_sha256='a'*64);self.store.confirm_forward(token)
        self.assertEqual(self.store.job_device,'cpu');self.assertEqual(self.store.job_id,job)
        self.assertEqual([x[0] for x in self.ledger.events if x[0] in ('model_enter','model_confirm','forward_enter','forward_confirm')],
                         ['model_enter','model_confirm','forward_enter','forward_confirm'])

    def test_descriptor_verifier_does_not_follow_links_inside_json_bodies(self):
        body=json.dumps({'old_descriptor':{'path':'unread-original.pt','bytes':999,'sha256':'a'*64}}).encode()
        with self.resources():
            d=self.store.write('manifest.json',body,kind='JSON')
            result=runtime.verify_descriptors({'artifact':d,'duplicate':d},self.store)
            self.assertEqual(result['files'],1)
            with self.assertRaisesRegex(ValueError,'Malformed descriptor'):
                runtime.verify_descriptors({'good':d,'bad':{**d,'extra':True}},self.store)


if __name__=='__main__':unittest.main()
