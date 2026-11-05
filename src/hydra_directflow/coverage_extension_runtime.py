"""Bounded artifact and resource glue for one Phase A worker.

The original campaign is read-only here. The separate inference ledger owns all
charges; a precharged global bookkeeping slot covers mutable heartbeats, logs
and database state. This module never imports Torch or constructs a model.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import time

import psutil

from .campaign_ledger import canonical, digest
from .corpus_dataset import _read
from .coverage_extension_ledger import validate_extension_plan
from .reference import require

ROOT=Path(__file__).resolve().parents[2]
BOOKKEEPING_BYTES=64*1024**2
MAX_ARTIFACT_BYTES=512*1024**2
HEARTBEAT_BYTES=64*1024


def _sha(raw):return hashlib.sha256(raw).hexdigest()
def _hex(value):return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None


def _directory(path):
    path=Path(path)
    require(path.absolute()==path.resolve(),'Directory aliases and symlinks are forbidden')
    if path.exists():require(path.is_dir() and not path.is_symlink(),'A real directory is required');return
    _directory(path.parent);path.mkdir(mode=0o700);_sync(path.parent)


def _sync(directory):
    fd=os.open(directory,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _relative(root,relative):
    require(isinstance(relative,str) and relative and not Path(relative).is_absolute()
            and str(Path(relative))==relative and all(p not in ('.','..') for p in Path(relative).parts),
            'Canonical relative artifact path required')
    path=root/relative
    require(path==path.resolve() and path.is_relative_to(root),'Artifact escaped its bound root')
    return path


class PhaseStore:
    def __init__(self,plan,ledger,lease,original_run_directory,*,supervisor_pid=None,job_device=None):
        self.plan=validate_extension_plan(plan);self.ledger,self.lease=ledger,lease
        require(Path(plan['run_directory']).is_absolute() and Path(original_run_directory).is_absolute(),
                'Explicit absolute phase and original roots required')
        self.run=Path(plan['run_directory']).absolute();self.original_run=Path(original_run_directory).absolute()
        require(self.run==self.run.resolve() and self.run.is_relative_to(ROOT/'.local')
                and self.original_run==self.original_run.resolve() and self.original_run.is_relative_to(ROOT/'.local')
                and self.run!=self.original_run,'Distinct stable private phase/original roots required')
        require(canonical(ledger.plan)==canonical(self.plan)
                and ledger.receipt_bindings(lease)['phase_plan_sha256']==self.plan['plan_sha256'],
                'Store must use the active Phase A plan/lease')
        if 'original_run_directory' in plan:
            require(plan['original_run_directory']==str(self.original_run),'Original run directory differs from plan')
        require(ledger.bookkeeping_reserved_bytes()>=BOOKKEEPING_BYTES,'Global64MiB bookkeeping must be precharged before workers')
        key=lease.task_key
        require(isinstance(key,str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,220}',key) and '..' not in key
                and key not in ('.','..') and type(lease.attempt_id) is int and lease.attempt_id>0,
                'Safe single-component task key and attempt required')
        self.directory=self.run/'tasks'/key/f'attempt-{lease.attempt_id:06d}'
        _directory(self.directory)
        require(supervisor_pid is None or type(supervisor_pid) is int and supervisor_pid>0 and supervisor_pid!=os.getpid(),
                'Explicit distinct supervisor PID required')
        self.supervisor_pid=supervisor_pid;self.job_device=None;self.set_job_device(job_device)
        self.stage=lease.stage;self.job_id=None;self.last_measurement=None;self.last_heartbeat=-float('inf')
        self.heartbeat_path=self.directory/'heartbeat.json'

    def set_job_device(self,device):
        require(device in (None,'cpu','mps'),'Phase A permits original CPU/MPS backends only')
        self.job_device=device

    def _set_job(self,job_id):
        require(job_id in self.plan['checkpoint_sha256_by_job'],'Only a declared checkpoint job is permitted')
        architecture=job_id.split('__')[2]
        expected='cpu' if architecture in ('LSTM','GRU_Transformer') else 'mps'
        require(self.job_device in (None,expected),'Runtime memory backend differs from the checkpoint backend')
        self.job_id=job_id;self.job_device=expected

    def _measure(self):
        own=psutil.Process();seen={own.pid};own_rss=own.memory_info().rss;child_rss=supervisor_rss=0
        for process in own.children(recursive=True):
            if process.pid in seen:continue
            try:child_rss+=process.memory_info().rss;seen.add(process.pid)
            except psutil.NoSuchProcess:pass
        supervisor_alive=self.supervisor_pid is None or self.supervisor_pid in seen
        if self.supervisor_pid is not None and self.supervisor_pid not in seen:
            try:
                supervisor=psutil.Process(self.supervisor_pid);supervisor_rss=supervisor.memory_info().rss
                supervisor_alive=True
            except psutil.NoSuchProcess:supervisor_alive=False
        driver=allocator=0
        torch=sys.modules.get('torch')
        if self.job_device=='mps' and torch is not None:
            driver=int(torch.mps.driver_allocated_memory())
            allocator=int(torch.mps.current_allocated_memory())
        require(all(type(x) is int and x>=0 for x in (own_rss,child_rss,supervisor_rss,driver,allocator)),
                'Measured resource values must be nonnegative integer bytes')
        return dict(worker_rss_bytes=own_rss,child_rss_bytes=child_rss,supervisor_rss_bytes=supervisor_rss,
                    supervisor_alive=supervisor_alive,mps_driver_bytes=driver,mps_allocator_bytes=allocator,
                    combined_rss_mps_bytes=own_rss+child_rss+supervisor_rss+driver,
                    free_disk_bytes=int(shutil.disk_usage(self.run).free))

    def _emit(self,measurement,*,force=False):
        now=time.monotonic()
        if not force and now-self.last_heartbeat<2:return
        value=dict(schema='hydra-phase-a-worker-heartbeat/v1',phase_plan_sha256=self.plan['plan_sha256'],
            task_key=self.lease.task_key,attempt_id=self.lease.attempt_id,pid=os.getpid(),
            supervisor_pid=self.supervisor_pid,stage=self.stage,job_id=self.job_id,device=self.job_device,
            observed_at_utc=datetime.now(timezone.utc).isoformat(),**measurement,**self.ledger.telemetry())
        raw=(canonical(value)+'\n').encode();require(0<len(raw)<=HEARTBEAT_BYTES,'Heartbeat exceeds bookkeeping slot')
        stage=self.heartbeat_path.with_name('heartbeat.json.writing')
        require(not stage.exists() and not stage.is_symlink(),'Unfinished heartbeat publication must be preserved')
        with stage.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.replace(stage,self.heartbeat_path);_sync(self.directory);self.last_heartbeat=now

    def check(self):
        if self.job_id is not None:self._set_job(self.job_id)
        measurement=self._measure()
        self.ledger.check_resources(combined_rss_bytes=measurement['combined_rss_mps_bytes'],
            mps_bytes=max(measurement['mps_driver_bytes'],measurement['mps_allocator_bytes']),free_bytes=measurement['free_disk_bytes'])
        self.last_measurement=measurement;self._emit(measurement)

    def heartbeat(self,stage=None,job_id=None,*,force=False):
        if stage is not None:
            require(isinstance(stage,str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,96}',stage),'Bounded heartbeat stage required')
            self.stage=stage
        if job_id is not None:
            self._set_job(job_id)
        measurement=self._measure()
        self.ledger.check_resources(combined_rss_bytes=measurement['combined_rss_mps_bytes'],
            mps_bytes=max(measurement['mps_driver_bytes'],measurement['mps_allocator_bytes']),free_bytes=measurement['free_disk_bytes'])
        self.last_measurement=measurement;self._emit(measurement,force=force)

    def read_budget(self,size):
        require(type(size) is int and size>=0,'Exact nonnegative logical read reservation required')
        self.check();return self.ledger.reserve_read(size,self.lease)

    def _read_path(self,path,maximum,expected_sha=None,expected_bytes=None):
        path=Path(path);require(path.absolute()==path.resolve(),'Read aliases and symlinks are forbidden')
        require(type(maximum) is int and 0<maximum<=MAX_ARTIFACT_BYTES,'Bounded read size required')
        before=path.lstat()
        require(stat.S_ISREG(before.st_mode) and not getattr(before,'st_flags',0)&0x40000000,
                'Read source must be resident regular file; no implicit hydration')
        require(0<before.st_size<=maximum and (expected_bytes is None or before.st_size==expected_bytes),
                'Read source size differs')
        require(expected_sha is None or _hex(expected_sha),'Expected source SHA must be explicit hex')
        self.read_budget(before.st_size)
        raw=_read(path,maximum)
        require(len(raw)==before.st_size and (expected_sha is None or _sha(raw)==expected_sha),'Read source bytes/SHA differ')
        self.check();return raw

    def read(self,descriptor,maximum):
        require(isinstance(descriptor,dict) and set(descriptor)=={'path','bytes','sha256'}
                and type(descriptor['bytes']) is int and descriptor['bytes']>0 and _hex(descriptor['sha256']),
                'Exact phase artifact descriptor required')
        return self._read_path(_relative(self.run,descriptor['path']),maximum,descriptor['sha256'],descriptor['bytes'])

    def read_original(self,relative,maximum,expected_sha=None):
        return self._read_path(_relative(self.original_run,relative),maximum,expected_sha)

    def read_source(self,path,maximum,expected_sha=None):
        path=Path(path).absolute();require(path==path.resolve(),'Source alias or symlink refused')
        roots=[Path(self.plan[k]).absolute() for k in ('corpus_root','reference_cache_root') if k in self.plan]
        roots += [Path(x).absolute() for x in self.plan.get('source_read_roots',[])]
        allowed=[Path(x).absolute() for x in self.plan.get('source_read_paths',[])]
        require(all(p==p.resolve() for p in (*roots,*allowed))
                and (path in allowed or any(path.is_relative_to(p) for p in roots)), 'Source path outside explicit Phase A read scope')
        return self._read_path(path,maximum,expected_sha)

    def write(self,relative,raw,*,kind):
        require(isinstance(raw,bytes) and 0<len(raw)<=MAX_ARTIFACT_BYTES,'Bounded nonempty artifact bytes required')
        path=_relative(self.directory,relative);stage=path.with_name(path.name+'.writing')
        require(not path.exists() and not path.is_symlink() and not stage.exists() and not stage.is_symlink(),
                'Existing or partial immutable artifact must be preserved')
        self.check();token=self.ledger.reserve_output(len(raw),self.lease,kind=kind)
        try:
            _directory(path.parent)
            with stage.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
            os.link(stage,path);stage.unlink();_sync(path.parent)
            verified=self._read_path(path,len(raw),_sha(raw),len(raw))
            descriptor=dict(path=str(path.relative_to(self.run)),bytes=len(verified),sha256=_sha(verified))
            self.ledger.settle_output(self.lease,token,actual_bytes=len(verified),sha256=descriptor['sha256'])
            return descriptor
        except BaseException:
            try:
                actual=path.stat().st_size if path.exists() else stage.stat().st_size if stage.exists() else 0
                self.ledger.settle_output(self.lease,token,actual_bytes=min(actual,len(raw)),failed=True)
            except BaseException:pass
            raise

    def precharge_forward(self,**kwargs):
        self.check();token=self.ledger.enter_forward(self.lease,**kwargs);return token

    def confirm_forward(self,token):
        self.ledger.confirm_forward(self.lease,token);self.check()

    def precharge_model(self,**kwargs):
        self._set_job(kwargs['job_id']);self.check();return self.ledger.enter_model(self.lease,**kwargs)

    def confirm_model(self,token):
        self.ledger.confirm_model(self.lease,token);self.check()


def verify_descriptors(data,store):
    """Verify descriptors present in the passed object, not links in file bodies.

The controller should pass only its own newly committed artifact inventory.
An original-campaign descriptor is never redirected to the original namespace.
"""
    checked={}
    def visit(value):
        if isinstance(value,dict):
            if 'path' in value:
                require(set(value)=={'path','bytes','sha256'},'Malformed descriptor in explicit artifact inventory')
                if value['path'] in checked:require(checked[value['path']]==value,'Conflicting duplicate descriptor');return
                store.read(value,MAX_ARTIFACT_BYTES);checked[value['path']]=value.copy()
            else:
                for child in value.values():visit(child)
        elif isinstance(value,(tuple,list)):
            for child in value:visit(child)
    visit(data)
    require(checked,'At least one explicit phase artifact descriptor required')
    return dict(files=len(checked),bytes=sum(x['bytes'] for x in checked.values()),descriptors_sha256=digest(checked))
