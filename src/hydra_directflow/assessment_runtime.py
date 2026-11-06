"""Small durable budget/store for the 36-model assessment first pass.

The orchestrator authenticates the plan, read release and scientific bindings.
This module never reads source values semantically, loads a model, downloads a
source, or authorizes scoring. Source GET accounting belongs to its separate
approved staging budget. Logical reads here are artifact/source-file bytes,
not network traffic or SQLite/OS paging.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import sys
import time
from urllib.parse import quote
import uuid

from .contract import GAUGE_IDS, ARCHITECTURES

GIB = 1024**3
BOOKKEEPING_BYTES = 64*1024**2
LIMITS = dict(max_case_checkpoint_evaluations=350000, max_forward_batches=6000,
              max_batch_size=64, max_model_constructions=36,
              max_model_constructions_with_resume=40, max_numeric_seconds=21600,
              max_combined_rss_mps_bytes=4*GIB, max_mps_bytes=2*GIB,
              max_read_bytes=64*GIB, max_output_bytes=2*GIB,
              minimum_free_disk_bytes=8*GIB)
MODEL_IDS = frozenset(f'final_development__{g}__{a}__full_L168__seed{s}'
                      for g in GAUGE_IDS for a in ARCHITECTURES for s in (17,29,43))
SCHEMA = 'hydra-assessment-firstpass-runtime/v1'


class AssessmentRuntimeError(ValueError): pass
class AssessmentLimitError(AssessmentRuntimeError): pass
class AssessmentIntegrityError(AssessmentRuntimeError): pass


def require(test, message, error=AssessmentRuntimeError):
    if not test: raise error(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def sha(raw): return hashlib.sha256(raw).hexdigest()
def is_sha(value): return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def validate_plan(plan):
    require(type(plan) is dict and is_sha(plan.get('plan_sha256'))
            and digest({k:v for k,v in plan.items() if k!='plan_sha256'})==plan['plan_sha256'],
            'An exact frozen assessment plan digest is required')
    require(plan.get('runtime_limits')==LIMITS and all(type(v) is int for v in plan['runtime_limits'].values()),
            'Only the declared first-pass limits are supported')
    models=plan.get('checkpoint_sha256_by_job')
    require(type(models) is dict and set(models)==MODEL_IDS and all(is_sha(v) for v in models.values()),
            'Exactly the36 full-L168 checkpoint identities are required')
    chunks=plan.get('required_prediction_chunks',[])
    require(type(chunks) in (list,tuple) and len(chunks)==len(set(chunks))
            and all(type(k) is str and re.fullmatch(r'[A-Za-z0-9_.:/-]{1,240}',k)
                    and k not in MODEL_IDS and k!='PREDICTION_BARRIER' for k in chunks),
            'Explicit additional prediction-commit keys required')
    p=Path(plan['run_directory'])
    require(p.is_absolute() and p==p.resolve(), 'A canonical frozen run directory is required')
    return json.loads(canonical(plan))


def _sync(directory):
    fd=os.open(directory,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


class AssessmentRuntime:
    def __init__(self, plan, *, clock=time.time, resource_probe=None, additional_pids=()):
        self.plan=validate_plan(plan);self.directory=Path(self.plan['run_directory'])
        self.database=self.directory/'assessment-runtime.sqlite3'
        self.clock=clock;self.resource_probe=resource_probe;self.additional_pids=tuple(additional_pids)
        require(all(type(p) is int and p>0 for p in self.additional_pids), 'Explicit extra process IDs required')
        self.con=None;self.owner=None

    def _held(self): require(self.con is not None and self.owner is not None, 'One runtime writer is required')

    @contextmanager
    def _transaction(self):
        self._held();self.con.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.con.execute('COMMIT')
        except BaseException:
            if self.con.in_transaction:self.con.execute('ROLLBACK')
            raise

    def _state(self):
        row=self.con.execute('SELECT body FROM state WHERE id=1').fetchone()
        require(row is not None,'Initialized runtime required')
        return json.loads(row[0])

    def _save(self,state):self.con.execute('UPDATE state SET body=? WHERE id=1',(canonical(state),))

    def _failure(self, kind, reason):
        with self._transaction():
            s=self._state()
            if s['failure'] is None:
                s['failure_sequence']+=1
                s['failure']=dict(id=s['failure_sequence'],kind=kind,reason=reason,
                                  active_job=s['active_job'],at=float(self.clock()))
                self._save(s)

    def record_failure(self, kind, reason):
        require(kind in ('INFRASTRUCTURE','NUMERICAL','INTEGRITY','RESOURCE','IMPLEMENTATION','UNCLASSIFIED')
                and type(reason) is str and 0<len(reason)<=512, 'Explicit bounded failure classification required')
        self._failure(kind,reason)

    @contextmanager
    def writer(self):
        require(self.con is None and self.directory==self.directory.resolve(), 'Canonical single writer required')
        self.directory.mkdir(parents=True,exist_ok=True)
        marker=self.directory/'RUNTIME_IDENTITY.json'
        require(not marker.exists() or self.database.exists(),'A lost ledger cannot reset an existing runtime')
        for p in [self.database,Path(str(self.database)+'-wal'),Path(str(self.database)+'-shm'),self.directory/'writer.lock',marker]:
            if p.exists() or p.is_symlink():
                st=p.lstat();require(stat.S_ISREG(st.st_mode) and not getattr(st,'st_flags',0)&0x40000000,
                                    'Runtime files must remain resident regular files')
        fd=os.open(self.directory/'writer.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:os.close(fd);raise AssessmentRuntimeError('Another runtime writer is active')
        self.owner=uuid.uuid4().hex
        try:
            self.con=sqlite3.connect(self.database,isolation_level=None,timeout=2)
            self.con.row_factory=sqlite3.Row
            self.con.execute('PRAGMA journal_mode=WAL');self.con.execute('PRAGMA synchronous=FULL')
            self.con.execute('PRAGMA fullfsync=ON');self.con.execute('PRAGMA checkpoint_fullfsync=ON')
            self.con.executescript('''CREATE TABLE IF NOT EXISTS state(id INTEGER PRIMARY KEY,body TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY,kind TEXT,job_id TEXT,resume_id INTEGER,
                cases INTEGER,batches INTEGER,status TEXT,detail TEXT);
              CREATE TABLE IF NOT EXISTS reads(id INTEGER PRIMARY KEY,bytes INTEGER,purpose TEXT,numeric INTEGER);
              CREATE TABLE IF NOT EXISTS outputs(id INTEGER PRIMARY KEY,path TEXT,reserved INTEGER,actual INTEGER,
                sha256 TEXT,status TEXT,kind TEXT);
              CREATE TABLE IF NOT EXISTS commits(key TEXT PRIMARY KEY,kind TEXT,job_id TEXT,body TEXT,sha256 TEXT);
              CREATE UNIQUE INDEX IF NOT EXISTS completed_model ON commits(job_id) WHERE kind='model';
              CREATE TABLE IF NOT EXISTS resumes(id INTEGER PRIMARY KEY,body TEXT,sha256 TEXT);''')
            with self._transaction():
                row=self.con.execute('SELECT body FROM state WHERE id=1').fetchone();now=float(self.clock())
                require(math.isfinite(now),'Finite clock required')
                if row is None:
                    # Existing artifacts without their ledger are never silently adopted.
                    require(not (self.directory/'artifacts').exists(), 'Existing artifacts require their original ledger')
                    s=dict(schema=SCHEMA,plan=self.plan,created_at=now,last_clock=now,numeric_started_at=None,deadline=None,
                      owner=None,active_job=None,active_model_token=None,resume_required_job=None,resume_id=0,
                      entered_cases=0,confirmed_cases=0,entered_batches=0,confirmed_batches=0,
                      entered_models=0,confirmed_models=0,read_reserved=0,output_reserved=BOOKKEEPING_BYTES,
                      bookkeeping_reserved=BOOKKEEPING_BYTES,peak_combined_rss_mps_bytes=0,peak_mps_bytes=0,
                      minimum_free_bytes=None,failure=None,failure_sequence=0)
                    self.con.execute('INSERT INTO state VALUES(1,?)',(canonical(s),))
                else:
                    s=json.loads(row['body']);require(s['schema']==SCHEMA and s['plan']==self.plan,
                                                     'Original namespace, plan and consumed allowance must be retained')
                    require(now>=s['last_clock'],'Backward clock cannot reset the assessment allowance')
                if s['owner'] is not None and s['failure'] is None:
                    s['failure_sequence']+=1;s['failure']=dict(id=s['failure_sequence'],kind='ORPHANED',
                        reason='Previous writer did not close',active_job=s['active_job'],at=now)
                s['owner']=self.owner;self._save(s)
            identity=dict(schema=SCHEMA,plan_sha256=self.plan['plan_sha256'],run_directory=str(self.directory),created_at=s['created_at'])
            if marker.exists():
                require(marker.stat().st_size<=4096 and json.loads(marker.read_text())==identity,
                        'Runtime identity marker changed',AssessmentIntegrityError)
            else:
                with marker.open('x') as stream:stream.write(canonical(identity));stream.flush();os.fsync(stream.fileno())
            _sync(self.directory)
            try:yield self
            except BaseException as error:
                kind=('RESOURCE' if isinstance(error,AssessmentLimitError) else 'INTEGRITY'
                      if isinstance(error,AssessmentIntegrityError) else 'NUMERICAL_OR_INTEGRITY'
                      if isinstance(error,ValueError) else 'INFRASTRUCTURE'
                      if isinstance(error,(OSError,KeyboardInterrupt)) else 'UNCLASSIFIED')
                self._failure(kind,type(error).__name__);raise
            finally:
                if self.con is not None:
                    with self._transaction():
                        s=self._state()
                        if s['active_job'] is not None and s['failure'] is None:
                            s['failure_sequence']+=1;s['failure']=dict(id=s['failure_sequence'],kind='UNCLASSIFIED',
                                reason='Writer exited with unfinished model',active_job=s['active_job'],at=float(self.clock()))
                        s['owner']=None;self._save(s)
        finally:
            if self.con is not None:self.con.close()
            self.con=None;self.owner=None;fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)

    def _measure(self):
        if self.resource_probe is not None:return self.resource_probe()
        import psutil
        seen=set();rss=0
        for pid in (os.getpid(),)+self.additional_pids:
            try:process=psutil.Process(pid);processes=[process]+process.children(recursive=True)
            except psutil.NoSuchProcess:continue
            for p in processes:
                if p.pid in seen:continue
                try:rss+=int(p.memory_info().rss);seen.add(p.pid)
                except psutil.NoSuchProcess:pass
        driver=allocator=0;torch=sys.modules.get('torch')
        if torch is not None and torch.backends.mps.is_available():
            driver=int(torch.mps.driver_allocated_memory());allocator=int(torch.mps.current_allocated_memory())
        return dict(combined_rss_bytes=rss+driver,mps_bytes=max(driver,allocator),
                    free_bytes=int(shutil.disk_usage(self.directory).free))

    def _check(self, numeric=False, *, measurement=None, allow_failure=False):
        require(type(numeric) is bool and type(allow_failure) is bool,'Explicit runtime phase flags required')
        self._held();m=self._measure() if measurement is None else measurement
        require(set(m)=={'combined_rss_bytes','mps_bytes','free_bytes'}
                and all(type(x) is int and x>=0 for x in m.values()),'Exact resource observations required')
        error=None
        with self._transaction():
            s=self._state();now=float(self.clock())
            require(math.isfinite(now) and now>=s['last_clock'],'Clock moved backwards')
            s['last_clock']=now
            if numeric and s['numeric_started_at'] is None and s['failure'] is None:
                s['numeric_started_at']=now;s['deadline']=now+LIMITS['max_numeric_seconds']
            s['peak_combined_rss_mps_bytes']=max(s['peak_combined_rss_mps_bytes'],m['combined_rss_bytes'])
            s['peak_mps_bytes']=max(s['peak_mps_bytes'],m['mps_bytes'])
            s['minimum_free_bytes']=m['free_bytes'] if s['minimum_free_bytes'] is None else min(s['minimum_free_bytes'],m['free_bytes'])
            if s['deadline'] is not None and now>=s['deadline']:error='Persistent numerical execution deadline exceeded'
            if m['combined_rss_bytes']>LIMITS['max_combined_rss_mps_bytes'] or m['mps_bytes']>LIMITS['max_mps_bytes'] or m['free_bytes']<LIMITS['minimum_free_disk_bytes']:error='Memory/MPS/free-disk limit exceeded'
            metadata_bytes=sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file() and p.name.startswith(('assessment-runtime.sqlite3','writer.lock','RUNTIME_IDENTITY')))
            if metadata_bytes>BOOKKEEPING_BYTES:error='Runtime bookkeeping reservation exceeded'
            if error and s['failure'] is None:
                s['failure_sequence']+=1;s['failure']=dict(id=s['failure_sequence'],kind='RESOURCE',reason=error,active_job=s['active_job'],at=now)
            failure=s['failure'];self._save(s)
        if error:raise AssessmentLimitError(error)
        require(allow_failure or failure is None,'Run has a preserved failure; no automatic retry')
        return m

    def start_numeric(self):self._check(True);return self.snapshot()['deadline']
    def check_resources(self, *, combined_rss_bytes, mps_bytes, free_bytes):
        return self._check(measurement=dict(combined_rss_bytes=combined_rss_bytes,mps_bytes=mps_bytes,free_bytes=free_bytes))
    def check(self):return self._check()

    def _budget(self, condition, reason):
        if not condition:raise AssessmentLimitError(reason)

    def reserve_read(self, bytes_count, *, numeric=True, purpose='ARTIFACT'):
        require(type(bytes_count) is int and bytes_count>=0 and type(numeric) is bool
                and type(purpose) is str and 0<len(purpose)<=96,'Bounded explicit read charge required')
        self._check(numeric)
        try:
            with self._transaction():
                s=self._state();self._budget(s['read_reserved']+bytes_count<=LIMITS['max_read_bytes'],'Logical read limit exceeded')
                s['read_reserved']+=bytes_count;self._save(s)
                return self.con.execute('INSERT INTO reads(bytes,purpose,numeric) VALUES(?,?,?)',(bytes_count,purpose,int(numeric))).lastrowid
        except AssessmentLimitError as e:self._failure('RESOURCE',str(e));raise

    def precharge_model(self, job_id, *, checkpoint_sha256=None):
        require(type(job_id) is str and job_id in MODEL_IDS and (checkpoint_sha256 is None or checkpoint_sha256==self.plan['checkpoint_sha256_by_job'][job_id]),'Declared model/checkpoint required')
        self._check(True)
        try:
            with self._transaction():
                s=self._state();require(s['active_job'] is None,'Only one active model is allowed')
                require(self.con.execute("SELECT 1 FROM commits WHERE kind='model' AND job_id=?",(job_id,)).fetchone() is None,'Completed model must be reused, never reconstructed')
                last=self.con.execute("SELECT MAX(resume_id) FROM calls WHERE kind='MODEL' AND job_id=?",(job_id,)).fetchone()[0]
                require(last is None or last<s['resume_id'],'Repeat construction needs a new explicit diagnosed resume')
                require(s['resume_required_job'] in (None,job_id),'Resume must continue the interrupted model')
                cap=LIMITS['max_model_constructions_with_resume'] if s['resume_id'] else LIMITS['max_model_constructions']
                self._budget(s['entered_models']<cap,'Model-construction limit exceeded')
                token=self.con.execute("INSERT INTO calls(kind,job_id,resume_id,cases,batches,status,detail) VALUES('MODEL',?,?,0,0,'ENTERED',?)",(job_id,s['resume_id'],canonical(dict(checkpoint_sha256=self.plan['checkpoint_sha256_by_job'][job_id])))).lastrowid
                s['entered_models']+=1;s['active_job']=job_id;s['active_model_token']=token;s['resume_required_job']=None;self._save(s);return token
        except AssessmentLimitError as e:self._failure('RESOURCE',str(e));raise

    def precharge_forward(self, *, case_count, batch_count=1, stage='ASSESSMENT_FULL_L168', case_ids_sha256):
        require(type(case_count) is int and 1<=case_count<=LIMITS['max_batch_size']
                and type(batch_count) is int and batch_count==1 and is_sha(case_ids_sha256)
                and stage=='ASSESSMENT_FULL_L168','One explicitly identified B64 assessment forward required')
        self._check(True)
        try:
            with self._transaction():
                s=self._state();model=self.con.execute('SELECT status FROM calls WHERE id=?',(s['active_model_token'],)).fetchone()
                require(s['active_job'] is not None and model is not None and model[0]=='CONFIRMED','Confirmed active model construction required')
                self._budget(s['entered_cases']+case_count<=LIMITS['max_case_checkpoint_evaluations'] and s['entered_batches']+1<=LIMITS['max_forward_batches'],'Forward case/batch limit exceeded')
                token=self.con.execute("INSERT INTO calls(kind,job_id,resume_id,cases,batches,status,detail) VALUES('FORWARD',?,?,?,?, 'ENTERED',?)",(s['active_job'],s['resume_id'],case_count,1,canonical(dict(stage=stage,case_ids_sha256=case_ids_sha256)))).lastrowid
                s['entered_cases']+=case_count;s['entered_batches']+=1;self._save(s);return token
        except AssessmentLimitError as e:self._failure('RESOURCE',str(e));raise

    def _confirm(self, token, kind):
        require(type(token) is int and token>0,'Exact positive operation token required')
        with self._transaction():
            s=self._state();row=self.con.execute('SELECT * FROM calls WHERE id=?',(token,)).fetchone()
            require(row is not None and row['kind']==kind and row['status']=='ENTERED'
                    and row['job_id']==s['active_job'] and row['resume_id']==s['resume_id'],'Only the exact pending operation can be confirmed')
            self.con.execute("UPDATE calls SET status='CONFIRMED' WHERE id=?",(token,))
            if kind=='MODEL':s['confirmed_models']+=1
            else:s['confirmed_cases']+=row['cases'];s['confirmed_batches']+=row['batches']
            self._save(s)
        self._check()

    def confirm_model(self, token):self._confirm(token,'MODEL')
    def confirm_forward(self, token):self._confirm(token,'FORWARD')

    def authorize_resume(self, diagnosis):
        self._check(allow_failure=True)
        require(type(diagnosis) is dict and diagnosis.get('classification')=='INFRASTRUCTURE'
                and diagnosis.get('plan_sha256')==self.plan['plan_sha256']
                and type(diagnosis.get('reason')) is str and 0<len(diagnosis['reason'])<=512
                and type(diagnosis.get('evidence_sha256')) in (list,tuple)
                and 1<=len(diagnosis['evidence_sha256'])<=8 and all(is_sha(v) for v in diagnosis['evidence_sha256']),
                'Explicit evidence-bound infrastructure diagnosis required')
        with self._transaction():
            s=self._state();f=s['failure'];require(f is not None and f['kind'] in ('INFRASTRUCTURE','ORPHANED','UNCLASSIFIED')
                    and diagnosis.get('failure_id')==f['id'],'Numerical, integrity, resource or unrelated failures cannot resume')
            require(len(canonical(diagnosis))<=8192,'Bounded diagnosis required')
            s['resume_id']+=1;self.con.execute('INSERT INTO resumes VALUES(?,?,?)',(s['resume_id'],canonical(diagnosis),digest(diagnosis)))
            self.con.execute("UPDATE calls SET status='ABANDONED' WHERE status='ENTERED'")
            self.con.execute("UPDATE outputs SET status='FAILED' WHERE status='PRECHARGED'")
            s['resume_required_job']=s['active_job'];s['active_job']=None;s['active_model_token']=None;s['failure']=None;self._save(s)
        return s['resume_id']

    def _path(self, relative):
        require(type(relative) is str and PurePosixPath(relative).as_posix()==relative
                and not PurePosixPath(relative).is_absolute() and '..' not in PurePosixPath(relative).parts
                and relative.startswith('artifacts/'),'Canonical artifact-relative path required')
        p=self.directory/relative;require(p==p.resolve(),'Artifact aliases and symlinks are forbidden');return p

    def read(self, descriptor, *, numeric=True):
        require(type(descriptor) is dict and set(descriptor)=={'path','bytes','sha256'}
                and type(descriptor['bytes']) is int and 0<descriptor['bytes']<=LIMITS['max_output_bytes']
                and is_sha(descriptor['sha256']),'Exact artifact descriptor required')
        self._check(numeric);p=self._path(descriptor['path'])
        try:
            before=p.lstat();require(stat.S_ISREG(before.st_mode) and not getattr(before,'st_flags',0)&0x40000000
                and before.st_size==descriptor['bytes'],'Artifact residency or size differs',AssessmentIntegrityError)
            measured=self._measure();self._budget(measured['combined_rss_bytes']+before.st_size<=LIMITS['max_combined_rss_mps_bytes'],'Insufficient memory headroom for artifact bytes')
            self.reserve_read(before.st_size,numeric=numeric);raw=p.read_bytes();after=p.lstat()
            require((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
                    and len(raw)==descriptor['bytes'] and sha(raw)==descriptor['sha256'],'Artifact changed or hash differs',AssessmentIntegrityError)
            self._check();return raw
        except AssessmentLimitError as e:self._failure('RESOURCE',str(e));raise
        except (OSError,AssessmentIntegrityError) as e:
            self._failure('INTEGRITY','Artifact read authentication failed')
            raise AssessmentIntegrityError('Artifact read authentication failed') from e

    def write(self, relative_path, raw, *, kind='ARTIFACT', numeric=True):
        require(type(raw) is bytes and 0<len(raw)<=LIMITS['max_output_bytes']
                and type(kind) is str and 0<len(kind)<=96,'Bounded nonempty artifact bytes required')
        self._check(numeric);p=self._path('artifacts/'+relative_path);stage=p.with_name(p.name+'.writing')
        require(not p.exists() and not stage.exists(),'Existing or partial immutable artifact must be preserved')
        try:
            with self._transaction():
                s=self._state();self._budget(s['output_reserved']+len(raw)<=LIMITS['max_output_bytes'],'Derived/result output limit exceeded')
                s['output_reserved']+=len(raw);self._save(s)
                token=self.con.execute("INSERT INTO outputs(path,reserved,status,kind) VALUES(?,?,'PRECHARGED',?)",(str(p.relative_to(self.directory)),len(raw),kind)).lastrowid
            p.parent.mkdir(parents=True,exist_ok=True)
            with stage.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            os.link(stage,p);stage.unlink();_sync(p.parent)
            descriptor=dict(path=str(p.relative_to(self.directory)),bytes=len(raw),sha256=sha(raw))
            self.read(descriptor,numeric=numeric)
            with self._transaction():self.con.execute("UPDATE outputs SET actual=?,sha256=?,status='VERIFIED' WHERE id=?",(len(raw),descriptor['sha256'],token))
            return descriptor
        except BaseException as error:
            if 'token' in locals():
                with self._transaction():self.con.execute("UPDATE outputs SET actual=?,status='FAILED' WHERE id=?",(min(len(raw),p.stat().st_size if p.exists() else stage.stat().st_size if stage.exists() else 0),token))
            if isinstance(error,AssessmentLimitError):self._failure('RESOURCE',str(error))
            raise

    def _descriptors(self, artifacts):
        found={}
        def visit(value):
            if type(value) is dict:
                if 'path' in value:
                    require(set(value)=={'path','bytes','sha256'},'Malformed explicit artifact inventory')
                    if value['path'] in found:require(found[value['path']]==value,'Conflicting artifact identity')
                    found[value['path']]=value
                else:
                    for item in value.values():visit(item)
            elif type(value) in (list,tuple):
                for item in value:visit(item)
        visit(artifacts);require(found,'A nonempty explicit artifact inventory is required')
        return found

    def commit(self, key, artifacts, *, kind=None, job_id=None, metadata=None):
        require(type(key) is str and re.fullmatch(r'[A-Za-z0-9_.:/-]{1,240}',key),'A stable commit key is required')
        kind=kind or ('model' if key in MODEL_IDS else 'chunk');job_id=job_id or (key if kind=='model' else None)
        require(kind in ('chunk','model') and (job_id in MODEL_IDS if kind=='model' else job_id is None),'Explicit chunk/model commit scope required')
        artifacts=json.loads(canonical(artifacts));metadata=json.loads(canonical(metadata))
        self._check();descriptors=self._descriptors(artifacts)
        for d in descriptors.values():self.read(d,numeric=False)
        body=dict(schema=SCHEMA,plan_sha256=self.plan['plan_sha256'],key=key,kind=kind,job_id=job_id,artifacts=artifacts,metadata=metadata)
        require(len(canonical(body).encode())<=256*1024,'Commit metadata exceeds its bound')
        with self._transaction():
            s=self._state();require(self.con.execute('SELECT 1 FROM commits WHERE key=?',(key,)).fetchone() is None,'Completed commit must be reused, not replaced')
            if kind=='model':
                require(s['active_job'] in (None,job_id),'Cannot commit a different active model')
                require(self.con.execute("SELECT 1 FROM calls WHERE job_id=? AND status='ENTERED'",(job_id,)).fetchone() is None,'Pending model/forward operations cannot be committed')
                if s['active_job']==job_id:s['active_job']=None;s['active_model_token']=None
            self.con.execute('INSERT INTO commits VALUES(?,?,?,?,?)',(key,kind,job_id,canonical(body),digest(body)));self._save(s)
        return body

    def reuse(self, key, *, expected_metadata=None):
        self._check();row=self.con.execute('SELECT * FROM commits WHERE key=?',(key,)).fetchone()
        if row is None:return None
        try:
            body=json.loads(row['body']);require(digest(body)==row['sha256'] and body['schema']==SCHEMA
                    and body['plan_sha256']==self.plan['plan_sha256'] and body['key']==key
                    and body['kind']==row['kind'] and body['job_id']==row['job_id'],
                    'Commit identity changed',AssessmentIntegrityError)
            if expected_metadata is not None:require(body['metadata']==expected_metadata,'Completed metadata changed',AssessmentIntegrityError)
            for d in self._descriptors(body['artifacts']).values():self.read(d,numeric=False)
            return body
        except AssessmentIntegrityError:
            self._failure('INTEGRITY','Completed commit authentication failed');raise

    def prediction_barrier(self):
        """Hash-authenticate all36 model commits and declared comparator chunks.

        This is a durable structural barrier. The orchestrator must separately
        validate prediction semantics and any authorization to read targets.
        """
        self._check()
        models=[dict(x) for x in self.con.execute("SELECT key,job_id,sha256 FROM commits WHERE kind='model'")]
        require(len(models)==36 and {x['job_id'] for x in models}==MODEL_IDS,
                'All36 declared model commits are required before the prediction barrier')
        for item in models:self.reuse(item['key'])
        chunks={}
        for key in self.plan.get('required_prediction_chunks',[]):
            require(self.reuse(key) is not None,'Required comparator prediction commit is missing')
            chunks[key]=self.con.execute('SELECT sha256 FROM commits WHERE key=?',(key,)).fetchone()[0]
        body=dict(schema='hydra-assessment-prediction-barrier/v1',
                  status='ALL_DECLARED_PREDICTION_COMMITS_HASH_VERIFIED',plan_sha256=self.plan['plan_sha256'],
                  model_commit_sha256_by_job={x['job_id']:x['sha256'] for x in models},
                  additional_prediction_commits=chunks,runner_semantic_validation_and_release_still_required=True)
        old=self.reuse('PREDICTION_BARRIER',expected_metadata=body)
        if old is not None:return old
        desc=self.write('PREDICTION_BARRIER.json',canonical(body).encode(),kind='PREDICTION_BARRIER',numeric=False)
        return self.commit('PREDICTION_BARRIER',{'barrier':desc},metadata=body)

    def snapshot(self):
        own=self.con is None
        con=sqlite3.connect('file:'+quote(str(self.database),safe='/')+'?mode=ro',uri=True) if own else self.con
        con.row_factory=sqlite3.Row
        try:
            con.execute('BEGIN')
            row=con.execute('SELECT body FROM state WHERE id=1').fetchone();require(row is not None,'Runtime not initialized')
            result=json.loads(row['body']);require(result['plan']==self.plan,'Runtime plan differs')
            result.pop('plan');result['plan_sha256']=self.plan['plan_sha256'];result['limits']=dict(LIMITS)
            result['commits']=[dict(x) for x in con.execute('SELECT key,kind,job_id,sha256 FROM commits ORDER BY key')]
            result['pending_calls']=con.execute("SELECT COUNT(*) FROM calls WHERE status='ENTERED'").fetchone()[0]
            result['abandoned_calls']=con.execute("SELECT COUNT(*) FROM calls WHERE status='ABANDONED'").fetchone()[0]
            con.execute('COMMIT');return result
        except BaseException:
            if con.in_transaction:con.execute('ROLLBACK')
            raise
        finally:
            if own:con.close()
