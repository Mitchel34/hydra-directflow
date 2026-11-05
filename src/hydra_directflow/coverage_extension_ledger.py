"""Durable Phase A inference accounting, separate from the training ledger.

Every model/forward/read/output charge is persisted before its operation. Failed
charges are retained, completed tasks cannot be rerun, and one explicit
infrastructure resume shares the original phase clock and allowance. This class
never edits the original campaign ledger or constructs a model.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import time
from urllib.parse import quote
import uuid

from .campaign_ledger import canonical, digest
from .contract import ARCHITECTURES, GAUGE_IDS
from .reference import require

SCHEMA = 'hydra-phase-a-inference-ledger/v1'
LIMITS = dict(max_case_checkpoint_evaluations=1280000, max_forward_batches=20000,
              max_batch_size=64, max_model_constructions=145, max_seconds=21600,
              max_rss_bytes=4*1024**3, max_mps_bytes=2*1024**3, max_read_bytes=8*1024**3,
              max_output_bytes=1024**3, min_free_bytes=8*1024**3, max_infrastructure_resumes=1)
CHECKPOINT_IDS = {f'final_development__{g}__{a}__full_L{h}__seed{s}'
                  for g in GAUGE_IDS for a in ARCHITECTURES for h in (168,24) for s in (17,29,43)}


class ExtensionAllowanceExhausted(ValueError):
    pass


@dataclass(frozen=True)
class ExtensionLease:
    task_key: str
    stage: str
    attempt_id: int
    lease_id: str


def _safe(value):
    return isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_.:/-]{1,240}',value) is not None


def _sha(value):
    return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None


def validate_extension_plan(plan):
    require(isinstance(plan,dict) and _sha(plan.get('plan_sha256'))
            and digest({k:v for k,v in plan.items() if k!='plan_sha256'})==plan['plan_sha256'],
            'Exact immutable Phase A plan digest required')
    require(plan.get('limits')==LIMITS and all(type(v) is int for v in plan['limits'].values()),
            'Only the adopted finite Phase A limits are supported')
    models=plan.get('checkpoint_sha256_by_job',{})
    require(isinstance(models,dict) and set(models)==CHECKPOINT_IDS and all(_sha(x) for x in models.values()),
            'Exactly the72 approved final-development checkpoints are required')
    require(type(plan.get('original_campaign_deadline')) in (int,float)
            and math.isfinite(plan['original_campaign_deadline']), 'Original campaign deadline required')
    require(plan.get('protected_assessment_access') is False and plan.get('paid_cloud_jobs',0)==0,
            'Phase A cannot enable protected assessment or paid work')
    return json.loads(canonical(plan))


def _resident(path, *, optional=False):
    path=Path(path)
    if optional and not path.exists() and not path.is_symlink():return
    info=path.lstat()
    require(stat.S_ISREG(info.st_mode) and not getattr(info,'st_flags',0)&0x40000000,
            'Phase A ledger files must be resident regular files')


def _sync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


class PhaseALedger:
    def __init__(self,directory,*,clock=time.time):
        self.directory=Path(directory).absolute();self.database=self.directory/'phase-a.sqlite3'
        self.clock=clock;self.connection=None;self.plan=None;self._owner=None;self._lock_fd=None

    def _held(self):
        require(self.connection is not None and self._owner is not None,'Phase A exclusive writer lease required')

    def _schema(self):
        self.connection.executescript('''
        CREATE TABLE IF NOT EXISTS phase(
          id INTEGER PRIMARY KEY CHECK(id=1),schema TEXT NOT NULL,plan_json TEXT NOT NULL,
          reservation_token INTEGER NOT NULL,started REAL NOT NULL,deadline REAL NOT NULL,last_clock REAL NOT NULL,
          entered_cases INTEGER NOT NULL DEFAULT 0,confirmed_cases INTEGER NOT NULL DEFAULT 0,
          entered_batches INTEGER NOT NULL DEFAULT 0,confirmed_batches INTEGER NOT NULL DEFAULT 0,
          entered_models INTEGER NOT NULL DEFAULT 0,confirmed_models INTEGER NOT NULL DEFAULT 0,
          read_reserved INTEGER NOT NULL DEFAULT 0,output_reserved INTEGER NOT NULL DEFAULT 0,
          infrastructure_resumes INTEGER NOT NULL DEFAULT 0,peak_combined_rss INTEGER NOT NULL DEFAULT 0,
          peak_mps INTEGER NOT NULL DEFAULT 0,minimum_free INTEGER);
        CREATE TABLE IF NOT EXISTS tasks(
          task_key TEXT PRIMARY KEY,stage TEXT NOT NULL,state TEXT NOT NULL,model_job_id TEXT,
          receipt_json TEXT,receipt_sha TEXT);
        CREATE TABLE IF NOT EXISTS attempts(
          id INTEGER PRIMARY KEY,task_key TEXT NOT NULL,lease_id TEXT NOT NULL,status TEXT NOT NULL,
          started REAL NOT NULL,finished REAL,reason TEXT);
        CREATE TABLE IF NOT EXISTS calls(
          id INTEGER PRIMARY KEY,attempt_id INTEGER NOT NULL,kind TEXT NOT NULL,cases INTEGER NOT NULL,
          batches INTEGER NOT NULL,detail_json TEXT NOT NULL,confirmed INTEGER NOT NULL DEFAULT 0,created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS reads(
          id INTEGER PRIMARY KEY,attempt_id INTEGER,bytes INTEGER NOT NULL,purpose TEXT NOT NULL,created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS outputs(
          id INTEGER PRIMARY KEY,attempt_id INTEGER,kind TEXT NOT NULL,reserved INTEGER NOT NULL,
          actual INTEGER,sha256 TEXT,status TEXT NOT NULL,created REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS calls_attempt_kind ON calls(attempt_id,kind);
        CREATE INDEX IF NOT EXISTS outputs_attempt_status ON outputs(attempt_id,status);
        ''')

    @contextmanager
    def _transaction(self):
        self._held();self.connection.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.connection.execute('COMMIT')
        except BaseException:
            if self.connection.in_transaction:self.connection.execute('ROLLBACK')
            raise

    @contextmanager
    def writer(self):
        require(self.connection is None and self.directory==self.directory.resolve(),'Stable single writer required')
        missing=[];candidate=self.directory
        while not candidate.exists():
            missing.append(candidate);candidate=candidate.parent
        self.directory.mkdir(parents=True,mode=0o700,exist_ok=True)
        for created in reversed(missing):
            _sync_directory(created.parent);_sync_directory(created)
        _sync_directory(self.directory.parent)
        for suffix in ('','-wal','-shm'):_resident(Path(str(self.database)+suffix),optional=True)
        _resident(self.directory/'writer.lock',optional=True)
        fd=os.open(self.directory/'writer.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:os.close(fd);raise
        self._lock_fd=fd;self._owner=uuid.uuid4().hex
        schema_ready=False;failure=None
        try:
            self.connection=sqlite3.connect(self.database,isolation_level=None,timeout=2)
            self.connection.row_factory=sqlite3.Row
            self.connection.execute('PRAGMA journal_mode=WAL');self.connection.execute('PRAGMA synchronous=FULL')
            self.connection.execute('PRAGMA fullfsync=ON');self.connection.execute('PRAGMA checkpoint_fullfsync=ON')
            self._schema();schema_ready=True
            _sync_directory(self.directory)
            row=self.connection.execute('SELECT plan_json FROM phase WHERE id=1').fetchone()
            if row is not None:
                self.plan=validate_extension_plan(json.loads(row[0]))
                with self._transaction():
                    # Acquiring the OS lock proves no previous writer still owns
                    # its lease. Only infrastructure-orphan state is changed.
                    now=float(self.clock())
                    last=self.connection.execute('SELECT last_clock FROM phase WHERE id=1').fetchone()[0]
                    require(math.isfinite(now) and now>=last,'Invalid/backward clock cannot settle orphan state')
                    active=self.connection.execute("SELECT id,task_key FROM attempts WHERE status='ACTIVE'").fetchall()
                    for old in active:
                        self.connection.execute("UPDATE attempts SET status='INTERRUPTED',finished=?,reason='ORPHANED_WRITER' WHERE id=?",(now,old['id']))
                        self.connection.execute("UPDATE tasks SET state='INTERRUPTED' WHERE task_key=?",(old['task_key'],))
            yield self
        except BaseException as error:
            failure=error
            raise
        finally:
            try:
                if self.connection is not None:
                    if self.connection.in_transaction:self.connection.execute('ROLLBACK')
                    # An exception with a live task is an interruption, never
                    # a successful commit or an allowance refund.
                    if schema_ready:
                        with self._transaction():
                            now=float(self.clock())
                            require(math.isfinite(now),'Nonfinite completion clock')
                            active=self.connection.execute("SELECT id,task_key FROM attempts WHERE lease_id=? AND status='ACTIVE'",(self._owner,)).fetchall()
                            if failure is None or isinstance(failure,(KeyboardInterrupt,ExtensionAllowanceExhausted)):
                                state,reason='INTERRUPTED','WRITER_EXIT_WITH_ACTIVE_TASK'
                            else:
                                state,reason='FAILED_IMPLEMENTATION','UNCLASSIFIED_EXCEPTION_REQUIRES_REVIEW'
                            for old in active:
                                self.connection.execute('UPDATE attempts SET status=?,finished=?,reason=? WHERE id=?',
                                                        (state,now,reason,old['id']))
                                self.connection.execute('UPDATE tasks SET state=? WHERE task_key=?',(state,old['task_key']))
            finally:
                if self.connection is not None:self.connection.close()
                self.connection=None;self._owner=None;self.plan=None
                fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd);self._lock_fd=None

    def initialize(self,plan,original_reservation_token,*,initial_read_bytes=0):
        self._held();plan=validate_extension_plan(plan)
        require(type(original_reservation_token) is int and original_reservation_token>0
                and type(initial_read_bytes) is int and 0<=initial_read_bytes<=LIMITS['max_read_bytes'],
                'Original reservation token and bounded initial read charge required')
        if 'ledger_directory' in plan:require(plan['ledger_directory']==str(self.directory),'Phase A ledger location differs')
        with self._transaction():
            existing=self.connection.execute('SELECT * FROM phase WHERE id=1').fetchone()
            if existing is not None:
                require(existing['schema']==SCHEMA and existing['plan_json']==canonical(plan)
                        and existing['reservation_token']==original_reservation_token,
                        'Existing Phase A plan/reservation cannot be replaced')
                require(initial_read_bytes==0,'Initial read charge cannot be replayed or reset')
                self.plan=plan;self._time();return
            now=float(self.clock());deadline=min(now+LIMITS['max_seconds'],float(plan['original_campaign_deadline']))
            require(math.isfinite(now) and now<deadline,'Original campaign deadline has expired')
            self.connection.execute('INSERT INTO phase(id,schema,plan_json,reservation_token,started,deadline,last_clock,read_reserved) VALUES(1,?,?,?,?,?,?,?)',
                                    (SCHEMA,canonical(plan),original_reservation_token,now,deadline,now,initial_read_bytes))
            if initial_read_bytes:self.connection.execute('INSERT INTO reads(attempt_id,bytes,purpose,created) VALUES(NULL,?,?,?)',
                                                          (initial_read_bytes,'INITIAL_AUTHENTICATED_METADATA',now))
            self.plan=plan

    def _time(self):
        self._held();require(self.plan is not None,'Initialize the immutable Phase A plan first')
        row=self.connection.execute('SELECT last_clock,deadline FROM phase WHERE id=1').fetchone();now=float(self.clock())
        require(math.isfinite(now) and now>=row['last_clock'],'Phase A clock moved backward')
        if now>=row['deadline']:raise ExtensionAllowanceExhausted('Persistent Phase A/original campaign deadline expired')
        self.connection.execute('UPDATE phase SET last_clock=? WHERE id=1',(now,));return now

    def _active(self,lease,*,clock=True):
        self._held();require(isinstance(lease,ExtensionLease) and lease.lease_id==self._owner,'Foreign Phase A task lease')
        row=self.connection.execute('SELECT a.*,t.stage,t.state,t.model_job_id FROM attempts a JOIN tasks t ON t.task_key=a.task_key WHERE a.id=?',(lease.attempt_id,)).fetchone()
        require(row is not None and row['task_key']==lease.task_key and row['stage']==lease.stage
                and row['lease_id']==self._owner and row['status']==row['state']=='ACTIVE','Inactive or mismatched Phase A task')
        if clock:self._time()
        return row

    def claim_task(self,key,stage,*,resume=False):
        require(_safe(key) and _safe(stage) and type(resume) is bool,'Explicit bounded task/stage required')
        with self._transaction():
            now=self._time()
            require(self.connection.execute("SELECT COUNT(*) FROM attempts WHERE status='ACTIVE'").fetchone()[0]==0,
                    'Only one Phase A task/worker may be active')
            require(self.connection.execute("SELECT COUNT(*) FROM tasks WHERE state LIKE 'FAILED_%'").fetchone()[0]==0,
                    'A numerical/integrity/implementation failure blocks subsequent execution tasks')
            interrupted=self.connection.execute("SELECT task_key FROM tasks WHERE state='INTERRUPTED'").fetchall()
            require(not interrupted or len(interrupted)==1 and interrupted[0]['task_key']==key and resume,
                    'Interrupted work cannot be skipped without its explicit same-task resume')
            task=self.connection.execute('SELECT * FROM tasks WHERE task_key=?',(key,)).fetchone()
            if task is None:
                require(not resume,'Cannot resume an unstarted task')
                self.connection.execute("INSERT INTO tasks(task_key,stage,state) VALUES(?,?,'ACTIVE')",(key,stage))
            else:
                require(task['stage']==stage and task['state']=='INTERRUPTED' and resume,
                        'Completed/failed/existing work cannot be rerun or silently replaced')
                used=self.connection.execute('SELECT infrastructure_resumes FROM phase WHERE id=1').fetchone()[0]
                if used>=LIMITS['max_infrastructure_resumes']:raise ExtensionAllowanceExhausted('Single supervised infrastructure resume already consumed')
                self.connection.execute('UPDATE phase SET infrastructure_resumes=infrastructure_resumes+1 WHERE id=1')
                self.connection.execute("UPDATE tasks SET state='ACTIVE' WHERE task_key=?",(key,))
            attempt=self.connection.execute("INSERT INTO attempts(task_key,lease_id,status,started) VALUES(?,?,'ACTIVE',?)",(key,self._owner,now)).lastrowid
        return ExtensionLease(key,stage,attempt,self._owner)

    def enter_model(self,lease,*,job_id,checkpoint_sha256):
        with self._transaction():
            task=self._active(lease);now=self._time()
            require(job_id in self.plan['checkpoint_sha256_by_job']
                    and self.plan['checkpoint_sha256_by_job'][job_id]==checkpoint_sha256,'Unapproved checkpoint identity')
            require(task['model_job_id'] is None or task['model_job_id']==job_id,'A task cannot switch its model identity')
            count=self.connection.execute('SELECT entered_models FROM phase WHERE id=1').fetchone()[0]
            if count>=LIMITS['max_model_constructions']:raise ExtensionAllowanceExhausted('Model-construction allowance exhausted')
            self.connection.execute('UPDATE tasks SET model_job_id=? WHERE task_key=?',(job_id,lease.task_key))
            self.connection.execute('UPDATE phase SET entered_models=entered_models+1 WHERE id=1')
            detail=canonical(dict(job_id=job_id,checkpoint_sha256=checkpoint_sha256))
            return self.connection.execute("INSERT INTO calls(attempt_id,kind,cases,batches,detail_json,created) VALUES(?,'MODEL',0,0,?,?)",(lease.attempt_id,detail,now)).lastrowid

    def enter_forward(self,lease,*,case_count,batch_count,stage,case_ids_sha256):
        require(type(case_count) is int and 0<case_count<=LIMITS['max_batch_size'] and type(batch_count) is int
                and batch_count==1 and _safe(stage) and _sha(case_ids_sha256),'One bounded model-forward batch required')
        with self._transaction():
            task=self._active(lease);now=self._time()
            require(task['model_job_id'] is not None and self.connection.execute(
                "SELECT COUNT(*) FROM calls WHERE attempt_id=? AND kind='MODEL' AND confirmed=1",(lease.attempt_id,)).fetchone()[0]>0,
                'A confirmed checkpoint model load must precede every forward')
            row=self.connection.execute('SELECT entered_cases,entered_batches FROM phase WHERE id=1').fetchone()
            if row['entered_cases']+case_count>LIMITS['max_case_checkpoint_evaluations'] or row['entered_batches']+1>LIMITS['max_forward_batches']:
                raise ExtensionAllowanceExhausted('Case/checkpoint or forward-batch allowance exhausted')
            self.connection.execute('UPDATE phase SET entered_cases=entered_cases+?,entered_batches=entered_batches+1 WHERE id=1',(case_count,))
            detail=canonical(dict(stage=stage,case_ids_sha256=case_ids_sha256,job_id=task['model_job_id']))
            return self.connection.execute("INSERT INTO calls(attempt_id,kind,cases,batches,detail_json,created) VALUES(?,'FORWARD',?,1,?,?)",(lease.attempt_id,case_count,detail,now)).lastrowid

    def _confirm(self,lease,token,kind):
        require(type(token) is int and token>0,'Exact precharged call token required')
        with self._transaction():
            self._active(lease)
            row=self.connection.execute('SELECT * FROM calls WHERE id=?',(token,)).fetchone()
            require(row is not None and row['attempt_id']==lease.attempt_id and row['kind']==kind and row['confirmed']==0,
                    'Call is not an unconfirmed charge of this task/kind')
            self.connection.execute('UPDATE calls SET confirmed=1 WHERE id=?',(token,))
            if kind=='MODEL':self.connection.execute('UPDATE phase SET confirmed_models=confirmed_models+1 WHERE id=1')
            else:self.connection.execute('UPDATE phase SET confirmed_cases=confirmed_cases+?,confirmed_batches=confirmed_batches+1 WHERE id=1',(row['cases'],))

    def confirm_model(self,lease,token):return self._confirm(lease,token,'MODEL')
    def confirm_forward(self,lease,token):return self._confirm(lease,token,'FORWARD')

    def reserve_read(self,bytes_count,lease=None,*,purpose='IMMUTABLE_ARTIFACT'):
        require(type(bytes_count) is int and bytes_count>=0 and _safe(purpose),'Bounded nonnegative logical read charge required')
        with self._transaction():
            if lease is not None:self._active(lease)
            now=self._time();count=self.connection.execute('SELECT read_reserved FROM phase WHERE id=1').fetchone()[0]
            if count+bytes_count>LIMITS['max_read_bytes']:raise ExtensionAllowanceExhausted('Logical-read reservation allowance exhausted')
            self.connection.execute('UPDATE phase SET read_reserved=read_reserved+? WHERE id=1',(bytes_count,))
            return self.connection.execute('INSERT INTO reads(attempt_id,bytes,purpose,created) VALUES(?,?,?,?)',
                (lease.attempt_id if lease else None,bytes_count,purpose,now)).lastrowid

    def reserve_output(self,bytes_count,lease=None,*,kind):
        require(type(bytes_count) is int and bytes_count>0 and _safe(kind),'Positive bounded output charge required')
        with self._transaction():
            if lease is not None:self._active(lease)
            now=self._time();count=self.connection.execute('SELECT output_reserved FROM phase WHERE id=1').fetchone()[0]
            if count+bytes_count>LIMITS['max_output_bytes']:raise ExtensionAllowanceExhausted('Phase A output reservation allowance exhausted')
            self.connection.execute('UPDATE phase SET output_reserved=output_reserved+? WHERE id=1',(bytes_count,))
            return self.connection.execute("INSERT INTO outputs(attempt_id,kind,reserved,status,created) VALUES(?,?,?,'PRECHARGED',?)",
                (lease.attempt_id if lease else None,kind,bytes_count,now)).lastrowid

    def settle_output(self,lease,token,*,actual_bytes,sha256=None,failed=False):
        require(type(token) is int and token>0 and type(actual_bytes) is int and actual_bytes>=0
                and type(failed) is bool and (sha256 is None if failed else _sha(sha256)), 'Exact output settlement required')
        with self._transaction():
            if lease is not None:self._active(lease,clock=False)
            row=self.connection.execute('SELECT * FROM outputs WHERE id=?',(token,)).fetchone()
            require(row is not None and row['attempt_id']==(lease.attempt_id if lease else None)
                    and row['status']=='PRECHARGED' and actual_bytes<=row['reserved'],'Output reservation cannot be replaced or reset')
            self.connection.execute('UPDATE outputs SET actual=?,sha256=?,status=? WHERE id=?',
                                    (actual_bytes,sha256,'FAILED' if failed else 'VERIFIED',token))

    def check_resources(self,*,combined_rss_bytes,mps_bytes,free_bytes):
        require(all(type(x) is int and x>=0 for x in (combined_rss_bytes,mps_bytes,free_bytes)), 'Exact measured resource values required')
        with self._transaction():
            self._time()
            self.connection.execute('UPDATE phase SET peak_combined_rss=MAX(peak_combined_rss,?),peak_mps=MAX(peak_mps,?),minimum_free=CASE WHEN minimum_free IS NULL THEN ? ELSE MIN(minimum_free,?) END WHERE id=1',
                                    (combined_rss_bytes,mps_bytes,free_bytes,free_bytes))
        # Retain the observation that triggered a stop, not just earlier
        # in-bound samples. The caller must still terminate/settle its task.
        if combined_rss_bytes>LIMITS['max_rss_bytes'] or mps_bytes>LIMITS['max_mps_bytes'] or free_bytes<LIMITS['min_free_bytes']:
            raise ExtensionAllowanceExhausted('Phase A memory/MPS/free-disk bound reached')

    def receipt_bindings(self,lease):
        self._active(lease,clock=False)
        return dict(phase_plan_sha256=self.plan['plan_sha256'],task_key=lease.task_key,task_stage=lease.stage,attempt_id=lease.attempt_id)

    def bookkeeping_reserved_bytes(self):
        self._held()
        return self.connection.execute("SELECT COALESCE(SUM(reserved),0) FROM outputs WHERE attempt_id IS NULL AND kind='BOOKKEEPING'").fetchone()[0]

    def telemetry(self):
        """Small current counters for heartbeat publication; no full plan load."""
        self._held()
        row=self.connection.execute('''SELECT started,deadline,entered_cases,confirmed_cases,entered_batches,
            confirmed_batches,entered_models,confirmed_models,read_reserved,output_reserved,
            infrastructure_resumes FROM phase WHERE id=1''').fetchone()
        require(row is not None,'Initialized Phase A telemetry required')
        return dict(row)

    def complete_task(self,lease,receipt,*,validator):
        self._active(lease);bindings=self.receipt_bindings(lease)
        require(isinstance(receipt,dict) and all(receipt.get(k)==v for k,v in bindings.items())
                and len(canonical(receipt).encode())<=256*1024 and callable(validator),'Bounded task-bound completion receipt required')
        receipt_sha=digest(receipt)
        require(validator(json.loads(canonical(receipt)),bindings)==receipt_sha,'Completion artifacts were not authenticated')
        with self._transaction():
            self._active(lease);now=self._time()
            require(self.connection.execute('SELECT COUNT(*) FROM calls WHERE attempt_id=? AND confirmed=0',(lease.attempt_id,)).fetchone()[0]==0,
                    'Unconfirmed model/forward charges prevent successful completion')
            require(self.connection.execute("SELECT COUNT(*) FROM outputs WHERE attempt_id=? AND status!='VERIFIED'",(lease.attempt_id,)).fetchone()[0]==0,
                    'Unsettled or failed output prevents successful completion')
            self.connection.execute("UPDATE tasks SET state='COMPLETED',receipt_json=?,receipt_sha=? WHERE task_key=?",
                                    (canonical(receipt),receipt_sha,lease.task_key))
            self.connection.execute("UPDATE attempts SET status='COMPLETED',finished=? WHERE id=?",(now,lease.attempt_id))
        return receipt_sha

    def interrupt(self,lease,reason='INFRASTRUCTURE_INTERRUPTION'):
        self._finish_failed(lease,'INTERRUPTED',reason)

    def fail(self,lease,category,reason):
        require(category in ('INTEGRITY','NUMERICAL','IMPLEMENTATION'),'Explicit failure category required')
        self._finish_failed(lease,'FAILED_'+category,reason)

    def _finish_failed(self,lease,state,reason):
        require(_safe(reason),'Bounded failure code required')
        with self._transaction():
            attempt=self._active(lease,clock=False);now=float(self.clock())
            require(math.isfinite(now) and now>=attempt['started'],'Invalid failure completion clock')
            self.connection.execute('UPDATE attempts SET status=?,finished=?,reason=? WHERE id=?',(state,now,reason,lease.attempt_id))
            self.connection.execute('UPDATE tasks SET state=? WHERE task_key=?',(state,lease.task_key))

    def snapshot(self):
        own=self.connection is None
        if own:
            require(self.directory==self.directory.resolve(),'Phase A ledger directory link refused')
            for suffix in ('','-wal','-shm'):_resident(Path(str(self.database)+suffix),optional=bool(suffix))
            con=sqlite3.connect('file:'+quote(str(self.database),safe='/')+'?mode=ro',uri=True,isolation_level=None)
            con.row_factory=sqlite3.Row
        else:self._held();con=self.connection
        try:
            con.execute('BEGIN');row=con.execute('SELECT * FROM phase WHERE id=1').fetchone()
            require(row is not None and row['schema']==SCHEMA,'Initialized Phase A state required')
            plan=validate_extension_plan(json.loads(row['plan_json']))
            result={k:row[k] for k in row.keys() if k!='plan_json'}
            result.update(plan_sha256=plan['plan_sha256'],limits=plan['limits'],
                tasks=[dict(x) for x in con.execute('SELECT * FROM tasks ORDER BY task_key')],
                attempts=[dict(x) for x in con.execute('''SELECT a.*,
                    COALESCE(SUM(CASE WHEN c.kind='FORWARD' THEN c.cases ELSE 0 END),0) AS entered_cases,
                    COALESCE(SUM(CASE WHEN c.kind='FORWARD' AND c.confirmed=1 THEN c.cases ELSE 0 END),0) AS confirmed_cases,
                    COALESCE(SUM(CASE WHEN c.kind='FORWARD' THEN 1 ELSE 0 END),0) AS entered_batches,
                    COALESCE(SUM(CASE WHEN c.kind='FORWARD' AND c.confirmed=1 THEN 1 ELSE 0 END),0) AS confirmed_batches,
                    COALESCE(SUM(CASE WHEN c.kind='MODEL' THEN 1 ELSE 0 END),0) AS entered_models,
                    COALESCE(SUM(CASE WHEN c.kind='MODEL' AND c.confirmed=1 THEN 1 ELSE 0 END),0) AS confirmed_models
                    FROM attempts a LEFT JOIN calls c ON c.attempt_id=a.id GROUP BY a.id ORDER BY a.id''')],
                distinct_checkpoint_jobs=con.execute("SELECT COUNT(DISTINCT json_extract(detail_json,'$.job_id')) FROM calls WHERE kind='MODEL'").fetchone()[0],
                training_forbidden=True,logical_reads_are_reserved_artifact_bytes_not_OS_paging=True)
            attempts={a['id']:a for a in result['attempts']}
            for task in result['tasks']:
                if task['state']=='COMPLETED':
                    raw=task.pop('receipt_json')
                    require(isinstance(raw,str) and 0<len(raw.encode())<=256*1024 and _sha(task['receipt_sha']),
                            'Completed task lacks its bounded authenticated receipt')
                    task['receipt']=json.loads(raw)
                    require(digest(task['receipt'])==task['receipt_sha'],'Stored Phase A completion receipt digest changed')
                    receipt=task['receipt'];attempt=attempts.get(receipt.get('attempt_id'))
                    require(receipt.get('phase_plan_sha256')==plan['plan_sha256']
                            and receipt.get('task_key')==task['task_key'] and receipt.get('task_stage')==task['stage']
                            and type(receipt.get('attempt_id')) is int and attempt is not None
                            and attempt['task_key']==task['task_key'] and attempt['status']=='COMPLETED',
                            'Completed task receipt/attempt bindings differ')
                else:
                    require(task.pop('receipt_json') is None and task['receipt_sha'] is None,
                            'Uncompleted task cannot carry a completed receipt')
            con.execute('COMMIT');return result
        except BaseException:
            if con.in_transaction:con.execute('ROLLBACK')
            raise
        finally:
            if own:con.close()

    def task(self,key):
        require(_safe(key),'Exact task key required')
        return next((x for x in self.snapshot()['tasks'] if x['task_key']==key),None)
