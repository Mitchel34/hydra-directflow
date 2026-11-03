"""Durable local campaign accounting; no model, source-array or network API.

A worker holds ``writer()`` for its lifetime; a supervisor may call ``snapshot``
read-only. Entered updates and output reservations are committed before work and
never refunded. Losing the OS lease interrupts the same job, never starts a new
fit. Receipt validators are trusted integration boundaries: they must authenticate
the actual artifact bytes and their semantic identities, not just return a digest
of unverified JSON. This module does not implement a resource supervisor.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
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

SCHEMA = "hydra-campaign-ledger/v1"
GIB = 1024**3
MAX_LOGICAL_UPDATES = 18_200_800
MAX_LOGICAL_JOB_UPDATES = 126_800
# Three explicit resumes, each potentially replaying one ten-epoch interval.
MAX_UPDATES = 20_930_920
MAX_JOB_UPDATES = 145_820
MAX_RESUMES = 3
LIMITS = dict(max_optimizer_updates=MAX_UPDATES, max_job_seconds=6*3600,
              max_campaign_seconds=336*3600, max_output_bytes=8*GIB,
              min_free_bytes=8*GIB, max_combined_rss_bytes=4*GIB)
PHASES = (("cv_1",17),("cv_2",17),("cv_3",17),
          ("final_development",17),("final_development",29),("final_development",43))
GAUGES = ("03161000", "03164000", "03479000")
ARCHITECTURES = ("LSTM", "Transformer", "GRU_Transformer", "Mamba2_Transformer")
ARMS = (("full",168),("q_only",168),("full",24))


class LedgerError(ValueError):
    pass


class LedgerBusy(LedgerError):
    pass


class AllowanceExhausted(LedgerError):
    pass


def require(condition, message):
    if not condition:
        raise LedgerError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _hash(value):
    return isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value) is not None


def _sync(directory):
    fd=os.open(directory,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _directory(path):
    require(path.absolute()==path.resolve(), "Ledger paths must not traverse links or parent aliases")
    missing=[]; cursor=path
    while not cursor.exists():
        require(not cursor.is_symlink(),"Ledger directory link refused")
        missing.append(cursor); cursor=cursor.parent
    info=cursor.lstat()
    require(stat.S_ISDIR(info.st_mode) and not getattr(info,'st_flags',0)&0x40000000,
            "Ledger parent must be a resident real directory")
    for item in reversed(missing): item.mkdir(mode=0o700); _sync(item.parent)


def _regular(path):
    if path.exists() or path.is_symlink():
        info=path.lstat()
        require(stat.S_ISREG(info.st_mode) and not getattr(info,'st_flags',0)&0x40000000,
                "Ledger files must be resident regular files, never links")


def validate_plan(plan):
    require(isinstance(plan,dict) and _hash(plan.get('plan_sha256'))
            and plan['plan_sha256']==digest({k:v for k,v in plan.items() if k!='plan_sha256'}),
            "Immutable campaign plan digest differs")
    for name in ('protocol_sha256','source_plan_sha256','snapshot_sha256'):
        require(_hash(plan.get(name)), "Protocol/data/snapshot identity required")
    code=plan.get('implementation_sha256')
    require(isinstance(code,dict) and code and all(isinstance(k,str) and k and _hash(v) for k,v in code.items()),
            "Implementation identities required")
    limits=plan.get('limits')
    require(isinstance(limits,dict) and set(limits)==set(LIMITS), "Exact local resource limits required")
    for name,value in limits.items():
        require(type(value) is int and value>0, "Positive integer campaign limits required")
        require(value>=LIMITS[name] if name=='min_free_bytes' else value<=LIMITS[name],
                "Campaign guard cannot be enlarged or free-space floor lowered")
    require(plan.get('protected_assessment_access',False) is False
            and plan.get('paid_cloud_jobs_started',0)==0, "Protected/cloud execution is outside the ledger")
    expected=[(fold,seed,gauge,architecture,arm,length)
              for fold,seed in PHASES for gauge in GAUGES for architecture in ARCHITECTURES
              for arm,length in (ARMS if fold=='final_development' else ARMS[:2])]
    jobs=plan.get('jobs')
    require(isinstance(jobs,list) and len(jobs)==180, "Exact ordered 180-job roster required")
    ids=set()
    for order,(job,coordinates) in enumerate(zip(jobs,expected)):
        require(isinstance(job,dict) and tuple(job.get(k) for k in
            ('fold','seed','gauge_id','architecture','input_arm','history_hours'))==coordinates,
            "Campaign job geometry, seed or ordering differs")
        require(job.get('order')==order and job.get('device') in ('cpu','mps')
                and job.get('batch_size')==64 and job.get('precision')=='FP32', "Local job geometry/device differs")
        identity=job.get('job_id')
        require(isinstance(identity,str) and 0<len(identity)<=256 and identity not in ids,
                "Unique bounded job identity required")
        ids.add(identity)
        require(type(job.get('max_updates')) is int and 0<job['max_updates']<=MAX_JOB_UPDATES,
                "Per-job update ceiling exceeds the reviewed bound")
        logical=job.get('logical_update_limit',job['max_updates'])
        batches=job.get('gradient_batches_per_epoch')
        interval=job.get('checkpoint_interval_epochs'); resumes=job.get('max_resumes')
        require(type(logical) is int and 0<logical<=MAX_LOGICAL_JOB_UPDATES
                and type(batches) is int and 0<batches<=634
                and type(interval) is int and interval==10 and type(resumes) is int and resumes==MAX_RESUMES
                and logical<=job['max_updates']<=logical+resumes*interval*batches,
                "Explicit logical ceiling and bounded physical replay reserve required")
        role_hashes=job.get('role_case_ids_sha256')
        require(isinstance(role_hashes,dict) and set(role_hashes)=={'gradient_fit','inner_stopping','outer_scoring'}
                and all(_hash(v) for v in role_hashes.values()), "Exact source role membership hashes required")
    require(sum(j.get('logical_update_limit',j['max_updates']) for j in jobs)<=MAX_LOGICAL_UPDATES,
            "Logical campaign update ceiling exceeded")
    return json.loads(canonical(plan))


@dataclass(frozen=True)
class AttemptLease:
    job_id: str
    attempt_id: int
    lease_id: str


class CampaignLedger:
    """Single-worker persistent state. Construction alone performs no I/O."""
    def __init__(self,directory,*,clock=time.time):
        self.directory=Path(directory).absolute()
        self.database=self.directory/'campaign.sqlite3'
        self.lock_path=self.directory/'writer.lock'
        self.clock=clock; self.connection=None; self._fd=None; self._owner=None; self._pid=None; self.plan=None

    @contextmanager
    def writer(self):
        require(self.connection is None, "Writer contexts cannot be nested")
        _directory(self.directory); _regular(self.lock_path)
        fd=os.open(self.lock_path,os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
        try:
            try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error: raise LedgerBusy("Campaign writer lease is already held") from error
            _sync(self.directory)
            for suffix in ('','-wal','-shm'): _regular(Path(str(self.database)+suffix))
            self._fd,self._owner,self._pid=fd,uuid.uuid4().hex,os.getpid()
            self.connection=sqlite3.connect(self.database,timeout=0,isolation_level=None)
            self.connection.row_factory=sqlite3.Row
            for setting in ('journal_mode=WAL','synchronous=FULL','fullfsync=ON','checkpoint_fullfsync=ON','foreign_keys=ON'):
                self.connection.execute('PRAGMA '+setting)
            self._schema()
            row=self.connection.execute('SELECT plan_json FROM campaign WHERE id=1').fetchone()
            if row:
                self.plan=validate_plan(json.loads(row[0])); self._reconcile(); self._orphan_active()
            try: yield self
            finally:
                if self.plan is not None: self._orphan_active(reason='LEASE_RELEASED')
        finally:
            if self.connection is not None: self.connection.close()
            self.connection=None; self._fd=self._owner=self._pid=None; self.plan=None
            fcntl.flock(fd,fcntl.LOCK_UN); os.close(fd)

    def _schema(self):
        self.connection.executescript('''
        CREATE TABLE IF NOT EXISTS campaign(id INTEGER PRIMARY KEY CHECK(id=1),schema TEXT NOT NULL,
          plan_json TEXT NOT NULL,started REAL,deadline REAL,last_clock REAL,entered INTEGER NOT NULL DEFAULT 0,
          confirmed INTEGER NOT NULL DEFAULT 0,output_reserved INTEGER NOT NULL DEFAULT 0,
          peak_rss INTEGER NOT NULL DEFAULT 0,minimum_free INTEGER);
        CREATE TABLE IF NOT EXISTS jobs(position INTEGER PRIMARY KEY,job_id TEXT UNIQUE NOT NULL,job_json TEXT NOT NULL,
          state TEXT NOT NULL,entered INTEGER NOT NULL DEFAULT 0,confirmed INTEGER NOT NULL DEFAULT 0,
          started REAL,deadline REAL,completion_json TEXT);
        CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY,job_position INTEGER NOT NULL REFERENCES jobs(position),
          lease_id TEXT NOT NULL,status TEXT NOT NULL,started REAL NOT NULL,finished REAL,error_type TEXT,
          resume_logical_updates INTEGER NOT NULL,entered INTEGER NOT NULL DEFAULT 0,confirmed INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS updates(id INTEGER PRIMARY KEY,attempt_id INTEGER NOT NULL REFERENCES attempts(id),
          logical_update INTEGER NOT NULL,entered_at REAL NOT NULL,confirmed_at REAL);
        CREATE TABLE IF NOT EXISTS artifacts(id INTEGER PRIMARY KEY,job_position INTEGER NOT NULL REFERENCES jobs(position),
          attempt_id INTEGER NOT NULL REFERENCES attempts(id),kind TEXT NOT NULL,receipt_sha TEXT NOT NULL,
          receipt_json TEXT NOT NULL,logical_updates INTEGER NOT NULL,created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS outputs(id INTEGER PRIMARY KEY,attempt_id INTEGER REFERENCES attempts(id),
          label TEXT NOT NULL,reserved INTEGER NOT NULL,status TEXT NOT NULL,actual INTEGER,sha256 TEXT);
        ''')
        _sync(self.directory)

    def _held(self):
        require(self.connection is not None and self._pid==os.getpid() and self._fd is not None,
                "A live same-process OS writer lease is required")
        try:
            fd=os.fstat(self._fd); path=self.lock_path.lstat()
            require((fd.st_dev,fd.st_ino)==(path.st_dev,path.st_ino), "Writer lock identity changed")
        except OSError as error: raise LedgerError("Writer lock was lost") from error

    @contextmanager
    def _transaction(self):
        self._held(); self.connection.execute('BEGIN IMMEDIATE')
        try: yield; self.connection.execute('COMMIT')
        except BaseException:
            self.connection.execute('ROLLBACK'); raise

    def initialize(self,immutable_plan):
        plan=validate_plan(immutable_plan)
        if self.connection is None:
            with self.writer(): return self.initialize(plan)
        self._held()
        with self._transaction():
            old=self.connection.execute('SELECT schema,plan_json FROM campaign WHERE id=1').fetchone()
            if old:
                require(old['schema']==SCHEMA and old['plan_json']==canonical(plan),
                        "Existing campaign identity or allowance cannot be reset")
            else:
                self.connection.execute('INSERT INTO campaign(id,schema,plan_json) VALUES(1,?,?)',(SCHEMA,canonical(plan)))
                self.connection.executemany('INSERT INTO jobs(position,job_id,job_json,state) VALUES(?,?,?,?)',
                    [(i,j['job_id'],canonical(j),'PENDING') for i,j in enumerate(plan['jobs'])])
        _sync(self.directory); self.plan=plan; self._reconcile()
        return dict(plan_sha256=plan['plan_sha256'],jobs=180,status='INITIALIZED_NO_FITS')

    def _reconcile(self):
        self._held(); require(self.plan is not None,"Initialize the immutable plan first")
        campaign=self.connection.execute('SELECT * FROM campaign WHERE id=1').fetchone()
        require(campaign['schema']==SCHEMA,"Unknown ledger schema")
        jobs=self.connection.execute('SELECT * FROM jobs ORDER BY position').fetchall()
        require(len(jobs)==180 and all(r['position']==i and r['job_id']==j['job_id'] and r['job_json']==canonical(j)
                for i,(r,j) in enumerate(zip(jobs,self.plan['jobs']))),"Persistent job roster changed")
        require(sum(r['entered'] for r in jobs)==campaign['entered'] and sum(r['confirmed'] for r in jobs)==campaign['confirmed']
                and 0<=campaign['confirmed']<=campaign['entered']<=self.plan['limits']['max_optimizer_updates'],
                "Persistent update accounting does not reconcile")
        totals={r[0]:(r[1],r[2]) for r in self.connection.execute(
            'SELECT job_position,SUM(entered),SUM(confirmed) FROM attempts GROUP BY job_position')}
        for row,job in zip(jobs,self.plan['jobs']):
            require(totals.get(row['position'],(0,0))==(row['entered'],row['confirmed'])
                    and 0<=row['confirmed']<=row['entered']<=job['max_updates'],"Job/attempt accounting differs")
        require(all(row['state'] in ('PENDING','ACTIVE','INTERRUPTED','FAILED','VERIFIED_TRAIN_EVAL') for row in jobs),
                "Unknown job state")
        first=next((i for i,r in enumerate(jobs) if r['state']!='VERIFIED_TRAIN_EVAL'),len(jobs))
        require(all(r['state']=='PENDING' for r in jobs[first+1:]),"Later jobs cannot bypass incomplete predecessors")
        for row in jobs[:first]:
            require(row['completion_json'] is not None,"Verified state lacks its authenticated completion")
            receipt=json.loads(row['completion_json'])
            require(all(receipt.get(k)==v for k,v in self._bindings(row).items())
                    and receipt.get('status')=='VERIFIED_TRAIN_EVAL'
                    and set(receipt.get('hashes',{}))=={'checkpoint','predictions','metrics'}
                    and all(_hash(v) for v in receipt['hashes'].values()),"Completion evidence identity differs")
            require(self.connection.execute("SELECT 1 FROM artifacts WHERE job_position=? AND kind='COMPLETE' AND receipt_sha=? AND receipt_json=?",
                    (row['position'],digest(receipt),row['completion_json'])).fetchone() is not None,
                    "Verified completion lacks its immutable artifact record")
        reserved=self.connection.execute('SELECT COALESCE(SUM(reserved),0) FROM outputs').fetchone()[0]
        require(reserved==campaign['output_reserved']<=self.plan['limits']['max_output_bytes'],"Output reservations differ")

    def _orphan_active(self,reason='LOST_LEASE'):
        self._held()
        with self._transaction():
            now=float(self.clock())
            active=self.connection.execute("SELECT id,job_position FROM attempts WHERE status='ACTIVE'").fetchall()
            for row in active:
                self.connection.execute("UPDATE attempts SET status='INTERRUPTED',finished=?,error_type=? WHERE id=?",
                                        (now,reason,row['id']))
                self.connection.execute("UPDATE jobs SET state='INTERRUPTED' WHERE position=?",(row['job_position'],))

    def _time(self,job=None):
        now=self.clock(); require(type(now) in (int,float) and math.isfinite(now),"Finite ledger clock required")
        row=self.connection.execute('SELECT * FROM campaign WHERE id=1').fetchone()
        require(row is not None,"Campaign is not initialized")
        require(row['last_clock'] is None or now>=row['last_clock'],"Wall clock moved backwards; do not extend an allowance")
        if row['deadline'] is not None and now>=row['deadline']: raise AllowanceExhausted('Campaign wall allowance expired')
        if job is not None and job['deadline'] is not None and now>=job['deadline']: raise AllowanceExhausted('Same-job wall allowance expired')
        return float(now)

    def _first(self):
        return self.connection.execute("SELECT * FROM jobs WHERE state!='VERIFIED_TRAIN_EVAL' ORDER BY position LIMIT 1").fetchone()

    def _start(self,job,now,resume_updates=0):
        require(job['entered']<=json.loads(job['job_json'])['max_updates'],"Same-job update allowance exceeded")
        entered=self.connection.execute('SELECT entered FROM campaign WHERE id=1').fetchone()[0]
        if entered>=self.plan['limits']['max_optimizer_updates'] and not resume_updates:
            raise AllowanceExhausted('Campaign optimizer allowance exhausted before another fit')
        if job['started'] is None:
            self.connection.execute('UPDATE jobs SET started=?,deadline=? WHERE position=?',
                (now,now+self.plan['limits']['max_job_seconds'],job['position']))
        campaign=self.connection.execute('SELECT started FROM campaign WHERE id=1').fetchone()
        if campaign[0] is None:
            self.connection.execute('UPDATE campaign SET started=?,deadline=? WHERE id=1',
                (now,now+self.plan['limits']['max_campaign_seconds']))
        self.connection.execute('UPDATE campaign SET last_clock=? WHERE id=1',(now,))
        cur=self.connection.execute("INSERT INTO attempts(job_position,lease_id,status,started,resume_logical_updates) VALUES(?,?,'ACTIVE',?,?)",
                                    (job['position'],self._owner,now,resume_updates))
        self.connection.execute("UPDATE jobs SET state='ACTIVE' WHERE position=?",(job['position'],))
        return AttemptLease(job['job_id'],cur.lastrowid,self._owner)

    def claim_next(self):
        self._held(); require(self.plan is not None,"Initialize first")
        with self._transaction():
            job=self._first()
            if job is None: return None
            require(job['state']=='PENDING',"Earlier job requires explicit same-job recovery or failure review")
            return self._start(job,self._time(job))

    def _active(self,lease,*,time_check=True):
        self._held(); require(type(lease) is AttemptLease and type(lease.attempt_id) is int and lease.attempt_id>0
                and lease.lease_id==self._owner,"Stale or foreign attempt lease")
        row=self.connection.execute('SELECT * FROM attempts WHERE id=?',(lease.attempt_id,)).fetchone()
        require(row is not None and row['lease_id']==self._owner and row['status']=='ACTIVE',"Attempt is not active under this lease")
        job=self.connection.execute('SELECT * FROM jobs WHERE position=?',(row['job_position'],)).fetchone()
        require(job['job_id']==lease.job_id and job['state']=='ACTIVE',"Attempt/job state differs")
        if time_check: self._time(job)
        return row,job

    def enter_update(self,lease):
        """Return only after durable precharge; invoke optimizer.step afterward."""
        with self._transaction():
            attempt,job=self._active(lease); now=self._time(job)
            require(attempt['entered']==attempt['confirmed'],"Unconfirmed optimizer entry requires interruption and explicit recovery")
            spec=json.loads(job['job_json']); cap=spec['max_updates']
            total=self.connection.execute('SELECT entered FROM campaign WHERE id=1').fetchone()[0]
            if job['entered']>=cap or total>=self.plan['limits']['max_optimizer_updates']:
                raise AllowanceExhausted('Persistent optimizer allowance exhausted; no refund or reset')
            logical=attempt['resume_logical_updates']+attempt['confirmed']+1
            require(logical<=spec.get('logical_update_limit',cap),"Logical optimizer ceiling exceeded")
            token=self.connection.execute('INSERT INTO updates(attempt_id,logical_update,entered_at) VALUES(?,?,?)',
                                          (lease.attempt_id,logical,now)).lastrowid
            self.connection.execute('UPDATE attempts SET entered=entered+1 WHERE id=?',(lease.attempt_id,))
            self.connection.execute('UPDATE jobs SET entered=entered+1 WHERE position=?',(job['position'],))
            self.connection.execute('UPDATE campaign SET entered=entered+1,last_clock=? WHERE id=1',(now,))
        return token

    def confirm_update(self,lease,token):
        require(type(token) is int and token>0,"Positive update token required")
        with self._transaction():
            attempt,job=self._active(lease,time_check=False)
            row=self.connection.execute('SELECT * FROM updates WHERE id=?',(token,)).fetchone()
            require(row is not None and row['attempt_id']==lease.attempt_id and row['confirmed_at'] is None
                    and attempt['entered']==attempt['confirmed']+1,"Update token already settled or belongs to another attempt")
            now=float(self.clock())
            self.connection.execute('UPDATE updates SET confirmed_at=? WHERE id=?',(now,token))
            self.connection.execute('UPDATE attempts SET confirmed=confirmed+1 WHERE id=?',(lease.attempt_id,))
            self.connection.execute('UPDATE jobs SET confirmed=confirmed+1 WHERE position=?',(job['position'],))
            self.connection.execute('UPDATE campaign SET confirmed=confirmed+1,last_clock=MAX(COALESCE(last_clock,?),?) WHERE id=1',(now,now))

    def receipt_bindings(self,lease):
        _,job=self._active(lease,time_check=False)
        return self._bindings(job)

    def _bindings(self,job):
        return dict(plan_sha256=self.plan['plan_sha256'],protocol_sha256=self.plan['protocol_sha256'],
            source_plan_sha256=self.plan['source_plan_sha256'],snapshot_sha256=self.plan['snapshot_sha256'],
            implementation_sha256=self.plan['implementation_sha256'],job_id=job['job_id'],job_sha256=digest(json.loads(job['job_json'])))

    def _evidence(self,receipt,job,validator,kind):
        require(isinstance(receipt,dict) and callable(validator),"An artifact-authenticating receipt validator is mandatory")
        bindings=self._bindings(job)
        require(all(receipt.get(k)==v for k,v in bindings.items()),"Receipt job/protocol/data/implementation identity differs")
        expected={'checkpoint'} if kind=='CHECKPOINT' else {'checkpoint','predictions','metrics'}
        require(isinstance(receipt.get('hashes'),dict) and set(receipt['hashes'])==expected
                and all(_hash(x) for x in receipt['hashes'].values()),"All authenticated artifact hashes are required")
        require(receipt.get('status')==('VERIFIED_CHECKPOINT' if kind=='CHECKPOINT' else 'VERIFIED_TRAIN_EVAL')
                and type(receipt.get('logical_updates')) is int and receipt['logical_updates']>=0,
                "Verified artifact state and logical update count required")
        expected_sha=digest(receipt)
        require(validator(json.loads(canonical(receipt)),json.loads(canonical(bindings)))==expected_sha,
                "Artifact verification did not authenticate the exact receipt")
        return expected_sha

    def checkpoint(self,lease,receipt,*,validator):
        attempt,job=self._active(lease)
        require(attempt['entered']==attempt['confirmed'] and receipt.get('logical_updates')==attempt['resume_logical_updates']+attempt['confirmed'],
                "Checkpoint is not the confirmed model-state prefix")
        receipt_sha=self._evidence(receipt,job,validator,'CHECKPOINT')
        with self._transaction():
            current,_=self._active(lease); now=self._time(job)
            require(current['entered']==current['confirmed'] and receipt['logical_updates']==current['resume_logical_updates']+current['confirmed'],
                    "Model prefix changed during checkpoint verification")
            self.connection.execute('INSERT INTO artifacts(job_position,attempt_id,kind,receipt_sha,receipt_json,logical_updates,created) VALUES(?,?,?,?,?,?,?)',
                (job['position'],lease.attempt_id,'CHECKPOINT',receipt_sha,canonical(receipt),receipt['logical_updates'],now))
        return receipt_sha

    def resume(self,job_id,checkpoint_receipt,*,validator):
        self._held(); job=self._first()
        require(job is not None and job['job_id']==job_id and job['state']=='INTERRUPTED',
                "Only the first interrupted job can resume; failed/integrity/nonfinite jobs require review")
        now=self._time(job); receipt_sha=self._evidence(checkpoint_receipt,job,validator,'CHECKPOINT')
        recorded=self.connection.execute("SELECT 1 FROM artifacts WHERE job_position=? AND kind='CHECKPOINT' AND receipt_sha=? AND receipt_json=?",
                                        (job['position'],receipt_sha,canonical(checkpoint_receipt))).fetchone()
        require(recorded is not None,"Resume checkpoint was never authenticated and recorded for this job")
        latest=self.connection.execute("SELECT receipt_sha FROM artifacts WHERE job_position=? AND kind='CHECKPOINT' ORDER BY logical_updates DESC,id DESC LIMIT 1",
                                       (job['position'],)).fetchone()
        require(latest is not None and latest[0]==receipt_sha,"Resume requires the latest highest-progress checkpoint; no older fallback")
        count=self.connection.execute('SELECT COUNT(*) FROM attempts WHERE job_position=?',(job['position'],)).fetchone()[0]
        require(count<1+json.loads(job['job_json'])['max_resumes'],"Explicit same-job resume allowance exhausted")
        with self._transaction(): return self._start(job,self._time(job),checkpoint_receipt['logical_updates'])

    def interrupt(self,lease,reason='INTERRUPTED'):
        self._settle_attempt(lease,'INTERRUPTED',reason)

    def fail(self,lease,error_type):
        """Failure is terminal to automatic scheduling; it is never a retry."""
        self._settle_attempt(lease,'FAILED',error_type)

    def _settle_attempt(self,lease,state,reason):
        require(isinstance(reason,str) and re.fullmatch(r'[A-Z0-9_]{1,96}',reason),"Bounded sanitized failure reason required")
        with self._transaction():
            _,job=self._active(lease,time_check=False)
            self.connection.execute('UPDATE attempts SET status=?,finished=?,error_type=? WHERE id=?',
                                    (state,float(self.clock()),reason,lease.attempt_id))
            self.connection.execute('UPDATE jobs SET state=? WHERE position=?',(state,job['position']))

    def complete(self,lease,receipt,*,validator):
        attempt,job=self._active(lease)
        require(attempt['entered']==attempt['confirmed'] and receipt.get('logical_updates')==attempt['resume_logical_updates']+attempt['confirmed']
                and receipt['logical_updates']>0,"Completion requires a positive confirmed model-state prefix")
        receipt_sha=self._evidence(receipt,job,validator,'COMPLETE')
        with self._transaction():
            current,_=self._active(lease); now=self._time(job)
            require(current['entered']==current['confirmed'] and receipt['logical_updates']==current['resume_logical_updates']+current['confirmed'],
                    "Model prefix changed during completion verification")
            require(self._first()['position']==job['position'],"All preceding jobs must have verified training and evaluation")
            require(self.connection.execute("SELECT COUNT(*) FROM outputs WHERE attempt_id=? AND status='PRECHARGED'",(lease.attempt_id,)).fetchone()[0]==0,
                    "Unsettled output reservations must be preserved and settled before completion")
            self.connection.execute('INSERT INTO artifacts(job_position,attempt_id,kind,receipt_sha,receipt_json,logical_updates,created) VALUES(?,?,?,?,?,?,?)',
                (job['position'],lease.attempt_id,'COMPLETE',receipt_sha,canonical(receipt),receipt['logical_updates'],now))
            self.connection.execute("UPDATE attempts SET status='VERIFIED_TRAIN_EVAL',finished=? WHERE id=?",(now,lease.attempt_id))
            self.connection.execute("UPDATE jobs SET state='VERIFIED_TRAIN_EVAL',completion_json=? WHERE position=?",(canonical(receipt),job['position']))
        return receipt_sha

    def check_resources(self,lease,*,free_bytes,combined_rss_bytes):
        self._active(lease)
        require(type(free_bytes) is int and type(combined_rss_bytes) is int and free_bytes>=0 and combined_rss_bytes>=0,
                "Explicit nonnegative resource observations required")
        limits=self.plan['limits']
        if free_bytes<limits['min_free_bytes'] or combined_rss_bytes>limits['max_combined_rss_bytes']:
            raise AllowanceExhausted('Free disk or combined host/MPS resource guard reached')
        with self._transaction():
            self.connection.execute('UPDATE campaign SET peak_rss=MAX(peak_rss,?),minimum_free=MIN(COALESCE(minimum_free,?),?) WHERE id=1',
                                    (combined_rss_bytes,free_bytes,free_bytes))

    def reserve_output(self,bytes_count,lease=None,*,kind):
        """Reserve before writing; lease=None reserves global bookkeeping/scratch."""
        require(type(bytes_count) is int and bytes_count>0 and isinstance(kind,str)
                and re.fullmatch(r'[A-Za-z0-9_.-]{1,96}',kind),"Positive output reservation and safe kind required")
        with self._transaction():
            require(self.plan is not None,"Initialize first")
            if lease is not None:
                _,job=self._active(lease)
                if kind=='CHECKPOINT':
                    count=self.connection.execute("SELECT COUNT(*) FROM outputs o JOIN attempts a ON a.id=o.attempt_id WHERE a.job_position=? AND o.label='CHECKPOINT'",
                                                  (job['position'],)).fetchone()[0]
                    require(count<21,"At most21 checkpoint write reservations per fit; failed/partial writes remain counted")
            total=self.connection.execute('SELECT output_reserved FROM campaign WHERE id=1').fetchone()[0]
            if total+bytes_count>self.plan['limits']['max_output_bytes']: raise AllowanceExhausted('Persistent output allowance exhausted')
            self.connection.execute('UPDATE campaign SET output_reserved=output_reserved+? WHERE id=1',(bytes_count,))
            return self.connection.execute("INSERT INTO outputs(attempt_id,label,reserved,status) VALUES(?,?,?,'PRECHARGED')",
                                           (lease.attempt_id if lease else None,kind,bytes_count)).lastrowid

    def settle_output(self,lease,token,*,actual_bytes,sha256=None,failed=False):
        require(type(token) is int and token>0 and type(actual_bytes) is int and actual_bytes>=0 and type(failed) is bool
                and (sha256 is None if failed else _hash(sha256)),"Exact output settlement required")
        with self._transaction():
            if lease is not None: self._active(lease,time_check=False)
            row=self.connection.execute('SELECT * FROM outputs WHERE id=?',(token,)).fetchone()
            require(row is not None and row['attempt_id']==(lease.attempt_id if lease else None) and row['status']=='PRECHARGED'
                    and actual_bytes<=row['reserved'],"Output reservation differs or is already settled")
            self.connection.execute('UPDATE outputs SET status=?,actual=?,sha256=? WHERE id=?',
                ('FAILED' if failed else 'VERIFIED',actual_bytes,sha256,token))

    @contextmanager
    def _reader(self):
        if self.connection is not None:
            self._held(); yield self.connection; return
        require(self.directory==self.directory.resolve(),"Ledger directory link refused")
        for suffix in ('','-wal','-shm'): _regular(Path(str(self.database)+suffix))
        connection=sqlite3.connect('file:'+quote(str(self.database),safe='/')+'?mode=ro',uri=True,isolation_level=None)
        connection.row_factory=sqlite3.Row
        try:
            connection.execute('BEGIN'); yield connection; connection.execute('COMMIT')
        except BaseException:
            if connection.in_transaction: connection.execute('ROLLBACK')
            raise
        finally: connection.close()

    def next_job(self):
        """Inspect the first unfinished job; never claim it or settle a lease."""
        with self._reader() as connection:
            row=connection.execute("SELECT * FROM jobs WHERE state!='VERIFIED_TRAIN_EVAL' ORDER BY position LIMIT 1").fetchone()
            if row is None: return None
            return dict(job=json.loads(row['job_json']),**{k:row[k] for k in ('state','entered','confirmed','started','deadline')})

    def checkpoints(self,job_id):
        """Recorded checkpoint receipt metadata only; artifact bodies remain private."""
        with self._reader() as connection:
            job=connection.execute('SELECT position FROM jobs WHERE job_id=?',(job_id,)).fetchone()
            require(job is not None,"Unknown planned job")
            rows=connection.execute("SELECT receipt_json,receipt_sha FROM artifacts WHERE job_position=? AND kind='CHECKPOINT' ORDER BY logical_updates,id",
                                    (job[0],)).fetchall()
            result=[]
            for row in rows:
                receipt=json.loads(row['receipt_json'])
                require(digest(receipt)==row['receipt_sha'],"Recorded checkpoint metadata digest changed")
                result.append(receipt)
            return result

    def latest_checkpoint(self,job_id):
        rows=self.checkpoints(job_id)
        return rows[-1] if rows else None

    def snapshot(self):
        """Read-only consistent metadata view; never settle, claim or create files."""
        own=self.connection is None
        if own:
            require(self.directory.absolute()==self.directory.resolve(),"Ledger directory link refused")
            for suffix in ('','-wal','-shm'): _regular(Path(str(self.database)+suffix))
            connection=sqlite3.connect('file:'+quote(str(self.database),safe='/')+'?mode=ro',uri=True,isolation_level=None)
            connection.row_factory=sqlite3.Row
        else: self._held(); connection=self.connection
        try:
            connection.execute('BEGIN')
            campaign=connection.execute('SELECT * FROM campaign WHERE id=1').fetchone()
            require(campaign is not None,"Campaign is not initialized")
            plan=validate_plan(json.loads(campaign['plan_json'])); jobs=[dict(x) for x in connection.execute('SELECT position,job_id,state,entered,confirmed,started,deadline FROM jobs ORDER BY position')]
            require(len(jobs)==180 and all(row['position']==i and row['job_id']==spec['job_id']
                    for i,(row,spec) in enumerate(zip(jobs,plan['jobs']))),"Snapshot roster is incomplete or changed")
            attempts=[dict(x) for x in connection.execute('SELECT id,job_position,status,started,finished,error_type,resume_logical_updates,entered,confirmed FROM attempts ORDER BY id')]
            ready=[]
            for seed,through in ((17,108),(29,144),(43,180)):
                if all(j['state']=='VERIFIED_TRAIN_EVAL' for j in jobs[:through]): ready.append(seed)
            result=dict(schema=SCHEMA,plan_sha256=plan['plan_sha256'],jobs=jobs,attempts=attempts,
                optimizer_updates_entered=campaign['entered'],optimizer_updates_confirmed=campaign['confirmed'],
                output_bytes_reserved=campaign['output_reserved'],campaign_started_at=campaign['started'],
                bookkeeping_bytes_reserved=connection.execute("SELECT COALESCE(SUM(reserved),0) FROM outputs WHERE attempt_id IS NULL AND label='BOOKKEEPING'").fetchone()[0],
                campaign_deadline=campaign['deadline'],seed_reports_ready=ready,complete=len(ready)==3,
                limits=plan['limits'],observed_peak_combined_rss_bytes=campaign['peak_rss'],observed_minimum_free_bytes=campaign['minimum_free'],
                qualification='Ledger metadata; callback authenticates artifacts, external supervisor enforces measured resources')
            connection.execute('COMMIT'); return result
        except BaseException:
            if connection.in_transaction: connection.execute('ROLLBACK')
            raise
        finally:
            if own: connection.close()
