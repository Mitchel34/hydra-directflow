#!/usr/bin/env python3
"""Recover an exact qualified development corpus from existing owned files.

Default is plan-only. Opaque copies use a persistent source-read budget and one
attempt per file. A separately recorded single post-copy pass checks the
development corpus; no acquisition, census, model or protected-data work occurs.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import stat
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
SOURCE_PLAN='0dad073824984fd7eeba88e4c313d77b4015dcccb94e8eb27297fcf7c6b39b0f'
SNAPSHOT='b41630b7df9a216874eb4ba38a10176d4b72092bd6f1ce2ad4c4f369db56948c'
MAX_BODY=1024**3;MAX_SECONDS=900;MAX_RSS=4*1024**3;PER_FILE_SECONDS=45
SCHEMA='hydra-owned-development-residency-recovery/v1'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def utc():return datetime.now(timezone.utc).isoformat()
def hex64(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def directory(path,create=False):
    path=Path(path)
    if create and not path.exists():
        directory(path.parent,True);path.mkdir();sync_directory(path.parent)
    info=path.lstat()
    require(stat.S_ISDIR(info.st_mode) and not getattr(info,'st_flags',0)&0x40000000,'Resident real directory required')
    return path


def guarded_parents(path,root):
    path,root=Path(path),Path(root)
    require(path.is_relative_to(root),'Path escaped approved root')
    directory(root)
    for parent in reversed(path.parent.relative_to(root).parents):
        if str(parent)!='.':directory(root/parent)
    if path.parent!=root:directory(path.parent)


def read_resident(path,maximum):
    path=Path(path);before=path.lstat()
    require(stat.S_ISREG(before.st_mode) and not getattr(before,'st_flags',0)&0x40000000
            and 0<before.st_size<=maximum,'Resident bounded regular metadata required')
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    with os.fdopen(fd,'rb') as stream:raw=stream.read(maximum+1)
    after=path.lstat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
            and len(raw)==before.st_size,'Metadata changed during read')
    return raw


def write_new_json(path,value):
    path=Path(path);directory(path.parent)
    require(not path.exists() and not path.is_symlink(),'Fresh report required')
    temporary=path.with_name(path.name+'.writing');raw=canonical(value)+b'\n'
    with temporary.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    try:os.link(temporary,path);sync_directory(path.parent)
    finally:temporary.unlink()
    return sha(raw)


def validate_manifest(value):
    require(value.get('schema')=='hydra-qualified-development-corpus-residency-recovery-plan/v1'
        and value.get('status')=='PLAN_ONLY_MODEL_PAYLOAD_COPY_NOT_EXECUTED'
        and value['source_plan_sha256']==SOURCE_PLAN and value['snapshot_sha256']==SNAPSHOT,
        'Exact qualified development source/snapshot required')
    require(value['plan_content_sha256']==sha(canonical({k:v for k,v in value.items() if k!='plan_content_sha256'})),
            'Recovery plan content identity differs')
    protect=value['protection']
    require(protect['all_H18_targets_strictly_before_utc']=='2024-07-31T00:00:00Z'
        and protect['original36_boundary_purges_preserved'] is True
        and not any(protect[k] for k in ('raw_source_archives_included','monthly_parquet_included','derived_cache_values_included','assessment_files_included')),
        'Development-only protection scope differs')
    source,target=Path(value['source_root']),Path(value['target_root'])
    require(source.is_absolute() and target.is_absolute() and source!=target
            and source.name==target.name=='development-corpus-L168-v4'
            and not target.is_relative_to(source),'Separate exact corpus roots required')
    require(target.is_relative_to(Path.home()/'.codex/hydra-directflow-runtime'),'Recovery target must stay outside cloud Documents in the local runtime')
    rows=value['files'];require(isinstance(rows,list) and len(rows)==1432,'Exact1432-file whitelist required')
    seen=set();cache_root=None;jobs=[]
    for row in rows:
        relative=row['relative_path']
        require(relative not in seen and (relative=='PLAN.json' or re.fullmatch(r'chunks/[A-Za-z0-9_-]+/\d{3}/(?:COMMIT\.json|case-index\.json|model\.npz)',relative)),
                'Unsafe/duplicate corpus relative path')
        seen.add(relative)
        require(row['source']==str(source/relative) and row['destination']==str(target/relative)
                and hex64(row['sha256']) and type(row['bytes']) is int and 0<row['bytes']<=16*1024**2,
                'Exact source/destination/byte/hash identity required')
        if relative.endswith('model.npz'):
            require(row['kind']=='DEVELOPMENT_MODEL_NPZ' and row['verified_metadata_cache'] is None,'Only declared model payloads use owned cloud files')
            selected=source/relative;selected_root=source
        else:
            require(row['kind'] in ('PLAN_METADATA','COMMIT_METADATA','CASE_INDEX_METADATA') and row['verified_metadata_cache'], 'Authenticated metadata copy required')
            selected=Path(row['verified_metadata_cache']);relative_parts=Path(relative).parts
            selected_root=selected.parents[len(relative_parts)-1]
            require(selected_root.is_relative_to(Path.home()/'.codex/hydra-directflow-runtime') and selected==selected_root/relative,'Metadata copy escaped its local root')
            if cache_root is None:cache_root=selected_root
            require(selected_root==cache_root,'One authenticated metadata cache root required')
        jobs.append(dict(relative=relative,source=str(selected),source_root=str(selected_root),target=str(target/relative),
                         expected_bytes=row['bytes'],expected_sha256=row['sha256'],kind=row['kind']))
    require(sum(r['expected_bytes'] for r in jobs)==value['totals']['exact_source_bytes']==75805012
            and sum(r['kind']=='DEVELOPMENT_MODEL_NPZ' for r in jobs)==477,'Exact qualified byte/payload totals differ')
    return jobs


def load_plan(path,expected_file_sha,expected_content_sha):
    require(hex64(expected_file_sha) and hex64(expected_content_sha),'Explicit approved plan identities required')
    raw=read_resident(path,4*1024**2);require(sha(raw)==expected_file_sha,'Approved plan file SHA differs')
    value=json.loads(raw);require(value['plan_content_sha256']==expected_content_sha,'Approved plan content SHA differs')
    return value,validate_manifest(value)


def connect(control):
    control=directory(control);path=control/'transfer.sqlite3'
    for candidate in (path,path.with_name(path.name+'-wal'),path.with_name(path.name+'-shm')):
        if candidate.exists() or candidate.is_symlink():
            info=candidate.lstat();require(stat.S_ISREG(info.st_mode) and not getattr(info,'st_flags',0)&0x40000000,'Resident regular transfer database required')
    db=sqlite3.connect(path,timeout=10);db.row_factory=sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');return db


@contextmanager
def lease(control):
    directory(control,True);path=Path(control)/'controller.lock'
    fd=os.open(path,os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode),'Regular recovery lease required')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);yield fd
    finally:os.close(fd)


def verify_lease(control,fd):
    actual=os.fstat(fd);expected=(Path(control)/'controller.lock').lstat()
    require(stat.S_ISREG(actual.st_mode) and (actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino),'Inherited controller lease differs')
    probe=os.open(Path(control)/'controller.lock',os.O_RDWR|getattr(os,'O_NOFOLLOW',0))
    try:
        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        raise ValueError('Controller lease is not held')
    finally:os.close(probe)


def initialize(db,context,jobs):
    db.executescript('CREATE TABLE IF NOT EXISTS context(id INTEGER PRIMARY KEY,data TEXT NOT NULL,charged INTEGER NOT NULL,verification TEXT NOT NULL);'
        'CREATE TABLE IF NOT EXISTS files(relative TEXT PRIMARY KEY,data TEXT NOT NULL,status TEXT NOT NULL,token TEXT,reserved INTEGER NOT NULL DEFAULT0,observed INTEGER NOT NULL DEFAULT0,error_type TEXT);'.replace('DEFAULT0','DEFAULT 0'))
    saved=db.execute('SELECT data FROM context WHERE id=1').fetchone()
    if saved:
        old=json.loads(saved['data'])
        require(all(old[k]==context[k] for k in context if k!='deadline_epoch'),'Existing recovery context differs; no reset')
        return old
    with db:
        db.execute('INSERT INTO context VALUES(1,?,0,?)',(canonical(context).decode(),'PENDING'))
        db.executemany('INSERT INTO files(relative,data,status) VALUES(?,?,?)',[(job['relative'],canonical(job).decode(),'PENDING') for job in jobs])
    return context


def reserve(db,relative):
    db.execute('BEGIN IMMEDIATE')
    try:
        row=db.execute('SELECT * FROM files WHERE relative=?',(relative,)).fetchone();require(row is not None,'Unknown file')
        if row['status']!='PENDING':db.rollback();return None
        context=db.execute('SELECT * FROM context WHERE id=1').fetchone();value=json.loads(context['data']);job=json.loads(row['data'])
        charge=job['expected_bytes']+1
        require(time.time()<value['deadline_epoch'] and context['charged']+charge<=value['body_cap'],'Recovery time/source-read budget exhausted')
        token=uuid.uuid4().hex
        db.execute('UPDATE context SET charged=charged+? WHERE id=1',(charge,))
        db.execute('UPDATE files SET status=?,token=?,reserved=? WHERE relative=?',('PRECHARGED',token,charge,relative));db.commit()
        return token
    except BaseException:db.rollback();raise


def claim(db,relative,token):
    db.execute('BEGIN IMMEDIATE')
    try:
        row=db.execute('SELECT * FROM files WHERE relative=?',(relative,)).fetchone()
        require(row is not None and row['status']=='PRECHARGED' and row['token']==token,'Unreserved/replayed source attempt rejected')
        context=json.loads(db.execute('SELECT data FROM context WHERE id=1').fetchone()['data'])
        require(time.time()<context['deadline_epoch'],'Source attempt deadline expired')
        require(context['copier_sha256']==sha(Path(__file__).read_bytes()),'Copier implementation changed')
        db.execute('UPDATE files SET status=? WHERE relative=?',('CLAIMED',relative));db.commit();return json.loads(row['data']),context
    except BaseException:db.rollback();raise


def reconcile_published(db,job):
    """A durable publication may outlive a crash before its ledger settlement."""
    row=db.execute('SELECT status FROM files WHERE relative=?',(job['relative'],)).fetchone()
    target=Path(job['target'])
    if row['status'] not in ('PRECHARGED','CLAIMED','FAILED') or not target.exists():return False
    raw=read_resident(target,job['expected_bytes'])
    require(len(raw)==job['expected_bytes'] and sha(raw)==job['expected_sha256'],'Interrupted publication identity differs')
    with db:db.execute('UPDATE files SET status=? WHERE relative=?',('VERIFIED',job['relative']))
    return True


def claim_verification(db):
    db.execute('BEGIN IMMEDIATE')
    try:
        row=db.execute('SELECT * FROM context WHERE id=1').fetchone();context=json.loads(row['data'])
        require(row['verification']=='PRECHARGED' and context['copier_sha256']==sha(Path(__file__).read_bytes())
                and time.time()<context['deadline_epoch'],'Single integrity pass is not available')
        require(db.execute('SELECT count(*) FROM files').fetchone()[0]==context['expected_files']
                and db.execute("SELECT count(*) FROM files WHERE status!='VERIFIED'").fetchone()[0]==0,
                'Incomplete copy cannot enter numeric validation')
        db.execute('UPDATE context SET verification=? WHERE id=1',('CLAIMED',));db.commit();return context
    except BaseException:db.rollback();raise


@contextmanager
def local_deadline(deadline):
    """The inherited lease must not let an orphan worker outlive its budget."""
    remaining=deadline-time.time()
    if remaining<=0:raise TimeoutError('Worker deadline expired before source access')
    old_handler=signal.getsignal(signal.SIGALRM)
    def timeout(*unused):raise TimeoutError('Worker local deadline reached')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,remaining)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)


def copy_worker(control,relative,token,fd):
    verify_lease(control,fd);db=connect(control);job,context=claim(db,relative,token)
    partial=Path(control)/'partials'/(sha(relative.encode())+'.partial');observed=0
    try:
        with local_deadline(min(time.time()+PER_FILE_SECONDS,context['deadline_epoch'])):
            source,target=Path(job['source']),Path(job['target']);guarded_parents(source,Path(job['source_root']))
            guarded_parents(target,Path(context['target_root']))
            info=source.lstat();require(stat.S_ISREG(info.st_mode) and info.st_size==job['expected_bytes'],'Source stat differs')
            if job['kind']!='DEVELOPMENT_MODEL_NPZ':require(not getattr(info,'st_flags',0)&0x40000000,'Authenticated metadata cache must remain resident')
            require(not target.exists() and not target.is_symlink() and not partial.exists(),'Fresh destination/partial required')
            hasher=hashlib.sha256()
            with partial.open('xb') as out:
                source_fd=os.open(source,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
                with os.fdopen(source_fd,'rb') as inp:
                    while observed<job['expected_bytes']+1:
                        body=inp.read(min(65536,job['expected_bytes']+1-observed))
                        if not body:break
                        observed+=len(body);out.write(body);hasher.update(body)
                        with db:db.execute('UPDATE files SET observed=? WHERE relative=?',(observed,relative))
                out.flush();os.fsync(out.fileno())
            require(observed==job['expected_bytes'] and hasher.hexdigest()==job['expected_sha256'],'Copied source SHA/bytes differ')
            os.link(partial,target);sync_directory(target.parent);partial.unlink();sync_directory(partial.parent)
            with db:db.execute('UPDATE files SET status=? WHERE relative=?',('VERIFIED',relative))
    except BaseException as error:
        with db:db.execute('UPDATE files SET status=?,error_type=? WHERE relative=?',('FAILED',type(error).__name__,relative))
        raise
    finally:db.close()


def verify_worker(control,fd):
    verify_lease(control,fd);db=connect(control);context=claim_verification(db)
    try:
        with local_deadline(context['deadline_epoch']):
            from hydra_directflow.corpus_dataset import inspect_corpus
            snapshot=inspect_corpus(context['target_root'],expected_plan_sha256=SOURCE_PLAN)
            require(snapshot.complete and snapshot.snapshot_sha256==SNAPSHOT,'Recovered development snapshot differs')
            write_new_json(Path(control)/'integrity-verification.json',snapshot.report())
            with db:db.execute('UPDATE context SET verification=? WHERE id=1',('PASSED',))
    except BaseException:
        with db:db.execute('UPDATE context SET verification=? WHERE id=1',('FAILED',))
        raise
    finally:db.close()


def supervise(command,*,fd,deadline,max_rss,log_path):
    import psutil
    log_path=Path(log_path);directory(log_path.parent)
    with log_path.open('xb') as log:
        sync_directory(log_path.parent)
        child=None;peak=0;error=None
        try:
            child=subprocess.Popen(command,pass_fds=(fd,),stdout=log,stderr=log,cwd=ROOT)
            parent=psutil.Process();process=psutil.Process(child.pid)
            while child.poll() is None:
                try:peak=max(peak,parent.memory_info().rss+process.memory_info().rss)
                except psutil.NoSuchProcess:pass
                require(peak<=max_rss and time.time()<deadline,'Copy worker time/RSS bound reached')
                time.sleep(.02)
        except BaseException as exc:
            error=type(exc).__name__
            log.write(('CONTROLLER_ERROR:'+error+'\n').encode())
            if child is not None and child.poll() is None:child.kill()
        finally:
            if child is not None:child.wait(timeout=5)
            log.flush();os.fsync(log.fileno())
    size=log_path.stat().st_size
    log_hash=sha(log_path.read_bytes()) if size<=1024**2 else None
    return dict(exit_code=child.returncode if child is not None else None,error_type=error,peak_parent_child_rss_bytes=peak,log_bytes=size,log_sha256=log_hash)


def summarize(db,context,plan_file_sha):
    records=db.execute('SELECT status,count(*) AS n,sum(reserved) AS reserved,sum(observed) AS observed FROM files GROUP BY status').fetchall()
    head=db.execute('SELECT * FROM context WHERE id=1').fetchone();states={r['status']:r['n'] for r in records}
    complete=states=={'VERIFIED':1432} and head['verification']=='PASSED'
    return dict(schema=SCHEMA,status='RECOVERED_AND_INTEGRITY_VERIFIED' if complete else 'INCOMPLETE_NO_SILENT_RETRY',
        created_at_utc=utc(),source_plan_sha256=SOURCE_PLAN,snapshot_sha256=SNAPSHOT,recovery_plan_file_sha256=plan_file_sha,
        copier_sha256=context['copier_sha256'],file_states=states,expected_files=1432,expected_bytes=75805012,
        source_read_precharged_bytes=head['charged'],source_observed_bytes_lower_bound=sum(r['observed'] or 0 for r in records),
        limits=dict(source_read_bytes=MAX_BODY,wall_seconds=MAX_SECONDS,parent_child_rss_bytes=MAX_RSS,per_file_seconds=PER_FILE_SECONDS,attempts_per_file=1),
        opaque_copy_phase=dict(numerical_values_decoded=False,original_source_files_rewritten=False,raw_source_or_monthly_parquet_reads=False,
            source_API_requests=0,paid_cloud_jobs=0,FileProvider_wire_bytes='UNOBSERVED'),
        postcopy_integrity_phase=dict(state=head['verification'],single_pass=True,
            development_arrays_decoded=True if head['verification']=='PASSED' else False if head['verification'] in ('PENDING','PRECHARGED') else None,
            protected_assessment_access=False,new_census_or_scaler_or_model_work=False),
        all_required_files_resident_outside_cloud=False)


def execute(plan_path,expected_file_sha,expected_content_sha,approved_copier_sha,report):
    plan,jobs=load_plan(plan_path,expected_file_sha,expected_content_sha)
    require(approved_copier_sha==sha(Path(__file__).read_bytes()),'Exact reviewed copier SHA required')
    target=Path(plan['target_root']);control=target.with_name(target.name+'-recovery-control-v1');report=Path(report)
    require(not report.exists() and not report.is_symlink(),'Fresh sanitized report required')
    require(not report.resolve().is_relative_to(Path(plan['source_root']).resolve()) and not report.resolve().is_relative_to(target.resolve()),
            'Report must remain outside original and recovered corpora')
    require(not target.exists() or (control/'transfer.sqlite3').exists(),'Cannot adopt an unrelated target directory')
    with lease(control) as fd:
        directory(control/'partials',True);directory(control/'worker-logs',True);db=connect(control)
        context=initialize(db,dict(schema=SCHEMA,plan_file_sha256=expected_file_sha,plan_content_sha256=expected_content_sha,
            copier_sha256=approved_copier_sha,target_root=str(target),expected_files=len(jobs),body_cap=MAX_BODY,deadline_epoch=time.time()+MAX_SECONDS),jobs)
        directory(target,True)
        for job in jobs:directory(Path(job['target']).parent,True)
        results=[]
        for number,job in enumerate(jobs,1):
            if time.time()>=context['deadline_epoch']:break
            reconcile_published(db,job)
            token=reserve(db,job['relative'])
            if token is None:continue
            command=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'--copy-worker',job['relative'],'--token',token,'--control',str(control),'--lease-fd',str(fd)]
            outcome=supervise(command,fd=fd,deadline=min(time.time()+PER_FILE_SECONDS,context['deadline_epoch']),max_rss=MAX_RSS,
                log_path=control/'worker-logs'/(sha(job['relative'].encode())+'.log'))
            results.append(outcome)
            with db:db.execute("UPDATE files SET status='FAILED',error_type=? WHERE relative=? AND status IN ('PRECHARGED','CLAIMED')",
                (outcome['error_type'] or 'WORKER_INCOMPLETE',job['relative']))
            if number%100==0:print(json.dumps(dict(files_attempted=number,source_bytes_precharged=db.execute('SELECT charged FROM context').fetchone()[0])),flush=True)
        if db.execute("SELECT count(*) FROM files WHERE status!='VERIFIED'").fetchone()[0]==0:
            for job in jobs:
                info=Path(job['target']).lstat();require(stat.S_ISREG(info.st_mode) and info.st_size==job['expected_bytes']
                    and not getattr(info,'st_flags',0)&0x40000000,'Destination residency/stat changed')
            if db.execute('SELECT verification FROM context').fetchone()[0]=='PENDING':
                with db:db.execute('UPDATE context SET verification=? WHERE id=1',('PRECHARGED',))
                command=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'--verify-worker','--control',str(control),'--lease-fd',str(fd)]
                results.append(supervise(command,fd=fd,deadline=context['deadline_epoch'],max_rss=MAX_RSS,
                    log_path=control/'worker-logs'/'integrity-verification.log'))
                with db:db.execute("UPDATE context SET verification='FAILED' WHERE verification IN ('PRECHARGED','CLAIMED')")
        result=summarize(db,context,expected_file_sha)
        result['peak_observed_parent_child_rss_bytes']=max((r['peak_parent_child_rss_bytes'] for r in results),default=0)
        result['private_worker_logs']=dict(count=len(results),nonempty_log_hashes=[r['log_sha256'] for r in results if r['log_bytes']>0])
        result['elapsed_seconds']=MAX_SECONDS-(context['deadline_epoch']-time.time())
        if result['status']=='RECOVERED_AND_INTEGRITY_VERIFIED':
            result['all_required_files_resident_outside_cloud']=all(not getattr(Path(j['target']).lstat(),'st_flags',0)&0x40000000 for j in jobs)
            require(result['all_required_files_resident_outside_cloud'],'Final residency changed')
        write_new_json(control/('execution-'+uuid.uuid4().hex+'.json'),result);write_new_json(report,result);db.close()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path);parser.add_argument('--expected-plan-sha256');parser.add_argument('--expected-content-sha256')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--approved-copier-sha256');parser.add_argument('--report',type=Path)
    parser.add_argument('--copy-worker');parser.add_argument('--verify-worker',action='store_true');parser.add_argument('--control',type=Path)
    parser.add_argument('--lease-fd',type=int);parser.add_argument('--token')
    args=parser.parse_args()
    if args.copy_worker:copy_worker(args.control,args.copy_worker,args.token,args.lease_fd);return
    if args.verify_worker:verify_worker(args.control,args.lease_fd);return
    plan,jobs=load_plan(args.plan,args.expected_plan_sha256,args.expected_content_sha256)
    if not args.execute:
        print(json.dumps(dict(status='PLAN_ONLY_NO_COPY',files=len(jobs),bytes=sum(j['expected_bytes'] for j in jobs),snapshot_sha256=SNAPSHOT)));return
    require(args.report is not None and args.approved_copier_sha256,'Explicit reviewed execution identity/report required')
    result=execute(args.plan,args.expected_plan_sha256,args.expected_content_sha256,args.approved_copier_sha256,args.report)
    print(json.dumps({key:result[key] for key in ('status','file_states','source_read_precharged_bytes','elapsed_seconds')}))
    require(result['status']=='RECOVERED_AND_INTEGRITY_VERIFIED','Recovery incomplete; preserve journal/partials and never reset attempts')


if __name__=='__main__':main()
