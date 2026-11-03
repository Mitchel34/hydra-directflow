#!/usr/bin/env python3
"""Plan-first, resident-only completion plus one fresh development integrity pass.

The old transport clock, charges, errors and reports remain unchanged. This
controller never requests hydration or acquisition and never runs a model.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import recover_development_residency as recovery

PLAN_FILE_SHA='a43eb952ac0d440186c13a82f313f90a2e1f275962b3b93f9ce45dad9689bae7'
PLAN_CONTENT_SHA='2b024be358b4047cc52676c731db9f34730352b2b2554120657538c386864a53'
MAX_BODY=2*1024**2;MAX_COMBINED_SOURCE_BYTES=1024**3;MAX_SECONDS=120;MAX_RSS=4*1024**3
REMAINING=('chunks/candidate_training__2021-10__L168/000/model.npz',
           'chunks/candidate_training__2023-03__L168/005/model.npz')


def require(ok,message):
    if not ok:raise ValueError(message)


def self_sha():return recovery.sha(Path(__file__).read_bytes())


def combined_source_reservation(prior,new):
    require(type(prior) is int and type(new) is int and prior>=0 and 0<=new<=MAX_BODY
            and prior+new<=MAX_COMBINED_SOURCE_BYTES,'Combined source-read reservation exceeds approved1GiB')
    return prior+new


def readonly_old(control):
    path=Path(control)/'transfer.sqlite3'
    for item in (path,path.with_name(path.name+'-wal'),path.with_name(path.name+'-shm')):
        if item.exists() or item.is_symlink():
            info=item.lstat();require(stat.S_ISREG(info.st_mode) and not getattr(info,'st_flags',0)&0x40000000,'Resident regular original ledger required')
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');return db


def old_evidence(plan):
    """Authenticate unchanged failure history without modifying its database."""
    control=Path(plan['original_control']);db=readonly_old(control)
    try:
        head=db.execute('SELECT * FROM context WHERE id=1').fetchone();context=json.loads(head['data'])
        rows=list(db.execute('SELECT * FROM files ORDER BY relative'))
        logical=[];failed={}
        for row in rows:
            job=json.loads(row['data']);logical.append(dict(relative=row['relative'],status=row['status'],expected_sha256=job['expected_sha256'],
                reserved=row['reserved'],observed=row['observed'],error_type=row['error_type']))
            if row['status']=='FAILED':failed[row['relative']]=job
        require(head['verification']=='PENDING' and context['deadline_epoch']==plan['original_deadline_epoch']
            and context['plan_file_sha256']==plan['original_recovery_plan_file_sha256']
            and context['copier_sha256']==plan['original_copier_sha256'], 'Original integrity state/clock/authority changed')
        require(len(rows)==1432 and set(failed)==set(REMAINING)
            and sum(row['status']=='VERIFIED' for row in rows)==1430
            and recovery.sha(recovery.canonical(logical))==plan['original_logical_ledger_state_sha256'], 'Original failed-file history differs')
        require(head['charged']==plan['budget']['first_wave_reserved'],'Original charges changed')
        for item in plan['remaining_files']:
            require(all(item[key]==value for key,value in failed[item['relative']].items()),'Finalization changed an original source/target identity')
        return dict(context=context,charged=head['charged'],logical_sha256=plan['original_logical_ledger_state_sha256'])
    finally:db.close()


def resident_sources(items):
    states=[]
    for item in items:
        source=Path(item['source']);recovery.guarded_parents(source,Path(item['source_root']))
        info=source.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_size==item['expected_bytes'],'Exact regular source byte identity required')
        states.append(not bool(getattr(info,'st_flags',0)&0x40000000))
    return states


def require_resident_sources(items):
    require(all(resident_sources(items)),'Both sources must be resident; no hydration is permitted')


def load_authority(plan_path,recovery_plan_path):
    raw=recovery.read_resident(plan_path,128*1024);require(recovery.sha(raw)==PLAN_FILE_SHA,'Approved finalization plan file differs')
    plan=json.loads(raw)
    require(plan['plan_sha256']==PLAN_CONTENT_SHA==recovery.sha(recovery.canonical({k:v for k,v in plan.items() if k!='plan_sha256'})),
            'Approved finalization content differs')
    require(plan['status']=='PROPOSED_NOT_EXECUTED_AWAIT_ROOT_UNLOCK_AND_RESIDENCY_CONFIRMATION'
            and plan['source_plan_sha256']==recovery.SOURCE_PLAN and plan['snapshot_sha256']==recovery.SNAPSHOT
            and plan['fresh_finalization_authority']['prior_integrity_state']=='PENDING'
            and plan['fresh_finalization_authority']['prior_integrity_pass_count']==0, 'Unused exact development finalization authority required')
    require(tuple(item['relative'] for item in plan['remaining_files'])==REMAINING
            and sum(item['expected_bytes'] for item in plan['remaining_files'])==206468,
            'Exact two-file206468-byte completion scope required')
    control=Path(plan['original_control']);local=Path.home()/'.codex/hydra-directflow-runtime'
    require(control.is_relative_to(local) and control.resolve().is_relative_to(local.resolve()),'Original control must remain in local runtime')
    evidence=old_evidence(plan)
    require(recovery.sha(Path(recovery.__file__).read_bytes())==plan['original_copier_sha256'],'Reviewed transfer helper changed')
    original,jobs=recovery.load_plan(recovery_plan_path,plan['original_recovery_plan_file_sha256'],evidence['context']['plan_content_sha256'])
    require(original['target_root']==plan['target_root'],'Recovery target differs')
    job_map={job['relative']:job for job in jobs}
    for item in plan['remaining_files']:
        require(all(item[key]==value for key,value in job_map[item['relative']].items()),'Unapproved completion source or target')
        prefix=str(Path(item['relative']).parent)
        commit_job=job_map[prefix+'/COMMIT.json'];case_job=job_map[prefix+'/case-index.json']
        commit_raw=recovery.read_resident(commit_job['target'],65536);case_raw=recovery.read_resident(case_job['target'],256*1024)
        require(recovery.sha(commit_raw)==commit_job['expected_sha256'] and recovery.sha(case_raw)==case_job['expected_sha256'],
                'Already copied case/commit metadata changed')
        commit=json.loads(commit_raw)
        require(commit['plan_sha256']==recovery.SOURCE_PLAN and commit['payloads']['model.npz']=={'sha256':item['expected_sha256'],'bytes':item['expected_bytes']},
                'Model identity differs from authenticated case commitment')
    first=recovery.read_resident(ROOT/'reports/development-corpus-residency-recovery-v1.json',1024**2)
    supplement=recovery.read_resident(control/'resident-supplement-v1/summary.json',1024**2)
    require(recovery.sha(first)==plan['first_wave_report_sha256'] and recovery.sha(supplement)==plan['resident_supplement_summary_sha256'],
            'Preserved transport/supplement receipt changed')
    prior_reserved=plan['budget']['first_wave_reserved']+plan['budget']['first_resident_supplement_reserved']
    require(combined_source_reservation(prior_reserved,sum(item['expected_bytes']+1 for item in plan['remaining_files']))==plan['budget']['combined_proposed_reserved'],
            'Cumulative reservation bound differs')
    return plan,jobs,prior_reserved


def context_authority(control):
    db=recovery.connect(control)
    try:context=json.loads(db.execute('SELECT data FROM context WHERE id=1').fetchone()[0])
    finally:db.close()
    require(context['finalizer_sha256']==self_sha(),'Finalizer code changed')
    plan,jobs,prior=load_authority(context['plan_path'],context['recovery_plan_path'])
    require(context['prior_reserved_bytes']==prior and time.time()<context['deadline_epoch'],'Fresh finalization window expired or budget changed')
    return context,plan,jobs


def copy_worker(control,relative,token,new_fd,old_fd):
    recovery.verify_lease(control,new_fd)
    context,plan,_=context_authority(control);recovery.verify_lease(plan['original_control'],old_fd)
    # Both sources must be resident before the once-only claim, even if this
    # worker copies only one of them. No FileProvider request is attempted.
    require_resident_sources(plan['remaining_files'])
    db=recovery.connect(control);job,context=recovery.claim(db,relative,token)
    partial=Path(control)/'partials'/(recovery.sha(relative.encode())+'.partial');observed=0
    try:
        with recovery.local_deadline(context['deadline_epoch']):
            source,target=Path(job['source']),Path(job['target']);require_resident_sources(plan['remaining_files'])
            recovery.guarded_parents(target,Path(context['target_root']))
            require(not target.exists() and not target.is_symlink() and not partial.exists(),'No source copy may overwrite a target or partial')
            with partial.open('xb') as out:
                fd=os.open(source,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
                with os.fdopen(fd,'rb') as stream:
                    info=os.fstat(stream.fileno());require(stat.S_ISREG(info.st_mode) and info.st_size==job['expected_bytes']
                        and not getattr(info,'st_flags',0)&0x40000000,'Resident-only source changed before read')
                    body=stream.read(job['expected_bytes']+1)
                observed=len(body);out.write(body);out.flush();os.fsync(out.fileno())
            with db:db.execute('UPDATE files SET observed=? WHERE relative=?',(observed,relative))
            require(observed==job['expected_bytes'] and recovery.sha(body)==job['expected_sha256'],'Final resident-only copy identity differs')
            os.link(partial,target);recovery.sync_directory(target.parent);partial.unlink();recovery.sync_directory(partial.parent)
            with db:db.execute('UPDATE files SET status=? WHERE relative=?',('VERIFIED',relative))
    except BaseException as error:
        with db:db.execute('UPDATE files SET status=?,error_type=? WHERE relative=?',('FAILED',type(error).__name__,relative))
        raise
    finally:db.close()


def all_targets_resident(jobs):
    for job in jobs:
        target=Path(job['target']);info=target.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_size==job['expected_bytes']
            and not getattr(info,'st_flags',0)&0x40000000,'All1432 exact target files must be resident before integrity verification')


def verify_worker(control,new_fd,old_fd):
    recovery.verify_lease(control,new_fd);context,plan,jobs=context_authority(control)
    recovery.verify_lease(plan['original_control'],old_fd);old_evidence(plan);all_targets_resident(jobs)
    # The separately authorized pass uses only the local target corpus. The old
    # expired transport ledger is read-only and remains PENDING/unused history.
    recovery.verify_worker(control,new_fd)


def supervise(command,new_fd,old_fd,deadline,log):
    import psutil
    process=None;peak=0;error=None;log=Path(log)
    with log.open('xb') as stream:
        recovery.sync_directory(log.parent)
        try:
            process=subprocess.Popen(command,pass_fds=(new_fd,old_fd),stdout=stream,stderr=stream,cwd=ROOT)
            parent=psutil.Process();child=psutil.Process(process.pid)
            while process.poll() is None:
                try:peak=max(peak,parent.memory_info().rss+child.memory_info().rss)
                except psutil.NoSuchProcess:pass
                require(peak<=MAX_RSS and time.time()<deadline,'Finalization time/RSS limit reached')
                time.sleep(.02)
        except BaseException as exc:
            error=type(exc).__name__;stream.write(('CONTROLLER_ERROR:'+error+'\n').encode())
            if process is not None and process.poll() is None:process.kill()
        finally:
            if process is not None:process.wait(timeout=5)
            stream.flush();os.fsync(stream.fileno())
    body=log.read_bytes() if log.stat().st_size<=1024**2 else None
    return dict(exit_code=None if process is None else process.returncode,error_type=error,peak_parent_child_rss_bytes=peak,
                log_bytes=log.stat().st_size,log_sha256=None if body is None else recovery.sha(body))


def execute(plan_path,recovery_plan_path,approved_sha,report,confirmed_resident):
    require(confirmed_resident is True,'Root must explicitly confirm resident source readiness before execution')
    plan,jobs,prior=load_authority(plan_path,recovery_plan_path)
    require(approved_sha==self_sha(),'Reviewed finalizer hash required')
    require_resident_sources(plan['remaining_files'])  # Before ledger/claim/source access.
    original=Path(plan['original_control']);control=original/'finalization-v1';report=Path(report)
    require(not report.exists() and not report.is_symlink() and not report.resolve().is_relative_to(Path(plan['target_root']).resolve()),'Fresh report outside recovered corpus required')
    require(not any(report.resolve().is_relative_to(Path(item['source_root']).resolve()) for item in plan['remaining_files']),
            'Report must remain outside original source corpus')
    require(not control.exists(),'Distinct finalization allowance already created; no reset/replay')
    outcomes=[]
    with recovery.lease(original) as old_fd,recovery.lease(control) as new_fd:
        old_evidence(plan);require_resident_sources(plan['remaining_files'])
        recovery.directory(control/'partials',True);recovery.directory(control/'worker-logs',True)
        db=recovery.connect(control)
        selected=[{key:item[key] for key in ('relative','source','source_root','target','expected_bytes','expected_sha256','kind')} for item in plan['remaining_files']]
        context=recovery.initialize(db,dict(schema='hydra-resident-only-finalization-authority/v1',plan_path=str(Path(plan_path).resolve()),
            recovery_plan_path=str(Path(recovery_plan_path).resolve()),finalizer_sha256=approved_sha,copier_sha256=plan['original_copier_sha256'],
            target_root=plan['target_root'],expected_files=2,body_cap=MAX_BODY,prior_reserved_bytes=prior,
            deadline_epoch=time.time()+MAX_SECONDS,root_confirmed_sources_resident=True,ui_unlock_asserted=False,old_clock_never_reset=True),selected)
        tokens={job['relative']:recovery.reserve(db,job['relative']) for job in selected}
        charged=db.execute('SELECT charged FROM context').fetchone()[0]
        require(charged==206470 and combined_source_reservation(prior,charged)==plan['budget']['combined_proposed_reserved'],
                'Cumulative source read reservation differs')
        for index,job in enumerate(selected):
            command=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'--copy-worker',job['relative'],'--token',tokens[job['relative']],
                '--control',str(control),'--new-fd',str(new_fd),'--old-fd',str(old_fd)]
            outcome=supervise(command,new_fd,old_fd,context['deadline_epoch'],control/'worker-logs'/f'copy-{index}.log');outcomes.append(outcome)
            with db:db.execute("UPDATE files SET status='FAILED',error_type=? WHERE relative=? AND status IN ('PRECHARGED','CLAIMED')",
                (outcome['error_type'] or 'WORKER_INCOMPLETE',job['relative']))
        if db.execute("SELECT count(*) FROM files WHERE status='VERIFIED'").fetchone()[0]==2:
            all_targets_resident(jobs);old_evidence(plan)
            with db:db.execute("UPDATE context SET verification='PRECHARGED' WHERE verification='PENDING'")
            command=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'--verify-worker','--control',str(control),'--new-fd',str(new_fd),'--old-fd',str(old_fd)]
            outcomes.append(supervise(command,new_fd,old_fd,context['deadline_epoch'],control/'worker-logs'/'integrity.log'))
            with db:db.execute("UPDATE context SET verification='FAILED' WHERE verification IN ('PRECHARGED','CLAIMED')")
        head=db.execute('SELECT * FROM context WHERE id=1').fetchone();states=dict(db.execute('SELECT status,count(*) FROM files GROUP BY status').fetchall())
        passed=states=={'VERIFIED':2} and head['verification']=='PASSED'
        old_evidence(plan)
        result=dict(schema='hydra-development-residency-finalization/v1',status='COMPLETE_EXACT_DEVELOPMENT_SNAPSHOT_VERIFIED' if passed else 'INCOMPLETE_FRESH_ALLOWANCE_CONSUMED',
            created_at_utc=datetime.now(timezone.utc).isoformat(),source_plan_sha256=recovery.SOURCE_PLAN,snapshot_sha256=recovery.SNAPSHOT,
            finalization_plan_sha256=PLAN_CONTENT_SHA,finalizer_sha256=approved_sha,first_wave_report_sha256=plan['first_wave_report_sha256'],
            prior_resident_supplement_summary_sha256=plan['resident_supplement_summary_sha256'],final_copy_states=states,
            final_source_precharge_bytes=head['charged'],prior_source_precharge_bytes=prior,cumulative_source_precharge_bytes=prior+head['charged'],
            source_read_cap_bytes=MAX_BODY,combined_source_read_cap_bytes=MAX_COMBINED_SOURCE_BYTES,parent_child_rss_cap_bytes=MAX_RSS,
            target_integrity_scope_bytes=75805012,expected_files=1432,expected_shards=477,
            original_expired_clock_unchanged=True,original_errors_charges_and_reports_unchanged=True,old_ledger_integrity_state_still_pending=True,
            execution_gate='ROOT_RESIDENT_CONFIRMATION_AND_ACTUAL_RESIDENCY_CHECKS_NO_UI_ACTIONS',
            fresh_integrity_authority=True,integrity_state=head['verification'],single_integrity_pass_consumed=head['verification']!='PENDING',
            recovered_files_all_resident_outside_cloud=passed,numerical_decode_scope='ONE_LOCAL_PREASSESSMENT_DEVELOPMENT_INTEGRITY_PASS' if passed else 'UNKNOWN_IF_FAILED_INTEGRITY_STARTED',
            scientific_census_scaler_model_work=False,protected_assessment_access=False,source_API_or_hydration_requests=0,paid_cloud_jobs=0,
            elapsed_seconds=MAX_SECONDS-(context['deadline_epoch']-time.time()),maximum_wall_seconds=MAX_SECONDS,
            peak_observed_parent_child_rss_bytes=max((x['peak_parent_child_rss_bytes'] for x in outcomes),default=0),
            private_worker_logs=dict(count=len(outcomes),hashes=[x['log_sha256'] for x in outcomes]))
        if passed:all_targets_resident(jobs)
        recovery.write_new_json(control/'summary.json',result);recovery.write_new_json(report,result);db.close()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path);parser.add_argument('--recovery-plan',type=Path);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--approved-finalizer-sha256');parser.add_argument('--confirmed-resident',action='store_true');parser.add_argument('--report',type=Path)
    parser.add_argument('--copy-worker');parser.add_argument('--verify-worker',action='store_true');parser.add_argument('--control',type=Path)
    parser.add_argument('--token');parser.add_argument('--new-fd',type=int);parser.add_argument('--old-fd',type=int);args=parser.parse_args()
    if args.copy_worker:copy_worker(args.control,args.copy_worker,args.token,args.new_fd,args.old_fd);return
    if args.verify_worker:verify_worker(args.control,args.new_fd,args.old_fd);return
    plan,_,_=load_authority(args.plan,args.recovery_plan)
    if not args.execute:
        print(json.dumps(dict(status='PLAN_ONLY_NO_SOURCE_READS',resident_sources=sum(resident_sources(plan['remaining_files'])),required_resident_sources=2,
            source_bytes=206468,source_read_cap=MAX_BODY,combined_source_read_cap=MAX_COMBINED_SOURCE_BYTES,
            parent_child_rss_cap=MAX_RSS,new_authority_window_seconds=MAX_SECONDS,old_clock_never_reset=True)));return
    require(args.report is not None,'Explicit fresh sanitized report required')
    result=execute(args.plan,args.recovery_plan,args.approved_finalizer_sha256,args.report,args.confirmed_resident)
    print(json.dumps({k:result[k] for k in ('status','final_copy_states','integrity_state','elapsed_seconds')}))
    require(result['status']=='COMPLETE_EXACT_DEVELOPMENT_SNAPSHOT_VERIFIED','Finalization incomplete; retain evidence and do not silently retry')


if __name__=='__main__':main()
