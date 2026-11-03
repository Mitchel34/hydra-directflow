#!/usr/bin/env python3
"""Qualified, frozen local campaign supervisor; plan creation never starts fits."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from hydra_directflow.campaign_ledger import CampaignLedger,digest,validate_plan
from hydra_directflow.campaign_worker import validate_execution_plan,run_job
from hydra_directflow.reference import require

ROSTER=ROOT/'reports/local-seed-sequence-fastest-v3.json'
ROSTER_SHA='f45aa1afe114db80b194288ecb06c7757ebe3928de3702fac9ff1114238310e9'
MODEL_QUAL=ROOT/'reports/primary-campaign-qualification-v1.json'
DATA_QUAL=ROOT/'reports/primary-campaign-data-qualification-v1.json'
MODEL_QUAL_SHA='b69ae171e6972ae51fb141a586d59fe89938a65febfd9254fab48b0d9ffc6e1a'
DATA_QUAL_SHA='4d0e45a1f2b08a21c93b0355e53073401f593ed4e9bc43f2e2270096e19fc222'
GLOBAL_RELEASE=ROOT/'.cache/primary-scientific-release-v1.json'
SOURCE_ROOT=Path.home()/'.codex/hydra-directflow-runtime/development-corpus-L168-v4'
BOOKKEEPING=2*1024**3
CODE=('scripts/run_primary_campaign.py','scripts/report_primary_seed.py',
      'src/hydra_directflow/campaign_worker.py','src/hydra_directflow/campaign_ledger.py',
      'src/hydra_directflow/campaign_data.py','src/hydra_directflow/training.py',
      'src/hydra_directflow/inference.py','src/hydra_directflow/scoring.py',
      'src/hydra_directflow/evaluation.py','src/hydra_directflow/trainer_bundle.py',
      'src/hydra_directflow/corpus_dataset.py','src/hydra_directflow/ablations.py',
      'src/hydra_directflow/local_data.py','src/hydra_directflow/quality.py',
      'src/hydra_directflow/contract.py','src/hydra_directflow/reference.py',
      'src/hydra_directflow/engineering.py','src/hydra_directflow/models/directflow.py',
      'src/hydra_directflow/models/mamba2_math.py')


def utc():return datetime.now(timezone.utc).isoformat()
def sha(raw):return hashlib.sha256(raw).hexdigest()


def read(path,maximum=8*1024**2):
    from hydra_directflow.corpus_dataset import _read
    return _read(Path(path),maximum)


def sync(directory):
    fd=os.open(directory,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def private(path):
    path=Path(path).absolute()
    require(path==path.resolve() and path.is_relative_to(ROOT/'.local'),'Stable ignored private path required')
    return path


def atomic(path,value,*,new=False):
    path=Path(path)
    # Mutable progress can recover after an interrupted publication without
    # removing old partial evidence. Immutable receipts retain their fixed name.
    temporary=path.with_name(path.name+('.writing' if new else '.writing-'+uuid.uuid4().hex))
    require(not temporary.exists() and not temporary.is_symlink(),'Unsettled controller artifact preserved')
    require(not new or not path.exists(),'Fresh controller artifact required')
    with temporary.open('xb') as stream:
        stream.write((json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())
    if new:
        os.link(temporary,path);temporary.unlink()
    else:os.replace(temporary,path)
    sync(path.parent)


def implementation():return {name:sha(read(ROOT/name,2*1024**2)) for name in CODE}


def qualification_evidence():
    model_raw,data_raw=read(MODEL_QUAL),read(DATA_QUAL)
    require(sha(model_raw)==MODEL_QUAL_SHA and sha(data_raw)==DATA_QUAL_SHA,'Executed qualification receipt hashes differ')
    model,data=json.loads(model_raw),json.loads(data_raw)
    require(model['status']=='PASSED_SYNTHETIC_PRIMARY_CAMPAIGN_QUALIFICATION'
            and model['optimizer_updates_entered']==model['optimizer_updates_confirmed']==36
            and model['scaler_fits_entered']==model['scaler_fits_confirmed']==12
            and len(model['checks'])==12 and model['count_exact'] is True,'Completed exact12-geometry trainer qualification required')
    require(all(c['exact_next_loss_samples_weights_optimizer_rng'] and c['exact_fresh_best_prediction']
                and c['best_differs_from_last'] for c in model['checks']),'Checkpoint recovery or best-state qualification incomplete')
    require(data['status']=='PASSED' and data['qualification']['status']=='PASSED'
            and len(data['qualification']['views'])==27 and data['scope']['data_views']==27
            and data['model_calls']==data['optimizer_updates']==data['scaler_fits']==0
            and data['protected_assessment_access'] is False,'Complete27-view no-fit data qualification required')
    for name,expected in model['plan']['spec']['implementation']['code'].items():
        require(sha(read(ROOT/name,2*1024**2))==expected,'Qualified trainer implementation changed: '+name)
    for name,expected in data['scope']['implementation']['controller'].items():
        require(sha(read(ROOT/name,2*1024**2))==expected,'Qualified data implementation changed: '+name)
    for name,expected in data['scope']['implementation']['loader'].items():
        require(Path(name).name==name and name not in ('.','..'),'Qualified loader filename is unsafe')
        require(sha(read(ROOT/'src/hydra_directflow'/name,2*1024**2))==expected,'Qualified loader implementation changed: '+name)
    for name,expected in model['plan']['spec']['implementation']['dependencies'].items():
        require(importlib.metadata.version(name)==expected,'Qualified dependency changed')
    return dict(model_report_sha256=sha(model_raw),model_plan_sha256=model['plan']['plan_sha256'],
                data_report_sha256=sha(data_raw),data_plan_sha256=data['plan_sha256'])


def reviewed_protocol(protocol_path,expected_sha=None):
    protocol_raw=read(protocol_path);protocol=json.loads(protocol_raw)
    require(expected_sha is None or sha(protocol_raw)==expected_sha,'Frozen protocol changed')
    require(protocol['protocol_frozen'] is True and protocol['scientific_training_authorized'] is True,
            'User-reviewed frozen protocol is required before campaign preparation')
    confirmation=protocol.get('human_confirmation',{})
    require(confirmation.get('received') is True and isinstance(confirmation.get('message_reference'),str)
            and confirmation['message_reference'] and isinstance(confirmation.get('confirmed_at_utc'),str),
            'Recorded human method confirmation required; a protocol hash is not approval')
    require(protocol.get('protected_assessment_access') is False and protocol.get('paid_cloud_jobs_started')==0,
            'Local development-only scope required')
    return protocol,sha(protocol_raw)


def controller_evidence(protocol):
    record=protocol.get('qualification_evidence',{}).get('durable_controller',{})
    require(record.get('artifact')=='reports/primary-campaign-controller-check-v1.json'
            and isinstance(record.get('sha256'),str),'Qualified controller receipt required before launch')
    raw=read(ROOT/record['artifact']);require(sha(raw)==record['sha256'],'Controller qualification receipt changed')
    value=json.loads(raw)
    require(value['status']=='PASSED_UNIT_AND_PEER_REVIEW' and value['unit_tests']['failures']==value['unit_tests']['errors']==0
            and value['scientific_fits_started']==0,'Completed controller verification required')
    require(value['implementation_sha256']==implementation(),'Controller implementation changed after qualification')
    return dict(report_sha256=sha(raw),status=value['status'])


def prepare(protocol_path,run_directory):
    protocol_path=Path(protocol_path).absolute();protocol,protocol_sha=reviewed_protocol(protocol_path)
    run=private(run_directory);ledger=run/'ledger'
    raw=read(ROSTER);require(sha(raw)==ROSTER_SHA,'Exact fastest local180 roster required');roster=json.loads(raw)
    require(protocol['source_plan_sha256']==roster['source_plan_sha256'] and protocol['snapshot_sha256']==roster['snapshot_sha256'],
            'Frozen protocol uses a different data snapshot')
    require(protocol['training']['max_epochs'] in (100,200),'Explicit reviewed high epoch ceiling required')
    jobs=[]
    for original in roster['jobs']:
        job=deepcopy(original);batches=job['gradient_batches_per_epoch']
        job.update(data_job=deepcopy(original),logical_update_limit=protocol['training']['max_epochs']*batches,
            max_updates=(protocol['training']['max_epochs']+30)*batches,max_resumes=3,checkpoint_interval_epochs=10)
        jobs.append(job)
    require(protocol['limits']['max_optimizer_updates']==sum(j['max_updates'] for j in jobs),'Global physical update allowance differs')
    result=dict(schema='hydra-primary-local-campaign-plan/v1',created_at_utc=utc(),
        protocol_path=str(protocol_path),protocol_sha256=protocol_sha,protocol_frozen=True,scientific_training_authorized=True,
        source_plan_sha256=roster['source_plan_sha256'],snapshot_sha256=roster['snapshot_sha256'],
        roster_sha256=ROSTER_SHA,roster_plan_sha256=roster['plan_sha256'],
        corpus_root=str(SOURCE_ROOT),run_directory=str(run),ledger_directory=str(ledger),
        training=protocol['training'],limits=protocol['limits'],jobs=jobs,
        implementation_sha256=implementation(),qualification=qualification_evidence(),controller_qualification=controller_evidence(protocol),
        protected_assessment_access=False,paid_cloud_jobs_started=0,
        bookkeeping_reservation_bytes=BOOKKEEPING,maximum_bookkeeping_bytes=BOOKKEEPING,
        per_worker_log_limit_bytes=16*1024**2,maximum_reports_output_bytes=64*1024**2,
        checkpoint_retention='INITIAL_EVERY10_COMPLETED_EPOCHS_AND_TERMINAL_ALL_RETAINED_MAX21_PER_JOB',
        no_automatic_retry=True,seed_report_barriers=[108,144,180],scientific_seed_order=[17,29,43])
    result['plan_sha256']=digest(result)
    return validate_execution_plan(result)


def authenticated_plan(path,expected):
    path=private(path);plan=json.loads(read(path,8*1024**2))
    require(plan['plan_sha256']==expected,'Exact campaign plan SHA required')
    validate_execution_plan(plan)
    protocol,_=reviewed_protocol(plan['protocol_path'],plan['protocol_sha256'])
    require(all(plan[key]==protocol[key] for key in ('training','limits','source_plan_sha256','snapshot_sha256')),
            'Execution plan differs from the reviewed protocol')
    require(plan['qualification']==qualification_evidence(),'Qualified sources or data evidence changed')
    require(plan['controller_qualification']==controller_evidence(protocol),'Controller qualification differs from reviewed protocol')
    require(plan['corpus_root']==str(SOURCE_ROOT) and Path(plan['run_directory'])==private(plan['run_directory'])
            and plan['ledger_directory']==str(Path(plan['run_directory'])/'ledger'),'Fixed source/run/ledger bindings required')
    return plan


def initialize(plan):
    """One global release prevents relocating the same180-fit allowance."""
    require(not GLOBAL_RELEASE.exists() and not GLOBAL_RELEASE.is_symlink(),'Primary release already claimed; use explicit same-run continuation')
    run=private(plan['run_directory']);run.mkdir(mode=0o700,exist_ok=False);sync(run.parent)
    marker=dict(schema='hydra-primary-local-release/v1',plan_sha256=plan['plan_sha256'],protocol_sha256=plan['protocol_sha256'],
                run_directory=str(run),ledger_directory=plan['ledger_directory'],created_at_utc=utc(),max_optimizer_updates=plan['limits']['max_optimizer_updates'])
    GLOBAL_RELEASE.parent.mkdir(exist_ok=True);atomic(GLOBAL_RELEASE,marker,new=True)
    atomic(run/'PLAN.json',plan,new=True)
    ledger=CampaignLedger(plan['ledger_directory'])
    with ledger.writer():
        ledger.initialize(plan);ledger.reserve_output(BOOKKEEPING,kind='BOOKKEEPING')
    atomic(run/'start-receipt.json',dict(status='INITIALIZED_NO_FITS_YET',plan_sha256=plan['plan_sha256'],created_at_utc=utc()),new=True)


def verify_release(plan):
    marker=json.loads(read(GLOBAL_RELEASE))
    require(all(marker[k]==plan[k] for k in ('plan_sha256','protocol_sha256','run_directory','ledger_directory')),
            'Global release cannot be moved or reset')
    require(json.loads(read(Path(plan['run_directory'])/'PLAN.json'))==plan,'Original private campaign plan changed')


def folder_bytes(root):
    total=0
    for directory,subdirectories,files in os.walk(root,followlinks=False):
        for name in subdirectories:require(not (Path(directory)/name).is_symlink(),'Run directory symlink refused')
        for name in files:
            p=Path(directory)/name;require(not p.is_symlink(),'Run artifact symlink refused')
            try:total+=p.stat().st_size
            except FileNotFoundError:pass # Atomic publication may rename an owned temporary.
    return total


def progress(plan,ledger,*,status,child_pid=None,error=None,heartbeat=None):
    snapshot=ledger.snapshot();counts=dict(Counter(j['state'] for j in snapshot['jobs']))
    next_job=ledger.next_job()
    value=dict(schema='hydra-primary-campaign-progress/v1',status=status,updated_at_utc=utc(),
        plan_sha256=plan['plan_sha256'],protocol_sha256=plan['protocol_sha256'],supervisor_pid=os.getpid(),worker_pid=child_pid,
        job_counts=counts,completed_jobs=counts.get('VERIFIED_TRAIN_EVAL',0),total_jobs=180,
        optimizer_updates_entered=snapshot['optimizer_updates_entered'],optimizer_updates_confirmed=snapshot['optimizer_updates_confirmed'],
        output_bytes_reserved=snapshot['output_bytes_reserved'],seed_reports_ready=snapshot['seed_reports_ready'],
        next_job=None if next_job is None else dict(job_id=next_job['job']['job_id'],state=next_job['state'],
            architecture=next_job['job']['architecture'],gauge_id=next_job['job']['gauge_id'],seed=next_job['job']['seed'],device=next_job['job']['device']),
        heartbeat=heartbeat,error=error,protected_assessment_access=False,cloud_jobs=0)
    atomic(Path(plan['run_directory'])/'progress.json',value)
    target=Path(plan['run_directory'])/'progress.html';temporary=target.with_name('progress.html.writing-'+uuid.uuid4().hex)
    current=value['next_job'] or {};pulse=heartbeat or {}
    fields={'Status':status,'Completed fits':f"{value['completed_jobs']} /180",'Current job':current.get('job_id','—'),
        'Backend':current.get('device','—'),'Completed epochs':pulse.get('completed_epochs','—'),
        'Logical updates in current fit':pulse.get('logical_updates','—'),'Confirmed optimizer updates':value['optimizer_updates_confirmed'],
        'Worker stage':pulse.get('stage','—'),'Updated UTC':value['updated_at_utc']}
    rows=''.join('<tr><th>'+html.escape(str(k))+'</th><td>'+html.escape(str(v))+'</td></tr>' for k,v in fields.items())
    links=' '.join(f'<a href="reports/seed-{s}/index.html">Seed {s} report</a>' for s in value['seed_reports_ready'])
    body=('<!doctype html><html><meta charset="utf-8"><meta http-equiv="refresh" content="15"><title>HYDRA training progress</title>'
        '<style>body{font:16px system-ui;background:#101c24;color:#eef4f6;max-width:1050px;margin:40px auto;padding:0 24px}'
        'h1{border-bottom:3px solid #bf5700;padding-bottom:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #31434e}'
        'a{color:#91d8ef;margin-right:20px}td{overflow-wrap:anywhere}p{color:#b9cbd5}</style>'
        '<h1>HYDRA DirectFlow · local training</h1><p>Development only. Protected assessment remains sealed.</p>'
        '<table>'+rows+'</table><p>'+links+'</p><p>Early stopping uses inner data. First-seed results remain provisional until all seeds finish.</p></html>')
    require(not temporary.exists(),'Unsettled progress page preserved')
    with temporary.open('x') as page:page.write(body);page.flush();os.fsync(page.fileno())
    os.replace(temporary,target);sync(target.parent)
    return value


def publish_ready_seeds(plan,ledger):
    from report_primary_seed import publish_seed_report,_ledger_records
    run=Path(plan['run_directory'])
    for seed in ledger.snapshot()['seed_reports_ready']:
        if not (run/'reports'/f'seed-{seed}'/'COMMIT.json').is_file():
            publish_seed_report(plan,run,Path(plan['ledger_directory']),seed)
        directory=run/'reports'/f'seed-{seed}'
        commit=json.loads(read(directory/'COMMIT.json',2*1024**2))
        require(commit['schema']=='hydra-primary-seed-report-commit/v1' and commit['plan_sha256']==plan['plan_sha256']
                and commit['seed']==seed and set(commit['payloads'])=={'summary.json','index.html'},
                'Complete seed report identity differs before later seeds advance')
        through={17:108,29:144,43:180}[seed]
        require(set(commit['completion_receipts'])=={j['job_id'] for j in plan['jobs'][:through]},
                'Seed report omits expected completed jobs')
        records=_ledger_records(plan,Path(plan['ledger_directory']),seed)
        require(commit['completion_receipts']=={row['job']['job_id']:row['receipt_sha256'] for row in records},
                'Seed report completion receipts differ from the verified ledger')
        for name,identity in commit['payloads'].items():
            raw=read(directory/name,plan['maximum_reports_output_bytes'])
            require(len(raw)==identity['bytes'] and sha(raw)==identity['sha256'],'Seed report bytes changed')
            if name=='summary.json':
                summary=json.loads(raw)
                require(summary['plan_sha256']==plan['plan_sha256'] and summary['seed']==seed
                        and summary['content_sha256']==digest({k:v for k,v in summary.items() if k!='content_sha256'}),
                        'Seed summary payload identity differs')


def supervise(plan_path,plan,*,resume=False):
    """One fresh child perfit; no automatic retry or overlapping science workers."""
    import psutil
    verify_release(plan);run=Path(plan['run_directory']);ledger=CampaignLedger(plan['ledger_directory'])
    lock=run/'supervisor.lock'
    fd=os.open(lock,os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:
        os.close(fd);raise
    child=None;request_resume=resume
    def stop_signal(*unused):raise KeyboardInterrupt('Supervisor termination requested')
    previous_term=signal.signal(signal.SIGTERM,stop_signal)
    try:
        with ledger.writer():ledger.initialize(plan) # Settles any orphan after explicit controller restart.
        while True:
            publish_ready_seeds(plan,ledger)
            upcoming=ledger.next_job()
            if upcoming is None:
                return progress(plan,ledger,status='COMPLETE_ALL180_TRAINING_AND_DEVELOPMENT_METRICS')
            require(upcoming['state']=='PENDING' or request_resume and upcoming['state']=='INTERRUPTED',
                    'Earlier failed/interrupted job requires explicit recovery; no automatic replacement')
            require(shutil.disk_usage(run).free>=plan['limits']['min_free_bytes'],'Free-space floor reached before another fit')
            require(folder_bytes(run)<=plan['limits']['max_output_bytes'],'Actual run artifact limit reached')
            validate_execution_plan(plan)
            ordinal=len(ledger.snapshot()['attempts'])+1
            log_path=run/f'worker-{ordinal:06d}.log'
            env={**os.environ,'PYTHONNOUSERSITE':'1','PYTORCH_ENABLE_MPS_FALLBACK':'0',
                 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
            command=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'--worker','--plan',str(plan_path),
                     '--approved-plan-sha256',plan['plan_sha256']]
            if request_resume:command.append('--resume')
            with log_path.open('xb') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
                parent=psutil.Process();worker=psutil.Process(child.pid);start=time.monotonic();last=0.;last_storage=0.;failure=None
                while child.poll() is None:
                    current=ledger.snapshot();active=ledger.next_job();now=time.time();heartbeat=None
                    if current['attempts'] and current['attempts'][-1]['status']=='ACTIVE':
                        attempt=current['attempts'][-1]
                        path=run/'jobs'/upcoming['job']['job_id']/f'attempt-{attempt["id"]:06d}'/'heartbeat.json'
                        if path.exists():
                            heartbeat=json.loads(read(path,64*1024))
                            require(heartbeat['pid']==child.pid and heartbeat['plan_sha256']==plan['plan_sha256']
                                    and heartbeat['job_id']==upcoming['job']['job_id'],'Worker heartbeat identity differs')
                    driver=0 if heartbeat is None else heartbeat['mps_driver_bytes']
                    require(type(driver) is int and 0<=driver<=2*1024**3,'Worker MPS memory telemetry invalid')
                    try:rss=parent.memory_info().rss+worker.memory_info().rss+driver
                    except psutil.NoSuchProcess:break
                    require(rss<=plan['limits']['max_combined_rss_bytes'],'Supervisor memory guard reached')
                    require(now<(current['campaign_deadline'] or now+1) and now<(active['deadline'] if active and active['deadline'] else now+plan['limits']['max_job_seconds']),
                            'Persistent campaign/job deadline reached')
                    require(time.monotonic()-start<plan['limits']['max_job_seconds'],'Worker hard wall cap reached')
                    require(log_path.stat().st_size<=plan['per_worker_log_limit_bytes'],'Worker log bound reached')
                    if time.monotonic()-last_storage>=5:
                        require(shutil.disk_usage(run).free>=plan['limits']['min_free_bytes'],'Supervisor free-space floor reached')
                        require(folder_bytes(run)<=plan['limits']['max_output_bytes'],'Supervisor disk limit reached')
                        last_storage=time.monotonic()
                    if time.monotonic()-last>=2:
                        progress(plan,ledger,status='RUNNING',child_pid=child.pid,heartbeat=heartbeat);last=time.monotonic()
                    time.sleep(.5)
                child.wait(timeout=10);log.flush();os.fsync(log.fileno())
            if child.returncode!=0:
                with ledger.writer():pass # Orphaned charge is retained; no next job is started.
                return progress(plan,ledger,status='STOPPED_WORKER_FAILURE',error=dict(job_id=upcoming['job']['job_id'],exit_code=child.returncode))
            settled=ledger.snapshot()['jobs'][upcoming['job']['order']]
            require(settled['state']=='VERIFIED_TRAIN_EVAL','Worker exited without authenticated training/evaluation completion')
            child=None;request_resume=False
            progress(plan,ledger,status='JOB_VERIFIED')
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        with ledger.writer():pass
        progress(plan,ledger,status='STOPPED_CONTROLLER_GUARD',error={'type':type(error).__name__,'message':str(error)[:500]})
        raise
    finally:
        signal.signal(signal.SIGTERM,previous_term)
        fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','run-directory','plan'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--continue-run',action='store_true');parser.add_argument('--approved-plan-sha256')
    args=parser.parse_args()
    if args.prepare:
        require(args.protocol and args.run_directory and args.plan,'Explicit protocol/run/plan required')
        plan=prepare(args.protocol,args.run_directory);private(args.plan);atomic(args.plan,plan,new=True)
        print(json.dumps(dict(status='PREPARED_NOT_EXECUTED',plan_sha256=plan['plan_sha256'],jobs=180)));return
    require(args.plan and args.approved_plan_sha256 and (args.execute or args.worker),'Explicit reviewed execution mode and plan required')
    plan=authenticated_plan(args.plan,args.approved_plan_sha256)
    if args.worker:
        verify_release(plan)
        result=run_job(plan,plan['corpus_root'],plan['run_directory'],plan['ledger_directory'],resume=args.resume)
        print(json.dumps(dict(status=result['status'],job_id=result.get('job_id'))));return
    if args.continue_run or args.resume:verify_release(plan)
    else:initialize(plan)
    result=supervise(args.plan,plan,resume=args.resume)
    print(json.dumps({k:result[k] for k in ('status','completed_jobs','optimizer_updates_confirmed')}))


if __name__=='__main__':main()
