#!/usr/bin/env python3
"""Plan-first eight-update synthetic trainer-to-best-inference qualification.

Separate once-only allowance. No scientific arrays, downloads or network. The
prior twelve-update recovery qualification and its artifacts are untouched.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
MAX_UPDATES, MAX_SECONDS, MAX_BYTES, MPS_BYTES = 8,900,4*1024**3,2*1024**3
FULL='FULL_HRRR_H2_W4_S2_V1'; Q_ONLY='Q_ONLY_H2_HORIZON1_S0_V1'
LEDGER='best-inference-update-budget-v1.json'


def require(condition,message):
    if not condition: raise ValueError(message)


def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(value): return hashlib.sha256(value).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def mkdir_durable(path):
    path=Path(path)
    if path.exists():
        require(path.is_dir() and not path.is_symlink(),'Resident real directory required');return
    mkdir_durable(path.parent);path.mkdir();sync_directory(path.parent)


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.writing')
    require(not temporary.exists() and not temporary.is_symlink(),'Unsettled JSON staging file; no replacement')
    with temporary.open('xb') as stream:
        stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path);sync_directory(path.parent)


def read_bytes(path,maximum):
    from hydra_directflow.engineering import _local_bytes
    return _local_bytes(path,maximum)


def read_json(path,maximum=1024**2): return json.loads(read_bytes(path,maximum))


def matrix():
    return [dict(id=f'{architecture}-{device}-{mode}',architecture=architecture,device=device,feature_schema=mode,
                 training_updates=2,epochs=2,batch_size=8,history_hours=24,fit_cases=8,inner_stopping_cases=4,inference_cases=4)
            for mode in (FULL,Q_ONLY) for architecture,device in (('LSTM','cpu'),('Mamba2_Transformer','mps'))]


def implementation_identity():
    names=('scripts/qualify_best_inference.py','src/hydra_directflow/training.py','src/hydra_directflow/inference.py',
           'src/hydra_directflow/engineering.py','src/hydra_directflow/trainer_bundle.py','src/hydra_directflow/ablations.py','src/hydra_directflow/contract.py',
           'src/hydra_directflow/reference.py','src/hydra_directflow/local_data.py',
           'src/hydra_directflow/models/directflow.py','src/hydra_directflow/models/mamba2_math.py')
    return dict(code={name:sha(read_bytes(ROOT/name,512*1024)) for name in names},
                dependencies={name:importlib.metadata.version(name) for name in ('numpy','torch','transformers','psutil')})


def plan():
    spec=dict(cases=matrix(),source='SYNTHETIC_ONLY',maximum_optimizer_updates=MAX_UPDATES,
        maximum_wall_seconds=MAX_SECONDS,maximum_combined_rss_mps_bytes=MAX_BYTES,mps_driver_cap_bytes=MPS_BYTES,
        seed=1729,learning_rate=.0001,epochs=2,minimum_epochs=2,patience=1,min_improvement=0.,
        dtype='float32',independent_inference_process=True,training_gauge_id='03161000',
        implementation=implementation_identity())
    require(sum(case['training_updates'] for case in matrix())==MAX_UPDATES,'Fixed eight-update scope differs')
    return dict(schema='hydra-best-inference-qualification-plan/v1',status='PREPARED_NOT_EXECUTED',
                spec=spec,plan_sha256=sha(canonical(spec)),optimizer_updates=0,network_requests=0)


def ledger_path(): return ROOT/'.cache'/LEDGER

def ledger_lock():
    directory=ROOT/'.cache';mkdir_durable(directory)
    lock=(directory/'best-inference-update-budget-v1.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX);return lock


def precharge(prepared,run_directory):
    run=Path(run_directory).resolve()
    require(run.is_relative_to((ROOT/'.local').resolve()),'Use an ignored local qualification directory')
    require(prepared['plan_sha256']==sha(canonical(prepared['spec'])) and prepared['spec']['maximum_optimizer_updates']==MAX_UPDATES
            and prepared['spec']['cases']==matrix(),
            'Exact eight-update qualification plan required')
    with ledger_lock():
        path=ledger_path();require(not path.exists() and not path.is_symlink(),'Eight-update allowance already consumed; no retry or reset')
        record=dict(schema='hydra-best-inference-consumed-allowance/v1',plan_sha256=prepared['plan_sha256'],
            run_directory=str(run),maximum_updates=MAX_UPDATES,precharged_updates=MAX_UPDATES,created_at_utc=utc(),
            deadline_epoch=time.time()+MAX_SECONDS,cases={case['id']:{phase:{'status':'PENDING','entered':0,'confirmed':0}
                for phase in ('train','infer')} for case in matrix()})
        atomic_json(path,record)
    return record


def claim_worker(run_directory,case_id,phase,prepared):
    require(phase in ('train','infer') and case_id in {c['id'] for c in matrix()},'Exact prepared worker case/phase required')
    with ledger_lock():
        ledger=read_json(ledger_path())
        require(ledger['plan_sha256']==prepared['plan_sha256'] and ledger['run_directory']==str(Path(run_directory).resolve())
                and ledger['maximum_updates']==ledger['precharged_updates']==MAX_UPDATES,'Worker lacks its exact consumed allowance')
        require(time.time()<ledger['deadline_epoch'],'Qualification campaign deadline expired')
        item=ledger['cases'][case_id][phase]
        require(item['status']=='PENDING','Worker phase already claimed or settled; replay forbidden')
        if phase=='infer':
            train=ledger['cases'][case_id]['train']
            require(train['status']=='SUCCEEDED' and train['entered']==train['confirmed']==2,'Inference requires its successful two-update trainer')
            require(sha(read_bytes(Path(run_directory)/case_id/'train-result.json',1024**2))==train['result_sha256'],
                    'Settled training result changed')
        item.update(status='CLAIMED',pid=os.getpid(),claimed_at_utc=utc())
        atomic_json(ledger_path(),ledger)
    return ledger['deadline_epoch']


def settle_worker(run_directory,case_id,phase,attempt):
    with ledger_lock():
        ledger=read_json(ledger_path());item=ledger['cases'][case_id][phase]
        require(ledger['run_directory']==str(Path(run_directory).resolve()) and item['status']=='CLAIMED',
                'Only an exact claimed worker can settle')
        require(attempt['pid']==item['pid'],'Settlement worker PID differs')
        item.update(status='SUCCEEDED' if attempt['passed'] else 'FAILED_OR_UNSETTLED',
            entered=attempt['optimizer_updates_entered'],confirmed=attempt['optimizer_updates_confirmed'],
            count_exact=attempt['count_exact'],result_sha256=attempt.get('result_sha256'))
        atomic_json(ledger_path(),ledger)


def training_fixture(schema):
    import numpy as np
    from hydra_directflow.local_data import utc_epoch,iso
    from hydra_directflow.training import q_only_training_bundle
    rng=np.random.default_rng(1729)
    origins=np.array([utc_epoch('2021-01-01T00:00:00Z')+h*3600 for h in range(8)]+
                     [utc_epoch('2021-01-03T00:00:00Z')+h*3600 for h in range(4)],dtype=np.int64)
    arrays=dict(origin_time=origins,history_time=origins[:,None]-np.arange(25,1,-1)*3600,
        target_time=origins[:,None]+np.arange(1,19)*3600,site_id=np.zeros(12,np.int64),
        case_id=np.array(['03161000:'+iso(int(c)) for c in origins]))
    for name,shape in (('history',(12,24,2)),('weather',(12,18,4)),('state',(12,2)),('target',(12,18))):
        arrays[name]=rng.uniform(1.,10.,shape);arrays[name+'_mask']=np.ones(shape,bool)
    arrays['history'][...,1]=np.arange(25,1,-1);arrays['state'][:,0]=.25;arrays['state'][:,1]=4.
    arrays['weather'][...,1]+=273.;arrays['weather'][...,2]/=1000.
    cohort=sha(canonical(dict(fixture='SYNTHETIC_TRAINING',case_ids=arrays['case_id'].tolist())))
    arrays['feature_schema']=np.array(FULL);arrays['cohort_admission_digest']=np.array(cohort)
    if schema==Q_ONLY:arrays=q_only_training_bundle(arrays,cohort_admission_digest=cohort)
    return arrays,cohort


def inference_fixture(schema):
    """Generate only future inputs/coordinates; no discharge targets exist here."""
    import numpy as np
    from hydra_directflow.local_data import utc_epoch,iso
    from hydra_directflow.training import array_identity
    from hydra_directflow.inference import input_only_view
    rng=np.random.default_rng(2718);origins=np.array([utc_epoch('2021-01-05T00:00:00Z')+h*3600 for h in range(4)],np.int64)
    arrays=dict(origin_time=origins,history_time=origins[:,None]-np.arange(25,1,-1)*3600,
        forecast_time=origins[:,None]+np.arange(1,19)*3600,site_id=np.zeros(4,np.int64),
        case_id=np.array(['03161000:'+iso(int(c)) for c in origins]),feature_schema=np.array(schema))
    arrays['history']=rng.uniform(1.,10.,(4,24,2));arrays['history'][...,1]=np.arange(25,1,-1)
    if schema==FULL:
        arrays['weather']=rng.uniform(1.,10.,(4,18,4));arrays['weather'][...,1]+=273.;arrays['weather'][...,2]/=1000.
        arrays['state']=np.tile(np.array([.25,4.]),(4,1))
    else:
        arrays['weather']=np.broadcast_to(np.arange(1,19,dtype=np.float64)[None,:,None],(4,18,1)).copy()
        arrays['state']=np.empty((4,0),np.float64)
    for name in ('history','weather','state'):arrays[name+'_mask']=np.ones(arrays[name].shape,bool)
    cohort=sha(canonical(dict(fixture='SYNTHETIC_INFERENCE',case_ids=arrays['case_id'].tolist())))
    arrays['cohort_admission_digest']=np.array(cohort);source=array_identity(arrays)
    return input_only_view(PoisonLabels(arrays),inference_source_identity=source),source,cohort


class PoisonLabels(dict):
    def __getitem__(self,key):
        if key in ('target','target_mask'):raise AssertionError('Inference requested labels')
        return super().__getitem__(key)


def tree_digest(value):
    import torch
    result=hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):
            tensor=item.detach().cpu().contiguous();result.update(str((str(tensor.dtype),tuple(tensor.shape))).encode());result.update(tensor.numpy().tobytes())
        elif isinstance(item,dict):
            for key in sorted(item):result.update(key.encode());visit(item[key])
        else:result.update(canonical(item))
    visit(value);return result.hexdigest()


class Progress:
    def __init__(self,path,case,phase,torch,deadline):
        import psutil
        self.path,self.case,self.phase,self.torch,self.deadline=path,case,phase,torch,deadline
        self.process,self.parent=psutil.Process(),psutil.Process(os.getppid())
        self.lock,self.stop=threading.RLock(),threading.Event()
        self.state=dict(pid=os.getpid(),phase=phase,case_id=case['id'],optimizer_updates_entered=0,
            optimizer_updates_confirmed=0,mps_driver_bytes=0,peak_combined_bytes=0)
        self.update()
        self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start()
    def update(self,entered=False,confirmed=False):
        with self.lock:
            if entered:
                require(self.phase=='train' and self.state['optimizer_updates_entered']<2,'Worker update allowance exhausted')
                self.state['optimizer_updates_entered']+=1
            if confirmed:
                require(self.state['optimizer_updates_confirmed']<self.state['optimizer_updates_entered'],'Update confirmation without entry')
                self.state['optimizer_updates_confirmed']+=1
            driver=self.torch.mps.driver_allocated_memory() if self.case['device']=='mps' else 0
            self.state['mps_driver_bytes']=driver
            combined=self.process.memory_info().rss+self.parent.memory_info().rss+driver
            self.state['peak_combined_bytes']=max(self.state['peak_combined_bytes'],combined)
            atomic_json(self.path,self.state)  # Entered charge is durable before optimizer entry.
            require(combined<=MAX_BYTES and driver<=MPS_BYTES,'Worker memory allowance reached')
            require(time.time()<self.deadline,'Worker campaign deadline reached')
    def watch(self):
        while not self.stop.wait(.05):
            try:self.update()
            except BaseException:
                os._exit(73)  # Last durable entered/confirmed counters remain authoritative bounds.
    def close(self):
        self.stop.set();self.thread.join(timeout=2);self.update()


def train_case(case,directory,progress):
    import numpy as np
    import torch
    from hydra_directflow.training import DevelopmentTrainer,TrainerSpec,EarlyStopping,array_identity,feature_schema_fields
    from hydra_directflow.models.directflow import DirectFlowModel,ModelConfig
    from hydra_directflow.inference import transform_inputs
    from hydra_directflow.local_data import utc_epoch
    from hydra_directflow.trainer_bundle import refit_label_mask
    arrays,cohort=training_fixture(case['feature_schema']);source=array_identity(arrays)
    require(refit_label_mask(arrays['target_time'],refit_at_utc='2021-01-05T00:00:00Z',
            simulated_label_delay_hours=2).all(),'Synthetic fit/stopping labels unavailable at inference refit')
    spec=TrainerSpec(case['architecture'],case['device'],24,8,.0001,1729,True,2,feature_schema=case['feature_schema'])
    trainer=DevelopmentTrainer(arrays,train_indices=tuple(range(8)),validation_indices=tuple(range(8,12)),
        boundary_epoch=utc_epoch('2021-01-03T00:00:00Z'),spec=spec,source_identity=source)
    original_update=trainer.update
    def guarded_update():
        progress.update(entered=True);value=original_update();progress.update(confirmed=True);return value
    trainer.update=guarded_update
    trainer.fit_epochs(EarlyStopping(2,2,1,0.))
    require(trainer.updates==trainer.stopping.completed_epochs==2,'Exactly two complete synthetic epochs/updates required')
    checkpoint_sha=trainer.checkpoint(directory/'completed.pt')
    inputs,input_source,input_cohort=inference_fixture(case['feature_schema'])
    mapped=transform_inputs(PoisonLabels(inputs),trainer.scaler);fields=feature_schema_fields(case['feature_schema'])
    with torch.random.fork_rng(devices=[]):
        reference=DirectFlowModel(ModelConfig(case['architecture'],fields['history'],fields['weather'],fields['state'],24,tuple(range(1,19))))
    reference.load_state_dict(trainer.best_model_state,strict=True);reference.to(case['device']);reference.eval()
    with torch.inference_mode():
        standardized=reference(*(torch.from_numpy(mapped[name]).to(case['device']) for field in ('history','weather','state') for name in (field,field+'_mask')))
    predicted=trainer.scaler.inverse_target(standardized.detach().cpu().numpy().astype(np.float64))
    require(predicted.shape==(4,18) and np.isfinite(predicted).all(),'Synthetic reference prediction invalid')
    buffer=io.BytesIO();np.save(buffer,predicted,allow_pickle=False);raw=buffer.getvalue()
    with (directory/'reference.npy').open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    sync_directory(directory)
    return dict(checkpoint_sha256=checkpoint_sha,training_source_identity=source,training_cohort=cohort,
        gradient_fit_case_ids=arrays['case_id'][:8].tolist(),gradient_fit_case_ids_sha256=trainer.scaler.training_case_ids_sha256,
        transform_sha256=sha(canonical(asdict(trainer.scaler))),best_model_sha256=tree_digest(trainer.best_model_state),
        selected_best_epoch=max(row['epoch'] for row in trainer.epoch_records if row['improved']),completed_epochs=2,
        reference_file_sha256=sha(raw),reference_prediction_sha256=array_identity({'prediction_q':predicted}),
        mapped_input_sha256=array_identity(mapped),inference_source_identity=input_source,inference_cohort=input_cohort,
        inference_case_ids_sha256=sha(canonical(inputs['case_id'].tolist())))


def infer_case(case,directory,progress):
    import numpy as np
    import torch
    from unittest.mock import patch
    from hydra_directflow.inference import BestCheckpointPredictor,transform_inputs
    from hydra_directflow.training import DevelopmentTrainer,TrainOnlyTransform,array_identity
    train_raw=read_bytes(directory/'train-result.json',1024**2)
    require(sha(train_raw)==read_json(ledger_path())['cases'][case['id']]['train']['result_sha256'],
            'Pinned training result changed before inference')
    train=json.loads(train_raw);inputs,input_source,input_cohort=inference_fixture(case['feature_schema'])
    require(input_source==train['inference_source_identity'] and input_cohort==train['inference_cohort'],'Fresh inference input identities differ')
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('Inference constructed optimizer')),\
         patch.object(TrainOnlyTransform,'fit',side_effect=AssertionError('Inference fitted scaler')),\
         patch.object(DevelopmentTrainer,'__init__',side_effect=AssertionError('Inference constructed trainer')):
        predictor=BestCheckpointPredictor.load(directory/'completed.pt',expected_sha256=train['checkpoint_sha256'],
            expected_source_identity=train['training_source_identity'],expected_feature_schema=case['feature_schema'],
            expected_training_cohort_admission_digest=train['training_cohort'],expected_gauge_id='03161000',
            gradient_fit_case_ids=tuple(train['gradient_fit_case_ids']))
        predicted=predictor.predict(PoisonLabels(inputs),expected_inference_source_identity=input_source,
                                    expected_inference_cohort_admission_digest=input_cohort)
    raw=read_bytes(directory/'reference.npy',65536);require(sha(raw)==train['reference_file_sha256'],'Reference prediction identity differs')
    expected=np.load(io.BytesIO(raw),allow_pickle=False)
    require(expected.dtype==np.dtype('<f8') and expected.shape==(4,18) and np.array_equal(predicted['prediction_q'],expected),
            'Fresh-process best-forward prediction differs')
    require(sha(canonical(asdict(predictor.transform)))==train['transform_sha256']
            and tree_digest(predictor.model.state_dict())==train['best_model_sha256']
            and predicted['receipt']['selected_best_epoch']==train['selected_best_epoch']
            and array_identity(transform_inputs(PoisonLabels(inputs),predictor.transform))==train['mapped_input_sha256'],
            'Saved transform/best-state/reference input mapping differs')
    return dict(exact_physical_prediction=True,exact_saved_transform=True,exact_selected_best_state=True,
        selected_best_epoch=train['selected_best_epoch'],completed_epochs=2,
        checkpoint_sha256=train['checkpoint_sha256'],reference_prediction_sha256=train['reference_prediction_sha256'],
        inference_receipt=predicted['receipt'],labels_available=False,optimizer_constructed=False,scaler_fitted=False)


def worker(run_directory,case_id,phase):
    run=Path(run_directory).resolve();prepared=read_json(run/'plan.json');current=plan()
    require(prepared==current,'Qualification plan/code/dependency identity changed')
    deadline=claim_worker(run,case_id,phase,prepared)  # Before imports, arrays, model, or optimizer.
    case=next(value for value in matrix() if value['id']==case_id);directory=run/case_id
    def deny_network(event,unused):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise RuntimeError('Network forbidden in synthetic qualification')
    sys.addaudithook(deny_network)
    def timeout(signum,frame):raise TimeoutError('Qualification worker deadline reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,min(MAX_SECONDS,math.ceil(deadline-time.time()))))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if case['device']=='mps':
        require(torch.backends.mps.is_available(),'MPS unavailable; no fallback')
        torch.mps.set_per_process_memory_fraction(min(1.,MPS_BYTES/torch.mps.recommended_max_memory()))
    progress=Progress(directory/(phase+'-status.json'),case,phase,torch,deadline)
    try:
        details=train_case(case,directory,progress) if phase=='train' else infer_case(case,directory,progress)
        progress.update();expected=2 if phase=='train' else 0
        require(progress.state['optimizer_updates_entered']==progress.state['optimizer_updates_confirmed']==expected,'Worker update accounting differs')
        result=dict(**details,**progress.state,status='PASSED',plan_sha256=prepared['plan_sha256'])
        atomic_json(directory/(phase+'-result.json'),result)
    finally:progress.close();signal.alarm(0)


def counters(value,limit):
    entered=value.get('optimizer_updates_entered',0);confirmed=value.get('optimizer_updates_confirmed',0)
    require(type(entered) is int and type(confirmed) is int and 0<=confirmed<=entered<=limit,'Invalid observed update counters')
    return entered,confirmed


def supervise(command,*,run,case,phase,deadline,env,plan_sha256):
    import psutil
    directory=run/case['id'];status_path=directory/(phase+'-status.json');result_path=directory/(phase+'-result.json')
    entered=confirmed=peak=0;child=None;error_type=None;result=None;code=None
    try:
        with (directory/(phase+'.log')).open('x') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=log);process=psutil.Process(child.pid);parent=psutil.Process()
            while child.poll() is None:
                status=read_json(status_path) if status_path.exists() else {}
                seen=counters(status,2 if phase=='train' else 0);entered=max(entered,seen[0]);confirmed=max(confirmed,seen[1])
                try:peak=max(peak,process.memory_info().rss+parent.memory_info().rss+status.get('mps_driver_bytes',0),status.get('peak_combined_bytes',0))
                except psutil.NoSuchProcess:pass
                require(peak<=MAX_BYTES and time.time()<deadline,'Qualification campaign resource limit reached')
                time.sleep(.05)
            code=child.wait()
    except BaseException as error:error_type=type(error).__name__
    finally:
        if child is not None and child.poll() is None:child.kill();child.wait(timeout=5)
        if child is not None:code=child.returncode
        for path in (status_path,result_path):
            if path.exists():
                try:
                    value=read_json(path);seen=counters(value,2 if phase=='train' else 0)
                    entered=max(entered,seen[0]);confirmed=max(confirmed,seen[1]);peak=max(peak,value.get('peak_combined_bytes',0))
                    if path==result_path:result=value
                except BaseException as error:error_type=error_type or type(error).__name__
    expected=2 if phase=='train' else 0
    passed=bool(child is not None and code==0 and error_type is None and result is not None and result.get('status')=='PASSED'
                and result.get('pid')==child.pid and result.get('case_id')==case['id'] and result.get('phase')==phase
                and result.get('plan_sha256')==plan_sha256 and entered==confirmed==expected and peak<=MAX_BYTES)
    return dict(case_id=case['id'],phase=phase,pid=child.pid if child else None,exit_code=code,passed=passed,
        optimizer_updates_entered=entered,optimizer_updates_confirmed=confirmed,count_exact=passed,
        peak_observed_combined_bytes=peak,error_type=error_type,
        result_sha256=sha(read_bytes(result_path,1024**2)) if passed else None)


def execute(prepared,*,run_directory,report_path,approved_plan_sha256):
    run,report=Path(run_directory).absolute(),Path(report_path).absolute()
    require(prepared==plan() and approved_plan_sha256==prepared['plan_sha256'],'Explicit exact reviewed plan required')
    require(not run.exists() and not run.is_symlink() and not report.exists() and not report.is_symlink(),'Use new run directory and report')
    require(run.resolve().is_relative_to((ROOT/'.local').resolve()),'Run must remain in ignored local storage')
    consumed=precharge(prepared,run);mkdir_durable(run);atomic_json(run/'plan.json',prepared)
    mkdir_durable(report.parent)
    receipt=dict(schema='hydra-best-inference-qualification/v1',status='RUNNING',plan=prepared,
        precharged_optimizer_updates=MAX_UPDATES,optimizer_updates_entered=0,optimizer_updates_confirmed=0,
        update_count_exact=False,attempts=[],checks=[],created_at_utc=utc(),network_requests=0,scientific_data_reads=0)
    atomic_json(report,receipt);started=time.monotonic()
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTORCH_ENABLE_MPS_FALLBACK':'0'}
    try:
        for case in matrix():
            directory=run/case['id'];mkdir_durable(directory)
            for phase in ('train','infer'):
                command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',phase,'--run-directory',str(run),'--case-id',case['id']]
                attempt=supervise(command,run=run,case=case,phase=phase,deadline=consumed['deadline_epoch'],env=env,plan_sha256=prepared['plan_sha256'])
                receipt['attempts'].append(attempt)
                receipt['optimizer_updates_entered']+=attempt['optimizer_updates_entered']
                receipt['optimizer_updates_confirmed']+=attempt['optimizer_updates_confirmed']
                atomic_json(report,receipt)
                # A pre-claim import/launch failure leaves no claimed worker to settle; charge still remains consumed.
                ledger=read_json(ledger_path())
                if ledger['cases'][case['id']][phase]['status']=='CLAIMED':settle_worker(run,case['id'],phase,attempt)
                require(attempt['passed'],'Qualification worker failed; consumed allowance retained')
            train,infer=(read_json(directory/(phase+'-result.json')) for phase in ('train','infer'))
            require(train['pid']!=infer['pid'] and infer['exact_physical_prediction'] and infer['exact_saved_transform']
                    and infer['exact_selected_best_state'],'Fresh-process qualification differs')
            receipt['checks'].append(dict(**case,distinct_worker_pids=True,selected_best_epoch=infer['selected_best_epoch'],
                completed_epochs=2,exact_physical_prediction=True,exact_saved_transform=True,exact_selected_best_state=True,
                checkpoint_sha256=infer['checkpoint_sha256'],reference_prediction_sha256=infer['reference_prediction_sha256'],
                training_case_ids_sha256=train['gradient_fit_case_ids_sha256'],inference_case_ids_sha256=train['inference_case_ids_sha256'],
                no_inference_labels=True,no_inference_optimizer=True,no_inference_scaler_fit=True))
        require(receipt['optimizer_updates_entered']==receipt['optimizer_updates_confirmed']==MAX_UPDATES,'Eight-update total differs')
        receipt.update(status='PASSED_SYNTHETIC_TRAINER_TO_LABEL_FREE_BEST_INFERENCE',update_count_exact=True)
    except BaseException as error:receipt.update(status='FAILED_WITH_CONSUMED_ALLOWANCE',error_type=type(error).__name__)
    finally:
        receipt['elapsed_seconds']=time.monotonic()-started;atomic_json(report,receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--approved-plan-sha256')
    parser.add_argument('--run-directory',type=Path);parser.add_argument('--report',type=Path)
    parser.add_argument('--worker',choices=('train','infer'));parser.add_argument('--case-id')
    args=parser.parse_args()
    if args.worker:
        require(args.run_directory is not None and args.case_id is not None,'Internal worker requires exact run/case')
        worker(args.run_directory,args.case_id,args.worker);return
    prepared=plan()
    if not args.execute:print(json.dumps(prepared,sort_keys=True,indent=2));return
    require(args.run_directory is not None and args.report is not None and args.approved_plan_sha256 is not None,
            'Execution requires reviewed plan hash, new run directory and report')
    result=execute(prepared,run_directory=args.run_directory,report_path=args.report,approved_plan_sha256=args.approved_plan_sha256)
    print(json.dumps({key:result[key] for key in ('status','precharged_optimizer_updates','optimizer_updates_entered','optimizer_updates_confirmed','update_count_exact','elapsed_seconds')}))
    if result['status']!='PASSED_SYNTHETIC_TRAINER_TO_LABEL_FREE_BEST_INFERENCE':raise SystemExit(1)


if __name__=='__main__':main()
