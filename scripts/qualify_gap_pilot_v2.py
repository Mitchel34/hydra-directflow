#!/usr/bin/env python3
"""Plan-first synthetic H4 qualification; separate v2 16-update allowance, no study data."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from qualify_best_inference import atomic_json, mkdir_durable, read_bytes, read_json, canonical, sha

MAX_UPDATES=16
MAX_SECONDS=900
MAX_BYTES=4*1024**3
LEDGER=ROOT/'.cache/gap-pilot-qualification-budget-v2.json'
LOCK=ROOT/'.cache/gap-pilot-qualification-budget-v2.lock'
LEGACY_RUN=ROOT/'.local/gap-pilot-qualification-v1'
LEGACY_REPORT=ROOT/'reports/gap-pilot-qualification-v1.json'
LEGACY_LEDGER=ROOT/'.cache/gap-pilot-qualification-budget-v1.json'
CODE=('scripts/qualify_gap_pilot_v2.py','scripts/qualify_gap_pilot.py','scripts/gap_pilot_fixture.py','scripts/qualify_best_inference.py',
      'src/hydra_directflow/gap_policies.py','src/hydra_directflow/gap_filling.py',
      'src/hydra_directflow/pilot_inputs.py','src/hydra_directflow/pilot_training.py','src/hydra_directflow/pilot_provenance.py',
      'src/hydra_directflow/training.py','src/hydra_directflow/inference.py',
      'src/hydra_directflow/models/directflow.py','src/hydra_directflow/quality.py',
      'src/hydra_directflow/reference.py','src/hydra_directflow/engineering.py',
      'src/hydra_directflow/trainer_bundle.py','src/hydra_directflow/contract.py',
      'src/hydra_directflow/ablations.py','src/hydra_directflow/local_data.py',
      'src/hydra_directflow/models/mamba2_math.py')


def require(condition,message):
    if not condition:raise ValueError(message)


def utc():return datetime.now(timezone.utc).isoformat()


def tree_digest(value):
    """Bind tensor/optimizer trees, including integer state keys and lists."""
    import torch
    result=hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):
            t=item.detach().cpu().contiguous()
            result.update(canonical(['tensor',str(t.dtype),list(t.shape)]));result.update(t.numpy().tobytes())
        elif isinstance(item,dict):
            result.update(canonical(['dict',len(item)]))
            for key in sorted(item,key=lambda k:canonical([type(k).__name__,k])):
                result.update(canonical([type(key).__name__,key]));visit(item[key])
        elif isinstance(item,(list,tuple)):
            result.update(canonical([type(item).__name__,len(item)]))
            for element in item:visit(element)
        else:result.update(canonical([type(item).__name__,item]))
    visit(value);return result.hexdigest()


def workers():
    return [{'id':'ridge','mode':None,'phase':'ridge','updates':0,'ridge_fits':0}]+[
        {'id':mode+'-'+phase,'mode':mode,'phase':phase,'updates':updates,'ridge_fits':0}
        for mode in ('clean','augmented') for phase,updates in
        (('baseline',4),('prefix',2),('resume',2),('infer',0))]


def checked_counts(value,limit):
    entered=value.get('optimizer_updates_entered',0);confirmed=value.get('optimizer_updates_confirmed',0)
    require(type(entered) is int and type(confirmed) is int and 0<=confirmed<=entered<=limit,
            'Worker update counters outside its declared allowance')
    return entered,confirmed


def implementation():
    return {'code':{p:sha(read_bytes(ROOT/p,512*1024)) for p in CODE},
            'dependencies':{p:importlib.metadata.version(p) for p in ('numpy','torch','psutil','transformers')}}


def expected_inadmissibility(error,admissible):
    return (isinstance(error,ValueError)
            and str(error)=='Pilot inference has failed/unsupported fills; no silent filtering or forwarding'
            and len(admissible)>0 and not all(admissible))


def resident_artifact(path,maximum):
    """Only resident copies inside this stable checkout; never follow old paths."""
    path=Path(path)
    require(path.is_relative_to(ROOT) and path.resolve().is_relative_to(ROOT.resolve()),
            'Reference artifact must be copied inside the stable checkout')
    return read_bytes(path,maximum)  # _local_bytes rejects dataless/cloud placeholders.


def legacy_report():
    value=json.loads(resident_artifact(LEGACY_REPORT,1024**2))
    require(value['status']=='PASSED_SYNTHETIC_H4_GAP_PILOT_QUALIFICATION'
            and value['optimizer_updates_entered']==value['optimizer_updates_confirmed']==16
            and value['plan']['spec']['source']=='SYNTHETIC_GENERATOR_ONLY_NO_EXTERNAL_ARRAY_ARGUMENT',
            'Qualified v1 synthetic reference required; no re-execution or fallback')
    return value


def legacy_result(identifier):
    report=legacy_report()
    attempts=[row for row in report['attempts'] if row['worker_id']==identifier]
    require(len(attempts)==1 and attempts[0]['passed'] and attempts[0]['count_exact'], 'Passed v1 phase identity required')
    raw=resident_artifact(LEGACY_RUN/identifier/'result.json',2*1024**2)
    require(sha(raw)==attempts[0]['result_sha256'],'Resident v1 result SHA differs')
    result=json.loads(raw)
    require(result['status']=='PASSED' and result['worker_id']==identifier
            and result['plan_sha256']==report['plan']['plan_sha256'],'v1 result context differs')
    return result


def legacy_binding():
    report=legacy_report();files={}
    ledger_raw=resident_artifact(LEGACY_LEDGER,1024**2);ledger=json.loads(ledger_raw)
    require(ledger['precharged_optimizer_updates']==16 and ledger['plan_sha256']==report['plan']['plan_sha256'],
            'Preserved consumed v1 allowance differs')
    for identifier in ('ridge','clean-baseline','augmented-baseline'):
        result=legacy_result(identifier)
        files[identifier+'/result.json']=sha(resident_artifact(LEGACY_RUN/identifier/'result.json',2*1024**2))
        name='ridge.json' if identifier=='ridge' else 'checkpoint.pt'
        raw=resident_artifact(LEGACY_RUN/identifier/name,64*1024**2)
        expected=result['artifact_sha256'] if identifier=='ridge' else result['checkpoint_sha256']
        require(sha(raw)==expected,'Resident v1 checkpoint/ridge artifact differs')
        files[identifier+'/'+name]=expected
    return dict(report_sha256=sha(resident_artifact(LEGACY_REPORT,1024**2)),
        qualification_plan_sha256=report['plan']['plan_sha256'],resident_artifacts_sha256=files,
        old_consumed_optimizer_updates=16,old_ledger_sha256=sha(ledger_raw),old_allowance_reused=False)


def compare_legacy_baseline(run,mode,new_result):
    """Read-only tensor comparison; never restore old code-bound checkpoints."""
    import torch
    old=legacy_result(mode+'-baseline')
    keys=('trace','model_sha256','optimizer_sha256','sampler','scaler','best_model_sha256',
          'epoch_records','reference_prediction_sha256','mapped_inputs_sha256')
    for key in keys:require(new_result[key]==old[key],'Optimized baseline differs from v1: '+key)
    old_raw=resident_artifact(LEGACY_RUN/(mode+'-baseline')/'checkpoint.pt',64*1024**2)
    new_raw=resident_artifact(run/(mode+'-baseline')/'checkpoint.pt',64*1024**2)
    require(sha(old_raw)==old['checkpoint_sha256'] and sha(new_raw)==new_result['checkpoint_sha256'],
            'Baseline checkpoint changed during equivalence comparison')
    old_value=torch.load(io.BytesIO(old_raw),map_location='cpu',weights_only=True)
    new_value=torch.load(io.BytesIO(new_raw),map_location='cpu',weights_only=True)
    require(torch.equal(old_value['cpu_rng'],new_value['cpu_rng']),'Optimized baseline CPU RNG differs from v1')
    return dict(exact_v1_trace_model_optimizer_scaler_masks_best_predictions=True,exact_v1_cpu_rng=True,
        v1_checkpoint_sha256=old['checkpoint_sha256'],new_checkpoint_sha256=new_result['checkpoint_sha256'],
        code_fingerprint_bytes_expected_to_differ=True,old_checkpoint_restored_under_new_code=False)


def plan():
    current=implementation();previous=legacy_report()['plan']['spec']['implementation']
    for name,expected in previous['code'].items():
        if name not in ('src/hydra_directflow/pilot_training.py','src/hydra_directflow/pilot_inputs.py'):
            require(current['code'].get(name)==expected,'Unapproved noncanonical v1 helper change: '+name)
    require(current['dependencies']==previous['dependencies'],'Exact v1 runtime package versions required for equivalence')
    spec={'source':'SYNTHETIC_GENERATOR_ONLY_NO_EXTERNAL_ARRAY_ARGUMENT',
        'workers':workers(),'maximum_optimizer_updates':MAX_UPDATES,'synthetic_ridge_fits':0,
        'resident_v1_ridge_reused':True,'v1_reference':legacy_binding(),
        'maximum_wall_seconds':MAX_SECONDS,'maximum_parent_child_rss_bytes':MAX_BYTES,
        'architecture':'LSTM','device':'cpu','batch_size':64,'history_hours':168,
        'width':64,'dtype':'float32','fit_cases':128,'inner_cases':64,'inference_cases':64,
        'seed':17,'mask_seed':1729,'epochs':2,'min_epochs':2,'patience':1,'learning_rate':.0001,
        'torch_threads':1,'fresh_process_restart_and_inference':True,
        'scientific_fits_authorized':False,'protected_assessment_authorized':False,
        'allowed_v1_helper_changes':['src/hydra_directflow/pilot_training.py','src/hydra_directflow/pilot_inputs.py'],
        'new_provenance_helper':'src/hydra_directflow/pilot_provenance.py',
        'optimization_scope':'Reviewed canonicalization and immutable original-metadata reuse only; no Q/target/fill caching or receipt-semantic changes',
        'implementation':current}
    require(sum(w['updates'] for w in spec['workers'])==MAX_UPDATES,'Fixed update scope differs')
    return {'schema':'hydra-gap-pilot-qualification-plan/v2','status':'PREPARED_NOT_EXECUTED',
            'spec':spec,'plan_sha256':sha(canonical(spec))}


def locked():
    require(LOCK.parent.resolve().is_relative_to(ROOT.resolve()),'v2 ledger directory must stay inside stable checkout')
    mkdir_durable(LOCK.parent);f=LOCK.open('a+');fcntl.flock(f.fileno(),fcntl.LOCK_EX);return f


def precharge(prepared,run):
    require(prepared==plan(),'Prepared qualification identity changed')
    with locked():
        require(not LEDGER.exists() and not LEDGER.is_symlink(),'Qualification allowance consumed; no retry/reset')
        ledger={'schema':'hydra-gap-pilot-consumed-allowance/v2','run_directory':str(run),
            'plan_sha256':prepared['plan_sha256'],'precharged_optimizer_updates':MAX_UPDATES,
            'precharged_synthetic_ridge_fits':0,'resident_v1_ridge_reused':True,'deadline_epoch':time.time()+MAX_SECONDS,
            'created_at_utc':utc(),'workers':{w['id']:{'status':'PENDING'} for w in workers()}}
        atomic_json(LEDGER,ledger)
    return ledger


def claim(run,w,prepared):
    with locked():
        value=read_json(LEDGER)
        require(value['run_directory']==str(run) and value['plan_sha256']==prepared['plan_sha256']
                and time.time()<value['deadline_epoch'] and value['workers'][w['id']]['status']=='PENDING',
                'Exact unclaimed qualification worker required')
        value['workers'][w['id']]={'status':'CLAIMED','pid':os.getpid(),'claimed_at_utc':utc()}
        atomic_json(LEDGER,value)
        return value['deadline_epoch']


def prior_result(run,identifier):
    ledger=read_json(LEDGER);record=ledger['workers'][identifier]
    require(record['status']=='PASSED','Required qualification phase has not passed')
    raw=read_bytes(run/identifier/'result.json',2*1024**2)
    require(sha(raw)==record['result_sha256'],'Prior result bytes changed')
    return json.loads(raw)


def ridge_phase(directory):
    prior=legacy_result('ridge');raw=resident_artifact(LEGACY_RUN/'ridge/ridge.json',2*1024**2)
    require(sha(raw)==prior['artifact_sha256'],'Resident v1 ridge bytes changed')
    with (directory/'ridge.json').open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    fd=os.open(directory,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return {key:prior[key] for key in ('ridge_identity','source_identity','artifact_sha256','gauge_id','refit_at_utc')} | {
        'synthetic_ridge_fits':0,'ridge_imported_from_v1':True,'legacy_ridge_fit_count':1,'fit_seconds':0.}


def load_ridge(run):
    from gap_pilot_fixture import ROLE_BOUNDS,REFIT,GAUGE,epoch
    from hydra_directflow.gap_filling import RidgeFallback
    prior=prior_result(run,'ridge');raw=read_bytes(run/'ridge/ridge.json',2*1024**2)
    require(sha(raw)==prior['artifact_sha256'],'Synthetic ridge artifact changed')
    model=RidgeFallback.from_dict(json.loads(raw),expected_source_identity=prior['source_identity'],
        expected_gauge_id=GAUGE,expected_gradient_start_epoch=epoch(ROLE_BOUNDS['fit']['start_utc']),
        expected_gradient_end_epoch=epoch(ROLE_BOUNDS['fit']['end_utc']),expected_refit_epoch=epoch(REFIT),
        expected_model_identity=prior['ridge_identity'])
    return model


def make_trainer(mode, ridge):
    from gap_pilot_fixture import training_fixture,ROLE_BOUNDS,REFIT
    from hydra_directflow.pilot_training import PilotTrainer,PilotSpec
    from hydra_directflow.pilot_inputs import make_fill_policy
    arrays,provenance,source,cohort=training_fixture()
    trainer=PilotTrainer(arrays,history_provenance=provenance,train_indices=tuple(range(128)),
        inner_indices=tuple(range(128,192)),role_time_bounds=ROLE_BOUNDS,refit_at_utc=REFIT,
        source_identity=source,cohort_digest=cohort,
        spec=PilotSpec(mode=mode,seed=17,batch_size=64,learning_rate=.0001,shuffle_training=True,max_updates=4),
        fill_policy=make_fill_policy(),
        augmentation_recipe=qualification_augmentation(source,arrays['case_id'][:128].tolist()),ridge_model=ridge)
    return trainer


def qualification_augmentation(source, gradient_ids):
    # Explicit synthetic stress mixture, not an empirical study policy.
    from hydra_directflow.pilot_inputs import make_augmentation_recipe
    from hydra_directflow.reference import digest
    mixture={'interior_1_early':.25,'interior_2_recent':.25,'trailing_2_carry':.25,'mixed_middle_carry':.25}
    return make_augmentation_recipe(mixture=mixture,gradient_case_ids_sha256=digest(gradient_ids),
        source_identity=source,evidence_sha256=digest({'synthetic_qualification_mixture':mixture}))


def train_phase(run,w,directory):
    import torch
    import numpy as np
    from gap_pilot_fixture import training_fixture,inference_fixture,ROLE_BOUNDS,REFIT
    from hydra_directflow.pilot_training import PilotTrainer
    from hydra_directflow.training import EarlyStopping,array_identity
    from hydra_directflow.pilot_inputs import make_fill_policy
    ridge=load_ridge(run)
    if w['phase']=='resume':
        saved=prior_result(run,w['mode']+'-prefix')
        arrays,provenance,source,cohort=training_fixture()
        trainer=PilotTrainer.restore(run/(w['mode']+'-prefix')/'checkpoint.pt',arrays,
            history_provenance=provenance,expected_sha256=saved['checkpoint_sha256'],
            expected_source_identity=source,expected_cohort_digest=cohort,
            expected_mode=w['mode'],expected_fill_policy=make_fill_policy(),
            expected_augmentation_recipe=qualification_augmentation(source,arrays['case_id'][:128].tolist()),ridge_model=ridge)
    else:trainer=make_trainer(w['mode'],ridge)
    trace=[];times=[];original=trainer.update
    def update():
        start=time.monotonic();r=original();times.append(time.monotonic()-start);trace.append(r);return r
    trainer.update=update
    stop=trainer.stopping if trainer.stopping is not None else EarlyStopping(2,2,1,0.)
    for _ in range(2 if w['phase']=='baseline' else 1):trainer.run_epoch(stop if trainer.stopping is None else trainer.stopping)
    checkpoint_sha=trainer.checkpoint(directory/'checkpoint.pt')
    result={'trace':trace,'update_seconds':times,'checkpoint_sha256':checkpoint_sha,
        'model_sha256':tree_digest(trainer.model.state_dict()),'optimizer_sha256':tree_digest(trainer.optimizer.state_dict()),
        'sampler':trainer.sampler.state(),'scaler':asdict(trainer.scaler),
        'source_identity':trainer.source_identity,'cohort_digest':trainer.cohort_admission_digest,
        'fill_policy':trainer.fill_policy,'augmentation_recipe':trainer.augmentation_recipe,
        'gradient_case_ids':trainer.arrays['case_id'][list(trainer.train)].tolist(),
        'best_model_sha256':tree_digest(trainer.best_model_state),'epoch_records':trainer.epoch_records,
        'parameter_count':sum(p.numel() for p in trainer.model.parameters()),
        'current_updates':trainer.updates,'next_batch':trainer.preview_next_batch()[1] if trainer.updates<4 else None}
    require(result['parameter_count']==68161,'Qualified H4 LSTM parameter count differs')
    if w['phase'] in ('baseline','resume'):
        from hydra_directflow.pilot_inputs import prepare_h4_inputs
        arrays,provenance,source,cohort=inference_fixture()
        prepared=prepare_h4_inputs(arrays,history_provenance=provenance,saved_h2_transform=trainer.scaler,
            inference_source_identity=source,fill_policy=trainer.fill_policy,scenario_id='clean')
        require(prepared['case_admissible'].all(),'Synthetic clean inference support differs')
        import copy
        model=copy.deepcopy(trainer.model);model.load_state_dict(trainer.best_model_state);model.eval()
        mapped=prepared['transformed_inputs']
        with torch.inference_mode():
            tensor={k:torch.from_numpy(np.array(v,copy=True)) for k,v in mapped.items() if k in ('history','history_mask','weather','weather_mask','state','state_mask')}
            y=model(*(tensor[k] for k in ('history','history_mask','weather','weather_mask','state','state_mask')))
        physical=trainer.scaler.inverse_target(y.numpy().astype(np.float64))
        with (directory/'reference.npy').open('xb') as stream:np.save(stream,physical,allow_pickle=False)
        result.update(reference_sha256=sha(read_bytes(directory/'reference.npy',128*1024)),
            reference_prediction_sha256=array_identity({'prediction_q':physical}),
            mapped_inputs_sha256=array_identity(mapped),inference_source_identity=source,inference_cohort=cohort)
    return result


def infer_phase(run,w,directory):
    # Late binding keeps planning and ordinary unit tests free of model work.
    import numpy as np
    import torch
    from unittest.mock import patch
    from gap_pilot_fixture import inference_fixture,PoisonLabels,GAUGE
    from hydra_directflow.pilot_training import PilotBestPredictor,PilotTrainer
    from hydra_directflow.training import TrainOnlyTransform,array_identity
    from hydra_directflow.pilot_inputs import scenario_manifest
    base=prior_result(run,w['mode']+'-baseline')
    arrays,provenance,source,cohort=inference_fixture();ridge=load_ridge(run)
    require(source==base['inference_source_identity'] and cohort==base['inference_cohort'],
            'Fresh inference fixture source/cohort differs')
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('Inference constructed optimizer')),\
         patch.object(TrainOnlyTransform,'fit',side_effect=AssertionError('Inference fitted transform')),\
         patch.object(PilotTrainer,'__init__',side_effect=AssertionError('Inference constructed trainer')):
        predictor=PilotBestPredictor.load(run/(w['mode']+'-baseline')/'checkpoint.pt',
            expected_sha256=base['checkpoint_sha256'],expected_source_identity=base['source_identity'],
            expected_cohort_digest=base['cohort_digest'],expected_gauge_id=GAUGE,expected_mode=w['mode'],
            expected_fill_policy=base['fill_policy'],expected_augmentation_recipe=base['augmentation_recipe'],
            gradient_fit_case_ids=tuple(base['gradient_case_ids']),ridge_model=ridge)
        output=predictor.predict(PoisonLabels(arrays),history_provenance=provenance,
            expected_inference_source_identity=source,expected_inference_cohort_digest=cohort,scenario_id='clean')
        raw=read_bytes(run/(w['mode']+'-baseline')/'reference.npy',128*1024)
        require(sha(raw)==base['reference_sha256'],'Reference file changed')
        reference=np.load(io.BytesIO(raw),allow_pickle=False)
        require(np.array_equal(output['prediction_q'],reference),'Fresh best physical predictions differ')
        require(tree_digest(predictor.model.state_dict())==base['best_model_sha256'],'Inference loaded nonbest state')
        scenario_results=[]
        for scenario in scenario_manifest()['scenarios']:
            identifier=scenario['id']
            start=time.monotonic()
            try:
                result=predictor.predict(PoisonLabels(arrays),history_provenance=provenance,
                    expected_inference_source_identity=source,expected_inference_cohort_digest=cohort,scenario_id=identifier)
                scenario_results.append({'id':identifier,'status':'PREDICTED','case_count':len(result['prediction_q']),
                    'seconds':time.monotonic()-start,'prediction_sha256':array_identity({'prediction_q':result['prediction_q']})})
            except ValueError as error:
                # Only independently demonstrated adapter inadmissibility may
                # count as an expected failure; implementation errors still fail.
                from hydra_directflow.pilot_inputs import prepare_h4_inputs
                prepared=prepare_h4_inputs(PoisonLabels(arrays),history_provenance=provenance,
                    saved_h2_transform=predictor.scaler,inference_source_identity=source,
                    fill_policy=predictor.fill_policy,scenario_id=identifier,ridge_model=ridge,
                    expected_ridge_identity=ridge.identity)
                require(expected_inadmissibility(error,prepared['case_admissible']),
                        'Unexpected scenario failure: '+type(error).__name__)
                scenario_results.append({'id':identifier,'status':'EXPECTED_EXPLICIT_INADMISSIBILITY',
                    'structurally_usable_cases':int(prepared['case_admissible'].sum()),'requested_cases':64,
                    'seconds':time.monotonic()-start,'preparation_receipt_sha256':sha(canonical(prepared['receipt']))})
    require(len(scenario_results)==17,'Seventeen fixed scenarios required')
    return {'exact_fresh_best_predictions':True,'saved_transform_sha256':sha(canonical(asdict(predictor.scaler))),
            'best_model_sha256':tree_digest(predictor.model.state_dict()),'scenario_results':scenario_results,
            'target_values_read':False,'optimizer_constructed':False,'scaler_fitted':False,'synthetic_ridge_fits':0}


def worker(run,identifier):
    run=Path(run).resolve();require(run.is_relative_to(ROOT.resolve()),'Worker must run inside stable checkout')
    prepared=read_json(run/'plan.json');require(prepared==plan(),'Worker plan changed')
    w=next(v for v in workers() if v['id']==identifier);deadline=claim(run,w,prepared)
    def deny_network(event,unused):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise RuntimeError('Qualification network forbidden')
    sys.addaudithook(deny_network)
    def timeout(*unused):raise TimeoutError('Qualification worker deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(deadline-time.time())))
    import torch
    import psutil
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    directory=run/identifier;process=psutil.Process()
    state={'pid':os.getpid(),'worker_id':identifier,'optimizer_updates_entered':0,'optimizer_updates_confirmed':0,
           'peak_rss_bytes':0,'plan_sha256':prepared['plan_sha256']}
    def progress():
        state['peak_rss_bytes']=max(state['peak_rss_bytes'],process.memory_info().rss)
        require(time.time()<deadline and state['peak_rss_bytes']<=MAX_BYTES,'Worker resource cap exceeded')
        atomic_json(directory/'status.json',state)
    progress();original_step=torch.optim.AdamW.step
    def step(optimizer,*args,**kwargs):
        require(state['optimizer_updates_entered']<w['updates'],'Worker update allowance exhausted')
        state['optimizer_updates_entered']+=1;progress()
        value=original_step(optimizer,*args,**kwargs)
        state['optimizer_updates_confirmed']+=1;progress();return value
    torch.optim.AdamW.step=step
    try:
        result=ridge_phase(directory) if w['phase']=='ridge' else infer_phase(run,w,directory) if w['phase']=='infer' else train_phase(run,w,directory)
        require(state['optimizer_updates_entered']==state['optimizer_updates_confirmed']==w['updates'],'Worker update count differs')
        progress();atomic_json(directory/'result.json',{**result,**state,'status':'PASSED'})
    finally:torch.optim.AdamW.step=original_step;signal.alarm(0)


def supervise(run,w,deadline,env,plan_sha,*,worker_script=None):
    import psutil
    directory=run/w['id'];mkdir_durable(directory);peak=0;child=None;result=None;error=None
    entered=confirmed=0
    try:
        with (directory/'worker.log').open('x') as log:
            script=Path(__file__).resolve() if worker_script is None else Path(worker_script).resolve()
            require(script.parent==ROOT/'scripts' and script.name in ('qualify_gap_pilot_v2.py','qualify_gap_pilot_constructor_v2.py'),
                    'Explicit prepared v2 worker script required')
            child=subprocess.Popen([sys.executable,'-B',str(script),'--worker',w['id'],'--run-directory',str(run)],cwd=ROOT,env=env,stdout=log,stderr=log)
            process=psutil.Process(child.pid);parent=psutil.Process()
            while child.poll() is None:
                if (directory/'status.json').exists():
                    status=read_json(directory/'status.json');seen=checked_counts(status,w['updates'])
                    entered=max(entered,seen[0]);confirmed=max(confirmed,seen[1])
                try:peak=max(peak,process.memory_info().rss+parent.memory_info().rss)
                except psutil.NoSuchProcess:pass
                require(peak<=MAX_BYTES and time.time()<deadline,'Qualification supervisor cap exceeded')
                time.sleep(.05)
            if (directory/'result.json').exists():result=read_json(directory/'result.json',2*1024**2)
    except BaseException as e:error=type(e).__name__
    finally:
        if child is not None and child.poll() is None:child.kill();child.wait(timeout=5)
        for filename in ('status.json','result.json'):
            if (directory/filename).exists():
                try:
                    value=read_json(directory/filename,2*1024**2);seen=checked_counts(value,w['updates'])
                    entered=max(entered,seen[0]);confirmed=max(confirmed,seen[1])
                except BaseException as exception:error=error or type(exception).__name__
    passed=bool(child is not None and child.returncode==0 and result is not None and result.get('status')=='PASSED'
        and result.get('pid')==child.pid and result.get('worker_id')==w['id'] and result.get('plan_sha256')==plan_sha
        and error is None and entered==confirmed==w['updates'])
    return {'worker_id':w['id'],'passed':passed,'pid':child.pid if child else None,
        'exit_code':child.returncode if child else None,'error_type':error,'optimizer_updates_entered':entered,
        'optimizer_updates_confirmed':confirmed,'count_exact':passed,'peak_parent_child_rss_bytes':peak,
        'result_sha256':sha(read_bytes(directory/'result.json',2*1024**2)) if passed else None}


def execute(prepared,run,report,approved_plan_sha256):
    run=Path(run).absolute();report=Path(report).absolute()
    require(prepared==plan() and prepared['plan_sha256']==approved_plan_sha256,'Exact reviewed plan required')
    require(run.resolve().is_relative_to(ROOT.resolve()) and run.resolve().is_relative_to((ROOT/'.local').resolve())
            and not run.exists() and not report.exists(),
            'Fresh ignored run directory and report required')
    consumed=precharge(prepared,run);mkdir_durable(run);atomic_json(run/'plan.json',prepared)
    receipt={'schema':'hydra-gap-pilot-qualification/v2','status':'RUNNING','plan':prepared,
        'created_at_utc':utc(),'precharged_optimizer_updates':MAX_UPDATES,'optimizer_updates_entered':0,
        'optimizer_updates_confirmed':0,'update_count_exact':False,'attempts':[],'checks':[],
        'scientific_fits':0,'scientific_data_reads':0,'network_requests':0,'protected_assessment_access':False}
    mkdir_durable(report.parent);atomic_json(report,receipt);started=time.monotonic()
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTORCH_ENABLE_MPS_FALLBACK':'0'}
    try:
        for w in workers():
            attempt=supervise(run,w,consumed['deadline_epoch'],env,prepared['plan_sha256'])
            receipt['attempts'].append(attempt)
            for key in ('optimizer_updates_entered','optimizer_updates_confirmed'):receipt[key]+=attempt[key]
            atomic_json(report,receipt)
            with locked():
                ledger=read_json(LEDGER);ledger['workers'][w['id']]={**ledger['workers'][w['id']],**attempt,'status':'PASSED' if attempt['passed'] else 'FAILED'};atomic_json(LEDGER,ledger)
            require(attempt['passed'],'Qualification failed; consumed allowance and evidence preserved')
        for mode in ('clean','augmented'):
            baseline,prefix,resume,infer=(prior_result(run,mode+'-'+phase) for phase in ('baseline','prefix','resume','infer'))
            require(baseline['trace']==prefix['trace']+resume['trace'],'Restart next-update/sample/mask/loss mismatch')
            for key in ('model_sha256','optimizer_sha256','sampler','scaler','best_model_sha256','epoch_records','reference_prediction_sha256'):
                require(baseline[key]==resume[key],'Restored state differs: '+key)
            require(infer['best_model_sha256']==baseline['best_model_sha256'] and infer['saved_transform_sha256']==sha(canonical(baseline['scaler'])),
                    'Saved transform or best state not replayed')
            legacy_check=compare_legacy_baseline(run,mode,baseline)
            receipt['checks'].append({'mode':mode,'exact_restart_sample_mask_loss_optimizer':True,'exact_best_state':True,
                'exact_fresh_label_free_prediction':infer['exact_fresh_best_predictions'],
                'parameter_count':baseline['parameter_count'],'training_update_seconds':baseline['update_seconds'],
                'scenarios':infer['scenario_results'],'v1_equivalence':legacy_check})
        require(receipt['optimizer_updates_confirmed']==MAX_UPDATES,'Qualification total differs')
        require(prepared['spec']['v1_reference']==legacy_binding(),'v1 reference artifacts changed during requalification')
        receipt.update(status='PASSED_SYNTHETIC_H4_GAP_PILOT_REQUALIFICATION_V2',update_count_exact=True,synthetic_ridge_fits=0,
                       old16_update_allowance_untouched=True,resident_v1_ridge_reused=True)
    except BaseException as error:receipt.update(status='FAILED_WITH_CONSUMED_ALLOWANCE',error_type=type(error).__name__)
    finally:receipt['elapsed_seconds']=time.monotonic()-started;atomic_json(report,receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--approved-plan-sha256')
    parser.add_argument('--run-directory',type=Path);parser.add_argument('--report',type=Path)
    parser.add_argument('--worker',choices=[w['id'] for w in workers()]);args=parser.parse_args()
    if args.worker:
        require(args.run_directory is not None,'Worker requires claimed directory');worker(args.run_directory,args.worker);return
    prepared=plan()
    if not args.execute:print(json.dumps(prepared,indent=2));return
    require(args.run_directory is not None and args.report is not None,'Explicit run and report required')
    result=execute(prepared,args.run_directory,args.report,args.approved_plan_sha256)
    print(json.dumps({k:result.get(k) for k in ('status','optimizer_updates_entered','optimizer_updates_confirmed','elapsed_seconds')}))
    require(result['status']=='PASSED_SYNTHETIC_H4_GAP_PILOT_REQUALIFICATION_V2','Qualification incomplete')


if __name__=='__main__':main()
