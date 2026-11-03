#!/usr/bin/env python3
"""Plan-first full-CV1-count SYNTHETIC constructor check; zero neural updates.

One explicit fit+inner scope (17869 rows), or an explicitly selected outer-inclusive
scope (26425 rows). No smaller automatic fallback. One synthetic H2 scaler fit is
counted; no optimizer, ridge refit, forward, scientific arrays or cloud operation.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from qualify_best_inference import atomic_json,mkdir_durable,read_bytes,read_json,canonical,sha
from qualify_gap_pilot_v2 import supervise,checked_counts

COUNTS=ROOT/'reports/confirmed-180-fit-admitted-runtime.json'
LEDGER=ROOT/'.cache/gap-pilot-constructor-budget-v2.json'
LOCK=ROOT/'.cache/gap-pilot-constructor-budget-v2.lock'
RUN=ROOT/'.local/gap-pilot-constructor-v2'
REPORT=ROOT/'reports/gap-pilot-constructor-qualification-v2.json'
MAX_SECONDS=120
MAX_BYTES=4*1024**3
CODE=('scripts/qualify_gap_pilot_constructor_v2.py','scripts/qualify_gap_pilot_v2.py','scripts/qualify_best_inference.py',
      'scripts/gap_pilot_fixture.py','src/hydra_directflow/pilot_training.py','src/hydra_directflow/pilot_inputs.py',
      'src/hydra_directflow/pilot_provenance.py',
      'src/hydra_directflow/gap_filling.py','src/hydra_directflow/gap_policies.py','src/hydra_directflow/training.py',
      'src/hydra_directflow/inference.py','src/hydra_directflow/trainer_bundle.py','src/hydra_directflow/contract.py',
      'src/hydra_directflow/reference.py','src/hydra_directflow/quality.py','src/hydra_directflow/local_data.py',
      'src/hydra_directflow/ablations.py','src/hydra_directflow/engineering.py','src/hydra_directflow/models/directflow.py',
      'src/hydra_directflow/models/mamba2_math.py')


def require(ok,message):
    if not ok:raise ValueError(message)


def authority(include_outer=False):
    require(type(include_outer) is bool,'Explicit source-extent choice required')
    raw=read_bytes(COUNTS,1024**2);value=json.loads(raw)
    require(value['schema']=='hydra-confirmed180-admitted-runtime/v1' and value['scope_fit_count']==180,
            'Confirmed180 count authority required')
    scenarios=[x for x in value['scenarios'] if x['epochs']==20 and x['simulated_label_delay_hours']==2]
    require(len(scenarios)==1,'Unique two-hour count scenario required')
    source=scenarios[0]
    matches=[x for x in source['parts'] if x['fold']=='cv_1' and x['gauge_id']=='03161000'
             and x['architecture']=='LSTM' and x['input_arm']=='full' and x['history_hours']==168]
    require(len(matches)==1,'Exact largestCV1 gauge geometry required')
    selected=matches[0]
    require(selected['actual_common_cases']=={'gradient_fit':15717,'inner_stopping':2152,'outer_scoring':8556},
            'CV1 cardinalities changed; do not silently retarget constructor experiment')
    window=next(x for x in source['fold_windows'] if x['name']=='cv_1')
    bounds={short:{'start_utc':window[long]['start_utc'],'end_utc':window[long]['end_exclusive_utc']}
            for short,long in (('fit','gradient_fit'),('inner_stop','inner_stopping'),('outer','outer_scoring'))}
    counts=dict(fit=15717,inner=2152,outer=8556 if include_outer else 0)
    return dict(report_sha256=sha(raw),snapshot_sha256=value['snapshot_sha256'],
        scientific_case_membership_not_reused=True,synthetic_cases_use_same_counts_not_observed_calendar_selection=True,
        gauge_id='03161000',counts=counts,total_source_rows=sum(counts.values()),role_time_bounds=bounds,
        refit_at_utc=window['inner_stopping']['refit_at_utc'],unassigned_rows_in_this_fixture=0,
        source_extent='FIT_INNER_AND_OUTER' if include_outer else 'FIT_AND_INNER_ONLY',
        limitation='If the actual future constructor retains further unassigned/source rows, this smaller geometry does not qualify that larger source extent')


def implementation():
    return dict(code={name:sha(read_bytes(ROOT/name,512*1024)) for name in CODE},
        dependencies={name:importlib.metadata.version(name) for name in ('numpy','torch','transformers','psutil')})


def plan(include_outer=False):
    spec=dict(source='SYNTHETIC_VALUES_AND_EXPLICIT_SYNTHETIC_PROVENANCE_ONLY',authority=authority(include_outer),
        implementation=implementation(),maximum_optimizer_updates=0,maximum_synthetic_scaler_fits=1,
        maximum_ridge_fits=0,maximum_model_forward_calls=0,maximum_wall_seconds=MAX_SECONDS,
        maximum_parent_child_rss_bytes=MAX_BYTES,device='cpu',torch_threads=1,batch_size=64,history_hours=168,
        prepared_clean_batches=1,prepared_augmented_batches=1,automatic_size_reduction=False,
        scientific_fits_authorized=False,include_outer=include_outer)
    return dict(schema='hydra-gap-pilot-constructor-plan/v2',status='PREPARED_NOT_EXECUTED',spec=spec,plan_sha256=sha(canonical(spec)))


def locked():
    require(LOCK.parent.resolve().is_relative_to(ROOT.resolve()),'Constructor ledger must stay inside stable checkout')
    mkdir_durable(LOCK.parent);stream=LOCK.open('a+');fcntl.flock(stream.fileno(),fcntl.LOCK_EX);return stream


def precharge(prepared):
    require(prepared==plan(prepared['spec']['include_outer']),'Constructor plan changed')
    with locked():
        require(not LEDGER.exists() and not LEDGER.is_symlink(),'Constructor allowance already consumed; no reset/retry or size switch')
        value=dict(schema='hydra-gap-pilot-constructor-allowance/v2',plan_sha256=prepared['plan_sha256'],
            source_rows=prepared['spec']['authority']['total_source_rows'],status='PRECHARGED',
            precharged_optimizer_updates=0,precharged_synthetic_scaler_fits=1,
            deadline_epoch=time.time()+MAX_SECONDS,created_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(LEDGER,value)
    return value


def claim(prepared):
    with locked():
        value=read_json(LEDGER)
        require(value['status']=='PRECHARGED' and value['plan_sha256']==prepared['plan_sha256']
                and time.time()<value['deadline_epoch'],'Exact unclaimed constructor allowance required')
        value.update(status='CLAIMED',pid=os.getpid());atomic_json(LEDGER,value)
    return value['deadline_epoch']


def synthetic_fixture(authority_value):
    """Same physical input geometry; no actual source observations are loaded."""
    import numpy as np
    from gap_pilot_fixture import inputs,synthetic_q,epoch
    from hydra_directflow.training import array_identity
    from hydra_directflow.reference import digest
    counts=authority_value['counts'];bounds=authority_value['role_time_bounds']
    origins=np.concatenate([epoch(bounds[role]['start_utc'])+np.arange(counts[key],dtype=np.int64)*3600
        for role,key in (('fit','fit'),('inner_stop','inner'),('outer','outer')) if counts[key]])
    arrays,provenance,_,cohort=inputs(origins)
    arrays['target_time']=arrays.pop('forecast_time');arrays['target']=synthetic_q(arrays['target_time'])
    arrays['target_mask']=np.ones(arrays['target'].shape,bool)
    source=digest({'fixture':'SYNTHETIC_FULL_CV1_CONSTRUCTOR_V2','authority_sha256':authority_value['report_sha256'],
        'source_extent':authority_value['source_extent'],'arrays_sha256':array_identity(arrays),'provenance_sha256':digest(provenance)})
    return arrays,provenance,source,cohort


def worker(run,identifier):
    require(RUN.resolve().is_relative_to(ROOT.resolve()) and Path(run).resolve()==RUN.resolve()
            and identifier=='constructor','Fixed stable constructor worker path/identity required')
    prepared=read_json(RUN/'plan.json');require(prepared==plan(prepared['spec']['include_outer']),'Worker constructor plan changed')
    deadline=claim(prepared)
    def deny_network(event,unused):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise RuntimeError('Constructor network forbidden')
    sys.addaudithook(deny_network)
    def timeout(*unused):raise TimeoutError('Full-cohort constructor deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(deadline-time.time())))
    import numpy as np
    import torch
    import psutil
    from hydra_directflow.pilot_training import PilotTrainer,PilotSpec
    from hydra_directflow.pilot_inputs import make_fill_policy,make_augmentation_recipe
    from hydra_directflow.training import TrainOnlyTransform
    from hydra_directflow.reference import digest
    from hydra_directflow.models.directflow import DirectFlowModel
    import hydra_directflow.gap_filling as filling
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    directory=RUN/'constructor';process=psutil.Process();started=time.monotonic()
    state=dict(pid=os.getpid(),worker_id='constructor',plan_sha256=prepared['plan_sha256'],optimizer_updates_entered=0,
        optimizer_updates_confirmed=0,synthetic_scaler_fits_entered=0,synthetic_scaler_fits_confirmed=0,model_forward_calls=0,
        ridge_fits=0,observed_peak_process_rss_bytes=0,phase='STARTING')
    def progress(phase):
        state['phase']=phase;state['observed_peak_process_rss_bytes']=max(state['observed_peak_process_rss_bytes'],process.memory_info().rss)
        require(time.time()<deadline and state['observed_peak_process_rss_bytes']<=MAX_BYTES,'Constructor resource ceiling reached')
        atomic_json(directory/'status.json',state)
    def forbid(*args,**kwargs):raise AssertionError('No optimizer/backward/model forward/ridge fit allowed in constructor check')
    original_fit=TrainOnlyTransform.fit
    def counted_fit(*args,**kwargs):
        require(state['synthetic_scaler_fits_entered']==0,'Only one original synthetic gradient scaler fit is permitted')
        state['synthetic_scaler_fits_entered']+=1;progress('SCALER_FIT_ENTERED')
        result=original_fit(*args,**kwargs);state['synthetic_scaler_fits_confirmed']+=1;progress('SCALER_FIT_CONFIRMED');return result
    with (patch.object(torch.optim.Optimizer,'__init__',side_effect=forbid),patch.object(torch.Tensor,'backward',side_effect=forbid),
          patch.object(PilotTrainer,'update',side_effect=forbid),patch.object(DirectFlowModel,'forward',side_effect=forbid),
          patch.object(filling,'fit_ridge_fallback',side_effect=forbid),patch.object(TrainOnlyTransform,'fit',side_effect=counted_fit)):
        progress('FIXTURE_BUILD');tick=time.monotonic()
        authority_value=prepared['spec']['authority'];arrays,provenance,source,cohort=synthetic_fixture(authority_value)
        fixture_seconds=time.monotonic()-tick;progress('FIXTURE_READY')
        counts=authority_value['counts'];train=tuple(range(counts['fit']));inner=tuple(range(counts['fit'],counts['fit']+counts['inner']))
        recipe=make_augmentation_recipe(mixture={'interior_1_middle':.5,'trailing_2_carry':.5},
            gradient_case_ids_sha256=digest(arrays['case_id'][list(train)].tolist()),source_identity=source,
            evidence_sha256=digest({'synthetic_full_cohort_recipe':'V2'}))
        source_bytes=sum(value.nbytes for value in arrays.values());tick=time.monotonic()
        trainer=PilotTrainer(arrays,history_provenance=provenance,train_indices=train,inner_indices=inner,
            role_time_bounds=authority_value['role_time_bounds'],refit_at_utc=authority_value['refit_at_utc'],
            spec=PilotSpec(mode='augmented',max_updates=0),source_identity=source,cohort_digest=cohort,
            fill_policy=make_fill_policy(),augmentation_recipe=recipe)
        constructor_seconds=time.monotonic()-tick;progress('CONSTRUCTOR_READY')
        tick=time.monotonic();clean,clean_receipt=trainer.preview_batch(tuple(range(64)),epoch=0,augment=False)
        clean_seconds=time.monotonic()-tick;progress('CLEAN_BATCH_READY')
        tick=time.monotonic();aug,aug_receipt=trainer.preview_batch(tuple(range(64)),epoch=0,augment=True)
        augmented_seconds=time.monotonic()-tick;progress('AUGMENTED_BATCH_READY')
        require(trainer.optimizer is None and trainer.updates==trainer.updates_entered==0
                and state['synthetic_scaler_fits_confirmed']==1,'Constructor work counters differ')
        require(clean['history'].shape==aug['history'].shape==(64,168,4)
                and clean['history_mask'].all() and aug['history_mask'].all(),'Prepared full-cohort H4 batch geometry/support differs')
        require(len(trainer.arrays['case_id'])==authority_value['total_source_rows'],'Constructor silently changed source extent')
        result=dict(status='PASSED',authority=authority_value,synthetic_source_identity=source,synthetic_cohort_digest=cohort,
            source_numpy_bytes=source_bytes,history_provenance_slots=168*authority_value['total_source_rows'],
            fixture_seconds=fixture_seconds,constructor_seconds=constructor_seconds,clean_batch_seconds=clean_seconds,
            augmented_batch_seconds=augmented_seconds,clean_batch_receipt=clean_receipt,augmented_batch_receipt=aug_receipt,
            parameter_count=sum(p.numel() for p in trainer.model.parameters()),transform_sha256=trainer.transform_sha256,
            original_gradient_transform_case_ids_sha256=trainer.scaler.training_case_ids_sha256,
            optimizer_constructed=False,scientific_fits=0,study_data_reads=0,
            automatic_size_reduction=False,elapsed_seconds=time.monotonic()-started,**state)
        require(prepared==plan(prepared['spec']['include_outer']),'Constructor implementation/count authority changed')
        atomic_json(directory/'result.json',result)
    signal.alarm(0)


def execute(prepared,approved_plan_sha256):
    require(prepared==plan(prepared['spec']['include_outer']) and prepared['plan_sha256']==approved_plan_sha256,'Exact reviewed constructor plan required')
    require(RUN.resolve().is_relative_to(ROOT.resolve()) and not RUN.exists() and not REPORT.exists()
            and not RUN.is_symlink() and not REPORT.is_symlink(),'Fresh fixed stable constructor outputs required')
    consumed=precharge(prepared);mkdir_durable(RUN);atomic_json(RUN/'plan.json',prepared)
    result=dict(schema='hydra-gap-pilot-constructor-qualification/v2',status='RUNNING',plan=prepared,
        precharged_optimizer_updates=0,precharged_synthetic_scaler_fits=1,scientific_fits=0,protected_assessment_access=False)
    atomic_json(REPORT,result)
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTORCH_ENABLE_MPS_FALLBACK':'0'}
    attempt=supervise(RUN,dict(id='constructor',updates=0),consumed['deadline_epoch'],env,prepared['plan_sha256'],worker_script=Path(__file__))
    result['attempt']=attempt
    result['status']='PASSED_SYNTHETIC_FULL_COHORT_CONSTRUCTOR' if attempt['passed'] else 'FAILED_WITH_CONSUMED_CONSTRUCTOR_ALLOWANCE'
    status_path=RUN/'constructor/status.json'
    if status_path.exists():
        try:
            status=read_json(status_path)
            entered=status.get('synthetic_scaler_fits_entered',0);confirmed=status.get('synthetic_scaler_fits_confirmed',0)
            require(type(entered) is int and type(confirmed) is int and 0<=confirmed<=entered<=1,'Constructor scaler counters differ')
            result.update(synthetic_scaler_fits_entered=entered,synthetic_scaler_fits_confirmed=confirmed,
                          scaler_count_exact=attempt['passed'],last_observed_phase=status.get('phase'))
        except BaseException as error:
            result.update(status='FAILED_WITH_CONSUMED_CONSTRUCTOR_ALLOWANCE',synthetic_scaler_fits_entered=None,
                          synthetic_scaler_fits_confirmed=None,scaler_count_exact=False,scaler_status_error_type=type(error).__name__)
    else:result.update(synthetic_scaler_fits_entered=0,synthetic_scaler_fits_confirmed=0,scaler_count_exact=False)
    if attempt['passed']:
        try:
            raw=read_bytes(RUN/'constructor/result.json',2*1024**2)
            require(sha(raw)==attempt['result_sha256'],'Constructor result bytes changed after supervision')
            result['constructor']=json.loads(raw)
            require(result.get('synthetic_scaler_fits_entered')==result.get('synthetic_scaler_fits_confirmed')==1,
                    'Exactly one synthetic scaler fit must be accounted for')
        except BaseException as error:
            result.update(status='FAILED_WITH_CONSUMED_CONSTRUCTOR_ALLOWANCE',scaler_count_exact=False,
                          result_validation_error_type=type(error).__name__)
    passed=attempt['passed'] and result['status']=='PASSED_SYNTHETIC_FULL_COHORT_CONSTRUCTOR'
    with locked():
        ledger=read_json(LEDGER);ledger.update(status='PASSED' if passed else 'FAILED',attempt=attempt);atomic_json(LEDGER,ledger)
    atomic_json(REPORT,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--include-outer',action='store_true')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--approved-plan-sha256')
    parser.add_argument('--worker',choices=('constructor',));parser.add_argument('--run-directory',type=Path)
    args=parser.parse_args()
    if args.worker:worker(args.run_directory,args.worker);return
    prepared=plan(args.include_outer)
    if not args.execute:print(json.dumps(prepared,indent=2));return
    result=execute(prepared,args.approved_plan_sha256);print(json.dumps({'status':result['status']}))
    require(result['status']=='PASSED_SYNTHETIC_FULL_COHORT_CONSTRUCTOR','Full-cohort constructor qualification incomplete')


if __name__=='__main__':main()
