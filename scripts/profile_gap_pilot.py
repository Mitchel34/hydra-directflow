#!/usr/bin/env python3
"""Bounded synthetic-only, zero-update profile of an already qualified H4 model.

Default is plan-only. This fixed one-run report is created before timing starts;
an interrupted/failed profile is not silently replayed. No existing code, source,
qualification checkpoint or optimizer allowance is modified.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import qualify_gap_pilot as qualification
from qualify_best_inference import atomic_json,canonical,read_bytes,read_json,sha

RUN=ROOT/'.local/gap-pilot-qualification-v1'
QUALIFIED_REPORT=ROOT/'reports/gap-pilot-qualification-v1.json'
REPORT=ROOT/'reports/gap-pilot-preparation-profile.json'
CALLS=20
MAX_SECONDS=120
MAX_BYTES=4*1024**3
STAGES=('clean_adapter','augmented_adapter','provenance_json_digest','model_only_forward','full_clean_predictor')


def require(ok,message):
    if not ok:raise ValueError(message)


def implementation():
    names=sorted(set(qualification.CODE+('scripts/profile_gap_pilot.py',)))
    return dict(code={name:sha(read_bytes(ROOT/name,512*1024)) for name in names},
        dependencies={name:importlib.metadata.version(name) for name in ('numpy','torch','transformers','psutil')})


def plan():
    raw=read_bytes(QUALIFIED_REPORT,1024**2);qualified=json.loads(raw)
    require(qualified['status']=='PASSED_SYNTHETIC_H4_GAP_PILOT_QUALIFICATION'
            and qualified['optimizer_updates_entered']==qualified['optimizer_updates_confirmed']==16
            and qualified['plan']['spec']['source']=='SYNTHETIC_GENERATOR_ONLY_NO_EXTERNAL_ARRAY_ARGUMENT',
            'Existing exact synthetic qualification required')
    current=implementation()
    for name,expected in qualified['plan']['spec']['implementation']['code'].items():
        require(current['code'][name]==expected,'Qualified helper code changed: '+name)
    for name,expected in qualified['plan']['spec']['implementation']['dependencies'].items():
        require(current['dependencies'][name]==expected,'Qualified dependency changed: '+name)
    base=qualification.prior_result(RUN,'clean-baseline')
    checkpoint=read_bytes(RUN/'clean-baseline/checkpoint.pt',64*1024**2)
    require(sha(checkpoint)==base['checkpoint_sha256'],'Qualified synthetic checkpoint changed')
    spec=dict(source='EXISTING_QUALIFIED_SYNTHETIC_FIXTURE_ONLY',qualified_report_sha256=sha(raw),
        qualification_plan_sha256=qualified['plan']['plan_sha256'],checkpoint_sha256=base['checkpoint_sha256'],
        baseline_result_sha256=sha(read_bytes(RUN/'clean-baseline/result.json',2*1024**2)),
        stages=list(STAGES),timed_calls_per_stage=CALLS,maximum_wall_seconds=MAX_SECONDS,
        maximum_process_rss_bytes=MAX_BYTES,device='cpu',torch_threads=1,batch_size=64,history_hours=168,
        dtype='float32',width=64,parameter_count=68161,optimizer_updates=0,
        backward_calls=0,model_fits=0,scaler_fits=0,ridge_fits=0,study_data_reads=0,network_requests=0,
        implementation=current)
    return dict(schema='hydra-gap-pilot-profile-plan/v1',status='PREPARED_NOT_EXECUTED',spec=spec,plan_sha256=sha(canonical(spec)))


def timing_summary(values):
    require(values and all(type(v) in (float,int) and 0<=v<MAX_SECONDS for v in values),'Finite bounded timing samples required')
    ordered=sorted(values)
    return dict(calls=len(values),seconds=values,first_seconds=values[0],mean_seconds=statistics.fmean(values),
        median_seconds=statistics.median(values),min_seconds=min(values),max_seconds=max(values),
        later_mean_seconds=statistics.fmean(values[1:]) if len(values)>1 else None,
        p10_observed_order_statistic_seconds=ordered[max(0,int(.1*(len(ordered)-1)))],
        p90_observed_order_statistic_seconds=ordered[int(.9*(len(ordered)-1))],
        confidence_interval=False)


def start_report(path,value):
    """Durable exclusive first claim; later atomic replacements belong to its owner."""
    path=Path(path)
    with path.open('xb') as stream:
        stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
    descriptor=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def execute(prepared,approved_plan_sha256):
    require(prepared==plan() and prepared['plan_sha256']==approved_plan_sha256,'Exact unchanged zero-update profile plan required')
    require(not REPORT.exists() and not REPORT.is_symlink(),'Fixed profile report already exists; no silent rerun')
    receipt=dict(schema='hydra-gap-pilot-preparation-profile/v1',status='RUNNING',plan=prepared,
        created_at_utc=datetime.now(timezone.utc).isoformat(),stages={},setup={},
        calls_entered={name:0 for name in STAGES},calls_completed={name:0 for name in STAGES},
        optimizer_updates=0,backward_calls=0,model_fits=0,scaler_fits=0,ridge_fits=0,
        study_data_reads=0,scientific_scores=0,network_requests=0,protected_assessment_access=False)
    start_report(REPORT,receipt)
    started=time.monotonic();peak=0;old_handler=None
    def deny_network(event,unused):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise RuntimeError('Profile network forbidden')
    sys.addaudithook(deny_network)
    def timeout(*unused):raise TimeoutError('Synthetic profile120second deadline')
    old_handler=signal.signal(signal.SIGALRM,timeout);signal.alarm(MAX_SECONDS)
    try:
        import numpy as np
        import torch
        import psutil
        from gap_pilot_fixture import training_fixture,inference_fixture,PoisonLabels,GAUGE
        from hydra_directflow.pilot_inputs import prepare_h4_inputs
        from hydra_directflow.pilot_training import PilotBestPredictor,PilotTrainer,_canonical
        from hydra_directflow.training import TrainOnlyTransform,array_identity
        from hydra_directflow.reference import digest
        import hydra_directflow.gap_filling as filling
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        process=psutil.Process()
        def guard():
            nonlocal peak
            peak=max(peak,process.memory_info().rss)
            require(time.monotonic()-started<MAX_SECONDS and peak<=MAX_BYTES,'Synthetic profile resource bound reached')
        def measure(name,fn):
            seconds=[];last=None
            for i in range(CALLS):
                guard();receipt['calls_entered'][name]+=1
                start=time.perf_counter();last=fn(i);seconds.append(time.perf_counter()-start)
                receipt['calls_completed'][name]+=1;guard()
            receipt['stages'][name]=timing_summary(seconds)
            atomic_json(REPORT,receipt)
            return last
        def forbidden(*unused,**kwargs):raise AssertionError('Profile attempted an optimizer/backward/scaler/ridge fit')
        ledger_before=sha(read_bytes(qualification.LEDGER,1024**2))
        with (patch.object(torch.optim.Optimizer,'__init__',side_effect=forbidden),
              patch.object(torch.optim.AdamW,'step',side_effect=forbidden),
              patch.object(torch.Tensor,'backward',side_effect=forbidden),
              patch.object(torch.autograd,'backward',side_effect=forbidden),
              patch.object(TrainOnlyTransform,'fit',side_effect=forbidden),
              patch.object(PilotTrainer,'__init__',side_effect=forbidden),
              patch.object(filling,'fit_ridge_fallback',side_effect=forbidden)):
            guard();load_start=time.perf_counter()
            baseline=qualification.prior_result(RUN,'clean-baseline');ridge=qualification.load_ridge(RUN)
            predictor=PilotBestPredictor.load(RUN/'clean-baseline/checkpoint.pt',
                expected_sha256=baseline['checkpoint_sha256'],expected_source_identity=baseline['source_identity'],
                expected_cohort_digest=baseline['cohort_digest'],expected_gauge_id=GAUGE,expected_mode='clean',
                expected_fill_policy=baseline['fill_policy'],expected_augmentation_recipe=baseline['augmentation_recipe'],
                gradient_fit_case_ids=tuple(baseline['gradient_case_ids']),ridge_model=ridge)
            receipt['setup']['checkpoint_and_ridge_load_seconds']=time.perf_counter()-load_start
            receipt['checkpoint_receipt']=predictor.receipt
            arrays,provenance,source,cohort=training_fixture()
            fields=('history','history_mask','weather','weather_mask','state','state_mask','origin_time','history_time','target_time','site_id','case_id')
            batch={name:arrays[name][:64].copy() for name in fields}
            batch.update(feature_schema=arrays['feature_schema'],cohort_admission_digest=arrays['cohort_admission_digest'])
            provenance={name:value[:64] for name,value in provenance.items()}
            batch=PoisonLabels(batch)
            def adapter(epoch,augment):
                value=prepare_h4_inputs(batch,history_provenance=provenance,saved_h2_transform=predictor.scaler,
                    inference_source_identity=source,fill_policy=baseline['fill_policy'],
                    scenario_id='augmentation' if augment else 'clean',epoch=epoch,mask_seed=1729,
                    augmentation_recipe=baseline['augmentation_recipe'] if augment else None,
                    ridge_model=ridge,expected_ridge_identity=ridge.identity)
                require(value['case_admissible'].all(),'Synthetic profile adapter lost admitted cases')
                return value
            clean=measure('clean_adapter',lambda _:adapter(0,False))
            augmented=measure('augmented_adapter',lambda i:adapter(i,True))
            receipt['adapter_bindings']=dict(source_identity=source,cohort_digest=cohort,
                original_case_ids_sha256=digest(batch['case_id'].tolist()),clean_receipt_sha256=digest(clean['receipt']),
                augmented_last_receipt_sha256=digest(augmented['receipt']),
                augmented_epochs=list(range(CALLS)),recipe_sha256=digest(baseline['augmentation_recipe']),
                labels_present_in_mapping=False,source_values_synthetic=True)
            provenance_digest=measure('provenance_json_digest',lambda _:digest(_canonical(clean['provenance'])))
            receipt['adapter_bindings']['full_clean_provenance_sha256']=provenance_digest
            outer,outer_provenance,outer_source,outer_cohort=inference_fixture();outer=PoisonLabels(outer)
            setup_start=time.perf_counter()
            prepared_outer=prepare_h4_inputs(outer,history_provenance=outer_provenance,saved_h2_transform=predictor.scaler,
                inference_source_identity=outer_source,fill_policy=baseline['fill_policy'],scenario_id='clean',
                ridge_model=ridge,expected_ridge_identity=ridge.identity)
            require(prepared_outer['case_admissible'].all(),'Synthetic outer preparation failed')
            receipt['setup']['untimed_outer_adapter_calls']=1
            receipt['setup']['outer_adapter_seconds']=time.perf_counter()-setup_start
            tensor_start=time.perf_counter()
            tensors=tuple(torch.from_numpy(np.array(prepared_outer['transformed_inputs'][name],copy=True))
                for name in ('history','history_mask','weather','weather_mask','state','state_mask'))
            receipt['setup']['tensorization_seconds']=time.perf_counter()-tensor_start
            predictor.model.eval()
            with torch.inference_mode():
                raw=measure('model_only_forward',lambda _:predictor.model(*tensors))
            physical=predictor.scaler.inverse_target(raw.numpy().astype(np.float64))
            require(array_identity({'prediction_q':physical})==baseline['reference_prediction_sha256'],
                'Profile model-only output differs from qualified best-state reference')
            output=measure('full_clean_predictor',lambda _:predictor.predict(outer,history_provenance=outer_provenance,
                expected_inference_source_identity=outer_source,expected_inference_cohort_digest=outer_cohort,scenario_id='clean'))
            require(np.array_equal(output['prediction_q'],physical),'Full prediction differs from model-only result')
            receipt['prediction_binding']=dict(prediction_sha256=array_identity({'prediction_q':physical}),
                inference_source_identity=outer_source,inference_cohort_digest=outer_cohort,
                qualified_best_reference_exact=True,labels_read=False,posthoc_clipping=False)
            require(sha(read_bytes(qualification.LEDGER,1024**2))==ledger_before,'Qualification update ledger changed')
            require(prepared==plan(),'Profile code/source/checkpoint binding changed during run')
            receipt['execution_environment']=dict(python=platform.python_version(),system=platform.system(),machine=platform.machine(),
                torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),dtype=str(torch.get_default_dtype()),device='cpu')
            receipt['qualification_ledger_unchanged']=True
            receipt['stage_scope']=dict(clean_adapter='Coordinate/provenance checks, gap classification/fill, H4 assembly, saved transform, copies and adapter receipt',
                augmented_adapter='Same operations plus deterministic case/epoch gap selection; first64 original-gradient synthetic rows',
                provenance_json_digest='Pilot wrapper full provenance normalization and SHA digest after clean adapter; excludes adapter work',
                model_only_forward='Pre-tensorized eval-mode CPU forward only; excludes prep/copy/inverse transform/receipt',
                full_clean_predictor='Best-predictor full outer input path including checks/adapter/tensorization/forward/inverse/receipt')
            receipt['interpretation']='SMALL_SAME_PROCESS_SYNTHETIC_PROFILE_NOT_STABLE_THROUGHPUT_OR_CONFIDENCE_BOUNDS;STAGES_NOT_A_DISJOINT_ADDITIVE_TRACE'
            receipt['status']='PASSED_SYNTHETIC_ZERO_UPDATE_PROFILE'
    except BaseException as error:
        receipt.update(status='FAILED_NO_AUTOMATIC_RETRY',error_type=type(error).__name__)
    finally:
        signal.alarm(0)
        if old_handler is not None:signal.signal(signal.SIGALRM,old_handler)
        receipt['elapsed_seconds']=time.monotonic()-started;receipt['observed_peak_process_rss_bytes']=peak
        atomic_json(REPORT,receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--approved-plan-sha256')
    args=parser.parse_args();prepared=plan()
    if not args.execute:print(json.dumps(prepared,indent=2));return
    result=execute(prepared,args.approved_plan_sha256)
    print(json.dumps({key:result.get(key) for key in ('status','elapsed_seconds','observed_peak_process_rss_bytes')}))
    require(result['status']=='PASSED_SYNTHETIC_ZERO_UPDATE_PROFILE','Synthetic preparation profile incomplete')


if __name__=='__main__':main()
