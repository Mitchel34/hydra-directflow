#!/usr/bin/env python3
"""Fixed27-view resident primary data qualification; zero fits or model imports.

Metadata-only by default. A fresh once-only allowance is required for the
bounded local numeric read. No raw source caches, cloud, hydration or scoring.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from assemble_gap_pilot import canonical,require,sha,sync_directory,utc,write_new
from qualify_pilot_assembly_loaders import ReadGuard,stream_array_identity,forbid

CORPUS=Path.home()/'.codex/hydra-directflow-runtime/development-corpus-L168-v4'
RUN=ROOT/'.local/primary-campaign-data-qualification-v1'
REPORT=ROOT/'reports/primary-campaign-data-qualification-v1.json'
LEDGER=ROOT/'.cache/primary-campaign-data-qualification-budget-v1.json'
CLAIM=ROOT/'.cache/primary-campaign-data-qualification-claim-v1.json'
MAX_SECONDS=900
MAX_RSS=4*1024**3
MAX_BUNDLE=1024**3
MAX_READ=8*1024**3
CODE=('scripts/qualify_campaign_data.py','scripts/qualify_pilot_assembly_loaders.py','scripts/assemble_gap_pilot.py',
      'src/hydra_directflow/campaign_data.py','src/hydra_directflow/training.py',
      'src/hydra_directflow/inference.py','src/hydra_directflow/trainer_bundle.py')


def implementation():
    from hydra_directflow import campaign_data as data
    from hydra_directflow.corpus_dataset import _read
    return dict(controller={name:sha(_read(ROOT/name,1024**2)) for name in CODE},loader=data.implementation_identity())


def unique_jobs(roster):
    from hydra_directflow.local_data import GAUGES
    require(len(roster['jobs'])==roster['fits']==180,'Exact180-job roster required')
    selected=[]; seen=set()
    for job in roster['jobs']:
        key=(job['fold'],job['gauge_id'],job['input_arm'],job['history_hours'])
        if key not in seen: selected.append(job); seen.add(key)
    expected={(fold,gauge,arm,168) for fold in ('cv_1','cv_2','cv_3','final_development') for gauge in GAUGES for arm in ('full','q_only')}
    expected.update(('final_development',gauge,'full',24) for gauge in GAUGES)
    require(seen==expected and len(selected)==27,'Exact27 distinct primary data views required')
    require(all(job['architecture']=='LSTM' and job['seed']==17 for job in selected),'Use first exact seed17/LSTM roster representatives; no architecture/seed duplication')
    return selected


def validate_paths():
    require(ROOT==ROOT.resolve() and CORPUS==CORPUS.resolve() and CORPUS.is_dir(),'Stable resident corpus path required')
    require(RUN==RUN.resolve() and REPORT==REPORT.resolve() and LEDGER==LEDGER.resolve() and CLAIM==CLAIM.resolve(),
            'Fixed real controller/report/allowance paths required')
    require(RUN.is_relative_to(ROOT/'.local') and REPORT.parent==ROOT/'reports'
            and LEDGER.parent==CLAIM.parent==ROOT/'.cache','Fixed private/public artifact boundaries required')


def plan():
    """Read roster/integrity/PLAN metadata only, never shard NPZ values or masks."""
    from hydra_directflow import campaign_data as data
    validate_paths()
    roster,_=data._pinned(data.ROSTER_PATH,data.ROSTER_SHA256,4*1024**2)
    jobs=unique_jobs(roster)
    authority=data._authority(CORPUS,jobs[0],data.INTEGRITY_SHA256)
    spec=dict(source_plan_sha256=data.SOURCE_PLAN,snapshot_sha256=data.SNAPSHOT,roster_sha256=data.ROSTER_SHA256,
        integrity_receipt_sha256=data.INTEGRITY_SHA256,source_plan_file_sha256=authority['plan_file_sha256'],
        jobs=jobs,data_views=27,training_source_extent='FIT_AND_INNER_ONLY',outer_source_extent='ALL_DECLARED_COMMON_OUTER_INPUTS_AND_LABELS',
        inspected_prediction_batch='FIRST_MAX64_CASES_ONLY_NO_FORWARD_CALL',maximum_wall_seconds=MAX_SECONDS,
        maximum_parent_child_rss_bytes=MAX_RSS,maximum_bundle_estimated_bytes=MAX_BUNDLE,
        maximum_cumulative_corpus_logical_read_bytes=MAX_READ,maximum_each_loader_read_bytes=MAX_BUNDLE,
        torch_imports=0,model_calls=0,scaler_fits=0,optimizer_updates=0,scientific_scores=0,network_requests=0,
        source_cache_reads=0,protected_assessment_access=False,implementation=implementation())
    return dict(schema='hydra-primary-campaign-data-qualification-plan/v1',status='PREPARED_NOT_EXECUTED',spec=spec,plan_sha256=sha(canonical(spec)))


def precharge(prepared):
    require(prepared==plan(),'Qualification source/code/scope changed')
    require(not RUN.exists() and not REPORT.exists() and not LEDGER.exists() and not CLAIM.exists(),
            'Fresh once-only data qualification required; preserve all prior attempts')
    LEDGER.parent.mkdir(exist_ok=True); sync_directory(LEDGER.parent.parent)
    allowance=dict(plan_sha256=prepared['plan_sha256'],created_at_utc=utc(),deadline_epoch=time.time()+MAX_SECONDS,
        maximum_cumulative_corpus_logical_read_bytes=MAX_READ,maximum_seconds=MAX_SECONDS,
        maximum_parent_child_rss_bytes=MAX_RSS,maximum_bundle_estimated_bytes=MAX_BUNDLE,
        model_calls=0,scaler_fits=0,optimizer_updates=0)
    write_new(LEDGER,allowance)
    RUN.mkdir(); sync_directory(RUN.parent); write_new(RUN/'plan.json',prepared)
    return allowance


def claim(prepared):
    allowance=json.loads(LEDGER.read_text())
    require(allowance['plan_sha256']==prepared['plan_sha256'] and time.time()<allowance['deadline_epoch']
            and allowance['maximum_cumulative_corpus_logical_read_bytes']==MAX_READ
            and allowance['maximum_parent_child_rss_bytes']==MAX_RSS,'Exact live data qualification allowance required')
    write_new(CLAIM,dict(plan_sha256=prepared['plan_sha256'],pid=os.getpid(),at_utc=utc()))
    return allowance


def _membership(bundle,job):
    import numpy as np
    from hydra_directflow.reference import digest
    from hydra_directflow.training import validate_development_bundle
    from hydra_directflow.trainer_bundle import _hour,refit_label_mask
    arrays=bundle['arrays']; n=validate_development_bundle(arrays,job['history_hours'],feature_schema=arrays['feature_schema'].item())
    train,inner=bundle['train_indices'],bundle['inner_indices']
    require(tuple(sorted((*train,*inner)))==tuple(range(n)) and len(set((*train,*inner)))==n,'Only disjoint fit/inner cases may enter training')
    for short,long,indices in (('fit','gradient_fit',train),('inner_stop','inner_stopping',inner)):
        ids=arrays['case_id'][list(indices)].tolist(); bounds=bundle['role_time_bounds'][short]
        require(len(ids)==job['actual_common_cases'][long] and digest(ids)==job['role_case_ids_sha256'][long], 'Primary role case hash/count differs')
        require(np.all(arrays['origin_time'][list(indices)]>=_hour(bounds['start_utc']))
                and np.all(arrays['target_time'][list(indices)]<_hour(bounds['end_utc'])), 'Explicit strict role target boundary differs')
        require(refit_label_mask(arrays['target_time'][list(indices)],refit_at_utc=bundle['refit_at_utc'],simulated_label_delay_hours=2).all(),
                'Primary labels not available at simulated refit')
    require(tuple(arrays['case_id'][list(train)].tolist())==bundle['gradient_case_ids'],'Gradient case tuple differs')
    require(all(arrays[field+'_mask'].all() for field in ('history','weather','state','target')),'Common-cohort values must be complete')
    require(all(not x.flags.writeable for x in arrays.values()),'Campaign arrays must be read-only')
    return n


def verify_pairs(rows):
    """Exact shared membership and targets across full/Q/L24, with no magnitude summaries."""
    groups={}
    for row in rows: groups.setdefault((row['fold'],row['gauge_id']),[]).append(row)
    require(len(groups)==12,'All12 fold/gauge pairs required')
    for key,views in groups.items():
        expected={('full',168),('q_only',168)} | ({('full',24)} if key[0]=='final_development' else set())
        require({(r['input_arm'],r['history_hours']) for r in views}==expected and len(views)==len(expected),'Paired view inventory differs')
        for field in ('fit_cohort','fit_case_ids_sha256','fit_target_sha256','fit_history24_sha256',
                      'outer_cohort','outer_case_ids_sha256','outer_label_sha256'):
            require(len({r[field] for r in views})==1,'Paired primary view identity differs: '+field)
        require(len({r['fit_source_identity'] for r in views})==len(views),'Different feature views need distinct training source identities')
    return dict(fold_gauge_pairs=12,view_pairs_matched=True,shared_common_l168_membership=True,matched_original_targets=True)


def qualify(prepared,guard,check):
    import numpy as np
    from hydra_directflow import campaign_data as data,corpus_dataset as ds
    from hydra_directflow.training import TrainOnlyTransform,DevelopmentTrainer
    from hydra_directflow.inference import validate_inputs
    from hydra_directflow.reference import digest
    require(not any(name.split('.')[0] in ('torch','transformers') for name in sys.modules),'No model runtime may be imported')
    rows=[]
    with ExitStack() as stack:
        stack.enter_context(patch.object(TrainOnlyTransform,'fit',side_effect=forbid))
        stack.enter_context(patch.object(DevelopmentTrainer,'__init__',side_effect=forbid))
        stack.enter_context(patch.object(DevelopmentTrainer,'restore',side_effect=forbid))
        stack.enter_context(patch.object(np.linalg,'solve',side_effect=forbid))
        for job in prepared['spec']['jobs']:
            check(); started=time.monotonic()
            common=dict(expected_integrity_hash=prepared['spec']['integrity_receipt_sha256'],max_bytes=MAX_BUNDLE,
                        max_rss_bytes=MAX_RSS,max_read_bytes=MAX_BUNDLE,check=check)
            fit=data.load_fit_bundle(CORPUS,job,**common); arrays=fit['arrays']; count=_membership(fit,job)
            require(stream_array_identity(arrays)==fit['manifest']['returned_arrays_sha256'],'Training raw array identity differs')
            fit_target=stream_array_identity({k:arrays[k] for k in ('case_id','target_time','target','target_mask')})
            fit_history24=stream_array_identity({k:arrays[k][:,-24:] for k in ('history','history_mask','history_time')})
            row=dict(job_id=job['job_id'],fold=job['fold'],gauge_id=job['gauge_id'],input_arm=job['input_arm'],history_hours=job['history_hours'],
                fit_inner_cases=count,fit_cases=len(fit['train_indices']),inner_cases=len(fit['inner_indices']),
                fit_cohort=fit['cohort_admission_digest'],fit_case_ids_sha256=fit['case_ids_sha256'],fit_source_identity=fit['source_identity'],
                fit_arrays_sha256=fit['manifest']['returned_arrays_sha256'],fit_target_sha256=fit_target,fit_history24_sha256=fit_history24,
                fit_array_bytes=sum(x.nbytes for x in arrays.values()),fit_source_shard_reads=fit['manifest']['construction']['source_shards_read'])
            del arrays,fit; gc.collect(); check()
            original=ds._decode
            def no_target(archive,commit,name,shape):
                require(name!='target','Outer input loading attempted numeric verification-label decode')
                return original(archive,commit,name,shape)
            with patch.object(ds,'_decode',side_effect=no_target): outer=data.load_outer_inputs(CORPUS,job,**common)
            arrays=outer['arrays']; ids=tuple(arrays['case_id'].tolist()); origins=arrays['origin_time'].copy(); future=arrays['forecast_time'].copy()
            require('target' not in arrays and 'target_mask' not in arrays and len(ids)==job['actual_common_cases']['outer_scoring']
                    and digest(list(ids))==job['role_case_ids_sha256']['outer_scoring'],'Label-free outer membership differs')
            require(stream_array_identity(arrays)==outer['manifest']['returned_arrays_sha256'],'Outer input raw array identity differs')
            batch=next(data.iter_outer_batches(outer,batch_size=64))
            require(validate_inputs(batch,history_hours=job['history_hours'],feature_schema=batch['feature_schema'].item(),
                expected_cohort_admission_digest=outer['cohort_admission_digest'],expected_gauge_id=job['gauge_id'])==min(64,len(ids)),
                'First bounded predictor batch differs')
            outer_cohort,outer_ids=outer['cohort_admission_digest'],outer['case_ids_sha256']
            row.update(outer_cases=len(ids),outer_cohort=outer_cohort,outer_case_ids_sha256=outer_ids,
                       outer_source_identity=outer['source_identity'],outer_input_arrays_sha256=outer['manifest']['returned_arrays_sha256'],
                       predictor_batch_cases=min(64,len(ids)),outer_target_values_decoded=False)
            del batch,arrays,outer; gc.collect(); check()
            labels=data.load_outer_labels(CORPUS,job,expected_outer_cohort_digest=outer_cohort,expected_case_ids_sha256=outer_ids,**common)
            arrays=labels['arrays']
            require(tuple(arrays['case_id'].tolist())==ids and np.array_equal(arrays['origin_time'],origins)
                    and np.array_equal(arrays['target_time'],future) and arrays['target_mask'].all(),'Separate full outer labels do not align')
            label_sha=stream_array_identity(arrays)
            require(label_sha==labels['manifest']['returned_arrays_sha256'],'Outer label raw identity differs')
            row.update(outer_label_sha256=label_sha,label_cells=arrays['target_mask'].size,
                       elapsed_seconds=time.monotonic()-started)
            rows.append(row)
            del arrays,labels,ids,origins,future; gc.collect(); check()
    pairing=verify_pairs(rows)
    require(len(rows)==27 and not any(name.split('.')[0] in ('torch','transformers') for name in sys.modules),'Exact27 zero-model qualification required')
    require(implementation()==prepared['spec']['implementation'],'Qualification implementation changed during execution')
    return dict(status='PASSED',views=rows,paired_views=pairing,read_telemetry=guard.report(),torch_imports=0,model_calls=0,
        scaler_fits=0,optimizer_updates=0,scientific_scores=0,network_requests=0,protected_assessment_access=False,
        limitations=['All declared outer input/label arrays are loaded and hash/coordinate checked; only the first64-case predictor slice per view is exercised',
                    'Raw source provenance caches and the full corpus inspection are not replayed',
                    'Numeric target members are never decoded during input-only loading; compressed shard reads still contain archived target bytes',
                    'Logical read telemetry reserves each full resident file size at open, not physical disk or network bytes'])


def worker():
    prepared=json.loads((RUN/'plan.json').read_text())
    guard=ReadGuard(CORPUS,maximum=MAX_READ); sys.addaudithook(guard.audit)
    require(prepared==plan(),'Worker plan/source/code changed')
    allowance=claim(prepared)
    import psutil
    process=psutil.Process(); peak=0
    def check():
        nonlocal peak
        peak=max(peak,process.memory_info().rss)
        require(time.time()<allowance['deadline_epoch'] and peak<=MAX_RSS,'Data qualification resource cap exceeded')
    def deadline(*args): raise TimeoutError('Primary data qualification deadline')
    signal.signal(signal.SIGALRM,deadline); signal.setitimer(signal.ITIMER_REAL,max(.001,allowance['deadline_epoch']-time.time()))
    result=qualify(prepared,guard,check); check()
    result.update(plan_sha256=prepared['plan_sha256'],worker_pid=os.getpid(),observed_worker_peak_rss_bytes=peak)
    write_new(RUN/'worker-result.json',result)


def execute(prepared,approved_plan_sha256):
    require(prepared==plan() and prepared['plan_sha256']==approved_plan_sha256,'Exact reviewed qualification plan required')
    allowance=precharge(prepared)
    import psutil
    parent=psutil.Process(); start=time.monotonic(); peak=0; reason=None; child=None
    env={**os.environ,'PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    try:
        with (RUN/'worker.log').open('xb') as log:
            child=subprocess.Popen([sys.executable,'-I','-B',str(Path(__file__).resolve()),'--worker'],cwd=ROOT,env=env,stdout=log,stderr=log)
            process=psutil.Process(child.pid)
            try:
                while child.poll() is None:
                    try: peak=max(peak,parent.memory_info().rss+process.memory_info().rss)
                    except psutil.NoSuchProcess: pass
                    if peak>MAX_RSS or time.time()>allowance['deadline_epoch']:
                        reason='RESOURCE_LIMIT'; child.kill(); break
                    time.sleep(.05)
                child.wait(timeout=10)
            finally:
                if child.poll() is None: child.kill(); child.wait(timeout=10)
                log.flush(); os.fsync(log.fileno())
    except BaseException as error:
        reason=type(error).__name__
        if child is not None and child.poll() is None: child.kill(); child.wait(timeout=10)
    passed=child is not None and child.returncode==0 and reason is None and (RUN/'worker-result.json').is_file()
    from hydra_directflow.corpus_dataset import _read
    result=dict(schema='hydra-primary-campaign-data-qualification/v1',status='PASSED' if passed else 'FAILED_PRESERVED_ATTEMPT',
        plan_sha256=prepared['plan_sha256'],scope=prepared['spec'],created_at_utc=utc(),elapsed_seconds=time.monotonic()-start,
        peak_parent_child_rss_bytes=peak,worker_exit_code=child.returncode if child else None,failure_reason=reason,
        model_calls=0,scaler_fits=0,optimizer_updates=0,network_requests=0,protected_assessment_access=False)
    if passed:
        try:
            raw=_read(RUN/'worker-result.json',4*1024**2); value=json.loads(raw)
            require(value['status']=='PASSED' and value['plan_sha256']==prepared['plan_sha256']
                    and value['worker_pid']==child.pid and len(value['views'])==27,'Worker result identity/scope differs')
            require(value['read_telemetry']['logical_bytes_reserved_at_open']<=MAX_READ,'Worker read allowance exceeded')
            peak=max(peak,value['observed_worker_peak_rss_bytes']+parent.memory_info().rss)
            require(peak<=MAX_RSS and time.time()<=allowance['deadline_epoch'],'Final resource limits exceeded')
            require(implementation()==prepared['spec']['implementation'],'Qualification implementation changed before publication')
            result.update(qualification=value,worker_result_sha256=sha(raw),peak_parent_child_rss_bytes=peak)
        except BaseException as error:
            result.update(status='FAILED_PRESERVED_ATTEMPT',failure_reason=type(error).__name__,peak_parent_child_rss_bytes=peak)
    log=RUN/'worker.log'
    if log.exists(): result['private_worker_log_sha256']=sha(_read(log,4*1024**2)) if log.stat().st_size else sha(b'')
    write_new(RUN/'result.json',result); write_new(REPORT,result); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true'); parser.add_argument('--approved-plan-sha256'); parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker: worker(); return
    prepared=plan()
    if not args.execute:
        print(json.dumps(prepared,indent=2)); return
    result=execute(prepared,args.approved_plan_sha256)
    print(json.dumps({k:result[k] for k in ('status','plan_sha256','elapsed_seconds')})); require(result['status']=='PASSED','Qualification failed; preserve attempt')


if __name__=='__main__': main()
