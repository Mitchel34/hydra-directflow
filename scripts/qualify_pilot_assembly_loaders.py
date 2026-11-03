#!/usr/bin/env python3
"""Plan-first real assembly loader checks: zero Torch, transforms, fits or scores.

The assembly must already be complete and externally SHA-pinned. This controller
reads only its immutable derived artifacts; it neither reopens raw source caches
nor replays full-corpus inspection. Public output contains hashes/counts only.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from assemble_gap_pilot import (AUTHORITIES, SOURCE_PLAN, SNAPSHOT, authority, canonical, private_path,
                               public_report_path, require, sha, sync_directory, utc, write_new)

MAX_SECONDS=900
MAX_RSS=4*1024**3
MAX_BUNDLE=1024**3
MAX_READ=1024**3
ALLOWANCES=Path.home()/'.codex/hydra-directflow-runtime/pilot-loader-qualification-allowances'
CODE=('scripts/qualify_pilot_assembly_loaders.py','scripts/assemble_gap_pilot.py',
      'src/hydra_directflow/pilot_assembly.py','src/hydra_directflow/pilot_provenance.py',
      'src/hydra_directflow/pilot_training.py','src/hydra_directflow/training.py',
      'src/hydra_directflow/trainer_bundle.py','src/hydra_directflow/inference.py',
      'src/hydra_directflow/gap_filling.py','src/hydra_directflow/quality.py',
      'src/hydra_directflow/reference.py','src/hydra_directflow/contract.py')


def implementation():
    from hydra_directflow.corpus_dataset import _read
    from hydra_directflow.pilot_assembly import code_identity
    return dict(controller={name:sha(_read(ROOT/name,1024**2)) for name in CODE},assembly=code_identity())


def prepare(directory,expected_manifest_sha256):
    """Metadata only: no prepared NPZ values/provenance lookup bodies opened."""
    from hydra_directflow.pilot_assembly import _manifest
    directory=private_path(directory)
    _,manifest=_manifest(directory,expected_manifest_sha256)
    _,bounds,refit,expected=authority(Path.home()/'.codex/hydra-directflow-runtime/pilot-cv1-usgs-cache-v1')
    require(manifest['source_plan_sha256']==SOURCE_PLAN and manifest['snapshot_sha256']==SNAPSHOT
            and manifest['roles']==expected and manifest['role_time_bounds']==bounds
            and manifest['refit_at_utc']==refit,'Exact published CV1 two-hour roles/source required')
    require(manifest['execution']==dict(scaler_fits=0,ridge_fits=0,optimizer_updates=0,network_requests=0,protected_access=False)
            and manifest['scientific_fit_authorization'] is False,'Data-only completed assembly required')
    require(set(manifest['coverage_outer'])==set(expected)
            and all(x['cases']==8742 and x['cohort_mode']=='all_outer_candidates' for x in manifest['coverage_outer'].values()),
            'Exact CV1 outer candidate calendar required')
    scope=dict(training_gauges=list(expected),training_roles=['fit','inner_stop'],outer_sampling='FIRST_RETURNED_BATCH_PER_GAUGE_PER_COHORT_MAX64',
        sampled_outer_cohorts=['outer','coverage_outer'],ridge_series_per_gauge=1,
        maximum_bundle_estimated_bytes=MAX_BUNDLE,maximum_parent_child_rss_bytes=MAX_RSS,
        maximum_seconds=MAX_SECONDS,maximum_assembly_logical_read_bytes=MAX_READ,
        torch_imports=0,model_calls=0,scaler_fits=0,ridge_fits=0,optimizer_updates=0,scientific_scores=0,network_requests=0)
    result=dict(schema='hydra-pilot-loader-qualification-plan/v1',status='PREPARED_NOT_EXECUTED',
        assembly_directory=str(directory),assembly_manifest_sha256=expected_manifest_sha256,
        assembly_manifest_content_sha256=manifest['content_sha256'],assembly_manifest_bytes=(directory/'MANIFEST.json').stat().st_size,
        source_plan_sha256=SOURCE_PLAN,snapshot_sha256=SNAPSHOT,roles=expected,role_time_bounds=bounds,
        refit_at_utc=refit,coverage_outer=manifest['coverage_outer'],implementation=implementation(),authorities=AUTHORITIES,scope=scope)
    result['plan_sha256']=sha(canonical(result)); return result


def validate_plan(plan,expected):
    require(plan['plan_sha256']==expected==sha(canonical({k:v for k,v in plan.items() if k!='plan_sha256'})),
            'Exact loader qualification plan required')
    require(plan==prepare(plan['assembly_directory'],plan['assembly_manifest_sha256']), 'Qualification source/code/scope changed')


class ReadGuard:
    """Audit-hook accounting: conservatively reserve full logical bytes per open.

    This is not physical disk I/O or a wire-byte measurement. Source NPZ/JSON
    files may be warm in the OS cache. Repeated opens are charged repeatedly.
    """
    def __init__(self,directory,maximum=MAX_READ):
        self.directory=Path(directory).absolute(); self.maximum=maximum
        self.read_bytes=0; self.opens=0; self.labels_forbidden=False; self.kinds={}
    def audit(self,event,args):
        if event in ('socket.connect','socket.connect_ex','socket.getaddrinfo','socket.bind','socket.sendto'):
            raise RuntimeError('Loader qualification is offline')
        if event=='import' and args and str(args[0]).split('.')[0] in ('torch','transformers'):
            raise RuntimeError('No model runtime imports in loader qualification')
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)): return
        path=Path(os.fsdecode(args[0])).absolute()
        if not path.is_relative_to(self.directory): return
        mode=args[1] if len(args)>1 else None; flags=args[2] if len(args)>2 else 0
        require(not (isinstance(mode,str) and any(x in mode for x in ('w','a','+','x')))
                and not (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)),
                'Prepared assembly is immutable')
        require(not (self.labels_forbidden and path.name=='labels.npz'), 'Outer input loader attempted label access')
        info=path.lstat()
        require(stat.S_ISREG(info.st_mode) and not getattr(info,'st_flags',0)&0x40000000 and 0<info.st_size<=16*1024**2,
                'Only bounded resident assembly files may be read')
        require(self.read_bytes+info.st_size<=self.maximum,'Cumulative assembly read allowance exceeded before open')
        self.read_bytes+=info.st_size; self.opens+=1; self.kinds[path.name]=self.kinds.get(path.name,0)+1
    def report(self):
        return dict(logical_bytes_reserved_at_open=self.read_bytes,file_opens=self.opens,file_opens_by_basename=dict(self.kinds),
                    maximum_logical_bytes= self.maximum,physical_disk_or_network_bytes_measured=False)


def stream_array_identity(arrays):
    """Same array_identity bytes without a full-tensor temporary byte string."""
    import numpy as np
    result=hashlib.sha256()
    for name in sorted(arrays):
        value=np.asarray(arrays[name]); require(value.dtype.kind in 'fbiuUS','Non-object typed arrays required')
        result.update(json.dumps([name,value.dtype.str,value.shape],separators=(',',':')).encode())
        flat=value.reshape(-1); step=max(1,1024**2//value.dtype.itemsize)
        for start in range(0,len(flat),step): result.update(flat[start:start+step].tobytes(order='C'))
    return result.hexdigest()


def forbid(*unused,**kwargs):
    raise RuntimeError('Fitting, scaler construction, model construction and optimizer work are forbidden')


def validate_role_membership(bundle,expected):
    from hydra_directflow.reference import digest
    from hydra_directflow.pilot_training import validate_pilot_roles
    train,inner,receipt=validate_pilot_roles(bundle['arrays'],train_indices=bundle['train_indices'],
        inner_indices=bundle['inner_indices'],role_time_bounds=bundle['role_time_bounds'],refit_at_utc=bundle['refit_at_utc'])
    n=len(bundle['arrays']['case_id'])
    require(tuple(sorted((*train,*inner)))==tuple(range(n)), 'Fit+inner bundle includes unassigned or outer rows')
    for name,indices in (('fit',train),('inner_stop',inner)):
        ids=bundle['arrays']['case_id'][list(indices)].tolist()
        require(dict(cases=len(ids),case_ids_sha256=digest(ids))==expected[name], 'Loaded role membership differs from frozen count authority')
    return receipt


def _verify_provenance(arrays,provenance):
    from hydra_directflow.pilot_provenance import _rows,_native_digest
    import numpy as np
    mask=np.asarray(arrays['history_mask']); times=np.asarray(arrays['history_time'])
    require(np.array_equal(mask[...,0],mask[...,1]),'Q and coordinate-age masks differ')
    count=0
    for _ in _rows(provenance,mask[...,0],times): count+=1
    require(count==len(times),'Provenance row count differs')
    return _native_digest(provenance)


def _input_check(batch,gauge):
    from hydra_directflow.inference import validate_inputs
    from hydra_directflow.training import FULL_FEATURE_SCHEMA
    arrays=batch['arrays']
    require('target' not in arrays and 'target_mask' not in arrays,'Input-only loader returned labels')
    n=validate_inputs(arrays,history_hours=168,feature_schema=FULL_FEATURE_SCHEMA,
        expected_cohort_admission_digest=batch['cohort_admission_digest'],expected_gauge_id=gauge)
    require(0<n<=64 and arrays['inference_source_identity'].item()==batch['source_identity'],'Bounded input source identity differs')
    return dict(cases=n,case_ids_sha256=sha(canonical(arrays['case_id'].tolist())),raw_arrays_sha256=stream_array_identity(arrays),
        original_provenance_sha256=_verify_provenance(arrays,batch['history_provenance']),
        source_identity=batch['source_identity'],cohort_admission_digest=batch['cohort_admission_digest'])


def qualify(plan,guard,check):
    """Called only inside the claimed fresh worker; sequential gauge lifetimes."""
    import numpy as np
    from hydra_directflow import pilot_assembly as assembly
    from hydra_directflow.training import validate_development_bundle,TrainOnlyTransform,FULL_FEATURE_SCHEMA
    from hydra_directflow.pilot_training import PilotTrainer
    from hydra_directflow import gap_filling
    from hydra_directflow.reference import digest,ASSESSMENT_START
    from hydra_directflow.trainer_bundle import _hour
    directory=plan['assembly_directory']; pinned=plan['assembly_manifest_sha256']; training=[]; inference=[]; ridge=[]
    require(not any(x.split('.')[0] in ('torch','transformers') for x in sys.modules),'Model runtime was imported before qualification')
    with ExitStack() as stack:
        stack.enter_context(patch.object(TrainOnlyTransform,'fit',side_effect=forbid))
        stack.enter_context(patch.object(PilotTrainer,'__init__',side_effect=forbid))
        stack.enter_context(patch.object(PilotTrainer,'update',side_effect=forbid))
        stack.enter_context(patch.object(gap_filling,'fit_ridge_fallback',side_effect=forbid))
        stack.enter_context(patch.object(np.linalg,'solve',side_effect=forbid))
        for gauge in plan['scope']['training_gauges']:
            check(); start=time.monotonic()
            bundle=assembly.load_training_bundle(directory,expected_manifest_sha256=pinned,gauge_id=gauge,max_bytes=MAX_BUNDLE)
            arrays=bundle['arrays']; n=validate_development_bundle(arrays,168,feature_schema=FULL_FEATURE_SCHEMA)
            require(all(arrays[name+'_mask'].all() for name in ('history','weather','state','target')), 'Training common cohort lost complete support')
            receipt=validate_role_membership(bundle,plan['roles'][gauge]); check()
            numeric_bytes=sum(value.nbytes for value in arrays.values())
            require(numeric_bytes<=MAX_BUNDLE,'Actual loaded array bytes exceed bundle cap')
            training.append(dict(gauge_id=gauge,cases=n,fit_cases=len(bundle['train_indices']),inner_cases=len(bundle['inner_indices']),outer_cases=0,
                arrays_bytes=numeric_bytes,raw_arrays_sha256=stream_array_identity(arrays),
                original_provenance_sha256=_verify_provenance(arrays,bundle['history_provenance']),
                source_identity=bundle['source_identity'],cohort_admission_digest=bundle['cohort_admission_digest'],
                role_validation=receipt,elapsed_seconds=time.monotonic()-start))
            del arrays,bundle; gc.collect(); check()
            for role,loader,label_loader in (('outer',assembly.iter_outer_inputs,assembly.load_outer_labels),
                    ('coverage_outer',assembly.iter_coverage_inputs,assembly.load_coverage_labels)):
                guard.labels_forbidden=True
                try:
                    iterator=loader(directory,expected_manifest_sha256=pinned,gauge_id=gauge,batch_size=64)
                    batch=next(iterator); iterator.close(); row=_input_check(batch,gauge)
                finally: guard.labels_forbidden=False
                check(); ids=batch['arrays']['case_id'].tolist()
                labels=label_loader(directory,expected_manifest_sha256=pinned,gauge_id=gauge,case_ids=ids)
                require(np.array_equal(labels['case_id'],batch['arrays']['case_id']) and np.array_equal(labels['site_id'],batch['arrays']['site_id'])
                        and np.array_equal(labels['origin_time'],batch['arrays']['origin_time'])
                        and np.array_equal(labels['target_time'],batch['arrays']['forecast_time']), 'Explicit label coordinates differ from input-only cases')
                require(labels['target'].shape==labels['target_mask'].shape==(len(ids),18) and labels['target_mask'].dtype==np.dtype(bool)
                        and np.isfinite(labels['target'][labels['target_mask']]).all()
                        and np.isnan(labels['target'][~labels['target_mask']]).all(), 'Explicit label masks/finite values differ')
                if role=='outer': require(labels['target_mask'].all(),'Strict outer labels unexpectedly incomplete')
                inference.append(dict(gauge_id=gauge,cohort=role,**row,label_arrays_sha256=stream_array_identity(labels),
                    supported_label_cells=int(labels['target_mask'].sum()),requested_label_cells=labels['target_mask'].size,
                    input_phase_label_opens=0,label_values_scored=False))
                del batch,labels,iterator; gc.collect(); check()
            value=assembly.load_ridge_series(directory,expected_manifest_sha256=pinned,gauge_id=gauge)
            arrays,metadata=value['arrays'],value['metadata']; times=arrays['observation_times']; mask=arrays['approved_mask']
            start,end=(_hour(plan['role_time_bounds']['fit'][k]) for k in ('start_utc','end_utc')); refit=_hour(plan['refit_at_utc'])
            require(np.array_equal(times,np.arange(start,end,3600,dtype=np.int64)) and end<=refit<=ASSESSMENT_START
                    and metadata['gradient_start_epoch']==start and metadata['gradient_end_epoch']==end and metadata['refit_epoch']==refit
                    and metadata['gauge_id']==gauge and metadata['ridge_fits']==0,'Ridge store exceeds exact gradient scope')
            valid_times,valid_masks,_=gap_filling._metadata(times,mask,metadata['source_observation_times'],
                tuple(metadata['source_ids']),tuple(tuple(flags) for flags in metadata['source_flags']))
            expected=np.array([i for i in range(7,len(times)) if mask[i-7:i+1].all() and int(times[i])+7200<=refit],np.int64)
            require(np.array_equal(expected,arrays['eligible_pair_stop_indices']) and len(expected)==metadata['eligible_pairs'], 'Ridge pair census differs')
            ridge.append(dict(gauge_id=gauge,unique_gradient_hours=len(times),eligible_pairs=len(expected),
                arrays_bytes=sum(x.nbytes for x in arrays.values()),raw_arrays_sha256=stream_array_identity(arrays),
                original_metadata_sha256=digest(metadata),source_identity=value['source_identity'],ridge_fits=0))
            del arrays,metadata,times,mask,value,valid_times,valid_masks,expected; gc.collect(); check()
    require(not any(x.split('.')[0] in ('torch','transformers') for x in sys.modules),'Model runtime imported during qualification')
    return dict(training=training,outer_batch_checks=inference,ridge=ridge,read_telemetry=guard.report(),
        torch_imports=0,model_calls=0,scaler_fits=0,ridge_fits=0,optimizer_updates=0,scientific_scores=0,network_requests=0,
        protected_assessment_access=False,qualification_scope='FULL_TRAIN_INNER_BUNDLES_AND_FIRST_OUTER_BATCHES_ONLY',
        limitations=['Outer and coverage loaders are sampled at their first returned batch of at most64 cases per gauge, not exhausted',
                    'All prepared source data were already authenticated by assembly; no raw-source or full-corpus audit is repeated',
                    'Read telemetry counts logical file-size reservations at each open, not physical disk or network bytes',
                    'No scientific fitting, transforms, gap filling, parameter selection, scoring or protocol adoption occurs'])


def claim_worker(plan,allowance,control):
    global_allowance=json.loads((ALLOWANCES/(plan['plan_sha256']+'.json')).read_text())
    require(global_allowance==allowance and allowance['control_directory']==str(control)
            and time.time()<allowance['deadline_epoch'],'Exact live precharged worker/control required')
    claim=dict(plan_sha256=plan['plan_sha256'],pid=os.getpid(),at_utc=utc())
    write_new(ALLOWANCES/(plan['plan_sha256']+'.claimed.json'),claim)
    write_new(control/'claimed.json',claim)


def worker(plan_path,control):
    # Read the private control only, then enforce the import/read guard before
    # any hydra module or assembly manifest is opened in this fresh process.
    plan_path,control=private_path(plan_path),private_path(control)
    plan=json.loads(plan_path.read_text()); allowance=json.loads((control/'precharged.json').read_text())
    guard=ReadGuard(plan['assembly_directory']); sys.addaudithook(guard.audit)
    require(time.time()<allowance['deadline_epoch'],'Qualification allowance expired')
    validate_plan(plan,allowance['plan_sha256'])
    claim_worker(plan,allowance,control)
    import psutil
    process=psutil.Process(); peak=0; started=time.monotonic()
    def check():
        nonlocal peak
        peak=max(peak,process.memory_info().rss)
        require(time.time()<allowance['deadline_epoch'] and peak<=MAX_RSS,'Loader qualification resource limit reached')
    def deadline(*args): raise TimeoutError('Loader qualification deadline')
    signal.signal(signal.SIGALRM,deadline); signal.setitimer(signal.ITIMER_REAL,max(.001,allowance['deadline_epoch']-time.time()))
    result=qualify(plan,guard,check); check()
    result.update(status='PASSED',plan_sha256=plan['plan_sha256'],assembly_manifest_sha256=plan['assembly_manifest_sha256'],
                  elapsed_seconds=time.monotonic()-started,observed_worker_peak_rss_bytes=peak)
    write_new(control/'worker-result.json',result)


def execute_controller(plan_path,*,approved_plan_sha256,control,report):
    from hydra_directflow.corpus_dataset import _read
    plan_path=private_path(plan_path); plan=json.loads(_read(plan_path,8*1024**2)); validate_plan(plan,approved_plan_sha256)
    control,report=private_path(control),public_report_path(report)
    require(not control.exists() and not report.exists(),'Fresh qualification control/report required; preserve prior attempts')
    require(control.parent.is_dir(),'Existing stable control parent required')
    ALLOWANCES.mkdir(exist_ok=True); sync_directory(ALLOWANCES.parent)
    deadline=time.time()+MAX_SECONDS
    precharge=dict(plan_sha256=approved_plan_sha256,assembly_manifest_sha256=plan['assembly_manifest_sha256'],
        created_at_utc=utc(),deadline_epoch=deadline,control_directory=str(control),maximum_read_bytes=MAX_READ,maximum_seconds=MAX_SECONDS,
        maximum_parent_child_rss_bytes=MAX_RSS,maximum_bundle_estimated_bytes=MAX_BUNDLE,
        model_calls=0,scaler_fits=0,ridge_fits=0,optimizer_updates=0)
    write_new(ALLOWANCES/(approved_plan_sha256+'.json'),precharge)
    control.mkdir(); sync_directory(control.parent); write_new(control/'precharged.json',precharge)
    import psutil
    parent=psutil.Process(); start=time.monotonic(); peak=0; reason=None; child=None
    env={**os.environ,'PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    try:
        with (control/'worker.log').open('xb') as log:
            child=subprocess.Popen([sys.executable,'-I','-B',str(Path(__file__).resolve()),'--worker','--plan',str(plan_path),'--control',str(control)],
                cwd=ROOT,env=env,stdout=log,stderr=log)
            process=psutil.Process(child.pid)
            try:
                while child.poll() is None:
                    try: peak=max(peak,parent.memory_info().rss+process.memory_info().rss)
                    except psutil.NoSuchProcess: pass
                    if peak>MAX_RSS or time.time()>deadline:
                        reason='RESOURCE_LIMIT'; child.kill(); break
                    time.sleep(.05)
                child.wait(timeout=10)
            finally:
                if child.poll() is None: child.kill(); child.wait(timeout=10)
                log.flush(); os.fsync(log.fileno())
    except BaseException as error:
        reason=type(error).__name__
        if child is not None and child.poll() is None: child.kill(); child.wait(timeout=10)
    passed=child is not None and child.returncode==0 and reason is None and (control/'worker-result.json').is_file()
    result=dict(schema='hydra-pilot-assembly-loader-qualification/v1',status='PASSED' if passed else 'FAILED_PRESERVED_ATTEMPT',
        plan_sha256=approved_plan_sha256,assembly_manifest_sha256=plan['assembly_manifest_sha256'],
        assembly_manifest_content_sha256=plan['assembly_manifest_content_sha256'],source_plan_sha256=SOURCE_PLAN,snapshot_sha256=SNAPSHOT,
        implementation=plan['implementation'],authorities=plan['authorities'],scope=plan['scope'],
        elapsed_seconds=time.monotonic()-start,peak_parent_child_rss_bytes=peak,worker_exit_code=child.returncode if child else None,
        failure_reason=reason,created_at_utc=utc(),scientific_execution_authorized=False)
    if passed:
        raw=_read(control/'worker-result.json',4*1024**2); value=json.loads(raw)
        require(value['plan_sha256']==approved_plan_sha256 and value['assembly_manifest_sha256']==plan['assembly_manifest_sha256'], 'Worker qualification identity differs')
        result.update(worker_result_sha256=sha(raw),qualification=value)
    log_path=control/'worker.log'
    if log_path.exists(): result['private_worker_log_sha256']=sha(_read(log_path,4*1024**2)) if log_path.stat().st_size else sha(b'')
    write_new(control/'result.json',result); write_new(report,result); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('assembly','plan','control','report'): parser.add_argument('--'+name,type=Path)
    parser.add_argument('--manifest-sha256'); parser.add_argument('--approved-plan-sha256')
    parser.add_argument('--execute',action='store_true'); parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:
        require(args.plan and args.control,'Explicit worker plan/control required'); worker(args.plan,args.control); return
    if args.execute:
        require(args.plan and args.control and args.report and args.approved_plan_sha256,'Explicit reviewed plan/control/report required')
        result=execute_controller(args.plan,approved_plan_sha256=args.approved_plan_sha256,control=args.control,report=args.report)
        print(json.dumps({k:result[k] for k in ('status','plan_sha256','elapsed_seconds')})); require(result['status']=='PASSED','Qualification failed; preserve attempt'); return
    require(args.assembly and args.manifest_sha256 and args.plan,'Plan-only mode requires explicit assembly/manifest hash/plan path')
    value=prepare(args.assembly,args.manifest_sha256); write_new(private_path(args.plan),value)
    print(json.dumps(dict(status='PLAN_ONLY_NO_NUMERIC_READS',plan_sha256=value['plan_sha256'])))


if __name__=='__main__': main()
