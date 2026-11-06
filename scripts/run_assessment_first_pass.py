#!/usr/bin/env python3
"""Local saved-weight assessment controller; each phase is a fresh process.

Source staging has its own approved persistent budget. Numerical execution uses
one new immutable plan, and never reuses the closed development allowance.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'))
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
from hydra_directflow.reference import require, digest
from hydra_directflow.local_data import verified_local_bytes
from hydra_directflow.assessment_runtime import AssessmentRuntime, LIMITS
from hydra_directflow.assessment_contract import validate_primary_checkpoint_roster
from hydra_directflow.assessment_data import AssessmentReadAuthority
from hydra_directflow.training import array_identity
from hydra_directflow.staging import atomic_json

PREP = ROOT/'.local/assessment-preparation-v1'
RUN = ROOT/'.local/assessment-v1'
OLD = ROOT/'.local/primary-campaign-v1'
PROTOCOL = ROOT/'configs/assessment-firstpass-v1.json'

def encoded(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def sha(raw): return hashlib.sha256(raw).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def descriptor(path):
    p=Path(path); raw=resident(p,32*1024**2)
    return dict(path=str(p.resolve()), bytes=len(raw), sha256=sha(raw))

def resident(path, maximum, expected=None, read_budget=None):
    p=Path(path);s=p.lstat()
    require(p.is_absolute() and p==p.resolve() and stat.S_ISREG(s.st_mode)
            and not getattr(s,'st_flags',0)&0x40000000 and 0<s.st_size<=maximum,
            'Bounded resident regular source required')
    if read_budget is not None:read_budget(s.st_size)
    raw=p.read_bytes(); after=p.lstat()
    require((s.st_ino,s.st_size,s.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
            and len(raw)==s.st_size,'Source changed while reading')
    if expected is not None:
        require(set(expected)=={'path','bytes','sha256'} and str(p)==expected['path']
                and len(raw)==expected['bytes'] and sha(raw)==expected['sha256'],'Pinned source identity differs')
    return raw

def npz(arrays):
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);return stream.getvalue()

def arrays_from_bytes(raw, maximum=256*1024**2):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        entries=z.infolist();require(len(entries)<=64 and len({e.filename for e in entries})==len(entries)
            and sum(e.file_size for e in entries)<=maximum,'Bounded unique numerical members required')
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    require(all(a.dtype.kind in 'fbiuUS' and a.ndim<=4 for a in arrays.values()),'Plain typed arrays required')
    return arrays

def implementation():
    paths=[p for p in (ROOT/'src/hydra_directflow').glob('*.py') if p.name!='assessment_scoring.py']
    paths += [ROOT/'scripts'/n for n in ('run_assessment_first_pass.py','assessment_prepare_inputs.py')]
    return {str(p.relative_to(ROOT)):sha(resident(p,2*1024**2)) for p in sorted(paths)}

def authenticate(path, expected):
    raw=resident(Path(path),16*1024**2);p=json.loads(raw)
    require(p['plan_sha256']==expected==digest({k:v for k,v in p.items() if k!='plan_sha256'}),'Exact frozen execution plan required')
    require(p['implementation']==implementation(),'Qualified implementation changed')
    sources={k:json.loads(resident(Path(d['path']),32*1024**2,d)) for k,d in p['json_sources'].items()}
    selection=sources['selection'];validate_primary_checkpoint_roster(r['job_id'] for r in selection['records'])
    require(selection['selection_sha256']==digest({k:v for k,v in selection.items() if k!='selection_sha256'}), 'Selection digest differs')
    require(p['checkpoint_sha256_by_job']=={r['job_id']:r['checkpoint']['sha256'] for r in selection['records']},'Selected model roster changed')
    old=json.loads((OLD/'PLAN.json').read_text())
    from hydra_directflow.campaign_ledger import CampaignLedger
    ledger=CampaignLedger(old['ledger_directory']);s=ledger.snapshot()
    require(s['complete'] and s['optimizer_updates_entered']==s['optimizer_updates_confirmed']==1151690,'Completed training counters changed')
    authority=AssessmentReadAuthority(execution_plan_sha256=p['plan_sha256'],protocol_sha256=p['json_sources']['protocol']['sha256'],
        selection_sha256=selection['selection_sha256'],source_inventory_sha256=p['json_sources']['inventory']['sha256'],
        cycle_row_hashes=tuple((r['cycle_id'],r['sealed_row_sha256']) for r in sources['staging_plan']['cycles']),inputs_enabled=True)
    return p,sources,authority

def prepare(weights_path,nwm_root):
    require(not (RUN/'EXECUTION_PLAN.json').exists(),'Existing frozen plan must be reused')
    names={'protocol':PROTOCOL,'selection':PREP/'PRIMARY_SELECTION.json','inventory':PREP/'SOURCE_INVENTORY.json',
        'staging_plan':PREP/'STAGING_PLAN.json','staging_approval':PREP/'STAGING_APPROVAL.json','index_rows':PREP/'INDEX_ROWS.json'}
    sources={k:descriptor(v) for k,v in names.items()}
    selection=json.loads(Path(sources['selection']['path']).read_text());validate_primary_checkpoint_roster(r['job_id'] for r in selection['records'])
    stage=json.loads(Path(sources['staging_plan']['path']).read_text())
    require(stage['plan_sha256']==digest({k:v for k,v in stage.items() if k!='plan_sha256'}),'Staging plan changed')
    from hydra_directflow.local_data import WEIGHTS_SHA256
    wd=descriptor(weights_path);require(wd['sha256']==WEIGHTS_SHA256,'Original frozen weights required')
    phase=OLD/'coverage-extension-v1';mp=phase/'tasks/A0_RECONCILE/attempt-000001/MODEL_MANIFEST.json'
    mr=resident(mp,2*1024**2);require(sha(mr)=='aeb9e9a7ae0d4f26e95a16354790a2b7fdc7f283bcf515981fc248aa2409b2d4','Closed membership manifest differs')
    membership={}
    for key,d in json.loads(mr)['gradient_membership_artifacts'].items():
        dd=dict(d,path=str(phase/d['path']));v=json.loads(resident(Path(dd['path']),2*1024**2,dd))
        require(digest(v['case_ids'])==v['case_ids_sha256']==key,'Gradient membership digest differs');membership[key]=dd
    nwm=Path(nwm_root).resolve()
    nd=descriptor(nwm/'unique-record-view.csv.gz')
    require(nd['sha256']=='ca1dd222eeef12be890034e6bd2d5bc98706806107c6ca2d6ae5502f9f0327f4','Owned unique NWM snapshot differs')
    pd=descriptor(nwm/'PARITY.json')
    require(pd['sha256']=='5dc5cebc3b3ea5e560a504ec12385f79801465e1c67d33c6019eb446a8d654c8','Recorded NWM parity differs')
    p=dict(schema='hydra-assessment-firstpass-execution/v1',created_at_utc=utc(),run_directory=str(RUN),
        json_sources=sources,weights=wd,gradient_memberships=membership,runtime_limits=LIMITS,
        nwm_snapshot=nd,nwm_parity=pd,scoring_implementation_freeze='SEPARATE_BEFORE_VERIFICATION_ROLE_READS',
        checkpoint_sha256_by_job={r['job_id']:r['checkpoint']['sha256'] for r in selection['records']},
        implementation=implementation(),original_training_updates=1151690,
        input_source_identity=digest(dict(source_inventory=sources['inventory']['sha256'],weights=wd['sha256'],
                                        protocol=sources['protocol']['sha256'])),
        verification_gate='ALL36_COMMITTED_PREDICTIONS',new_training=False)
    p['plan_sha256']=digest(p);atomic_json(RUN/'EXECUTION_PLAN.json',p)
    return p

def progress(runtime, status, **extra):
    s=runtime.snapshot();value=dict(status=status,updated_at_utc=utc(),plan_sha256=runtime.plan['plan_sha256'],
        model_products_complete=sum(x['kind']=='model' for x in s['commits']),models_expected=36,
        forward_cases=s['confirmed_cases'],forward_batches=s['confirmed_batches'],runtime=s,
        new_training_updates=0,new_scaler_fits=0,**extra)
    atomic_json(RUN/'progress.json',value);print(json.dumps({k:v for k,v in value.items() if k!='runtime'}),flush=True)

def inputs(runtime,p,sources,authority):
    existing=runtime.reuse('INPUTS')
    if existing:return existing
    from assessment_prepare_inputs import prepare_inputs
    from hydra_directflow.assessment_sources import AssessmentSourceProvider
    from hydra_directflow.staging import SourceBudget
    stage=sources['staging_plan'];ap=stage['sources']['budget_approval'];folder=RUN/'source-staging'
    budget=SourceBudget(folder/'budget',ap['sha256'],body_cap=16*1024**3,get_cap=4000,usd_cap=5)
    try:
        with AssessmentSourceProvider(plan=stage,directory=folder,budget=budget,expected_plan_sha256=stage['plan_sha256']) as provider:
            progress(runtime,'ASSEMBLING_INPUTS',verification_targets_assembled=False)
            paths={x.stem:str(x) for x in (folder/'usgs').glob('*.parquet')}
            result=prepare_inputs(runtime,authority,provider,stage,sources['index_rows']['rows'],paths,
                str(folder/'usgs/source_manifest.json'),p['weights']['path'],p['input_source_identity'])
            result['source_budget']=budget.snapshot()
            d=runtime.write('INPUTS.json',encoded(result),kind='INPUT_INVENTORY')
            c=runtime.commit('INPUTS',dict(inventory=d),metadata=dict(source_identity=p['input_source_identity']))
            progress(runtime,'INPUTS_COMPLETE',verification_targets_assembled=False);return c
    finally:budget.close()

def batch_view(arrays,indices):
    from hydra_directflow.assessment_inference import COORDINATES,METADATA,INPUT_ARRAYS
    result={k:np.array(arrays[k] if k in METADATA else arrays[k][indices],copy=True) for k in COORDINATES+METADATA+INPUT_ARRAYS}
    result['input_admitted']=np.array(arrays['input_admitted'][indices],copy=True)
    return result

def model(runtime,p,sources,authority,job_id):
    if runtime.reuse(job_id):return
    from hydra_directflow.inference import BestCheckpointPredictor
    from hydra_directflow.assessment_inference import predict_assessment_batch,validate_selected_predictor
    from hydra_directflow.coverage_extension_inference import no_learning, configure_runtime
    j=next(x for x in sources['selection']['records'] if x['job_id']==job_id)
    ic=runtime.reuse('INPUTS');require(ic is not None,'Committed inputs required')
    inventory=json.loads(runtime.read(ic['artifacts']['inventory']));bundle=inventory['gauge_bundles'][j['gauge_id']]
    a=arrays_from_bytes(runtime.read(bundle['arrays']));manifest=json.loads(runtime.read(bundle['manifest']))
    require(array_identity(a)==manifest['arrays_sha256'],'Compact input typed identity differs')
    require(len(a['origin_time'])==8760 and np.array_equal(a['forecast_time'],a['origin_time'][:,None]+np.arange(1,19)*3600), 'Complete origin calendar required')
    gd=p['gradient_memberships'][j['gradient_case_ids_sha256']]
    ids=json.loads(resident(Path(gd['path']),2*1024**2,gd,lambda n:runtime.reserve_read(n,numeric=False)))
    require(digest(ids['case_ids'])==j['gradient_case_ids_sha256'],'Original gradient membership differs')
    torch=configure_runtime(j['device'])
    cp=dict(j['checkpoint'],path=str(OLD/j['checkpoint']['path']))
    resident(Path(cp['path']),16*1024**2,cp,lambda n:runtime.reserve_read(n,numeric=False))
    with no_learning(torch):
        runtime.reserve_read(cp['bytes'],purpose='SAVED_CHECKPOINT_LOAD')
        token=runtime.precharge_model(job_id,checkpoint_sha256=cp['sha256'])
        predictor=BestCheckpointPredictor.load(cp['path'],expected_sha256=cp['sha256'],expected_source_identity=j['training_source_identity'],
            expected_feature_schema=j['feature_schema'],expected_training_cohort_admission_digest=j['training_cohort_admission_digest'],
            expected_gauge_id=j['gauge_id'],gradient_fit_case_ids=tuple(ids['case_ids']))
        validate_selected_predictor(predictor,j,authority);runtime.confirm_model(token)
        y=np.full((8760,18),np.nan,dtype=np.float64);positions=np.flatnonzero(a['input_admitted']);receipts=[]
        progress(runtime,'RUNNING_SAVED_WEIGHT_INFERENCE',job_id=job_id,admitted_origins=len(positions),verification_targets_assembled=False)
        for start in range(0,len(positions),64):
            take=positions[start:start+64];view=batch_view(a,take)
            result=predict_assessment_batch(predictor,view,authority,selected_job=j,
                expected_inference_source_identity=a['inference_source_identity'].item(),
                expected_inference_cohort_admission_digest=a['cohort_admission_digest'].item(),
                precharge_forward=runtime.precharge_forward,confirm_forward=runtime.confirm_forward,check=runtime.check)
            require(np.array_equal(result['case_id'],a['case_id'][take]),'Forward identity shifted')
            y[take]=result['prediction_q'];receipts.append(result['receipt'])
            if (start//64+1)%32==0:progress(runtime,'RUNNING_SAVED_WEIGHT_INFERENCE',job_id=job_id,job_cases_done=min(start+64,len(positions)))
    require(np.isfinite(y[a['input_admitted']]).all() and np.isnan(y[~a['input_admitted']]).all(),'Prediction admission changed')
    out={k:a[k] for k in ('case_id','origin_time','forecast_time','site_id','input_admitted')};out['prediction_q']=y
    receipt=dict(job=j,predictions_array_sha256=array_identity(out),input_arrays_sha256=manifest['arrays_sha256'],
        input_artifact=bundle['arrays'],batches=receipts,forecast_origins=int(a['input_admitted'].sum()),candidate_origins=8760,
        target_values_assembled=False,optimizer_updates=0,scaler_fits=0,checkpoint_sha256=cp['sha256'])
    artifacts=dict(predictions=runtime.write('predictions/'+job_id+'.npz',npz(out),kind='PREDICTIONS'),
                   receipt=runtime.write('predictions/'+job_id+'.json',encoded(receipt),kind='PREDICTION_RECEIPT'))
    runtime.commit(job_id,artifacts,kind='model',job_id=job_id,metadata=dict(checkpoint_sha256=cp['sha256'],forecast_origins=receipt['forecast_origins']))
    progress(runtime,'MODEL_COMPLETE',job_id=job_id,verification_targets_assembled=False)

def execute(path,expected,phase,job_id=None):
    p=json.loads(resident(Path(path),16*1024**2))
    require(p['plan_sha256']==expected==digest({k:v for k,v in p.items() if k!='plan_sha256'})
            and p['run_directory']==str(RUN),'Frozen local execution envelope required')
    parent=os.environ.get('HYDRA_ASSESSMENT_SUPERVISOR_PID')
    extra=(int(parent),) if parent is not None else ()
    with AssessmentRuntime(p,additional_pids=extra).writer() as runtime:
        runtime.reserve_read(16*1024**2,numeric=False,purpose='BOUNDED_PHASE_AUTHENTICATION_METADATA')
        p,sources,authority=authenticate(path,expected)
        if phase=='inputs':inputs(runtime,p,sources,authority)
        elif phase=='model':model(runtime,p,sources,authority,job_id)
        elif phase=='barrier':
            barrier=runtime.prediction_barrier()
            progress(runtime,'ALL36_PREDICTIONS_COMMITTED',prediction_barrier=barrier,verification_targets_assembled=False)
        else:raise ValueError('Unknown bounded phase')

def supervised_child(args,p,phase,job):
    """An independent parent enforces the numerical wall clock during blocking calls."""
    import psutil
    worker=subprocess.Popen(args,env=dict(os.environ,PYTHONNOUSERSITE='1',HYDRA_ASSESSMENT_SUPERVISOR_PID=str(os.getpid())))
    runtime=AssessmentRuntime(p);bootstrap=time.monotonic()
    try:
        while True:
            try:
                code=worker.wait(timeout=15)
                require(code==0,'Assessment worker exited unsuccessfully; no automatic retry')
                return
            except subprocess.TimeoutExpired:pass
            reason=None
            if runtime.database.exists():
                state=runtime.snapshot()
                if state['deadline'] is not None and time.time()>state['deadline']:reason='PERSISTENT_NUMERICAL_DEADLINE'
            elif time.monotonic()-bootstrap>120:reason='WORKER_BOOTSTRAP_TIMEOUT'
            parent=psutil.Process();rss=0
            for process in [parent]+parent.children(recursive=True):
                try:rss+=process.memory_info().rss
                except psutil.NoSuchProcess:pass
            if rss>LIMITS['max_combined_rss_mps_bytes']:reason='OBSERVED_PROCESS_RSS_LIMIT'
            if reason:
                atomic_json(RUN/('WATCHDOG_'+str(worker.pid)+'.json'),dict(reason=reason,phase=phase,job_id=job,
                    plan_sha256=p['plan_sha256'],pid=worker.pid,at_utc=utc(),automatic_retry=False))
                worker.terminate()
                try:worker.wait(timeout=10)
                except subprocess.TimeoutExpired:worker.kill();worker.wait(timeout=10)
                raise RuntimeError('Assessment worker stopped by '+reason+'; retained evidence requires diagnosis')
    except BaseException:
        if worker.poll() is None:
            worker.terminate()
            try:worker.wait(timeout=10)
            except subprocess.TimeoutExpired:worker.kill();worker.wait(timeout=10)
        raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--weights');ap.add_argument('--nwm-root')
    ap.add_argument('--plan',default=str(RUN/'EXECUTION_PLAN.json'));ap.add_argument('--plan-sha256')
    ap.add_argument('--phase',choices=('inputs','model','barrier','all'));ap.add_argument('--job-id');a=ap.parse_args()
    if a.prepare:print(prepare(a.weights,a.nwm_root)['plan_sha256']);return
    require(a.plan_sha256 is not None,'Frozen execution digest required')
    if a.phase=='all':
        p,sources,_=authenticate(a.plan,a.plan_sha256)
        steps=[('inputs',None)]+[('model',r['job_id']) for r in sources['selection']['records']]+[('barrier',None)]
        for phase,job in steps:
            args=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'--plan',a.plan,'--plan-sha256',a.plan_sha256,'--phase',phase]
            if job:args+=['--job-id',job]
            supervised_child(args,p,phase,job)
    else:execute(a.plan,a.plan_sha256,a.phase,a.job_id)

if __name__=='__main__':main()
