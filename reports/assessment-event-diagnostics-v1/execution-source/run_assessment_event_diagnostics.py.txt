"""Bounded post-assessment diagnostics of saved forecasts; no execution of models."""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import stat
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from hydra_directflow.reference import digest,require
from hydra_directflow.training import array_identity
from hydra_directflow.assessment_event_diagnostics import unique_reference,select_events,analyze_gauge

SOURCE=ROOT/'.local/assessment-v1'
OUT=ROOT/'.local/assessment-event-diagnostics-v1'
PROTOCOL=ROOT/'configs/assessment-event-diagnostics-v1.json'

def encoded(x):return (json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(x):return hashlib.sha256(x).hexdigest()

class Store:
    def __init__(self):
        self.started=time.monotonic();self.read_bytes=0;self.output_bytes=0;self.sources={}
        self.protocol=json.loads(PROTOCOL.read_text());self.limits=self.protocol['local_diagnostic_limits']
        self.protocol_sha=sha(PROTOCOL.read_bytes());record=json.loads((OUT/'METHOD_BEFORE_VALUES.json').read_text())
        require(record['protocol_sha256']==self.protocol_sha,'Recorded diagnostic method changed')
    def check(self):
        import psutil,shutil
        p=psutil.Process();rss=sum(x.memory_info().rss for x in [p]+p.children(recursive=True))
        require(rss<=self.limits['memory_bytes'] and shutil.disk_usage(OUT).free>=self.limits['minimum_free_disk_bytes'], 'Diagnostic memory/disk bound exceeded')
        require(time.monotonic()-self.started<self.limits['execution_seconds'],'Diagnostic wall bound exceeded')
    def read(self,path,descriptor=None,maximum=16*1024**2):
        self.check();p=Path(path).resolve();s=p.lstat()
        require(str(Path(path).absolute())==str(p) and stat.S_ISREG(s.st_mode) and not getattr(s,'st_flags',0)&0x40000000
            and 0<s.st_size<=maximum,'Exact resident diagnostic source required')
        require(self.read_bytes+s.st_size<=self.limits['logical_read_bytes'],'Diagnostic read cap exceeded')
        self.read_bytes+=s.st_size;raw=p.read_bytes();after=p.lstat()
        require((s.st_ino,s.st_size,s.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'Diagnostic source changed')
        d=dict(path=str(p),bytes=len(raw),sha256=sha(raw))
        if descriptor:require(d==dict(descriptor,path=str(p)),'Diagnostic source hash/size differs')
        self.sources[str(p)]=d;return raw
    def artifact(self,d):return self.read(SOURCE/d['path'],d,maximum=256*1024**2)
    def write(self,name,raw):
        self.check();require(self.output_bytes+len(raw)<=self.limits['output_bytes'],'Diagnostic output cap exceeded')
        p=OUT/name;require(not p.exists(),'Preserve existing diagnostic outputs')
        p.parent.mkdir(exist_ok=True,parents=True)
        with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        self.output_bytes+=len(raw);return dict(path=str(p),bytes=len(raw),sha256=sha(raw))
    def arrays(self,d):
        raw=self.artifact(d)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            entries=z.infolist();require(len(entries)<=64 and len({x.filename for x in entries})==len(entries)
                and sum(x.file_size for x in entries)<=512*1024**2,'Bounded saved arrays required')
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:return {k:z[k] for k in z.files}

def commit(con,key,plan_sha):
    row=con.execute('SELECT body,sha256 FROM commits WHERE key=?',(key,)).fetchone();require(row is not None,'Missing original commitment '+key)
    body=json.loads(row[0]);require(digest(body)==row[1] and body['plan_sha256']==plan_sha,'Original commit changed')
    return body

def run():
    store=Store();protocol=store.protocol
    p=json.loads(store.read(SOURCE/'EXECUTION_PLAN.json'))
    require(p['plan_sha256']==protocol['source_assessment_plan_sha256']==digest({k:v for k,v in p.items() if k!='plan_sha256'}),'Source assessment plan differs')
    # Authenticate original code entries individually; additive diagnostic files
    # are not retroactively part of the frozen first-pass implementation.
    for name,expected in p['implementation'].items():require(sha(store.read(ROOT/name,maximum=2*1024**2))==expected,'Frozen source implementation changed: '+name)
    con=sqlite3.connect('file:'+str(SOURCE/'assessment-runtime.sqlite3')+'?mode=ro',uri=True);con.execute('PRAGMA query_only=ON')
    initial=json.loads(con.execute('select body from state').fetchone()[0]);require(initial['owner'] is None and initial['failure'] is None,'Completed source runtime required')
    audit_commit=commit(con,'INDEPENDENT_POINT_AUDIT',p['plan_sha256']);audit_raw=store.artifact(audit_commit['artifacts']['private_audit'])
    require(sha(audit_raw)==protocol['source_audit_sha256'],'Exact independent source audit required');audit=json.loads(audit_raw);require(audit['status']=='PASS','Source audit did not pass')
    aggregate_raw=store.artifact(audit_commit['artifacts']['public_summary']);require(sha(aggregate_raw)==protocol['source_aggregate_sha256'],'Source aggregate differs')
    aggregate=json.loads(aggregate_raw)
    semantic=commit(con,'SEMANTIC_PREDICTION_BARRIER',p['plan_sha256']);require(semantic['artifacts']['barrier']['sha256']==audit['hashes']['semantic_barrier_sha256'],'Audited semantic barrier differs')
    store.artifact(semantic['artifacts']['barrier'])
    targets=commit(con,'VERIFICATION_TARGETS',p['plan_sha256']);require(targets['metadata']['prediction_barrier_sha256']==audit['hashes']['semantic_barrier_sha256'],'Target release changed')
    nw=commit(con,'NWM_MATCHED_INPUT',p['plan_sha256']);nw_receipt=json.loads(store.artifact(nw['artifacts']['audit']))
    inputs=commit(con,'INPUTS',p['plan_sha256']);inventory=json.loads(store.artifact(inputs['artifacts']['inventory']))
    sd=p['json_sources']['selection'];selected=json.loads(store.read(Path(sd['path']),sd))['records']
    thresh_raw=store.read(ROOT/protocol['threshold_authority']);require(sha(thresh_raw)==protocol['threshold_authority_sha256'],'Frozen gradient threshold report differs')
    thresholds=json.loads(thresh_raw)['fold_regime_and_event_diagnostics']
    outputs={};origin=np.arange(1722384000,1753920000,3600,dtype=np.int64)
    require(len(origin)==8760,'Exact origin calendar')
    for g in protocol['gauges']:
        threshold=next(x['thresholds'] for x in thresholds if x['gauge_id']==g and x['fold']=='final_development')
        truth=store.arrays(targets['artifacts'][g]);inp=store.arrays(inventory['gauge_bundles'][g]['arrays'])
        manifest=json.loads(store.artifact(inventory['gauge_bundles'][g]['manifest']))
        require(array_identity(inp)==manifest['arrays_sha256'] and np.array_equal(inp['origin_time'],origin),'Original input bundle changed')
        require(np.array_equal(truth['target_time'],origin[:,None]+np.arange(1,19)*3600),'Target valid times changed')
        times,q,valid=unique_reference(truth['target'],truth['target_mask'],truth['target_time'])
        catalog=select_events(times,q,valid,threshold['q95_cms'])
        # Selection is committed before any model/NWM prediction values are read.
        catalog_desc=store.write(g+'/CATALOG_BEFORE_ERRORS.json',encoded(dict(gauge_id=g,threshold=threshold,catalog=catalog,
            protocol_sha256=store.protocol_sha,event_selection_uses_model_errors=False)))
        nwm=store.arrays(nw['artifacts'][g]);require(array_identity(nwm)==nw_receipt['arrays_sha256_by_gauge'][g],'Saved NWM identity differs')
        preds={};prediction_sources={}
        for j in selected:
            if j['gauge_id']!=g:continue
            c=commit(con,j['job_id'],p['plan_sha256']);a=store.arrays(c['artifacts']['predictions']);r=json.loads(store.artifact(c['artifacts']['receipt']))
            require(r['job']==j and array_identity(a)==r['predictions_array_sha256'] and r['input_arrays_sha256']==manifest['arrays_sha256'],'Saved prediction identity differs')
            require(all(np.array_equal(a[k],inp[k]) for k in ('origin_time','forecast_time','case_id','site_id','input_admitted')),'Saved prediction alignment differs')
            preds[j['architecture'],j['seed']]=a['prediction_q'];prediction_sources[j['job_id']]=c['artifacts']['predictions']
        result=analyze_gauge(origin,truth['target'],truth['target_mask'],inp['input_admitted'],preds,nwm['prediction_q'],nwm['source_valid'],inp['history'],threshold['q95_cms'],catalog)
        result['summary'].update(gauge_id=g,threshold=threshold,protocol_sha256=store.protocol_sha,source_audit_sha256=protocol['source_audit_sha256'],
            sources=dict(targets=targets['artifacts'][g],nwm=nw['artifacts'][g],inputs=inventory['gauge_bundles'][g]['arrays'],predictions=prediction_sources),
            original_audited_annual=aggregate['gauges'][g])
        flat={}
        for k,v in result['plot'].items():
            if isinstance(v,dict):
                for arch,arr in v.items():flat[k.replace('_by_arch','')+'__'+arch]=arr
            else:flat[k]=v
        stream=io.BytesIO();np.savez_compressed(stream,**flat)
        outputs[g]=dict(catalog=catalog_desc,summary=store.write(g+'/SUMMARY.json',encoded(result['summary'])),plot=store.write(g+'/PLOT.npz',stream.getvalue()))
        print(json.dumps({'gauge':g,'selected_events':len(result['summary']['events']),'source_reads_bytes':store.read_bytes}),flush=True)
        del truth,inp,preds,nwm,result,flat
    final=json.loads(con.execute('select body from state').fetchone()[0]);require(final==initial,'Original completed runtime changed during diagnosis');con.close()
    body=dict(schema='hydra-assessment-event-diagnostic-run/v1',status='COMPUTED_PENDING_INDEPENDENT_CHECK',protocol_sha256=store.protocol_sha,
        outputs=outputs,sources=store.sources,logical_read_bytes=store.read_bytes,output_bytes=store.output_bytes,elapsed_seconds=time.monotonic()-store.started,
        original_runtime_unchanged=True,new_forwards=0,new_training_updates=0,new_scaler_fits=0,new_downloads=0)
    d=store.write('RESULTS.json',encoded(body));print(json.dumps(d))

if __name__=='__main__':run()
