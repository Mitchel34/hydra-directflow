"""Export independently audited aggregates, without raw cases or private paths."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

GAUGES=('03161000','03164000','03479000')
MODELS=('LSTM','Transformer','GRU_Transformer','Mamba2_Transformer')
SITES=('Jefferson','Galax','Sugar Grove')

def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()

def export(source,expected,directory):
    raw=Path(source).read_bytes()
    assert sha(raw)==expected and len(raw)<8*1024**2
    data=json.loads(raw);assert data['status']=='INDEPENDENT_POINT_METRICS_AUDIT_PASSED'
    assert set(data['gauges'])==set(GAUGES) and data['models']==36
    # Human-written first-pass interpretation is checked against the actual values.
    for g in GAUGES[:2]:
        v=data['gauges'][g]
        assert all(v['nwm']['matched']['equal_lead_nse']>v['models'][a]['seed_aggregation']['nwm_matched']['equal_lead_nse']['mean'] for a in MODELS)
    v=data['gauges'][GAUGES[2]]
    assert {a for a in MODELS if v['models'][a]['seed_aggregation']['nwm_matched']['equal_lead_nse']['mean']>v['nwm']['matched']['equal_lead_nse']}=={'LSTM','Mamba2_Transformer'}
    out=Path(directory);out.mkdir(parents=True,exist_ok=True);files={};csv_rows=[]
    summary={k:v for k,v in data.items() if k!='gauges'};summary['source_aggregate_sha256']=expected;summary['gauges']={}
    def write(name,body):
        assert len(body)<1000000 and b'/Users/' not in body and b'gs://' not in body
        p=out/name
        if p.exists():assert p.read_bytes()==body,'Existing export differs'
        else:p.write_bytes(body)
        files[name]={'bytes':len(body),'sha256':sha(body)}
    for g in GAUGES:
        gauge=data['gauges'][g];assert set(gauge['models'])==set(MODELS)
        item={'coverage':gauge['coverage'],'nwm':gauge['nwm'],'models':{}}
        for a in MODELS:
            model=gauge['models'][a];assert set(model['seeds'])=={'17','29','43'}
            name=g+'__'+a+'.json'
            write(name,encoded({'schema':'hydra-assessment-public-model/v1','gauge_id':g,'architecture':a,
                'source_aggregate_sha256':expected,'model':model}))
            item['models'][a]={'artifact':dict(path=name,**files[name]),'seed_aggregation':{
                scope:{k:panel[k] for k in ('seeds','equal_lead_nse','equal_lead_nse_leads','support')}
                for scope,panel in model['seed_aggregation'].items()}}
            for seed,panels in model['seeds'].items():
                for scope,panel in panels.items():
                    for row in panel['leads']:
                        csv_rows.append(dict(gauge_id=g,model=a,seed=seed,scope=scope,lead_hours=row['lead_hours'],
                            n=row['n'],**{k:row[k] for k in ('nse','rmse','mae','bias','physical_mse','negative_frequency','pbias','pearson_r')}))
        for row in gauge['nwm']['matched']['leads']:
            csv_rows.append(dict(gauge_id=g,model='NWM',seed='',scope='nwm_matched',lead_hours=row['lead_hours'],
                n=row['n'],**{k:row[k] for k in ('nse','rmse','mae','bias','physical_mse','negative_frequency','pbias','pearson_r')}))
        summary['gauges'][g]=item
    # Keeping aggregates once in summary and all seeds in separate model files
    # avoids a multi-megabyte public blob while preserving every audit value.
    write('summary.json',encoded(summary))
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    write('lead-metrics.csv',stream.getvalue().encode())
    rows=[]
    for g,site in zip(GAUGES,SITES):
        d=data['gauges'][g]
        vals=[d['nwm']['matched']['equal_lead_nse']]+[d['models'][a]['seed_aggregation']['nwm_matched']['equal_lead_nse']['mean'] for a in MODELS]
        rows.append('| '+site+' | '+' | '.join('undefined' if x is None else f'{x:.3f}' for x in vals)+' |')
    report='''# Assessment first pass: saved FULL L168 weights

All 36 saved models completed and the independent numerical/provenance audit passed.
Origins: 2024-07-31 through 2025-07-30 UTC; the final H18 target is 2025-07-31 17:00 UTC.
No training, transform refit, clipping, repair or seed selection was performed.

Matched H1-16 equal-lead NSE (HYDRA: arithmetic mean of all 3 seed metrics):

| Gauge | NWM | LSTM | Transformer | GRU-Transformer | Mamba2-Transformer |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

NWM has higher matched mean NSE than all four models at Jefferson and Galax.
At Sugar Grove, LSTM and Mamba2-Transformer have higher seed means than NWM;
the seed ranges are substantial. These are descriptive point comparisons,
not significance claims or model-selection decisions. Sugar Grove's NWM lead
profile is nonmonotonic; a higher H1-16 mean does not establish superiority
at every lead. Event-level diagnosis remains pending.

| Gauge | Forecasts available / 8,760 | Complete-H18 target origins | NWM-matched origins / lead |
|---|---:|---:|---:|
'''
    for g,site in zip(GAUGES,SITES):
        d=data['gauges'][g];c=d['coverage'];n=[x['n'] for x in d['nwm']['matched']['leads'][:16]]
        report+=f"| {site} | {c['input_admitted_origins']:,} ({100*c['forecast_availability_fraction']:.2f}%) | {c['primary_complete_target_origins']:,} | {min(n):,}"+(f"-{max(n):,}" if min(n)!=max(n) else '')+' |\n'
    report+='''
NWM is an external comparator. Exact C-2 issues support origin H1-16; H17-18
are explicitly unavailable. Independent HYDRA H1-18 primary and per-lead
valid-target secondary scores remain separate from the matched H1-16 comparison.

This is the strict assessment baseline. Prior development-pool or coverage-router
availability does not describe these assessment forecasts. No assessment Q-only,
L24/router or peak-event analysis is claimed here. The poster retains the known
development peak-underprediction limitation with its development-period label.

All three seeds 17/29/43 are retained. Min/max ranges are descriptive, not confidence
intervals. Forecast windows overlap and are not independent floods. The observation
delay is simulated using revised approved A/A+R data; it is not a reconstruction
of historical USGS publication times. Basin footprints remain approximate.

`summary.json` holds coverage, NWM and all seed aggregates. Each gauge/model JSON
retains every seed and support panel; `lead-metrics.csv` provides physical-unit
metrics. Empty CSV metrics mean undefined or structurally unsupported, never zero.
PBIAS uses 100*sum(prediction-observation)/sum(observation).
'''
    write('README.md',report.encode())
    # Exact reconstruction verifies that splitting did not omit or alter a seed.
    for g in GAUGES:
        for a in MODELS:
            found=json.loads((out/(g+'__'+a+'.json')).read_text())['model']
            assert found==data['gauges'][g]['models'][a]
    inventory=dict(schema='hydra-assessment-public-export/v1',status='EXACT_AGGREGATE_EXPORT_VERIFIED',
        source_aggregate_sha256=expected,raw_arrays_or_weights=False,files=files,metric_csv_rows=len(csv_rows),
        all36_seed_products_preserved=True)
    body=encoded(inventory);(out/'ARTIFACTS.json').write_bytes(body)
    return inventory

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=export(a.source,a.sha256,a.output);print(json.dumps({'status':result['status'],'files':len(result['files']),'metric_rows':result['metric_csv_rows']}))
