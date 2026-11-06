"""Export audited event aggregates and figures; raw arrays and weights stay local."""
from pathlib import Path
import csv
import hashlib
import io
import json
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'.local/assessment-event-diagnostics-v1'
OUTPUT=ROOT/'reports/assessment-event-diagnostics-v1'
GAUGES=('03161000','03164000')
SITES={'03161000':'Jefferson','03164000':'Galax'}
MODELS=('LSTM','Transformer','GRU_Transformer','Mamba2_Transformer')

def encoded(x):return (json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def read(path,expected):
    raw=Path(path).read_bytes();assert sha(raw)==expected;return json.loads(raw)

def publication_svg(raw):
    cleaned=b'\n'.join(line.rstrip(b' \t') for line in raw.split(b'\n'))
    def signature(node):
        attrs={k:(' '.join(v.split()) if k in ('d','points') else v) for k,v in node.attrib.items()}
        text=node.text if node.text and node.text.strip() else None
        tail=node.tail if node.tail and node.tail.strip() else None
        return node.tag,attrs,text,tail,[signature(c) for c in node]
    assert signature(ET.fromstring(raw))==signature(ET.fromstring(cleaned)), 'SVG publication formatting changed semantics'
    return cleaned

def export(figure_directory):
    checks=read(SOURCE/'CHECKS.json','9eb49b2f189dd86dea7ec81affd2e78482843d02fc60fbb2bda2b17ddb68286f');assert checks['status']=='PASS'
    bounds=read(SOURCE/'HEAD_BOUNDS.json',checks['headbounds_sha256'])
    assert all(checks[k]==0 for k in ('model_constructions','model_forwards','optimizer_updates','scaler_fits','downloads'))
    OUTPUT.mkdir(exist_ok=True,parents=True);inventory={}
    def write(name,raw):
        assert len(raw)<1000000 and b'/Users/' not in raw and b'gs://' not in raw
        p=OUTPUT/name;p.parent.mkdir(exist_ok=True,parents=True)
        if p.exists():assert p.read_bytes()==raw,'Existing export changed: '+name
        else:p.write_bytes(raw)
        inventory[name]=dict(bytes=len(raw),sha256=sha(raw));return dict(path=name,**inventory[name])
    compact=dict(schema='hydra-assessment-event-public-report/v1',status='INDEPENDENTLY_VERIFIED_DESCRIPTIVE_DIAGNOSTIC',
        protocol_sha256=checks['protocol_sha256'],source_assessment_audit_sha256=checks['source_assessment_audit_sha256'],
        event_audit_sha256='9eb49b2f189dd86dea7ec81affd2e78482843d02fc60fbb2bda2b17ddb68286f',
        headbounds_sha256=checks['headbounds_sha256'],checks=checks['checks'],gauges={},
        new_training=0,new_forward_calls=0,new_downloads=0,assessment_used_for_tuning=False,
        interpretation='Post-assessment error accounting and fixed-weight capacity diagnosis; no causal architecture trial or confidence intervals')
    metrics=[]
    for g in GAUGES:
        s=read(SOURCE/g/'SUMMARY.json',checks['summary_sha256_by_gauge'][g]);events=[]
        for e in s['events']:
            rank=e['event']['observed_peak_rank'];name=g+'__event_'+str(rank)+'.json'
            d=write(name,encoded(dict(gauge_id=g,source_summary_sha256=checks['summary_sha256_by_gauge'][g],event=e)))
            events.append(dict(event=e['event'],artifact=d))
            for a,v in e['models'].items():
                for seed,panels in v['seeds'].items():
                    for scope,rows in [('PAIRED_NWM_H1_H16',panels['paired']),('HYDRA_ONLY_H18',[panels['hydra_h18']])]:
                        for r in rows:
                            metrics.append(dict(gauge_id=g,event_rank=rank,peak_epoch=e['event']['peak_epoch'],model=a,seed=seed,scope=scope,
                                lead_hours=r['lead_hours'],**{k:r[k] for k in ('n','sse','physical_mse','rmse','mae','bias')},
                                peak_prediction_q_cms=r['at_observed_peak']['prediction_q_cms'],peak_error_cms=r['at_observed_peak']['error_cms'],
                                peak_ratio=r['at_observed_peak']['ratio_to_observed_peak']))
            for r in e['nwm_paired']:
                metrics.append(dict(gauge_id=g,event_rank=rank,peak_epoch=e['event']['peak_epoch'],model='NWM',seed='',scope='PAIRED_NWM_H1_H16',
                    lead_hours=r['lead_hours'],**{k:r[k] for k in ('n','sse','physical_mse','rmse','mae','bias')},
                    peak_prediction_q_cms=r['at_observed_peak']['prediction_q_cms'],peak_error_cms=r['at_observed_peak']['error_cms'],peak_ratio=r['at_observed_peak']['ratio_to_observed_peak']))
        bb=[b for b in bounds['records'] if b['gauge_id']==g];assert len(bb)==12
        retained=[{k:b[k] for k in ('job_id','gauge_id','architecture','seed','checkpoint_sha256','physical_lower','physical_upper','comparison_allowance_cms','model_parameter_count')} for b in bb]
        first=s['events'][0];h1=first['coverage_by_lead'][0]
        compact['gauges'][g]=dict(q95_cms=s['q95_cms'],threshold=s['threshold'],catalog=s['catalog'],selected_events=events,
            decomposition=s['decomposition'],fixed_saved_model_envelopes=retained,
            largest_peak_h1=dict(observed_q_cms=first['event']['observed_peak_q_cms'],
                latest_admitted_input_q_cms=h1['latest_observation_context']['q_cms'],
                latest_input_context=h1['latest_observation_context'],nwm_q_cms=first['nwm_paired'][0]['at_observed_peak']['prediction_q_cms'],
                hydra_family_seed_means={a:first['models'][a]['seed_mean_and_range'][0]['at_observed_peak']['prediction_q_cms']['mean'] for a in MODELS}),
            annual_published_matched_nse=dict(nwm=s['original_audited_annual']['nwm']['matched']['equal_lead_nse'],
                models={a:s['original_audited_annual']['models'][a]['seed_aggregation']['nwm_matched']['equal_lead_nse'] for a in MODELS}))
    write('summary.json',encoded(compact))
    stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(metrics[0]),lineterminator='\n');w.writeheader();w.writerows(metrics)
    write('event-lead-metrics.csv',stream.getvalue().encode())
    figs=Path(figure_directory);qa=read(figs/'FINAL_FIGURE_ARTIFACTS.json','ce0566297f8cf56fa636d05b85f8870a155e44452b486f87f16a0ec0a9f80cc3')
    assert qa['status']=='PASS' and qa['source_checks_sha256']==compact['event_audit_sha256'] and qa['figures']==7
    for p in sorted(figs.glob('*.svg')):
        raw=p.read_bytes();assert dict(bytes=len(raw),sha256=sha(raw))==qa['files'][p.name]
        write('figures/'+p.name,publication_svg(raw))
    write('FIGURE_SOURCE.json',encoded(dict(source_figure_manifest_sha256=sha((figs/'FINAL_FIGURE_ARTIFACTS.json').read_bytes()),
        event_audit_sha256=compact['event_audit_sha256'],hydrograph_trace_csvs_published=False,raw_year_arrays_published=False,raw_weights_published=False,
        svg_publication_formatting='Trailing ASCII whitespace removed; XML text, attributes and path-token semantics verified unchanged; original hashes in source figure manifest')))
    manifest=dict(schema='hydra-event-report-export/v1',status='AUDITED_AGGREGATE_EXPORT',files=inventory,
        metric_rows=len(metrics),all24_saved_models_and_all3_seeds_retained=True)
    write('ARTIFACTS.json',encoded(manifest))
    return compact

if __name__=='__main__':
    result=export(sys.argv[1]);print(json.dumps({'gauges':list(result['gauges']),'status':result['status']}))
