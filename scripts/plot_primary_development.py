#!/usr/bin/env python3
"""Static development-review figures from authenticated completed artifacts.

Uses the existing local matplotlib runtime; never imports models or raw sources.
All forecasts remain raw. Event curves show descriptive seed means/ranges only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parents[1]
MODELS=('LSTM','Transformer','GRU_Transformer','Mamba2_Transformer')
NAMES=('LSTM','Transformer','GRU–Transformer','Mamba-2–Transformer')
GAUGES=('03161000','03164000','03479000')
SITES=('Jefferson','Galax','Sugar Grove')
COLORS=('#bf5700','#0d6074','#815ca4','#58752d')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'svg.hashsalt':'hydra-directflow-development-v1'})


def read(path,maximum=256*1024**2,expected=None):
    path=Path(path);stat=path.lstat()
    assert path==path.resolve() and 0<stat.st_size<=maximum and not getattr(stat,'st_flags',0)&0x40000000
    raw=path.read_bytes();assert len(raw)==stat.st_size
    if expected is not None: assert hashlib.sha256(raw).hexdigest()==expected
    return raw


def descriptor(run,item):
    assert set(item)=={'path','bytes','sha256'} and not Path(item['path']).is_absolute()
    path=run/item['path'];assert path.is_relative_to(run) and path==path.resolve()
    raw=read(path,expected=item['sha256']);assert len(raw)==item['bytes'];return raw


def npz(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        assert sum(x.file_size for x in z.infolist())<=512*1024**2
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:return {k:z[k] for k in z.files}


def save(fig,directory,name,caption):
    paths=[]
    for suffix in ('svg','png'):
        path=directory/(name+'.'+suffix);assert not path.exists()
        metadata={'Date':None,'Creator':'HYDRA DirectFlow'} if suffix=='svg' else {'Software':'HYDRA DirectFlow'}
        fig.savefig(path,dpi=180,bbox_inches='tight',metadata=metadata)
        raw=path.read_bytes();paths.append(dict(path=path.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    plt.close(fig);return dict(name=name,caption=caption,files=paths)


def lead_figures(summary,directory):
    rows={(r['gauge_id'],r['architecture'],r['input_arm'],r['history_hours']):r for r in summary['final_seed_aggregates']}
    output=[];hours=np.arange(1,19)
    for difference in (False,True):
        fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharex=True,sharey=True)
        for site,(gauge,ax) in enumerate(zip(GAUGES,axes)):
            for i,model in enumerate(MODELS):
                full=np.array([x['nse'] for x in rows[gauge,model,'full',168]['score']['per_lead_mean_metrics']])
                q=np.array([x['nse'] for x in rows[gauge,model,'q_only',168]['score']['per_lead_mean_metrics']])
                ax.plot(hours,full-q if difference else full,color=COLORS[i],label=NAMES[i],lw=1.9)
                if not difference: ax.plot(hours,q,color=COLORS[i],ls='--',lw=1.25,alpha=.8)
            n=next(j['panel']['leads'][0]['n'] for j in summary['jobs'] if j['fold']=='final_development' and j['gauge_id']==gauge)
            ax.set_title(f'{SITES[site]} · {gauge}\nn = {n:,} admitted origins per lead',fontsize=11)
            ax.set_xlabel('Lead (hours)');ax.set_xticks([1,3,6,9,12,15,18]);ax.grid(alpha=.15)
            if difference:ax.axhline(0,color='#777777',lw=.8)
        axes[0].set_ylabel('NSE difference: full − Q-only' if difference else 'NSE (raw physical discharge)')
        title='Joint weather + soil contribution by lead' if difference else 'Full inputs and discharge history by lead'
        fig.suptitle(title+' · development validation',fontsize=15,y=1.02)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,-.04))
        note='Mean of three seed NSEs; common L168 support. '+('Zero marks no difference.' if difference else 'Solid: full inputs; dashed: Q-only.')
        fig.text(.5,-.075,note+'  No uncertainty band shown.',ha='center',fontsize=9)
        fig.tight_layout();output.append(save(fig,directory,'development-'+('joint-input-delta-by-lead' if difference else 'nse-by-lead'),note))
    return output


def event_figures(run,audit,events,catalog,directory):
    groups={(x['fold'],x['gauge_id']):x for x in audit['groups']}
    event_groups={(x['fold'],x['gauge_id']):x for x in events['groups']}
    rows={(x['fold'],x['gauge_id']):x for x in catalog['fold_regime_and_event_diagnostics']}
    data={}
    for gauge in GAUGES:
        group=groups['final_development',gauge];eg=event_groups['final_development',gauge]
        arrays=npz(descriptor(run,group['artifacts']['arrays']));manifest=json.loads(descriptor(run,group['artifacts']['manifest']))
        ref=npz(descriptor(run,eg['artifacts']['reference']))
        data[gauge]=(arrays,manifest,ref)
    output=[]
    for lead in (1,18):
        fig,axes=plt.subplots(3,3,figsize=(16,11))
        for gi,gauge in enumerate(GAUGES):
            arrays,manifest,ref=data[gauge]
            fixed=[e for e in rows['final_development',gauge]['observed_high_flow_events'] if e['representative_top3']]
            assert len(fixed)<=3
            for ei,ax in enumerate(axes[gi]):
                if ei>=len(fixed):ax.text(.5,.5,'No further catalogued event',ha='center',transform=ax.transAxes);continue
                event=fixed[ei];lo,hi=event['start_epoch'],event['end_epoch']
                selected=(ref['times']>=lo)&(ref['times']<hi)
                obs=np.where(ref['admitted'][selected],ref['values'][selected],np.nan)
                times=[datetime.fromtimestamp(int(t),timezone.utc) for t in ref['times'][selected]]
                ax.plot(times,obs,color='#111111',lw=1.7,label='Approved observed flow',zorder=5)
                valid_times=arrays['forecast_time'][:,lead-1];sel=(valid_times>=lo)&(valid_times<hi)
                x=[datetime.fromtimestamp(int(t),timezone.utc) for t in valid_times[sel]]
                for mi,model in enumerate(MODELS):
                    indices=[i for i,p in enumerate(manifest['panels']) if p['architecture']==model and p['input_arm']=='full' and p['history_hours']==168]
                    assert len(indices)==3
                    q=arrays['predictions'][indices][:,sel,lead-1]
                    complete=np.isfinite(q).all(axis=0);center=np.full(q.shape[1],np.nan);lower=center.copy();upper=center.copy()
                    center[complete]=q[:,complete].mean(axis=0);lower[complete]=q[:,complete].min(axis=0);upper[complete]=q[:,complete].max(axis=0)
                    ax.plot(x,center,color=COLORS[mi],lw=1.3,label=NAMES[mi]);ax.fill_between(x,lower,upper,color=COLORS[mi],alpha=.10)
                support=event['forecast_case_coverage_by_lead'][lead-1]
                peak='covered' if support['observed_peak_has_common_prediction_case'] else 'UNAVAILABLE'
                ax.set_title(f"{SITES[gi]} · catalog event {ei+1}\n{event['event_id'][:10]} · peak forecast {peak}",fontsize=10)
                ax.text(.02,.96,f"{support['common_comparison_cells']}/{support['requested_cells']} forecast target hours",transform=ax.transAxes,va='top',fontsize=8)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d',tz=timezone.utc));ax.grid(alpha=.15)
                if ei==0:ax.set_ylabel('Discharge (m³/s)')
                if gi==2:ax.set_xlabel('Valid time (UTC)')
        fig.suptitle(f'Fixed top-three development events · H{lead} forecasts',fontsize=16,y=1.01)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='lower center',bbox_to_anchor=(.5,-.025))
        note='Full L168; lines = seed-mean predictions, shading = seed min–max (not confidence intervals). Metrics remain means of individual-seed scores.'
        fig.text(.5,-.048,note,ha='center',fontsize=9);fig.text(.5,-.068,'Events selected from observed peaks before model errors. Missing forecasts and rejected observations remain gaps; no replacement events.',ha='center',fontsize=9)
        fig.tight_layout();output.append(save(fig,directory,f'development-fixed-events-H{lead:02d}',note))
    return output


def contrast_figures(run,index,directory):
    families={}
    for block in index['blocks']:
        families[block['block_hours']]=json.loads(descriptor(run,block['families']['final']['report']))
    assert set(families)=={168,336,672}
    output=[];short=dict(zip(MODELS,('LSTM','Transformer','GRU–T','Mamba2–T')))
    for kind,title in [('full_minus_q','Joint weather + soil effect'),('L168_minus_L24','History length effect: L168 − L24'),
                       ('architecture_first_minus_second','Full-input L168 architecture contrasts')]:
        fig,axes=plt.subplots(1,3,figsize=(15,5.4),sharex=True,sharey=True)
        for gi,(gauge,ax) in enumerate(zip(GAUGES,axes)):
            base=[r for r in families[336]['contrasts'] if r['kind']==kind and r['gauge_id']==gauge]
            labels=[short[r['positive_panel'][1]]+(' − '+short[r['negative_panel'][1]] if kind=='architecture_first_minus_second' else '') for r in base]
            for bi,(hours,color,offset) in enumerate(((168,'#8a8a8a',-.20),(336,'#bf5700',0),(672,'#0d6074',.20))):
                rows=[r for r in families[hours]['contrasts'] if r['kind']==kind and r['gauge_id']==gauge]
                assert len(rows)==len(base)
                for i,(row,anchor) in enumerate(zip(rows,base)):
                    assert row['positive_panel']==anchor['positive_panel'] and row['negative_panel']==anchor['negative_panel']
                    y=i+offset;point=row['point_difference'];assert point is not None
                    if row['lower'] is not None:
                        ax.plot([row['lower'],row['upper']],[y,y],color=color,lw=2 if hours==336 else 1.2)
                    ax.scatter([point],[y],s=24 if hours==336 else 16,color=color,marker='o' if row['lower'] is not None else 'x',
                               label=f'{hours//24}-day blocks'+(' (primary)' if hours==336 else '') if i==0 else None,zorder=3)
            ax.set_yticks(np.arange(len(base)),labels);ax.axvline(0,color='#555555',lw=.8);ax.grid(axis='x',alpha=.15)
            ax.set_title(SITES[gi]+' · '+gauge);ax.set_xlabel('Difference in mean equal-lead NSE')
            counts=', '.join(f"{h//24}d: {families[h]['jointly_valid_draws']}/2000" for h in (168,336,672))
            ax.text(.5,-.18,counts,ha='center',transform=ax.transAxes,fontsize=8)
        axes[0].invert_yaxis();fig.suptitle(title+' · development validation',fontsize=15,y=1.02)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.06))
        note='95% conditional, pointwise percentile intervals; paired calendar blocks, same jointly valid pool. Not simultaneous or interannual coverage.'
        fig.text(.5,-.095,note,ha='center',fontsize=9);fig.tight_layout()
        output.append(save(fig,directory,'development-'+kind+'-uncertainty',note))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--summary',type=Path,required=True);parser.add_argument('--summary-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run=args.run.resolve();directory=args.output.absolute();assert directory==directory.resolve() and directory.is_relative_to(run)
    assert not directory.exists();directory.mkdir(parents=True)
    summary=json.loads(read(args.summary,16*1024**2,args.summary_sha256));assert summary['verified_cumulative_jobs']==180 and not summary['protected_assessment_access']
    audit=json.loads(read(run/'AUDIT.json'));events=json.loads(read(run/'EVENTS.json'));uncertainty=json.loads(read(run/'UNCERTAINTY.json'))
    assert audit['plan_sha256']==events['plan_sha256']==uncertainty['plan_sha256'] and audit['groups_total']==12
    assert summary['plan_sha256']==uncertainty['original_plan_sha256']
    assert summary['protocol_sha256']==uncertainty['protocol_sha256']
    assert events['catalog_sha256']=='8e82f613165a5d452c74a30783b02e1c0ae5fb5d93dbe2e338d0010c215e8c0a'
    catalog_raw=read(ROOT/'reports/development-coverage-regimes-v2.json',16*1024**2,'8e82f613165a5d452c74a30783b02e1c0ae5fb5d93dbe2e338d0010c215e8c0a')
    figures=lead_figures(summary,directory)+contrast_figures(run,uncertainty,directory)+event_figures(run,audit,events,json.loads(catalog_raw),directory)
    manifest=dict(schema='hydra-primary-development-figures/v1',status='RENDERED_REQUIRES_VISUAL_REVIEW',figures=figures,
        source_summary_sha256=args.summary_sha256,postprocessing_plan_sha256=audit['plan_sha256'],
        matplotlib_version=matplotlib.__version__,numpy_version=np.__version__,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protected_assessment_access=False,model_calls=0,seed_shading_is_uncertainty_interval=False)
    (directory/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(status=manifest['status'],figures=len(figures))))


if __name__=='__main__':main()
