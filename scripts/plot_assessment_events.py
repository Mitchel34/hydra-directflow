"""Static scientific figures from authenticated, already computed diagnostics.

The root controller owns source authentication and the independent analysis
check. This module has no data-reader CLI, model, fitting or selection entry
point. Call plot_gauge only after the diagnostic artifacts have passed review.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

GAUGES=('03161000','03164000')
SITES={'03161000':'Jefferson','03164000':'Galax'}
ARCHITECTURES=('LSTM','Transformer','GRU_Transformer','Mamba2_Transformer')
NAMES={'LSTM':'LSTM','Transformer':'Transformer','GRU_Transformer':'GRU–Transformer','Mamba2_Transformer':'Mamba-2–Transformer'}
COLORS={'LSTM':'#CA6500','Transformer':'#11748A','GRU_Transformer':'#8860A9','Mamba2_Transformer':'#597B32'}
BLUE='#244E5B';MUTED='#657D86';GRID='#E0E8EB'
HOUR=3600
START=int(datetime(2024,7,31,tzinfo=timezone.utc).timestamp())
END=int(datetime(2025,7,31,tzinfo=timezone.utc).timestamp())
HYDRO_LEADS=(1,6,12,16)


def require(value,message):
    if not value:raise ValueError(message)


def _number(value):
    if value is None:return np.nan
    require(type(value) in (int,float) and math.isfinite(value),'Finite summary value or explicit None required')
    return float(value)


def _json_number(value):
    return float(value) if np.isfinite(value) else None


def _iso(epoch):return datetime.fromtimestamp(int(epoch),timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
def _date(epoch):return datetime.fromtimestamp(int(epoch),timezone.utc).strftime('%d %b %Y %HZ')
def _dates(values):return mdates.date2num(np.asarray(values,dtype='datetime64[s]'))


def validate(summary,arrays,gauge):
    """Validate plotting semantics without choosing events or changing data."""
    require(gauge in GAUGES and summary.get('schema')=='hydra-assessment-event-diagnostics/v1',
            'Declared Jefferson/Galax event diagnostics required')
    require(summary.get('gauge_id',gauge)==gauge and summary['targets_filled'] is False
        and summary['predictions_clipped'] is False and summary['model_calls']==summary['new_training']==0
        and summary['fixed_lead_curves_are_successive_origins'] is True
        and summary['formal_confidence_intervals'] is False,'Saved-prediction diagnostic scope differs')
    required={'reference_time','observed_q','reference_valid','forecast_time','matched_mask','primary_mask','nwm'}
    required|={f'prediction_{kind}__{a}' for kind in ('mean','min','max') for a in ARCHITECTURES}
    require(required<=set(arrays),'Complete declared plot-array fields required')
    t,q,valid=(np.asarray(arrays[k]) for k in ('reference_time','observed_q','reference_valid'))
    calendar=np.arange(START+HOUR,END+18*HOUR,HOUR,dtype=np.int64)
    require(t.dtype.kind in 'iu' and np.array_equal(t,calendar) and q.shape==valid.shape==t.shape
        and q.dtype.kind=='f' and valid.dtype==np.dtype(bool)
        and np.isfinite(q[valid]).all() and (q[valid]>=0).all() and np.isnan(q[~valid]).all(),
        'Full original hourly observed calendar and explicit gaps required')
    f,match,primary=(np.asarray(arrays[k]) for k in ('forecast_time','matched_mask','primary_mask'))
    require(f.dtype.kind in 'iu' and f.shape==match.shape==(8760,18)
        and np.array_equal(f,np.arange(START,END,HOUR)[:,None]+np.arange(1,19)*HOUR)
        and match.dtype==np.dtype(bool) and primary.dtype==np.dtype(bool) and primary.shape==(8760,)
        and not match[:,16:].any() and not (match & ~primary[:,None]).any(),
        'Exact frozen forecast coordinates and matched support required')
    nwm=np.asarray(arrays['nwm'])
    require(nwm.shape==f.shape and nwm.dtype.kind=='f' and np.isfinite(nwm[match]).all()
        and np.isnan(nwm[:,16:]).all(),'Supported NWM fields and absent H17/H18 required')
    for a in ARCHITECTURES:
        lo,mean,hi=(np.asarray(arrays[f'prediction_{k}__{a}']) for k in ('min','mean','max'))
        require(all(v.shape==f.shape and v.dtype.kind=='f' and np.isfinite(v[match]).all() for v in (lo,mean,hi)),
                'Finite saved seed statistics required on every matched case')
        tol=1e-12*np.maximum(1,np.maximum(np.abs(lo[match]),np.abs(hi[match])))
        require(np.all(lo[match]<=mean[match]+tol) and np.all(mean[match]<=hi[match]+tol),
                'Seed minimum, mean and maximum are reversed')
    events=summary['events'];require(0<=len(events)<=3 and len(events)==summary['representative_event_count'],
                                     'Use only the existing top-three event selection')
    selected=[x for x in summary['catalog'] if x['representative_top3']]
    require([x['event'] for x in events]==selected,'Do not replace selected events using forecast results')
    for item in events:
        e=item['event'];take=(t>=e['start_epoch'])&(t<e['end_exclusive_epoch']);at=np.flatnonzero(t==e['peak_epoch'])
        require(len(at)==1 and valid[at[0]] and float(q[at[0]])==e['observed_peak_q_cms']
            and e['start_epoch']<=e['peak_epoch']<e['end_exclusive_epoch']
            and e['reference_requested_hours']==int(take.sum())
            and e['reference_approved_hours']==int((take&valid).sum()),
            'Selected approved observed anchor/window differs')
        require(len(item['coverage_by_lead'])==18 and len(item['nwm_paired'])==16
            and set(item['models'])==set(ARCHITECTURES),'Event reporting roster differs')
        for j in range(16):
            inside=(f[:,j]>=e['start_epoch'])&(f[:,j]<e['end_exclusive_epoch'])
            n=int((inside&match[:,j]).sum())
            require(item['nwm_paired'][j]['lead_hours']==j+1 and item['nwm_paired'][j]['n']==n
                and item['coverage_by_lead'][j]['matched_nwm_target_hours']==n,
                'Plotted event support differs from its scored support')
            require(all(item['models'][a]['seed_mean_and_range'][j]['n']==n
                and item['models'][a]['seed_mean_and_range'][j]['lead_hours']==j+1 for a in ARCHITECTURES),
                    'Model and NWM event metric support differs')
    expected={e['event']['event_id'] for e in events}|{'REST_OF_ASSESSMENT'}
    require(set(summary['decomposition'])==set(ARCHITECTURES),'All four decomposition panels required')
    for a in ARCHITECTURES:
        row=summary['decomposition'][a]
        require(set(row['contributions'])==expected,'Exact disjoint events plus remainder required')
        total=row['seed_mean_and_range']['mean'];values=[x['mean'] for x in row['contributions'].values()]
        if total is not None:
            require(all(v is not None for v in values) and math.isclose(math.fsum(values),total,rel_tol=1e-10,abs_tol=1e-10),
                    'Signed event contributions do not reconcile to the annual mean gap')
    return events


def _configure():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':11,
        'text.color':BLUE,'axes.labelcolor':BLUE,'axes.edgecolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none',
        'svg.hashsalt':'hydra-assessment-events-v1','savefig.facecolor':'white'})


def _save(fig,directory,stem):
    paths={}
    for extension in ('svg','png','pdf'):
        p=directory/(stem+'.'+extension)
        metadata={'Date':None} if extension=='svg' else {'CreationDate':None,'ModDate':None} if extension=='pdf' else None
        fig.savefig(p,dpi=160,bbox_inches='tight',pad_inches=.15,metadata=metadata)
        paths[extension]=str(p)
    plt.close(fig);return paths


def _legend(observed=True):
    handles=([Line2D([0],[0],color='#202020',lw=1.6,label='Approved observed Q')] if observed else [])
    handles += [Line2D([0],[0],color=COLORS[a],lw=1.6,label=NAMES[a]) for a in ARCHITECTURES]
    handles += [Line2D([0],[0],color='#111111',lw=1.6,ls='--',label='NWM')]
    return handles


def _window(summary_row,arrays):
    e=summary_row['event'];t=arrays['reference_time'];take=(t>=e['start_epoch'])&(t<e['end_exclusive_epoch'])
    return t[take],np.where(arrays['reference_valid'][take],arrays['observed_q'][take],np.nan)


def fixed_lead_trace(arrays,times,lead):
    """Reindex saved curves to full hourly x; unsupported cells remain NaN."""
    require(lead in range(1,17),'Only genuinely supported NWM H1–16 traces may be paired')
    t=np.asarray(times);j=lead-1;index=(t-(START+lead*HOUR))//HOUR
    require(np.all(t%HOUR==0),'Exact UTC target hours required')
    in_scope=(index>=0)&(index<8760);matched=np.zeros(len(t),bool)
    matched[in_scope]=arrays['matched_mask'][index[in_scope],j]
    def values(key):
        out=np.full(len(t),np.nan);out[matched]=arrays[key][index[matched],j];return out
    curves={'NWM':dict(mean=values('nwm'),min=None,max=None)}
    for a in ARCHITECTURES:curves[a]={k:values(f'prediction_{k}__{a}') for k in ('mean','min','max')}
    return matched,curves


def hydrographs(summary,arrays,directory,gauge):
    rows=summary['events'];n=max(1,len(rows));fig,axes=plt.subplots(n,4,figsize=(19,3.35*n+1.8),squeeze=False,sharey='row')
    fig.subplots_adjust(left=.065,right=.99,top={1:.68,2:.80,3:.86}[n],bottom=.12,hspace=.53,wspace=.12)
    fig.suptitle(SITES[gauge]+' — largest known approved peaks\nFixed-lead rolling-origin hydrographs',fontsize=19,fontweight='bold',y=.985)
    trace=[]
    if not rows:
        for ax in axes.flat:ax.set_axis_off()
        axes[0,0].text(0,.5,'No approved peaks exceed the frozen development q95.',transform=axes[0,0].transAxes)
    for i,item in enumerate(rows):
        e=item['event'];t,obs=_window(item,arrays);x=_dates(t);finite=[obs[np.isfinite(obs)]]
        descriptor=('Peak' if e['selected_peak_is_window_max'] else 'Ranked anchor')+f" #{e['observed_peak_rank']} · {_date(e['peak_epoch'])} · Q={e['observed_peak_q_cms']:,.1f} m³/s"
        for j,h in enumerate(HYDRO_LEADS):
            ax=axes[i,j];mask,curves=fixed_lead_trace(arrays,t,h)
            ax.plot(x,obs,color='#202020',lw=1.6,zorder=5)
            for a in ARCHITECTURES:
                curve=curves[a];ax.fill_between(x,curve['min'],curve['max'],color=COLORS[a],alpha=.13,linewidth=0)
                ax.plot(x,curve['mean'],color=COLORS[a],lw=1.3)
                finite.extend([curve['min'][np.isfinite(curve['min'])],curve['max'][np.isfinite(curve['max'])]])
            ax.plot(x,curves['NWM']['mean'],color='#111111',lw=1.5,ls='--',zorder=4)
            finite.append(curves['NWM']['mean'][np.isfinite(curves['NWM']['mean'])])
            ax.axvline(_dates([e['peak_epoch']])[0],color=MUTED,ls=':',lw=.8)
            ax.set_title(f"H{h:02d} · matched n={int(mask.sum())}/{len(t)} h",pad=7)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1,tz=timezone.utc));ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%HZ',tz=timezone.utc))
            ax.set_xlim(x[0],x[-1]);ax.grid(axis='y',color=GRID,lw=.7)
            if j==0:ax.set_ylabel('Discharge (m³/s)')
            for k,epoch in enumerate(t):
                trace.append(dict(gauge_id=gauge,event_id=e['event_id'],lead_hours=h,valid_epoch=int(epoch),
                    origin_epoch=int(epoch-h*HOUR),observed_q_cms=_json_number(obs[k]),matched=bool(mask[k]),
                    nwm_q_cms=_json_number(curves['NWM']['mean'][k]),**{
                    a+'_'+kind:_json_number(curves[a][kind][k]) for a in ARCHITECTURES for kind in ('mean','min','max')}))
        values=np.concatenate([v for v in finite if len(v)])
        bottom=min(0,float(values.min()));top=max(1.,float(values.max()));span=max(1.,top-bottom)
        axes[i,0].set_ylim(bottom-.025*span,top+.09*span)
        axes[i,0].text(0,1.27,descriptor,transform=axes[i,0].transAxes,fontweight='bold',fontsize=11,clip_on=False)
        notes=[]
        if not e['selected_peak_is_window_max']:notes.append('Window contains a higher observed hour')
        if e.get('left_truncated') or e.get('right_truncated'):notes.append('Window truncated at assessment boundary')
        if notes:axes[i,-1].text(1,1.27,'; '.join(notes),transform=axes[i,-1].transAxes,ha='right',color='#9B4F00',fontsize=9,clip_on=False)
    fig.legend(handles=_legend(),loc='lower center',bbox_to_anchor=(.5,.059),ncol=6,frameon=False,fontsize=11)
    fig.text(.065,.016,'Each curve links successive issuances C=t−h, not one issuance trajectory. All times UTC; unknown observations and unsupported forecasts remain gaps.\nHYDRA curves = three-seed prediction means; shading = min–max, not confidence intervals. Peaks are known approved observations, not guaranteed maxima during outages.',fontsize=10,color=MUTED)
    return _save(fig,directory,gauge+'_EVENT_HYDROGRAPHS'),trace


def event_metric_series(item,model,metric):
    rows=item['nwm_paired'] if model=='NWM' else item['models'][model]['seed_mean_and_range']
    result={k:[] for k in ('mean','min','max')}
    for row in rows:
        if metric=='anchor_percent':
            value=row['at_observed_peak']['ratio_to_observed_peak']
            convert=lambda x:100*(_number(x)-1)
        else:value=row[metric];convert=_number
        if model=='NWM':
            y=convert(value)
            for k in result:result[k].append(y)
        else:
            for k in result:result[k].append(convert(value[k]))
    return {k:np.asarray(v,dtype=np.float64) for k,v in result.items()}


def lead_errors(summary,directory,gauge):
    rows=summary['events'];n=max(1,len(rows));fig,axes=plt.subplots(n,2,figsize=(13.8,3.25*n+1.75),squeeze=False)
    fig.subplots_adjust(left=.08,right=.985,top={1:.67,2:.79,3:.85}[n],bottom=.17,hspace=.56,wspace=.22)
    fig.suptitle(SITES[gauge]+' — event errors by lead\nSame original matched cases at every model comparison',fontsize=18,fontweight='bold',y=.985)
    trace=[];h=np.arange(1,17)
    if not rows:
        for ax in axes.flat:ax.set_axis_off()
        axes[0,0].text(0,.5,'No selected approved peaks.',transform=axes[0,0].transAxes)
    for i,item in enumerate(rows):
        e=item['event'];counts=[v['n'] for v in item['nwm_paired']]
        annotation=f"Anchor #{e['observed_peak_rank']} · {_date(e['peak_epoch'])} · Q={e['observed_peak_q_cms']:,.1f} m³/s · matched n={min(counts)}–{max(counts)}/lead"
        axes[i,0].text(0,1.16,annotation,transform=axes[i,0].transAxes,fontweight='bold',fontsize=10,clip_on=False)
        for j,(metric,label) in enumerate((('rmse','Event-window RMSE (m³/s)'),('anchor_percent','Error at observed anchor (%)'))):
            ax=axes[i,j];defined=False
            for model in (*ARCHITECTURES,'NWM'):
                values=event_metric_series(item,model,metric);color='#111111' if model=='NWM' else COLORS[model]
                ax.plot(h,values['mean'],color=color,lw=1.6,ls='--' if model=='NWM' else '-',marker='o',ms=2.5)
                if model!='NWM':ax.fill_between(h,values['min'],values['max'],color=color,alpha=.13,linewidth=0)
                defined|=bool(np.isfinite(values['mean']).any())
                for k,lead in enumerate(h):trace.append(dict(gauge_id=gauge,event_id=e['event_id'],lead_hours=int(lead),model=model,
                    metric=metric,mean=_json_number(values['mean'][k]),min=_json_number(values['min'][k]),max=_json_number(values['max'][k]),n=counts[k]))
            ax.set_xlim(1,16);ax.set_xticks([1,4,8,12,16]);ax.set_xlabel('Origin lead (h)');ax.set_ylabel(label);ax.grid(color=GRID,lw=.7)
            if metric=='rmse':ax.set_ylim(bottom=0)
            else:ax.axhline(0,color=MUTED,lw=.8,ls=':')
            if not defined:ax.text(.5,.5,'Unavailable on selected support',transform=ax.transAxes,ha='center',color=MUTED)
    fig.legend(handles=_legend(False),loc='lower center',bbox_to_anchor=(.5,.085),ncol=5,frameon=False,fontsize=11)
    fig.text(.08,.016,'RMSE is computed separately for each seed, then averaged; it is not the RMSE of the displayed mean hydrograph. Bands = three-seed min–max, not CI.\nAnchor error = 100×(forecast−observed anchor Q)/observed anchor Q; absent anchor forecasts remain gaps. NWM H17–18 are unsupported.',fontsize=10,color=MUTED)
    return _save(fig,directory,gauge+'_EVENT_LEAD_ERRORS'),trace


def decomposition(summary,directory,gauge):
    events=summary['events'];keys=[e['event']['event_id'] for e in events]+['REST_OF_ASSESSMENT']
    colors=['#3B8295','#D79B36','#7A6A9D'][:len(events)]+['#B8C4C7']
    labels=[f"Anchor #{e['event']['observed_peak_rank']} · {datetime.fromtimestamp(e['event']['peak_epoch'],timezone.utc):%d %b %Y}" for e in events]+['Rest of assessment']
    fig,ax=plt.subplots(figsize=(12.5,5.4));fig.subplots_adjust(left=.19,right=.98,top=.80,bottom=.29)
    fig.suptitle(SITES[gauge]+' — where the annual matched NSE gap occurs',fontsize=17,fontweight='bold',y=.98)
    trace=[];shown=False
    for i,a in enumerate(ARCHITECTURES):
        row=summary['decomposition'][a];total=row['seed_mean_and_range'];positive=negative=0.
        for key,color,label in zip(keys,colors,labels):
            value=row['contributions'][key]['mean'];trace.append(dict(gauge_id=gauge,architecture=a,bucket=key,contribution_mean=value,
                contribution_min=row['contributions'][key]['min'],contribution_max=row['contributions'][key]['max'],annual_gap_mean=total['mean']))
            if value is None:continue
            left=positive if value>=0 else negative;ax.barh(i,value,left=left,height=.55,color=color,edgecolor='white',lw=.6,label=label if i==0 else None)
            if value>=0:positive+=value
            else:negative+=value
            shown=True
        if total['mean'] is not None:
            require(math.isclose(positive+negative,total['mean'],rel_tol=1e-10,abs_tol=1e-10),'Stacked means do not reconcile')
            ax.plot([total['min'],total['max']],[i,i],color='#111111',lw=1.2)
            ax.plot(total['mean'],i,marker='D',ms=5,color='#111111')
        else:ax.text(0,i,'Undefined annual NSE',va='center',fontsize=10,color=MUTED)
    ax.axvline(0,color=MUTED,lw=.9);ax.set_yticks(range(4),[NAMES[a] for a in ARCHITECTURES]);ax.invert_yaxis()
    ax.set_xlabel('NWM NSE − HYDRA NSE (equal-lead H1–16 mean)');ax.grid(axis='x',color=GRID,lw=.7);ax.set_axisbelow(True)
    if not shown:ax.set_xlim(-1,1)
    handles=[Line2D([0],[0],color=c,lw=8,label=label) for c,label in zip(colors,labels)]
    handles.append(Line2D([0],[0],color='#111111',marker='D',lw=1,label='Total mean / seed range'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.11),ncol=min(3,len(handles)),frameon=False,fontsize=10)
    fig.text(.04,.025,'Signed SSE differences use each lead’s original annual SST denominator; disjoint selected windows + remainder exactly reconstruct the annual gap.\nPositive favors NWM; negative favors HYDRA. This is error accounting, not causal attribution. Diamond/range retain all three seeds; no confidence interval.',fontsize=10,color=MUTED)
    return _save(fig,directory,gauge+'_ANNUAL_NSE_GAP_DECOMPOSITION'),trace


def _write_csv(path,rows):
    if not rows:return None
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    return dict(path=str(path),rows=len(rows),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def plot_gauge(summary,arrays,output_dir,gauge_id):
    """Root must authenticate and independently check artifacts before this call."""
    rows=validate(summary,arrays,gauge_id);_configure();out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    hydro,ht=hydrographs(summary,arrays,out,gauge_id);errors,et=lead_errors(summary,out,gauge_id);gap,gt=decomposition(summary,out,gauge_id)
    traces={name:_write_csv(out/(gauge_id+'_'+name+'.csv'),data) for name,data in
            [('HYDROGRAPH_PLOT_TRACE',ht),('EVENT_METRIC_PLOT_TRACE',et),('NSE_GAP_PLOT_TRACE',gt)]}
    artifacts={name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for name in [Path(v).name for f in (hydro,errors,gap) for v in f.values()]
        for p in [out/name]}
    qa=dict(schema='hydra-assessment-event-figure-qa/v1',status='RENDERED_REQUIRES_VISUAL_INSPECTION',gauge_id=gauge_id,
        event_count=len(rows),hydrograph_leads=list(HYDRO_LEADS),error_plot_leads=list(range(1,17)),
        full_hourly_grid_with_nan_gaps=True,forecasts_are_successive_rolling_origins=True,seed_range_is_confidence_interval=False,
        metrics_average_seed_scores_not_mean_prediction_scores=True,original_event_selection_unchanged=True,
        decomposition_reconciles=True,causal_attribution=False,original_fitting_or_forecasts_performed=False,
        input_authentication_performed_by_plotter=False,input_authentication_required_from_controller=True,
        new_model_forwards=0,new_optimizer_updates=0,new_scaler_fits=0,artifacts=artifacts,traces=traces,
        trace_privacy='Hydrograph trace contains displayed event-hour observed and forecast values; retain privately unless separately reviewed for publication')
    p=out/(gauge_id+'_FIGURE_QA.json');p.write_text(json.dumps(qa,indent=2,allow_nan=False)+'\n')
    return dict(hydrographs=hydro,lead_errors=errors,decomposition=gap,traces=traces,qa=str(p))


def plot_largest_pair(summaries,arrays_by_gauge,output_dir):
    """Display the already-selected rank1 event at both gauges, H1 and H16."""
    require(set(summaries)==set(arrays_by_gauge)==set(GAUGES),'Exactly the two reviewed gauges required')
    for g in GAUGES:
        events=validate(summaries[g],arrays_by_gauge[g],g)
        require(events and events[0]['event']['observed_peak_rank']==1,
                'A previously selected rank1 approved peak is required; no replacement')
    _configure();out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(14.4,9.6),sharey='row')
    fig.subplots_adjust(left=.075,right=.985,top=.80,bottom=.18,hspace=.58,wspace=.13)
    fig.suptitle('Largest known approved peak at each gauge\nFixed-lead rolling-origin curves at H1 and H16',fontsize=18,fontweight='bold',y=.98)
    rows=[]
    for i,g in enumerate(GAUGES):
        item=summaries[g]['events'][0];e=item['event'];arrays=arrays_by_gauge[g]
        t,obs=_window(item,arrays);x=_dates(t);finite=[obs[np.isfinite(obs)]]
        label=f"{SITES[g]} · {_date(e['peak_epoch'])} · observed Q={e['observed_peak_q_cms']:,.1f} m³/s"
        axes[i,0].text(0,1.23,label,transform=axes[i,0].transAxes,fontweight='bold',fontsize=12,clip_on=False)
        for j,h in enumerate((1,16)):
            ax=axes[i,j];match,curves=fixed_lead_trace(arrays,t,h)
            ax.plot(x,obs,color='#202020',lw=2,zorder=5)
            for a in ARCHITECTURES:
                c=curves[a];ax.fill_between(x,c['min'],c['max'],color=COLORS[a],alpha=.13,linewidth=0)
                ax.plot(x,c['mean'],color=COLORS[a],lw=1.6)
                finite.extend([c[k][np.isfinite(c[k])] for k in ('min','max')])
            ax.plot(x,curves['NWM']['mean'],color='#111111',ls='--',lw=1.8,zorder=4)
            finite.append(curves['NWM']['mean'][np.isfinite(curves['NWM']['mean'])])
            ax.axvline(_dates([e['peak_epoch']])[0],color=MUTED,ls=':',lw=.8)
            ax.set_title(f'H{h:02d} · matched n={int(match.sum())}/{len(t)} h',fontsize=12,pad=8)
            ax.set_xlim(x[0],x[-1]);ax.xaxis.set_major_locator(mdates.DayLocator(interval=1,tz=timezone.utc))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%HZ',tz=timezone.utc));ax.grid(axis='y',color=GRID,lw=.7)
            if j==0:ax.set_ylabel('Discharge (m³/s)',fontsize=12)
            for k,epoch in enumerate(t):
                rows.append(dict(gauge_id=g,event_id=e['event_id'],lead_hours=h,valid_epoch=int(epoch),origin_epoch=int(epoch-h*HOUR),
                    observed_q_cms=_json_number(obs[k]),matched=bool(match[k]),nwm_q_cms=_json_number(curves['NWM']['mean'][k]),
                    **{a+'_'+kind:_json_number(curves[a][kind][k]) for a in ARCHITECTURES for kind in ('mean','min','max')}))
        values=np.concatenate([v for v in finite if len(v)]);bottom=min(0,float(values.min()));top=max(1,float(values.max()))
        span=max(1.,top-bottom);axes[i,0].set_ylim(bottom-.025*span,top+.09*span)
    fig.legend(handles=_legend(),loc='lower center',bbox_to_anchor=(.5,.075),ncol=3,frameon=False,fontsize=11)
    fig.text(.075,.015,'UTC; each line links successive origins C=t−h, not a single issuance. Approved observations remain visible even without matched forecasts.\nHYDRA = three-seed prediction mean; bands = min–max, not CI. Unknown hours remain gaps. First-ranked events are fixed, not selected by model error.',fontsize=10,color=MUTED)
    files=_save(fig,out,'LARGEST_EVENT_COMPARISON');trace=_write_csv(out/'LARGEST_EVENT_COMPARISON_TRACE.csv',rows)
    qa=dict(status='RENDERED_REQUIRES_VISUAL_INSPECTION',ranked_event_ids={g:summaries[g]['events'][0]['event']['event_id'] for g in GAUGES},
        original_selection_unchanged=True,leads=[1,16],input_authentication_performed_by_plotter=False,
        input_authentication_required_from_controller=True,new_forecasts=0,new_fits=0,trace=trace,
        files={Path(p).name:dict(bytes=Path(p).stat().st_size,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in files.values()})
    p=out/'LARGEST_EVENT_COMPARISON_QA.json';p.write_text(json.dumps(qa,indent=2,allow_nan=False)+'\n')
    return dict(**files,trace=trace,qa=str(p))
