"""Post-assessment diagnostics from fixed predictions; no I/O or model calls.

Event selection depends only on approved observed hours and a previously fixed
gradient-calendar q95. None of these summaries selects a model or changes the
assessment protocol. Fixed-lead hydrographs contain successive rolling origins,
not a single forecast trajectory. All performance summaries retain every seed.
"""
from collections.abc import Mapping
import math

import numpy as np

from .contract import ARCHITECTURES
from .reference import ASSESSMENT_START, ASSESSMENT_END, HOUR, require

SEEDS=(17,29,43)
LEADS=tuple(range(1,19))
PAIRED_LEADS=tuple(range(1,17))
REFERENCE_START=ASSESSMENT_START+HOUR
REFERENCE_END=ASSESSMENT_END+18*HOUR  # exclusive; retains final H18 at17Z


def unique_reference(target,target_mask,target_time):
    """Return full hourly calendar with original valid values and explicit gaps."""
    q,valid,t=map(np.asarray,(target,target_mask,target_time))
    require(q.shape==valid.shape==t.shape and q.ndim==2 and q.shape[1]==18
        and q.dtype.kind=='f' and valid.dtype==np.dtype(bool) and t.dtype.kind in 'iu',
        'Aligned typed H18 target values, quality masks and clocks required')
    require(t.size>0 and np.all(t%HOUR==0) and np.isfinite(q[valid]).all() and (q[valid]>=0).all(),
        'Exact hourly times and approved nonnegative observations required')
    stamps,first,inverse=np.unique(t.ravel(),return_index=True,return_inverse=True)
    mask=valid.ravel();values=q.ravel()
    require(np.array_equal(mask,mask[first][inverse]),'Repeated target-hour quality masks differ')
    require(np.array_equal(values[mask],values[first][inverse][mask]),'Repeated approved target-hour values differ')
    calendar=np.arange(int(stamps[0]),int(stamps[-1])+HOUR,HOUR,dtype=np.int64)
    out=np.full(len(calendar),np.nan);present=np.zeros(len(calendar),dtype=bool)
    locations=(stamps-calendar[0])//HOUR
    present[locations]=mask[first];out[locations[present[locations]]]=values[first][mask[first]]
    return calendar,out,present


def select_events(times,q,valid,q95,*,min_separation_hours=168,window_hours=(-48,72)):
    """Complete observed-only ranked catalog; top3 marked without forecast access."""
    t,y,mask=map(np.asarray,(times,q,valid))
    require(t.ndim==1 and t.size>0 and y.shape==mask.shape==t.shape and t.dtype.kind in 'iu'
        and y.dtype.kind=='f' and mask.dtype==np.dtype(bool) and np.all(t%HOUR==0)
        and np.array_equal(t,np.arange(int(t[0]),int(t[-1])+HOUR,HOUR)), 'Complete exact-hour reference calendar required')
    require(np.isfinite(y[mask]).all() and (y[mask]>=0).all() and np.isnan(y[~mask]).all(),
        'Keep rejected or unknown references NaN, never imputed')
    require(type(q95) in (int,float) and math.isfinite(q95) and q95>=0
        and min_separation_hours==168 and tuple(window_hours)==(-48,72),
        'Frozen gradient q95,168h separation and[-48,+72) window required')
    candidates=np.flatnonzero(mask & (y>q95));order=sorted(candidates,key=lambda i:(-float(y[i]),int(t[i])))
    chosen=[];result=[];left_bound=int(t[0]);right_bound=int(t[-1])+HOUR
    for i in order:
        peak=int(t[i])
        if any(abs(peak-old)<168*HOUR for old in chosen):continue
        chosen.append(peak);left=max(left_bound,peak-48*HOUR);right=min(right_bound,peak+72*HOUR)
        window=(t>=left)&(t<right);known=window & mask;indices=np.flatnonzero(known)
        maximum=max(indices,key=lambda j:(float(y[j]),-int(t[j])))
        rank=len(result)+1
        result.append(dict(event_id=str(peak),observed_peak_rank=rank,representative_top3=rank<=3,
            peak_epoch=peak,observed_peak_q_cms=float(y[i]),start_epoch=left,end_exclusive_epoch=right,
            left_truncated=left!=peak-48*HOUR,right_truncated=right!=peak+72*HOUR,
            reference_requested_hours=int(window.sum()),reference_approved_hours=int(known.sum()),
            reference_unusable_hours=int((window & ~mask).sum()),
            window_observed_max_epoch=int(t[maximum]),window_observed_max_q_cms=float(y[maximum]),
            selected_peak_is_window_max=peak==int(t[maximum]),
            selection_uses_model_errors=False,selection_requires_forecast_admission=False,
            event_episodes_are_proven_independent=False))
    return result


def _score(y,p):
    n=len(y)
    if n==0:return dict(status='UNAVAILABLE_NO_MATCHED_TARGETS',n=0,sse=None,physical_mse=None,rmse=None,mae=None,bias=None)
    errors=[float(a)-float(b) for a,b in zip(p,y)];sse=math.fsum(x*x for x in errors)
    require(math.isfinite(sse),'Nonfinite event squared error')
    return dict(status='AVAILABLE_DESCRIPTIVE',n=n,sse=sse,physical_mse=sse/n,rmse=math.sqrt(sse/n),
                mae=math.fsum(abs(x) for x in errors)/n,bias=math.fsum(errors)/n)


def _spread(values):
    if any(v is None for v in values):return dict(mean=None,min=None,max=None,defined_seeds=sum(v is not None for v in values),total_seeds=3)
    return dict(mean=math.fsum(v/3 for v in values),min=min(values),max=max(values),defined_seeds=3,total_seeds=3)


def _peak_prediction(prediction,forecast_times,selected,event):
    indices=np.flatnonzero(selected)
    if not len(indices):return dict(status='UNAVAILABLE_NO_MATCHED_TARGETS',q_cms=None,valid_epoch=None,
                                  timing_from_selected_observed_peak_hours=None,coverage_complete=False)
    index=max(indices,key=lambda i:(float(prediction[i]),-int(forecast_times[i])))
    peak_matched=bool(np.any(forecast_times[indices]==event['peak_epoch']))
    result=dict(status='MAXIMUM_ON_MATCHED_FIXED_LEAD_TRACE',q_cms=float(prediction[index]),valid_epoch=int(forecast_times[index]),
        timing_from_selected_observed_peak_hours=((int(forecast_times[index])-event['peak_epoch'])/HOUR if peak_matched else None),
        coverage_complete=len(indices)==event['reference_requested_hours'],
        limitation='May not be the true predicted event peak when trace support is incomplete; not a single issuance trajectory')
    if not peak_matched:result['timing_status']='UNAVAILABLE_OBSERVED_PEAK_NOT_MATCHED'
    return result


def _at_observed_peak(prediction,forecast_times,selected,event):
    take=np.flatnonzero((forecast_times==event['peak_epoch']) & selected)
    if not len(take):return dict(status='UNAVAILABLE_AT_OBSERVED_PEAK',prediction_q_cms=None,error_cms=None,ratio_to_observed_peak=None)
    require(len(take)==1,'One exact forecast cell per fixed lead and observed peak required')
    value=float(prediction[take[0]]);truth=event['observed_peak_q_cms']
    return dict(status='AVAILABLE_AT_OBSERVED_PEAK',prediction_q_cms=value,error_cms=value-truth,
                ratio_to_observed_peak=value/truth if truth>0 else None)


def _validate(origins,truth,valid,input_admitted,predictions,nwm,nwm_mask,history):
    c=np.asarray(origins);y=np.asarray(truth);mask=np.asarray(valid);admitted=np.asarray(input_admitted)
    nw=np.asarray(nwm);nw_mask=np.asarray(nwm_mask);hist=np.asarray(history)
    require(c.dtype.kind in 'iu' and np.array_equal(c,np.arange(ASSESSMENT_START,ASSESSMENT_END,HOUR)),
            'Exact full8760 assessment-origin calendar required')
    require(y.shape==mask.shape==nw.shape==nw_mask.shape==(8760,18) and hist.shape==(8760,168,2)
        and y.dtype.kind==nw.dtype.kind==hist.dtype.kind=='f' and admitted.shape==(8760,)
        and mask.dtype==admitted.dtype==nw_mask.dtype==np.dtype(bool),'Exact saved array geometry required')
    require(np.isfinite(y[mask]).all() and (y[mask]>=0).all() and np.isnan(y[~mask]).all(),
            'Verification targets must retain approved values and NaN rejected hours')
    require(not nw_mask[:,16:].any() and np.isnan(nw[:,16:]).all() and np.isfinite(nw[nw_mask]).all(),
            'NWM must retain exact supportedH1–16 and absentH17–18')
    expected={(a,s) for a in ARCHITECTURES for s in SEEDS}
    require(isinstance(predictions,Mapping) and set(predictions)==expected,'All four models and three seeds required')
    checked={}
    for key in sorted(expected):
        p=np.asarray(predictions[key]);require(p.shape==(8760,18) and p.dtype.kind=='f'
            and np.isfinite(p[admitted]).all() and np.isnan(p[~admitted]).all(),
            'Saved predictions must be finite on admitted cases and NaN elsewhere')
        checked[key]=p
    require(np.isfinite(hist[admitted]).all() and np.all(hist[admitted,-1,1]==2),
            'Admitted original history and its two-hour input age must be retained')
    future=c[:,None]+np.arange(1,19)*HOUR
    ref=unique_reference(y,mask,future)
    require(int(ref[0][0])==REFERENCE_START and int(ref[0][-1])+HOUR==REFERENCE_END,'Full verification target tail differs')
    return c,y,mask,admitted,checked,nw,nw_mask,hist,future,ref


def analyze_gauge(origins,truth,valid,input_admitted,predictions,nwm,nwm_mask,history,q95,catalog):
    """Explain fixed matched errors without changing forecasts or annual support."""
    c,y,mask,admitted,pred,nw,nw_mask,hist,future,ref=_validate(
        origins,truth,valid,input_admitted,predictions,nwm,nwm_mask,history)
    require(catalog==select_events(*ref,q95),'Catalog differs from the fixed observed-only selection rule')
    primary=admitted & mask.all(axis=1);matched=primary[:,None] & nw_mask
    representatives=[e for e in catalog if e['representative_top3']]
    rows=[]
    for event in representatives:
        window=(future>=event['start_epoch']) & (future<event['end_exclusive_epoch'])
        coverage=[];nwm_rows=[];models={a:dict(seeds={}) for a in ARCHITECTURES}
        for h in LEADS:
            j=h-1;selected=window[:,j] & matched[:,j];independent=window[:,j]&primary
            origin=event['peak_epoch']-h*HOUR;i=(origin-ASSESSMENT_START)//HOUR
            latest=None;latest_time=event['peak_epoch']-(h+2)*HOUR;case=False;complete=False
            if 0<=i<8760:
                value,age=hist[i,-1]
                if np.isfinite(value) and value>=0 and age==2:latest=float(value)
                case=bool(admitted[i]);complete=bool(primary[i])
            coverage.append(dict(lead_hours=h,requested_target_hours=int(window[:,j].sum()),
                valid_observed_target_hours=int((window[:,j]&mask[:,j]).sum()),
                input_admitted_target_hours=int((window[:,j]&admitted).sum()),
                strict_primary_target_hours=int(independent.sum()),matched_nwm_target_hours=int(selected.sum()),
                observed_peak_matched=bool(((future[:,j]==event['peak_epoch']) & selected).any()),
                observed_peak_hydra_primary=bool(((future[:,j]==event['peak_epoch']) & independent).any()),
                latest_observation_context=dict(valid_epoch=latest_time,q_cms=latest,
                    forecast_origin_epoch=origin,full_history_and_inputs_admitted=case,complete_H18_targets=complete,
                    role='ISSUANCE_INPUT_CONTEXT_ONLY_NOT_A_PERSISTENCE_FORECAST_OR_BASELINE')))
            if h<=16:
                score=_score(y[selected,j],nw[selected,j]);score.update(lead_hours=h,
                    at_observed_peak=_at_observed_peak(nw[:,j],future[:,j],selected,event),
                    fixed_lead_trace_maximum=_peak_prediction(nw[:,j],future[:,j],selected,event));nwm_rows.append(score)
        for a in ARCHITECTURES:
            for seed in SEEDS:
                p=pred[a,seed];paired=[]
                for h in PAIRED_LEADS:
                    j=h-1;selected=window[:,j]&matched[:,j];score=_score(y[selected,j],p[selected,j])
                    score.update(lead_hours=h,at_observed_peak=_at_observed_peak(p[:,j],future[:,j],selected,event),
                        fixed_lead_trace_maximum=_peak_prediction(p[:,j],future[:,j],selected,event));paired.append(score)
                selected=window[:,17]&primary;h18=_score(y[selected,17],p[selected,17]);h18.update(lead_hours=18,
                    at_observed_peak=_at_observed_peak(p[:,17],future[:,17],selected,event),
                    fixed_lead_trace_maximum=_peak_prediction(p[:,17],future[:,17],selected,event))
                models[a]['seeds'][str(seed)]=dict(paired=paired,hydra_h18=h18)
            mean=[]
            for j,h in enumerate(PAIRED_LEADS):
                seed_rows=[models[a]['seeds'][str(s)]['paired'][j] for s in SEEDS]
                mean.append(dict(lead_hours=h,n=seed_rows[0]['n'],
                    **{k:_spread([x[k] for x in seed_rows]) for k in ('sse','physical_mse','rmse','mae','bias')},
                    at_observed_peak={k:_spread([x['at_observed_peak'][k] for x in seed_rows])
                        for k in ('prediction_q_cms','error_cms','ratio_to_observed_peak')}))
            models[a]['seed_mean_and_range']=mean
        event_mask=window&matched
        rows.append(dict(event=event,coverage_by_lead=coverage,nwm_paired=nwm_rows,models=models,
            matched_forecast_target_pairs=int(event_mask.sum()),unique_matched_target_hours=int(np.unique(future[event_mask]).size),
            metrics='DESCRIPTIVE_ON_ORIGINAL_ANNUAL_MATCHED_SUPPORT',uncertainty='SEED_MIN_MAX_NOT_CONFIDENCE_INTERVAL'))
    # Disjoint target-time windows partition every original annual matched pair.
    buckets={e['event_id']:(future>=e['start_epoch'])&(future<e['end_exclusive_epoch']) for e in representatives}
    union=np.zeros(future.shape,bool)
    for selected in buckets.values():require(not (union&selected).any(),'Representative windows overlap');union|=selected
    buckets['REST_OF_ASSESSMENT']=~union
    decomp={}
    for a in ARCHITECTURES:
        seed_rows={}
        for seed in SEEDS:
            p=pred[a,seed];lead_rows=[]
            for h in PAIRED_LEADS:
                j=h-1;take=matched[:,j];truth_values=y[take,j]
                mean=math.fsum(float(v) for v in truth_values)/len(truth_values) if len(truth_values) else None
                sst=math.fsum((float(v)-mean)**2 for v in truth_values) if mean is not None else None
                if sst is None or sst<=0:
                    lead_rows.append(dict(lead_hours=h,status='UNDEFINED_NSE_DENOMINATOR',n=int(take.sum()),
                        sst=sst,nwm_minus_hydra_nse=None,contributions={k:None for k in buckets}));continue
                differences=(p[:,j]-y[:,j])**2-(nw[:,j]-y[:,j])**2
                delta=math.fsum(float(v) for v in differences[take])/sst
                contribution={key:math.fsum(float(v) for v in differences[take&selected[:,j]])/sst for key,selected in buckets.items()}
                require(math.isclose(math.fsum(contribution.values()),delta,rel_tol=1e-10,abs_tol=1e-10),'Event and residual contributions do not reconcile')
                lead_rows.append(dict(lead_hours=h,status='DEFINED',n=int(take.sum()),sst=sst,
                    nwm_minus_hydra_nse=delta,contributions=contribution))
            all_defined=all(x['status']=='DEFINED' for x in lead_rows)
            gap=math.fsum(x['nwm_minus_hydra_nse']/16 for x in lead_rows) if all_defined else None
            contributions={key:math.fsum(x['contributions'][key]/16 for x in lead_rows) if all_defined else None for key in buckets}
            seed_rows[str(seed)]=dict(equal_lead_nwm_minus_hydra_nse=gap,contributions=contributions,leads=lead_rows)
        mean_gap=_spread([seed_rows[str(s)]['equal_lead_nwm_minus_hydra_nse'] for s in SEEDS])
        contributions={k:_spread([seed_rows[str(s)]['contributions'][k] for s in SEEDS]) for k in buckets}
        mean_values={k:v['mean'] for k,v in contributions.items()}
        if mean_gap['mean'] is not None:require(math.isclose(math.fsum(mean_values.values()),mean_gap['mean'],rel_tol=1e-10,abs_tol=1e-10),'Seed mean decomposition differs')
        decomp[a]=dict(seeds=seed_rows,seed_mean_and_range=mean_gap,contributions=contributions,
            signed_fraction_of_gap={k:v/mean_gap['mean'] if mean_gap['mean'] not in (None,0) and v is not None else None for k,v in mean_values.items()},
            interpretation='Signed SSE/SST accounting; may be negative or exceed100%; not a causal explanation')
    summary=dict(schema='hydra-assessment-event-diagnostics/v1',event_rule='Q95_EXCEEDING_OBSERVED_HOURS_DESCENDING_EARLIEST_TIE_GREEDY_168H',
        q95_cms=q95,reference_start_epoch=REFERENCE_START,reference_end_exclusive_epoch=REFERENCE_END,
        catalog=catalog,representative_event_count=len(representatives),events=rows,decomposition=decomp,
        fixed_lead_curves_are_successive_origins=True,seed_metrics_averaged_after_scoring=True,
        targets_filled=False,predictions_clipped=False,model_calls=0,new_training=0,
        no_claim_of_independent_floods=True,formal_confidence_intervals=False,
        selection_requires_forecast_admission=False,unsupported_events_replaced=False)
    plot=dict(reference_time=ref[0],observed_q=ref[1],reference_valid=ref[2],forecast_time=future,
        matched_mask=matched,primary_mask=primary,nwm=nw,
        prediction_mean_by_arch={a:np.mean(np.stack([pred[a,s] for s in SEEDS]),axis=0) for a in ARCHITECTURES},
        prediction_min_by_arch={a:np.min(np.stack([pred[a,s] for s in SEEDS]),axis=0) for a in ARCHITECTURES},
        prediction_max_by_arch={a:np.max(np.stack([pred[a,s] for s in SEEDS]),axis=0) for a in ARCHITECTURES})
    return dict(summary=summary,plot=plot)
