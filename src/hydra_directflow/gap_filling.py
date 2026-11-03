"""Proposed causal L168 Q fillers and a source-bound direct ridge fallback.

Only explicitly supplied original observation data are consumed. No weather or
forecast H18 targets are accepted. Real-data fitting/integration remains an
external authorization and provenance requirement; synthetic tests do not grant it.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .gap_policies import classify_history_window, policy_catalog, POLICY_CATALOG_SHA256
from .local_data import GAUGES
from .quality import PRIMARY_APPROVED, admit_usgs_quality
from .reference import ASSESSMENT_START, HOUR, digest, require

IMPLEMENTATION_SHA256 = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in ('gap_filling.py','gap_policies.py','quality.py')}
RIDGE_SPEC = dict(schema='hydra-causal-history-ridge-policy/v1',alpha=1.0,real_consecutive_lags=6,
                 direct_output_hours=[1,2],feature_standardization='GRADIENT_ONLY_MEAN_POPULATION_STD_CONSTANT_SCALE1',
                 intercept_penalized=False,simulated_lag_hours=2,quality_policy=PRIMARY_APPROVED,
                 recursive_self_feeding=False,negative_or_nonfinite_fill='FAIL_NO_CLIP_NO_FALLBACK')
RIDGE_POLICY_SHA256 = digest(RIDGE_SPEC)


def _hash(value): return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def _epoch(value):
    require(isinstance(value,Integral) and not isinstance(value,(bool,np.bool_)),'Explicit integer UTC epoch required')
    value=int(value);require(0<=value<ASSESSMENT_START+1 and value%HOUR==0,'Exact development UTC hour required');return value


def _metadata(times, approved_mask, source_observation_times, source_ids, source_flags):
    times=np.asarray(times);mask=np.asarray(approved_mask)
    require(times.ndim==1 and 1<=len(times)<=100000 and times.dtype.kind in 'iu'
            and np.all((times>=0)&(times<ASSESSMENT_START)&(times%HOUR==0))
            and np.all(np.diff(times.astype(np.int64))>0),'Unique ordered exact-hour development observation clocks required')
    require(mask.shape==times.shape and mask.dtype==np.dtype(bool),'Exact Boolean original approved mask required')
    require(type(source_ids) is tuple and type(source_flags) is tuple and len(source_ids)==len(source_flags)==len(times),
            'Explicit aligned tuples of original source IDs and qualifier tuples required')
    sources=tuple(source_observation_times);require(len(sources)==len(times),'Source timestamps must align')
    validated=[]
    for i,(source,identity,flags) in enumerate(zip(sources,source_ids,source_flags)):
        require(identity is None or isinstance(identity,str) and 0<len(identity)<=2048,'Original source ID must be nonempty text or None')
        require(type(flags) is tuple and all(type(flag) is str and flag for flag in flags),'Preserve exact original qualifier tuples')
        clock=None if source is None else _epoch(source)
        require(clock is None or clock==int(times[i]),'Source observation timestamp differs from its exact slot')
        if mask[i]:
            require(clock is not None and identity is not None and admit_usgs_quality(flags,policy_id=PRIMARY_APPROVED),
                    'Approved observation requires original clock, source ID and A with optional R qualifiers')
        validated.append(clock)
    return times.astype(np.int64,copy=False),mask,tuple(validated)


def _q_value(values,index):
    value=values[index]  # Deliberately never convert/read the whole Q array.
    require(isinstance(value,Real) and not isinstance(value,(bool,np.bool_)) and math.isfinite(value) and value>=0,
            'Original approved Q values must be finite nonnegative scalars')
    return float(value)


def _bounds(start,end,refit):
    start,end,refit=(_epoch(value) for value in (start,end,refit))
    require(start<end<=refit<=ASSESSMENT_START,'Explicit gradient interval and later refit cutoff required')
    return start,end,refit


def _validate_model(payload):
    expected={'schema','policy','policy_sha256','implementation_sha256','gauge_id','source_identity',
              'gradient_start_epoch','gradient_end_epoch','refit_epoch','training_pairs','training_pairs_sha256',
              'source_metadata_sha256','training_values_sha256','feature_mean','feature_scale','constant_features',
              'coefficients','intercept','model_identity'}
    require(isinstance(payload,dict) and set(payload)==expected,'Exact ridge artifact fields required')
    require(payload['schema']=='hydra-causal-history-ridge/v1' and payload['policy']==RIDGE_SPEC
            and payload['policy_sha256']==RIDGE_POLICY_SHA256 and payload['implementation_sha256']==IMPLEMENTATION_SHA256,
            'Ridge policy/implementation fingerprint differs')
    require(payload['gauge_id'] in GAUGES and all(_hash(payload[key]) for key in
            ('source_identity','training_pairs_sha256','source_metadata_sha256','training_values_sha256','model_identity')),
            'Ridge gauge/source/training identities required')
    _bounds(payload['gradient_start_epoch'],payload['gradient_end_epoch'],payload['refit_epoch'])
    require(type(payload['training_pairs']) is int and 1<=payload['training_pairs']<=100000,'Bounded ridge fit support required')
    for name,shape in (('feature_mean',(6,)),('feature_scale',(6,)),('coefficients',(6,2)),('intercept',(2,))):
        value=np.asarray(payload[name],dtype=np.float64)
        require(value.shape==shape and np.isfinite(value).all(),'Malformed saved ridge parameter: '+name)
    require(np.all(np.asarray(payload['feature_scale'])>0) and type(payload['constant_features']) is list
            and len(payload['constant_features'])==6 and all(type(v) is bool for v in payload['constant_features']),
            'Ridge scale/constant-feature metadata differs')
    require(all(not flag or scale==1. for flag,scale in zip(payload['constant_features'],payload['feature_scale'])),
            'Constant ridge features must retain scale1')
    require(digest({key:value for key,value in payload.items() if key!='model_identity'})==payload['model_identity'],
            'Ridge serialized content identity differs')


@dataclass(frozen=True)
class RidgeFallback:
    _payload_json: str

    @property
    def identity(self): return self.to_dict()['model_identity']
    def to_dict(self): return json.loads(self._payload_json)

    @classmethod
    def from_dict(cls,payload,*,expected_source_identity,expected_gauge_id,expected_gradient_start_epoch,
                  expected_gradient_end_epoch,expected_refit_epoch,expected_model_identity):
        _validate_model(payload)
        require(payload['source_identity']==expected_source_identity and payload['gauge_id']==expected_gauge_id
                and payload['gradient_start_epoch']==expected_gradient_start_epoch and payload['gradient_end_epoch']==expected_gradient_end_epoch
                and payload['refit_epoch']==expected_refit_epoch and payload['model_identity']==expected_model_identity,
                'Expected ridge source/gauge/fit cutoff/identity differs')
        return cls(json.dumps(payload,sort_keys=True,allow_nan=False))

    def _predict_values(self,original_six_values):
        payload=self.to_dict();_validate_model(payload)
        value=np.asarray(original_six_values,dtype=np.float64)
        require(value.shape==(6,) and np.isfinite(value).all() and (value>=0).all(),'Six original observed ridge lag values required')
        with np.errstate(over='ignore',invalid='ignore'):
            standardized=(value-np.asarray(payload['feature_mean']))/np.asarray(payload['feature_scale'])
            return standardized@np.asarray(payload['coefficients'])+np.asarray(payload['intercept'])


def fit_ridge_fallback(*,observation_times,q_values,approved_mask,source_observation_times,source_ids,source_flags,
                       gauge_id,gradient_start_epoch,gradient_end_epoch,refit_epoch,source_identity):
    """Fit alpha1 ridge only on qualified contiguous original-Q 6+2 windows.

    All metadata, pair clocks, masks, qualifiers and refit availability are
    examined BEFORE any Q value is fetched. Non-gradient/unsupported series
    rows may remain in the supplied series, but their values are never read.
    This is NOT a fit on neural forecast targets or a fitted history filler
    from stopping/outer observations. It does not choose an execution budget.
    """
    require(gauge_id in GAUGES and _hash(source_identity),'Explicit ridge gauge/source identity required')
    start,end,refit=_bounds(gradient_start_epoch,gradient_end_epoch,refit_epoch)
    times,mask,sources=_metadata(observation_times,approved_mask,source_observation_times,source_ids,source_flags)
    require(len(q_values)==len(times),'Q values must align with observation metadata')
    pairs=[]
    for stop in range(7,len(times)):
        rows=tuple(range(stop-7,stop+1));clocks=times[list(rows)]
        if not (np.all(np.diff(clocks)==HOUR) and start<=int(clocks[0]) and int(clocks[-1])<end
                and int(clocks[-1])+2*HOUR<=refit and mask[list(rows)].all()):continue
        pairs.append(rows)
    require(pairs,'No approved contiguous gradient-only ridge training pairs satisfy target/refit bounds')
    # No values from an unqualified pair, withheld observation or future role are read.
    used=sorted({index for rows in pairs for index in rows});observed={i:_q_value(q_values,i) for i in used}
    x=np.array([[observed[i] for i in rows[:6]] for rows in pairs],dtype=np.float64)
    y=np.array([[observed[i] for i in rows[6:]] for rows in pairs],dtype=np.float64)
    mean=x.mean(0);deviation=x.std(0);constant=deviation==0;scale=np.where(constant,1.,deviation)
    require(np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all(),'Ridge gradient feature statistics overflowed')
    z=(x-mean)/scale;zmean=z.mean(0);ymean=y.mean(0);centered=z-zmean
    gram=centered.T@centered+np.eye(6);cross=centered.T@(y-ymean)
    require(np.isfinite(gram).all() and np.isfinite(cross).all() and np.isfinite(ymean).all(),'Ridge fit arithmetic overflowed')
    coefficients=np.linalg.solve(gram,cross);intercept=ymean-zmean@coefficients
    core=dict(schema='hydra-causal-history-ridge/v1',policy=RIDGE_SPEC,policy_sha256=RIDGE_POLICY_SHA256,
        implementation_sha256=IMPLEMENTATION_SHA256,gauge_id=gauge_id,source_identity=source_identity,
        gradient_start_epoch=start,gradient_end_epoch=end,refit_epoch=refit,training_pairs=len(pairs),
        training_pairs_sha256=digest([[{'time':int(times[i]),'source_id':source_ids[i],'flags':source_flags[i]} for i in rows] for rows in pairs]),
        source_metadata_sha256=digest(dict(times=times.tolist(),approved_mask=mask.tolist(),source_times=sources,source_ids=source_ids,source_flags=source_flags)),
        training_values_sha256=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),
        feature_mean=mean.tolist(),feature_scale=scale.tolist(),constant_features=constant.tolist(),
        coefficients=coefficients.tolist(),intercept=intercept.tolist())
    payload={**core,'model_identity':digest(core)}
    return RidgeFallback.from_dict(payload,expected_source_identity=source_identity,expected_gauge_id=gauge_id,
        expected_gradient_start_epoch=start,expected_gradient_end_epoch=end,expected_refit_epoch=refit,
        expected_model_identity=payload['model_identity'])


def filling_policy_fingerprint(policy_id,trailing_method,*,ridge_identity=None):
    require(policy_id in policy_catalog() and trailing_method in ('carry','ridge'),'Explicit supported history repair policy/method required')
    require((trailing_method=='carry' and ridge_identity is None) or (trailing_method=='ridge' and _hash(ridge_identity)),
            'Ridge method requires its bound model identity; carry cannot hide an unused model')
    return digest(dict(schema='hydra-causal-history-fill-policy/v1',gap_policy_id=policy_id,
        gap_policy_catalog_sha256=POLICY_CATALOG_SHA256,trailing_method=trailing_method,ridge_identity=ridge_identity,
        interior='LINEAR_ORIGINAL_APPROVED_ENDPOINTS',quality_policy=PRIMARY_APPROVED,
        window_failure='ATOMIC_NO_DERIVED_VALUES_APPLIED',implementation_sha256=IMPLEMENTATION_SHA256))


def fill_history(*,q_values,approved_mask,history_times,source_observation_times,source_ids,source_flags,
                 cutoff_epoch,gauge_id,policy_id,trailing_method='carry',ridge_model=None,expected_ridge_identity=None):
    """Return separate usable values/mask; original masks/qualifiers never change.

    The supplied approved mask is the input-available mask AFTER any artificial
    hiding, not a claim that a truly observed source record never existed. The
    adapter must preserve its pre-hiding source mask separately. Successful
    fills become usable numeric inputs but never original observations or later
    endpoints. Any failed attempt leaves ALL missing slots unfilled.
    Ridge is direct, trained once externally, and available only at/after refit.
    """
    require(gauge_id in GAUGES and policy_id in policy_catalog() and trailing_method in ('carry','ridge'),
            'Explicit gauge, gap policy and trailing method required')
    classification=classify_history_window(history_times=history_times,approved_mask=approved_mask,
        source_observation_times=source_observation_times,cutoff_epoch=cutoff_epoch)
    times,mask,sources=_metadata(history_times,approved_mask,source_observation_times,source_ids,source_flags)
    require(len(q_values)==168,'Exact L168 Q input required')
    if trailing_method=='ridge':
        require(type(ridge_model) is RidgeFallback and _hash(expected_ridge_identity),'Explicit fitted ridge artifact and expected identity required')
        payload=ridge_model.to_dict();_validate_model(payload)
        require(ridge_model.identity==expected_ridge_identity and payload['gauge_id']==gauge_id
                and payload['refit_epoch']<=int(cutoff_epoch),'Ridge identity/gauge/refit is not available at this cutoff')
    else:require(ridge_model is None and expected_ridge_identity is None,'Carry policy cannot silently use a ridge artifact')
    policy_sha=filling_policy_fingerprint(policy_id,trailing_method,ridge_identity=expected_ridge_identity)
    values=np.full(168,np.nan);usable=mask.copy();observed={i:_q_value(q_values,i) for i in range(168) if mask[i]}
    for i,value in observed.items():values[i]=value
    provenance=[dict(slot_index=i,slot_epoch=int(times[i]),input_observation_available=bool(mask[i]),
        original_source_id=source_ids[i],original_source_flags=source_flags[i],original_source_observation_epoch=sources[i],
        fill_method=None,derived=False,usable=bool(mask[i]),fill_status='ORIGINAL_OBSERVATION' if mask[i] else 'NOT_FILLED',
        support_source_ids=(),support_source_flags=(),support_observation_epochs=()) for i in range(168)]
    decision=classification['candidates'][policy_id];failures=list(decision['reasons']);pending={}
    if decision['history_admissible']:
        for run in classification['runs']:
            start,end=run['start_index'],run['end_index_exclusive'];position=run['position']
            if position=='interior':
                support=(start-1,end);left,right=(observed[i] for i in support)
                proposed={i:left+(right-left)*(i-support[0])/(support[1]-support[0]) for i in range(start,end)}
                method='LINEAR_INTERPOLATION_ORIGINAL_ENDPOINTS'
            elif position=='trailing':
                anchor=start-1
                if trailing_method=='carry':support=(anchor,);proposed={i:observed[anchor] for i in range(start,end)};method='CARRY_FORWARD_ORIGINAL_READING'
                else:
                    support=tuple(range(anchor-5,anchor+1));method='RIDGE_DIRECT_ORIGINAL_SIX_LAGS'
                    if support[0]<0 or not mask[list(support)].all():
                        failures.append('RIDGE_ORIGINAL_SIX_LAG_CONTEXT_UNAVAILABLE')
                        for i in range(start,end):provenance[i].update(fill_method=method,fill_status='FAILED_CONTEXT')
                        continue
                    # Exact history clocks and the original masks prove all six are past, consecutive and available.
                    prediction=ridge_model._predict_values([observed[i] for i in support])
                    proposed={i:float(prediction[i-anchor-1]) for i in range(start,end)}
            else:raise ValueError('Unsupported gap position passed structural admission')
            for i,value in proposed.items():
                provenance[i].update(fill_method=method,support_source_ids=tuple(source_ids[j] for j in support),
                    support_source_flags=tuple(source_flags[j] for j in support),
                    support_observation_epochs=tuple(sources[j] for j in support),
                    simulated_support_available_epoch=max(sources[j]+2*HOUR for j in support),
                    ridge_model_identity=expected_ridge_identity if method.startswith('RIDGE') else None)
                reason='NONFINITE_DERIVED_VALUE' if not math.isfinite(value) else 'NEGATIVE_DERIVED_VALUE' if value<0 else None
                if reason:failures.append(reason);provenance[i].update(fill_status='FAILED_VALUE',failure_reason=reason)
                else:pending[i]=value;provenance[i]['fill_status']='PENDING_WINDOW_CHECK'
    success=not failures
    if success:
        for i,value in pending.items():values[i]=value;usable[i]=True;provenance[i].update(fill_status='FILLED',derived=True,usable=True)
    else:
        for i in pending:provenance[i]['fill_status']='NOT_APPLIED_WINDOW_FAILED'
    original=mask.copy()
    for value in (values,usable,original):value.setflags(write=False)
    return dict(usable_q=values,usable_mask=usable,supplied_approved_mask=original,slot_provenance=tuple(provenance),
        success=success,failure_reasons=list(dict.fromkeys(failures)),history_candidate=classification,
        policy_fingerprint=policy_sha,policy_id=policy_id,trailing_method=trailing_method,
        original_source_ids=source_ids,original_source_flags=source_flags,
        numerical_fills_applied=len(pending) if success else 0,
        filled_values_are_observations=False,targets_read_or_changed=False,weather_read_or_changed=False,
        actual_publication_vintage_verified=False,scientific_policy_adopted=False)
