"""Deterministic synthetic-only L168/H18 qualification data; no filesystem reads."""
from datetime import datetime, timezone
import numpy as np

from hydra_directflow.local_data import iso
from hydra_directflow.reference import digest
from hydra_directflow.training import FULL_FEATURE_SCHEMA, array_identity

GAUGE = '03161000'
VERSION = 'SYNTHETIC_GAP_PILOT_V1'
HOUR = 3600


def epoch(value):
    return int(datetime.fromisoformat(value.replace('Z','+00:00')).timestamp())


ROLE_BOUNDS = {
    'fit': {'start_utc':'2019-01-01T00:00:00Z','end_utc':'2019-01-10T00:00:00Z'},
    'inner_stop': {'start_utc':'2019-01-10T00:00:00Z','end_utc':'2019-01-20T00:00:00Z'},
    'outer': {'start_utc':'2019-01-20T00:00:00Z','end_utc':'2019-02-01T00:00:00Z'}}
REFIT = '2019-01-20T00:00:00Z'


def synthetic_q(times):
    hours=(np.asarray(times,dtype=np.float64)-epoch('2018-12-01T00:00:00Z'))/HOUR
    return 25. + 3.*np.sin(hours/24.) + .3*np.cos(hours/7.)


def inputs(origins):
    origins=np.asarray(origins,dtype=np.int64);n=len(origins)
    history=origins[:,None]-np.arange(169,1,-1)*HOUR
    future=origins[:,None]+np.arange(1,19)*HOUR
    ids=np.array([GAUGE+':'+iso(int(t)) for t in origins])
    values=np.stack([synthetic_q(history),np.broadcast_to(np.arange(169,1,-1),(n,168))],axis=-1)
    weather=np.empty((n,18,4),np.float64)
    weather[...,0]=.1+.03*np.sin(future/HOUR/13.)
    weather[...,1]=285.+4.*np.sin(future/HOUR/24.)
    weather[...,2]=.005+.0001*np.cos(future/HOUR/7.)
    weather[...,3]=.02
    state=np.tile(np.array([.25,4.]),(n,1))
    cohort=digest({'fixture':VERSION,'gauge':GAUGE,'case_ids':ids.tolist()})
    arrays={'origin_time':origins,'history_time':history,'forecast_time':future,
        'site_id':np.zeros(n,np.int64),'case_id':ids,'history':values,
        'history_mask':np.ones(values.shape,bool),'weather':weather,
        'weather_mask':np.ones(weather.shape,bool),'state':state,'state_mask':np.ones(state.shape,bool),
        'feature_schema':np.array(FULL_FEATURE_SCHEMA),'cohort_admission_digest':np.array(cohort)}
    provenance={'observation_times':history.tolist(),
        'quality_flags':[[['A'] for _ in range(168)] for _ in range(n)],
        'source_ids':[['synthetic-usgs/'+GAUGE+'/'+iso(int(t)) for t in row] for row in history]}
    source=digest({'fixture':VERSION,'arrays_sha256':array_identity(arrays),'provenance_sha256':digest(provenance)})
    return arrays,provenance,source,cohort


def training_fixture():
    origins=np.r_[epoch('2019-01-02T00:00:00Z')+np.arange(128)*HOUR,
                  epoch('2019-01-11T00:00:00Z')+np.arange(64)*HOUR]
    arrays,provenance,_,cohort=inputs(origins)
    arrays['target_time']=arrays.pop('forecast_time')
    arrays['target']=synthetic_q(arrays['target_time'])
    arrays['target_mask']=np.ones((192,18),bool)
    source=digest({'fixture':VERSION,'arrays_sha256':array_identity(arrays),'provenance_sha256':digest(provenance)})
    return arrays,provenance,source,cohort


def inference_fixture():
    return inputs(epoch('2019-01-21T00:00:00Z')+np.arange(64)*HOUR)


def ridge_fixture():
    times=np.arange(epoch(ROLE_BOUNDS['fit']['start_utc']),epoch(ROLE_BOUNDS['fit']['end_utc']),HOUR,dtype=np.int64)
    return {'times':times,'values':synthetic_q(times),'approved_mask':np.ones(len(times),bool),
        'source_ids':['synthetic-usgs/'+GAUGE+'/'+iso(int(t)) for t in times],
        'source_flags':[['A'] for _ in times]}


class PoisonLabels(dict):
    def __getitem__(self,key):
        if key in ('target','target_mask'):
            raise AssertionError('Synthetic inference must not access verification labels')
        return super().__getitem__(key)
