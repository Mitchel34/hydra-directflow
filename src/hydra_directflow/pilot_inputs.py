"""Qualification-only H4 inputs and deterministic artificial-gap mechanics.

No statistics are fitted here. Explicit original per-slot provenance is required;
targets are never accessed. Existing H2 source/primary artifacts are unchanged.
"""
from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import math
import re
from types import MappingProxyType

import numpy as np

from .gap_policies import POLICY_IDS, policy_catalog
from .inference import validate_input_coordinates, validate_inputs
from .local_data import GAUGES
from .reference import digest, require
from .training import FULL_FEATURE_SCHEMA, TrainOnlyTransform, INPUTS
from .pilot_provenance import ValidatedProvenanceBatch, validate_native_provenance

H4_FEATURE_SCHEMA = 'PILOT_HRRR_H4_W4_S2_V1'
H4_HISTORY_FEATURES = ('discharge_cms', 'age_hours', 'was_missing', 'elapsed_since_real_hours')
HISTORY_HOURS = 168
BANDS = {'early': (1,55), 'middle': (56,110), 'recent': (111,165)}
COORDINATES = ('origin_time','history_time','forecast_time','site_id','case_id')


def scenario_manifest():
    scenarios = [dict(id='clean',kind='clean',band=None,interior_hours=0,trailing_hours=0,trailing_method=None)]
    for length in (1,2):
        for band in BANDS:
            scenarios.append(dict(id=f'interior_{length}_{band}',kind='interior',band=band,interior_hours=length,trailing_hours=0,trailing_method=None))
    for length in (1,2):
        for method in ('carry','ridge'):
            scenarios.append(dict(id=f'trailing_{length}_{method}',kind='trailing',band=None,interior_hours=0,trailing_hours=length,trailing_method=method))
    for band in BANDS:
        for method in ('carry','ridge'):
            scenarios.append(dict(id=f'mixed_{band}_{method}',kind='mixed',band=band,interior_hours=1,trailing_hours=1,trailing_method=method))
    return dict(schema='hydra-pilot-artificial-gap-scenarios/v1',layout_policy_status='PROPOSED_NOT_FROZEN',history_hours=168,
        history_endpoint_lag_hours=2,scenario_count=17,scenarios=scenarios,
        interior_start_index_bands_inclusive={name:list(bounds) for name,bounds in BANDS.items()},
        mask_algorithm='SHA256_CANONICAL_SEED_CASE_ID_EPOCH_BAND_V1',paired_carry_ridge_masks_identical=True,
        maximum_artificial_missing_slots=2,leading_slots_never_hidden=True,
        mixed_definition='ONE_INTERIOR_SLOT_AND_FINAL_TRAILING_SLOT;RECENT_MAY_REMOVE_REQUIRED_ORIGINAL_RIDGE_CONTEXT',
        selection_uses_Q_values_or_targets=False,scientific_policy_adopted=False)


def _hex(value): return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None


def _seed_identity(case_id,epoch,mask_seed):
    require(isinstance(case_id,str) and re.fullmatch(r'\d{8}:\d{4}-\d{2}-\d{2}T\d{2}:00:00Z',case_id), 'Canonical case ID required for deterministic masks')
    require(type(epoch) is int and 0<=epoch<2**32 and type(mask_seed) is int and 0<=mask_seed<2**32,'Bounded explicit epoch/mask seed required')
    return dict(case_id=case_id,epoch=epoch,mask_seed=mask_seed)


def _draw(identity,purpose):
    return int(digest({**identity,'purpose':purpose}),16)


def artificial_mask(*, case_id, scenario_id, epoch=0, mask_seed=1729):
    """Select slots without Q values, flags, targets, or process-global RNG."""
    identity=_seed_identity(case_id,epoch,mask_seed)
    scenarios={row['id']:row for row in scenario_manifest()['scenarios']}
    require(scenario_id in scenarios,'Unknown explicit artificial-gap scenario')
    scenario=scenarios[scenario_id];mask=np.zeros(168,bool)
    if scenario['interior_hours']:
        low,high=BANDS[scenario['band']]
        start=low+_draw(identity,'interior-band:'+scenario['band'])%(high-low+1)
        mask[start:start+scenario['interior_hours']]=True
    if scenario['trailing_hours']:mask[-scenario['trailing_hours']:]=True
    require(not mask[0] and int(mask.sum())<=2,'Scenario violated leading/total-gap limits')
    mask.setflags(write=False)
    return mask


def make_fill_policy(*, trailing_method='carry', policy_id=None):
    return validate_fill_policy(dict(schema='hydra-pilot-fill-policy/v1',policy_id=POLICY_IDS['combined_2'] if policy_id is None else policy_id,
                                    trailing_method=trailing_method,scenario_methods='EXPLICIT_SCENARIO_MANIFEST'))


def validate_fill_policy(value):
    require(isinstance(value,dict) and set(value)=={'schema','policy_id','trailing_method','scenario_methods'}
            and value['schema']=='hydra-pilot-fill-policy/v1' and value['policy_id'] in policy_catalog()
            and value['trailing_method'] in ('carry','ridge') and value['scenario_methods']=='EXPLICIT_SCENARIO_MANIFEST',
            'Explicit supported JSON fill policy required')
    return dict(value)


def make_augmentation_recipe(*, mixture, gradient_case_ids_sha256, source_identity, evidence_sha256):
    return validate_augmentation_recipe(dict(schema='hydra-pilot-augmentation-recipe/v1',probability=.5,mixture=dict(mixture),
        evidence=dict(scope='GRADIENT_ONLY',gradient_case_ids_sha256=gradient_case_ids_sha256,source_identity=source_identity,evidence_sha256=evidence_sha256)))


def validate_augmentation_recipe(value):
    require(isinstance(value,dict) and set(value)=={'schema','probability','mixture','evidence'}
            and value['schema']=='hydra-pilot-augmentation-recipe/v1' and type(value['probability']) in (int,float)
            and value['probability']==.5,'Explicit probability0.5 augmentation recipe required')
    scenarios={row['id'] for row in scenario_manifest()['scenarios']} - {'clean'}
    mixture=value['mixture'];evidence=value['evidence']
    require(isinstance(mixture,dict) and 0<len(mixture)<=16 and set(mixture)<=scenarios
            and all(type(weight) in (int,float) and math.isfinite(weight) and weight>0 for weight in mixture.values())
            and abs(math.fsum(mixture.values())-1)<=1e-12,'Caller-supplied scenario probability mixture must sum to one')
    require(isinstance(evidence,dict) and set(evidence)=={'scope','gradient_case_ids_sha256','source_identity','evidence_sha256'}
            and evidence['scope']=='GRADIENT_ONLY' and all(_hex(evidence[name]) for name in ('gradient_case_ids_sha256','source_identity','evidence_sha256')),
            'Gradient-only mixture evidence identities required; no empirical fitting inside adapter')
    return dict(schema=value['schema'],probability=.5,mixture={name:float(mixture[name]) for name in sorted(mixture)},evidence=dict(evidence))


def augmentation_scenario(*, case_id, epoch, mask_seed, recipe):
    recipe=validate_augmentation_recipe(recipe);identity=_seed_identity(case_id,epoch,mask_seed)
    if _draw(identity,'augmentation-gate')>=2**255:return 'clean'
    point=_draw(identity,'augmentation-scenario')/2**256;total=0.
    for name,weight in recipe['mixture'].items():
        total+=weight
        if point<total:return name
    return next(reversed(recipe['mixture']))  # Floating-point partition endpoint only.


def _copy_readonly(value):
    result=np.array(value,copy=True);result.setflags(write=False);return result


def _input_coordinates(arrays):
    require(isinstance(arrays,Mapping),'Explicit input-only mapping required')
    keys=set(iter(arrays));clock='forecast_time' if 'forecast_time' in keys else 'target_time'
    coordinates={name:np.asarray(arrays[clock if name=='forecast_time' else name]) for name in COORDINATES}
    for name in ('feature_schema','cohort_admission_digest'):
        require(name in keys,'Explicit original H2 schema/cohort identity required')
        coordinates[name]=np.asarray(arrays[name])
    sites=coordinates['site_id'];require(sites.ndim==1 and len(sites)>0 and sites.dtype.kind in 'iu' and int(sites[0]) in range(3),'Explicit single gauge required')
    cohort=coordinates['cohort_admission_digest']
    require(cohort.shape==() and cohort.dtype.kind=='U' and _hex(cohort.item()),'Explicit original cohort digest required')
    origins=coordinates['origin_time'];require(origins.ndim==1 and origins.dtype.kind in 'iu' and len(origins)==len(sites),'Aligned integer batch origins required')
    order=np.argsort(origins,kind='stable')
    ordered={**coordinates,**{name:coordinates[name][order] for name in COORDINATES}}
    # Training batches may be shuffled. Validate a coordinate-only sorted view
    # before numerical access, while preserving the caller's batch order.
    validate_input_coordinates(ordered,history_hours=168,feature_schema=FULL_FEATURE_SCHEMA,
        expected_cohort_admission_digest=cohort.item(),expected_gauge_id=GAUGES[int(sites[0])])
    coordinates['case_id']=np.array([value.decode('utf-8') if isinstance(value,bytes) else str(value) for value in coordinates['case_id'].tolist()])
    return coordinates,GAUGES[int(sites[0])],cohort.item(),order


def _provenance(value,original_mask,times,**bindings):
    if type(value) is ValidatedProvenanceBatch:
        return value.checked_native(original_mask=original_mask,times=times,**bindings)
    return validate_native_provenance(value,original_mask,times)


def _saved_transform(value):
    state=asdict(value) if isinstance(value,TrainOnlyTransform) else value
    require(isinstance(state,dict),'Explicit saved original-gradient H2 transform required')
    restored=TrainOnlyTransform.from_dict(state)
    require(restored.feature_schema==FULL_FEATURE_SCHEMA and _hex(restored.training_case_ids_sha256)
            and _hex(restored.training_values_sha256),'Saved transform must bind original-gradient full H2 data')
    return restored


def prepare_h4_inputs(arrays, *, history_provenance, saved_h2_transform, inference_source_identity, fill_policy,
                      scenario_id='clean', epoch=0, mask_seed=1729, augmentation_recipe=None,
                      ridge_model=None, expected_ridge_identity=None):
    """Return raw H4 and saved-statistic model inputs, without touching labels.

    The caller must reject/filter case_admissible=False before a model forward.
    Artificially hidden true Q stays exclusively in the caller's reference-side
    object; this return value never contains those hidden original magnitudes.
    """
    coordinates,gauge,cohort,order=_input_coordinates(arrays)
    require(_hex(inference_source_identity),'Explicit source identity required')
    policy=validate_fill_policy(fill_policy);transform=_saved_transform(saved_h2_transform)
    require(type(epoch) is int and 0<=epoch<2**32 and type(mask_seed) is int and 0<=mask_seed<2**32,'Explicit epoch/seed required')
    manifest=scenario_manifest();scenarios={row['id']:row for row in manifest['scenarios']}
    require(scenario_id in scenarios or scenario_id=='augmentation','Explicit prepared scenario required')
    recipe=None if augmentation_recipe is None else validate_augmentation_recipe(augmentation_recipe)
    require((scenario_id=='augmentation') == (recipe is not None),'Augmentation recipe is used only with explicit augmentation scenario')
    if recipe is not None:
        require(recipe['evidence']['gradient_case_ids_sha256']==transform.training_case_ids_sha256
                and recipe['evidence']['source_identity']==inference_source_identity,'Augmentation evidence differs from original gradient transform/source')
    bindings=dict(origins=coordinates['origin_time'],forecast_times=coordinates['forecast_time'],
                  site_ids=coordinates['site_id'],case_ids=coordinates['case_id'],
                  source_identity=inference_source_identity,cohort_digest=cohort)
    if type(history_provenance) is ValidatedProvenanceBatch:
        original_masks=np.asarray(arrays['history_mask'])
        require(original_masks.shape==(len(coordinates['origin_time']),168,2) and original_masks.dtype==np.dtype(bool),
                'Original H2 masks required before provenance reuse')
        _provenance(history_provenance,original_masks[...,0],coordinates['history_time'],**bindings)
    h2={**coordinates}
    for name in INPUTS:
        h2[name],h2[name+'_mask']=np.asarray(arrays[name]),np.asarray(arrays[name+'_mask'])
    ordered={**h2,**{name:h2[name][order] for name in (*COORDINATES,*(x for field in INPUTS for x in (field,field+'_mask')))}}
    validate_inputs(ordered,history_hours=168,feature_schema=FULL_FEATURE_SCHEMA,expected_cohort_admission_digest=cohort,expected_gauge_id=gauge)
    require(np.array_equal(h2['history_mask'][...,0],h2['history_mask'][...,1]),'Original H2 Q/age masks must agree')
    original=h2['history_mask'][...,0].copy();times=coordinates['history_time'];origins=coordinates['origin_time'];n=len(origins)
    age=(origins[:,None]-times)/3600
    require(np.array_equal(h2['history'][...,1][original],age[original]),'Original observed coordinate-age values differ')
    native=_provenance(history_provenance,original,times,**bindings)
    hidden=np.zeros((n,168),bool);effective=original.copy();usable=np.zeros((n,168),bool)
    raw_history=np.full((n,168,4),np.nan,np.float64);raw_mask=np.zeros((n,168,4),bool)
    selected=[];methods=[];fill_receipts=[];failure_reasons=[];admissible=np.zeros(n,bool)
    from .gap_filling import fill_history
    for case in range(n):
        case_id=str(coordinates['case_id'][case]);choice=augmentation_scenario(case_id=case_id,epoch=epoch,mask_seed=mask_seed,recipe=recipe) if recipe is not None else scenario_id
        selected.append(choice);hidden[case]=artificial_mask(case_id=case_id,scenario_id=choice,epoch=epoch,mask_seed=mask_seed)
        require(not hidden[case].any() or original[case].all(),'Artificial paired qualification scenarios require originally complete Q histories')
        effective[case] &= ~hidden[case]
        # Copy only still-observed Q values; artificially hidden true magnitudes
        # never enter the filler or this function's returned objects.
        values=np.full(168,np.nan,np.float64);values[effective[case]]=h2['history'][case,:,0][effective[case]]
        method=scenarios[choice]['trailing_method'] or policy['trailing_method'];methods.append(method)
        result=fill_history(q_values=values,approved_mask=effective[case],history_times=times[case],
            source_observation_times=native['observation_times'][case],source_ids=native['source_ids'][case],source_flags=native['quality_flags'][case],
            cutoff_epoch=int(origins[case]),gauge_id=gauge,policy_id=policy['policy_id'],trailing_method=method,
            ridge_model=ridge_model if method=='ridge' else None,
            expected_ridge_identity=expected_ridge_identity if method=='ridge' else None)
        q=np.asarray(result['usable_q']);mask=np.asarray(result['usable_mask'])
        require(q.shape==mask.shape==(168,) and q.dtype.kind=='f' and mask.dtype==np.dtype(bool)
                and np.isfinite(q[mask]).all() and np.all(~effective[case]|mask),'Filler returned invalid numeric support')
        require(np.array_equal(q[effective[case]],values[effective[case]]),'Filler changed an original observed Q value')
        usable[case]=mask;raw_history[case,:,0]=np.where(mask,q,np.nan);raw_mask[case,:,0]=mask
        raw_history[case,:,1]=age[case];raw_mask[case,:,1]=True
        raw_history[case,:,2]=(~effective[case]).astype(float);raw_mask[case,:,2]=True
        last=None
        for slot in range(168):
            if effective[case,slot]:last=int(times[case,slot])
            if last is not None:
                raw_history[case,slot,3]=(int(times[case,slot])-last)/3600;raw_mask[case,slot,3]=True
        reasons=list(result['failure_reasons'])
        if not h2['weather_mask'][case].all():reasons.append('WEATHER_INPUT_INCOMPLETE')
        if not h2['state_mask'][case].all():reasons.append('SOIL_INPUT_INCOMPLETE')
        admissible[case]=bool(result['success'] and raw_mask[case].all() and h2['weather_mask'][case].all() and h2['state_mask'][case].all())
        failure_reasons.append(tuple(dict.fromkeys(reasons)))
        fill_receipts.append(dict(slot_provenance=result['slot_provenance'],policy_fingerprint=result['policy_fingerprint'],success=bool(result['success'])))
    raw_inputs={**coordinates,'history':raw_history,'history_mask':raw_mask,'feature_schema':np.array(H4_FEATURE_SCHEMA)}
    for name in ('weather','state'):raw_inputs[name]=h2[name].copy();raw_inputs[name+'_mask']=h2[name+'_mask'].copy()
    mapped={}
    for name in INPUTS:
        value,mask=raw_inputs[name],raw_inputs[name+'_mask'];output=np.zeros(value.shape,np.float32)
        channels=2 if name=='history' else value.shape[-1]
        for column in range(channels):
            stats=transform.parameters[name][column];good=mask[...,column]
            output[...,column][good]=(value[...,column][good]-stats['mean'])/stats['scale']
        if name=='history':
            output[...,2][mask[...,2]]=value[...,2][mask[...,2]]
            output[...,3][mask[...,3]]=value[...,3][mask[...,3]]/168.
        require(np.isfinite(output).all(),'Saved H2/fixed H4 transform overflowed')
        mapped[name],mapped[name+'_mask']=output,mask.copy()
    raw_inputs=MappingProxyType({name:_copy_readonly(value) for name,value in raw_inputs.items()})
    transformed=MappingProxyType({name:_copy_readonly(value) for name,value in mapped.items()})
    provenance=MappingProxyType({**native,'original_observation_mask':_copy_readonly(original),'artificial_hidden_mask':_copy_readonly(hidden),
        'effective_observation_mask':_copy_readonly(effective),'usable_q_mask':_copy_readonly(usable),
        'selected_scenarios':tuple(selected),'trailing_methods':tuple(methods),'failure_reasons':tuple(failure_reasons),'fill_receipts':tuple(fill_receipts)})
    receipt=dict(schema='hydra-pilot-h4-input-adapter/v1',feature_schema=H4_FEATURE_SCHEMA,history_features=list(H4_HISTORY_FEATURES),
        source_identity=inference_source_identity,cohort_admission_digest=cohort,case_ids_sha256=digest(coordinates['case_id'].tolist()),
        saved_h2_transform_sha256=digest(asdict(transform)),training_case_ids_sha256=transform.training_case_ids_sha256,
        scenario_manifest_sha256=digest(manifest),layout_policy_status='PROPOSED_NOT_FROZEN',fill_policy=policy,fill_policy_sha256=digest(policy),
        augmentation_recipe_sha256=None if recipe is None else digest(recipe),scenario_request=scenario_id,epoch=epoch,mask_seed=mask_seed,
        expected_ridge_identity=expected_ridge_identity,batch_order_preserved=True,
        selected_scenarios=selected,artificial_mask_sha256=digest(hidden.tolist()),original_provenance_sha256=digest(native),
        case_count=n,admissible_cases=int(admissible.sum()),failed_cases=n-int(admissible.sum()),
        fixed_channel_transform=dict(was_missing='IDENTITY_0_OR_1',elapsed_since_real_hours='DIVIDE_BY168'),
        elapsed_since_real_definition='SLOT_TIME_MINUS_LATEST_EFFECTIVELY_OBSERVED_SOURCE_AT_OR_BEFORE_SLOT;IMPUTED_SLOTS_NEVER_RESET_CLOCK',
        original_observed_flags_and_source_ids_preserved=True,hidden_true_Q_returned=False,artificial_hiding_precedes_filling=True,
        filler_observation_mask_is_effective_after_artificial_hiding=True,
        target_values_read=False,target_masks_read=False,weather_or_soil_filled=False,transform_fit_performed=False,
        scientific_policy_adopted=False,failed_cases_must_not_be_forwarded=True)
    return dict(raw_inputs=raw_inputs,transformed_inputs=transformed,case_admissible=_copy_readonly(admissible),provenance=provenance,receipt=receipt)
