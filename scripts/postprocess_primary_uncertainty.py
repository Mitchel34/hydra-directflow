#!/usr/bin/env python3
"""Bounded uncertainty stage over authenticated, already-admitted development arrays.

Called by the primary postprocessing controller with its resource guard. This
module has no standalone execution, model calls, raw-source fetches or new fits.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

from hydra_directflow.local_data import bounded_npy
from hydra_directflow.post_uncertainty import (ARCHITECTURES, FOLDS, GAUGES, SEEDS,
    make_draws, evaluate_matched, primary_family)
from hydra_directflow.quality import PRIMARY_APPROVED
from hydra_directflow.reference import digest, require
from hydra_directflow.scoring import ScoreIdentity, validate_score_coordinates
from hydra_directflow.trainer_bundle import _array_digest
from hydra_directflow.uncertainty import validate_origin_calendar, weighted_physical_h18

MAX_GROUP_BYTES = 256 * 1024**2
TOLERANCE = 1e-9
ARRAY_KEYS = {'origins', 'forecast_time', 'labels', 'label_mask', 'comparison_mask', 'predictions', 'panel_ids'}
BLOCKS = (168, 336, 672)


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _relative(run, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute()
            and all(x not in ('.', '..') for x in Path(name).parts), 'Safe relative group path required')
    path = run / name
    require(path == path.resolve() and path.is_relative_to(run), 'Group path escaped postprocessing directory')
    return path


def _artifact(guard, descriptor, maximum):
    require(isinstance(descriptor, dict) and set(descriptor) == {'path', 'bytes', 'sha256'}
            and type(descriptor['bytes']) is int and 0 < descriptor['bytes'] <= maximum,
            'Exact bounded group descriptor required')
    raw = guard.read(_relative(guard.run, descriptor['path']), maximum)
    require(len(raw) == descriptor['bytes'] and _sha(raw) == descriptor['sha256'], 'Prepared group artifact changed')
    return raw


def _unicode_array(raw, count):
    require(len(raw) <= 128 * 1024, 'Bounded panel identifier array required')
    stream = io.BytesIO(raw); version = np.lib.format.read_magic(stream)
    require(version in ((1, 0), (2, 0)), 'Supported panel identifier NPY version required')
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(stream)
    require(shape == (count,) and not fortran and dtype.kind == 'U' and dtype.itemsize <= 1024,
            'Bounded Unicode panel identifiers required')
    return bounded_npy(raw, shape=(count,), dtype=dtype)


def _load_group(plan, original, guard, record):
    fold, gauge = record['fold'], record['gauge_id']
    name = f'{fold}__{gauge}'
    require(record['group'] == name and record['plan_sha256'] == plan['plan_sha256']
            and fold in FOLDS and gauge in GAUGES and set(record['artifacts']) == {'arrays', 'manifest', 'audit'},
            'Exact prepared group identity required')
    for kind, descriptor in record['artifacts'].items():
        suffix = 'npz' if kind == 'arrays' else 'json'
        require(descriptor['path'] == f'groups/{name}/{kind}.{suffix}', 'Prepared group artifact path differs')
    commit = json.loads(guard.read(guard.run / 'groups' / name / 'COMMIT.json', 2 * 1024**2))
    require(commit == record, 'Audit group differs from immutable group commit')
    manifest = json.loads(_artifact(guard, record['artifacts']['manifest'], 4 * 1024**2))
    audit = json.loads(_artifact(guard, record['artifacts']['audit'], 8 * 1024**2))
    require(manifest['content_sha256'] == digest({k: v for k, v in manifest.items() if k != 'content_sha256'}),
            'Prepared manifest content digest differs')
    require(manifest['schema'] == 'hydra-postprocessing-development-group/v1'
            and manifest['fold'] == fold and manifest['gauge_id'] == gauge
            and manifest['plan_sha256'] == original['plan_sha256'] == plan['original_plan_sha256']
            and manifest['protocol_sha256'] == original['protocol_sha256'] == plan['protocol_sha256']
            and manifest['source_plan_sha256'] == original['source_plan_sha256']
            and manifest['snapshot_sha256'] == original['snapshot_sha256']
            and manifest['protected_assessment_access'] is False and manifest['target_imputation'] is False
            and manifest['negative_predictions_clipped'] is False and manifest['quality_policy'] == PRIMARY_APPROVED,
            'Prepared data provenance or scientific scope differs')
    expected = [{k: j[k] for k in ('job_id', 'architecture', 'input_arm', 'history_hours', 'seed', 'device')}
                for j in original['jobs'] if j['fold'] == fold and j['gauge_id'] == gauge]
    require(manifest['panels'] == expected and len(expected) == (36 if fold == 'final_development' else 8),
            'Complete original model/arm/seed ordering required')
    require(audit['status'] == 'ALL_RAW_PREDICTIONS_AND_PHYSICAL_METRICS_MATCH'
            and audit['panels'] == len(expected) and audit['leads'] == len(expected) * 18
            and audit['original_coordinates_and_masks_preserved'] is True
            and audit['protected_assessment_access'] is False
            and [r['job_id'] for r in audit['results']] == [p['job_id'] for p in expected],
            'Independent metric audit is incomplete or mismatched')
    n, panels = manifest['calendar_hours'], len(expected)
    require(type(n) is int and 0 < n <= 8784, 'Bounded development calendar required')
    raw = _artifact(guard, record['artifacts']['arrays'], MAX_GROUP_BYTES)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        require(len(entries) == len(ARRAY_KEYS) and {x.filename for x in entries} == {k+'.npy' for k in ARRAY_KEYS}
                and sum(x.file_size for x in entries) <= MAX_GROUP_BYTES
                and all(x.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) for x in entries),
                'Bounded prepared array archive inventory required')
        arrays = {name: bounded_npy(archive.read(name+'.npy'), shape=shape, dtype=np.dtype(dtype))
                  for name, shape, dtype in [('origins',(n,),'<i8'), ('forecast_time',(n,18),'<i8'),
                                            ('label_mask',(n,18),'?'), ('comparison_mask',(n,),'?')]}
        arrays['panel_ids'] = _unicode_array(archive.read('panel_ids.npy'), panels)
        validate_origin_calendar(arrays['origins'], start_utc=manifest['calendar_start_utc'],
                                 end_exclusive_utc=manifest['calendar_end_exclusive_utc'])
        require(np.array_equal(arrays['forecast_time'], arrays['origins'][:,None]+np.arange(1,19)[None,:]*3600)
                and arrays['panel_ids'].tolist() == [p['job_id'] for p in expected]
                and digest(arrays['comparison_mask'].tolist()) == manifest['comparison_mask_sha256'],
                'Prepared clocks, panel identities or admission mask changed')
        # Decode outcomes only after archive identity and full development clocks.
        arrays['labels'] = bounded_npy(archive.read('labels.npy'), shape=(n,18), dtype=np.dtype('<f8'))
        arrays['predictions'] = bounded_npy(archive.read('predictions.npy'), shape=(panels,n,18), dtype=np.dtype('<f8'))
    require(_array_digest(arrays) == manifest['group_arrays_sha256'], 'Prepared typed array digest differs')
    mask = arrays['comparison_mask']
    require(np.array_equal(arrays['label_mask'], np.broadcast_to(mask[:,None],(n,18)))
            and int(mask.sum()) == manifest['admitted_origins']
            and np.isfinite(arrays['labels'][mask]).all() and np.all(arrays['labels'][mask] >= 0)
            and np.isfinite(arrays['predictions'][:,mask]).all()
            and np.isnan(arrays['labels'][~mask]).all() and np.isnan(arrays['predictions'][:,~mask]).all(),
            'Original complete-case support or unavailable-calendar values changed')
    guard.check()
    return arrays, manifest, audit


def _case_ids(origins, gauge):
    from datetime import datetime, timezone
    return np.array([gauge+':'+datetime.fromtimestamp(int(t),timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') for t in origins])


def _baseline_check(result, audit):
    largest = 0.
    require(len(result['baseline_primary']) == len(audit['results']), 'Baseline panel count differs')
    for i, record in enumerate(audit['results']):
        old = record['independently_recomputed']
        actual, expected = float(result['baseline_primary'][i]), old['primary_score']
        require(expected is not None and np.isfinite(actual)
                and np.isclose(actual, expected, rtol=TOLERANCE, atol=TOLERANCE), 'Bootstrap baseline primary differs from audited metrics')
        largest = max(largest, abs(actual - expected))
        for h, lead in enumerate(old['leads']):
            require(np.isclose(result['baseline_per_lead_nse'][i,h], lead['nse'], rtol=TOLERANCE, atol=TOLERANCE)
                    and int(result['baseline_distinct_hours'][h]) == lead['distinct_target_hours']
                    and int(result['baseline_distinct_dates'][h]) == lead['distinct_utc_dates'],
                    'Bootstrap baseline lead metric/support differs')
    return dict(panels=len(audit['results']), maximum_absolute_primary_difference=largest,
                relative_and_absolute_tolerance=TOLERANCE)


def _reference_check(draws, arrays, manifest, result):
    """Fixed first draw/panel0 for Jefferson in each fold/block; never score-selected."""
    panel = manifest['panels'][0]
    comparison = np.broadcast_to(arrays['comparison_mask'][:,None], arrays['labels'].shape).copy()
    schema = 'Q_ONLY_H2_HORIZON1_S0_V1' if panel['input_arm'] == 'q_only' else 'FULL_HRRR_H2_W4_S2_V1'
    expected = weighted_physical_h18(case_ids=_case_ids(arrays['origins'], manifest['gauge_id']),
        origins=arrays['origins'],forecast_time=arrays['forecast_time'],
        start_utc=manifest['calendar_start_utc'],end_exclusive_utc=manifest['calendar_end_exclusive_utc'],
        labels=arrays['labels'],label_mask=arrays['label_mask'],predictions=arrays['predictions'][0],
        prediction_mask=comparison,comparison_mask=comparison,
        expected_comparison_mask_sha256=digest(comparison.tolist()),multiplicities=draws.multiplicities[0],
        identity=ScoreIdentity(panel['architecture'],f"{panel['architecture']}-{panel['input_arm']}-L{panel['history_hours']}",
            manifest['gauge_id'],schema,panel['history_hours'],manifest['fold'],panel['seed']),
        input_cohort_digest=manifest['cohort_admission_digest'],label_policy_digest=digest({'policy':PRIMARY_APPROVED}),
        label_source_digest=manifest['label_source_identity'])
    actual = result['primary'][0,0]
    require(expected['primary_score'] is None and np.isnan(actual)
            or expected['primary_score'] is not None and np.isclose(actual,expected['primary_score'],rtol=TOLERANCE,atol=TOLERANCE),
            'Fixed scalar weighted-reference primary qualification failed')
    differences = []
    for h, row in enumerate(expected['leads']):
        value = result['per_lead_nse'][0,0,h]
        reference_nse = row['nse'] if row['primary_eligible'] and result['baseline_lead_eligible'][h] else None
        require(reference_nse is None and np.isnan(value) or reference_nse is not None
                and np.isclose(value,reference_nse,rtol=TOLERANCE,atol=TOLERANCE),
                'Fixed scalar weighted-reference lead qualification failed')
        require(int(result['distinct_target_hours'][0,h]) == row['distinct_target_hours']
                and int(result['distinct_utc_dates'][0,h]) == row['distinct_utc_dates']
                and int(result['weighted_label_count'][0,h]) == row['weighted_label_count'],
                'Fixed scalar weighted-reference support qualification failed')
        if reference_nse is not None: differences.append(abs(float(value)-reference_nse))
    return dict(gauge_id=manifest['gauge_id'],fold=manifest['fold'],job_id=panel['job_id'],draw_index=0,panel_index=0,
                primary_defined=expected['primary_score'] is not None,
                maximum_absolute_lead_nse_difference=max(differences,default=0.),
                original_coordinate_support_matches=True,tolerance_relative=TOLERANCE,tolerance_absolute=TOLERANCE)


def _lead_bands(draw_scores, point_scores, family, *, mode):
    folds = FOLDS[:3] if mode == 'cv' else FOLDS[3:]
    seeds = (17,) if mode == 'cv' else SEEDS
    arms = (('full',168),('q_only',168)) if mode == 'cv' else (('full',168),('q_only',168),('full',24))
    rows, draws, points = [], [], []
    for gauge in GAUGES:
        for architecture in ARCHITECTURES:
            for arm,hours in arms:
                keys = [(fold,gauge,architecture,arm,hours,seed) for fold in folds for seed in seeds]
                require(all(k in draw_scores and k in point_scores for k in keys), 'Complete lead-band fold/seed panels required')
                value = np.sum([draw_scores[k]/len(keys) for k in keys],axis=0)
                point = np.sum([point_scores[k]/len(keys) for k in keys],axis=0)
                require(value.shape == (2000,18) and point.shape == (18,), 'Bounded all-H18 lead-band geometry required')
                draws.append(value);points.append(point)
                rows.append(dict(gauge_id=gauge,architecture=architecture,input_arm=arm,history_hours=hours))
    values = np.stack(draws,axis=1);baseline=np.stack(points)
    eligible = family['status'] == 'AVAILABLE_CONDITIONAL_POINTWISE'
    interval = np.percentile(values[family['joint_valid_mask']],(2.5,97.5),axis=0,method='linear') if eligible else None
    for i,row in enumerate(rows):
        row['leads'] = [dict(lead_hours=h+1,mean_nse=float(baseline[i,h]) if np.isfinite(baseline[i,h]) else None,
            lower=None if interval is None else float(interval[0,i,h]),
            upper=None if interval is None else float(interval[1,i,h])) for h in range(18)]
    return dict(series=rows,draws=values,points=baseline,
                interpretation='POINTWISE_EXPLORATORY_MEAN_SEED_OR_FOLD_NSE_NOT_PREDICTION_INTERVALS',
                same_joint_family_valid_draw_pool=True,seed_dispersion_is_interval=False)


def run_uncertainty(plan, guard):
    guard.check()
    require(plan['uncertainty'] == dict(draws=2000,master_seed=20261102,block_hours=[168,336,672],
        fold_order=list(FOLDS),stream_mapping='SeedSequence(master).spawn(12); block-major/fold-minor; one uint64 child seed',
        same_draws_for_all_gauges_models_seeds=True), 'Frozen uncertainty execution policy differs')
    original = json.loads(guard.read(Path(plan['original_plan_path']),8*1024**2))
    require(original['plan_sha256'] == plan['original_plan_sha256']
            and digest({k:v for k,v in original.items() if k!='plan_sha256'}) == original['plan_sha256'],
            'Original frozen campaign plan identity differs')
    audit = json.loads(guard.read(guard.run/'AUDIT.json',16*1024**2))
    require(audit['status']=='ALL180_PREDICTIONS_INDEPENDENTLY_AUDITED'
            and audit['plan_sha256']==plan['plan_sha256'] and audit['groups_total']==12
            and audit['protected_assessment_access'] is False and audit['model_calls']==0,
            'Complete independent audit required before uncertainty')
    groups = {(r['fold'],r['gauge_id']):r for r in audit['groups']}
    require(len(audit['groups'])==len(groups)==12 and set(groups)=={(f,g) for f in FOLDS for g in GAUGES},
            'All12 unique fold/gauge groups required')
    blocks=[]
    for block in BLOCKS:
        primary,points,lead_draws,lead_points={}, {}, {}, {}
        streams={};reference_checks=[];baseline_checks=[]
        for fold in FOLDS:
            draws=None
            for gauge in GAUGES:
                guard.check();arrays,manifest,group_audit=_load_group(plan,original,guard,groups[fold,gauge])
                if draws is None:
                    draws=make_draws(arrays['origins'],start_utc=manifest['calendar_start_utc'],
                        end_exclusive_utc=manifest['calendar_end_exclusive_utc'],fold=fold,block_hours=block,replicates=2000)
                    streams[fold]=guard.write(f'uncertainty/block-{block}/{fold}-draws.json',_json(draws.receipt))
                else:
                    require(np.array_equal(draws.origins,arrays['origins']), 'Gauge calendars differ; paired draws cannot change')
                present=np.broadcast_to(arrays['comparison_mask'][None,:,None],arrays['predictions'].shape)
                result=evaluate_matched(draws,labels=arrays['labels'],label_mask=arrays['label_mask'],
                    predictions=arrays['predictions'],prediction_mask=present,comparison_mask=arrays['comparison_mask'],
                    expected_comparison_mask_sha256=manifest['comparison_mask_sha256'])
                baseline_checks.append(dict(fold=fold,gauge_id=gauge,**_baseline_check(result,group_audit)))
                if gauge==GAUGES[0]:
                    reference_checks.append(_reference_check(draws,arrays,manifest,result))
                for i,panel in enumerate(manifest['panels']):
                    key=(fold,gauge,panel['architecture'],panel['input_arm'],panel['history_hours'],panel['seed'])
                    require(key not in primary, 'Duplicate uncertainty panel')
                    primary[key]=result['primary'][:,i].copy();points[key]=float(result['baseline_primary'][i])
                    lead_draws[key]=result['per_lead_nse'][:,i,:].copy();lead_points[key]=result['baseline_per_lead_nse'][i].copy()
                del arrays,manifest,group_audit,result,present
                guard.check();guard.save()
            del draws
        require(len(reference_checks)==4 and len(baseline_checks)==12, 'All fixed qualifications must precede interval publication')
        qualification=guard.write(f'uncertainty/block-{block}/qualification.json',_json(dict(
            status='ALL_GROUP_BASELINES_AND_FIXED_SCALAR_REFERENCES_MATCH',reference_checks=reference_checks,
            baseline_checks=baseline_checks,model_calls=0,protected_assessment_access=False)))
        families={}
        for mode in ('cv','final'):
            selected={k:v for k,v in primary.items() if (k[0]=='final_development') == (mode=='final')}
            selected_points={k:points[k] for k in selected}
            family=primary_family(selected,selected_points,mode=mode)
            bands=_lead_bands(lead_draws,lead_points,family,mode=mode)
            buffer=io.BytesIO();np.savez_compressed(buffer,joint_valid_mask=family['joint_valid_mask'],
                contrast_draws=family['contrast_draws'],per_lead_mean_nse_draws=bands['draws'],
                per_lead_mean_nse_points=bands['points'])
            draw_artifact=guard.write(f'uncertainty/block-{block}/{mode}-draws.npz',buffer.getvalue())
            body={k:v for k,v in family.items() if k not in ('joint_valid_mask','contrast_draws')}
            body.update(plan_sha256=plan['plan_sha256'],original_plan_sha256=original['plan_sha256'],
                protocol_sha256=plan['protocol_sha256'],block_hours=block,primary_block=block==336,
                draw_streams=streams,scalar_qualification=qualification,draw_artifact=draw_artifact,
                per_lead_mean_nse={k:v for k,v in bands.items() if k not in ('draws','points')},
                uncertainty_is_not_interannual_or_random_catchment_inference=True,
                negative_predictions_clipped=False,protected_assessment_access=False)
            report=guard.write(f'uncertainty/block-{block}/{mode}.json',_json(body))
            families[mode]=dict(status=family['status'],contrasts=family['contrast_count'],
                jointly_valid_draws=family['jointly_valid_draws'],jointly_invalid_draws=family['jointly_invalid_draws'],
                report=report,draws=draw_artifact)
            del family,bands,buffer,body
            guard.check();guard.save()
        blocks.append(dict(block_hours=block,primary_block=block==336,draw_streams=streams,
                           qualification=qualification,families=families))
        del primary,points,lead_draws,lead_points
    result=dict(schema='hydra-primary-development-uncertainty-stage/v1',status='COMPLETED_PREDECLARED_DEVELOPMENT_UNCERTAINTY',
        plan_sha256=plan['plan_sha256'],original_plan_sha256=original['plan_sha256'],protocol_sha256=plan['protocol_sha256'],
        block_hours=list(BLOCKS),primary_block_hours=336,replicates=2000,master_seed=20261102,blocks=blocks,
        fixed_scalar_reference_checks=12,baseline_panel_checks=540,model_calls=0,optimizer_updates=0,
        protected_assessment_access=False,paid_cloud_jobs=0,raw_source_values_read=False,
        interpretation='CONDITIONAL_WITHIN_YEAR_POINTWISE_EXPLORATORY_NOT_FAMILYWISE_OR_PREDICTION_INTERVALS',
        all_three_seed_metrics_retained=True,statistical_model_selection_performed=False)
    guard.write('UNCERTAINTY.json',_json(result));guard.save();return result
