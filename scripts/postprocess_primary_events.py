#!/usr/bin/env python3
"""Bounded event orchestration using saved forecasts and owned resident references.

No network, cache writes, model calls, threshold fitting or protected assessment.
The parent authenticates the postprocessing plan and owns its resource guard.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np

from audit_coverage_regimes import reference_month
from hydra_directflow.cache import DerivedCache
from hydra_directflow.contract import GAUGE_IDS
from hydra_directflow.coverage_analysis import reference_arrays
from hydra_directflow.local_data import bounded_npy
from hydra_directflow.post_events import summarize_events, summarize_regimes, validate_catalog_row
from hydra_directflow.quality import PRIMARY_APPROVED
from hydra_directflow.reference import ASSESSMENT_START, digest, require
from hydra_directflow.trainer_bundle import _array_digest
from hydra_directflow.uncertainty import validate_origin_calendar

FOLDS = ('cv_1', 'cv_2', 'cv_3', 'final_development')
GAUGES = tuple(GAUGE_IDS)
CATALOG = ROOT / 'reports/development-coverage-regimes-v2.json'
CATALOG_SHA = '8e82f613165a5d452c74a30783b02e1c0ae5fb5d93dbe2e338d0010c215e8c0a'
REFERENCE_ROOT = Path.home() / '.codex/hydra-directflow-runtime/postprocessing-usgs-reference-v1'
MAX_GROUP_BYTES = 256 * 1024**2
MAX_REFERENCE_ROWS = 3 * 49 * 31 * 24


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _resident(path, maximum, *, directory=False):
    path = Path(path)
    require(path == path.resolve(), 'Reference path cannot contain a symlink')
    info = path.lstat()
    require((stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
            and not getattr(info, 'st_flags', 0) & 0x40000000,
            'Only resident regular reference files/directories may be used')
    if not directory:
        require(0 < info.st_size <= maximum, 'Reference file size exceeds its bound')
    return info


def _months():
    return [f'{year:04d}-{month:02d}' for year in range(2020, 2025) for month in range(1, 13)
            if '2020-07' <= f'{year:04d}-{month:02d}' <= '2024-07']


def load_catalog(guard):
    raw = guard.read(CATALOG, 16 * 1024**2)
    require(_sha(raw) == CATALOG_SHA, 'Frozen event/threshold catalog bytes differ')
    catalog = json.loads(raw)
    require(catalog['coverage_content_sha256'] == digest({k:v for k,v in catalog.items()
            if k not in ('coverage_content_sha256', 'runtime')}), 'Catalog content identity differs')
    rows = catalog['fold_regime_and_event_diagnostics']
    require(len(rows) == len(FOLDS) * len(GAUGES)
            and {(r['fold'], r['gauge_id']) for r in rows} == {(f,g) for f in FOLDS for g in GAUGES},
            'Complete frozen gauge/fold event inventory required')
    for row in rows:
        validate_catalog_row(row)
    return catalog


def checked_reference_month(cache, identity, guard):
    """Precharge all helper reads, then use its existing July-boundary guard."""
    month, key = identity['month'], identity['cache_key']
    require(month in _months() and isinstance(key, str) and re.fullmatch(r'[0-9a-f]{64}', key),
            'Only the approved 2020-07 through 2024-07 reference months are allowed')
    directory = cache.path / 'entries' / key
    _resident(cache.path, 0, directory=True)
    _resident(cache.path / 'entries', 0, directory=True)
    _resident(directory, 0, directory=True)
    commit_path = directory / 'COMMIT.json'
    commit_size = _resident(commit_path, 128 * 1024).st_size
    raw = guard.read(commit_path, 128 * 1024)
    require(_sha(raw) == identity['commit_sha256'], 'Reference commitment differs from frozen catalog')
    commit = json.loads(raw)
    require(commit['kind'] == 'usgs_month' and commit['key'] == key
            and commit['identity']['month'] == month and set(commit['payloads']) == {'arrays', 'metadata'},
            'Reference monthly cache commitment differs')
    expected_files = {'COMMIT.json'}
    payload_bytes = 0
    for label, suffix, maximum in (('arrays', '.npz', 16 * 1024**2), ('metadata', '.json', 4 * 1024**2)):
        descriptor = commit['payloads'][label]
        require(re.fullmatch(r'[0-9a-f]{64}', descriptor['sha256'])
                and descriptor['name'] == descriptor['sha256'] + suffix,
                'Content-addressed reference payload name differs')
        info = _resident(directory / descriptor['name'], maximum)
        require(info.st_size == descriptor['bytes'], 'Reference payload size differs')
        expected_files.add(descriptor['name'])
        payload_bytes += info.st_size
    require({p.name for p in directory.iterdir()} == expected_files, 'Reference cache inventory differs')
    # reference_month reads COMMIT before/after DerivedCache._load, which reads
    # it once more and both payloads. Its internal reads are all precharged here.
    reserve = 3 * commit_size + payload_bytes
    guard.read_budget(reserve)
    observations, returned_identity = reference_month(cache, month, key)
    require(returned_identity == identity, 'Reference helper returned a different source identity')
    require(len(observations) <= 3 * 31 * 24
            and all(row.site in GAUGES and row.valid_at < ASSESSMENT_START for row in observations),
            'Reference helper returned protected or out-of-scope observations')
    guard.check()
    return observations, dict(identity=identity, helper_read_bytes_reserved=reserve,
                              preflight_commit_bytes=commit_size, source_files=3)


def build_reference_calendars(observations, *, start, end):
    """Retain original flag order/duplicates and exact-hour absence explicitly."""
    require(type(start) is int and type(end) is int and start < end <= ASSESSMENT_START
            and start % 3600 == end % 3600 == 0 and (end-start)//3600 <= 4*366*24,
            'Bounded development-only reference calendar required')
    require(len(observations) <= MAX_REFERENCE_ROWS and all(row.site in GAUGES for row in observations),
            'Reference observation count or gauge scope differs')
    output = {}
    for gauge in GAUGES:
        source = [row for row in observations if row.site == gauge]
        require(all(row.valid_at < ASSESSMENT_START for row in source), 'Protected reference row forbidden')
        reference = reference_arrays([(row.valid_at, row.value, row.quality_flags) for row in source], start=start, end=end)
        flags = ['null'] * len(reference['times'])
        identifiers = [''] * len(flags)
        present = np.zeros(len(flags), bool)
        for row in source:
            if not start <= row.valid_at < end:
                continue
            index = (row.valid_at-start)//3600
            require(not present[index] and row.valid_at % 3600 == 0, 'Duplicate/nonhourly reference source')
            text = json.dumps(list(row.quality_flags), ensure_ascii=True, separators=(',', ':'))
            require(len(text) <= 4096 and isinstance(row.source_id, str) and 0 < len(row.source_id) <= 2048,
                    'Bounded original reference qualifiers/source identity required')
            flags[index], identifiers[index], present[index] = text, row.source_id, True
        reference.update(quality_flags_json=np.asarray(flags), source_id=np.asarray(identifiers), row_present=present)
        for value in reference.values():
            value.setflags(write=False)
        output[gauge] = reference
    return output


def load_references(catalog, guard):
    require(REFERENCE_ROOT == REFERENCE_ROOT.resolve(), 'Stable approved reference root required')
    identities = sorted((item for item in catalog['monthly_reference_identities'] if item['month'] in _months()), key=lambda x:x['month'])
    require([x['month'] for x in identities] == _months(), 'All49 approved monthly identities required exactly once')
    cache = DerivedCache(REFERENCE_ROOT)
    observations, receipts = [], []
    for identity in identities:
        rows, receipt = checked_reference_month(cache, identity, guard)
        observations.extend(rows)
        require(len(observations) <= MAX_REFERENCE_ROWS, 'Cumulative reference observation bound exceeded')
        receipts.append(receipt)
    bounds = [validate_catalog_row(row) for row in catalog['fold_regime_and_event_diagnostics']]
    start, end = min(x[0] for x in bounds), max(x[1] for x in bounds)
    references = build_reference_calendars(observations, start=start, end=end)
    guard.check()
    return references, receipts


def _descriptor(guard, descriptor, maximum):
    require(isinstance(descriptor, dict) and set(descriptor) == {'path', 'bytes', 'sha256'}
            and isinstance(descriptor['path'], str) and not Path(descriptor['path']).is_absolute()
            and all(p not in ('.','..') for p in Path(descriptor['path']).parts), 'Safe postprocessing artifact descriptor required')
    path = guard.run / descriptor['path']
    require(path == path.resolve() and path.is_relative_to(guard.run), 'Postprocessing artifact escaped run')
    raw = guard.read(path, maximum)
    require(len(raw) == descriptor['bytes'] and _sha(raw) == descriptor['sha256'], 'Postprocessing artifact bytes changed')
    return raw


def load_group(plan, guard, record):
    name, fold, gauge = record['group'], record['fold'], record['gauge_id']
    require(fold in FOLDS and gauge in GAUGES and name == f'{fold}__{gauge}'
            and record['plan_sha256'] == plan['plan_sha256']
            and set(record['artifacts']) == {'arrays','manifest','audit'}, 'Audited group identity differs')
    for kind, item in record['artifacts'].items():
        suffix = 'npz' if kind == 'arrays' else 'json'
        require(item['path'] == f'groups/{name}/{kind}.{suffix}', 'Exact audited group paths required')
    committed = json.loads(guard.read(guard.run / 'groups' / name / 'COMMIT.json', 2 * 1024**2))
    require(committed == record, 'Audit entry differs from the immutable group commit')
    manifest = json.loads(_descriptor(guard, record['artifacts']['manifest'], 4 * 1024**2))
    audit = json.loads(_descriptor(guard, record['artifacts']['audit'], 8 * 1024**2))
    require(manifest['schema'] == 'hydra-postprocessing-development-group/v1'
            and manifest['fold'] == fold and manifest['gauge_id'] == gauge
            and manifest['plan_sha256'] == plan['original_plan_sha256']
            and manifest['protocol_sha256'] == plan['protocol_sha256']
            and manifest['protected_assessment_access'] is False and manifest['target_imputation'] is False
            and manifest['negative_predictions_clipped'] is False and manifest['quality_policy'] == PRIMARY_APPROVED
            and manifest['content_sha256'] == digest({k:v for k,v in manifest.items() if k != 'content_sha256'})
            and audit['status'] == 'ALL_RAW_PREDICTIONS_AND_PHYSICAL_METRICS_MATCH'
            and audit['protected_assessment_access'] is False, 'Authenticated prediction audit required')
    raw = _descriptor(guard, record['artifacts']['arrays'], MAX_GROUP_BYTES)
    n, panels = manifest['calendar_hours'], manifest['panels']
    p = len(panels)
    require(type(n) is int and 0 < n <= 8784 and 1 <= p <= 36, 'Bounded postprocessing group geometry required')
    ids = np.array([x['job_id'] for x in panels])
    specs = {'origins': ((n,), np.dtype('<i8')), 'forecast_time': ((n,18), np.dtype('<i8')),
             'comparison_mask': ((n,), np.dtype(bool)), 'panel_ids': ((p,), ids.dtype),
             'labels': ((n,18), np.dtype('<f8')), 'label_mask': ((n,18), np.dtype(bool)),
             'predictions': ((p,n,18), np.dtype('<f8'))}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        require(len(entries) == len(specs) and {x.filename for x in entries} == {x+'.npy' for x in specs}
                and sum(x.file_size for x in entries) <= MAX_GROUP_BYTES, 'Audited array archive differs')
        arrays = {name: bounded_npy(archive.read(name+'.npy'), shape=specs[name][0], dtype=specs[name][1])
                  for name in ('origins','forecast_time','comparison_mask','panel_ids')}
        validate_origin_calendar(arrays['origins'], start_utc=manifest['calendar_start_utc'],
                                 end_exclusive_utc=manifest['calendar_end_exclusive_utc'])
        require(np.array_equal(arrays['forecast_time'], arrays['origins'][:,None]+np.arange(1,19)*3600)
                and np.all(arrays['forecast_time'] < ASSESSMENT_START) and np.array_equal(arrays['panel_ids'], ids)
                and digest(arrays['comparison_mask'].tolist()) == manifest['comparison_mask_sha256'],
                'Audited coordinates or admission identity differs')
        for name in ('labels','label_mask','predictions'):
            arrays[name] = bounded_npy(archive.read(name+'.npy'), shape=specs[name][0], dtype=specs[name][1])
    require(_array_digest(arrays) == manifest['group_arrays_sha256'], 'Audited typed array hash differs')
    mask = arrays['comparison_mask']
    require(int(mask.sum()) == manifest['admitted_origins']
            and np.array_equal(arrays['label_mask'], np.broadcast_to(mask[:,None], (n,18)))
            and np.isnan(arrays['labels'][~mask]).all() and np.isnan(arrays['predictions'][:,~mask]).all()
            and np.isfinite(arrays['labels'][mask]).all() and np.all(arrays['labels'][mask] >= 0)
            and np.isfinite(arrays['predictions'][:,mask]).all(), 'Original common cohort values/masks differ')
    guard.check()
    return arrays, manifest


def corroborate_reference(row, arrays, reference):
    """Check frozen catalog peaks and every matched verification target."""
    start, end = validate_catalog_row(row)
    time = reference['times']
    require(time.ndim == 1 and time.dtype == np.dtype('<i8')
            and np.array_equal(time, np.arange(start, end, 3600, dtype=np.int64)), 'Full outer reference calendar required')
    require(np.array_equal(arrays['origins'], np.arange(start, end-18*3600, 3600, dtype=np.int64)),
            'Audited forecast calendar differs from the frozen event outer year')
    admitted, values = reference['admitted'], reference['values']
    require(admitted.shape == values.shape == time.shape and admitted.dtype == np.dtype(bool)
            and np.isfinite(values[admitted]).all() and (values[admitted] >= 0).all()
            and np.isnan(values[~admitted]).all(), 'Original admitted reference values/masks required')
    common = arrays['comparison_mask']
    positions = ((arrays['forecast_time'][common]-start)//3600).astype(np.int64)
    require(np.all((positions >= 0) & (positions < len(time)))
            and admitted[positions].all() and np.array_equal(values[positions], arrays['labels'][common]),
            'Saved verification targets differ from authenticated hourly references')
    for event in row['observed_high_flow_events']:
        peak = (event['peak_epoch']-start)//3600
        left, right = (event['start_epoch']-start)//3600, (event['end_epoch']-start)//3600
        require(admitted[peak] and values[peak] == event['observed_peak_q_cms']
                and int(admitted[left:right].sum()) == event['reference_approved_hours']
                and right-left == event['reference_requested_hours'], 'Frozen event peak/reference support differs')
    return dict(all_matched_target_cells=int(positions.size), all_catalog_peaks=len(row['observed_high_flow_events']),
                exact_reference_values=True, exact_event_reference_counts=True)


def _npz(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def run_events(plan, guard):
    require(guard.run == Path(plan['run_directory']) and plan['protected_assessment_access'] is False,
            'Bounded development postprocessing guard required')
    audit = json.loads(guard.read(guard.run / 'AUDIT.json', 4 * 1024**2))
    require(audit['plan_sha256'] == plan['plan_sha256'] and audit['status'] == 'ALL180_PREDICTIONS_INDEPENDENTLY_AUDITED'
            and len(audit['groups']) == len(FOLDS)*len(GAUGES)
            and {(r['fold'],r['gauge_id']) for r in audit['groups']} == {(f,g) for f in FOLDS for g in GAUGES},
            'All declared prediction groups must pass before event analysis')
    catalog = load_catalog(guard)
    references, reference_receipts = load_references(catalog, guard)
    by_group = {(r['fold'],r['gauge_id']): r for r in catalog['fold_regime_and_event_diagnostics']}
    outputs = []
    for record in sorted(audit['groups'], key=lambda x:(FOLDS.index(x['fold']), GAUGES.index(x['gauge_id']))):
        guard.check()
        arrays, manifest = load_group(plan, guard, record)
        name, fold, gauge = record['group'], record['fold'], record['gauge_id']
        row = by_group[fold,gauge]
        start, end = validate_catalog_row(row)
        reference = references[gauge]
        selected = (reference['times'] >= start) & (reference['times'] < end)
        reference = {k: v[selected] for k,v in reference.items()}
        corroboration = corroborate_reference(row, arrays, reference)
        common = arrays['comparison_mask']
        args = dict(origins=arrays['origins'][common], forecast_time=arrays['forecast_time'][common],
                    labels=arrays['labels'][common], label_mask=arrays['label_mask'][common],
                    predictions={p['job_id']: arrays['predictions'][i,common] for i,p in enumerate(manifest['panels'])})
        events = summarize_events(row, **args)
        regimes = summarize_regimes(row, **args)
        lineage = dict(schema='hydra-primary-outer-reference-calendar/v1', fold=fold, gauge_id=gauge,
                       plan_sha256=plan['plan_sha256'], original_plan_sha256=plan['original_plan_sha256'],
                       catalog_sha256=CATALOG_SHA, quality_policy=catalog['quality_policy'],
                       source_months=[r['identity'] for r in reference_receipts],
                       full_outer_calendar_hours=len(reference['times']), approved_hours=int(reference['admitted'].sum()),
                       status_counts=dict(Counter(reference['statuses'].tolist())),
                       arrays_sha256=_array_digest(reference), corroboration=corroboration,
                       original_qualifier_order_and_duplicates_retained=True,
                       original_numeric_values_retained_in_bound_monthly_cache=True,
                       targets_filled=False, thresholds_recomputed=False, protected_assessment_access=False)
        for result in (events, regimes):
            result.update(plan_sha256=plan['plan_sha256'], catalog_sha256=CATALOG_SHA,
                          audited_group_arrays_sha256=manifest['group_arrays_sha256'],
                          full_observed_reference_calendar_loaded=True,
                          original_reference_arrays_sha256=lineage['arrays_sha256'])
        artifacts = dict(events=guard.write(f'events/{name}/events.json', _json(events)),
                         regimes=guard.write(f'events/{name}/regimes.json', _json(regimes)),
                         reference=guard.write(f'events/{name}/reference.npz', _npz(reference)),
                         reference_manifest=guard.write(f'events/{name}/reference-manifest.json', _json(lineage)))
        outputs.append(dict(group=name, fold=fold, gauge_id=gauge, artifacts=artifacts))
        del arrays, args, reference, events, regimes
        guard.save()
        print(json.dumps(dict(stage='EVENTS_GROUP_COMPLETE', group=name)), flush=True)
    result = dict(schema='hydra-primary-event-postprocessing/v1', status='ALL12_GROUPS_EVENT_REGIME_REFERENCE_COMPLETE',
                  plan_sha256=plan['plan_sha256'], catalog_sha256=CATALOG_SHA, groups=outputs,
                  reference_months_identities=[r['identity'] for r in reference_receipts],
                  reference_source_files=len(reference_receipts)*3,
                  reference_helper_read_bytes_reserved=sum(r['helper_read_bytes_reserved'] for r in reference_receipts),
                  source_reference_scope='2020-07_THROUGH_2024-07_OWNED_RESIDENT_CACHE_ONLY',
                  thresholds_recomputed=False, event_selection_uses_model_errors=False,
                  model_calls=0, optimizer_updates=0, new_downloads=0, protected_assessment_access=False)
    guard.write('EVENTS.json', _json(result))
    guard.save()
    return result
