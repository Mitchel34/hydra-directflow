"""Authenticated saved-prediction preparation and independent physical metrics.

Only the already-admitted development labels are loaded. Outside that fixed
cohort, calendar values stay NaN with false masks; this is not a full observed
reference calendar, target imputation, model inference or assessment access.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from .campaign_data import load_outer_labels
from .contract import ARCHITECTURES, GAUGE_IDS
from .corpus_dataset import _read
from .local_data import GAUGES, bounded_npy
from .quality import PRIMARY_APPROVED
from .reference import ASSESSMENT_START, digest, require
from .scoring import validate_score_coordinates
from .trainer_bundle import _array_digest

LABEL_READ_RESERVE = 64 * 1024**2
MAX_GROUP_ARRAY_BYTES = 512 * 1024**2
MAX_RSS_BYTES = 3 * 1024**3
TOLERANCE = 1e-10
LEADS = tuple(range(1, 19))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _path(run, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
            and all(part not in ('.', '..') for part in Path(relative).parts), 'Safe relative artifact path required')
    path = run / relative
    require(path == path.resolve() and path.is_relative_to(run), 'Artifact escaped the exact run directory')
    return path


class _Reader:
    def __init__(self, run, check, read_budget):
        self.run, self.check, self.budget = run, check, read_budget
        self.bytes = self.files = 0

    def read(self, relative, maximum, *, expected_sha=None, expected_bytes=None):
        self.check()
        path = _path(self.run, relative)
        size = path.lstat().st_size
        require(0 < size <= maximum and (expected_bytes is None or size == expected_bytes), 'Artifact byte size differs')
        self.budget(size)  # Before opening a file body; failed reads retain charge.
        self.bytes += size
        self.files += 1
        raw = _read(path, maximum)
        require(expected_sha is None or _sha(raw) == expected_sha, 'Artifact SHA differs')
        self.check()
        return raw

    def descriptor(self, descriptor, maximum):
        require(isinstance(descriptor, dict) and set(descriptor) == {'path', 'bytes', 'sha256'}
                and type(descriptor['bytes']) is int and 0 < descriptor['bytes'] <= maximum,
                'Exact bounded artifact descriptor required')
        return self.read(descriptor['path'], maximum, expected_sha=descriptor['sha256'], expected_bytes=descriptor['bytes'])


def _identity(plan, job, receipt):
    expected = dict(plan_sha256=plan['plan_sha256'], protocol_sha256=plan['protocol_sha256'],
                    source_plan_sha256=plan['source_plan_sha256'], snapshot_sha256=plan['snapshot_sha256'],
                    implementation_sha256=plan['implementation_sha256'], job_id=job['job_id'], job_sha256=digest(job))
    require(all(receipt.get(k) == v for k, v in expected.items())
            and receipt.get('status') == 'VERIFIED_TRAIN_EVAL'
            and receipt.get('protected_assessment_access') is False, 'Verified original completion identity required')
    return expected


def _label_manifest(reader, plan, record):
    job, receipt = record['job'], record['receipt']
    relative = str(Path(receipt['receipt_path']).parent / 'outer-label-source.json')
    raw = reader.read(relative, 2 * 1024**2, expected_sha=record.get('label_source_manifest_sha256'))
    manifest = json.loads(raw)
    core = {k: v for k, v in manifest.items() if k not in ('source_identity', 'returned_arrays_sha256', 'construction')}
    require(manifest['source_identity'] == receipt['label_source_identity'] == digest(core),
            'Label source core does not match the completion receipt')
    require(manifest['schema'] == 'hydra-primary-campaign-data/v1' and manifest['operation'] == 'labels'
            and manifest['quality_policy'] == PRIMARY_APPROVED
            and manifest['source_plan_sha256'] == plan['source_plan_sha256']
            and manifest['snapshot_sha256'] == plan['snapshot_sha256']
            and all(manifest[k] == job[k] for k in ('fold', 'gauge_id', 'input_arm', 'history_hours')),
            'Label source view, quality or snapshot differs')
    count = job['data_job']['actual_common_cases']['outer_scoring']
    cases = job['data_job']['role_case_ids_sha256']['outer_scoring']
    require(manifest['roles'] == {'outer': {'cases': count, 'case_ids_sha256': cases}}
            and manifest['case_count'] == count and receipt['outer_case_ids_sha256'] == cases
            and manifest['cohort_admission_digest'] == receipt['inference_cohort_admission_digest']
            and manifest['raw_arrays_sha256'] == manifest['returned_arrays_sha256'], 'Label source cohort differs')
    return manifest, _sha(raw)


def _utc(text):
    value = datetime.fromisoformat(text.replace('Z', '+00:00'))
    require(value.tzinfo is not None and value.utcoffset().total_seconds() == 0
            and value.minute == value.second == value.microsecond == 0, 'Exact UTC calendar bounds required')
    return int(value.timestamp())


def _scalar_json(raw):
    require(len(raw) <= 256 * 1024, 'Prediction metadata exceeds bound')
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    require(version in ((1, 0), (2, 0)), 'Supported scalar metadata format required')
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(stream)
    require(shape == () and not fortran and dtype.kind == 'U' and dtype.itemsize <= 256 * 1024,
            'Bounded scalar text metadata required')
    return json.loads(bounded_npy(raw, shape=(), dtype=dtype).item())


def _predictions(reader, plan, record, start, calendar_end):
    job, receipt = record['job'], record['receipt']
    bindings = _identity(plan, job, receipt)
    descriptor = receipt['artifacts']['predictions']
    require(descriptor['sha256'] == receipt['hashes']['predictions'], 'Prediction descriptor hash differs')
    raw = reader.descriptor(descriptor, 64 * 1024**2)
    n = job['data_job']['actual_common_cases']['outer_scoring']
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        expected = {'prediction_q.npy', 'case_id.npy', 'origin_time.npy', 'forecast_time.npy', 'site_id.npy', 'metadata_json.npy'}
        entries = archive.infolist()
        require(len(entries) == len(expected) and {e.filename for e in entries} == expected
                and sum(e.file_size for e in entries) <= 64 * 1024**2, 'Prediction archive inventory or expansion differs')
        meta = _scalar_json(archive.read('metadata_json.npy'))
        for key, value in bindings.items():
            require(meta.get(key) == value, 'Prediction metadata identity differs')
        for key in ('training_source_identity', 'training_cohort_admission_digest', 'inference_source_identity',
                    'inference_cohort_admission_digest', 'selected_best_epoch'):
            require(meta.get(key) == receipt.get(key), 'Prediction source/checkpoint identity differs')
        require(meta['checkpoint_sha256'] == receipt['hashes']['checkpoint']
                and meta['case_ids_sha256'] == receipt['outer_case_ids_sha256'] and meta['case_count'] == n
                and meta['prediction_units'] == 'm3/s' and meta['label_values_opened'] is False
                and meta['posthoc_clipping'] is False and meta['best_only'] is True
                and meta['optimizer_constructed'] is False and meta['scaler_fitted'] is False,
                'Prediction units, selected weights or label separation differ')
        specs = {'case_id': ((n,), '<U29'), 'origin_time': ((n,), '<i8'),
                 'forecast_time': ((n, 18), '<i8'), 'site_id': ((n,), '<i8')}
        coords = {name: bounded_npy(archive.read(name + '.npy'), shape=shape, dtype=np.dtype(dtype))
                  for name, (shape, dtype) in specs.items()}
        ids, origin, _ = validate_score_coordinates(case_ids=coords['case_id'], origins=coords['origin_time'],
                                                   forecast_time=coords['forecast_time'], gauge_id=job['gauge_id'])
        require(np.all((origin >= start) & (origin < calendar_end))
                and np.all(coords['site_id'] == GAUGES.index(job['gauge_id']))
                and digest(list(ids)) == receipt['outer_case_ids_sha256'], 'Prediction clock/gauge membership differs')
        # Outcomes are decoded only after archive identity and all coordinates.
        values = bounded_npy(archive.read('prediction_q.npy'), shape=(n, 18), dtype=np.dtype('<f8'))
        require(np.isfinite(values).all(), 'Missing or nonfinite requested prediction')
    return coords, values


def independent_panel(labels, mask, predictions, forecast_time):
    """Numerical recomputation independent of the production metric functions."""
    y, m, p, clock = map(np.asarray, (labels, mask, predictions, forecast_time))
    require(y.ndim == 2 and y.shape[1] == 18 and y.dtype == p.dtype == np.dtype('<f8')
            and p.shape == m.shape == clock.shape == y.shape and m.dtype == np.dtype(bool)
            and clock.dtype.kind in 'iu', 'Physical metric arrays differ')
    require(np.isfinite(p).all() and np.isfinite(y[m]).all() and np.all(y[m] >= 0),
            'Finite predictions and admissible original references required')
    leads = []
    for i in range(18):
        truth, predicted = y[m[:, i], i], p[m[:, i], i]
        times = clock[m[:, i], i]
        count = len(truth)
        require(count > 0, 'No admitted support in requested panel')
        error = predicted - truth
        mse = float(np.mean(error * error, dtype=np.float64))
        center = float(np.mean(truth, dtype=np.float64))
        denominator = float(np.sum((truth - center) ** 2, dtype=np.float64))
        nse = None if np.all(truth == truth[0]) or denominator == 0 else 1 - float(np.sum(error * error, dtype=np.float64)) / denominator
        negative = int(np.count_nonzero(predicted < 0))
        days = int(len(np.unique(times // 86400)))
        unique = int(len(np.unique(times)))
        leads.append(dict(lead_hours=i + 1, n=count, n_requested=len(y), physical_mse=mse, rmse=math.sqrt(mse),
                          mae=float(np.mean(np.abs(error), dtype=np.float64)), bias=float(np.mean(error, dtype=np.float64)),
                          nse=nse, negative_predictions_scored=negative, negative_frequency=negative / count,
                          distinct_target_hours=unique, distinct_utc_dates=days,
                          primary_eligible=unique >= 168 and days >= 28 and nse is not None))
    primary = math.fsum(x['nse'] / 18 for x in leads) if all(x['primary_eligible'] for x in leads) else None
    return dict(primary_score=primary, leads=leads)


def _compare(computed, recorded):
    exact = ('lead_hours', 'n', 'n_requested', 'negative_predictions_scored', 'distinct_target_hours',
             'distinct_utc_dates', 'primary_eligible')
    numeric = ('physical_mse', 'rmse', 'mae', 'bias', 'nse', 'negative_frequency')
    require(len(recorded['leads']) == 18, 'Recorded H18 metrics are incomplete')
    largest = 0.
    for current, old in zip(computed['leads'], recorded['leads']):
        require(all(current[k] == old[k] for k in exact), 'Independent metric support/count differs')
        for key in numeric:
            a, b = current[key], old[key]
            require(a is None and b is None or a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
                    and math.isclose(a, b, rel_tol=TOLERANCE, abs_tol=TOLERANCE), 'Independent physical metric differs: ' + key)
            if a is not None:
                largest = max(largest, abs(a - b))
    a, b = computed['primary_score'], recorded['primary_score']
    require(a is None and b is None or a is not None and b is not None
            and math.isclose(a, b, rel_tol=TOLERANCE, abs_tol=TOLERANCE), 'Independent primary metric differs')
    return largest


def build_group(plan, run, corpus_root, records, fold, gauge, *, check=lambda: None, read_budget=lambda n: None):
    """Prepare one complete paired fold/gauge group with no model calls.

    Every artifact read precharges actual bytes. The one canonical label call
    precharges a conservative 64 MiB and has that exact internal read bound.
    Reported logical reservation and actual source telemetry remain distinct.
    """
    require(callable(check) and callable(read_budget) and gauge in GAUGE_IDS
            and fold in ('cv_1', 'cv_2', 'cv_3', 'final_development'), 'Explicit development group required')
    run = Path(run).absolute()
    require(run == run.resolve() and str(run) == plan['run_directory']
            and str(Path(corpus_root).absolute()) == plan['corpus_root'] and plan['protected_assessment_access'] is False,
            'Exact original development run/source paths required')
    selected = sorted((r for r in records if r['job']['fold'] == fold and r['job']['gauge_id'] == gauge), key=lambda r: r['job']['order'])
    arms = [('full', 168), ('q_only', 168), ('full', 24)] if fold == 'final_development' else [('full', 168), ('q_only', 168)]
    seeds = (17, 29, 43) if fold == 'final_development' else (17,)
    expected = {(a, arm, length, seed) for a in ARCHITECTURES for arm, length in arms for seed in seeds}
    require(len(selected) == len(expected) and {(r['job']['architecture'], r['job']['input_arm'], r['job']['history_hours'], r['job']['seed']) for r in selected} == expected,
            'Complete, unique paired group inventory required')
    reader = _Reader(run, check, read_budget)
    sidecars = []
    for record in selected:
        job = record['job']
        require(job == plan['jobs'][job['order']], 'Record differs from the immutable plan job')
        _identity(plan, job, record['receipt'])
        sidecars.append(_label_manifest(reader, plan, record))
    canonical_index = next(i for i, r in enumerate(selected) if r['job']['input_arm'] == 'full' and r['job']['history_hours'] == 168)
    canonical = selected[canonical_index]
    manifest = sidecars[canonical_index][0]
    window = manifest['role_time_bounds']['outer']
    start, end = _utc(window['start_utc']), _utc(window['end_utc']) - 18 * 3600
    require(start < end and (end - start) // 3600 <= 8784 and end - 3600 + 18 * 3600 < ASSESSMENT_START,
            'Bounded H18-eligible development calendar required')
    require(all(m['role_time_bounds']['outer'] == window and m['raw_arrays_sha256'] == manifest['raw_arrays_sha256']
                and m['roles'] == manifest['roles'] and m['cohort_admission_digest'] == manifest['cohort_admission_digest']
                for m, _ in sidecars), 'Matched views do not share labels, masks and coordinates')
    origins = np.arange(start, end, 3600, dtype=np.int64)
    future = origins[:, None] + np.arange(1, 19, dtype=np.int64) * 3600
    size = len(origins)
    estimated = size * 18 * 8 * (len(selected) + 2) + size * 18 * 2 + size * 16
    require(estimated <= MAX_GROUP_ARRAY_BYTES, 'Group array allocation exceeds bound')
    check()
    predictions = np.full((len(selected), size, 18), np.nan, dtype=np.float64)
    coordinates = None
    positions = None
    panels = []
    metric_panels = []
    for index, record in enumerate(selected):
        job, receipt = record['job'], record['receipt']
        coords, values = _predictions(reader, plan, record, start, end)
        if coordinates is None:
            coordinates = coords
            positions = ((coords['origin_time'] - start) // 3600).astype(np.int64)
        else:
            require(all(np.array_equal(coords[k], coordinates[k]) for k in coordinates), 'Paired prediction coordinates differ')
        predictions[index, positions] = values
        require(receipt['artifacts']['metrics']['sha256'] == receipt['hashes']['metrics'], 'Metric descriptor differs')
        metrics = json.loads(reader.descriptor(receipt['artifacts']['metrics'], 2 * 1024**2))
        bindings = _identity(plan, job, receipt)
        require(all(metrics.get(k) == v for k, v in bindings.items()) and metrics['checkpoint_sha256'] == receipt['hashes']['checkpoint']
                and metrics['predictions_sha256'] == receipt['hashes']['predictions'] and metrics['outer_case_ids_sha256'] == receipt['outer_case_ids_sha256'],
                'Metric artifact parent identity differs')
        panel = metrics['panel']
        require(panel['case_ids_sha256'] == receipt['outer_case_ids_sha256'] and panel['input_cohort_digest'] == receipt['inference_cohort_admission_digest']
                and panel['label_source_digest'] == receipt['label_source_identity'] and panel['label_policy_digest'] == digest({'policy': PRIMARY_APPROVED})
                and panel['negative_predictions_clipped'] is False and panel['protected_assessment_scores'] == 0,
                'Metric label/admission policy differs')
        panels.append({k: job[k] for k in ('job_id', 'architecture', 'input_arm', 'history_hours', 'seed', 'device')})
        metric_panels.append(panel)
        check()
    read_budget(LABEL_READ_RESERVE)
    label_bundle = load_outer_labels(corpus_root, canonical['job']['data_job'],
        expected_outer_cohort_digest=canonical['receipt']['inference_cohort_admission_digest'],
        expected_case_ids_sha256=canonical['receipt']['outer_case_ids_sha256'], max_bytes=LABEL_READ_RESERVE,
        max_read_bytes=LABEL_READ_RESERVE, max_rss_bytes=MAX_RSS_BYTES, check=check)
    la = label_bundle['arrays']
    require(label_bundle['source_identity'] == manifest['source_identity']
            and _array_digest(la) == manifest['raw_arrays_sha256'], 'Canonical label bytes/source differ from saved inference labels')
    require(np.array_equal(la['case_id'], coordinates['case_id']) and np.array_equal(la['origin_time'], coordinates['origin_time'])
            and np.array_equal(la['target_time'], coordinates['forecast_time']) and np.array_equal(la['site_id'], coordinates['site_id'])
            and la['target_mask'].dtype == np.dtype(bool) and la['target_mask'].all(), 'Exact common complete target support required')
    labels = np.full((size, 18), np.nan, dtype=np.float64)
    label_mask = np.zeros((size, 18), dtype=bool)
    comparison = np.zeros(size, dtype=bool)
    labels[positions] = la['target']
    label_mask[positions] = la['target_mask']
    comparison[positions] = True  # Derived from authenticated case coordinates, never predictions/errors.
    audits = []
    for index, panel in enumerate(metric_panels):
        recomputed = independent_panel(la['target'], la['target_mask'], predictions[index, positions], la['target_time'])
        largest = _compare(recomputed, panel)
        audits.append(dict(job_id=panels[index]['job_id'], maximum_absolute_numeric_difference=largest,
                           independently_recomputed=recomputed))
        check()
    arrays = dict(origins=origins, forecast_time=future, labels=labels, label_mask=label_mask,
                  comparison_mask=comparison, predictions=predictions,
                  panel_ids=np.array([p['job_id'] for p in panels]))
    require(sum(a.nbytes for a in arrays.values()) <= MAX_GROUP_ARRAY_BYTES, 'Actual group array byte limit reached')
    for value in arrays.values():
        value.setflags(write=False)
    result_manifest = dict(schema='hydra-postprocessing-development-group/v1', fold=fold, gauge_id=gauge,
        plan_sha256=plan['plan_sha256'], protocol_sha256=plan['protocol_sha256'], source_plan_sha256=plan['source_plan_sha256'],
        snapshot_sha256=plan['snapshot_sha256'], panels=panels, original_outer_window=window,
        calendar_start_utc=datetime.fromtimestamp(start, timezone.utc).isoformat(),
        calendar_end_exclusive_utc=datetime.fromtimestamp(end, timezone.utc).isoformat(),
        calendar_hours=size, admitted_origins=int(comparison.sum()), label_scope='ADMITTED_COMMON_L168_TARGETS_ONLY_NOT_FULL_REFERENCE_CALENDAR',
        quality_policy=PRIMARY_APPROVED, target_imputation=False, negative_predictions_clipped=False,
        label_source_identity=label_bundle['source_identity'], label_arrays_sha256=manifest['raw_arrays_sha256'],
        cohort_admission_digest=manifest['cohort_admission_digest'], case_ids_sha256=canonical['receipt']['outer_case_ids_sha256'],
        comparison_mask_sha256=digest(comparison.tolist()),
        comparison_mask_array_sha256=_array_digest({'comparison_mask': comparison}), group_arrays_sha256=_array_digest(arrays),
        outer_label_sidecar_sha256={r['job']['job_id']:sha for r, (_, sha) in zip(selected, sidecars)},
        prediction_sha256={r['job']['job_id']:r['receipt']['hashes']['predictions'] for r in selected},
        label_source_bytes_read=label_bundle['manifest']['construction']['source_bytes_read'], label_read_bytes_reserved=LABEL_READ_RESERVE,
        artifact_bytes_read=reader.bytes, artifact_files_read=reader.files, dense_array_bytes=sum(a.nbytes for a in arrays.values()),
        model_calls=0, optimizer_updates=0, protected_assessment_access=False)
    result_manifest['content_sha256'] = digest(result_manifest)
    audit = dict(status='ALL_RAW_PREDICTIONS_AND_PHYSICAL_METRICS_MATCH', panels=len(audits), leads=len(audits) * 18,
                 tolerance_absolute=TOLERANCE, tolerance_relative=TOLERANCE, results=audits,
                 original_coordinates_and_masks_preserved=True, protected_assessment_access=False)
    return arrays, result_manifest, audit
