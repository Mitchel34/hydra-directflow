#!/usr/bin/env python3
"""Independent, post-commit point-result audit. Never runs models or reads sources.

The CLI requires the already committed POINT_RESULTS artifact SHA. All 36
prediction semantics are rechecked before verification arrays are decoded.
Only committed runtime artifacts are read, plus resident metadata describing
the frozen selection and the original training/source accounting ledgers.
Metrics use scalar math.fsum formulas, not the production scoring module.
The public artifact is an explicit aggregate allowlist; publication is separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import stat
import sys
from urllib.parse import quote
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from hydra_directflow.assessment_runtime import AssessmentRuntime, MODEL_IDS, LIMITS, digest

GAUGES = ('03161000', '03164000', '03479000')
FAMILIES = ('LSTM', 'Transformer', 'GRU_Transformer', 'Mamba2_Transformer')
SEEDS = (17, 29, 43)
START = int(datetime(2024, 7, 31, tzinfo=timezone.utc).timestamp())
END = int(datetime(2025, 7, 31, tzinfo=timezone.utc).timestamp())
HOUR, N, ORIGINAL_UPDATES = 3600, 8760, 1151690
METRICS = ('nse', 'physical_mse', 'rmse', 'mae', 'bias', 'negative_frequency', 'pbias', 'pearson_r')
PANELS = ('primary', 'per_lead_valid_target', 'nwm_matched')
ATOL, RTOL = 1e-10, 1e-12
SCHEMA = 'hydra-assessment-independent-point-audit/v1'
CALENDAR = np.arange(START, END, HOUR, dtype=np.int64)
FUTURE = CALENDAR[:, None] + np.arange(1, 19) * HOUR


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def array_sha(value):
    return sha(np.ascontiguousarray(value).tobytes())


def typed_sha(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        v = arrays[name]
        require(v.dtype.kind in 'fbiuUS', 'Plain typed numerical artifacts required')
        h.update(json.dumps([name, v.dtype.str, v.shape], separators=(',', ':')).encode())
        h.update(v.tobytes(order='C'))
    return h.hexdigest()


def decode_arrays(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        entries = z.infolist()
        require(0 < len(entries) <= 64 and len({e.filename for e in entries}) == len(entries)
                and sum(e.file_size for e in entries) <= 256 * 1024**2,
                'Bounded unique array members required')
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    require(all(a.dtype.kind in 'fbiuUS' and a.ndim <= 4 for a in arrays.values()),
            'Only plain arrays are audited')
    return arrays


def compare(actual, expected, path='result'):
    """Compare an explicitly recomputed projection; never log raw values."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) <= set(actual), 'Missing audit fields: ' + path)
        for k, v in expected.items():
            compare(actual[k], v, path + '.' + k)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'Audit list differs: ' + path)
        for i, v in enumerate(expected):
            compare(actual[i], v, path + '[' + str(i) + ']')
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
                and abs(actual - expected) <= ATOL + RTOL * abs(expected), 'Audit float differs: ' + path)
    else:
        require(type(actual) is type(expected) and actual == expected, 'Audit value differs: ' + path)


def support(times, selected):
    hours = set(map(int, times[selected].ravel()))
    return dict(forecast_target_pairs=int(np.count_nonzero(selected)),
                distinct_target_hours=len(hours), distinct_utc_dates=len({t // 86400 for t in hours}))


def metric_row(truth, prediction, times, selected, lead, *, supported=True):
    info = support(times, selected)
    n = info['forecast_target_pairs']
    row = dict(lead_hours=lead, n=n, **info, metrics_units='PHYSICAL_Q_M3_PER_SECOND',
               support_mask_sha256=array_sha(selected), scored_target_times_sha256=array_sha(times[selected]),
               negative_predictions_scored=0, **{k: None for k in METRICS})
    undefined = {}
    if not supported or not n:
        require(supported or n == 0, 'Unsupported lead has numerical support')
        reason = 'NO_SCORED_PAIRS' if supported else 'STRUCTURALLY_UNAVAILABLE_NWM_SHORT_RANGE_LEAD'
        undefined = dict.fromkeys(METRICS, reason)
    else:
        y, p = list(map(float, truth[selected])), list(map(float, prediction[selected]))
        require(all(math.isfinite(v) for v in y + p), 'Nonfinite scored values cannot be dropped')
        errors = [pred - obs for obs, pred in zip(y, p)]
        sse = math.fsum(e * e for e in errors)
        mean_y, mean_p = math.fsum(y) / n, math.fsum(p) / n
        yc, pc = [v - mean_y for v in y], [v - mean_p for v in p]
        yy, pp = math.fsum(v * v for v in yc), math.fsum(v * v for v in pc)
        total_y = math.fsum(y)
        negatives = sum(v < 0 for v in p)
        row.update(negative_predictions_scored=negatives, physical_mse=sse / n,
                   rmse=math.sqrt(sse / n), mae=math.fsum(abs(e) for e in errors) / n,
                   bias=math.fsum(errors) / n, negative_frequency=negatives / n)
        if n >= 2 and yy > 0:
            row['nse'] = 1 - sse / yy
        else:
            undefined['nse'] = 'FEWER_THAN_TWO_TARGETS' if n < 2 else 'CONSTANT_OBSERVED_TARGETS'
        if total_y > 0:
            row['pbias'] = 100 * math.fsum(errors) / total_y
        else:
            undefined['pbias'] = 'ZERO_OBSERVED_SUM'
        if n >= 2 and yy > 0 and pp > 0:
            correlation = math.fsum(a * b for a, b in zip(yc, pc)) / math.sqrt(yy) / math.sqrt(pp)
            row['pearson_r'] = max(-1., min(1., correlation))
        else:
            undefined['pearson_r'] = 'FEWER_THAN_TWO_PAIRS_OR_CONSTANT_SERIES'
        require(all(row[k] is None or math.isfinite(row[k]) for k in METRICS), 'Metric overflow')
    reasons = []
    if not supported:
        reasons.append('STRUCTURALLY_UNAVAILABLE_NWM_SHORT_RANGE_LEAD')
    if info['distinct_target_hours'] < 168:
        reasons.append('TOO_FEW_DISTINCT_TARGET_HOURS')
    if info['distinct_utc_dates'] < 28:
        reasons.append('TOO_FEW_DISTINCT_UTC_DATES')
    if row['nse'] is None:
        reasons.append(undefined['nse'])
    row.update(undefined=undefined, primary_eligible=not reasons,
               primary_ineligibility=list(dict.fromkeys(reasons)))
    return row


def panel(truth, prediction, mask, *, paired=False):
    rows = [metric_row(truth[:, j], prediction[:, j], FUTURE[:, j], mask[:, j], j + 1,
                       supported=not paired or j < 16) for j in range(18)]
    count = 16 if paired else 18
    unavailable = [dict(lead_hours=r['lead_hours'], reasons=r['primary_ineligibility'])
                   for r in rows[:count] if not r['primary_eligible']]
    score = None if unavailable else math.fsum(r['nse'] for r in rows[:count]) / count
    return dict(equal_lead_nse=score, equal_lead_nse_leads=list(range(1, count + 1)),
                status='UNAVAILABLE' if unavailable else 'DEFINED', equal_lead_nse_unavailable=unavailable,
                support=support(FUTURE, mask), support_mask_sha256=array_sha(mask), leads=rows)


def describe(values):
    count = sum(v is not None for v in values)
    return dict(mean=math.fsum(values) / 3 if count == 3 else None,
                min=min(values) if count == 3 else None, max=max(values) if count == 3 else None,
                defined_seeds=count, total_seeds=3, undefined_seeds_omitted=False)


def aggregate(panels):
    require(len(panels) == 3 and len({p['support_mask_sha256'] for p in panels}) == 1,
            'All three seeds on identical support required')
    p = panels[0]
    return dict(seeds=list(SEEDS), equal_lead_nse=describe([x['equal_lead_nse'] for x in panels]),
                equal_lead_nse_leads=p['equal_lead_nse_leads'], support=p['support'],
                support_mask_sha256=p['support_mask_sha256'],
                leads=[dict(lead_hours=h + 1, n=p['leads'][h]['n'],
                            **{m: describe([x['leads'][h][m] for x in panels]) for m in METRICS}) for h in range(18)],
                interpretation='ARITHMETIC_MEAN_OF_SEED_METRICS_WITH_DESCRIPTIVE_MIN_MAX',
                seed_selection_performed=False, ensemble_predictions_scored=False, formal_confidence_intervals=False)


def audit_gauge(gauge, truth, valid, admitted, predictions, nwm, nwm_valid, reported):
    """Pure independent recomputation on synthetic or previously released arrays."""
    require(gauge in GAUGES and set(predictions) == {(a, s) for a in FAMILIES for s in SEEDS},
            'All four families and all three seeds are required')
    require(set(reported['models']) == set(FAMILIES), 'Reported model roster differs')
    shape = (N, 18)
    require(truth.shape == valid.shape == nwm.shape == nwm_valid.shape == shape and admitted.shape == (N,)
            and truth.dtype.kind == nwm.dtype.kind == 'f'
            and all(v.dtype == np.dtype(bool) for v in (valid, admitted, nwm_valid)), 'Fixed typed geometry differs')
    require(np.isfinite(truth[valid]).all() and (truth[valid] >= 0).all()
            and np.isnan(truth[~valid]).all(), 'Targets must retain invalid cells as unfilled NaN')
    require(not nwm_valid[:, 16:].any() and np.isnan(nwm[:, 16:]).all()
            and np.isfinite(nwm[nwm_valid]).all() and np.isnan(nwm[~nwm_valid]).all(), 'NWM source validity or horizon differs')
    for j in range(1, 18):
        left, right = valid[1:, j-1], valid[:-1, j]
        require(np.array_equal(left, right) and np.array_equal(truth[1:, j-1][left], truth[:-1, j][right]),
                'Repeated verification hours disagree')
    for values in predictions.values():
        require(values.shape == shape and values.dtype == np.dtype('float64')
                and np.isfinite(values[admitted]).all() and np.isnan(values[~admitted]).all(),
                'Missing admitted predictions or changed input eligibility')
    strict = admitted & np.logical_and.reduce(valid, axis=1)
    primary = np.repeat(strict[:, None], 18, axis=1)
    secondary = valid & admitted[:, None]
    matched = primary & nwm_valid
    counts = dict(origin_count=N, input_admitted_origins=int(admitted.sum()), input_excluded_origins=int((~admitted).sum()),
                  primary_complete_target_origins=int(strict.sum()),
                  input_admitted_with_incomplete_targets=int((admitted & ~valid.all(axis=1)).sum()),
                  forecast_availability_fraction=int(admitted.sum()) / N,
                  primary_support=support(FUTURE, primary), independent_per_lead_support=support(FUTURE, secondary),
                  matched_nwm_support=support(FUTURE, matched), per_lead=[])
    for j in range(18):
        evaluable, observed = int(secondary[:, j].sum()), int(valid[:, j].sum())
        counts['per_lead'].append(dict(lead_hours=j+1, forecast_available_origins=int(admitted.sum()),
            valid_observed_target_origins=observed, evaluable_origins=evaluable, evaluable_fraction_all_origins=evaluable/N,
            evaluable_fraction_valid_observed_targets=evaluable/observed if observed else None,
            strict_primary_origins=int(strict.sum()), nwm_matched_origins=int(matched[:, j].sum()),
            distinct_evaluable_target_hours=len(set(map(int, FUTURE[secondary[:, j], j])))))
    models = {}
    for family in FAMILIES:
        require(set(reported['models'][family]['seeds']) == {str(s) for s in SEEDS}, 'Reported seed roster differs')
        seeds = {}
        for seed in SEEDS:
            pred = predictions[family, seed]
            seeds[str(seed)] = dict(primary=panel(truth, pred, primary),
                per_lead_valid_target=panel(truth, pred, secondary), nwm_matched=panel(truth, pred, matched, paired=True))
        models[family] = dict(seeds=seeds, seed_aggregation={name: aggregate([seeds[str(s)][name] for s in SEEDS]) for name in PANELS})
    expected = dict(schema='hydra-assessment-point-scores/v1', gauge_id=gauge, units='m3/s',
        first_origin_epoch=START, last_origin_epoch=END-HOUR, last_target_epoch=END+17*HOUR, coverage=counts, models=models,
        nwm=dict(product='SHORT_RANGE_ONLY', supported_lead_hours=list(range(1,17)),
                 structurally_unavailable_lead_hours=[17,18], matched=panel(truth,nwm,matched,paired=True)),
        input_admission_sha256=array_sha(admitted), target_mask_sha256=array_sha(valid), forecast_coordinates_sha256=array_sha(FUTURE),
        policy=dict(primary='STRICT_INPUT_ADMITTED_AND_ALL_18_TARGETS_VALID',
            secondary='INDEPENDENT_PER_LEAD_VALID_TARGETS_ON_INPUT_ADMITTED_ORIGINS',
            nwm='SAME_STRICT_PRIMARY_HYDRA_ORIGINS_AND_EXACT_LEAD_NWM_AVAILABILITY',
            primary_metric='EQUAL_LEAD_PHYSICAL_NSE_H1_H18', paired_metric='EQUAL_LEAD_PHYSICAL_NSE_H1_H16',
            minimum_distinct_target_hours=168, minimum_distinct_utc_dates=28,
            pbias_sign='100 * SUM(PREDICTION_MINUS_OBSERVATION) / SUM(OBSERVATION)',
            negative_predictions_clipped=False, targets_imputed=False, missing_predictions='FAIL_ALL_INPUT_ADMITTED_CASES',
            undefined_leads_dropped=False, seed_selection_performed=False, ensemble_predictions_scored=False,
            formal_confidence_intervals=False, overlapping_forecasts_are_independent_events=False))
    compare(reported, expected, gauge)
    return expected


def target_provenance(arrays):
    mask=arrays['target_mask']
    require(np.array_equal(arrays['target_time'],FUTURE), 'Verification coordinates differ')
    for name in ('target_quality_flags','target_source_ids','target_exclusion_reason'):
        require(arrays[name].shape==(N,18) and arrays[name].dtype.kind in 'US', 'Original target provenance is required')
    for text in np.unique(arrays['target_quality_flags'][mask]):
        flags=json.loads(str(text))
        require(type(flags) is list and 'A' in flags and all(type(f) is str and f in ('A','R') for f in flags),
                'A scored target has unapproved original qualifiers')
    require(np.all(arrays['target_source_ids'][mask]!='') and np.all(arrays['target_exclusion_reason'][mask]=='')
        and np.all(arrays['target_exclusion_reason'][~mask]!=''), 'Target source identity or exclusion reason missing')


def nwm_provenance(arrays):
    shape=(N,18);valid=arrays['source_valid']
    require(all(arrays[k].shape==shape for k in ('prediction_q','source_valid','source_row','numeric_status',
                                               'raw_streamflow','source_forecast_offset')), 'NWM provenance geometry differs')
    require(np.array_equal(valid,arrays['numeric_status']=='NUMERIC_VALID')
        and np.all(arrays['source_row'][valid]>0)
        and np.all(arrays['numeric_status'][:,16:]=='NATIVE_HORIZON_UNAVAILABLE')
        and np.all(arrays['source_row'][:,16:]==-1), 'NWM source status or unavailable horizon differs')
    # The owned BigQuery values are already decoded m3/s. Test exact parsing;
    # never apply the native NetCDF scale/offset to these strings a second time.
    parsed=np.asarray([float(v) for v in arrays['raw_streamflow'][valid]],dtype=np.float64)
    require(np.isfinite(parsed).all() and np.array_equal(parsed,arrays['prediction_q'][valid]),
            'NWM original decoded source floats were rescaled or changed')


def public_projection(results, hashes, accounting):
    """Allowlist only: no raw arrays, local paths, cloud URIs, or per-case rows."""
    output = dict(schema=SCHEMA, status='INDEPENDENT_POINT_METRICS_AUDIT_PASSED',
        assessment_origins=['2024-07-31','2025-07-31'], models=36, gauges={}, hashes=dict(hashes),
        numerical_tolerance=dict(absolute=ATOL,relative=RTOL), accounting=dict(accounting),
        limitations=['Point estimates only; seed min/max are descriptive, not confidence intervals.',
                     'Overlapping forecast windows are not independent events.',
                     'NWM matching preserves the two-hour cutoff; origin H17/H18 are unavailable.',
                     'Strict L168 full-input primary; no coverage repair, target fill, retraining, or seed selection.'])
    for g in GAUGES:
        result = results[g]
        output['gauges'][g] = dict(coverage=result['coverage'], models=result['models'], nwm=result['nwm'])
    # Projection is applied only to freshly independently constructed metrics.
    return output


def resident_metadata(path, maximum, runtime=None, descriptor=None):
    p = Path(path)
    s = p.lstat()
    require(p.is_absolute() and p == p.resolve() and stat.S_ISREG(s.st_mode)
            and not getattr(s,'st_flags',0)&0x40000000 and 0 < s.st_size <= maximum, 'Bounded resident metadata required')
    if runtime:
        runtime.reserve_read(s.st_size, numeric=False, purpose='INDEPENDENT_AUDIT_METADATA')
    raw = p.read_bytes(); after = p.lstat()
    require((s.st_ino,s.st_size,s.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns), 'Metadata changed while reading')
    if descriptor:
        require(descriptor == dict(path=str(p),bytes=len(raw),sha256=sha(raw)), 'Frozen metadata descriptor differs')
    return raw


def read_sqlite_metadata(path, runtime):
    """Open accounting-only SQLite without constructor side effects or resets."""
    p = Path(path)
    for suffix in ('','-wal','-shm'):
        v = Path(str(p)+suffix)
        if v.exists():
            s = v.lstat()
            require(v == v.resolve() and stat.S_ISREG(s.st_mode) and not getattr(s,'st_flags',0)&0x40000000
                    and s.st_size <= 64*1024**2, 'Bounded resident accounting database required')
            runtime.reserve_read(s.st_size,numeric=False,purpose='INDEPENDENT_AUDIT_ACCOUNTING')
    c = sqlite3.connect('file:'+quote(str(p),safe='/')+'?mode=ro',uri=True)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA query_only=ON'); c.execute('BEGIN')
    return c


def source_accounting(runtime, expected_approval, recorded):
    c = read_sqlite_metadata(runtime.directory/'source-staging/budget/source-budget.sqlite3', runtime)
    try:
        row = dict(c.execute('SELECT * FROM budget WHERE id=1').fetchone())
        total = c.execute('SELECT COALESCE(SUM(reserved),0),COUNT(*),COALESCE(SUM(observed),0) FROM attempts').fetchone()
        states = dict(c.execute('SELECT status,COUNT(*) FROM attempts GROUP BY status'))
    finally:
        c.rollback(); c.close()
    require(row['approval']==expected_approval and row['body_cap']==16*1024**3 and row['get_cap']==4000
            and row['cost_cap_micro']==5000000 and row['charged_body']==total[0] and row['attempts']==total[1]
            and 0<=row['charged_body']<=row['body_cap'] and 0<=row['attempts']<=4000
            and 0<=total[2]<=total[0] and not states.get('PRECHARGED',0), 'Source accounting failed to reconcile')
    cost_micro = (row['charged_body']*280000+1024**3-1)//1024**3 + row['attempts']*65
    require(cost_micro<=5000000, 'Source estimate exceeded approved five dollars')
    expected = dict(body_cap=row['body_cap'],get_cap=4000,cost_estimate_cap_usd=5.0,
        precharged_body_bytes=row['charged_body'],attempted_gets=row['attempts'],observed_body_bytes=total[2],
        conservative_precharge_estimate_usd=cost_micro/1e6,attempt_states=states,billing_invoice_reconciled=False)
    compare(recorded,expected,'source_budget')
    return expected


def original_training_accounting(runtime):
    old = ROOT/'.local/primary-campaign-v1'
    plan = json.loads(resident_metadata(old/'PLAN.json',16*1024**2,runtime))
    from hydra_directflow.campaign_ledger import CampaignLedger
    ledger = CampaignLedger(plan['ledger_directory'])
    # Charge the bounded read-only ledger before its existing authenticated snapshot.
    c = read_sqlite_metadata(ledger.database,runtime); c.rollback(); c.close()
    s = ledger.snapshot()
    require(s['complete'] and s['optimizer_updates_entered']==s['optimizer_updates_confirmed']==ORIGINAL_UPDATES,
            'Original completed training counters changed')
    return dict(entered=s['optimizer_updates_entered'],confirmed=s['optimizer_updates_confirmed'],plan_sha256=s['plan_sha256'])


def runtime_accounting(s):
    require(s['failure'] is None and s['active_job'] is None and s['pending_calls']==0,
            'Assessment accounting is unfinished or failed')
    require({x['job_id'] for x in s['commits'] if x['kind']=='model'}==MODEL_IDS,
            'All36 model commits required')
    for key,limit in (('entered_cases','max_case_checkpoint_evaluations'),('entered_batches','max_forward_batches'),
                      ('read_reserved','max_read_bytes'),('output_reserved','max_output_bytes'),
                      ('peak_combined_rss_mps_bytes','max_combined_rss_mps_bytes'),('peak_mps_bytes','max_mps_bytes')):
        require(0<=s[key]<=LIMITS[limit], 'Runtime allowance differs: '+key)
    cap=LIMITS['max_model_constructions_with_resume'] if s['resume_id'] else LIMITS['max_model_constructions']
    require(0<=s['confirmed_models']<=s['entered_models']<=cap and s['confirmed_cases']<=s['entered_cases']
            and s['confirmed_batches']<=s['entered_batches'] and s['minimum_free_bytes']>=8*1024**3
            and s['numeric_started_at'] is not None and s['deadline']==s['numeric_started_at']+21600
            and s['last_clock']<s['deadline'], 'Runtime model/resource/deadline allowance differs')
    return {k:s[k] for k in ('entered_cases','confirmed_cases','entered_batches','confirmed_batches',
                             'entered_models','confirmed_models','resume_id')}


def reconcile_confirmations(counters,receipts):
    require(set(receipts)==MODEL_IDS, 'All36 completed model receipts required for accounting')
    expected=dict(confirmed_cases=sum(r['forecast_origins'] for r in receipts.values()),
        confirmed_batches=sum((r['forecast_origins']+63)//64 for r in receipts.values()),
        # The controller constructs and validates every selected model, even
        # when a gauge has no admitted origins and its forward loop is empty.
        confirmed_models=len(receipts))
    require(all(counters[k]>=v and (counters['resume_id']>0 or counters[k]==v) for k,v in expected.items()),
            'Runtime confirmations do not reconcile with completed chronological predictions')
    return expected


def require_commit(runtime,key):
    c=runtime.reuse(key)
    require(c is not None,'Required completed commit is absent: '+key)
    return c


def metadata(runtime,descriptor):
    return json.loads(runtime.read(descriptor,numeric=False))


def scoring_release(runtime,p):
    commit=require_commit(runtime,'SCORING_RELEASE')
    release=metadata(runtime,commit['artifacts']['release'])
    require(release==commit['metadata'] and release['schema']=='hydra-assessment-point-scoring-release/v1'
        and release['plan_sha256']==p['plan_sha256']
        and release['protocol_sha256']==p['json_sources']['protocol']['sha256']
        and release['selection_sha256']==p['json_sources']['selection']['sha256']
        and all(release[k] is True for k in ('no_inference','no_training','no_post_outcome_method_selection'))
        and release['formal_confidence_intervals'] is False,
        'Frozen no-training point-scoring release differs')
    expected={'scripts/run_assessment_scores.py','src/hydra_directflow/assessment_scoring.py'}
    require(set(release['implementation'])==expected, 'Exact scorer implementation required')
    for name in sorted(expected):
        require(sha(resident_metadata(ROOT/name,2*1024**2,runtime))==release['implementation'][name],
                'Scorer code changed after its separate release')
    return commit['artifacts']['release']['sha256']


def authenticate_barrier(runtime,p,selection):
    """Independently recheck all36 input/prediction products before target reads."""
    structural=require_commit(runtime,'PREDICTION_BARRIER')
    barrier=require_commit(runtime,'SEMANTIC_PREDICTION_BARRIER')
    body=metadata(runtime,barrier['artifacts']['barrier'])
    require(body==barrier['metadata'] and body['schema']=='hydra-assessment-semantic-prediction-barrier/v1'
        and body['status']=='ALL36_PREDICTION_SEMANTICS_VERIFIED_BEFORE_VERIFICATION_ROLE_READ'
        and body['plan_sha256']==p['plan_sha256'] and body['selection_sha256']==selection['selection_sha256']
        and body['structural_prediction_barrier_sha256']==structural['artifacts']['barrier']['sha256']
        and body['verification_values_read'] is False and body['nwm_values_read'] is False
        and set(body['models'])==MODEL_IDS and set(body['gauges'])==set(GAUGES), 'Explicit all36 semantic barrier differs')
    inventory_commit=require_commit(runtime,'INPUTS')
    require(body['input_inventory']==inventory_commit['artifacts']['inventory'], 'Semantic input inventory differs')
    inventory=metadata(runtime,body['input_inventory'])
    require(inventory['input_source_identity']==p['input_source_identity']
        and set(inventory['gauge_bundles'])==set(GAUGES), 'Input inventory source differs')
    records={r['job_id']:r for r in selection['records']}
    require(len(selection['records'])==36 and set(records)==MODEL_IDS
        and selection['selection_sha256']==digest({k:v for k,v in selection.items() if k!='selection_sha256'}),
        'Frozen selected roster differs')
    predictions={g:{} for g in GAUGES};cohorts={};receipts={}
    for si,g in enumerate(GAUGES):
        bundle=inventory['gauge_bundles'][g];b=body['gauges'][g]
        require(b['input_bundle']==bundle, 'Semantic gauge bundle differs')
        a=decode_arrays(runtime.read(bundle['arrays']));manifest=metadata(runtime,bundle['manifest'])
        typed=typed_sha(a)
        require(typed==b['input_arrays_sha256']==manifest['arrays_sha256']
            and manifest['source_identity']==p['input_source_identity'] and manifest['gauge_id']==g
            and manifest['binding']['execution_plan_sha256']==p['plan_sha256']
            and a['inference_source_identity'].item()==p['input_source_identity']
            and a['cohort_admission_digest'].item()==manifest['cohort_admission_digest']
            and a['feature_schema'].item()==manifest['feature_schema'], 'Input typed/source/release binding differs')
        require(np.array_equal(a['origin_time'],CALENDAR) and np.array_equal(a['forecast_time'],FUTURE)
            and np.array_equal(a['history_time'],CALENDAR[:,None]-np.arange(169,1,-1)*HOUR)
            and a['site_id'].shape==(N,) and np.all(a['site_id']==si)
            and a['case_id'].tolist()==[g+':'+datetime.fromtimestamp(int(t),timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') for t in CALENDAR],
            'Exact input gauge/calendar/case identities differ')
        admitted=a['input_admitted'];complete=np.ones(N,bool)
        require(admitted.dtype==np.dtype(bool) and admitted.shape==(N,), 'Input admission must be explicit')
        for name,shape in (('history',(N,168,2)),('weather',(N,18,4)),('state',(N,2))):
            v,m=a[name],a[name+'_mask']
            require(v.shape==m.shape==shape and m.dtype==np.dtype(bool) and np.isfinite(v[m]).all(), 'Input mask/value geometry differs')
            complete &= m.reshape(N,-1).all(axis=1)
        require(np.array_equal(admitted,complete) and int(admitted.sum())==b['admitted_origins'], 'Complete-input cohort changed')
        cohorts[g]=admitted
        for family in FAMILIES:
            for seed in SEEDS:
                job=f'final_development__{g}__{family}__full_L168__seed{seed}'
                selected=records[job];c=require_commit(runtime,job);saved=body['models'][job]
                require(selected['gauge_id']==g and selected['architecture']==family and selected['seed']==seed,
                        'Selected job gauge/family/seed identities differ')
                require(c['kind']=='model' and c['job_id']==job and c['artifacts']==saved['artifacts']
                    and digest(c)==saved['model_commit_sha256']==structural['metadata']['model_commit_sha256_by_job'][job]
                    and saved['checkpoint_sha256']==p['checkpoint_sha256_by_job'][job]==selected['checkpoint']['sha256'],
                    'Model commit/checkpoint differs from all36 barrier')
                z=decode_arrays(runtime.read(c['artifacts']['predictions']));r=metadata(runtime,c['artifacts']['receipt'])
                require(set(z)=={'origin_time','case_id','forecast_time','site_id','input_admitted','prediction_q'}
                    and all(np.array_equal(a[k],z[k]) for k in ('origin_time','case_id','forecast_time','site_id','input_admitted')),
                    'Saved prediction coordinates or admission differ')
                q=z['prediction_q'];pos=np.flatnonzero(admitted)
                require(q.dtype==np.dtype('float64') and q.shape==(N,18)
                    and np.isfinite(q[admitted]).all() and np.isnan(q[~admitted]).all(), 'Missing admitted forecast or cohort shrinking')
                require(typed_sha(z)==r['predictions_array_sha256']==saved['predictions_array_sha256']
                    and r['input_arrays_sha256']==saved['input_arrays_sha256']==typed and r['job']==selected
                    and r['input_artifact']==bundle['arrays'] and r['checkpoint_sha256']==selected['checkpoint']['sha256']
                    and r['forecast_origins']==saved['forecast_origins']==len(pos) and r['candidate_origins']==N
                    and r['target_values_assembled'] is False and r['optimizer_updates']==r['scaler_fits']==0,
                    'Prediction receipt source, target boundary or no-training claim differs')
                require(len(r['batches'])==(len(pos)+63)//64, 'Prediction batch count differs')
                for offset,batch in zip(range(0,len(pos),64),r['batches']):
                    take=pos[offset:offset+64]
                    require(batch['job_id']==job and batch['checkpoint_sha256']==selected['checkpoint']['sha256']
                        and batch['case_count']==len(take) and batch['input_case_ids_sha256']==digest(a['case_id'][take].tolist())
                        and batch['target_values_read'] is False and batch['target_masks_read'] is False
                        and batch['optimizer_updates']==batch['scaler_fits']==0,
                        'Chronological forward receipts or no-learning counters differ')
                predictions[g][family,seed]=q;receipts[job]=dict(checkpoint_sha256=r['checkpoint_sha256'],
                    prediction_artifact_sha256=c['artifacts']['predictions']['sha256'],forecast_origins=len(pos))
        runtime.check()
    return barrier,structural,inventory,predictions,cohorts,receipts


def run(plan_path,expected_plan,expected_results):
    p=json.loads(resident_metadata(Path(plan_path),16*1024**2))
    require(p['plan_sha256']==expected_plan==digest({k:v for k,v in p.items() if k!='plan_sha256'})
        and p['original_training_updates']==ORIGINAL_UPDATES and p['new_training'] is False,
        'Exact original no-training execution plan required')
    # Never create a fresh runtime just to audit an unstarted run.
    runtime=AssessmentRuntime(p)
    require(runtime.database.exists(), 'Original completed runtime must already exist')
    with runtime.writer():
        finished=require_commit(runtime,'POINT_RESULTS')
        require(finished['artifacts']['results']['sha256']==expected_results, 'Explicit committed result hash required')
        result=metadata(runtime,finished['artifacts']['results'])
        require(result['status']=='POINT_METRICS_COMPLETE_PENDING_INDEPENDENT_REVIEW' and result['models']==36
            and result['plan_sha256']==expected_plan and result['new_optimizer_updates']==result['new_scaler_fits']==0
            and result['formal_confidence_intervals'] is False and result['nwm_supported_origin_leads']==[1,16]
            and set(result['scores'])==set(GAUGES), 'Completed first-pass scope differs')
        prior=runtime.reuse('INDEPENDENT_POINT_AUDIT')
        if prior:
            require(prior['metadata']['point_results_sha256']==expected_results
                and prior['metadata']['audit_code_sha256']==sha(resident_metadata(Path(__file__).resolve(),2*1024**2,runtime)),
                'Completed audit identity changed')
            return prior
        before=runtime_accounting(runtime.snapshot());old_before=original_training_accounting(runtime)
        scorer_release_sha=scoring_release(runtime,p)
        sd=p['json_sources']['selection'];selection=json.loads(resident_metadata(Path(sd['path']),32*1024**2,runtime,sd))
        barrier,structural,inventory,predictions,cohorts,receipts=authenticate_barrier(runtime,p,selection)
        reconcile_confirmations(before,receipts)
        barrier_sha=barrier['artifacts']['barrier']['sha256']
        # This line is deliberately after all36 semantic revalidation.
        nwm=require_commit(runtime,'NWM_MATCHED_INPUT');targets=require_commit(runtime,'VERIFICATION_TARGETS')
        require(targets['metadata']==dict(prediction_barrier_sha256=barrier_sha), 'Targets were not released by the semantic barrier')
        nw_audit=metadata(runtime,nwm['artifacts']['audit'])
        require(nwm['metadata']==dict(snapshot_sha256=p['nwm_snapshot']['sha256'])
            and nw_audit['snapshot']==p['nwm_snapshot'] and nw_audit['reference_rule']=='EXACT_ORIGIN_MINUS_2H'
            and nw_audit['native_leads']==[3,18] and nw_audit['origin_leads']==[1,16]
            and nw_audit['unsupported_origin_leads']==[17,18] and nw_audit['no_rescaling'] is True,
            'Owned NWM snapshot, issuance rule or decoded-unit handling differs')
        require(nw_audit['units']=='m3 s-1' and nw_audit['representation']=='BIGQUERY_DECODED_FLOAT_M3_S',
                'Owned NWM decoded units differ')
        stage_d=p['json_sources']['staging_plan'];stage=json.loads(resident_metadata(Path(stage_d['path']),16*1024**2,runtime,stage_d))
        budget=source_accounting(runtime,stage['sources']['budget_approval']['sha256'],inventory['source_budget'])
        results={};score_hashes={}
        for g in GAUGES:
            score_commit=require_commit(runtime,'SCORE_'+g)
            require(result['scores'][g]==score_commit['artifacts']['scores'], 'Completion score descriptor differs')
            reported=metadata(runtime,result['scores'][g]);truth=decode_arrays(runtime.read(targets['artifacts'][g]));nw=decode_arrays(runtime.read(nwm['artifacts'][g]))
            target_provenance(truth);nwm_provenance(nw)
            require(typed_sha(nw)==nw_audit['arrays_sha256_by_gauge'][g], 'NWM typed source artifact differs')
            compare(reported['provenance'],dict(plan_sha256=expected_plan,prediction_barrier_sha256=barrier_sha,
                structural_prediction_barrier_sha256=structural['artifacts']['barrier']['sha256'],
                target_artifact=targets['artifacts'][g],nwm_artifact=nwm['artifacts'][g],input_artifact=inventory['gauge_bundles'][g]['arrays']))
            results[g]=audit_gauge(g,truth['target'],truth['target_mask'],cohorts[g],predictions[g],nw['prediction_q'],nw['source_valid'],reported)
            score_hashes[g]=result['scores'][g]['sha256'];runtime.check()
        after=runtime_accounting(runtime.snapshot());old_after=original_training_accounting(runtime)
        require(before==after and old_before==old_after, 'Audit performed or observed new inference/training work')
        code_sha=sha(resident_metadata(Path(__file__).resolve(),2*1024**2,runtime))
        hashes=dict(plan_sha256=expected_plan,point_results_sha256=expected_results,
                    semantic_barrier_sha256=barrier_sha,scoring_release_sha256=scorer_release_sha,
                    audit_code_sha256=code_sha,score_sha256_by_gauge=score_hashes)
        accounting=dict(original_training_updates=ORIGINAL_UPDATES,new_optimizer_updates=0,new_scaler_fits=0,
            audit_model_constructions=0,audit_forward_calls=0,runtime=after,source_budget=budget)
        public=public_projection(results,hashes,accounting)
        private=dict(schema=SCHEMA,status='PASS',hashes=hashes,accounting=accounting,model_products=receipts,
                     method='Independent scalar math.fsum; exact masks, coordinates, support, metrics and seed aggregation.',
                     numerical_tolerance=dict(absolute=ATOL,relative=RTOL),point_estimates_only=True)
        descriptors=dict(private_audit=runtime.write('audit/INDEPENDENT_AUDIT.json',encoded(private),kind='INDEPENDENT_RESULT_AUDIT'),
                         public_summary=runtime.write('audit/PUBLIC_AGGREGATE_SUMMARY.json',encoded(public),kind='AGGREGATE_RESULT_AUDIT'))
        require(before==runtime_accounting(runtime.snapshot()),'Audit changed inference counters')
        return runtime.commit('INDEPENDENT_POINT_AUDIT',descriptors,metadata=dict(point_results_sha256=expected_results,audit_code_sha256=code_sha))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--point-results-sha256',required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.plan,args.plan_sha256,args.point_results_sha256),sort_keys=True))
