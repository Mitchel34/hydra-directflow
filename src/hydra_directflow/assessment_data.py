"""Additive assessment-only source decoding and causal input assembly.

The consuming runner must authenticate the release, selection and source files
before constructing ``AssessmentReadAuthority``. The structural bindings here
are not an approval mechanism. Existing development readers remain unchanged.
No model, optimizer, scaler fitting, download or source mutation is performed.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import stat

import numpy as np

from .assessment_contract import clock
from .lineage import availability_view, normalize_manifest_summary
from .local_data import (GAUGES, FEATURE_GROUPS, GRID_SHA256, USGS_MANIFEST_SHA256,
                         WEIGHTS_SHA256, DEVELOPMENT_QUALIFICATION, load_candidate_weights,
                         CFS_TO_CMS, LocalCycle, bounded_npy, iso, utc_epoch,
                         sha256, verified_local_bytes, read_cycle_tar,
                         _source_bytes, _verify_grid_and_clocks,
                         _verify_soil_grid, _verify_weather_crop)
from .quality import PRIMARY_APPROVED, quality_admitter
from .reference import (ASSESSMENT_START, ASSESSMENT_END, ASSESSMENT_LAST_TARGET,
                        HOUR, AvailabilityPolicy, Observation, WeatherRun,
                        SoilInitialState, latest_weather_run, soil_state,
                        target_window, digest, require)

SCHEMA = 'hydra-assessment-input-data/v1'
FULL_FEATURE_SCHEMA = 'FULL_HRRR_H2_W4_S2_V1'
MAX_ORIGINS = 128
MAX_CYCLES = 32
FIRST_CYCLE = ASSESSMENT_START - 18 * HOUR
LAST_CYCLE = ((ASSESSMENT_END - 3 * HOUR) // (6 * HOUR)) * 6 * HOUR
FIRST_INPUT = ASSESSMENT_START - 169 * HOUR
LAST_INPUT = ASSESSMENT_END - 3 * HOUR
UNITS = {'temperature': 'K', 'humidity': 'kg kg-1',
         'precipitation': 'kg m-2', 'snow_water_equivalent': 'kg m-2'}
USGS_COLUMNS = ('site', 'sample_epoch', 'sample_time_utc', 'source_units',
                'discharge_cfs', 'source_qualifiers_json', 'numeric_valid',
                'is_exact_utc_hour', 'is_core', 'source_wire_sha256',
                'historical_available_at_utc')


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


@dataclass(frozen=True)
class AssessmentReadAuthority:
    """Bindings issued by an external authenticated runner, not self-approval.

    ``cycle_row_hashes`` binds supplied source rows to the runner's authenticated
    inventory. Verification requires an additional committed-prediction barrier.
    The runner must verify the actual artifacts behind all these identities.
    """
    execution_plan_sha256: str
    protocol_sha256: str
    selection_sha256: str
    source_inventory_sha256: str
    cycle_row_hashes: tuple[tuple[str, str], ...] = ()
    inputs_enabled: bool = False
    verification_enabled: bool = False
    prediction_barrier_sha256: str | None = None
    allowed_history_hours: tuple[int, ...] = (168,)
    status: str = 'RUNNER_AUTHENTICATED_ASSESSMENT_RELEASE_V1'

    def __post_init__(self):
        require(self.status == 'RUNNER_AUTHENTICATED_ASSESSMENT_RELEASE_V1'
                and all(_sha(x) for x in (self.execution_plan_sha256,
                    self.protocol_sha256, self.selection_sha256,
                    self.source_inventory_sha256)), 'Exact authenticated release bindings required')
        require(type(self.inputs_enabled) is bool and type(self.verification_enabled) is bool,
                'Read gates must be explicit booleans')
        require(type(self.allowed_history_hours) is tuple
                and len(set(self.allowed_history_hours)) == len(self.allowed_history_hours)
                and self.allowed_history_hours
                and all(type(x) is int and x in (24, 168) for x in self.allowed_history_hours),
                'Explicit retained history lengths required')
        require(type(self.cycle_row_hashes) is tuple, 'Immutable cycle-row bindings required')
        seen = set()
        for item in self.cycle_row_hashes:
            require(type(item) is tuple and len(item) == 2 and _sha(item[1]),
                    'Cycle-row binding must have a cycle and SHA256')
            assessment_cycle_reference(item[0])
            require(item[0] not in seen, 'Duplicate authorized cycle row')
            seen.add(item[0])
        require(self.prediction_barrier_sha256 is None or _sha(self.prediction_barrier_sha256),
                'Prediction barrier identity must be SHA256')
        if self.verification_enabled:
            require(self.inputs_enabled and _sha(self.prediction_barrier_sha256),
                    'Verification requires the committed prediction barrier')


def _gate(authority, *, verification=False):
    require(isinstance(authority, AssessmentReadAuthority),
            'A runner-authenticated assessment read authority is required before I/O')
    require(authority.inputs_enabled, 'Assessment numerical input decoding is disabled')
    if verification:
        require(authority.verification_enabled and _sha(authority.prediction_barrier_sha256),
                'Verification target decoding requires committed predictions')


def assessment_cycle_reference(cycle_id):
    require(isinstance(cycle_id, str) and re.fullmatch(r'\d{8}_\d{2}Z', cycle_id),
            'Explicit assessment cycle ID required')
    try:
        reference = int(datetime.strptime(cycle_id, '%Y%m%d_%HZ').replace(tzinfo=timezone.utc).timestamp())
    except ValueError as exc:
        raise ValueError('Invalid assessment cycle clock') from exc
    require(reference % (6 * HOUR) == 0 and FIRST_CYCLE <= reference <= LAST_CYCLE,
            'Cycle outside the fixed 1,463-candidate assessment calendar')
    return reference


def _resident_charge(path, maximum, read_budget, *, expected_bytes=None):
    p = Path(path)
    require(p.absolute() == p.resolve(), 'Source aliases and symlinks are forbidden')
    st = p.lstat()
    require(stat.S_ISREG(st.st_mode) and not getattr(st, 'st_flags', 0) & 0x40000000,
            'Assessment input must be resident; readers cannot hydrate sources')
    require(0 < st.st_size <= maximum and (expected_bytes is None or st.st_size == expected_bytes),
            'Bounded source byte count differs')
    read_budget(st.st_size)
    return st.st_size


def _bundle_identity(spec, index_row, expected):
    # Lazy import keeps pure assembly and synthetic fixtures independent of
    # transport initialization. This resolver reads only pinned commit metadata.
    from .assessment_sources import authenticate_assessment_bundle_identity
    return authenticate_assessment_bundle_identity(
        spec, index_row, expected_sealed_row_sha256=expected)


def read_assessment_weights(path, authority, *, read_budget=lambda n: None):
    """Reuse the immutable 514-row spatial source after the assessment gate.

    The existing parser's development qualification is a historical source
    contract, not a new scientific admission or a modification of its hold flags.
    The authenticated release must adopt the same approximate, independently
    treated basins. Missing positive-weight cells remain incomplete.
    """
    _gate(authority)
    _resident_charge(path, 128 * 1024, read_budget, expected_bytes=62601)
    weights = load_candidate_weights(path, qualification=DEVELOPMENT_QUALIFICATION)
    weights.setflags(write=False)
    return weights, dict(schema=SCHEMA, role='ASSESSMENT_FROZEN_SPATIAL_WEIGHTS',
        execution_plan_sha256=authority.execution_plan_sha256,
        weights_sha256=WEIGHTS_SHA256, bytes=62601, source_rows=514,
        interpretation='FROZEN_APPROXIMATE_BASINS_SEPARATE_NON_NESTING_GAUGES',
        original_candidate_status_unchanged=True, partial_coverage_renormalized=False,
        source_files_modified=False)


def load_assessment_cycle(spec, index_row, authority, *, read_budget=lambda n: None):
    """Authenticate and decode one original HRRR archive after the read gate."""
    _gate(authority)
    require(isinstance(spec, Mapping) and isinstance(index_row, Mapping), 'Explicit cycle and sealed row required')
    cycle_id = spec['cycle_id']
    reference = assessment_cycle_reference(cycle_id)
    expected = dict(authority.cycle_row_hashes).get(cycle_id)
    require(expected is not None and digest(index_row) == expected,
            'Cycle row is not bound to the authenticated assessment inventory')
    if spec.get('commit_path') is not None:
        _resident_charge(spec['commit_path'], 16384, read_budget)
    bundle, worker, lineage = _bundle_identity(spec, index_row, expected)
    _resident_charge(spec['archive_path'], 16 * 1024**2, read_budget, expected_bytes=bundle['bytes'])
    members, manifest = read_cycle_tar(spec['archive_path'], bundle['sha256'],
                                      cycle_id=cycle_id, allow_legacy=True)
    manifest_sha = sha256(members['CYCLE_MANIFEST.json'])
    require(worker is None or worker['manifest_sha256'] == manifest_sha,
            'Original worker/manifest binding differs')
    availability = availability_view(index_row, manifest, members)
    summary = normalize_manifest_summary(index_row, manifest, availability, worker)
    n = summary['record_count']
    if 'raw_values.npy' in members:
        require(manifest['raw_shape'] == [n, 957] and manifest['mask_shape'] == [n, 120],
                'Assessment cycle array shapes differ')
        values = bounded_npy(members['raw_values.npy'], shape=(n, 957), dtype='<f8')
        bits = bounded_npy(members['raw_missing_bits.npy'], shape=(n, 120), dtype='u1')
    else:
        require(n == 0 and manifest['raw_shape'] is None and manifest['mask_shape'] is None,
                'Metadata-only source gap has an inconsistent numerical inventory')
        values, bits = np.empty((0, 957)), np.empty((0, 120), dtype='u1')
    require(manifest['assets']['grid_sha256'] == GRID_SHA256, 'Frozen HRRR grid differs')
    missing = np.unpackbits(bits, axis=1, bitorder='little')
    require(not missing[:, 957:].any(), 'Nonzero source-mask padding')
    present = ~missing[:, :957].astype(bool)
    require(np.isfinite(values[present]).all(), 'Nonfinite admitted source value')
    records = json.loads(members['RAW_PROVENANCE.json'])
    fields = json.loads(members['FIELD_SOURCES.json'])
    require(len(records) == n and availability['cycle'] == cycle_id,
            'Original source and availability inventory differ')
    _verify_grid_and_clocks(members, fields, reference, cycle_id=cycle_id, minimum_leads=36)
    seen, groups = set(), {g['group']: g for g in availability['groups']}
    require(len(groups) == len(availability['groups']), 'Duplicate availability group')
    for i, record in enumerate(records):
        group, lead = record['group'], record['lead']
        require(record['record_index'] == i and (group, lead) not in seen, 'Duplicate or reordered source record')
        seen.add((group, lead))
        require(utc_epoch(record['source_reference_UTC']) == reference
                and utc_epoch(record['source_valid_UTC']) == reference + lead * HOUR,
                'Original record clocks differ')
        require(sha256(values[i].astype('<f8', copy=False).tobytes()) == record['source_values_sha256_float64_le']
                and sha256(bits[i].tobytes()) == record['missing_bits_sha256']
                and int((~present[i]).sum()) == record['missing_cells'],
                'Original value or mask identity differs')
        if group == 'soil_moisture':
            require(lead == 0 and record['native_metadata']['units'] == 'Proportion', 'Initial soil definition differs')
            _verify_soil_grid(_source_bytes(members, record['source'], cycle_id=cycle_id))
        else:
            require(group in UNITS and type(lead) is int and 1 <= lead <= 36
                    and record['units'] == UNITS[group], 'Weather group, native lead or units differ')
            require(record['APCP_interval_hours'] == ([lead - 1, lead] if group == 'precipitation' else None),
                    'Original one-hour precipitation interval differs')
            require(record['field_source_id'] in fields, 'Original weather field source missing')
    available = {group for group, _ in seen}
    require(available == set(summary['groups_available']), 'Original available groups differ')
    for group in available:
        leads = [0] if group == 'soil_moisture' else list(range(1, 37))
        require(sorted(lead for g, lead in seen if g == group) == leads
                and groups[group]['leads'] == leads and groups[group]['record_count'] == len(leads)
                and groups[group]['status'] in ('AVAILABLE_VERIFIED', 'ADAPTER_DERIVED_COMPLETE_GROUP'),
                'Available group has incomplete or conflicting native support')
    _verify_weather_crop(members, fields, records, values, present, cycle_id=cycle_id)
    values.setflags(write=False); present.setflags(write=False)
    cycle = LocalCycle(reference, values, present, tuple(records), len(members), fields, availability)
    receipt = dict(schema=SCHEMA, role='ASSESSMENT_HRRR_INPUT', cycle_id=cycle_id,
                   execution_plan_sha256=authority.execution_plan_sha256,
                   sealed_row_sha256=expected, archive_sha256=bundle['sha256'], archive_bytes=bundle['bytes'],
                   manifest_sha256=manifest_sha, lineage=lineage, normalized_summary=summary,
                   numeric_records_decoded=n, source_files_modified=False,
                   verification_target_arrays_read=False)
    return cycle, receipt


def _windows(windows, mode):
    require(mode in ('input', 'verification'), 'Explicit source-read role required')
    require(type(windows) in (list, tuple) and 1 <= len(windows) <= 128,
            'One to 128 explicit inclusive windows required')
    windows = tuple(windows)
    lo, hi = (FIRST_INPUT, LAST_INPUT) if mode == 'input' else (ASSESSMENT_START, ASSESSMENT_LAST_TARGET)
    for window in windows:
        require(type(window) is tuple and len(window) == 2
                and all(type(t) is int and t % HOUR == 0 for t in window)
                and lo <= window[0] <= window[1] <= hi,
                'Requested USGS window exceeds its assessment role')
    result = set()
    for start, end in windows:
        t = datetime.fromtimestamp(start, timezone.utc).replace(day=1, hour=0)
        while int(t.timestamp()) <= end:
            result.add(t.strftime('%Y-%m'))
            t = t.replace(year=t.year+1, month=1) if t.month==12 else t.replace(month=t.month+1)
    require(len(result) <= 13 and min(result) >= '2024-07' and max(result) <= '2025-07',
            'Only the 13 assessment native partitions are in scope')
    return windows, tuple(sorted(result))


def load_assessment_usgs(paths, manifest_path, windows, authority, *, mode='input', read_budget=lambda n: None):
    """Read pinned native sources, retaining only requested rows and qualifiers.

    Parquet may internally decode full row groups before applying its filter.
    The receipt explicitly distinguishes that from returned rows. Causal input
    hours can also be verification hours for earlier origins. This function
    never returns a target tensor; the verification role is separately gated.
    """
    _gate(authority, verification=mode == 'verification')
    windows, months = _windows(windows, mode)
    require(isinstance(paths, Mapping) and set(months) <= set(paths), 'Required resident native months missing')
    _resident_charge(manifest_path, 2 * 1024**2, read_budget)
    manifest = json.loads(verified_local_bytes(manifest_path, USGS_MANIFEST_SHA256, max_bytes=2 * 1024**2))
    entries = {}
    for item in manifest['native_partitions']:
        name = Path(item['path']).name
        require(name not in entries, 'Duplicate partition identity in pinned USGS manifest')
        entries[name] = item
    from pyarrow import BufferReader
    import pyarrow.parquet as pq
    filters = [[('sample_epoch', '>=', a), ('sample_epoch', '<=', b)] for a, b in windows]
    observations, originals, receipts, seen = [], [], [], set()
    for month in months:
        entry = entries.get(month + '.parquet')
        require(entry is not None, 'Native month absent from pinned USGS manifest')
        _resident_charge(paths[month], 512 * 1024, read_budget, expected_bytes=entry['bytes'])
        raw = verified_local_bytes(paths[month], entry['sha256'], expected_bytes=entry['bytes'], max_bytes=512 * 1024)
        native = pq.ParquetFile(BufferReader(raw)).metadata
        require(native.num_rows <= 30000 and sum(native.row_group(i).column(j).total_uncompressed_size
                for i in range(native.num_row_groups) for j in range(native.num_columns)) <= 64 * 1024**2,
                'USGS native rows or Parquet expansion exceed the bounded reader')
        rows = pq.read_table(BufferReader(raw), columns=list(USGS_COLUMNS), filters=filters).to_pylist()
        require(len(rows) <= 30000, 'USGS native row bound exceeded')
        exact, core, noncore = 0, 0, 0
        for row in rows:
            t, site = row['sample_epoch'], row['site']
            require(type(t) is int and any(a <= t <= b for a, b in windows)
                    and utc_epoch(row['sample_time_utc']) == t, 'USGS window or sample clocks differ')
            require(site in GAUGES and row['source_units'] == 'ft3/s'
                    and row['historical_available_at_utc'] is None, 'USGS site, units or vintage differs')
            require(type(row['is_exact_utc_hour']) is bool and row['is_exact_utc_hour'] == (t % HOUR == 0)
                    and type(row['is_core']) is bool and type(row['numeric_valid']) is bool,
                    'USGS native flags differ from their typed contract')
            if not row['is_exact_utc_hour']:
                continue
            # Original is_core describes acquisition-month ownership, not
            # observation quality. Keep an explicitly requested tail even if
            # it was acquisition padding. Adjacent-month padding duplicates
            # belong to the partition matching the timestamp's calendar month.
            if not iso(t).startswith(month):
                continue
            require((site, t) not in seen, 'Duplicate native exact-hour row')
            seen.add((site, t)); exact += 1
            core += int(row['is_core']); noncore += int(not row['is_core'])
            require(_sha(row['source_wire_sha256']), 'Original USGS wire identity missing')
            flags = json.loads(row['source_qualifiers_json'])
            require(type(flags) is list and all(type(x) is str for x in flags), 'Original qualifier array malformed')
            value = row['discharge_cfs']
            numeric = row['numeric_valid'] and type(value) in (int, float) and math.isfinite(value) and value >= 0
            observations.append(Observation(site, t, None, float(value) * CFS_TO_CMS if numeric else None,
                                            tuple(flags), 'usgs-wire-sha256:' + row['source_wire_sha256']))
            originals.append(dict(row, source_partition_sha256=entry['sha256'], source_partition_month=month))
        receipts.append(dict(month=month, sha256=entry['sha256'], bytes=entry['bytes'],
                             returned_rows=len(rows), exact_core_rows=core,
                             exact_requested_rows=exact, retained_noncore_rows=noncore,
                             temporal_core_flag_used_as_quality=False,
                             parquet_may_decode_full_row_groups=True))
    return dict(observations=tuple(observations), original_rows=tuple(originals), metadata=tuple(receipts),
                mode=mode, requested_windows=windows, source_manifest_sha256=USGS_MANIFEST_SHA256,
                execution_plan_sha256=authority.execution_plan_sha256,
                prediction_barrier_sha256=authority.prediction_barrier_sha256 if mode == 'verification' else None,
                original_qualifiers_preserved=True, targets_imputed=False, source_files_modified=False)


def _history(index, site, origin, length, policy, admit):
    times = clock(origin, length)['history']
    result = {k: [] for k in ('values', 'mask', 'age_hours', 'reasons', 'quality_flags', 'source_ids')}
    for t in times:
        row = index.get((site, t)); reason = None
        if row is None: reason = 'MISSING_EXACT_HOUR'
        elif policy.available_at(row.valid_at, row.available_at) > origin: reason = 'UNAVAILABLE_AT_CUTOFF'
        elif row.value is None or type(row.value) not in (int, float) or not math.isfinite(row.value) or row.value < 0: reason = 'INVALID_DISCHARGE'
        elif not admit(row.quality_flags): reason = 'QUALITY_NOT_ADMITTED'
        for key, value in dict(values=float(row.value) if reason is None else None,
                              mask=reason is None, age_hours=(origin-t)/HOUR if reason is None else None,
                              reasons=reason, quality_flags=row.quality_flags if row else None,
                              source_ids=row.source_id if row else None).items():result[key].append(value)
    return dict(times=times, **{k:tuple(v) for k,v in result.items()})


def _array_identity(arrays):
    return digest({name: dict(dtype=value.dtype.str, shape=list(value.shape),
                             sha256=sha256(np.ascontiguousarray(value).tobytes()))
                   for name, value in sorted(arrays.items())})


def _unicode_array(values):
    # NumPy fixed-width strings pad every element to the longest item. Check
    # that expansion before constructing the final persisted Unicode buffer.
    probe = np.asarray(values, dtype=object)
    require(all(type(v) is str for v in probe.flat), 'Original provenance strings required')
    maximum = max((len(v) for v in probe.flat), default=1)
    require(probe.size * max(1, maximum) * 4 <= 256 * 1024**2,
            'Provenance string-array expansion exceeds the explicit bound')
    return np.asarray(values, dtype=str)


def assemble_assessment_inputs(cycles, reduced, observations, origins, authority, *,
                               source_identity, history_hours=168):
    """Pure, bounded label-free assembly; retain unavailable cases explicitly."""
    _gate(authority)
    require(_sha(source_identity) and history_hours in authority.allowed_history_hours,
            'Source identity and released lookback required')
    require(type(origins) in (list, tuple) and 1 <= len(origins) <= MAX_ORIGINS,
            'One to 128 explicit assessment origins required')
    origins = tuple(origins)
    require(origins == tuple(sorted(set(origins))),
            'One to 128 increasing unique assessment origins required')
    for c in origins:clock(c, history_hours)
    require(isinstance(cycles, Mapping) and 1 <= len(cycles) <= MAX_CYCLES
            and set(cycles) == set(reduced) and set(cycles) <= set(dict(authority.cycle_row_hashes)),
            'One to 32 inventory-bound reduced cycles required')
    policy = AvailabilityPolicy('assessment_simulated_lag_2h_v1', 'retrospective_simulated', 2 * HOUR)
    admit = quality_admitter(PRIMARY_APPROVED)
    indexes, metadata, metadata_index = {}, [], {}
    for cid, cyc in sorted(cycles.items()):
        require(cyc.reference == assessment_cycle_reference(cid), 'Cycle identity/reference differs')
        ix = {};means, masks, coverage = reduced[cid]
        require(means.shape == masks.shape == coverage.shape == (len(cyc.records), 3)
                and means.dtype.kind == 'f' and masks.dtype == np.dtype(bool)
                and coverage.dtype.kind == 'f' and np.isfinite(coverage).all()
                and np.all((coverage >= -1e-10) & (coverage <= 1+1e-10))
                and np.all(coverage[masks] >= 1-1e-10)
                and np.isfinite(means[masks]).all(), 'Reduced basin arrays differ or renormalize partial coverage')
        for i, rec in enumerate(cyc.records):
            key = rec['group'], rec['lead']
            require(key not in ix and rec['record_index'] == i
                    and (key[0] == 'soil_moisture' and key[1] == 0 or key[0] in UNITS and type(key[1]) is int and 1 <= key[1] <= 36),
                    'Reduced source record inventory differs')
            ix[key] = i
            group = next((a for a in cyc.availability['groups'] if a['group'] == rec['group']), None)
            require(group is not None, 'Original group provenance missing')
            meta = dict(cycle_id=cid, record=rec, group_availability=group)
            if 'field_source_id' in rec:meta['field_source'] = cyc.fields[rec['field_source_id']]
            metadata_index[cid, *key] = len(metadata);metadata.append(json.dumps(meta,sort_keys=True,separators=(',',':'),allow_nan=False))
        indexes[cid] = ix
    obs_index = {}
    lo, hi = origins[0]-(history_hours+1)*HOUR, origins[-1]-2*HOUR
    for row in observations:
        # Check coordinates before touching values, flags or availability.
        if row.valid_at < lo or row.valid_at > hi:continue
        require(row.site in GAUGES and type(row.valid_at) is int and row.valid_at % HOUR == 0,
                'Exact declared observation coordinates required')
        require((row.site,row.valid_at) not in obs_index, 'Duplicate input observation')
        obs_index[row.site,row.valid_at] = row
    runs_by_site, soils_by_site, soil_lookup = {}, {}, {}
    for si, site in enumerate(GAUGES):
        runs, soils = [], []
        for cid, cyc in sorted(cycles.items()):
            means, masks, _ = reduced[cid];ix = indexes[cid]
            leads = sorted({h for group,h in ix if group in UNITS})
            support = tuple((feature,tuple(bool(masks[ix[group,h],si]) if (group,h) in ix else False for h in leads))
                            for feature,group in FEATURE_GROUPS)
            runs.append(WeatherRun(cid,cyc.reference,None,tuple(cyc.reference+h*HOUR for h in leads),support))
            if ('soil_moisture',0) in ix:
                i=ix['soil_moisture',0];rec=cyc.records[i];sid=rec['source']['key']
                require(sid not in soil_lookup or soil_lookup[sid]==cid, 'Competing soil source identities')
                soil_lookup[sid]=cid
                soils.append(SoilInitialState(sid,cyc.reference,None,float(means[i,si]) if masks[i,si] else None,
                                             (rec['availability_status'],)))
        runs_by_site[site],soils_by_site[site]=runs,soils
    names=('history','history_mask','weather','weather_mask','state','state_mask','site_id','origin_time','case_id',
           'history_time','forecast_time','history_quality_flags','history_source_ids','history_exclusion_reason',
           'source_cycle','soil_source_cycle','weather_source_leads','weather_metadata_index','soil_metadata_index',
           'soil_quality_flags','input_admitted','eligibility_reason')
    out={name:[] for name in names}
    for c in origins:
        future=clock(c,history_hours)['valid_times']
        for si,site in enumerate(GAUGES):
            h=_history(obs_index,site,c,history_hours,policy,admit)
            runs=[r for r in runs_by_site[site] if r.reference >= c-18*HOUR]
            chosen=latest_weather_run(runs,cutoff=c,origin_anchor=c,policy=policy,horizons_hours=tuple(range(1,19)),allow_older_extended_cycles=True)
            soil=soil_state(soils_by_site[site],cutoff=c,policy=policy,max_age_seconds=8*HOUR)
            weather=np.full((18,4),np.nan);wm=np.zeros((18,4),dtype=bool);wi=np.full((18,4),-1,dtype=np.int64);leads=np.full(18,-1,dtype=np.int64)
            cid=chosen.run_id if chosen else ''
            if chosen:
                means,masks,_=reduced[cid];ix=indexes[cid];leads=np.array([(t-chosen.reference)//HOUR for t in future],dtype=np.int64)
                for j,lead in enumerate(leads):
                    for k,(_,group) in enumerate(FEATURE_GROUPS):
                        weather[j,k]=means[ix[group,int(lead)],si];wm[j,k]=masks[ix[group,int(lead)],si];wi[j,k]=metadata_index[cid,group,int(lead)]
            soil_cid=soil_lookup.get(soil['source_id'],'')
            reasons=[reason for bad,reason in [(chosen is None,'NO_COMPLETE_SINGLE_WEATHER_RUN'),(not all(h['mask']),'HISTORY_INCOMPLETE'),(not soil['mask'],'SOIL_UNAVAILABLE_OR_STALE')] if bad]
            row=dict(history=list(zip(h['values'],h['age_hours'])),history_mask=np.repeat(np.asarray(h['mask'],dtype=bool)[:,None],2,axis=1),
                     weather=weather,weather_mask=wm,state=[soil['value'],soil['age_hours'] if soil['mask'] else None],state_mask=[soil['mask']]*2,
                     site_id=si,origin_time=c,case_id=site+':'+iso(c),history_time=h['times'],forecast_time=future,
                     history_quality_flags=[json.dumps(f) for f in h['quality_flags']],history_source_ids=[v or '' for v in h['source_ids']],
                     history_exclusion_reason=[v or '' for v in h['reasons']],source_cycle=cid,soil_source_cycle=soil_cid,weather_source_leads=leads,
                     weather_metadata_index=wi,soil_metadata_index=metadata_index.get((soil_cid,'soil_moisture',0),-1),
                     soil_quality_flags=json.dumps(soil['quality_flags']),input_admitted=not reasons,eligibility_reason='|'.join(reasons) or 'COMPLETE_INPUTS')
            for name,value in row.items():out[name].append(value)
    floats={'history','weather','state'};ints={'site_id','origin_time','history_time','forecast_time','weather_source_leads','weather_metadata_index','soil_metadata_index'}
    arrays={name:np.asarray(v,dtype=np.float64 if name in floats else bool if name.endswith('_mask') or name=='input_admitted' else np.int64)
            if name in floats or name in ints or name.endswith('_mask') or name=='input_admitted' else _unicode_array(v)
            for name,v in out.items()}
    arrays['source_metadata_json']=_unicode_array(metadata)
    n=3*len(origins)
    require(arrays['history'].shape==(n,history_hours,2) and arrays['weather'].shape==(n,18,4)
            and arrays['state'].shape==(n,2) and arrays['forecast_time'].shape==(n,18), 'Input tensor shapes differ')
    require(np.array_equal(arrays['input_admitted'],arrays['history_mask'].all(axis=(1,2))&arrays['weather_mask'].all(axis=(1,2))&arrays['state_mask'].all(axis=1)),
            'Input admission must depend only on required original inputs')
    cohort=digest(dict(source_identity=source_identity,case_ids=arrays['case_id'].tolist(),input_admitted=arrays['input_admitted'].tolist(),history_hours=history_hours))
    arrays['feature_schema']=np.asarray(FULL_FEATURE_SCHEMA);arrays['cohort_admission_digest']=np.asarray(cohort);arrays['inference_source_identity']=np.asarray(source_identity)
    manifest=dict(schema=SCHEMA,execution_plan_sha256=authority.execution_plan_sha256,source_identity=source_identity,
                  source_inventory_sha256=authority.source_inventory_sha256,cohort_admission_digest=cohort,
                  feature_schema=FULL_FEATURE_SCHEMA,history_hours=history_hours,requested_cases=n,admitted_cases=int(arrays['input_admitted'].sum()),
                  first_origin_utc=iso(origins[0]),last_origin_utc=iso(origins[-1]),last_target_utc=iso(int(arrays['forecast_time'][-1,-1])),
                  arrays_sha256=_array_identity(arrays),quality_policy=PRIMARY_APPROVED,target_values_read=False,target_masks_read=False,
                  causal_assessment_history_may_overlap_other_origins_targets=True,weather_max_age_hours=18,soil_max_age_hours=8,
                  numerical_input_filling=False,model_calls=0,optimizer_updates=0,scaler_fits=0)
    for value in arrays.values():value.setflags(write=False)
    return dict(arrays=arrays,manifest=manifest)


def assemble_assessment_targets(observations, origins, authority):
    """Verification-only target tensor after authenticated prediction commit."""
    _gate(authority, verification=True)
    require(type(origins) in (list,tuple) and 1<=len(origins)<=MAX_ORIGINS, 'Bounded explicit assessment origins required')
    origins=tuple(origins)
    require(origins==tuple(sorted(set(origins))), 'Bounded ordered assessment origins required')
    for c in origins:clock(c)
    values=[];masks=[];flags=[];sources=[];reasons=[];coordinates=[]
    admit=quality_admitter(PRIMARY_APPROVED)
    index={};lo=origins[0]+HOUR;hi=origins[-1]+18*HOUR
    for row in observations:
        if row.valid_at < lo or row.valid_at > hi:continue
        require(row.site in GAUGES and type(row.valid_at) is int and row.valid_at % HOUR==0
                and (row.site,row.valid_at) not in index, 'Duplicate or invalid verification coordinate')
        index[row.site,row.valid_at]=row
    for c in origins:
        for site in GAUGES:
            selected=[index[site,t] for t in clock(c)['valid_times'] if (site,t) in index]
            t=target_window(selected,site=site,origin_anchor=c,horizons_hours=tuple(range(1,19)),admit_quality=admit)
            values.append(t['values']);masks.append(t['mask']);flags.append([json.dumps(x) for x in t['quality_flags']]);sources.append([x or '' for x in t['source_ids']]);reasons.append([x or '' for x in t['reasons']]);coordinates.append(t['times'])
    arrays=dict(target=np.asarray(values,dtype=np.float64),target_mask=np.asarray(masks,dtype=bool),target_time=np.asarray(coordinates,dtype=np.int64),
                target_quality_flags=_unicode_array(flags),target_source_ids=_unicode_array(sources),target_exclusion_reason=_unicode_array(reasons))
    for value in arrays.values():value.setflags(write=False)
    return dict(arrays=arrays,prediction_barrier_sha256=authority.prediction_barrier_sha256,
                target_imputation=False,quality_policy=PRIMARY_APPROVED)
