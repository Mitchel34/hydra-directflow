#!/usr/bin/env python3
"""Count development-corpus admission and source diagnostics; never select a primary cohort."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import psutil

from hydra_directflow import corpus_dataset as dataset
from hydra_directflow.contract import CORE_WEATHER_FEATURES
from hydra_directflow.local_data import GAUGES
from hydra_directflow.quality import PRIMARY_APPROVED
from hydra_directflow.reference import digest, require

MAX_SECONDS = 900
MAX_RSS = 4 * 1024**3
MAX_SHARDS = 1024
RESOLVED = 'RESOLVED_WITHIN_ACQUIRED_CALENDAR'
CODE_FILES = ('scripts/audit_corpus_admission.py', 'src/hydra_directflow/corpus_dataset.py',
              'src/hydra_directflow/contract.py', 'src/hydra_directflow/quality.py',
              'src/hydra_directflow/local_data.py', 'src/hydra_directflow/multicycle.py',
              'src/hydra_directflow/reference.py')


def code_identity():
    return {name: hashlib.sha256(dataset._read(ROOT / name, 512 * 1024)).hexdigest() for name in CODE_FILES}


class Guard:
    def __init__(self, max_seconds, max_rss_bytes):
        require(type(max_seconds) is int and 1 <= max_seconds <= MAX_SECONDS
                and type(max_rss_bytes) is int and 0 < max_rss_bytes <= MAX_RSS,
                'Explicit bounded runtime/RSS allowance required')
        self.max_seconds, self.max_rss_bytes = max_seconds, max_rss_bytes
        self.start, self.peak, self.process = time.monotonic(), 0, psutil.Process()
        self.check()

    def check(self):
        high_water = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.peak = max(self.peak, self.process.memory_info().rss, high_water)
        require(self.peak <= self.max_rss_bytes, 'Audit process RSS allowance exceeded')
        require(time.monotonic() - self.start <= self.max_seconds, 'Audit runtime allowance exceeded')

    def report(self):
        self.check()
        return dict(elapsed_seconds=time.monotonic() - self.start, observed_process_peak_rss_bytes=self.peak,
                    max_seconds=self.max_seconds, max_rss_bytes=self.max_rss_bytes,
                    rss_enforcement='COOPERATIVE_BETWEEN_BOUNDED_SHARD_READS_WITH_PROCESS_HIGH_WATER_CHECK')


def _mask_summary(mask):
    return dict(total_cells=int(mask.size), observed_cells=int(mask.sum()), missing_cells=int((~mask).sum()),
                complete_cases=int(mask.reshape(len(mask), -1).all(axis=1).sum()))


def _add_counts(destination, name, counts):
    target = destination.setdefault(name, Counter())
    target.update(counts)


def _reasons(batch, length):
    reasons, dimensions = [], {}
    for key, label in (('weather', 'NO_COMPLETE_SINGLE_WEATHER_RUN'), ('history', 'HISTORY_INCOMPLETE'),
                       ('state', 'SOIL_UNAVAILABLE_OR_STALE'), ('target', 'TARGET_INCOMPLETE')):
        mask = batch[key + '_mask'][:, -length:] if key == 'history' else batch[key + '_mask']
        dimensions[label] = ~mask.reshape(len(mask), -1).all(axis=1)
    for index in range(len(batch['case_id'])):
        reasons.append('|'.join(label for label, excluded in dimensions.items() if excluded[index]) or 'COMPLETE')
    return Counter(reasons), Counter({name: int(values.sum()) for name, values in dimensions.items()})


def _domain_counts(values, mask, valid, common):
    bad = mask & ~valid
    bad_cases = bad.reshape(len(mask), -1).any(axis=1)
    return dict(observed_cells=int(mask.sum()), violation_cells=int(bad.sum()),
                cases_with_violation=int(bad_cases.sum()), common_l168_cases_with_violation=int((bad_cases & common).sum()))


def _new_row(gauge, split, selections):
    mask_names = [f'history_L{length}_{name}' for length in (24, 168) for name in ('discharge', 'age_hours')]
    mask_names += list(CORE_WEATHER_FEATURES) + ['soil_fraction', 'soil_age_hours', 'target']
    mask_names += [f'target_H{lead:02d}' for lead in range(1, 19)]
    empty_mask = dict(total_cells=0, observed_cells=0, missing_cells=0, complete_cases=0)
    return dict(gauge_id=gauge, split=split, candidate_cases=0,
                candidate_selections={name: selection.report() for name, selection in selections.items()},
                mask_counts={name: Counter(empty_mask) for name in mask_names}, source_eligibility_counts=Counter(), source_calendar_counts=Counter(),
                view_reason_counts={'L24': Counter(), 'L168': Counter()},
                view_exclusion_dimensions={'L24': Counter(), 'L168': Counter()},
                weather=dict(selected_cases=0, unavailable_cases=0, older_than_latest_nominal_cases=0,
                             latest_nominal_cases=0, nominal_cycle_before_acquisition_cases=0,
                             common_l168_selected_cases=0, common_l168_older_than_nominal_cases=0,
                             age_hours_counts=Counter(), older_by_hours_counts=Counter(),
                             first_native_lead_counts=Counter(), last_native_lead_counts=Counter()),
                soil=dict(observed_cases=0, unavailable_cases=0, selected_cycle_cases=0,
                          selected_cycle_without_observation_cases=0, age_hours_counts=Counter(),
                          differs_from_weather_cycle_cases=0),
                outside_calendar_warmup=dict(unresolved_cases=0, weather_unresolved_cases=0, soil_unresolved_cases=0),
                physical_domain_diagnostics={name: Counter(observed_cells=0, violation_cells=0, cases_with_violation=0,
                    common_l168_cases_with_violation=0) for name in (*CORE_WEATHER_FEATURES, 'soil_fraction')}, batch_reads=0)


def _accumulate(row, batch):
    n = len(batch['case_id']); row['candidate_cases'] += n; row['batch_reads'] += 1
    common = batch['source_admission_mask']
    row['source_eligibility_counts'].update(batch['source_eligibility_reason'].tolist())
    row['source_calendar_counts'].update(batch['source_calendar_status'].tolist())
    for length in (24, 168):
        reasons, dimensions = _reasons(batch, length)
        row['view_reason_counts'][f'L{length}'].update(reasons)
        row['view_exclusion_dimensions'][f'L{length}'].update(dimensions)
        history_mask = batch['history_mask'][:, -length:]
        for column, name in enumerate(('discharge', 'age_hours')):
            _add_counts(row['mask_counts'], f'history_L{length}_{name}', _mask_summary(history_mask[..., column]))
    for column, name in enumerate(CORE_WEATHER_FEATURES):
        _add_counts(row['mask_counts'], name, _mask_summary(batch['weather_mask'][..., column]))
    for column, name in enumerate(('soil_fraction', 'soil_age_hours')):
        _add_counts(row['mask_counts'], name, _mask_summary(batch['state_mask'][:, column, None]))
    _add_counts(row['mask_counts'], 'target', _mask_summary(batch['target_mask']))
    for lead in range(18):
        _add_counts(row['mask_counts'], f'target_H{lead + 1:02d}', _mask_summary(batch['target_mask'][:, lead, None]))
    weather, soil, warmup = row['weather'], row['soil'], row['outside_calendar_warmup']
    for i, origin in enumerate(batch['origin_time']):
        nominal = ((int(origin) - 2 * 3600) // (6 * 3600)) * 6 * 3600
        weather['nominal_cycle_before_acquisition_cases'] += int(nominal < dataset.MIN_ORIGIN)
        cycle = str(batch['source_cycle'][i])
        if cycle:
            reference = dataset._cycle_epoch(cycle)
            difference = nominal - reference
            require(difference >= 0 and difference % (6 * 3600) == 0, 'Selected weather differs from nominal six-hour calendar')
            age, older = (int(origin) - reference) // 3600, difference > 0
            weather['selected_cases'] += 1; weather['older_than_latest_nominal_cases'] += int(older)
            weather['latest_nominal_cases'] += int(not older)
            weather['common_l168_selected_cases'] += int(common[i])
            weather['common_l168_older_than_nominal_cases'] += int(older and common[i])
            weather['age_hours_counts'][str(age)] += 1
            if older: weather['older_by_hours_counts'][str(difference // 3600)] += 1
            weather['first_native_lead_counts'][str(int(batch['weather_source_leads'][i, 0]))] += 1
            weather['last_native_lead_counts'][str(int(batch['weather_source_leads'][i, -1]))] += 1
        else:
            weather['unavailable_cases'] += 1
        soil_cycle = str(batch['soil_source_cycle'][i]); observed = bool(batch['state_mask'][i, 0])
        soil['selected_cycle_cases'] += int(bool(soil_cycle)); soil['observed_cases'] += int(observed)
        soil['unavailable_cases'] += int(not observed)
        soil['selected_cycle_without_observation_cases'] += int(bool(soil_cycle) and not observed)
        if observed:
            soil['age_hours_counts'][str(int(batch['state'][i, 1]))] += 1
            soil['differs_from_weather_cycle_cases'] += int(bool(cycle) and cycle != soil_cycle)
        status = str(batch['source_calendar_status'][i])
        warmup['unresolved_cases'] += int(status != RESOLVED)
        warmup['weather_unresolved_cases'] += int(status != RESOLVED and 'WEATHER' in status)
        warmup['soil_unresolved_cases'] += int(status != RESOLVED and 'SOIL' in status)
    values, masks = batch['weather'], batch['weather_mask']
    domains = ((CORE_WEATHER_FEATURES[0], values[..., 0], masks[..., 0], values[..., 0] >= 0),
               (CORE_WEATHER_FEATURES[1], values[..., 1], masks[..., 1], values[..., 1] > 0),
               (CORE_WEATHER_FEATURES[2], values[..., 2], masks[..., 2], (values[..., 2] >= 0) & (values[..., 2] <= 1)),
               (CORE_WEATHER_FEATURES[3], values[..., 3], masks[..., 3], values[..., 3] >= 0),
               ('soil_fraction', batch['state'][:, :1], batch['state_mask'][:, :1],
                (batch['state'][:, :1] >= 0) & (batch['state'][:, :1] <= 1)))
    for name, value, mask, valid in domains:
        _add_counts(row['physical_domain_diagnostics'], name, _domain_counts(value, mask, valid, common))


def audit_corpus(root, *, expected_plan_sha256, diagnostic_partial=False, batch_size=128,
                 max_seconds=MAX_SECONDS, max_rss_bytes=MAX_RSS):
    require(type(diagnostic_partial) is bool and type(batch_size) is int and 1 <= batch_size <= 128,
            'Explicit diagnostic mode and bounded batch size required')
    guard = Guard(max_seconds, max_rss_bytes); code = code_identity()
    root = Path(root).absolute()
    plan_raw = dataset._read(root / 'PLAN.json', 4 * 1024**2); plan = json.loads(plan_raw)
    require(digest({key: plan[key] for key in dataset.PLAN_KEYS}) == expected_plan_sha256,
            'Expected source plan digest differs')
    require(plan['history_hours'] == 168, 'Both lookbacks require a staged L168 corpus')
    require(sum(len(unit['chunks']) for unit in plan['units']) <= MAX_SHARDS, 'Audit shard-count allowance exceeded')
    snapshot = dataset.inspect_corpus(root, expected_plan_sha256=expected_plan_sha256); guard.check()
    require(snapshot.plan_file_sha256 == hashlib.sha256(plan_raw).hexdigest(), 'Source plan changed during audit')
    if not snapshot.complete and not diagnostic_partial:
        raise dataset.IncompleteCorpusError('Full admission audit requires every planned shard; use explicit diagnostic_partial only for provisional diagnostics')
    rows = []
    for split in snapshot.splits:
        for gauge in GAUGES:
            selections = {}
            for name, length, mode in (('all_candidates', 168, 'all_candidates'), ('natural_L24', 24, 'natural_complete'),
                                      ('natural_L168', 168, 'natural_complete'), ('common_L24', 24, 'common_l168_complete'),
                                      ('common_L168', 168, 'common_l168_complete')):
                selections[name] = snapshot.select(gauge_id=gauge, split=split, history_hours=length,
                                                    cohort_mode=mode, allow_partial=diagnostic_partial)
            require(selections['natural_L168'].case_count == selections['common_L168'].case_count
                    and selections['common_L24'].cohort_admission_digest == selections['common_L168'].cohort_admission_digest,
                    'Explicit common-case selections disagree')
            row = _new_row(gauge, split, selections)
            for batch in dataset.iter_batches(selections['all_candidates'], feature_view='full', batch_size=batch_size):
                guard.check(); _accumulate(row, batch)
            require(row['candidate_cases'] == selections['all_candidates'].case_count, 'Audit omitted candidate cases')
            row['candidate_counts'] = {name: selection.case_count for name, selection in selections.items()}
            row['natural_L24_cases_rescued_from_L168'] = selections['natural_L24'].case_count - selections['natural_L168'].case_count
            require(row['natural_L24_cases_rescued_from_L168'] >= 0, 'Short history lost long-complete cases')
            row['physical_domain_review_needed'] = any(item['violation_cells'] for item in row['physical_domain_diagnostics'].values())
            rows.append(row)
    require(code_identity() == code, 'Audit implementation changed during execution')
    require(hashlib.sha256(dataset._read(root / 'PLAN.json', 4 * 1024**2)).hexdigest() == snapshot.plan_file_sha256,
            'Source plan changed during audit')
    snapshot_report = snapshot.report()
    result = dict(schema='hydra-corpus-admission-audit/v1',
                  status='COMPLETE_SNAPSHOT_CANDIDATE_AUDIT' if snapshot.complete else 'PARTIAL_DIAGNOSTIC_NOT_FINAL_COHORT',
                  snapshot=snapshot_report, implementation_sha256=code,
                  producer_implementation_sha256=plan['implementation_sha256'],
                  source_identities={key: plan[key] for key in ('qualified_index_sha256', 'usgs_manifest_sha256', 'candidate_weights_sha256')},
                  quality=dict(input_policy=PRIMARY_APPROVED, target_policy=PRIMARY_APPROVED, original_flags_reconstructed=False),
                  splits=plan['splits'], all_candidate_cases=sum(row['candidate_cases'] for row in rows),
                  calendar_unavailable_cycle_ids=plan['calendar_unavailable_cycle_ids'], results=rows,
                  weather_reference=dict(definition='FLOOR_TO_6H_OF_C_MINUS_2H', older_cycle_count_is_not_a_gap_cause_diagnosis=True,
                                         global_source_availability_reaudited=False),
                  domain_rules={'precip_mm':'>=0', 'temp_k':'>0 K', 'specific_humidity':'0..1 inclusive',
                                'swe_kg_m2':'>=0', 'soil_fraction':'0..1 inclusive'},
                  physical_domain_review_needed=any(row['physical_domain_review_needed'] for row in rows),
                  counting=dict(unit='GAUGE_ORIGIN_CASES_AND_REPEATED_TENSOR_CELLS',
                                cells_are_unique_observations=False, exclusion_dimensions_may_overlap=True,
                                physical_domain_scope='MASK_ADMITTED_INPUT_CELLS_WITH_SEPARATE_COMMON_L168_CASE_COUNTS'),
                  execution=dict(raw_source_reads=0, network_requests=0, scaler_fits=0, optimizer_updates=0,
                                 source_files_written=False, upstream_sources_reaudited=False, protected_assessment_access=False,
                                 derived_discharge_values_decoded_by_existing_validation=True,
                                 discharge_magnitude_summaries_computed=False, scientific_cohort_adopted=False),
                  bounds=dict(max_shards=MAX_SHARDS, max_shard_expanded_bytes=dataset.MAX_SHARD_BYTES, batch_size=batch_size))
    require(result['all_candidate_cases'] == snapshot_report['available_candidate_cases'], 'Audit total candidate count differs')
    result['audit_content_sha256'] = digest(result)
    result['runtime'] = guard.report()
    return result


def write_report(report, output, *, source_root):
    output, source = Path(output).absolute(), Path(source_root).resolve()
    require(output.suffix == '.json' and not output.exists() and not output.is_symlink(), 'Use a fresh JSON report; no overwrites')
    require(not output.resolve().is_relative_to(source), 'Report must be outside the source corpus')
    dataset._directory(output.parent)
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    require(len(raw) <= 16 * 1024**2, 'Report byte bound exceeded')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.admission-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name); stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, output)  # Atomic publication; fails if a report appeared concurrently.
        fd = os.open(output.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--diagnostic-partial', action='store_true', help='Explicit provisional diagnostics; never a final cohort')
    parser.add_argument('--max-seconds', type=int, default=MAX_SECONDS)
    parser.add_argument('--max-rss-bytes', type=int, default=MAX_RSS)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    require(not args.report.exists() and not args.report.is_symlink(), 'Refusing to overwrite an existing report')
    require(1 <= args.max_seconds <= MAX_SECONDS, 'Runtime allowance must be 1..900 seconds')
    def deadline(signum, frame): raise TimeoutError('Audit wall-clock deadline reached')
    previous = signal.signal(signal.SIGALRM, deadline); signal.alarm(args.max_seconds)
    try:
        report = audit_corpus(args.corpus, expected_plan_sha256=args.expected_plan_sha256,
                              diagnostic_partial=args.diagnostic_partial, batch_size=args.batch_size,
                              max_seconds=args.max_seconds, max_rss_bytes=args.max_rss_bytes)
        write_report(report, args.report, source_root=args.corpus)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
    print(json.dumps({key: report[key] for key in ('status', 'all_candidate_cases', 'physical_domain_review_needed', 'audit_content_sha256')}))


if __name__ == '__main__': main()
