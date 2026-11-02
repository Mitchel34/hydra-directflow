#!/usr/bin/env python3
"""Count verified common-cohort cases in proposed fold windows; no fits or policy adoption."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from hydra_directflow import corpus_dataset as dataset
from hydra_directflow.local_data import GAUGES
from hydra_directflow.quality import PRIMARY_APPROVED
from hydra_directflow.reference import digest, require
from audit_corpus_admission import Guard, MAX_RSS, MAX_SECONDS, MAX_SHARDS, code_identity as reader_identity, write_report
from plan_scientific_workload import fold_windows, epoch, iso, calendar_window

ROLES = ('gradient_fit', 'inner_stopping', 'outer_scoring')
CANONICAL_SPLITS = [
    dict(name='candidate_training', start_utc='2018-07-12T00:00:00Z', end_utc='2023-07-31T00:00:00Z'),
    dict(name='candidate_validation', start_utc='2023-07-31T00:00:00Z', end_utc='2024-07-31T00:00:00Z'),
]


def code_identity():
    return {**reader_identity(), **{name: hashlib.sha256(dataset._read(ROOT / name, 512 * 1024)).hexdigest()
        for name in ('scripts/plan_fold_admitted_counts.py', 'scripts/plan_scientific_workload.py')}}


def load_admission_report(path, *, expected_sha256, expected_plan_sha256):
    raw = dataset._read(path, 16 * 1024**2)
    require(dataset._hex(expected_sha256) and hashlib.sha256(raw).hexdigest() == expected_sha256,
            'Expected admission report SHA256 differs')
    report = json.loads(raw)
    require(report['schema'] == 'hydra-corpus-admission-audit/v1'
            and report['status'] == 'COMPLETE_SNAPSHOT_CANDIDATE_AUDIT'
            and report['snapshot']['status'] == 'COMPLETE_COMMITTED_CORPUS_SNAPSHOT'
            and not report['snapshot']['missing_chunks'], 'Complete admission audit required')
    require(report['audit_content_sha256'] == digest({key: value for key, value in report.items()
                                                     if key not in ('audit_content_sha256', 'runtime')}),
            'Admission audit content digest differs')
    require(report['snapshot']['plan_sha256'] == expected_plan_sha256 and dataset._hex(expected_plan_sha256),
            'Admission audit source plan differs')
    require(report['implementation_sha256'] == reader_identity(), 'Current reader/audit code differs from admitted snapshot')
    require(report['splits'] == CANONICAL_SPLITS and report['snapshot']['stored_history_hours'] == 168,
            'Canonical full development intervals and L168 store required')
    require(report['quality'] == dict(input_policy=PRIMARY_APPROVED, target_policy=PRIMARY_APPROVED,
                                     original_flags_reconstructed=False), 'Admission quality policy differs')
    require(report['execution']['scientific_cohort_adopted'] is False
            and report['execution']['protected_assessment_access'] is False
            and report['execution']['optimizer_updates'] == report['execution']['scaler_fits'] == 0,
            'Admission report execution scope differs')
    return report


def selected_coordinates(selection):
    """Use membership and coordinates already proven exactly by inspect_corpus.

    Its numerical shard gate has checked stored case IDs and every H18 target
    against these chunk origins. This does not reopen or summarize Q values.
    """
    require(selection.cohort_mode == 'common_l168_complete' and selection.history_hours == 168
            and selection.inventory.complete and not selection.allow_partial, 'Complete explicit common-L168 selection required')
    gauge_index = GAUGES.index(selection.gauge_id)
    result = []
    for shard_index, rows in selection.selected_rows:
        shard = selection.inventory.shards[shard_index]
        require(shard.unit['split'] == selection.split and rows == tuple(sorted(set(rows))),
                'Selected shard/row order differs')
        first = epoch(shard.chunk['first_origin_utc']); count = shard.chunk['origin_count']
        for row in rows:
            require(type(row) is int and 0 <= row < count*3 and row % 3 == gauge_index,
                    'Selected gauge/row differs')
            origin = first + row // 3 * 3600
            result.append((origin, selection.gauge_id + ':' + iso(origin)))
    require(len(result) == selection.case_count and all(a[0] < b[0] for a, b in zip(result, result[1:])),
            'Selected coordinates are duplicate, unordered or incomplete')
    return result


def count_role(records, window):
    expected = calendar_window(window['start_utc'], window['end_exclusive_utc'],
                               refit_at=window['refit_at_utc'],
                               simulated_label_delay_hours=window['simulated_label_delay_hours'])
    require(window == expected, 'Role calendar window differs from exact proposed mechanics')
    require(all(type(origin) is int and origin % 3600 == 0 for origin, _ in records)
            and all(a[0] < b[0] for a, b in zip(records, records[1:])), 'Unique chronological exact-hour coordinates required')
    start, end = epoch(window['start_utc']), epoch(window['end_exclusive_utc'])
    refit = epoch(window['refit_at_utc']) if window['refit_at_utc'] is not None else None
    origin_inside = [(origin, case_id) for origin, case_id in records if start <= origin < end]
    strict = [(origin, case_id) for origin, case_id in origin_inside if origin + 18*3600 < end]
    admitted = [(origin, case_id) for origin, case_id in strict
                if refit is None or origin + (18 + window['simulated_label_delay_hours'])*3600 <= refit]
    require(len(admitted) <= window['calendar_origins'], 'Admitted common cases exceed calendar support')
    return dict(window=window, admitted_cases=len(admitted), case_ids_sha256=digest([case_id for _, case_id in admitted]),
                parent_selection_cases=len(records), origin_window_common_cases=len(origin_inside),
                strict_target_window_excluded_common_cases=len(origin_inside) - len(strict),
                refit_label_delay_excluded_common_cases=len(strict) - len(admitted),
                calendar_positions_without_common_case=window['calendar_origins'] - len(admitted),
                first_admitted_origin_utc=iso(admitted[0][0]) if admitted else None,
                last_admitted_origin_utc=iso(admitted[-1][0]) if admitted else None,
                last_admitted_target_utc=iso(admitted[-1][0] + 18*3600) if admitted else None,
                zero_row_support=not admitted)


def count_scenarios(coordinates, selections, *, guard):
    scenarios = []
    for delay in (0, 2):
        folds = []
        for fold in fold_windows(simulated_label_delay_hours=delay):
            gauges = []
            for gauge in GAUGES:
                roles = {}
                for role in ROLES:
                    split = 'candidate_validation' if fold['name'] == 'final_development' and role == 'outer_scoring' else 'candidate_training'
                    selected = selections[(gauge, split)]
                    roles[role] = dict(**count_role(coordinates[(gauge, split)], fold[role]), source_split=split,
                                       parent_selection_digest=selected.selection_digest,
                                       parent_cohort_admission_digest=selected.cohort_admission_digest)
                    guard.check()
                gauges.append(dict(gauge_id=gauge, roles=roles))
            folds.append(dict(name=fold['name'], role=fold['role'], gauge_results=gauges))
        scenarios.append(dict(simulated_label_delay_hours=delay, folds=folds))
    return scenarios


def build_plan(corpus_root, admission_report_path, *, expected_plan_sha256, expected_admission_sha256,
               max_seconds=MAX_SECONDS, max_rss_bytes=MAX_RSS):
    guard = Guard(max_seconds, max_rss_bytes); code = code_identity()
    audit = load_admission_report(admission_report_path, expected_sha256=expected_admission_sha256,
                                  expected_plan_sha256=expected_plan_sha256)
    require(len(audit['snapshot']['observed_commits']) <= MAX_SHARDS, 'Bounded source shard inventory required')
    snapshot = dataset.inspect_corpus(corpus_root, expected_plan_sha256=expected_plan_sha256); guard.check()
    require(snapshot.complete and snapshot.report() == audit['snapshot'], 'Current complete snapshot differs from the admission report')
    require(tuple(split['name'] for split in CANONICAL_SPLITS) == snapshot.splits, 'Source split inventory differs')
    audit_rows = {(row['gauge_id'], row['split']): row for row in audit['results']}
    require(len(audit_rows) == len(audit['results']) == 6, 'One audit row per gauge/split required')
    selections, coordinates = {}, {}
    for split in snapshot.splits:
        for gauge in GAUGES:
            key = (gauge, split)
            selected = snapshot.select(gauge_id=gauge, split=split, history_hours=168,
                                       cohort_mode='common_l168_complete', allow_partial=False)
            require(selected.report() == audit_rows[key]['candidate_selections']['common_L168']
                    and selected.case_count == audit_rows[key]['candidate_counts']['common_L168'],
                    'Common-case selection differs from admission audit')
            selections[key] = selected; coordinates[key] = selected_coordinates(selected); guard.check()
    scenarios = count_scenarios(coordinates, selections, guard=guard)
    require(code_identity() == code, 'Count-plan implementation changed during execution')
    require(hashlib.sha256(dataset._read(admission_report_path, 16 * 1024**2)).hexdigest() == expected_admission_sha256,
            'Admission report changed during execution')
    result = dict(schema='hydra-proposed-fold-admitted-counts/v1', status='COMPLETE_SNAPSHOT_PROPOSED_FOLD_COUNTS',
                  scope_fit_count=180, scope_status='USER_CONFIRMED', window_policy_status='PROPOSED_UNRESOLVED',
                  final_scientific_cohort=False, cohort_mode='common_l168_complete',
                  snapshot_sha256=snapshot.snapshot_sha256, source_plan_sha256=snapshot.plan_sha256,
                  source_plan_file_sha256=snapshot.plan_file_sha256, admission_report_sha256=expected_admission_sha256,
                  audit_content_sha256=audit['audit_content_sha256'], source_identities=audit['source_identities'],
                  quality=audit['quality'], implementation_sha256=code,
                  producer_implementation_sha256=audit['producer_implementation_sha256'],
                  observed_shard_commits=audit['snapshot']['observed_commits'],
                  parent_selections=[selections[(gauge, split)].report() for split in snapshot.splits for gauge in GAUGES],
                  physical_domain_review_needed=audit['physical_domain_review_needed'], scenarios=scenarios,
                  case_id_hash_encoding='CANONICAL_JSON_LIST_IN_CHRONOLOGICAL_ORDER_SHA256',
                  execution=dict(raw_source_reads=0, network_requests=0, scaler_fits=0, optimizer_updates=0, cloud_jobs=0,
                                 protected_assessment_access=False, source_files_written=False,
                                 derived_values_decoded_by_existing_snapshot_validation=True,
                                 discharge_magnitude_summaries_computed=False, cohort_or_window_policy_adopted=False),
                  limitations=['Counts use actual common-L168 membership, not natural-L24 extra cases',
                               'Strict origin/H18 partition bounds apply under both conventions; optional label availability is inclusive at refit',
                               'Role cutoffs are declared dates, never inferred from surviving observations',
                               'Observed boundary exclusions do not include origins already purged from the global corpus',
                               'Calendar positions without common cases are not a diagnosis of upstream source gaps',
                               'Both label-delay scenarios remain proposed; no operational runtime or publication-vintage claim',
                               'Source raw/cache bodies are not reaudited; immutable derived shard membership and clocks are verified'])
    result['fold_counts_sha256'] = digest(result); result['runtime'] = guard.report()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--admission-report', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--expected-admission-sha256', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    require(not args.report.exists() and not args.report.is_symlink(), 'Refusing to overwrite existing report')
    require(args.report.resolve() != args.admission_report.resolve(), 'Output cannot replace the admission report')
    def deadline(signum, frame): raise TimeoutError('Fold-count plan deadline reached')
    previous = signal.signal(signal.SIGALRM, deadline); signal.alarm(MAX_SECONDS)
    try:
        result = build_plan(args.corpus, args.admission_report, expected_plan_sha256=args.expected_plan_sha256,
                            expected_admission_sha256=args.expected_admission_sha256)
        write_report(result, args.report, source_root=args.corpus)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
    print(json.dumps({key: result[key] for key in ('status', 'scope_fit_count', 'snapshot_sha256', 'fold_counts_sha256')}))


if __name__ == '__main__': main()
