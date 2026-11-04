#!/usr/bin/env python3
"""Explicit, hash-bound operational amendment for mutable heartbeat snapshots.

The frozen primary supervisor and scientific worker are not edited. Only the
supervisor's reads of exact planned heartbeat paths use the additive reader.
Every other read, source check, worker command, budget and checkpoint stays under
the original qualified controller. Preparation never starts or resumes a fit.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_primary_campaign as controller
from hydra_directflow.campaign_ledger import CampaignLedger, digest
from hydra_directflow.heartbeat_snapshot import read_snapshot
from hydra_directflow.reference import require

CODE = ('scripts/recover_primary_campaign_heartbeat.py',
        'src/hydra_directflow/heartbeat_snapshot.py')
TESTS = ('tests/test_heartbeat_snapshot.py', 'tests/test_primary_heartbeat_recovery.py')
SCOPE = 'EXPLICIT_SAME_JOB_RESUME_WITH_SUPERVISOR_HEARTBEAT_SNAPSHOT_ONLY'
READ = controller.read


def implementation():
    return {name: controller.sha(READ(ROOT / name, 2 * 1024**2)) for name in CODE}


def qualified_receipt(path):
    path = Path(path).absolute()
    require(path == path.resolve() and path.parent == ROOT / 'reports', 'Local qualification report required')
    raw = READ(path)
    report = json.loads(raw)
    require(report['status'] == 'PASSED_OPERATIONAL_HEARTBEAT_RECOVERY_QUALIFICATION'
            and report['implementation_sha256'] == implementation(), 'Recovery implementation is not qualified')
    require(report['test_sha256'] == {name: controller.sha(READ(ROOT / name)) for name in TESTS},
            'Qualified recovery tests changed')
    require(report['tests']['failures'] == report['tests']['errors'] == 0 and report['tests']['passed'] > 0
            and report['model_calls'] == report['optimizer_updates'] == 0,
            'Passing no-model recovery verification required')
    return dict(path=str(path), sha256=controller.sha(raw))


def recovery_state(plan):
    """Read-only receipts and opaque checkpoint bytes; no deserialization or fits."""
    controller.verify_release(plan)
    ledger = CampaignLedger(plan['ledger_directory'])
    snapshot = ledger.snapshot()
    require(snapshot['plan_sha256'] == plan['plan_sha256'], 'Original ledger plan differs')
    require(not any(a['status'] == 'ACTIVE' for a in snapshot['attempts']), 'An active attempt already exists')
    first = next((j for j in snapshot['jobs'] if j['state'] != 'VERIFIED_TRAIN_EVAL'), None)
    require(first is not None and first['state'] == 'INTERRUPTED', 'First unfinished job must be interrupted')
    require(time.time() < min(first['deadline'], snapshot['campaign_deadline']), 'Original recovery deadline expired')
    checkpoint = ledger.latest_checkpoint(first['job_id'])
    require(checkpoint is not None, 'Latest verified checkpoint required')
    run = Path(plan['run_directory'])
    receipt_path = run / checkpoint['receipt_path']
    require(receipt_path.resolve() == receipt_path and receipt_path.is_relative_to(run), 'Checkpoint receipt path changed')
    require(json.loads(READ(receipt_path)) == checkpoint, 'Checkpoint file differs from ledger')
    for descriptor in checkpoint['artifacts'].values():
        path = run / descriptor['path']
        require(path.resolve() == path and path.is_relative_to(run), 'Checkpoint artifact path changed')
        raw = READ(path, 16 * 1024**2)
        require(len(raw) == descriptor['bytes'] and controller.sha(raw) == descriptor['sha256'],
                'Checkpoint artifact bytes changed')
    with ledger._reader() as connection:
        completed = connection.execute("SELECT job_position,receipt_sha,receipt_json FROM artifacts WHERE kind='COMPLETE' ORDER BY job_position").fetchall()
    require(len(completed) == first['position'], 'Completed prefix receipt count differs')
    completion_hashes = {}
    for position, receipt_sha, raw in completed:
        receipt = json.loads(raw)
        require(position == len(completion_hashes) and digest(receipt) == receipt_sha
                and receipt['job_id'] == plan['jobs'][position]['job_id'], 'Completed receipt identity differs')
        completion_hashes[receipt['job_id']] = receipt_sha
    return dict(first_interrupted=first, latest_checkpoint_receipt_sha256=digest(checkpoint),
                checkpoint_logical_updates=checkpoint['logical_updates'],
                completed_receipt_sha256=completion_hashes, attempts=snapshot['attempts'],
                optimizer_updates_entered=snapshot['optimizer_updates_entered'],
                optimizer_updates_confirmed=snapshot['optimizer_updates_confirmed'],
                output_bytes_reserved=snapshot['output_bytes_reserved'],
                campaign_started_at=snapshot['campaign_started_at'],
                campaign_deadline=snapshot['campaign_deadline'])


def prepare(plan_path, plan_sha, qualification, authorization):
    plan_path = controller.private(plan_path)
    plan = controller.authenticated_plan(plan_path, plan_sha)
    require(authorization.get('source') == 'DIRECT_USER_MESSAGE'
            and authorization.get('scope') == SCOPE
            and isinstance(authorization.get('quote'), str) and authorization['quote'].strip()
            and isinstance(authorization.get('recorded_at_utc'), str), 'Explicit user recovery instruction required')
    result = dict(schema='hydra-primary-heartbeat-recovery-amendment/v1',
                  created_at_utc=controller.utc(), scope=SCOPE, authorization=authorization,
                  original_plan_path=str(plan_path), original_plan_sha256=plan_sha,
                  original_protocol_sha256=plan['protocol_sha256'],
                  implementation_sha256=implementation(), qualification=qualified_receipt(qualification),
                  original_state=recovery_state(plan), scientific_method_changes=False,
                  resource_limit_changes=False, paid_cloud_jobs=0, protected_assessment_access=False)
    result['amendment_sha256'] = digest(result)
    return result


def validate_amendment(amendment, expected):
    require(amendment['amendment_sha256'] == expected
            and digest({k: v for k, v in amendment.items() if k != 'amendment_sha256'}) == expected,
            'Exact recovery amendment SHA required')
    require(amendment['schema'] == 'hydra-primary-heartbeat-recovery-amendment/v1'
            and amendment['scope'] == SCOPE and amendment['scientific_method_changes'] is False
            and amendment['resource_limit_changes'] is False and amendment['paid_cloud_jobs'] == 0
            and amendment['protected_assessment_access'] is False, 'Only the bounded operational amendment is supported')
    require(amendment['implementation_sha256'] == implementation(), 'Recovery implementation changed')
    require(qualified_receipt(amendment['qualification']['path']) == amendment['qualification'], 'Recovery qualification changed')
    auth = amendment['authorization']
    require(auth.get('source') == 'DIRECT_USER_MESSAGE' and auth.get('scope') == SCOPE
            and isinstance(auth.get('quote'), str) and auth['quote'].strip()
            and isinstance(auth.get('recorded_at_utc'), str), 'Explicit recovery authorization missing')
    plan = controller.authenticated_plan(amendment['original_plan_path'], amendment['original_plan_sha256'])
    require(plan['protocol_sha256'] == amendment['original_protocol_sha256'], 'Frozen protocol identity differs')
    require(recovery_state(plan) == amendment['original_state'], 'Original recovery state changed; no implicit retry')
    return plan


@contextmanager
def heartbeat_reads(plan):
    """Apply only to supervisor telemetry; immutable data reads stay strict."""
    original = controller.read
    run = Path(plan['run_directory'])
    job_ids = {j['job_id'] for j in plan['jobs']}

    def scoped_read(path, maximum=8 * 1024**2):
        path = Path(path)
        if path.name != 'heartbeat.json' or not path.is_relative_to(run):
            return original(path, maximum)
        parts = path.relative_to(run).parts
        require(len(parts) == 4 and parts[0] == 'jobs' and parts[1] in job_ids
                and re.fullmatch(r'attempt-[0-9]{6}', parts[2]) is not None
                and int(parts[2][8:]) > 0 and maximum == 64 * 1024,
                'Unexpected heartbeat path or byte bound')
        require(path.absolute() == path and path.resolve() == path, 'Heartbeat path alias or symlink refused')
        return read_snapshot(path, maximum)

    controller.read = scoped_read
    try:
        yield
    finally:
        controller.read = original


def execute(amendment_path, expected):
    amendment_path = controller.private(amendment_path)
    amendment = json.loads(READ(amendment_path))
    plan = validate_amendment(amendment, expected)
    run = Path(plan['run_directory'])
    require(controller.shutil.disk_usage(run).free >= plan['limits']['min_free_bytes'], 'Free-space floor reached')
    require(controller.folder_bytes(run) + 1024**2 <= plan['limits']['max_output_bytes'], 'Recovery bookkeeping exceeds output bound')
    # Original two-GiB bookkeeping reservation covers this small immutable
    # operational receipt; no allowance, clock, source or completed fit is reset.
    directory = run / 'operational-recovery'
    require(not directory.is_symlink(), 'Recovery directory symlink refused')
    directory.mkdir(mode=0o700, exist_ok=True)
    receipt = dict(schema='hydra-primary-operational-recovery-start/v1', started_at_utc=controller.utc(),
                   amendment=amendment, original_plan_sha256=plan['plan_sha256'],
                   original_protocol_sha256=plan['protocol_sha256'], resume_requested=True)
    controller.atomic(directory / (expected + '.json'), receipt, new=True)
    # The unchanged supervisor retains its own exclusive process lock and
    # launches the unchanged worker with --resume only for the first job.
    with heartbeat_reads(plan):
        return controller.supervise(Path(amendment['original_plan_path']), plan, resume=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--execute', action='store_true')
    for name in ('plan', 'qualification', 'authorization', 'amendment'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--approved-plan-sha256')
    parser.add_argument('--approved-amendment-sha256')
    args = parser.parse_args()
    require(args.amendment is not None, 'Explicit private amendment path required')
    if args.prepare:
        require(all((args.plan, args.qualification, args.authorization, args.approved_plan_sha256)), 'Complete preparation inputs required')
        authorization = json.loads(READ(controller.private(args.authorization), 16 * 1024))
        amendment = prepare(args.plan, args.approved_plan_sha256, args.qualification, authorization)
        controller.atomic(controller.private(args.amendment), amendment, new=True)
        print(json.dumps(dict(status='PREPARED_NO_FITS_OR_RESUME', amendment_sha256=amendment['amendment_sha256'])))
    else:
        require(args.approved_amendment_sha256, 'Exact approved recovery amendment required')
        result = execute(args.amendment, args.approved_amendment_sha256)
        print(json.dumps({k: result[k] for k in ('status', 'completed_jobs', 'optimizer_updates_confirmed')}))


if __name__ == '__main__':
    main()
