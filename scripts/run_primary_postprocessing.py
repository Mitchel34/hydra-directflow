#!/usr/bin/env python3
"""Bounded development-only postprocessing of the completed frozen campaign.

No model calls, checkpoint loading, new data requests or protected data access.
All outputs share one precharged GiB inside the original campaign's output cap.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import psutil
import run_primary_campaign as driver
from hydra_directflow.campaign_ledger import CampaignLedger, digest
from hydra_directflow.corpus_dataset import _read
from hydra_directflow.reference import require

FOLDS = ('cv_1', 'cv_2', 'cv_3', 'final_development')
GAUGES = ('03161000', '03164000', '03479000')
CATALOG = 'reports/development-coverage-regimes-v2.json'
CATALOG_SHA = '8e82f613165a5d452c74a30783b02e1c0ae5fb5d93dbe2e338d0010c215e8c0a'
CODE = ('scripts/run_primary_postprocessing.py', 'src/hydra_directflow/post_data.py',
        'src/hydra_directflow/post_events.py', 'src/hydra_directflow/post_uncertainty.py',
        'scripts/postprocess_primary_events.py', 'scripts/postprocess_primary_uncertainty.py',
        'scripts/postprocess_primary_residuals.py',
        'scripts/plot_primary_development.py', 'scripts/audit_coverage_regimes.py',
        'src/hydra_directflow/coverage_analysis.py', 'src/hydra_directflow/cache.py',
        'src/hydra_directflow/development_export.py', 'src/hydra_directflow/multicycle.py',
        'src/hydra_directflow/telemetry.py')
GIB = 1024**3
UNCERTAINTY = dict(draws=2000, master_seed=20261102, block_hours=[168,336,672],
    fold_order=list(FOLDS), stream_mapping='SeedSequence(master).spawn(12); block-major/fold-minor; one uint64 child seed',
    same_draws_for_all_gauges_models_seeds=True)


def sha(raw): return hashlib.sha256(raw).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def canonical(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def prepare(original_path):
    original = json.loads(_read(original_path, 8 * 1024**2))
    original = driver.authenticated_plan(original_path, original['plan_sha256'])
    ledger = CampaignLedger(original['ledger_directory']); state = ledger.snapshot()
    require(state['complete'] and all(j['state'] == 'VERIFIED_TRAIN_EVAL' for j in state['jobs']), 'All180 completed fits required')
    require(state['output_bytes_reserved'] + GIB <= original['limits']['max_output_bytes'], 'No remaining output allowance')
    require(sha(_read(ROOT / CATALOG, 16 * 1024**2)) == CATALOG_SHA, 'Frozen event catalog differs')
    result = dict(schema='hydra-primary-postprocessing-plan/v1', created_at_utc=utc(),
                  original_plan_path=str(original_path), original_plan_sha256=original['plan_sha256'],
                  protocol_sha256=original['protocol_sha256'], corpus_root=original['corpus_root'],
                  run_directory=str(Path(original['run_directory']) / 'postprocessing-v1'),
                  campaign_directory=original['run_directory'], ledger_directory=original['ledger_directory'],
                  original_campaign_deadline=state['campaign_deadline'],
                  original_output_reserved=state['output_bytes_reserved'],
                  implementation_sha256={name: sha(_read(ROOT / name, 2 * 1024**2)) for name in CODE},
                  catalog_sha256=CATALOG_SHA,
                  limits=dict(max_seconds=21600, max_rss_bytes=3 * GIB, max_new_output_bytes=GIB,
                              max_read_bytes=8 * GIB, min_free_bytes=original['limits']['min_free_bytes']),
                  uncertainty=deepcopy(UNCERTAINTY),
                  residual_diagnostics=dict(lag_hours=[1,6,12,24,48,72,168,336,672],
                    metric='PAIRWISE_CENTERED_PEARSON_ON_ORIGINAL_HOURLY_CALENDAR',
                    significance_claim=False,automatic_block_selection=False),
                  protected_assessment_access=False, paid_cloud_jobs=0, model_calls=0)
    result['plan_sha256'] = digest(result)
    return result


class Guard:
    def __init__(self, plan):
        self.plan = plan; self.run = Path(plan['run_directory']); self.record_path = self.run / 'RESOURCE.json'
        self.lock_fd = os.open(self.run / 'writer.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.record = json.loads(_read(self.record_path, 128 * 1024))
            require(self.record['plan_sha256'] == plan['plan_sha256'], 'Postprocessing resource plan differs')
        except BaseException:
            os.close(self.lock_fd); raise

    def close(self):
        if self.lock_fd is not None:
            self.save(); fcntl.flock(self.lock_fd, fcntl.LOCK_UN); os.close(self.lock_fd); self.lock_fd = None

    def save(self): driver.atomic(self.record_path, self.record)

    def check(self):
        require('torch' not in sys.modules, 'Postprocessing cannot import Torch or models')
        now = time.time(); process = psutil.Process(); rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try: rss += child.memory_info().rss
            except psutil.NoSuchProcess: pass
        self.record['peak_rss_bytes'] = max(self.record['peak_rss_bytes'], rss)
        require(now < min(self.record['started_epoch'] + self.plan['limits']['max_seconds'],
                          self.plan['original_campaign_deadline']), 'Persistent postprocessing deadline reached')
        require(rss <= self.plan['limits']['max_rss_bytes'], 'Postprocessing memory bound reached')
        require(shutil.disk_usage(self.run).free >= self.plan['limits']['min_free_bytes'], 'Postprocessing free-space floor reached')
        require(driver.folder_bytes(self.run) <= self.plan['limits']['max_new_output_bytes'], 'Postprocessing output bound reached')

    def read_budget(self, size):
        require(type(size) is int and size >= 0, 'Exact nonnegative read charge required')
        self.check()
        require(self.record['read_bytes_reserved'] + size <= self.plan['limits']['max_read_bytes'], 'Postprocessing read allowance exhausted')
        self.record['read_bytes_reserved'] += size; self.save()

    def read(self, path, maximum):
        size = Path(path).lstat().st_size; self.read_budget(size)
        return _read(path, maximum)

    def write(self, relative, raw):
        require(isinstance(relative,str) and not Path(relative).is_absolute()
                and all(p not in ('.','..') for p in Path(relative).parts), 'Safe relative output path required')
        path = self.run / relative
        require(path == path.resolve() and path.is_relative_to(self.run), 'Output path escaped postprocessing run')
        require(len(raw) + driver.folder_bytes(self.run) <= self.plan['limits']['max_new_output_bytes'], 'Output size exceeds bound')
        self.check(); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            require(_read(path, len(raw)) == raw, 'Preserve conflicting previous postprocessing output')
        else:
            temporary = path.with_name(path.name + '.writing')
            require(not temporary.exists(), 'Preserve unfinished postprocessing output')
            with temporary.open('xb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            os.link(temporary, path); temporary.unlink(); driver.sync(path.parent)
        return dict(path=relative, bytes=len(raw), sha256=sha(raw))


def start(plan):
    run = Path(plan['run_directory'])
    if run.exists():
        require(json.loads(_read(run / 'PLAN.json', 2 * 1024**2)) == plan, 'Existing postprocessing plan differs')
        return Guard(plan)
    require(shutil.disk_usage(run.parent).free >= plan['limits']['min_free_bytes'], 'Free-space floor before reservation')
    original = json.loads(_read(Path(plan['original_plan_path']), 8 * 1024**2))
    ledger = CampaignLedger(plan['ledger_directory'])
    with ledger.writer():
        ledger.initialize(original)
        state = ledger.snapshot()
        require(state['complete'] and state['output_bytes_reserved'] == plan['original_output_reserved'], 'Campaign state/reservation changed')
        require(time.time() < state['campaign_deadline'], 'Original campaign deadline expired')
        token = ledger.reserve_output(plan['limits']['max_new_output_bytes'], kind='POSTPROCESSING_V1')
    run.mkdir(mode=0o700)
    driver.atomic(run / 'PLAN.json', plan, new=True)
    driver.atomic(run / 'RESOURCE.json', dict(plan_sha256=plan['plan_sha256'], started_epoch=time.time(),
                  started_at_utc=utc(), read_bytes_reserved=0, peak_rss_bytes=0,
                  original_ledger_output_reservation_token=token,
                  original_ledger_output_reservation_bytes=GIB), new=True)
    return Guard(plan)


def checked_plan(path, expected):
    plan = json.loads(_read(driver.private(path), 2 * 1024**2))
    require(plan['plan_sha256'] == expected and digest({k:v for k,v in plan.items() if k!='plan_sha256'}) == expected,
            'Exact postprocessing plan SHA required')
    original = driver.authenticated_plan(Path(plan['original_plan_path']), plan['original_plan_sha256'])
    require(plan['run_directory'] == str(Path(original['run_directory']) / 'postprocessing-v1')
            and plan['campaign_directory']==original['run_directory']
            and plan['corpus_root'] == original['corpus_root'] and plan['ledger_directory'] == original['ledger_directory']
            and plan['protocol_sha256'] == original['protocol_sha256'], 'Original study bindings changed')
    require(plan['implementation_sha256'] == {n:sha(_read(ROOT/n, 2*1024**2)) for n in CODE}, 'Postprocessing implementation changed')
    require(plan['limits'] == dict(max_seconds=21600, max_rss_bytes=3*GIB, max_new_output_bytes=GIB,
                max_read_bytes=8*GIB, min_free_bytes=original['limits']['min_free_bytes']), 'Fixed postprocessing limits differ')
    require(plan['uncertainty']==UNCERTAINTY, 'Frozen uncertainty execution settings differ')
    require(plan['residual_diagnostics']==dict(lag_hours=[1,6,12,24,48,72,168,336,672],
                metric='PAIRWISE_CENTERED_PEARSON_ON_ORIGINAL_HOURLY_CALENDAR',
                significance_claim=False,automatic_block_selection=False), 'Descriptive residual diagnostic settings differ')
    require(sha(_read(ROOT/CATALOG,16*1024**2)) == plan['catalog_sha256'] == CATALOG_SHA, 'Frozen event catalog differs')
    state=CampaignLedger(plan['ledger_directory']).snapshot()
    require(state['complete'] and state['campaign_deadline']==plan['original_campaign_deadline'], 'Original campaign deadline/state changed')
    reserved=plan['original_output_reserved'] + (GIB if Path(plan['run_directory']).exists() else 0)
    require(state['output_bytes_reserved']==reserved, 'Postprocessing reservation accounting differs')
    require(plan['protected_assessment_access'] is False and plan['paid_cloud_jobs']==plan['model_calls']==0,
            'Development-only no-model scope required')
    return plan, original


def audit(plan, original, guard):
    from report_primary_seed import _ledger_records
    from hydra_directflow.post_data import build_group
    summary = json.loads(guard.read(Path(original['run_directory'])/'reports/seed-43/summary.json', 16*1024**2))
    by_id = {row['job_id']:row for row in summary['jobs']}
    records = _ledger_records(original, Path(original['ledger_directory']), 43)
    for row in records: row['label_source_manifest_sha256'] = by_id[row['job']['job_id']]['label_source_manifest_sha256']
    groups=[]
    for fold in FOLDS:
        for gauge in GAUGES:
            guard.check(); name=f'{fold}__{gauge}'; done=guard.run/'groups'/name/'COMMIT.json'
            if done.exists():
                record=json.loads(guard.read(done, 2*1024**2))
                require(record['group']==name and record['fold']==fold and record['gauge_id']==gauge
                        and record['plan_sha256']==plan['plan_sha256']
                        and set(record['artifacts'])=={'arrays','manifest','audit'}, 'Committed group identity differs')
                for kind, descriptor in record['artifacts'].items():
                    suffix='npz' if kind=='arrays' else 'json'
                    require(set(descriptor)=={'path','bytes','sha256'}
                            and descriptor['path']==f'groups/{name}/{kind}.{suffix}', 'Committed group artifact path differs')
                    body=guard.read(guard.run/descriptor['path'], 256*1024**2)
                    require(len(body)==descriptor['bytes'] and sha(body)==descriptor['sha256'], 'Committed group changed')
                groups.append(record);continue
            arrays,manifest,result=build_group(original, Path(original['run_directory']), Path(original['corpus_root']),
                                               records,fold,gauge,check=guard.check,read_budget=guard.read_budget)
            buffer=io.BytesIO();np.savez_compressed(buffer,**arrays)
            artifacts=dict(arrays=guard.write(f'groups/{name}/arrays.npz',buffer.getvalue()),
                           manifest=guard.write(f'groups/{name}/manifest.json',canonical(manifest)),
                           audit=guard.write(f'groups/{name}/audit.json',canonical(result)))
            record=dict(group=name,fold=fold,gauge_id=gauge,plan_sha256=plan['plan_sha256'],artifacts=artifacts)
            guard.write(f'groups/{name}/COMMIT.json',canonical(record));groups.append(record)
            del arrays,manifest,result,buffer
            print(json.dumps(dict(stage='AUDITED_GROUP',group=name,groups=len(groups))),flush=True)
    result=dict(schema='hydra-primary-postprocessing-audit/v1',status='ALL180_PREDICTIONS_INDEPENDENTLY_AUDITED',
                plan_sha256=plan['plan_sha256'],groups=groups,groups_total=12,model_calls=0,
                protected_assessment_access=False)
    guard.write('AUDIT.json',canonical(result));guard.save();return result


def figures(plan,original,guard):
    summary=Path(original['run_directory'])/'reports/seed-43/summary.json'
    raw=guard.read(summary,16*1024**2)
    # Rendering has no source/model access. This conservative reservation
    # covers its re-reading of the prepared groups, references and summaries.
    guard.read_budget(GIB)
    for name in ('AUDIT.json','EVENTS.json','UNCERTAINTY.json'):
        value=json.loads(guard.read(guard.run/name,64*1024**2))
        require(value['plan_sha256']==plan['plan_sha256'],'Figure stage evidence differs')
    output=guard.run/'figures';require(not output.exists(),'Preserve already rendered figures')
    command=['/opt/homebrew/bin/python3.11','-I','-B',str(ROOT/'scripts/plot_primary_development.py'),
        '--run',str(guard.run),'--summary',str(summary),'--summary-sha256',sha(raw),'--output',str(output)]
    log=guard.run/'figure-render.log'
    child=None
    try:
        with log.open('xb') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'MPLBACKEND':'Agg','MPLCONFIGDIR':str(guard.run/'matplotlib-cache')},
                                   stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            while child.poll() is None:
                guard.check();require(log.stat().st_size<=1024**2,'Figure renderer log limit reached');time.sleep(.5)
            require(child.returncode==0,'Figure renderer failed; retained log and partial outputs')
        guard.check();manifest=json.loads(guard.read(output/'MANIFEST.json',1024**2));guard.save()
        return dict(status=manifest['status'],figures=len(manifest['figures']),plan_sha256=plan['plan_sha256'])
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    stages=p.add_mutually_exclusive_group(required=True)
    for stage in ('prepare','audit','events','uncertainty','diagnostics','figures'): stages.add_argument('--'+stage,action='store_true')
    p.add_argument('--original-plan',type=Path);p.add_argument('--plan',type=Path,required=True);p.add_argument('--approved-plan-sha256')
    args=p.parse_args()
    if args.prepare:
        require(args.original_plan is not None,'Original frozen plan required');plan=prepare(driver.private(args.original_plan))
        driver.atomic(driver.private(args.plan),plan,new=True);print(json.dumps(dict(status='PREPARED_NO_VALUES_READ',plan_sha256=plan['plan_sha256'])));return
    require(args.approved_plan_sha256,'Explicit execution stage and exact SHA required')
    plan,original=checked_plan(args.plan,args.approved_plan_sha256);guard=start(plan)
    try:
        if args.audit: result=audit(plan,original,guard)
        elif args.events:
            from postprocess_primary_events import run_events
            result=run_events(plan,guard)
        elif args.uncertainty:
            from postprocess_primary_uncertainty import run_uncertainty
            result=run_uncertainty(plan,guard)
        elif args.diagnostics:
            from postprocess_primary_residuals import run_diagnostics
            result=run_diagnostics(plan,guard)
        else: result=figures(plan,original,guard)
        print(json.dumps({k:v for k,v in result.items() if k not in ('groups','blocks')}))
    finally: guard.close()


if __name__=='__main__':main()
