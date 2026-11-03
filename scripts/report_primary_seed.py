#!/usr/bin/env python3
"""Publish deterministic private development reports from verified campaign jobs.

No models, Torch, source observations or scientific fits are loaded. Checkpoints
are hashed as opaque bytes; prediction coordinate/identity members are checked
without decoding prediction values. The ledger remains read-only.
"""
from collections import defaultdict
import hashlib
import html
import io
import json
import os
from pathlib import Path
import sqlite3
import stat
import statistics
import sys
import time
from urllib.parse import quote
import uuid
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hydra_directflow.campaign_ledger import CampaignLedger,validate_plan,digest,canonical,ARCHITECTURES
from hydra_directflow.reference import require
from hydra_directflow.local_data import GAUGES,bounded_npy
from hydra_directflow.quality import PRIMARY_APPROVED
from hydra_directflow.scoring import (ScoringPolicy,_validate_panel,validate_score_coordinates,
                                    aggregate_cv_folds,aggregate_final_seeds)

MAX_REPORT_BYTES=64*1024**2
MAX_READ_BYTES=8*1024**3
SEED_END={17:108,29:144,43:180}


def _bytes(value): return canonical(value).encode()+b'\n'
def _sha(raw): return hashlib.sha256(raw).hexdigest()


def _resident(path):
    before=path.lstat()
    require(stat.S_ISREG(before.st_mode) and not getattr(before,'st_flags',0)&0x40000000,
            'Reports require resident regular artifacts; no links or hydration')
    return before


def _relative(root,value):
    require(isinstance(value,str) and value and not Path(value).is_absolute()
            and all(x not in ('.','..') for x in Path(value).parts),'Safe private artifact path required')
    path=root/value
    require(path==path.resolve() and path.is_relative_to(root),'Artifact escaped its private run')
    return path


class _Read:
    def __init__(self): self.bytes=0; self.files=0; self.started=time.monotonic()
    def read(self,path,maximum,*,expected_sha=None,expected_bytes=None,collect=True):
        require(time.monotonic()-self.started<900,'Seed report wall bound exceeded')
        before=_resident(path)
        require(0<before.st_size<=maximum and (expected_bytes is None or before.st_size==expected_bytes),
                'Artifact size differs before read')
        require(self.bytes+before.st_size<=MAX_READ_BYTES,'Seed report logical read bound exceeded')
        self.bytes+=before.st_size; self.files+=1; hashed=hashlib.sha256(); parts=[]; observed=0
        with path.open('rb') as stream:
            while True:
                chunk=stream.read(min(1024**2,before.st_size-observed+1))
                if not chunk: break
                observed+=len(chunk); require(observed<=before.st_size,'Artifact grew during read')
                hashed.update(chunk)
                if collect: parts.append(chunk)
        after=path.lstat(); identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        require(identity(before)==identity(after) and observed==before.st_size,'Artifact changed during read')
        require(expected_sha is None or hashed.hexdigest()==expected_sha,'Artifact SHA differs')
        return b''.join(parts) if collect else hashed.hexdigest()


def _ledger_records(plan,ledger,seed):
    snapshot=CampaignLedger(ledger).snapshot()
    require(seed in snapshot['seed_reports_ready'],'Every required job needs verified training and development evaluation')
    require(snapshot['bookkeeping_bytes_reserved']>=2*1024**3,'Reserved report/bookkeeping allowance required')
    db=ledger/'campaign.sqlite3'; _resident(db)
    con=sqlite3.connect('file:'+quote(str(db),safe='/')+'?mode=ro',uri=True,isolation_level=None)
    con.row_factory=sqlite3.Row
    try:
        con.execute('BEGIN')
        parent=con.execute('SELECT plan_json FROM campaign WHERE id=1').fetchone()
        require(parent is not None and parent[0]==canonical(plan),'Ledger and requested immutable plan differ')
        jobs=con.execute('SELECT * FROM jobs ORDER BY position').fetchall()
        require(len(jobs)==180,'Incomplete ledger roster')
        records=[]
        for index,spec in enumerate(plan['jobs'][:SEED_END[seed]]):
            row=jobs[index]
            require(row['position']==index and row['job_id']==spec['job_id'] and row['job_json']==canonical(spec)
                    and row['state']=='VERIFIED_TRAIN_EVAL' and row['completion_json'],
                    'Required job is missing, stale or not verified')
            receipt=json.loads(row['completion_json']); receipt_sha=digest(receipt)
            verified=con.execute("SELECT * FROM artifacts WHERE job_position=? AND kind='COMPLETE' AND receipt_sha=? AND receipt_json=?",
                                 (index,receipt_sha,row['completion_json'])).fetchall()
            require(len(verified)==1,'Exactly one immutable completion record required per job')
            attempts=[dict(x) for x in con.execute('SELECT id,status,started,finished,error_type,entered,confirmed FROM attempts WHERE job_position=? ORDER BY id',(index,))]
            require(attempts and all(x['finished'] is not None and x['finished']>=x['started'] for x in attempts)
                    and attempts[-1]['status']=='VERIFIED_TRAIN_EVAL','Completed attempt timing/state missing')
            records.append(dict(job=spec,receipt=receipt,receipt_sha256=receipt_sha,attempts=attempts,
                entered=row['entered'],confirmed=row['confirmed'],started=row['started']))
        con.execute('COMMIT'); return records
    except BaseException:
        if con.in_transaction: con.execute('ROLLBACK')
        raise
    finally: con.close()


def _descriptor(run,value,maximum,guard,*,collect):
    require(isinstance(value,dict) and set(value)=={'path','bytes','sha256'} and type(value['bytes']) is int
            and isinstance(value['sha256'],str) and len(value['sha256'])==64,'Exact artifact descriptor required')
    return guard.read(_relative(run,value['path']),maximum,expected_sha=value['sha256'],expected_bytes=value['bytes'],collect=collect)


def _prediction_coordinates(raw,receipt,job):
    import numpy as np
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries=archive.infolist(); expected={'prediction_q.npy','case_id.npy','origin_time.npy','forecast_time.npy','site_id.npy','metadata_json.npy'}
        require(len(entries)==len(expected) and {x.filename for x in entries}==expected
                and sum(x.file_size for x in entries)<=64*1024**2,'Prediction ZIP inventory/expansion differs')
        n=job['data_job']['actual_common_cases']['outer_scoring']
        arrays={name:bounded_npy(archive.read(name+'.npy'),shape=shape,dtype=dtype) for name,shape,dtype in
                (('case_id',(n,),np.dtype('<U29')),('origin_time',(n,),np.dtype('<i8')),
                 ('forecast_time',(n,18),np.dtype('<i8')),('site_id',(n,),np.dtype('<i8')))}
        blob=archive.read('metadata_json.npy'); require(len(blob)<=256*1024,'Bounded prediction metadata required')
        stream=io.BytesIO(blob); version=np.lib.format.read_magic(stream)
        require(version in ((1,0),(2,0)),'Supported prediction metadata encoding required')
        shape,fortran,dtype=(np.lib.format.read_array_header_1_0(stream) if version==(1,0) else np.lib.format.read_array_header_2_0(stream))
        require(shape==() and not fortran and dtype.kind=='U' and dtype.itemsize<=256*1024,'Scalar text prediction metadata required')
        meta=json.loads(bounded_npy(blob,shape=(),dtype=dtype).item())
    ids,_,_=validate_score_coordinates(case_ids=tuple(arrays['case_id'].tolist()),origins=arrays['origin_time'],
        forecast_time=arrays['forecast_time'],gauge_id=job['gauge_id'])
    require(digest(list(ids))==job['role_case_ids_sha256']['outer_scoring']
            and np.all(arrays['site_id']==GAUGES.index(job['gauge_id'])),'Prediction case/gauge membership differs')
    for name in ('plan_sha256','protocol_sha256','source_plan_sha256','snapshot_sha256','implementation_sha256',
                 'job_id','job_sha256','training_source_identity','training_cohort_admission_digest',
                 'inference_source_identity','inference_cohort_admission_digest','selected_best_epoch'):
        require(meta.get(name)==receipt.get(name),'Prediction identity differs from its verified job')
    require(meta['checkpoint_sha256']==receipt['hashes']['checkpoint'] and meta['case_ids_sha256']==digest(list(ids))
            and meta['case_count']==n and meta['label_values_opened'] is False and meta['posthoc_clipping'] is False
            and meta['best_only'] is True,'Prediction selection/label separation differs')


def _feature_schema(job):
    require(job['input_arm'] in ('full','q_only'),'Declared primary feature arm required')
    return 'Q_ONLY_H2_HORIZON1_S0_V1' if job['input_arm']=='q_only' else 'FULL_HRRR_H2_W4_S2_V1'


def _label_source_identity(manifest,receipt,job,plan):
    core={k:v for k,v in manifest.items() if k not in ('source_identity','returned_arrays_sha256','construction')}
    require(manifest['source_identity']==receipt['label_source_identity']==digest(core),
            'Label sidecar core does not authenticate the registered label source')
    require(manifest['schema']=='hydra-primary-campaign-data/v1' and manifest['operation']=='labels'
            and manifest['source_plan_sha256']==plan['source_plan_sha256'] and manifest['snapshot_sha256']==plan['snapshot_sha256']
            and manifest['quality_policy']==PRIMARY_APPROVED
            and all(manifest[k]==job[k] for k in ('gauge_id','fold','input_arm','history_hours')),
            'Label sidecar source/view/quality identity differs')
    require(manifest['roles']=={'outer':{'cases':job['data_job']['actual_common_cases']['outer_scoring'],
                'case_ids_sha256':job['role_case_ids_sha256']['outer_scoring']}}
            and manifest['case_count']==job['data_job']['actual_common_cases']['outer_scoring']
            and manifest['cohort_admission_digest']==receipt['inference_cohort_admission_digest']
            and manifest['raw_arrays_sha256']==manifest['returned_arrays_sha256']
            and isinstance(manifest['raw_arrays_sha256'],str) and len(manifest['raw_arrays_sha256'])==64,
            'Label sidecar exact typed-array/cohort identity differs')
    return manifest['raw_arrays_sha256']


def _verified_panel(plan,run,record,guard):
    job,receipt=record['job'],record['receipt']
    bindings=dict(plan_sha256=plan['plan_sha256'],protocol_sha256=plan['protocol_sha256'],source_plan_sha256=plan['source_plan_sha256'],
        snapshot_sha256=plan['snapshot_sha256'],implementation_sha256=plan['implementation_sha256'],job_id=job['job_id'],job_sha256=digest(job))
    require(all(receipt.get(k)==v for k,v in bindings.items()) and receipt.get('status')=='VERIFIED_TRAIN_EVAL'
            and receipt.get('protected_assessment_access') is False,'Completion job/protocol/source identity differs')
    require(receipt['feature_schema']==_feature_schema(job),'Receipt feature schema differs from declared input arm')
    raw=guard.read(_relative(run,receipt['receipt_path']),2*1024**2)
    require(raw==_bytes(receipt),'Published completion receipt differs from the ledger')
    artifacts=receipt.get('artifacts',{}); hashes=receipt.get('hashes',{})
    require(set(artifacts)==set(hashes)=={'checkpoint','predictions','metrics'}
            and all(artifacts[k]['sha256']==hashes[k] for k in artifacts),'Complete checkpoint/prediction/metric evidence required')
    _descriptor(run,artifacts['checkpoint'],16*1024**2,guard,collect=False)
    _prediction_coordinates(_descriptor(run,artifacts['predictions'],64*1024**2,guard,collect=True),receipt,job)
    metrics=json.loads(_descriptor(run,artifacts['metrics'],2*1024**2,guard,collect=True)); panel=metrics['panel']
    sidecar_path=_relative(run,str(Path(receipt['receipt_path']).parent/'outer-label-source.json'))
    sidecar_raw=guard.read(sidecar_path,2*1024**2)
    label_arrays_sha=_label_source_identity(json.loads(sidecar_raw),receipt,job,plan)
    require(all(metrics.get(k)==v for k,v in bindings.items())
            and metrics['checkpoint_sha256']==hashes['checkpoint'] and metrics['predictions_sha256']==hashes['predictions']
            and metrics['outer_case_ids_sha256']==job['role_case_ids_sha256']['outer_scoring'], 'Metric parent identities differ')
    _validate_panel(panel,ScoringPolicy())
    require(panel['input_cohort_digest']==receipt['inference_cohort_admission_digest']
            and panel['label_source_digest']==receipt['label_source_identity']
            and panel['label_policy_digest']==digest({'policy':PRIMARY_APPROVED})
            and panel['case_ids_sha256']==job['role_case_ids_sha256']['outer_scoring']
            and panel['negative_predictions_clipped'] is False and panel['protected_assessment_scores']==0,
            'Metric cohort/label/assessment identity differs')
    expected=dict(architecture=job['architecture'],gauge_id=job['gauge_id'],history_hours=job['history_hours'],
        feature_schema=_feature_schema(job),fold_id=job['fold'],seed=job['seed'],
        configuration_id=f'{job["architecture"]}-{job["input_arm"]}-L{job["history_hours"]}')
    require(all(panel['identity'].get(k)==v for k,v in expected.items()),'Metric configuration differs')
    n=job['data_job']['actual_common_cases']['outer_scoring']
    require(all(row['n']==row['n_requested']==n for row in panel['leads'])
            and receipt['outer_case_ids_sha256']==panel['case_ids_sha256'] and receipt['scored_cases']==n
            and receipt['score_status']==panel['status'],'Missing or selectively dropped metric support')
    return dict(job_id=job['job_id'],order=job['order'],fold=job['fold'],seed=job['seed'],gauge_id=job['gauge_id'],
        architecture=job['architecture'],input_arm=job['input_arm'],history_hours=job['history_hours'],device=job['device'],
        completion_receipt_sha256=record['receipt_sha256'],completion_receipt_file_sha256=_sha(raw),artifact_sha256=hashes,
        label_arrays_sha256=label_arrays_sha,label_source_manifest_sha256=_sha(sidecar_raw),
        completed_epochs=receipt['completed_epochs'],selected_best_epoch=receipt['selected_best_epoch'],
        logical_updates=receipt['logical_updates'],physical_updates_entered=record['entered'],physical_updates_confirmed=record['confirmed'],
        attempts=record['attempts'],attempt_elapsed_seconds=sum(x['finished']-x['started'] for x in record['attempts']),
        job_elapsed_span_seconds=record['attempts'][-1]['finished']-record['started'],panel=panel)


def _matched_views(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['gauge_id'],row['fold'],row['seed']].append(row)
    for (_,fold,_),members in groups.items():
        arms=(('full',168),('q_only',168),('full',24)) if fold=='final_development' else (('full',168),('q_only',168))
        expected={(a,arm,length) for a in ARCHITECTURES for arm,length in arms}
        require(len(members)==len(expected) and {(r['architecture'],r['input_arm'],r['history_hours']) for r in members}==expected,
                'Missing or duplicate matched model/input panels')
        common=[];views=defaultdict(list)
        for row in members:
            panel=row['panel']
            require(panel['label_policy_digest']==digest({'policy':PRIMARY_APPROVED}), 'Matched panel quality policy differs')
            common.append((row['label_arrays_sha256'],panel['case_ids_sha256'],panel['input_cohort_digest'],
                tuple((x['lead_hours'],x['n'],x['n_requested'],x['scored_sample_digest']) for x in panel['leads'])))
            views[row['input_arm'],row['history_hours']].append((panel['label_source_digest'],tuple(x['label_series_digest'] for x in panel['leads'])))
        require(len(set(common))==1,'Matched arms/models used different typed labels, cohorts or scored support')
        require(all(len(set(values))==1 for values in views.values()),'Same-view model label series/source differs')
    return dict(gauge_fold_seed_groups=len(groups),matched_typed_labels_masks_coordinates=True,
                view_specific_label_sources_preserved=True)


def _aggregates(rows,seed):
    _matched_views(rows)
    cv=defaultdict(dict); final=defaultdict(dict)
    for row in rows:
        key=(row['gauge_id'],row['architecture'],row['input_arm'],row['history_hours'])
        if row['fold']=='final_development': final[key][row['seed']]=row['panel']
        else: cv[key][row['fold']]=row['panel']
    require(len(cv)==24 and len(final)==36,'Complete model/gauge/input comparison inventory required')
    cv_rows=[dict(gauge_id=k[0],architecture=k[1],input_arm=k[2],history_hours=k[3],score=aggregate_cv_folds(v)) for k,v in sorted(cv.items())]
    seeds=[]
    if seed==43:
        for key,panels in sorted(final.items()):
            aggregate=aggregate_final_seeds(panels); values=[panels[s]['primary_score'] for s in (17,29,43)]
            spread=None if any(x is None for x in values) else dict(sample_standard_deviation=statistics.stdev(values),minimum=min(values),maximum=max(values))
            seeds.append(dict(gauge_id=key[0],architecture=key[1],input_arm=key[2],history_hours=key[3],score=aggregate,
                descriptive_seed_spread=spread,spread_is_confidence_interval=False,selected_favorable_seed=False))
    return cv_rows,seeds


def _number(value): return 'UNDEFINED' if value is None else f'{value:.6g}'


def _html(summary):
    esc=lambda x:html.escape(str(x),quote=True)
    out=['<!doctype html><html><head><meta charset="utf-8"><title>DirectFlow seed report</title>',
         '<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#172334}table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}th,td{text-align:left;border-bottom:1px solid #d4dce5;padding:7px}th{background:#edf2f7}details{margin:14px 0}code{overflow-wrap:anywhere}.note{background:#eef4fa;padding:16px}</style></head><body>',
         f'<h1>DirectFlow · seed {summary["seed"]}</h1><p class="note">{esc(summary["evidence_status"])}. Development results only. Protected assessment has not been scored.</p>',
         f'<p>{summary["current_seed_jobs"]} jobs for this seed; {summary["verified_cumulative_jobs"]}/180 completed in the declared sequence. Event analysis and dependence-aware uncertainty remain pending.</p>',
         '<p>NSE is the equal-weight mean over all18 leads. RMSE, MAE and bias use m³/s. Negative predictions remain unclipped. Undefined primary panels remain visible.</p>']
    for gauge in GAUGES:
        out.append(f'<h2>Gauge {gauge} · current-seed final-development results</h2><table><tr><th>Model</th><th>Inputs / history</th><th>NSE</th><th>Best / completed epochs</th><th>Cases per lead</th></tr>')
        for row in summary['jobs']:
            if row['seed']!=summary['seed'] or row['gauge_id']!=gauge or row['fold']!='final_development': continue
            out.append(f'<tr><td>{esc(row["architecture"])}</td><td>{esc(row["input_arm"])} / {row["history_hours"]}h</td><td>{_number(row["panel"]["primary_score"])}</td><td>{row["selected_best_epoch"]} / {row["completed_epochs"]}</td><td>{row["panel"]["leads"][0]["n"]}</td></tr>')
        out.append('</table>')
    out.append('<h2>Equal-fold development CV · seed17</h2><table><tr><th>Gauge</th><th>Model</th><th>Inputs</th><th>Mean NSE</th></tr>')
    for row in summary['cv_aggregates']:
        out.append(f'<tr><td>{row["gauge_id"]}</td><td>{esc(row["architecture"])}</td><td>{esc(row["input_arm"])}</td><td>{_number(row["score"]["primary_score"])}</td></tr>')
    out.append('</table>')
    if summary['final_seed_aggregates']:
        out.append('<h2>All three final seeds · mean and descriptive spread</h2><p>Spread is not a confidence interval. No favorable seed is selected.</p><table><tr><th>Gauge</th><th>Model / input / history</th><th>Mean NSE</th><th>Sample SD</th></tr>')
        for row in summary['final_seed_aggregates']:
            spread=row['descriptive_seed_spread']
            out.append(f'<tr><td>{row["gauge_id"]}</td><td>{esc(row["architecture"])} / {esc(row["input_arm"])} / {row["history_hours"]}h</td><td>{_number(row["score"]["primary_score"])}</td><td>{_number(spread["sample_standard_deviation"] if spread else None)}</td></tr>')
        out.append('</table>')
    out.append('<h2>Per-job and per-lead evidence</h2>')
    for row in summary['jobs']:
        out.append(f'<details><summary>{esc(row["job_id"])} · NSE {_number(row["panel"]["primary_score"])}</summary><p>Best/completed epochs {row["selected_best_epoch"]}/{row["completed_epochs"]}; logical updates {row["logical_updates"]}; charged/confirmed physical updates {row["physical_updates_entered"]}/{row["physical_updates_confirmed"]}; ledger attempt span {_number(row["attempt_elapsed_seconds"])}s.</p>')
        out.append('<table><tr><th>Lead</th><th>NSE</th><th>RMSE</th><th>MAE</th><th>Bias</th><th>Negative fraction</th><th>n / requested</th><th>Distinct dates</th><th>Primary support</th></tr>')
        for lead in row['panel']['leads']:
            out.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in [lead['lead_hours'],*(_number(lead[k]) for k in ('nse','rmse','mae','bias','negative_frequency')),f'{lead["n"]}/{lead["n_requested"]}',lead['distinct_utc_dates'],'eligible' if lead['primary_eligible'] else '; '.join(lead['primary_ineligibility'])])+'</tr>')
        out.append('</table></details>')
    out.append(f'<p>Summary content SHA256: <code>{summary["content_sha256"]}</code></p></body></html>')
    return ''.join(out).encode()


def _publish(directory,payloads,commit):
    def sync(path):
        fd=os.open(path,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    parent=directory.parent
    require(parent==parent.resolve(),'Report directory link refused')
    if not parent.exists(): parent.mkdir(); sync(parent.parent)
    if directory.exists():
        require(directory==directory.resolve() and {p.name for p in directory.iterdir()}==set(payloads)|{'COMMIT.json'},'Existing report inventory differs')
        read=_Read()
        for name,raw in {**payloads,'COMMIT.json':_bytes(commit)}.items():
            require(read.read(directory/name,MAX_REPORT_BYTES)==raw,'Existing report differs; preserve and review')
        return 'VERIFIED_EXISTING_REPORT'
    stage=parent/('.'+directory.name+'.writing-'+uuid.uuid4().hex);stage.mkdir(mode=0o700);sync(parent)
    for name,raw in {**payloads,'COMMIT.json':_bytes(commit)}.items():
        with (stage/name).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    sync(stage);require(not directory.exists(),'Report appeared during publication')
    os.rename(stage,directory);sync(parent);return 'PUBLISHED'


def publish_seed_report(plan,run_directory,ledger_directory,seed):
    plan=validate_plan(plan)
    require(type(seed) is int and seed in SEED_END,'One declared seed report required')
    require(plan.get('protocol_frozen') is True and plan.get('scientific_training_authorized') is True,
            'Frozen authorized campaign metadata required')
    run,ledger=Path(run_directory).absolute(),Path(ledger_directory).absolute()
    require(run==run.resolve() and ledger==ledger.resolve() and run.is_relative_to(ROOT/'.local')
            and ledger.is_relative_to(ROOT/'.local') and plan.get('run_directory')==str(run)
            and plan.get('ledger_directory')==str(ledger),'Exact private run/ledger paths required')
    records=_ledger_records(plan,ledger,seed); guard=_Read()
    for name,expected in plan['implementation_sha256'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Relative implementation identity required')
        path=ROOT/name
        require(path==path.resolve() and path.is_relative_to(ROOT),'Implementation escaped the pinned checkout')
        guard.read(path,2*1024**2,expected_sha=expected,collect=False)
    rows=[_verified_panel(plan,run,record,guard) for record in records]
    cv,final=_aggregates(rows,seed)
    summary=dict(schema='hydra-primary-seed-development-report/v1',seed=seed,
        evidence_status='THREE_SEED_DEVELOPMENT' if seed==43 else 'SINGLE_SEED_DEVELOPMENT',
        plan_sha256=plan['plan_sha256'],protocol_sha256=plan['protocol_sha256'],source_plan_sha256=plan['source_plan_sha256'],snapshot_sha256=plan['snapshot_sha256'],
        implementation_sha256=plan['implementation_sha256'],current_seed_jobs=sum(r['seed']==seed for r in rows),
        verified_cumulative_jobs=len(rows),completed_seed_prefix=[s for s in SEED_END if s<=seed],
        jobs=rows,cv_aggregates=cv,final_seed_aggregates=final,matched_comparison=_matched_views(rows),all_seed_mean_available=seed==43,
        no_favorable_seed_selection=True,primary_support_panels_dropped=False,protected_assessment_access=False,
        events_analysis='PENDING_NOT_COMPUTED_BY_THIS_PUBLISHER',dependence_aware_uncertainty='PENDING_NOT_COMPUTED_BY_THIS_PUBLISHER',
        scientific_poster_ready=False,raw_predictions_published=False,
        runtime_qualification='Ledger attempt spans include recovery gaps for lost leases; they are not isolated accelerator timings',
        execution=dict(models=0,torch_loads=0,source_value_reads=0,prediction_value_members_decoded=0,checkpoint_deserialization=0,
                       logical_artifact_bytes_verified=guard.bytes,artifact_files_verified=guard.files,ledger_mutations=0))
    summary['content_sha256']=digest(summary); raw=_bytes(summary); page=_html(summary)
    payloads={'summary.json':raw,'index.html':page}
    commit=dict(schema='hydra-primary-seed-report-commit/v1',seed=seed,plan_sha256=plan['plan_sha256'],
        completion_receipts={row['job_id']:row['completion_receipt_sha256'] for row in rows},
        payloads={name:dict(bytes=len(body),sha256=_sha(body)) for name,body in payloads.items()})
    require(sum(map(len,payloads.values()))+len(_bytes(commit))<=MAX_REPORT_BYTES,'Report output exceeds reserved64MiB scope')
    directory=run/'reports'/f'seed-{seed}'; status=_publish(directory,payloads,commit)
    return dict(status=status,seed=seed,plan_sha256=plan['plan_sha256'],report_directory=str(directory),
                summary_sha256=_sha(raw),html_sha256=_sha(page),commit_sha256=_sha(_bytes(commit)))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--run-directory',type=Path,required=True)
    parser.add_argument('--ledger-directory',type=Path,required=True);parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args(); payload=json.loads(_Read().read(args.plan,8*1024**2))
    print(json.dumps(publish_seed_report(payload,args.run_directory,args.ledger_directory,args.seed),sort_keys=True))
