#!/usr/bin/env python3
"""Prime pinned HRRR commit metadata under the approved assessment budget.

No raw archive or numerical weather/discharge value is opened by this stage.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse, hashlib, json, os, sys, uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hydra_directflow.assessment_sources import AssessmentSourceProvider, authenticate_assessment_bundle_identity
from hydra_directflow.staging import SourceBudget, verified_hit, object_identity, atomic_json, sync_directory, MAX_COMMIT
from hydra_directflow.multicycle import qualified_cycle_rows
from hydra_directflow.local_data import verified_local_bytes
from hydra_directflow.reference import digest,require

def run(plan_path,expected):
    plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_text())
    require(plan['plan_sha256']==expected and digest({k:v for k,v in plan.items() if k!='plan_sha256'})==expected,'Exact staging plan required')
    ap=plan['sources']['budget_approval'];verified_local_bytes(ap['path'],ap['sha256'],expected_bytes=ap['bytes'],max_bytes=65536)
    stage=ROOT/'.local/assessment-v1/source-staging';prep=ROOT/'.local/assessment-preparation-v1'
    budget=SourceBudget(stage/'budget',ap['sha256'],body_cap=16*1024**3,get_cap=4000,usd_cap=5)
    try:
        with AssessmentSourceProvider(plan=plan,directory=stage,budget=budget,expected_plan_sha256=expected) as provider:
            rows=qualified_cycle_rows(plan['sources']['qualified_index']['path'],[r['cycle_id'] for r in plan['cycles']])
            saved=prep/'INDEX_ROWS.json'
            payload=dict(schema='hydra-assessment-index-rows/v1',index_sha256=plan['sources']['qualified_index']['sha256'],rows=rows)
            if saved.exists():require(json.loads(saved.read_text())==payload,'Existing index subset differs')
            else:atomic_json(saved,payload)
            resolved=[]
            for i,r in enumerate(plan['cycles']):
                cid=r['cycle_id'];require(digest(rows[cid])==r['sealed_row_sha256'],'Sealed row differs')
                ident=object_identity(r['commit_identity'],MAX_COMMIT,bytes_required=False,allowed_buckets=provider.allowed_buckets)
                target=stage/'commits'/(cid+'.json');provider.check_resources()
                if not verified_hit(target,ident,MAX_COMMIT):
                    require(not target.exists(),'Existing commit metadata is corrupt')
                    paths=[x['path'] if isinstance(x,dict) else x for x in r['local_commit_candidates']]
                    source=next((p for p in paths if verified_hit(p,ident,MAX_COMMIT)),None)
                    if source:
                        raw=verified_local_bytes(source,ident['sha256'],expected_bytes=ident['bytes'],max_bytes=MAX_COMMIT)
                        temp=target.with_name(target.name+'.partial-'+uuid.uuid4().hex)
                        with temp.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
                        os.link(temp,target);temp.unlink();sync_directory(target.parent)
                    else:provider._transfer(ident,target,MAX_COMMIT,'commit')
                bundle,worker,receipt=authenticate_assessment_bundle_identity(dict(cycle_id=cid,commit_path=str(target)),rows[cid],expected_sealed_row_sha256=r['sealed_row_sha256'])
                resolved.append(dict(cycle_id=cid,commit_path=str(target),bundle=bundle,lineage=receipt))
                if (i+1)%50==0 or i+1==len(plan['cycles']):
                    status=dict(schema='hydra-assessment-metadata-progress/v1',phase='STAGING_COMMIT_METADATA',resolved_cycles=i+1,total_cycles=len(plan['cycles']),source_budget=budget.snapshot(),model_forecasts=0,numeric_values_decoded=False,updated_at_utc=datetime.now(timezone.utc).isoformat())
                    atomic_json(ROOT/'.local/assessment-v1/progress.json',status);print(json.dumps(status),flush=True)
            final=dict(schema='hydra-assessment-resolved-source-bundles/v1',status='ALL1463_SOURCE_IDENTITIES_RESOLVED',staging_plan_sha256=expected,rows=resolved,archive_bytes=sum(x['bundle']['bytes'] for x in resolved),source_budget=budget.snapshot(),raw_archives_opened=False,numeric_values_decoded=False)
            atomic_json(prep/'RESOLVED_BUNDLES.json',final);print(json.dumps({k:v for k,v in final.items() if k!='rows'}),flush=True)
    finally:budget.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--approved-plan-sha256',required=True);a=p.parse_args();run(a.plan,a.approved_plan_sha256)
