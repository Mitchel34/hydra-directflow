#!/usr/bin/env python3
"""Prefetch a bounded chronological prefix as opaque, hash-pinned archives."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hydra_directflow.assessment_sources import AssessmentSourceProvider
from hydra_directflow.staging import SourceBudget,atomic_json
from hydra_directflow.local_data import verified_local_bytes
from hydra_directflow.reference import require,digest

def selected_interval(cycles,count,start=0):
    require(type(count) is int and 1<=count<=128,'At most128 chronological opaque archives per prefetch')
    require(type(start) is int and 0<=start and start+count<=len(cycles),'Exact bounded source interval required')
    return cycles[start:start+count]

def run(path,expected,count,start=0):
    plan=json.loads(Path(path).read_text());require(plan['plan_sha256']==expected and digest({k:v for k,v in plan.items() if k!='plan_sha256'})==expected,'Exact approved plan required')
    ap=plan['sources']['budget_approval'];verified_local_bytes(ap['path'],ap['sha256'],expected_bytes=ap['bytes'],max_bytes=65536)
    rows=json.loads((ROOT/'.local/assessment-preparation-v1/INDEX_ROWS.json').read_text())['rows']
    stage=ROOT/'.local/assessment-v1/source-staging';budget=SourceBudget(stage/'budget',ap['sha256'],body_cap=16*1024**3,get_cap=4000,usd_cap=5)
    try:
        with AssessmentSourceProvider(plan=plan,directory=stage,budget=budget,expected_plan_sha256=expected) as provider:
            selected=selected_interval(plan['cycles'],count,start)
            for i,r in enumerate(selected):
                result=provider(r['cycle_id'],rows[r['cycle_id']]);require(result['status']=='READY','Opaque source unavailable')
                if (i+1)%8==0 or i+1==count:
                    value=dict(status='PREFETCHING_OPAQUE_HRRR',interval_completed=i+1,interval_start=start,interval_count=count,total_candidate_cycles=1463,
                               budget=budget.snapshot(),raw_hotset=provider.check_resources(),numerical_values_decoded=False,forecasts_generated=0,
                               updated_at_utc=datetime.now(timezone.utc).isoformat())
                    atomic_json(ROOT/'.local/assessment-v1/progress.json',value);print(json.dumps(value),flush=True)
            name='PREFETCH_READY.json' if start==0 else 'PREFETCH_'+str(start)+'_'+str(start+count)+'_READY.json'
            atomic_json(ROOT/'.local/assessment-preparation-v1'/name,dict(status='OPAQUE_INTERVAL_VERIFIED',start=start,cycles=count,source_plan_sha256=expected,summary=provider.summary(),numeric_values_decoded=False))
    finally:budget.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--approved-plan-sha256',required=True);p.add_argument('--count',type=int,default=32);p.add_argument('--start',type=int,default=0);a=p.parse_args();run(a.plan,a.approved_plan_sha256,a.count,a.start)
