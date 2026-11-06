#!/usr/bin/env python3
"""Materialize the exact approved owned USGS files as opaque bytes only.

This does not enable numeric readers, model inference or scoring. Every source
open runs in a bounded child and has a durable retained attempt record.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, shutil, stat, subprocess, sys, uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hydra_directflow.reference import require
from hydra_directflow.staging import atomic_json, durable_directory

PREP=ROOT/'.local/assessment-preparation-v1'
DEST=ROOT/'.local/assessment-v1/source-staging/usgs'
APPROVAL=PREP/'STAGING_APPROVAL.json'
INVENTORY=PREP/'SOURCE_INVENTORY.json'
USGS_SHA='4c494b4587c394b1819261f6df0386eba19500a28f0d73aa06543870f933ea2a'
INVENTORY_SHA='18684bef5238ca936621d1cf65bbbfc41c9def3d10e1dba077e538e55c27e264'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def resident(path,maximum):
    s=path.lstat();require(stat.S_ISREG(s.st_mode) and not s.st_flags&0x40000000 and 0<s.st_size<=maximum,'Resident bounded control required')
    return path.read_bytes()

def authorities(expected_approval_sha):
    raw=resident(APPROVAL,65536);require(sha(raw)==expected_approval_sha,'Approval identity differs')
    approval=json.loads(raw);require(approval['status']=='USER_APPROVED' and approval['user_reply']=='Approve bounded assessment staging'
        and approval['owned_usgs_download_files']==5 and approval['owned_usgs_download_encoded_bytes']==1371323,'Approved materialization scope differs')
    raw=resident(INVENTORY,8*1024**2);require(sha(raw)==INVENTORY_SHA,'Source inventory changed')
    inv=json.loads(raw);return approval,inv

def opaque_copy(item, *, permitted_cloud_paths, destination_root):
    source,target=Path(item['source']),Path(item['destination'])
    require(source.is_absolute() and source==source.resolve() and target.parent==destination_root==destination_root.resolve(),
            'Exact source and stable destination paths required')
    require(type(item['bytes']) is int and 0<item['bytes']<=2*1024**2 and len(item['sha256'])==64,'Bounded exact source identity required')
    if target.exists():
        b=resident(target,2*1024**2);require(len(b)==item['bytes'] and sha(b)==item['sha256'],'Existing owned copy changed')
        return dict(status='VERIFIED_EXISTING_COPY',bytes=len(b),sha256=sha(b),numeric_values_decoded=False)
    before=source.lstat();require(stat.S_ISREG(before.st_mode) and before.st_size==item['bytes'],'Original source size/type differs')
    cloud=bool(before.st_flags&0x40000000)
    require(not cloud or str(source) in permitted_cloud_paths,'Unapproved additional cloud-only source')
    require(shutil.disk_usage(destination_root).free>=8*1024**3+2*item['bytes'],'Free disk floor reached')
    fd=os.open(source,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    with os.fdopen(fd,'rb') as stream:
        opened=os.fstat(stream.fileno());require(stat.S_ISREG(opened.st_mode) and opened.st_size==item['bytes'],'Opened source changed')
        raw=stream.read(item['bytes']+1)
    require(len(raw)==item['bytes'] and sha(raw)==item['sha256'],'Owned source checksum differs')
    tmp=target.with_name(target.name+'.partial-'+uuid.uuid4().hex)
    with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    os.link(tmp,target);tmp.unlink()
    return dict(status='VERIFIED_NEW_OPAQUE_COPY',bytes=len(raw),sha256=sha(raw),source_was_cloud_only=cloud,
                numeric_values_decoded=False,fileprovider_wire_bytes=None)

def worker(request):
    q=json.loads(resident(Path(request),65536));_,inv=authorities(q['approval_sha256'])
    allowed=[inv['usgs']['pinned_manifest_references'][0]['path'],*inv['usgs']['cloud_only_native_paths']]
    # Some inventory versions retain paths as small descriptors.
    allowed={x if isinstance(x,str) else x['path'] for x in allowed}
    item=q['item'];require(item['source'] in {inv['usgs']['pinned_manifest_references'][0]['path'],*[x['path'] for x in inv['usgs']['months']]},'Source was not inventoried')
    result=opaque_copy(item,permitted_cloud_paths=allowed,destination_root=DEST)
    atomic_json(Path(q['receipt']),dict(**result,completed_at_utc=utc(),source=item['source'],destination=item['destination']))

def run(expected_approval_sha):
    _,inv=authorities(expected_approval_sha);durable_directory(DEST)
    ledger_path=PREP/'OWNED_MATERIALIZATION_ATTEMPTS.json'
    ledger=json.loads(resident(ledger_path,1024**2)) if ledger_path.exists() else dict(schema='hydra-assessment-owned-materialization/v1',approval_sha256=expected_approval_sha,attempts=[])
    require(ledger['approval_sha256']==expected_approval_sha,'Cannot reset materialization approval')
    def copy(item):
        target=Path(item['destination'])
        if target.exists():
            b=resident(target,2*1024**2);require(len(b)==item['bytes'] and sha(b)==item['sha256'],'Existing copy changed');return
        require(not any(a['source']==item['source'] for a in ledger['attempts']),'Preserve previous attempt and diagnose before any retry')
        attempt=dict(number=len(ledger['attempts'])+1,source=item['source'],expected_bytes=item['bytes'],sha256=item['sha256'],status='PRECHARGED',started_at_utc=utc())
        receipt=PREP/f"owned-copy-{attempt['number']:02d}.json";req=PREP/f"owned-request-{attempt['number']:02d}.json"
        ledger['attempts'].append(attempt);atomic_json(ledger_path,ledger)
        atomic_json(req,dict(item=item,approval_sha256=expected_approval_sha,receipt=str(receipt)))
        try:
            p=subprocess.run([sys.executable,'-I','-B',str(Path(__file__).resolve()),'--worker',str(req)],capture_output=True,text=True,timeout=60)
            require(p.returncode==0,'Opaque worker failed: '+p.stderr[-1200:])
            r=json.loads(resident(receipt,65536));require(r['sha256']==item['sha256'] and r['bytes']==item['bytes'],'Worker receipt differs')
            attempt.update(status='VERIFIED',finished_at_utc=utc(),receipt=str(receipt));atomic_json(ledger_path,ledger)
            print(json.dumps({'file':target.name,'status':'VERIFIED','bytes':r['bytes']}),flush=True)
        except BaseException as e:
            attempt.update(status='FAILED',error_type=type(e).__name__,finished_at_utc=utc());atomic_json(ledger_path,ledger);raise
    first=inv['usgs']['pinned_manifest_references'][0]
    copy(dict(source=first['path'],destination=str(DEST/'source_manifest.json'),bytes=first['expected_bytes'],sha256=USGS_SHA))
    manifest=json.loads(resident(DEST/'source_manifest.json',2*1024**2))
    native={Path(x['path']).stem:x for x in manifest['native_partitions']}
    for row in inv['usgs']['months']:
        original=native[row['month']];require(original['bytes']==row['bytes'],'Native partition size differs from manifest')
        copy(dict(source=row['path'],destination=str(DEST/(row['month']+'.parquet')),bytes=original['bytes'],sha256=original['sha256']))
    result=dict(schema='hydra-assessment-owned-source-residency/v1',status='ALL13_MONTHS_AND_MANIFEST_VERIFIED',files=14,
                expected_download_files=5,expected_download_encoded_bytes=1371323,numeric_values_decoded=False,
                fileprovider_wire_bytes=None,approval_sha256=expected_approval_sha,finished_at_utc=utc())
    atomic_json(PREP/'OWNED_SOURCES_READY.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--approved-sha256');a=p.parse_args()
    if a.worker:worker(a.worker)
    else:require(bool(a.approved_sha256),'Explicit approved scope hash required');run(a.approved_sha256)
