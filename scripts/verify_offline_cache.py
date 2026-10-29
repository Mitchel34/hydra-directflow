"""Verify the private local cache against the public aggregate readiness receipt."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=p.parse_args();root=args.root.resolve()
    receipt=json.loads((root/'reports/offline-readiness.json').read_text())
    count=0
    for record in receipt['files']:
        path=root/record['path']
        if (not path.resolve().is_relative_to(root) or path.is_symlink()
                or not path.is_file() or getattr(path.stat(), 'st_flags', 0) & 0x40000000):
            raise ValueError('A required local cache member is missing or nonresident')
        if path.stat().st_size!=record['bytes'] or digest(path)!=record['sha256']:
            raise ValueError('Local cache member changed: '+record['path'])
        count+=1
    print(json.dumps({'status':'PASS_LOCAL_OFFLINE_CACHE','files':count,'network_requests':0}))


if __name__=='__main__':main()
