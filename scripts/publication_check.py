"""Check candidate Git files before publishing this public research repository."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    names = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT
    ).decode().split('\0')
    patterns = {
        'absolute local home path': re.compile('/' + r'Users/[^/\s]+/'),
        'private acquisition bucket': re.compile('hydra-' + r'hrrr-direct-acq-[A-Za-z0-9-]+'),
        'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
        'GitHub token': re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}'),
        'AWS access ID': re.compile(r'AKIA[A-Z0-9]{16}'),
    }
    errors = []
    for name in sorted(set(filter(None, names))):
        p = ROOT / name
        if not p.exists():
            continue
        if p.is_symlink() or p.stat().st_size > 1_000_000:
            errors.append((name, 'symlink or file over 1 MB'))
            continue
        if p.suffix in {'.pt', '.pth', '.npz', '.npy', '.parquet', '.sqlite', '.whl'}:
            errors.append((name, 'data/checkpoint/dependency binary'))
            continue
        try:
            body = p.read_text()
        except UnicodeDecodeError:
            errors.append((name, 'unreviewed binary'))
            continue
        for label, pattern in patterns.items():
            if pattern.search(body):
                errors.append((name, label))
    for name, reason in errors:
        print(f'{name}: {reason}')
    print(f'Publication check: {len(set(filter(None, names)))} candidate files; {len(errors)} findings')
    return bool(errors)

if __name__ == '__main__':
    sys.exit(main())
