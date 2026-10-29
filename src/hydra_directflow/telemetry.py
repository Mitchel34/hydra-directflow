"""Local run custody and low-cardinality metrics; no network or optional imports."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


class RunJournal:
    """One explicit attempt. Resumption creates a new attempt with parent evidence.

    Actual wall time and artifact creation are never replaced by historical
    replay timestamps. Run IDs belong in logs, not unbounded metric labels.
    """
    def __init__(self, directory, spec, *, max_wall_seconds=300,
                 max_rss_bytes=4 * 2**30, clock=time.monotonic):
        if not 0 < max_wall_seconds <= 8 * 3600 or max_rss_bytes <= 0:
            raise ValueError('Finite positive bounded run limits required')
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=False)
        self.spec = json.loads(canonical(spec))
        self.spec_sha256 = hashlib.sha256(canonical(spec)).hexdigest()
        self.clock = clock
        self.start = clock()
        self.max_wall_seconds = max_wall_seconds
        self.max_rss_bytes = max_rss_bytes
        self.status = 'RUNNING'
        self.latest = {}
        atomic_json(self.path / 'spec.json', self.spec)
        self.event('started', spec_sha256=self.spec_sha256)
        self.metric('run_active', 1)
        self._state()

    def _state(self):
        atomic_json(self.path / 'state.json', dict(
            schema='hydra-run-state/v1', status=self.status,
            spec_sha256=self.spec_sha256, elapsed_seconds=self.clock() - self.start,
            maximum_wall_seconds=self.max_wall_seconds, maximum_rss_bytes=self.max_rss_bytes,
            updated_UTC=datetime.now(timezone.utc).isoformat()))

    def event(self, event_type, **fields):
        value = dict(event=event_type, actual_UTC=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=self.clock() - self.start, **fields)
        body = canonical(value) + b'\n'
        if len(body) > 16 * 1024:
            raise ValueError('Event exceeds bounded metadata size')
        with (self.path / 'events.jsonl').open('ab') as stream:
            stream.write(body)
            stream.flush()

    def check(self):
        if self.status != 'RUNNING':
            raise RuntimeError('Attempt is already terminal')
        elapsed = self.clock() - self.start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss *= 1 if sys.platform == 'darwin' else 1024
        if elapsed >= self.max_wall_seconds or rss > self.max_rss_bytes:
            self.finish('STOPPED', reason='wall_or_RSS_limit')
            raise RuntimeError('Run resource limit reached')
        self.metric('elapsed_seconds', elapsed)
        self.metric('host_peak_rss_bytes', rss)
        self.metric('heartbeat_unixtime_seconds', time.time())
        self._state()

    def metric(self, name, value):
        if self.status != 'RUNNING':
            raise RuntimeError('Metrics cannot change after terminal state')
        if not re.fullmatch('[a-z][a-z0-9_]*', name) or type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('A finite scalar metric and bounded name are required')
        self.latest[name] = value
        self.event('metric', name=name, value=value)
        # Prometheus text exposition for a local textfile collector; no listener.
        body = ''.join(f'hydra_{k} {v}\n' for k, v in sorted(self.latest.items()))
        temp = self.path / 'metrics.prom.tmp'
        temp.write_text(body)
        temp.replace(self.path / 'metrics.prom')

    @contextmanager
    def stage(self, name):
        self.check()
        if not re.fullmatch('[a-z][a-z0-9_]*', name):
            raise ValueError('Invalid stage name')
        tick = self.clock()
        self.event('stage_started', name=name)
        try:
            yield
        except BaseException as exc:
            self.event('stage_failed', name=name, error_type=type(exc).__name__)
            raise
        else:
            self.metric(name + '_seconds', self.clock() - tick)
            self.check()

    def artifact(self, path):
        original = Path(path)
        if original.is_symlink():
            raise ValueError('Artifact symlinks are not accepted')
        path = original.resolve()
        if not path.is_relative_to(self.path.resolve()) or path.is_symlink() or not path.is_file():
            raise ValueError('Artifact must be a regular file within this run directory')
        before = path.stat()
        h = hashlib.sha256()
        count = 0
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(2**20), b''):
                h.update(block)
                count += len(block)
        after = path.stat()
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if stamp(before) != stamp(after) or count != before.st_size:
            raise ValueError('Artifact changed during hashing')
        record = dict(file=str(path.relative_to(self.path.resolve())),
                      bytes=path.stat().st_size, sha256=h.hexdigest())
        self.event('artifact', **record)
        return record

    def finish(self, status='COMPLETED', **fields):
        if self.status != 'RUNNING' or status not in {'COMPLETED', 'FAILED', 'STOPPED'}:
            raise ValueError('Exactly one valid terminal transition required')
        self.metric('run_active', 0)
        self.metric('terminal_status', {'COMPLETED': 1, 'FAILED': 2, 'STOPPED': 3}[status])
        self.status = status
        self.event('finished', status=status, **fields)
        self._state()
