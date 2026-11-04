"""Bounded reads of atomically replaced, local worker heartbeat snapshots.

This reader is only for mutable telemetry. Immutable source, checkpoint and
receipt readers must keep their stricter pathname-identity checks.
"""
from __future__ import annotations

import os
from pathlib import Path
import stat

from .reference import require


MAX_HEARTBEAT_BYTES = 64 * 1024
_OFFLOADED = 0x40000000


def _check_file(info, maximum):
    require(stat.S_ISREG(info.st_mode) and not getattr(info, "st_flags", 0) & _OFFLOADED,
            "Heartbeat must be a resident regular file; no symlinks or hydration")
    require(0 < info.st_size <= maximum, "Heartbeat byte bound exceeded")


def read_snapshot(path, maximum=MAX_HEARTBEAT_BYTES):
    """Read one complete heartbeat version even if its pathname is replaced.

    The open descriptor pins either the previous or new atomically published
    file. Its content size and modification timestamp must remain unchanged
    during the read. Link-count/ctime changes from replacing the pathname are
    allowed; replacing a pathname does not modify the pinned file's contents.
    JSON parsing and heartbeat identity validation remain the caller's job.
    """
    require(type(maximum) is int and 0 < maximum <= MAX_HEARTBEAT_BYTES,
            "Heartbeat maximum must be a positive bound of at most 64 KiB")
    path = Path(path)
    _check_file(path.lstat(), maximum)
    # NONBLOCK prevents a file-type substitution from blocking on a FIFO;
    # NOFOLLOW rejects a symlink substituted after the initial lstat.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _check_file(before, maximum)
        pieces = []
        remaining = maximum + 1
        while remaining:
            block = os.read(fd, remaining)
            if not block:
                break
            pieces.append(block)
            remaining -= len(block)
        raw = b"".join(pieces)
        after = os.fstat(fd)
        _check_file(after, maximum)
        content_key = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        require(content_key(before) == content_key(after) and len(raw) == before.st_size,
                "Heartbeat contents changed during read")
        return raw
    finally:
        os.close(fd)
