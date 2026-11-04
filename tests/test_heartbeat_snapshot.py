"""Deterministic local telemetry races; no model calls or scientific inputs."""
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from hydra_directflow.corpus_dataset import _read as immutable_read
from hydra_directflow.heartbeat_snapshot import MAX_HEARTBEAT_BYTES, read_snapshot


class HeartbeatSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "heartbeat.json"
        self.old = b'{"version":1}'
        self.new = b'{"version":2,"new":true}'
        self.path.write_bytes(self.old)

    def tearDown(self):
        self.temp.cleanup()

    def replacement(self):
        target = self.root / "heartbeat.json.writing"
        target.write_bytes(self.new)
        return target

    def test_regular_bytes_and_json_parsing_remains_callers_responsibility(self):
        self.assertEqual(read_snapshot(self.path), self.old)
        self.path.write_bytes(b"invalid json")
        self.assertEqual(read_snapshot(self.path), b"invalid json")

    def test_atomic_replacement_after_open_returns_coherent_old_snapshot(self):
        replacement = self.replacement()
        real_read = os.read
        replaced = False

        def replace_after_read(fd, count):
            nonlocal replaced
            result = real_read(fd, count)
            if not replaced:
                os.replace(replacement, self.path)
                replaced = True
            return result

        with patch("hydra_directflow.heartbeat_snapshot.os.read", side_effect=replace_after_read):
            self.assertEqual(read_snapshot(self.path), self.old)
        self.assertEqual(self.path.read_bytes(), self.new)

    def test_atomic_replacement_before_open_returns_coherent_new_snapshot(self):
        replacement = self.replacement()
        real_open = os.open

        def replace_before_open(path, flags):
            os.replace(replacement, self.path)
            return real_open(path, flags)

        with patch("hydra_directflow.heartbeat_snapshot.os.open", side_effect=replace_before_open):
            self.assertEqual(read_snapshot(self.path), self.new)

    def test_existing_immutable_reader_still_rejects_the_same_replacement(self):
        replacement = self.replacement()
        real_open = Path.open
        owner = self

        class ReplaceOnRead:
            def __enter__(self):
                self.stream = real_open(owner.path, "rb")
                return self

            def read(self, count):
                raw = self.stream.read(count)
                os.replace(replacement, owner.path)
                return raw

            def __exit__(self, *unused):
                self.stream.close()

        with patch.object(Path, "open", return_value=ReplaceOnRead()):
            with self.assertRaisesRegex(ValueError, "Corpus input changed during read"):
                immutable_read(self.path, MAX_HEARTBEAT_BYTES)

    def test_same_length_in_place_change_is_rejected_and_descriptor_closed(self):
        real_read, real_close = os.read, os.close
        before = self.path.stat()
        changed = False

        def mutate_after_read(fd, count):
            nonlocal changed
            result = real_read(fd, count)
            if not changed:
                self.path.write_bytes(b'{"version":9}')
                os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
                changed = True
            return result

        with patch("hydra_directflow.heartbeat_snapshot.os.read", side_effect=mutate_after_read), \
                patch("hydra_directflow.heartbeat_snapshot.os.close", wraps=real_close) as close:
            with self.assertRaisesRegex(ValueError, "Heartbeat contents changed during read"):
                read_snapshot(self.path)
            close.assert_called_once()

    def test_append_during_read_is_rejected(self):
        real_read = os.read
        changed = False

        def append_after_read(fd, count):
            nonlocal changed
            result = real_read(fd, count)
            if not changed:
                with self.path.open("ab") as stream:
                    stream.write(b"extra")
                changed = True
            return result

        with patch("hydra_directflow.heartbeat_snapshot.os.read", side_effect=append_after_read):
            with self.assertRaisesRegex(ValueError, "Heartbeat contents changed during read"):
                read_snapshot(self.path)

    def test_partial_reads_are_accumulated(self):
        real_read = os.read
        with patch("hydra_directflow.heartbeat_snapshot.os.read", side_effect=lambda fd, count: real_read(fd, min(count, 3))):
            self.assertEqual(read_snapshot(self.path), self.old)

    def test_empty_and_oversized_files_are_rejected_before_open(self):
        for raw in (b"", b"x" * (MAX_HEARTBEAT_BYTES + 1)):
            with self.subTest(size=len(raw)):
                self.path.write_bytes(raw)
                with patch("hydra_directflow.heartbeat_snapshot.os.open") as opened:
                    with self.assertRaisesRegex(ValueError, "Heartbeat byte bound"):
                        read_snapshot(self.path)
                    opened.assert_not_called()

    def test_caller_maximum_is_enforced_and_cannot_exceed_64k(self):
        for limit in (0, -1, True, 3.5, MAX_HEARTBEAT_BYTES + 1):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, "Heartbeat maximum"):
                    read_snapshot(self.path, limit)
        with self.assertRaisesRegex(ValueError, "Heartbeat byte bound"):
            read_snapshot(self.path, len(self.old) - 1)

    def test_symlink_directory_and_fifo_rejected_without_opening(self):
        link = self.root / "link.json"
        link.symlink_to(self.path)
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        for path in (link, self.root, fifo):
            with self.subTest(path=path.name), patch("hydra_directflow.heartbeat_snapshot.os.open") as opened:
                with self.assertRaisesRegex(ValueError, "resident regular file"):
                    read_snapshot(path)
                opened.assert_not_called()

    def test_symlink_substituted_after_lstat_is_rejected_by_nofollow(self):
        target = self.root / "target"
        target.write_bytes(self.new)
        real_open = os.open

        def symlink_before_open(path, flags):
            self.path.unlink()
            self.path.symlink_to(target)
            self.assertTrue(flags & os.O_NOFOLLOW)
            self.assertTrue(flags & os.O_NONBLOCK)
            return real_open(path, flags)

        with patch("hydra_directflow.heartbeat_snapshot.os.open", side_effect=symlink_before_open):
            with self.assertRaises(OSError):
                read_snapshot(self.path)

    def test_cloud_only_flag_rejected_before_open(self):
        info = self.path.lstat()
        offloaded = SimpleNamespace(st_mode=info.st_mode, st_size=info.st_size, st_flags=0x40000000)
        with patch.object(Path, "lstat", return_value=offloaded), \
                patch("hydra_directflow.heartbeat_snapshot.os.open") as opened:
            with self.assertRaisesRegex(ValueError, "resident regular file"):
                read_snapshot(self.path)
            opened.assert_not_called()

    def test_open_descriptor_is_rechecked_for_residency_before_read(self):
        info = self.path.lstat()
        offloaded = SimpleNamespace(st_mode=info.st_mode, st_size=info.st_size, st_flags=0x40000000)
        with patch("hydra_directflow.heartbeat_snapshot.os.fstat", return_value=offloaded), \
                patch("hydra_directflow.heartbeat_snapshot.os.read") as read:
            with self.assertRaisesRegex(ValueError, "resident regular file"):
                read_snapshot(self.path)
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
