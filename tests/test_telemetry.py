import json
from pathlib import Path
import tempfile
import unittest

from hydra_directflow.telemetry import RunJournal


class JournalTests(unittest.TestCase):
    def test_terminal_state_and_portable_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            run = RunJournal(Path(d)/'run', {'stage':'synthetic'})
            with run.stage('compute'):
                output = run.path/'result.json'
                output.write_text('{"ok":true}')
            self.assertEqual(run.artifact(output)['file'], 'result.json')
            run.finish()
            self.assertEqual(json.loads((run.path/'state.json').read_text())['status'], 'COMPLETED')
            self.assertIn('hydra_run_active 0', (run.path/'metrics.prom').read_text())
            self.assertIn('hydra_terminal_status 1', (run.path/'metrics.prom').read_text())
            with self.assertRaises(RuntimeError):run.metric('loss',1.)
            with self.assertRaises(ValueError):run.finish()

    def test_nonfinite_metrics_reject_before_log(self):
        with tempfile.TemporaryDirectory() as d:
            run = RunJournal(Path(d)/'run', {})
            for value in (float('nan'),float('inf'),True):
                with self.assertRaises(ValueError):run.metric('loss',value)
            self.assertNotIn('loss',run.latest)
            run.finish('FAILED')

    def test_expired_attempt_stops(self):
        tick=[0.]
        with tempfile.TemporaryDirectory() as d:
            run = RunJournal(Path(d)/'run',{},max_wall_seconds=1,clock=lambda:tick[0])
            tick[0]=1.
            with self.assertRaises(RuntimeError):run.check()
            self.assertEqual(run.status,'STOPPED')

    def test_existing_attempt_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run';run=RunJournal(p,{})
            with self.assertRaises(FileExistsError):RunJournal(p,{})
            run.finish('STOPPED')

    def test_external_artifacts_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            run=RunJournal(Path(d)/'run',{});p=Path(d)/'outside';p.write_text('x')
            with self.assertRaises(ValueError):run.artifact(p)
            inside=run.path/'inside';inside.write_text('x')
            link=run.path/'link';link.symlink_to(inside)
            with self.assertRaises(ValueError):run.artifact(link)
            run.finish()

if __name__=='__main__':unittest.main()
