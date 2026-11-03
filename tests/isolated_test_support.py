"""Keep no-model loader checks independent of other tests' Torch imports."""
from pathlib import Path
import os
import subprocess
import sys


def rerun_without_model_imports_if_needed(case):
    if not any(name.split('.')[0] in ('torch','transformers') for name in sys.modules):
        return False
    root=Path(__file__).resolve().parents[1]
    code=('import sys, unittest; from pathlib import Path; '
          'root=Path.cwd(); sys.path[:0]=[str(root/"src"),str(root/"tests")]; '
          'suite=unittest.defaultTestLoader.loadTestsFromName(sys.argv[1]); '
          'result=unittest.TextTestRunner(verbosity=2).run(suite); '
          'sys.exit(0 if result.wasSuccessful() else 1)')
    result=subprocess.run([sys.executable,'-I','-B','-c',code,case.id()],cwd=root,
        env={**os.environ,'PYTHONNOUSERSITE':'1'},capture_output=True,text=True,timeout=60)
    case.assertEqual(result.returncode,0,result.stdout+result.stderr)
    return True
