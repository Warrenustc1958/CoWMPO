# Copyright 2026 The RLinf Authors.
"""Run staged contracts by file path and refuse an accidentally shadowing package."""
import sys
import unittest
from pathlib import Path
import cowm
root=Path(__file__).parent
if Path(cowm.__file__).resolve()!=root/'toolkits/cowm/__init__.py':
    raise RuntimeError('unexpected cowm import: '+str(cowm.__file__))
print('TEST_PACKAGE',cowm.__file__,flush=True)
suite=unittest.defaultTestLoader.discover(str(root/'contract_tests_v2'),pattern='test_*.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
