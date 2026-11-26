"""Reproducible, signer-free regression suite for this research package."""
import importlib
import io
import unittest
from datetime import datetime,timezone
from .protocol import ROOT,OUT,save,sha


def main():
    modules=['production_replay_20261025.'+p.stem for p in sorted((ROOT/'production_replay_20261025').glob('test_*.py'))]
    modules+=['candidate_handoff.test_handoff','candidate_handoff.test_risk_priority']
    suite=unittest.TestSuite();files={}
    for name in modules:
        module=importlib.import_module(name)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        for key,value in vars(module).items():
            if key.startswith('test_') and callable(value) and getattr(value,'__module__',None)==name:
                suite.addTest(unittest.FunctionTestCase(value))
        files[str(module.__file__)]=sha(module.__file__)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=1).run(suite)
    report=dict(at=datetime.now(timezone.utc).isoformat(),tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        log=stream.getvalue(),files=files,orders_sent=0)
    save(OUT/'validation_tests.json',report)
    print(stream.getvalue())
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
