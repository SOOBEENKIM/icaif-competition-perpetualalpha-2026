import copy,io,json,shutil,tempfile,unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .switch_policy import main
from .migration import load_existing
from competition_live.profile import ROOT,load_profile
from competition_live.runtime import fingerprint,json_save

class SwitchTests(unittest.TestCase):
    def test_switch_and_rollback_use_latest_book_and_preserve_fills(self):
        fixture=ROOT/'reports/pullback_activation_20261023_0200'
        packet=json.loads((fixture/'stage/signal.json').read_text())
        now=packet['body']['published_at']+1
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'runtime';d.mkdir()
            for f in (fixture/'rollback/runtime').iterdir():shutil.copy2(f,d/f.name)
            before,_=load_existing(d/'coordinator.pkl',load_profile())
            for number,policy in enumerate(['pullback_any3','budget_lifecycle']):
                body=copy.deepcopy(packet['body']);body['execution_policy']=policy;body['frame']['execution_policy']=policy
                if policy=='budget_lifecycle':body['frame'].pop('venue_history',None)
                signal=Path(tmp)/f'signal{number}.json';config=Path(tmp)/f'config{number}.json'
                json_save(signal,dict(body=body,sha256=fingerprint(body)));json_save(config,dict(policy=policy))
                argv=['switch_policy','--directory',str(d),'--signal',str(signal),'--config',str(config),'--archive',str(Path(tmp)/f'before{number}')]
                with patch('sys.argv',argv),patch('candidate_handoff.switch_policy.WriterLease'),patch('time.time',return_value=now),redirect_stdout(io.StringIO()):main()
                after,_=load_existing(d/'coordinator.pkl',load_profile())
                self.assertEqual(after.execution_policy,policy)
                self.assertEqual(len(after.account.physical.fills),len(before.account.physical.fills))
                self.assertEqual(after.account.physical.fees,before.account.physical.fees)
                self.assertEqual(after.last_decision,before.last_decision)
                self.assertEqual(after.counter,before.counter)
                np.testing.assert_array_equal(after.account.physical.qty,before.account.physical.qty)
                np.testing.assert_array_equal(after.account.reserved.qty,before.account.reserved.qty)
                self.assertIsNone(after.account.executor)

if __name__=='__main__':unittest.main()
