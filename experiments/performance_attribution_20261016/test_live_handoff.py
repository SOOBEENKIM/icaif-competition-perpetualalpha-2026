import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from practice_live.live_store import Store
from practice_live.runner import COINS
from strategy.agent import TAIL
from experiments.risk_preserving_20261016.policy import RiskCapController
from .live import BudgetRunner,epoch_snapshot,PROFILE

class LiveHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.store=Store(Path(self.temp.name)/'state.sqlite3')
        self.p=json.loads((TAIL/'baseline.json').read_text())['protocol']
        self.runner=BudgetRunner.__new__(BudgetRunner);self.runner.store=self.store;self.runner.protocol=self.p;self.runner.signal_error=None;self.runner.publication=None
        self.t=1792173600 # UTC 18:00 / KST 03:00 next morning
        self.account={'ledger':{'equity':1e6,'external_capital':0.,'fees':39.,'funding':32.,'fills':11},
            'qty':np.array([300.,0.,60.,0.,0.,0.,0.,0.,0.,0.]),'prices':np.ones(10)*100,
            'universe':{c:({'szDecimals':4},{'oraclePx':'100'}) for c in COINS}}
        self.inputs={'source_hour':self.t,'weights':np.array([.02,0.,.006,0.,0.,0.,0.,0.,0.,0.]),
            'retention':np.array([.75,1.,1.,1.,1.,1.,1.,1.,1.,1.]),'sigma':np.eye(10)*1e-6,
            'volume':np.ones(10)*1e6,'trades':np.ones(10)*1000,'input_hash':'a'*64,'protocol':self.p}
    def tearDown(self):self.store.close();self.temp.cleanup()
    def test_live_decision_matches_compared_controller(self):
        plan=self.runner.decide(self.account,self.inputs,self.t+120)
        p=json.loads(json.dumps(self.p));p['execution']['decimals']=[4]*10
        expected=RiskCapController().propose(self.t,self.account['prices'],self.account['qty'],1e6,self.inputs['weights'],
            self.inputs['sigma'],self.inputs['volume'],self.inputs['trades'],p,buffer=.2,retention=self.inputs['retention'])
        self.assertEqual(plan['target_qty'],expected['target_qty']);self.assertEqual(plan['profile'],PROFILE)
        self.assertLess(plan['target_qty'][0],self.account['qty'][0])
    def test_completed_incumbent_round_is_never_repeated(self):
        self.store.put(f'round:normal:{self.t}',{'status':'complete'})
        self.assertIsNone(self.runner.decide(self.account,self.inputs,self.t+120))
    def test_connection_between_rounds_does_not_force_rebalance(self):
        self.inputs['source_hour']=self.t+3600
        self.assertIsNone(self.runner.decide(self.account,self.inputs,self.t+3600+120))
    def test_stale_signal_blocks_normal_but_allows_hard_risk_reduction(self):
        self.inputs['source_hour']=self.t-3600
        self.assertIsNone(self.runner.decide(self.account,self.inputs,self.t+120))
        self.account['qty'][0]=10000
        plan=self.runner.decide(self.account,self.inputs,self.t+120)
        self.assertTrue(plan['risk_forced']);self.assertLess(plan['target_qty'][0],10000)
        self.assertTrue(np.all(np.abs(plan['target_qty'])<=abs(self.account['qty'])))
    def test_drawdown_halt_operates_without_a_signal(self):
        self.store.put('peak',2e6)
        plan=self.runner.decide(self.account,None,self.t+123)
        self.assertTrue(plan['risk_forced']);np.testing.assert_array_equal(plan['target_qty'],np.zeros(10))
    def test_handoff_keeps_actual_inventory_and_original_anchor(self):
        anchor={'initial_equity':1e6,'start_ms':123,'profile':'default'};self.store.put('anchor',anchor);self.store.put('peak',1000500)
        value=epoch_snapshot(self.account,self.store)
        self.assertEqual(value['previous_anchor'],anchor);self.assertEqual(value['actual_qty'],self.account['qty'].tolist())
        self.assertEqual(self.store.get('anchor'),anchor);self.assertEqual(value['previous_peak'],1000500)
    def test_invalid_signal_cannot_enable_increases(self):
        self.runner.signal_error='bad digest'
        self.assertIsNone(self.runner.decide(self.account,self.inputs,self.t+100))

if __name__=='__main__':unittest.main()
