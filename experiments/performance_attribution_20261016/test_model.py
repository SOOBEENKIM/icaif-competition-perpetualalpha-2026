import unittest
import numpy as np
from .model import economic_target,CalibratedController
from replay_lab.ema_rebuild.engine import Controller
from .agent import apply_budget

class EconomicTests(unittest.TestCase):
    def test_low_edge_blocks_only_increase(self):
        raw=np.array([.04,.005,0.,-.02]);current=np.array([.01,.01,.02,.01])
        target,blocked,edge=economic_target(raw,current,np.zeros(4),np.zeros(4),.0038)
        np.testing.assert_array_equal(target,[.01,.005,0.,0.])
    def test_good_edges_allow_both_directions(self):
        raw=np.array([.04,-.04]);target,_,_=economic_target(raw,np.zeros(2),np.array([.01,-.01]),np.zeros(2),.0038)
        np.testing.assert_array_equal(target,raw)
    def test_funding_cost_reserves_only_adverse_direction(self):
        raw=np.array([.04,-.04]);target,_,_=economic_target(raw,np.zeros(2),np.array([.01,-.01]),np.array([.001,.001]),.0038)
        np.testing.assert_array_equal(target,[0.,-.04])
    def test_nonfinite_economics_rejected(self):
        with self.assertRaises(ValueError):economic_target(np.array([.04]),np.zeros(1),np.array([float('nan')]),np.zeros(1),.0038)

    def test_economic_filter_does_not_block_emergency_reduction(self):
        import json
        from strategy.agent import TAIL
        p=json.loads((TAIL/'baseline.json').read_text())['protocol']
        prices=np.ones(10)*100;qty=np.zeros(10);qty[0]=10000
        raw=np.zeros(10);cov=np.eye(10)*1e-4
        args=(3600,prices,qty,1e6,raw,cov,np.ones(10),np.ones(10),p)
        actual=CalibratedController([3600],np.zeros((1,10)),np.zeros((1,10)),.0038).propose(*args)
        wanted=Controller().propose(*args)
        self.assertTrue(actual['risk_forced']);self.assertEqual(actual['target_qty'],wanted['target_qty'])

class BudgetTests(unittest.TestCase):
    def test_warned_assets_still_reduced_after_budget_calibration(self):
        before=np.array([.04,-.04,.04]);retention=np.array([1.,.75,.5])
        after=apply_budget(before,retention,1.295869)
        self.assertTrue(np.all(abs(after[1:])<abs(before[1:])))
        self.assertGreater(after[0],before[0])
        np.testing.assert_allclose(after,before*retention*1.295869)
    def test_budget_that_erases_reduction_is_rejected(self):
        with self.assertRaises(ValueError):apply_budget(np.array([.04]),np.array([.75]),1.4)
    def test_invalid_retention_is_rejected(self):
        with self.assertRaises(ValueError):apply_budget(np.array([.04]),np.array([.9]),1.29)

if __name__=='__main__':unittest.main()
