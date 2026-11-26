import unittest
import numpy as np
from .net_value import state
from experiments.decision_gate_pairs_20261017.entry import filter_proposals
from experiments.quote_anchored_pairs_20261017.pricing import reservation_limits


class TotalValueGateTests(unittest.TestCase):
    def fixture(self):
        return dict(mu=np.array([[0.,.01]]),economic_available=np.array([[True,True]]),economic_action=np.array([[0,1]]),beta=np.ones((1,2)),vol=np.ones((1,2))*.001,relationship_allowed=np.array([[True,False]]),regime_allowed=np.ones((1,2),bool))

    def test_positive_funding_value_was_vetoed_by_price_only_gate(self):
        # 1000gross, expectedprice0, discountedcarry10, explicitentry/exitfee
        # andimpact. Existing economic planner approves; old pricegate vetoes.
        q=np.array([5.,-5.]);prices=np.array([100.,100.]);preview=np.array([100.1,99.9])
        budget=reservation_limits(q,prices,preview,0.,10.,9.,10.,5.,10.)
        self.assertIsNotNone(budget);self.assertGreater(budget['expected_net_cash'],0.)
        self.assertEqual(filter_proposals([dict(pair_index=0,direction=1)],np.array([0]),np.array([True])),[])
        repaired=state(self.fixture());self.assertEqual(repaired['directions'][0,0],1)
        self.assertNotIn('economic_action',repaired)

    def test_failed_relation_stays_blocked(self):
        self.assertEqual(state(self.fixture())['directions'][0,1],0)

    def test_zero_price_and_no_carry_does_not_cover_cost(self):
        self.assertIsNone(reservation_limits(np.array([5.,-5.]),np.array([100.,100.]),np.array([100.1,99.9]),0.,0.,9.,10.,5.,10.))

    def test_carry_ablation_is_explicit_zero_not_fabricated_price_prediction(self):
        original=self.fixture();out=state(original,carry_only=True)
        self.assertEqual(out['mu'][0,0],0.);self.assertTrue(np.isnan(out['mu'][0,1]));self.assertEqual(original['mu'][0,1],.01)


if __name__=='__main__':unittest.main()
