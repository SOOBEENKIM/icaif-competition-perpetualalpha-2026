import unittest
from unittest.mock import patch
import numpy as np
from .policy import intersect_limits,renewal_plan
from experiments.remaining_value_pairs_20261017.test_policy import RemainingTests


class BudgetTests(unittest.TestCase):
    def test_integer_price_input_keeps_fractional_execution_cost(self):
        r=intersect_limits([1,-2],[2,-4],[100,50],[0,1],{'0':120,'1':40},5.,
                           {'execution':{'decimals':[2,2]}},{'fee_bps':0,'impact_bps':10})
        self.assertAlmostEqual(r['preview_cost'],.2)

    def test_limits_tighten_both_sides_and_fund_worst_cost(self):
        p={'execution':{'decimals':[2,2]}};sc={'fee_bps':10,'impact_bps':0}
        limits=intersect_limits([1,-2],[2,-4],[100,50],[0,1],{'0':120,'1':40},5.,p,sc)
        self.assertIsNotNone(limits)
        self.assertLess(limits['entry_limits']['0'],120)
        self.assertGreater(limits['entry_limits']['1'],40)
        # Buy one and sell two: loss vs marks plus one fee on every traded unit.
        buy,sell=limits['entry_limits']['0'],limits['entry_limits']['1']
        independent=(buy-100)+2*(50-sell)+.001*(buy+2*sell)
        self.assertAlmostEqual(independent,5.)

    def test_large_budget_never_widens_existing_limits(self):
        r=intersect_limits([1,-2],[2,-4],[100,50],[0,1],{'0':101,'1':49.5},1000.,
                           {'execution':{'decimals':[2,2]}},{'fee_bps':10,'impact_bps':0})
        self.assertEqual(r['entry_limits'],{'0':101.,'1':49.5})

    def test_insufficient_preview_budget_and_empty_trade_rejected(self):
        args=([1,-2],[2,-4],[100,50],[0,1],{'0':120,'1':40},.1,
              {'execution':{'decimals':[2,2]}},{'fee_bps':10,'impact_bps':0})
        self.assertIsNone(intersect_limits(*args))
        with self.assertRaises(ValueError):intersect_limits([1,-2],[1,-2],*args[2:])

    def test_wide_limit_veto_becomes_valid_bounded_renewal(self):
        pair,d,de,cfg,p,sc=RemainingTests().fixture();cfg['remaining_value']['mode']='budget'
        de['holding_terminal'][:]=[103.,49.]
        plan=dict(accepted=True,requires_adjustment=True,target=[2.,-4.],entry_limits={'0':120.,'1':40.},
                  new_maturity=86400,old_maturity=21600,buffer_cash=.04,
                  expected_hold_cash=100.,immediate_close_cash=-.2)
        with patch('experiments.remaining_budget_pairs_20261017.policy.base_renewal',return_value=plan):
            got=renewal_plan(pair,np.array([1.,-2.]),0,d,de,cfg,p,sc,np.array([1.,-2.]),1e6)
        self.assertTrue(got['accepted'])
        self.assertEqual(got['target'],plan['target'])
        self.assertEqual(plan['entry_limits'],{'0':120.,'1':40.})
        self.assertGreaterEqual(got['joint_advantage_at_sealed_limits'],.04-1e-8)
        self.assertGreater(got['original_advantage_at_joint_limits'],.04)


if __name__=='__main__':unittest.main()
