import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from .policy import reconcile_adjustment


class HysteresisTests(unittest.TestCase):
    def check(self,terminal,outcome='renewed'):
        pair=dict(ids=[0,1],rehedge_plan=dict(joint_terminal_prices=terminal,new_maturity=86400))
        d=dict(times=np.array([0,3600]),par_close=np.array([[100.,50.],[100.,50.]]),
               reference_close=np.array([[100.,50.],[100.,50.]]),known_hourly_funding=np.zeros((2,2)))
        cfg=dict(remaining_value=dict(arrival_hysteresis=True,exit_buffer_bps=2),renewal_policy=dict(incremental_buffer_bps=5))
        p=dict(execution=dict(decimals=[2,2]));sc=dict(fee_bps=0,impact_bps=0)
        sentinel=dict(outcome=outcome)
        with patch('experiments.remaining_hysteresis_pairs_20261017.policy.original_reconcile',return_value=sentinel):
            result=reconcile_adjustment(pair,SimpleNamespace(qty=np.array([1.,-2.])),0,d,{},cfg,p,sc,np.zeros(2),1e6,
                                         all_outcomes_observed=outcome!='awaiting_reconciliation')
        return result,pair,sentinel

    def test_positive_value_below_new_order_buffer_retains(self):
        result,pair,_=self.check([100.05,50.])
        self.assertEqual(result['outcome'],'renewed')
        self.assertLess(result['joint_arrival_advantage'],result['joint_pretrade_acceptance_buffer'])
        self.assertGreater(result['joint_arrival_advantage'],0)
        self.assertNotIn('closing',pair)

    def test_inside_negative_no_trade_band_retains(self):
        result,pair,_=self.check([99.98,50.])
        self.assertEqual(result['outcome'],'renewed')
        self.assertAlmostEqual(result['joint_arrival_advantage'],-.02)
        self.assertNotIn('closing',pair)

    def test_negative_value_beyond_band_requests_sticky_close(self):
        result,pair,_=self.check([99.9,50.])
        self.assertEqual(result['outcome'],'unwind')
        self.assertTrue(pair['closing']);self.assertTrue(pair['rehedge_failed'])

    def test_unknown_or_original_risk_unwind_has_priority(self):
        for status in ['awaiting_reconciliation','unwind','unfilled']:
            result,pair,sentinel=self.check([200.,1.],status)
            self.assertIs(result,sentinel)
            self.assertIn('rehedge_plan',pair)


if __name__=='__main__':unittest.main()
