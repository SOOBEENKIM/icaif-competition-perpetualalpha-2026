import copy,json,unittest
from pathlib import Path
import numpy as np
from strategy.agent import TAIL
from .book_plan import plan_entry

HERE=Path(__file__).resolve().parent


class BookPlannerTests(unittest.TestCase):
    def setUp(self):
        self.p=json.loads((TAIL/'baseline.json').read_text())['protocol']
        self.cfg=json.loads((HERE/'par_protocol.json').read_text())
        self.cfg.update(book_intent_lifetime_hours=6,book_max_age_ms=15000,context_max_age_ms=3000,
            visible_depth_band_bps=10,entry_limit_extra_bps=10,pair_max_gross=1000)
        self.sc=dict(depth_fraction=1.,fee_bps=9,exit_impact_bps=10)
        self.now=1800000000
        self.intent=dict(pair=['ETH','BTC'],ids=[1,0],direction=1,beta=1.,entry_mean=.05,entry_sd=.01,
            terminal_prices=[103.,97.],forecast_source=self.now-60,account_marks=[100.]*10)
        q=dict(book={'time':self.now*1000,'levels':[[{'px':'99.99','sz':'100'}],[{'px':'100.01','sz':'100'}]]},
            ctx={'funding':'0'},oracle=100.,received={'book':self.now*1000,'ctx':self.now*1000})
        self.books={s:copy.deepcopy(q) for s in ('ETH','BTC')}

    def plan(self):return plan_entry(self.intent,self.books,self.now,self.cfg,self.p,self.sc,np.zeros(10),1e6,np.eye(10)*1e-7)

    def test_visible_depth_can_plan_without_any_candle_volume_input(self):
        r=self.plan();self.assertTrue(r['accepted']);self.assertEqual(len(r['orders']),2)
        self.assertGreaterEqual(r['minimum_expected_cash_at_limits'],r['buffer_cash']-1e-8)
        self.assertTrue(r['requires_independent_fill_reconciliation']);self.assertEqual(r['orders_sent'],0)

    def test_missing_or_stale_book_is_not_invented_liquidity(self):
        self.books['BTC']['received']['book']-=15001;self.assertFalse(self.plan()['accepted'])
        del self.books['BTC'];self.assertEqual(self.plan()['reason'],'unobserved_book')

    def test_source_in_future_or_expired_rejects(self):
        self.intent['forecast_source']=self.now+1;self.assertEqual(self.plan()['reason'],'future_signal')
        self.intent['forecast_source']=self.now-6*3600;self.assertEqual(self.plan()['reason'],'expired_signal')

    def test_entry_depth_without_reverse_depth_cannot_imply_free_exit(self):
        self.books['ETH']['book']['levels'][0][0]['sz']='0.001'
        self.assertEqual(self.plan()['reason'],'visible_entry_or_exit_depth_missing')

    def test_price_edge_must_cover_both_fees_and_exit_reserve(self):
        self.intent['terminal_prices']=[100.02,99.98]
        self.assertEqual(self.plan()['reason'],'edge_below_full_cost')

    def test_funding_cost_changes_budget_and_positive_forecast_is_haircut(self):
        flat=self.plan();self.books['ETH']['ctx']['funding']='0.0001';pay=self.plan()
        self.assertTrue(pay['accepted']);self.assertLess(pay['expected_net_cash'],flat['expected_net_cash'])
        self.books['ETH']['ctx']['funding']='-0.0001';receive=self.plan()
        self.assertAlmostEqual(receive['expected_funding_cash'],-.5*pay['expected_funding_cash'])

    def test_partial_depth_lowers_both_leg_size(self):
        self.cfg['pair_max_gross']=50000;a=self.plan();self.sc['depth_fraction']=.5;b=self.plan()
        self.assertTrue(a['accepted'] and b['accepted']);self.assertLess(abs(b['target'][0]),abs(a['target'][0]))
        self.assertAlmostEqual(abs(b['target'][0]/b['target'][1]),1.)


if __name__=='__main__':unittest.main()
