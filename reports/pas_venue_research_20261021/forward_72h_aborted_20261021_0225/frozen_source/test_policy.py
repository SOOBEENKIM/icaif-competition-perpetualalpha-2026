import copy
import unittest
from .policy import propose,quote_price
from .shadow import Queue,consume


class VenuePolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(notional=250.,depth_fraction=.25,maker_fee_bps=3.,taker_fee_bps=9.,
            holding_seconds=60,book_max_age_seconds=15.,context_max_age_seconds=3.,
            basis_convergence_fraction=.5,exit_impact_bps=10.,entry_buffer_bps=5.,risk_z=1.645)
        self.sample=dict(now=1000.,book_received=999.,context_received=999.,
            book=dict(time=999000,levels=[[dict(px='99.98',sz='100')],[dict(px='100.02',sz='100')]]),
            context=dict(oraclePx='100.1',funding='-.0001'),size_decimals=3,reference_hourly_sigma=.001)

    def test_small_discount_does_not_override_fees_or_risk(self):
        for mode in ('maker_basis','taker_basis'):
            r=propose(self.sample,self.cfg,mode)
            self.assertFalse(r['accepted'])
            self.assertGreater(r['required_edge_bps'],r['hypothesized_price_edge_bps'])
            self.assertFalse(r['live_eligible'])

    def test_large_gap_is_only_an_unvalidated_hypothesis(self):
        self.sample['context']['oraclePx']='103'
        row=propose(self.sample,self.cfg,'maker_basis')
        self.assertTrue(row['accepted'])
        self.assertFalse(row['alpha_validated']);self.assertFalse(row['live_eligible'])
        self.assertLess(row['limit'],float(self.sample['book']['levels'][1][0]['px']))

    def test_stale_book_or_context_is_not_a_profit_opportunity(self):
        self.sample['context']['oraclePx']='103'
        for field in ('book_received','context_received'):
            bad=copy.deepcopy(self.sample);bad[field]=900
            self.assertFalse(propose(bad,self.cfg,'taker_basis')['accepted'])

    def test_limit_price_rounding_does_not_cross_to_improve_a_fill(self):
        self.assertLessEqual(quote_price(100.12345,3,True),100.12345)
        self.assertGreaterEqual(quote_price(100.12345,3,False),100.12345)
        self.assertLessEqual(quote_price(82645.89,5,True),82645.89)

    def test_passive_queue_requires_trades_and_does_not_credit_cancellations(self):
        q=Queue(1,100.,2.,5.,10.,41.)
        def trade(tid,size,when=15):
            return dict(coin='pas:BTC',tid=tid,side='A',px='100',sz=str(size),time=int(when*1000))
        self.assertEqual(q.remaining,2.) # Book touches/cancellations do nothing.
        self.assertEqual(q.trade(trade(1,4),15.1),0.)
        self.assertEqual(q.trade(trade(2,2),16.),1.)
        self.assertEqual(q.trade(trade(2,2),16.),0.) # Duplicate cannot fill twice.
        self.assertEqual(q.trade(trade(3,5,when=42),42.1),0.) # Cancel acknowledged.

    def test_passive_rejects_past_snapshot_and_wrong_aggressor(self):
        q=Queue(1,100.,2.,0.,10.,41.)
        row=dict(coin='pas:BTC',tid=1,side='A',px='99',sz='2',time=9000)
        self.assertEqual(q.trade(row,11),0)
        row.update(tid=2,side='B',time=11000)
        self.assertEqual(q.trade(row,11.1),0)

    def test_taker_limit_and_depth_are_binding_on_actual_shadow_quantity(self):
        book=dict(levels=[[dict(px='99',sz='100')],[dict(px='100',sz='1'),dict(px='102',sz='100')]])
        qty,price=consume(book,1,2,limit=101,fraction=1)
        self.assertEqual(qty,1);self.assertEqual(price,100)
        qty,_=consume(book,1,2,limit=101,fraction=.25)
        self.assertEqual(qty,.25)


if __name__=='__main__':
    unittest.main()
