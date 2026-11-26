import copy,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from competition_live.profile import load_profile
from competition_live.execution import prepare_pas_order
from .engine import SyntheticVenue,build,metrics

class VenueTests(unittest.TestCase):
    def case(self,side=1):
        p=load_profile();start=p['start_unix'];t=np.arange(start-120,start+360,60)*1000
        d=dict(times=t,close=np.full((len(t),10),100.),high=np.full((len(t),10),100.),low=np.full((len(t),10),100.),
            volume=np.full((len(t),10),1000.),active=np.ones((len(t),10),bool),age_ms=np.zeros((len(t),10)))
        sc=dict(fee_bps=9.,impact_bps=0.,participation=.01,receipt_multiplier=1.)
        v=SyntheticVenue(d,p,sc,[],start,start+300);port=build(v,'incumbent')
        target=np.array([side*1.]+[0.]*9)
        return p,d,v,port,target

    def execute(self,v,port,target):
        with patch('competition_live.live_account.time',SimpleNamespace(time=lambda:v.now)):
            port.submit_baseline(int(v.now),target,v.streams(),v.marks(),np.eye(10)*1e-6,False)

    def test_production_path_allocates_fill_and_fee_and_accounts_conserve(self):
        p,d,v,port,target=self.case();self.execute(v,port,target)
        self.assertEqual(v.now,p['start_unix']+60)
        self.assertEqual(port.account.physical.qty[0],1.)
        self.assertEqual(port.account.owners['base'].qty[0],1.)
        self.assertAlmostEqual(port.account.reconcile(v.marks()),49999.91)
        self.assertEqual(len(port.account.applied_fills),1)
        self.assertEqual(port.account.pending_orders,{})
        self.assertTrue(v.rows[1][0]['synthetic'])

    def test_buy_limit_cannot_fill_future_high_above_limit(self):
        p,d,v,port,target=self.case();d['high'][v.index()+1,0]=101.
        self.execute(v,port,target)
        self.assertEqual(port.account.physical.qty[0],0.)
        self.assertEqual(v.audit['limit_not_crossed'],1)

    def test_short_limit_cannot_fill_future_low_below_limit(self):
        p,d,v,port,target=self.case(-1);d['low'][v.index()+1,0]=99.
        self.execute(v,port,target)
        self.assertEqual(port.account.physical.qty[0],0.)

    def test_volume_requires_prior_and_future_activity(self):
        p,d,v,port,target=self.case();d['active'][v.index()+1,0]=False
        self.execute(v,port,target);self.assertEqual(port.account.physical.qty[0],0.)

    def test_future_price_changes_do_not_change_visible_book_or_mark(self):
        p,d,v,port,target=self.case();before=copy.deepcopy(v.streams());m=v.marks()
        d['close'][v.index()+1:]*=2
        self.assertEqual(v.streams(),before);np.testing.assert_array_equal(v.marks(),m)

    def test_missing_nonzero_funding_cannot_become_zero_or_price_proxy(self):
        p,d,v,port,target=self.case();self.execute(v,port,target)
        v.funding=[dict(time=int(v.now*1000),coin='pas:BTC',rate=.001)]
        with self.assertRaisesRegex(ValueError,'Missing settled funding cash'):v.settle(int(v.now*1000),port.account)

    def test_funding_before_arrival_does_not_pay_unheld_position(self):
        p,d,v,port,target=self.case();v.funding=[dict(time=int((v.now+60)*1000),coin='pas:BTC',rate=.001,cash_per_unit=-.1)]
        self.execute(v,port,target);self.assertEqual(port.account.physical.funding,0.)

    def test_actual_held_funding_once_with_full_payment(self):
        p,d,v,port,target=self.case();self.execute(v,port,target)
        v.funding=[dict(time=int((v.now+60)*1000),coin='pas:BTC',rate=.001,cash_per_unit=-.1)]
        v.settle(int((v.now+60)*1000),port.account);v.settle(int((v.now+60)*1000),port.account)
        self.assertAlmostEqual(port.account.physical.funding,-.1)
        self.assertAlmostEqual(port.account.owners['base'].funding,-.1)

    def test_actual_reversal_closes_before_opening_other_side(self):
        p,d,v,port,target=self.case();self.execute(v,port,target)
        self.execute(v,port,-target)
        self.assertEqual(port.account.physical.qty[0],0.)
        self.assertTrue(v.orders[-1]['reduce_only'])

    def test_half_receipt_haircut_is_not_applied_to_liability(self):
        p,d,v,port,target=self.case();self.execute(v,port,target);v.sc['receipt_multiplier']=.5
        v.funding=[dict(time=int((v.now+60)*1000),coin='pas:BTC',rate=-.001,cash_per_unit=.1),
                   dict(time=int((v.now+120)*1000),coin='pas:BTC',rate=.001,cash_per_unit=-.1)]
        v.settle(int((v.now+120)*1000),port.account)
        self.assertAlmostEqual(port.account.physical.funding,-.05)

if __name__=='__main__':unittest.main()
