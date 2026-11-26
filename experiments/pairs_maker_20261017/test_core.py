import unittest
import numpy as np
from .core import FlowFilter, PassiveOrder, queue_at_arrival


def trade(tid, side='A', price=100, qty=1, time=1100):
    return dict(coin='par:BTC',time=time,tid=tid,side=side,px=str(price),sz=str(qty))


class QueueTests(unittest.TestCase):
    def test_queue_volume_precedes_our_fill_and_duplicate_does_not_fill_twice(self):
        o=PassiveOrder('par:BTC',1,100,2,3,1000,3000,3)
        self.assertEqual(o.trade(1100,trade(1,qty=2)),0)
        self.assertEqual(o.trade(1100,trade(1,qty=2)),0)
        self.assertEqual(o.trade(1100,trade(2,qty=2)),1)
        self.assertEqual(o.trade(1100,trade(3,qty=4)),1)
        self.assertEqual(o.trade(1100,trade(4,qty=4)),0)

    def test_wrong_side_price_and_old_snapshot_cannot_fill(self):
        o=PassiveOrder('par:BTC',1,100,2,0,1000,3000,3)
        for tr in (trade(1,side='B'),trade(2,price=101),trade(3,time=500),trade(4,time=3000),trade(6,time=1000)):
            self.assertEqual(o.trade(1100,tr),0)
        self.assertEqual(o.trade(2999,trade(5,time=2999)),1)

    def test_cancellation_does_not_have_zero_latency(self):
        o=PassiveOrder('par:BTC',-1,100,2,0,1000,3000,3)
        # Cancel requested at 2000 but only effective at 3000.
        self.assertEqual(o.trade(2500,trade(1,side='B',time=2500)),1)
        self.assertEqual(o.trade(3000,trade(2,side='B',time=3000)),0)

    def test_post_only_rejection_and_unknown_queue(self):
        b=dict(levels=[[dict(px='99',sz='4'),dict(px='98',sz='2')],[dict(px='101',sz='3')]])
        self.assertEqual(queue_at_arrival(b,99,1),(4.,'accepted'))
        self.assertEqual(queue_at_arrival(b,101,1),(None,'post_only_rejected'))
        self.assertEqual(queue_at_arrival(b,97,1),(None,'queue_outside_observed_depth'))


class FilterTests(unittest.TestCase):
    def test_bernoulli_update_and_direction_change(self):
        f=FlowFilter(); self.assertEqual(f.predict()['buy_probability'],.5)
        self.assertAlmostEqual(f.observe(1)['buy_probability'],2/3)
        for _ in range(29): f.observe(1)
        self.assertGreater(f.predict()['buy_probability'],.9)
        for _ in range(20): f.observe(0)
        self.assertLess(f.predict()['buy_probability'],.15)
        self.assertAlmostEqual(f.mass.sum(),1.)

    def test_future_observations_do_not_change_saved_past_predictions(self):
        sides=[0,1,1,0,0,1,1,1,0]*10
        a=FlowFilter(); b=FlowFilter()
        first=[a.observe(x) for x in sides[:30]]
        second=[b.observe(x) for x in sides[:30]]
        for x in sides[30:]: a.observe(x)
        self.assertEqual(first,second)
        self.assertTrue(np.isfinite(a.mass).all())


if __name__=='__main__':unittest.main()
