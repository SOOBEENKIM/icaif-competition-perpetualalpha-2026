import time
import unittest
from .forward import StrictPaper,evaluate,events,RELEASE


class ForwardTests(unittest.TestCase):
    def paper(self):
        p=StrictPaper('baseline',{},dict(visible_depth_fraction=1.,fee_bps=9),{'par:BTC':{'szDecimals':4}})
        order=dict(coin='par:BTC',phase='increase',buy=True,size=1.,limit=101.,arrives=2000,id='test',submitted=1000)
        p.pending=[order]
        return p

    def book(self,at):
        d={'time':at,'levels':[[{'px':'99','sz':'2'}],[{'px':'100','sz':'2'}]]}
        return {'par:BTC':dict(received_ms=at,data=d)}

    def test_no_execution_on_pre_order_snapshot(self):
        p=self.paper();b=self.book(1500);p.depth.update('par:BTC',b['par:BTC']['data'])
        p.apply_pending(2000,b)
        self.assertEqual(len(p.fills),0);self.assertEqual(len(p.pending),1)
        p.apply_pending(7001,b)
        self.assertEqual(len(p.fills),0);self.assertEqual(len(p.pending),0)

    def test_new_post_latency_book_fills_and_reconciles(self):
        p=self.paper();b=self.book(2500);p.depth.update('par:BTC',b['par:BTC']['data'])
        p.apply_pending(2500,b)
        self.assertEqual(len(p.pending),0);self.assertEqual(p.qty['par:BTC'],1.)
        self.assertAlmostEqual(p.cash,1e6-100-.09)

    def test_results_cannot_be_read_before_release(self):
        future={'release_unix':time.time()+3600}
        with self.assertRaises(ValueError):evaluate(future)
        with self.assertRaises(ValueError):next(events(future))


if __name__=='__main__':unittest.main()
