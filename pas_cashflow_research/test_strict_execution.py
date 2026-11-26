import unittest
import numpy as np
from .strict_execution import simulate
from competition_live.profile import load_profile


class TemporalContract(unittest.TestCase):
    def test_bar_overlapping_decision_cannot_fill(self):
        p=load_profile()['trading'];start=43200000;end=start+3600000
        prices=np.full(10,100.);sigma=np.eye(10)*1e-6
        times=np.arange(start,end+1,60000)
        d=dict(times=times,close=np.tile(prices,(len(times),1)),age_ms=np.zeros((len(times),10)),active=np.ones((len(times),10),bool))
        at=start+150000
        signal=dict(time=at,source_time=start//1000,sigma=sigma.tolist(),weights=[.1]+[0.]*9,volume=[1.]*10,trades=[1.]*10)
        tape=[dict(time=start+offset,coin='pas:BTC',qty=1000.,vwap=100.,buy_price=100.,sell_price=100.) for offset in [180000,240000]]
        class Fixed:
            def propose(self,*args,**kwargs):return dict(target_qty=[1.]+[0.]*9)
        sc=dict(fee_bps=9.,impact_bps=10.,participation=.01,receipt_multiplier=1.)
        r=simulate(dict(start_ms=start,cutoff_ms=end),d,[signal],tape,[],p,sc,controller=Fixed())
        self.assertEqual(len(r['fills']),1)
        self.assertEqual(r['fills'][0]['time'],(start+240000)/1000)
        self.assertTrue(r['metrics']['timing_pass'])

if __name__=='__main__':unittest.main()
