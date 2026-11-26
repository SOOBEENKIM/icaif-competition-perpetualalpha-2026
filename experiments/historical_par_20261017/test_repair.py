import unittest
import numpy as np
from .test_replay import ReplayTests
from .repair_engine import run


class RepairTests(ReplayTests):
    def replay(self,arm='pairs_carry',scenario='base'):
        return run(self.d,self.s,self.cfg,self.p,arm,scenario,int(self.d['times'][0]),int(self.d['times'][-1]))

    def test_repair_reduces_only_then_holds_both_legs(self):
        # $500 per leg target is 5 units. ETH gets only 2.5, BTC gets 5.
        self.d['volume'][1,1]=250
        r=self.replay()
        e=next(x for x in r['events'] if x['event']=='hedge_repair_requested')
        ready=next(x for x in r['events'] if x['event']=='balanced_after_repair')
        self.assertGreater(ready['time'],e['time'])
        repair_fills=[f for f in r['fills'] if e['time']<f['time']<=ready['time']]
        self.assertTrue(repair_fills)
        self.assertTrue(all(f['reduce_only'] for f in repair_fills))
        self.assertGreater(r['metrics']['completed_pairs'],0)
        self.assertAlmostEqual(abs(ready['qty'][0]),abs(ready['qty'][1]))

    def test_repair_cannot_fill_without_liquidity(self):
        self.d['volume'][1,1]=250
        self.d['volume'][2:5,0]=0
        r=self.replay()
        self.assertTrue(any(e['event']=='hedge_repair_timeout' for e in r['events']))
        self.assertFalse(any(f['asset']==0 and self.d['times'][2]<=f['time']<=self.d['times'][4] for f in r['fills']))


if __name__=='__main__':unittest.main()
