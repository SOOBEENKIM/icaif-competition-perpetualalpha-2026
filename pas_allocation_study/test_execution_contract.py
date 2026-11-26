"""Exercise candidate targets through the actual shared account/controller seam."""
import copy
import unittest
import numpy as np
from competition_live.profile import load_profile
from strategy_recheck.pas import simulate


class ExecutionContracts(unittest.TestCase):
    def test_all_ten_short_close_and_long_with_paid_fills(self):
        p=load_profile()['trading'];start=1792411200000
        times=np.arange(start-3600000,start+13*3600000+1,60000,dtype=np.int64)
        # No attempt to infer performance; deterministic fill mechanics fixture.
        close=np.full((len(times),10),100.)
        d=dict(times=times,close=close,open=close,high=close,low=close,
            volume=np.full_like(close,10000.),active=np.ones_like(close,bool),
            age_ms=np.zeros_like(close),funding_events=[])
        cfg=dict(start_ms=start,cutoff_ms=int(times[-1]),symbols=p['symbols'])
        events=[]
        for h,target in [(0,-.01),(6,0.),(12,.01)]:
            events.append(dict(time=start+h*3600000+120000,source_time=start//1000+h*3600,
                weights=np.full(10,target).tolist(),sigma=(np.eye(10)*.0005**2).tolist(),
                volume=np.full(10,10000.).tolist(),trades=np.ones(10).tolist()))
        sc=dict(fee_bps=9.,impact_bps=10.,participation=.01,receipt_multiplier=1.)
        r=simulate(cfg,d,events,p,sc)
        at={s['time']:s for s in r['snapshots']}
        self.assertTrue(np.all(np.array(at[start+5*60000]['qty'])<0))
        np.testing.assert_array_equal(at[start+6*3600000+5*60000]['qty'],np.zeros(10))
        self.assertTrue(np.all(np.array(at[start+12*3600000+5*60000]['qty'])>0))
        self.assertEqual(set(r['metrics']['traded_symbols']),set(p['symbols']))
        self.assertGreaterEqual(r['metrics']['short_opening_fills'],10)
        self.assertGreaterEqual(r['metrics']['reduction_fills'],10)
        self.assertGreater(r['metrics']['fees'],0)
        self.assertLess(r['metrics']['net_pnl'],0)  # Constant mid; paid spread/fees.
        self.assertLess(r['metrics']['maximum_accounting_error'],1e-6)
        self.assertTrue(r['metrics']['timing_pass'])
        # Unavailable later liquidity cannot retroactively erase an earlier fill.
        changed=copy.deepcopy(d);changed['active'][times>=start+7*3600000]=False
        changed['volume'][times>=start+7*3600000]=0
        other=simulate(cfg,changed,events,p,sc)
        cutoff=start+7*3600000
        self.assertEqual([f for f in r['fills'] if f['time']<cutoff],
                         [f for f in other['fills'] if f['time']<cutoff])
        self.assertEqual(other['metrics']['final_notional'],0.)


if __name__=='__main__':unittest.main()
