import unittest
import numpy as np
from .study import ROOT,read
from .candidate import snapshot


class CandidateTests(unittest.TestCase):
    def test_adapter_matches_frozen_backtest_and_keeps_all45_audits(self):
        out=ROOT/'reports/pair_model_research_20261023/wave9';p=read(out/'protocol.json');raw=dict(np.load(ROOT/p['input']));state=dict(np.load(out/'adaptive_hedge_72.npz'))
        t=raw['market_times']
        for i in [len(t)-25,len(t)-1]:
            d={k:raw[k][:i+1] for k in ['market_times','close','volume','trades']};r=snapshot(d,int(t[i]),int(t[i])+5)
            self.assertEqual(len(r['pair_audit']),45);self.assertEqual(r['orders_sent'],0);self.assertFalse(r['live_connected'])
            self.assertEqual(len(r['forecasts']),int(np.isfinite(state['mu'][i]).sum()))
            for f in r['forecasts']:
                k=f['stable_pair_index'];self.assertAlmostEqual(f['mu'],state['mu'][i,k],places=13)
                self.assertAlmostEqual(f['source_beta'],state['beta'][i,k],places=13)
                self.assertEqual(f['economic_action'],state['economic_action'][i,k]);self.assertEqual(f['horizon'],72)
            with self.assertRaises(ValueError):snapshot(d,int(t[i])-3600,int(t[i]))
            with self.assertRaises(ValueError):snapshot(d,int(t[i]),int(t[i])+3598)


if __name__=='__main__':unittest.main()
