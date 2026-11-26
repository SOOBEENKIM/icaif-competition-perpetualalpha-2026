import unittest
import numpy as np
from .candidate import snapshot
from .study import ROOT,INPUT


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=dict(np.load(INPUT))
        cls.state=dict(np.load(ROOT/'reports/pair_value_learning_20261023/scaled_coherence/coherent_scaled_72.npz'))

    def prefix(self,t):
        n=int(np.searchsorted(self.data['market_times'],t))+1
        return {k:(v[:n] if v.ndim and len(v)==len(self.data['market_times']) else v) for k,v in self.data.items()}

    def test_actual_prefix_adapter_matches_frozen_backtest(self):
        for t in [1785585600,1788264000,1792065600,1792411200,1792713600]:
            out=snapshot(self.prefix(t),t,t+10);i=int(np.searchsorted(self.state['times'],t))
            self.assertEqual(len(out['pair_audit']),45);self.assertEqual(out['orders_sent'],0)
            expected=set(np.flatnonzero(self.state['economic_available'][i]))
            self.assertEqual({f['stable_pair_index'] for f in out['forecasts']},expected)
            for f in out['forecasts']:
                k=f['stable_pair_index']
                self.assertAlmostEqual(f['mu'],self.state['mu'][i,k],places=12)
                self.assertEqual(f['economic_action'],self.state['economic_action'][i,k])
                self.assertAlmostEqual(f['source_beta'],self.state['beta'][i,k],places=12)

    def test_future_data_or_stale_receipt_rejected(self):
        t=1792411200
        for data,receipt in [(self.data,t+10),(self.prefix(t),t-1),(self.prefix(t),t+3600)]:
            with self.assertRaises(ValueError):snapshot(data,t,receipt)


if __name__=='__main__':unittest.main()
