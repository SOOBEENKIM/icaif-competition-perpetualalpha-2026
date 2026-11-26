import unittest
import numpy as np
from pair_value_learning.study import ROOT,INPUT
from .candidate import snapshot,research_contract


class FinalCandidateTests(unittest.TestCase):
    def test_candidate_reproduces_historical_forecasts_and_explicit_pair_only_contract(self):
        d=dict(np.load(INPUT));s=dict(np.load(ROOT/'reports/pair_economic_learning_20261023/wave3/unbounded_pairs_72.npz'))
        for source in [1785585600,1788264000,1792065600,1792411200,1792713600]:
            n=int(np.searchsorted(d['market_times'],source))+1
            prefix={k:(v[:n] if v.ndim and len(v)==len(d['market_times']) else v) for k,v in d.items()}
            r=snapshot(prefix,source,source+10);i=int(np.searchsorted(s['times'],source))
            self.assertEqual(len(r['pair_audit']),45)
            self.assertFalse(r['research_contract']['include_directional_core'])
            self.assertFalse(r['live_connected']);self.assertEqual(r['orders_sent'],0)
            self.assertEqual({f['stable_pair_index'] for f in r['forecasts']},set(np.flatnonzero(s['economic_available'][i])))
            for f in r['forecasts']:
                k=f['stable_pair_index'];self.assertAlmostEqual(f['mu'],s['mu'][i,k],places=12)
                self.assertAlmostEqual(f['source_beta'],s['beta'][i,k],places=12)
                self.assertEqual(f['economic_action'],s['economic_action'][i,k])

    def test_contract_preserves_funded_sleeve_and_caps(self):
        c=research_contract()
        self.assertEqual(c['trading']['initial'],50000.)
        self.assertEqual(c['config']['pair_max_gross'],6000.)
        self.assertEqual(len(c['config']['dense_pairs']),45)
        self.assertTrue(c['arm'].startswith('pairs_'))


if __name__=='__main__':unittest.main()
