import copy,json
from pathlib import Path
import unittest
import numpy as np
from .model import infer

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


class SectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs=json.loads((HERE/'protocol.json').read_text())['pairs']
        cls.spec=json.loads((ROOT/'experiments/rolling_feature_pairs_20261017/protocol.json').read_text())
        cls.p=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['trading_protocol']
        with np.load(ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz') as z:
            cls.reference={k:z[k].copy() for k in ['market_times','close','volume','trades']}
        cls.state,cls.records,cls.arrays=infer(cls.reference['market_times'],cls.reference,cls.p['symbols'],cls.pairs,cls.spec)

    def test_original_two_pair_forecasts_unchanged(self):
        with np.load(ROOT/'experiments/rolling_feature_pairs_20261017/artifacts/reference_direct_24_state.npz') as z:
            for key in ['mu','beta','vol']:
                np.testing.assert_allclose(self.state[key][:,:2],z[key],atol=1e-14,rtol=1e-12,equal_nan=True)
            for key in ['directions','signal_times']:
                np.testing.assert_array_equal(self.state[key][:,:2],z[key])
            np.testing.assert_array_equal(self.state['source_prices'],z['source_prices'])

    def test_training_is_pair_separate_daily_and_strictly_mature(self):
        self.assertEqual({r['pair_index'] for r in self.records},set(range(6)))
        for record in self.records:
            self.assertEqual(record['fit_time']%86400,0)
            self.assertLess(record['last_maturity'],record['fit_time'])
            self.assertGreaterEqual(record['rows'],200)
        for k,pair in enumerate(self.pairs):
            self.assertEqual([self.p['symbols'][j] for j in self.state['pairs'][k]],pair)

    def test_future_changes_cannot_change_prior_forecasts(self):
        cutoff=int(np.datetime64('2026-09-01T12:00:00','s').astype('int64'))
        changed=copy.deepcopy(self.reference);changed['close'][changed['market_times']>cutoff]*=1.2
        after,records,arrays=infer(changed['market_times'],changed,self.p['symbols'],self.pairs,self.spec)
        before=changed['market_times']<=cutoff
        np.testing.assert_allclose(after['mu'][before],self.state['mu'][before],atol=1e-14,rtol=1e-12,equal_nan=True)
        self.assertEqual([r for r in self.records if r['fit_time']<=cutoff],[r for r in records if r['fit_time']<=cutoff])

    def test_reordering_pair_list_preserves_symbol_identified_forecasts(self):
        reversed_pairs=list(reversed(self.pairs))
        state,_,_=infer(self.reference['market_times'],self.reference,self.p['symbols'],reversed_pairs,self.spec)
        np.testing.assert_allclose(state['mu'][:,::-1],self.state['mu'],atol=1e-14,rtol=1e-12,equal_nan=True)


if __name__=='__main__':unittest.main()
