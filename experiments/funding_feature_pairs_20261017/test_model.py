import copy, json, unittest
from pathlib import Path
import numpy as np
from experiments.dense_history_pairs_20261017.model import design as price_design, labels
from experiments.sector_pairs_20261017.model import infer as price_infer
from .model import design, infer

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


class FundingFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = json.loads((HERE/'protocol.json').read_text())['pairs']
        cls.spec = json.loads((ROOT/'experiments/rolling_feature_pairs_20261017/protocol.json').read_text())
        cls.symbols = json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['trading_protocol']['symbols']
        with np.load(ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz') as z:
            cls.reference = {k:z[k].copy() for k in ['market_times','close','volume','trades','funding','observed']}
        cls.features = design(cls.reference, cls.symbols, cls.pairs)
        cls.state, cls.records, cls.arrays = infer(cls.reference['market_times'], cls.reference, cls.symbols, cls.pairs, cls.spec)

    def test_price_features_and_labels_unchanged(self):
        r = self.reference
        price = price_design(r['market_times'],r['close'],r['volume'],r['trades'],self.symbols,self.pairs)
        np.testing.assert_array_equal(self.features['x'][:,:,:-2],price['x'])
        for key in ['beta','vol','fresh','pairs']:
            np.testing.assert_array_equal(self.features[key],price[key])
        old,_ = labels(r['market_times'],r['close'],price,24)
        np.testing.assert_array_equal(self.arrays['target'],old)

    def test_price_only_control_reproduces_retained(self):
        state,_,_ = price_infer(self.reference['market_times'],self.reference,self.symbols,self.pairs,self.spec)
        with np.load(ROOT/'experiments/rolling_feature_pairs_20261017/artifacts/reference_direct_24_state.npz') as z:
            for key in ['mu','beta','vol']:
                np.testing.assert_allclose(state[key],z[key],atol=1e-14,rtol=1e-12,equal_nan=True)
            np.testing.assert_array_equal(state['directions'],z['directions'])

    def test_exact_hourly_units_and_source_rate_exclusion(self):
        i = 4000
        for k,(a,b) in enumerate(self.features['pairs']):
            beta,vol = self.features['beta'][i,k],self.features['vol'][i,k]
            for col,h in enumerate((24,168)):
                rates = self.reference['funding'][i-h:i].mean(axis=0)
                expected = np.clip((rates[a]-beta*rates[b])/(1+beta)/vol,-8,8)
                self.assertAlmostEqual(self.features['x'][i,k,-2+col],expected,places=12)
        changed = copy.deepcopy(self.reference)
        changed['funding'][i:] = 3.0
        new = design(changed,self.symbols,self.pairs)
        np.testing.assert_allclose(new['x'][:i+1],self.features['x'][:i+1],atol=1e-14,equal_nan=True)

    def test_unobserved_is_missing_observed_zero_is_valid(self):
        i = 4000
        missing = copy.deepcopy(self.reference)
        missing['funding'][i,0] = 0.
        missing['observed'][i,0] = False
        d = design(missing,self.symbols,self.pairs)
        np.testing.assert_array_equal(d['x'][:i+1],self.features['x'][:i+1])
        for col,h in enumerate((24,168)):
            self.assertTrue(np.isnan(d['x'][i+1:i+h+1,:,-2+col]).all())
            self.assertTrue(np.isfinite(d['x'][i+h+1,:,-2+col]).all())
        missing['observed'][i,0] = True
        zero = design(missing,self.symbols,self.pairs)
        self.assertTrue(np.isfinite(zero['x'][i+1:i+169]).all())

    def test_future_changes_preserve_forecasts_and_fit_scaling(self):
        cutoff = int(np.datetime64('2026-09-01T12:00:00','s').astype('int64'))
        changed = copy.deepcopy(self.reference)
        after = changed['market_times'] > cutoff
        changed['close'][after] *= 1.2
        changed['funding'][changed['market_times'] >= cutoff] = 3.
        state,records,_ = infer(changed['market_times'],changed,self.symbols,self.pairs,self.spec)
        np.testing.assert_allclose(state['mu'][~after],self.state['mu'][~after],atol=1e-14,rtol=1e-12,equal_nan=True)
        self.assertEqual([r for r in self.records if r['fit_time']<=cutoff],
                         [r for r in records if r['fit_time']<=cutoff])

    def test_daily_fit_only_strictly_mature_training_rows(self):
        self.assertEqual({r['pair_index'] for r in self.records},{0,1})
        for record in self.records:
            self.assertEqual(record['fit_time']%86400,0)
            self.assertLess(record['last_maturity'],record['fit_time'])
            self.assertGreaterEqual(record['rows'],200)
            self.assertEqual(record['names'][-2:],['funding_24','funding_168'])
        # Independently reconstruct one fit's weighted feature center.
        r=self.records[-1];t=self.reference['market_times'];k=r['pair_index']
        mask=(t>=r['fit_time']-90*86400)&(self.arrays['maturity']<r['fit_time'])&(t%(6*3600)==0)
        mask &= np.isfinite(self.features['x'][:,k]).all(1)&self.features['fresh'][:,k]
        mask &= self.arrays['terminal_fresh'][:,k]&np.isfinite(self.arrays['target'][:,k])
        w=np.exp2(-(r['fit_time']-self.arrays['maturity'][mask])/(30*86400));w/=w.sum()
        np.testing.assert_allclose(r['center'],w@self.features['x'][mask,k],atol=1e-14)


if __name__=='__main__':unittest.main()
