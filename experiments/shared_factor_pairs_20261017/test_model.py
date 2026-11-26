import copy
import json
import unittest
from pathlib import Path
import numpy as np
from .model import prepare, fit, rolling
from experiments.dense_history_pairs_20261017.model import design


class SharedFactorTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((Path(__file__).parent / 'protocol.json').read_text())

    def source(self):
        rng = np.random.default_rng(171017)
        returns = rng.normal(0, .003, (1000, 3))
        returns[:, 1] += .7 * returns[:, 0]
        close = 100 * np.exp(np.cumsum(returns, axis=0))
        ref = dict(market_times=np.arange(len(close)) * 3600, close=close,
                   volume=np.ones_like(close), trades=np.ones_like(close))
        f = design(ref['market_times'], close, ref['volume'], ref['trades'], ['BTC', 'ETH', 'HYPE'], [['BTC', 'ETH'], ['BTC', 'HYPE']])
        return ref, f

    def fixture(self, n=1500):
        rng = np.random.default_rng(171018)
        t = np.arange(n) * 3600
        x = rng.normal(size=(n, 2, 3))
        mx = rng.normal(size=(n, 3))
        market = .2 * mx[:, 0] + .1 * mx[:, 1]
        unit = np.full(n, .02)
        pu = np.full((n, 2), .01)
        gamma = .2 + .05 * x[..., 2]
        target = .2 * x[..., 0] - .1 * x[..., 1]
        f = dict(x=x, fresh=np.ones((n, 2), bool), names=['a', 'b', 'c'])
        factor = dict(x=mx, names=['a', 'b', 'c'], unit=unit, pair_unit=pu, gamma=gamma,
            maturity=t + 86400, fresh=np.ones(n, bool), market_target=market, residual_target=target,
            terminal_market=np.ones(n, bool), terminal_pair=np.ones((n, 2), bool))
        return t, f, factor, np.zeros((n, 2))

    def test_target_identity_uses_fixed_source_hedge_and_loading(self):
        ref, f = self.source(); a = prepare(ref, f)
        reconstructed = a['gamma'] * a['market_return'][:, None] + a['residual_target'] * a['pair_unit']
        mask = np.isfinite(reconstructed)
        np.testing.assert_allclose(reconstructed[mask], a['actual'][mask], atol=1e-15)
        self.assertTrue(mask.any())
        np.testing.assert_allclose(a['source_beta_to_btc'][168:, 0], 1, atol=1e-13)
        self.assertTrue(np.isnan(a['residual_target'][-24:]).all())

    def test_market_features_independent_of_other_assets(self):
        ref, f = self.source(); a = prepare(ref, f)
        ref['close'][:, 1:] *= np.exp(np.linspace(0, .5, len(ref['close'])))[:, None]
        b = prepare(ref, f)
        for key in ['x', 'unit', 'market_target', 'market_return']:
            np.testing.assert_array_equal(a[key], b[key])

    def test_source_loadings_and_features_do_not_use_future_prices(self):
        ref, f = self.source(); a = prepare(ref, f); cut = 850
        ref['close'][cut+1:] *= [1.1, 1.4, 1.8]
        b = prepare(ref, f)
        for key in ['x', 'unit', 'gamma', 'source_beta_to_btc']:
            np.testing.assert_array_equal(a[key][:cut+1], b[key][:cut+1])

    def test_weighted_ridge_matches_augmented_least_squares(self):
        t, f, a, _ = self.fixture(300); x = a['x']; y = a['market_target']; now = a['maturity'][-1] + 1
        m = fit(x, y, a['maturity'], now, self.spec)
        w = np.exp2(-(now - a['maturity']) / (30 * 86400)); w /= w.sum()
        z = np.c_[np.ones(len(t)), (x-m['center'])/m['scale']]
        p = np.diag([12/len(t), 1, 1, 1])
        coef = np.linalg.lstsq(np.r_[np.sqrt(w[:,None])*z, np.sqrt(p)], np.r_[np.sqrt(w)*y, np.zeros(4)], rcond=None)[0]
        np.testing.assert_allclose(m['coefficient'], coef, atol=1e-13)
        with self.assertRaises(ValueError): fit(x, y, a['maturity'], a['maturity'][-1], self.spec)

    def test_single_shared_model_per_day_and_component_sum(self):
        t, f, a, raw = self.fixture(); p, btc, c, st, records = rolling(t, f, a, raw, self.spec)
        days = [r['fit_time'] for r in records if r['role'] == 'market']
        self.assertTrue(days); self.assertEqual(len(days), len(set(days)))
        expected = np.clip(c['market'] + c['residual'], -4*a['pair_unit'], 4*a['pair_unit'])
        np.testing.assert_array_equal(p['combined'], expected)
        self.assertTrue(np.isfinite(p['combined'][1248]).all())
        self.assertTrue(all(r['last_maturity'] < r['fit_time'] for r in records))

    def test_pair_changes_do_not_retrain_shared_market_or_other_pair(self):
        t, f, a, raw = self.fixture(); p, btc, _, _, _ = rolling(t, f, a, raw, self.spec)
        f['x'][:,1] *= 3; a['residual_target'][:,1] *= -2; a['gamma'][:,1] *= -1
        q, other_btc, _, _, _ = rolling(t, f, a, raw, self.spec)
        np.testing.assert_array_equal(btc, other_btc)
        np.testing.assert_array_equal(p['combined'][:,0], q['combined'][:,0])

    def test_future_labels_and_activity_do_not_change_past_forecasts(self):
        t, f, a, raw = self.fixture(); cut = 1350
        p, btc, c, st, records = rolling(t, f, a, raw, self.spec)
        f['x'][cut+1:] *= 10; a['x'][cut+1:] *= 5; a['gamma'][cut+1:] *= 10
        immature = a['maturity'] > t[cut]
        a['market_target'][immature] = 100; a['residual_target'][immature] = 100
        a['terminal_market'][immature] = False; a['terminal_pair'][immature] = False
        q, other_btc, _, ss, rr = rolling(t, f, a, raw, self.spec)
        for key in p: np.testing.assert_array_equal(p[key][:cut+1], q[key][:cut+1])
        np.testing.assert_array_equal(btc[:cut+1], other_btc[:cut+1])
        np.testing.assert_array_equal(st[:cut+1], ss[:cut+1])
        self.assertEqual([r for r in records if r['fit_time'] <= t[cut]], [r for r in rr if r['fit_time'] <= t[cut]])

    def test_inactive_terminal_excluded_from_training_not_inference(self):
        t, f, a, raw = self.fixture(); a['terminal_market'][1248] = False; a['terminal_pair'][1248] = False
        p, _, _, _, rec = rolling(t, f, a, raw, self.spec)
        a['market_target'][1248] = 100; a['residual_target'][1248] = 100
        q, _, _, _, rr = rolling(t, f, a, raw, self.spec)
        for key in p: np.testing.assert_array_equal(p[key], q[key])
        self.assertEqual(rec, rr); self.assertTrue(np.isfinite(p['combined'][1248]).all())

    def test_insufficient_or_invalid_inputs(self):
        t, f, a, raw = self.fixture(200); p, _, _, st, rec = rolling(t, f, a, raw, self.spec)
        self.assertTrue(np.isnan(p['combined']).all()); self.assertFalse(rec); self.assertTrue((st == -1).all())
        a['maturity'] += 1
        with self.assertRaises(ValueError): rolling(t, f, a, raw, self.spec)


if __name__ == '__main__': unittest.main()
