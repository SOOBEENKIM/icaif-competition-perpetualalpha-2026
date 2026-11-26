import copy
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.optimize import minimize
from experiments.hourly_refresh_pairs_20261017.test_hourly import HourlyInferenceTests
from experiments.hourly_refresh_pairs_20261017.model import infer as old_infer
from .model import combine, infer, simplex_fit


class CausalExpertTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((Path(__file__).parent/'protocol.json').read_text())
        self.f = HourlyInferenceTests()
        self.f.setUp()

    def run_inference(self):
        f = self.f
        return infer(f.t, f.r, f.d, f.models, [['BTC', 'ETH']], f.spec, self.spec)

    def test_simplex_solution_matches_independent_constrained_optimizer(self):
        rng = np.random.default_rng(20261017)
        for prior in [0.1, 12.]:
            for target_kind in ['interior', 'boundary']:
                x = rng.normal(size=(80, 4))
                y = x@np.array([.2, .3, .1, .4]) if target_kind == 'interior' else x[:, 0]*8
                v = rng.uniform(.1, 2, len(x))
                objective = lambda w: float(v@(x@w-y)**2 + prior*np.sum((w-.25)**2))
                solved = minimize(objective, np.full(4, .25), method='SLSQP',
                                  bounds=[(0, 1)]*4, constraints={'type': 'eq', 'fun': lambda w: w.sum()-1},
                                  options={'ftol': 1e-10, 'maxiter': 300})
                got = simplex_fit(x, y, v, prior)
                self.assertTrue(solved.success, solved.message)
                self.assertLessEqual(abs(objective(got)-objective(solved.x)), 1e-6)
                self.assertTrue(np.all(got >= 0))
                self.assertAlmostEqual(got.sum(), 1.)

    def test_original_experts_remain_exact(self):
        _, controls, records, _ = self.run_inference()
        f = self.f
        old, _, _ = old_infer(f.t, f.r, f.d, f.models, [['BTC', 'ETH']], 24, f.spec)
        for kind in controls:
            for key in old[kind]:
                np.testing.assert_array_equal(controls[kind][key], old[kind][key])
        self.assertTrue(any(not r['fallback'] for r in records))
        for r in records:
            if r['last_maturity'] is not None:
                self.assertLess(r['last_maturity'], r['time'])

    def test_future_prices_and_features_do_not_change_earlier_weights(self):
        before, _, records, arrays = self.run_inference()
        cut = 1800
        self.f.r['close'][cut+1:] *= 5
        self.f.d['x'][cut+1:] *= -20
        self.f.d['fresh'][cut+1:] = False
        after, _, changed, aa = self.run_inference()
        for kind in before:
            for key in ['mu', 'signal_times', 'source_prices', 'directions', 'beta', 'vol']:
                np.testing.assert_array_equal(before[kind][key][:cut+1], after[kind][key][:cut+1])
            np.testing.assert_array_equal(arrays[f'weights_{kind}'][:cut+1], aa[f'weights_{kind}'][:cut+1])
        self.assertEqual([r for r in records if r['time'] <= self.f.t[cut]],
                         [r for r in changed if r['time'] <= self.f.t[cut]])

    def test_maturity_boundary_is_strict_and_weights_use_issued_predictions(self):
        t = np.arange(600)*3600
        x = np.tile([0., 1., 2., 3.], (len(t), 1, 1))
        y = np.zeros((len(t), 1))
        fresh = np.ones_like(y, bool)
        spec = {**self.spec, 'minimum_rows': 2}
        _, wa, _ = combine(t, x, y, t+86400, fresh, fresh, 24, spec)
        # Source 240 matures at 264: its changed error may affect hour 265,
        # never hour 264. No recomputation with the current predictor occurs.
        changed = x.copy(); changed[240, 0] = [4., 0., 0., 0.]
        _, wb, _ = combine(t, changed, y, t+86400, fresh, fresh, 24, spec)
        for k in ['imse', 'stack']:
            np.testing.assert_array_equal(wa[k][:265], wb[k][:265])
            self.assertFalse(np.array_equal(wa[k][265], wb[k][265]))
        self.assertGreater(wa['imse'][500, 0, 0], wa['imse'][500, 0, 3])

    def test_missing_terminal_or_expert_cannot_enter_weight_training(self):
        t = np.arange(500)*3600
        x = np.ones((len(t), 1, 4))
        y = np.zeros((len(t), 1))
        fresh = np.ones_like(y, bool); terminal = fresh.copy()
        terminal[0] = False; x[24, 0, 2] = np.nan
        forecasts, _, rows = combine(t, x, y, t+86400, fresh, terminal, 24, self.spec)
        self.assertTrue(np.isnan(forecasts['equal'][24, 0]))
        last = rows[-1]
        self.assertEqual(last['first_source'], 48*3600)

    def test_convex_bounds_and_uniform_fallback(self):
        _, _, records, a = self.run_inference()
        valid = np.isfinite(a['issued']).all(-1)
        for kind in self.spec['candidates']:
            w = a[f'weights_{kind}'][valid]
            np.testing.assert_allclose(w.sum(-1), 1, atol=1e-12)
            self.assertTrue(np.all(w >= 0))
            p = a[f'prediction_{kind}'][valid]; x = a['issued'][valid]
            self.assertTrue(np.all(p >= x.min(-1)-1e-12))
            self.assertTrue(np.all(p <= x.max(-1)+1e-12))
        for r in records:
            if r['fallback']:
                for weights in r['weights'].values():
                    self.assertEqual(weights, [.25]*4)

    def test_reject_wrong_horizon(self):
        t = np.arange(50)*3600; x = np.zeros((50, 1, 4)); y = np.zeros((50, 1))
        with self.assertRaises(ValueError):
            combine(t, x, y, t+72*3600, np.ones_like(y, bool), np.ones_like(y, bool), 24, self.spec)


if __name__ == '__main__':
    unittest.main()
