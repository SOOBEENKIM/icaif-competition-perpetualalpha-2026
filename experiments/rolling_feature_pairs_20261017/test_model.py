import copy
import json
from pathlib import Path
import unittest
import numpy as np
from experiments.hourly_refresh_pairs_20261017.test_hourly import HourlyInferenceTests
from experiments.hourly_refresh_pairs_20261017.model import infer as old_infer
from .model import fit, rolling, infer


class RollingFeatureTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((Path(__file__).parent/'protocol.json').read_text())

    def fixture(self, n=2200, pairs=1):
        rng = np.random.default_rng(23)
        times = np.arange(n)*3600
        x = rng.normal(size=(n, pairs, 2))
        raw = .1*x[:, :, 0]
        target = .5*x[:, :, 0]-.3*x[:, :, 1]+.1
        d = dict(x=x, fresh=np.ones((n, pairs), bool), vol=np.full((n, pairs), .002), names=['a', 'b'])
        return times, d, target, times+86400, raw, np.ones((n, pairs), bool)

    def test_fit_matches_augmented_least_squares_and_uses_train_scaler(self):
        t, d, target, maturity, raw, _ = self.fixture(300)
        now = t[-1]+172800
        model = fit(d['x'][:, 0], target[:, 0], raw[:, 0], maturity, now, self.spec)
        decay = np.exp2(-(now-maturity)/(30*86400)); w = decay/decay.sum()
        center = w@d['x'][:, 0]
        scale = np.sqrt(w@((d['x'][:, 0]-center)**2))
        np.testing.assert_allclose(model['center'], center, atol=1e-14)
        np.testing.assert_allclose(model['scale'], scale, atol=1e-14)
        z = np.c_[np.ones(len(t)), (d['x'][:, 0]-center)/scale]
        penalty = np.diag([12/len(t), 1, 1])
        a = np.r_[np.sqrt(w[:, None])*z, np.sqrt(penalty)]
        for kind, y in [('direct', target[:, 0]), ('residual', target[:, 0]-raw[:, 0])]:
            b = np.r_[np.sqrt(w)*y, np.zeros(3)]
            expected = np.linalg.lstsq(a, b, rcond=None)[0]
            np.testing.assert_allclose(model[kind], expected, atol=1e-12)

    def test_unmatured_label_is_rejected_and_insufficient_fit_stays_missing(self):
        t, d, y, maturity, raw, terminal = self.fixture(200)
        with self.assertRaises(ValueError):
            fit(d['x'][:, 0], y[:, 0], raw[:, 0], maturity, maturity[-1], self.spec)
        predictions, stamps, records = rolling(t, d, y, maturity, raw, terminal, self.spec)
        self.assertEqual(records, [])
        self.assertTrue(np.all(stamps == -1))
        self.assertTrue(all(np.isnan(v).all() for v in predictions.values()))

    def test_fits_are_daily_sources_are_hourly_and_use_strictly_mature_labels(self):
        t, d, y, maturity, raw, terminal = self.fixture()
        forecasts, stamps, records = rolling(t, d, y, maturity, raw, terminal, self.spec)
        self.assertTrue(records)
        for r in records:
            self.assertEqual(r['fit_time'] % 86400, 0)
            self.assertLess(r['last_maturity'], r['fit_time'])
            self.assertGreaterEqual(r['first_source'], r['fit_time']-90*86400)
        i = 1801
        self.assertEqual(stamps[i, 0], t[i]//86400*86400)
        self.assertNotEqual(forecasts['residual'][i, 0], forecasts['residual'][i-1, 0])

    def test_future_data_and_labels_do_not_change_existing_fits(self):
        t, d, y, maturity, raw, terminal = self.fixture()
        cut = 1800
        before, stamps, records = rolling(t, d, y, maturity, raw, terminal, self.spec)
        changed_d = copy.deepcopy(d); changed_d['x'][cut+1:] *= 10
        changed_y = y.copy(); changed_y[maturity > t[cut]] = -100
        changed_raw = raw.copy(); changed_raw[cut+1:] *= 100
        after, ss, rr = rolling(t, changed_d, changed_y, maturity, changed_raw, terminal, self.spec)
        for kind in before:
            np.testing.assert_array_equal(before[kind][:cut+1], after[kind][:cut+1])
        np.testing.assert_array_equal(stamps[:cut+1], ss[:cut+1])
        self.assertEqual([r for r in records if r['fit_time'] <= t[cut]], [r for r in rr if r['fit_time'] <= t[cut]])

    def test_pairs_are_fitted_independently(self):
        t, d, y, maturity, raw, terminal = self.fixture(pairs=2)
        before, _, _ = rolling(t, d, y, maturity, raw, terminal, self.spec)
        y[:, 1] *= -10; d['x'][:, 1] *= 10
        after, _, _ = rolling(t, d, y, maturity, raw, terminal, self.spec)
        for kind in before:
            np.testing.assert_array_equal(before[kind][:, 0], after[kind][:, 0])

    def test_scalar_ablation_has_no_feature_slopes(self):
        t, d, y, maturity, raw, terminal = self.fixture()
        before, _, records = rolling(t, d, y, maturity, raw, terminal, self.spec)
        i = 1801; model = next(r for r in records if r['fit_time'] == t[i]//86400*86400)
        expected = np.clip(raw[i, 0]+model['scalar'], -4, 4)*d['vol'][i, 0]*np.sqrt(24)
        self.assertEqual(before['scalar'][i, 0], expected)

    def test_terminal_freshness_excludes_training_row(self):
        t, d, y, maturity, raw, terminal = self.fixture()
        terminal[1200] = False
        before, _, records = rolling(t, d, y, maturity, raw, terminal, self.spec)
        y[1200] = 1000
        after, _, rr = rolling(t, d, y, maturity, raw, terminal, self.spec)
        for kind in before:
            np.testing.assert_array_equal(before[kind], after[kind])
        self.assertEqual(records, rr)

    def test_original_controls_exact_and_current_source_not_fit_time(self):
        f = HourlyInferenceTests(); f.setUp()
        states, controls, _, _ = infer(f.t, f.r, f.d, f.models, [['BTC', 'ETH']], f.spec, self.spec)
        old, _, _ = old_infer(f.t, f.r, f.d, f.models, [['BTC', 'ETH']], 24, f.spec)
        for kind in controls:
            for key in controls[kind]:
                np.testing.assert_array_equal(controls[kind][key], old[kind][key])
        for state in states.values():
            self.assertEqual(state['signal_times'][1801, 0], f.t[1801])
            np.testing.assert_array_equal(state['source_prices'][1801], f.r['close'][1801])
            self.assertTrue(np.all(state['directions'][~np.isfinite(state['mu'])] == 0))


if __name__ == '__main__':
    unittest.main()
