import copy
import unittest
import numpy as np
from competition_live.profile import load_profile
from .pas import simulate


class RecheckReplayTests(unittest.TestCase):
    def case(self, n=20):
        start = 1792411200000
        width = 10
        times = start + np.arange(n) * 60000
        prices = np.full((n, width), 100.)
        d = dict(times=times, close=prices.copy(), high=prices.copy(), low=prices.copy(),
                 active=np.ones_like(prices, bool), volume=np.full_like(prices, 100000.),
                 age_ms=np.zeros_like(prices), funding_events=[])
        cfg = dict(start_ms=start, cutoff_ms=int(times[-1]), symbols=load_profile()['trading']['symbols'])
        event = dict(time=start, source_time=start // 1000, weights=[.12]+[0.]*9,
                     sigma=(np.eye(width)*1e-8).tolist(), volume=[100000.]*width, trades=[100.]*width)
        sc = dict(fee_bps=9., impact_bps=10., participation=.01, receipt_multiplier=1.)
        return cfg, d, [event], copy.deepcopy(load_profile()['trading']), sc

    def test_no_artificial_two_thousand_cap(self):
        cfg, d, targets, p, sc = self.case()
        result = simulate(cfg, d, targets, p, sc)
        self.assertGreater(result['metrics']['maximum_notional'], 5900.)
        self.assertLess(result['metrics']['maximum_notional'], 6100.)
        self.assertEqual(result['metrics']['risk_breach_bars'], 0)

    def test_live_asset_and_turnover_limits_are_applied(self):
        cfg, d, targets, p, sc = self.case()
        targets[0]['weights'] = [1.] * 10
        result = simulate(cfg, d, targets, p, sc)
        decision = result['decisions'][0]
        self.assertLessEqual(decision['target_gross'], 50000 * .15 + 1e-6)
        self.assertTrue(all(abs(q) * 100 <= 7500 for q in decision['target_qty']))
        self.assertEqual(result['metrics']['risk_breach_bars'], 0)

    def test_no_future_fill_or_unobserved_volume(self):
        cfg, d, targets, p, sc = self.case()
        a = simulate(cfg, d, targets, p, sc)
        changed = copy.deepcopy(d)
        for key in ('close', 'high', 'low'):
            changed[key][10:] *= 1.5
        b = simulate(cfg, changed, targets, p, sc)
        cutoff = d['times'][10]
        self.assertEqual([f for f in a['fills'] if f['time'] < cutoff],
                         [f for f in b['fills'] if f['time'] < cutoff])
        self.assertTrue(all(f['time'] > f['decision_time'] for f in a['fills']))
        d['active'][:] = False
        self.assertEqual(simulate(cfg, d, targets, p, sc)['fills'], [])

    def test_unsupported_daily_score_is_not_zero(self):
        cfg, d, targets, p, sc = self.case()
        m = simulate(cfg, d, targets, p, sc)['metrics']
        self.assertIsNone(m['volatility'])
        self.assertIsNone(m['Q'])

    def test_missing_unheld_market_does_not_block_known_markets_or_use_future_price(self):
        cfg, d, targets, p, sc = self.case()
        for key in ('close', 'high', 'low'):
            d[key][:10, 1] = np.nan
        d['active'][:10, 1] = False
        targets[0]['weights'][1] = .05
        result = simulate(cfg, d, targets, p, sc)
        self.assertTrue(any(f['asset'] == 0 for f in result['fills']))
        self.assertFalse(any(f['asset'] == 1 for f in result['fills']))
        self.assertIn(1, result['decisions'][0]['unpriced_assets'])

    def test_close_before_short_on_next_six_hour_decision(self):
        cfg, d, targets, p, sc = self.case(n=380)
        short = copy.deepcopy(targets[0])
        short.update(time=short['time']+21600000, source_time=short['source_time']+21600,
                     weights=[-.12]+[0.]*9)
        targets.append(short)
        result = simulate(cfg, d, targets, p, sc)
        first_short = next(f for f in result['fills'] if f['signed_qty'] < 0 and not f['reduce_only'])
        previous = [f for f in result['fills'] if f['time'] < first_short['time']][-1]
        self.assertEqual(previous['position_after'], 0.)
        self.assertLess(result['metrics']['maximum_accounting_error'], 1e-6)


if __name__ == '__main__':
    unittest.main()
