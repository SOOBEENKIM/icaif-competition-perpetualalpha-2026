from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
from experiments.paired_venue_20261017.engine import quote_pair
from .quote import signed_beta, hedge_diagnostic, signed_quote


class SignedTests(unittest.TestCase):
    def fixture(self):
        coins = ['par:USDEUR', 'par:USDJPY']
        cfg = json.loads(Path('experiments/paired_venue_20261017/protocol.json').read_text())
        metadata = {coins[0]: dict(szDecimals=1), coins[1]: dict(szDecimals=2)}
        snapshots = {}
        for c, px in zip(coins, [1.1, 150.]):
            snapshots[c] = dict(oracle=px, mid=px, ctx=dict(funding=.0001), book=dict(levels=[
                [dict(px=str(px * .9999), sz='10000000')], [dict(px=str(px * 1.0001), sz='10000000')]]))
        return coins, snapshots, metadata, cfg

    def test_positive_beta_reproduces_existing_quotes(self):
        coins, s, m, cfg = self.fixture()
        for direction in (-1, 1):
            for scenario in cfg['scenarios'].values():
                old = quote_pair(coins, s, m, .6, direction, 72, cfg, scenario)
                new = signed_quote(coins, s, m, .6, direction, 72, cfg, scenario)
                for k in old:
                    if k in new:
                        if isinstance(old[k], (float, int)): self.assertAlmostEqual(old[k], new[k], places=9)
                        else: self.assertEqual(old[k], new[k])

    def test_negative_beta_requires_same_direction_but_not_dollar_neutral(self):
        coins, s, m, cfg = self.fixture()
        for d in (-1, 1):
            q = signed_quote(coins, s, m, -.6, d, 72, cfg, cfg['scenarios']['base'])
            self.assertTrue(all(d * x > 0 for x in q['target_qty']))
            self.assertAlmostEqual(abs(q['signed_oracle_notional']), sum(abs(qty) * s[c]['oracle'] for c, qty in zip(coins, q['target_qty'])))

    def test_funding_cash_keeps_each_leg_sign_and_only_haircuts_positive_carry(self):
        coins, s, m, cfg = self.fixture()
        for d in (-1, 1):
            q = signed_quote(coins, s, m, -.6, d, 24, cfg, cfg['scenarios']['base'])
            raw = -sum(qty * s[c]['oracle'] * s[c]['ctx']['funding'] * 24 for c, qty in zip(coins, q['target_qty']))
            self.assertAlmostEqual(q['predicted_funding'], raw * (.5 if raw > 0 else 1))
            self.assertAlmostEqual(q['predicted_net_pnl'], q['predicted_price_pnl'] + q['predicted_funding'] - q['predicted_fees'])

    def test_both_entry_and_exit_capacity_limit_both_legs(self):
        coins, s, m, cfg = self.fixture()
        full = signed_quote(coins, s, m, -.6, -1, 72, cfg, cfg['scenarios']['base'])
        # A sell can enter at the bid, but buying back requires the small ask.
        s[coins[1]]['book']['levels'][1][0]['sz'] = '1'
        small = signed_quote(coins, s, m, -.6, -1, 72, cfg, cfg['scenarios']['base'])
        ratios = np.array(small['target_qty']) / np.array(full['target_qty'])
        self.assertLess(small['gross_notional'], full['gross_notional'])
        self.assertLess(abs(ratios[0] / ratios[1] - 1), .05)

    def test_negative_beta_reduces_risk_for_known_inverse_returns(self):
        x = np.random.default_rng(7).normal(0, .001, 168)
        r = np.column_stack((-.6 * x, x)); d = hedge_diagnostic(r)
        self.assertAlmostEqual(d['beta'], -.6)
        self.assertLess(d['correct_hourly_sigma'], 1e-10)
        self.assertGreater(d['forced_opposite_hourly_sigma'], .0001)
        self.assertFalse(d['dollar_neutral']); self.assertFalse(d['mean_reversion_proven'])

    def test_zero_variance_and_zero_beta_are_not_tradeable(self):
        with self.assertRaises(ValueError): signed_beta(np.zeros((168, 2)))
        c, s, m, cfg = self.fixture()
        self.assertIsNone(signed_quote(c, s, m, 0., 1, 72, cfg, cfg['scenarios']['base']))


if __name__ == '__main__': unittest.main()
