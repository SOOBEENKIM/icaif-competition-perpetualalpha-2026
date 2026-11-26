import copy
import json
from pathlib import Path
import unittest
import numpy as np
from experiments.hourly_refresh_pairs_20261017.test_hourly import HourlyInferenceTests
from experiments.hourly_refresh_pairs_20261017.model import infer as old_infer
from .model import proxy, choose, infer


class CostSelectionTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((Path(__file__).parent/'protocol.json').read_text())

    def fixture(self):
        n = 100
        r = dict(market_times=np.arange(n)*3600, close=np.full((n, 2), 100.),
                 funding=np.full((n, 2), .0001), observed=np.ones((n, 2), bool))
        d = dict(pairs=[[0, 1]], beta=np.ones((n, 1)), fresh=np.ones((n, 1), bool))
        f = np.full((n, 1, 4), .02)
        return r, d, f

    def test_proxy_accounts_for_next_bar_two_leg_costs_and_settlements(self):
        r, d, f = self.fixture(); i = 2; end = i+24
        r['close'][i+1] = [101, 99]; r['close'][end] = [104, 98]
        result = proxy(r, d, f, self.spec)
        self.assertEqual(result['actions'][i, 0, 0], 1)
        q = np.array([.005, -.005]); impact = .001; fee = .0009
        entry = r['close'][i+1]*(1+np.sign(q)*impact)
        exit_price = r['close'][end]*(1-np.sign(q)*impact)
        cash = -q@entry-fee*(abs(q)@entry)
        for j in range(i+2, end+1):
            cash -= q@(r['close'][j]*r['funding'][j])
        cash += q@exit_price-fee*(abs(q)@exit_price)
        self.assertAlmostEqual(result['payoff'][i, 0, 0], cash, places=12)
        # Entry settlement is excluded because positions arrive after it.
        changed = copy.deepcopy(r); changed['funding'][i+1] = [10, 0]
        after = proxy(changed, d, f, self.spec)
        self.assertEqual(result['payoff'][i, 0, 0], after['payoff'][i, 0, 0])

    def test_price_gap_changes_payoff_not_source_decision(self):
        r, d, f = self.fixture(); i = 2
        before = proxy(r, d, f, self.spec)
        r['close'][i+1] = [110, 90]
        after = proxy(r, d, f, self.spec)
        self.assertEqual(before['actions'][i, 0, 0], after['actions'][i, 0, 0])
        self.assertEqual(before['expected'][i, 0, 0], after['expected'][i, 0, 0])
        self.assertLess(after['payoff'][i, 0, 0], before['payoff'][i, 0, 0])

    def test_missing_funding_or_terminal_does_not_become_zero_profit(self):
        r, d, f = self.fixture(); i = 2
        r['observed'][i+12, 0] = False
        result = proxy(r, d, f, self.spec)
        self.assertTrue(np.isnan(result['payoff'][i]).all())
        r['observed'][:] = True; d['fresh'][i+24] = False
        result = proxy(r, d, f, self.spec)
        self.assertTrue(np.isnan(result['payoff'][i]).all())

    def test_source_flat_has_zero_label_when_full_window_observed(self):
        r, d, f = self.fixture(); f[:] = 0; r['funding'][:] = 0
        result = proxy(r, d, f, self.spec)
        self.assertEqual(result['actions'][2, 0, 0], 0)
        self.assertEqual(result['payoff'][2, 0, 0], 0)
        self.assertTrue(np.isnan(result['payoff'][-1]).all())

    def test_mature_only_selector_preserves_exact_expert_value(self):
        n = 500; t = np.arange(n)*3600
        f = np.tile([.02, .03, -.02, .01], (n, 1, 1))
        actions = np.ones_like(f, dtype=int); payoff = np.tile([.001, .003, -.01, 0.], (n, 1, 1))
        ledger = dict(maturity=t+86400, actions=actions, preferred=actions.copy(), payoff=payoff,
                      expected=np.ones_like(f))
        fresh = np.ones((n, 1), bool)
        a, chosen, records = choose(t, f, ledger, fresh, self.spec)
        self.assertEqual(chosen['select'][-1, 0], 1)
        self.assertEqual(a['select'][-1, 0], f[-1, 0, 1])
        changed = copy.deepcopy(ledger); changed['payoff'][360, 0] = [100, -100, 0, 0]
        b, cc, _ = choose(t, f, changed, fresh, self.spec)
        for kind in self.spec['candidates']:
            np.testing.assert_array_equal(a[kind][:385], b[kind][:385])
        self.assertNotEqual(chosen['select'][385, 0], cc['select'][385, 0])
        for record in records:
            if record['last_maturity'] is not None:
                self.assertLess(record['last_maturity'], record['time'])

    def test_nonpositive_score_suppresses_source_instead_of_zero_mu(self):
        n = 500; t = np.arange(n)*3600; f = np.full((n, 1, 4), .02)
        ledger = dict(maturity=t+86400, actions=np.ones_like(f, int), preferred=np.ones_like(f, int),
                      payoff=np.full_like(f, -.001), expected=np.ones_like(f))
        output, chosen, _ = choose(t, f, ledger, np.ones((n, 1), bool), self.spec)
        for kind in output:
            self.assertTrue(np.isnan(output[kind]).all())
            self.assertTrue(np.all(chosen[kind] == -1))

    def test_directional_score_uses_matching_historical_actions(self):
        n = 900; t = np.arange(n)*3600; f = np.full((n, 1, 4), .02)
        actions = np.ones_like(f, int)
        actions[(np.arange(n)//24) % 2 == 0, 0, 0] = -1
        payoff = np.full_like(f, -.01)
        payoff[:, 0, 0] = np.where(actions[:, 0, 0] == 1, -.01, .1)
        ledger = dict(maturity=t+86400, actions=actions, preferred=np.ones_like(f, int), payoff=payoff,
                      expected=np.ones_like(f))
        _, chosen, _ = choose(t, f, ledger, np.ones((n, 1), bool), self.spec)
        self.assertEqual(chosen['select'][-1, 0], 0)
        self.assertEqual(chosen['directional'][-1, 0], -1)

    def test_future_changes_do_not_change_earlier_states_and_controls_exact(self):
        f = HourlyInferenceTests(); f.setUp()
        f.r['funding'] = np.zeros_like(f.r['close']); f.r['observed'] = np.ones_like(f.r['close'], bool)
        before, controls, records, _ = infer(f.t, f.r, f.d, f.models, [['BTC', 'ETH']], f.spec, self.spec)
        old, _, _ = old_infer(f.t, f.r, f.d, f.models, [['BTC', 'ETH']], 24, f.spec)
        for kind in controls:
            for key in controls[kind]:
                np.testing.assert_array_equal(controls[kind][key], old[kind][key])
        cut = 1800; f.r['close'][cut+1:] *= 2; f.r['funding'][cut+1:] = 1
        f.d['x'][cut+1:] *= -20; f.d['fresh'][cut+1:] = False
        after, _, changed, _ = infer(f.t, f.r, f.d, f.models, [['BTC', 'ETH']], f.spec, self.spec)
        for kind in before:
            for key in ['mu', 'directions', 'source_prices', 'signal_times', 'beta', 'vol']:
                np.testing.assert_array_equal(before[kind][key][:cut+1], after[kind][key][:cut+1])
        self.assertEqual([r for r in records if r['time'] <= f.t[cut]], [r for r in changed if r['time'] <= f.t[cut]])
        for state in before.values():
            self.assertTrue(np.all(state['directions'][~np.isfinite(state['mu'])] == 0))


if __name__ == '__main__':
    unittest.main()
