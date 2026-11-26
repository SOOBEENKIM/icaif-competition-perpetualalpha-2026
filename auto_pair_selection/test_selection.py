import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .selection import PAIRS, IDS, SYMBOLS, SelectionSpec, select
from .signals import AutomaticPairSignals, prepare_features, fit_day, state_for_arm


def data(days=95):
    rng = np.random.default_rng(20261023)
    n = days*24+1
    returns = rng.normal(0, .002, (n-1, 10))
    # Deliberately make a non-crypto pair eligible without naming it in rules.
    returns[:, 5] = returns[:, 4] + rng.normal(0, .0003, n-1)
    close = 100*np.exp(np.vstack([np.zeros(10), np.cumsum(returns, axis=0)]))
    return dict(market_times=np.arange(n)*3600, close=close,
                volume=np.full_like(close, 1000.), trades=np.ones_like(close))


class SelectionTests(unittest.TestCase):
    def test_all_45_identities_and_noncrypto_pair_can_pass(self):
        d = data(); result = select(d, 90*86400)
        self.assertEqual(len(PAIRS), 45)
        self.assertEqual(len(set(PAIRS)), 45)
        self.assertEqual(len(result['pairs']), 45)
        k = PAIRS.index(('EWY', 'EWJ'))
        self.assertTrue(result['selected'][k])
        self.assertFalse(result['selected'][PAIRS.index(('BTC', 'ETH'))])
        self.assertEqual(result['pairs'][k]['reasons'], [])

    def test_future_prices_cannot_change_selection(self):
        d = data(); original = select(d, 90*86400)
        other = {k:v.copy() for k,v in d.items()}
        future = d['market_times'] > 90*86400
        other['close'][future] *= np.arange(1, 11)*100
        other['volume'][future] = 0
        self.assertEqual(original, select(other, 90*86400))

    def test_stale_or_thin_leg_is_rejected_with_reason(self):
        d = data(); d['volume'][:, 4] = 0
        r = select(d, 90*86400)['pairs'][PAIRS.index(('EWY', 'EWJ'))]
        self.assertFalse(r['selected']); self.assertIn('insufficient_fresh_history', r['reasons'])
        d = data(); d['volume'][:, 4] = .001
        r = select(d, 90*86400)['pairs'][PAIRS.index(('EWY', 'EWJ'))]
        self.assertIn('low_reference_leg_liquidity', r['reasons'])

    def test_missing_hour_and_wrong_asset_order_are_errors(self):
        d = data(); d['market_times'][20] += 1
        with self.assertRaises(ValueError): select(d, 90*86400)
        d = data(); d['symbols'] = list(reversed(SYMBOLS))
        with self.assertRaises(ValueError): select(d, 90*86400)

    def test_deselection_only_blocks_new_entry_not_held_forecast(self):
        n = len(PAIRS)
        state = dict(mu=np.ones((2,n)), directions=np.ones((2,n),bool),
            selection_eligible=np.ones((2,n),bool), gate_available=np.ones((2,n),bool),
            gate_action=np.ones((2,n),int), pairs=np.asarray(IDS))
        k = PAIRS.index(('EWY', 'EWJ'));state['selection_eligible'][1,k] = False
        changed = state_for_arm(state, 'automatic_45')
        self.assertFalse(changed['economic_available'][1,k])
        self.assertEqual(changed['economic_action'][1,k], 0)
        self.assertEqual(changed['mu'][1,k], 1.)
        self.assertTrue(changed['directions'][1,k])
        np.testing.assert_array_equal(changed['pairs'], state['pairs'])

    def test_expansion_does_not_invent_gate_permission(self):
        n = len(PAIRS)
        state = dict(mu=np.ones((1,n)), directions=np.ones((1,n),bool),
            selection_eligible=np.ones((1,n),bool), gate_available=np.zeros((1,n),bool),
            gate_action=np.ones((1,n),int))
        changed = state_for_arm(state, 'automatic_45')
        self.assertFalse(changed['economic_available'].any())
        self.assertFalse(changed['economic_action'].any())

    def test_training_excludes_unmatured_labels(self):
        d = data(); f,y,maturity,terminal = prepare_features(d)
        spec = json.loads(Path('competition_live/strategy_plan.json').read_text())['direct_spec']
        gs = json.loads(Path('competition_live/config.json').read_text())['gate_spec']
        source = 90*86400
        # A strict screening threshold suppresses gate fits: this property
        # checks real direct coefficients and maturity, not solver runtime.
        strict = SelectionSpec(minimum_correlation=.99999)
        a = fit_day(d,f,y,maturity,terminal,source,spec,gs,strict)
        poison = y.copy();poison[maturity >= source] = 1e9
        b = fit_day(d,f,poison,maturity,terminal,source,spec,gs,strict)
        self.assertEqual(a['direct'], b['direct'])
        self.assertEqual(len(a['direct']), 45)
        self.assertTrue(all(v['last_maturity'] < source for v in a['direct'].values()))

    def test_adapter_rejects_backdated_or_revised_publication(self):
        d = data(5);source=int(d['market_times'][-1])
        spec = json.loads(Path('competition_live/strategy_plan.json').read_text())['direct_spec']
        gs = json.loads(Path('competition_live/config.json').read_text())['gate_spec']
        a = AutomaticPairSignals(spec,gs)
        with self.assertRaises(ValueError):a.ingest(d,source,source-1)
        # Insufficient history is an explicit no-trade output.
        self.assertEqual(a.ingest(d,source,source+5), [])
        self.assertEqual(a.ingest(d,source,source+7), [])
        revised = copy.deepcopy(d);revised['close'][-1,0] += 1
        with self.assertRaises(ValueError):a.ingest(revised,source,source+8)

    def test_midday_restart_does_not_backdate_midnight_model(self):
        d = data(5);d={k:v[:-1] for k,v in d.items()};source=int(d['market_times'][-1])
        spec = json.loads(Path('competition_live/strategy_plan.json').read_text())['direct_spec']
        gs = json.loads(Path('competition_live/config.json').read_text())['gate_spec']
        a = AutomaticPairSignals(spec,gs)
        self.assertEqual(a.ingest(d,source,source+5), [])
        self.assertIsNone(a.model)


if __name__ == '__main__':unittest.main()
