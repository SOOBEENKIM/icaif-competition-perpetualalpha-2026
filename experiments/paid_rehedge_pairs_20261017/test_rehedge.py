import copy
import json
import unittest
from pathlib import Path
import numpy as np
from experiments.inventory_renewal_pairs_20261017 import test_renewal as fixture
from experiments.inventory_renewal_pairs_20261017.portfolio import replay as previous_replay
from experiments.shared_execution_pairs_20261017.account_v2 import SharedAccount
from experiments.shared_execution_pairs_20261017 import test_account as book_fixture
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution
from .portfolio import replay
from .policy import anchor_stop, held_state, renewal_plan, reconcile_adjustment


class PaidAdjustmentTests(unittest.TestCase):
    def setUp(self):
        f = fixture.RenewalTests(); f.setUp()
        self.d, self.de, self.cfg, self.p = f.d, f.de, f.cfg, f.p
        self.cfg['rehedge_policy'] = json.loads((Path(__file__).parent/'par_protocol.json').read_text())['rehedge_policy']
        self.t = self.d['times']; self.change = int(self.t[0])+18*3600
        self.i = int(np.searchsorted(self.t, self.change))
        self.de['beta'][self.t >= self.change] = 2.

    def run_arm(self, arm='pair_dense_renewhedge_ridge_24', engine=replay):
        return engine(self.d, self.de, None, self.cfg, self.p, arm, 'base', int(self.t[0]), int(self.t[-1]))

    def direct(self):
        a = SharedAccount(1e6, self.p['symbols']); q = np.zeros(10); q[[1, 0]] = [5., -5.]
        for j in [1, 0]:
            a.physical_fill(j, q[j], 100., 9, int(self.t[0])+300)
            a.owner_fill('pair', j, q[j], 100., 9, int(self.t[0])+300, 'external_fill')
        pair = dict(renewable=True, dense=True, balanced=True, paid_rehedge_enabled=True,
                    ids=[1, 0], pair=['ETH', 'BTC'], pair_index=0, target=[5., -5.], direction=1,
                    forecast_source=int(self.t[0]), horizon=24, first_fill=int(self.t[0])+300,
                    entry_vol=.002, risk_entry_prices=[100., 100.], risk_quantities=[5., -5.])
        anchor_stop(pair, a.owners['pair'], self.d['par_close'][self.i])
        return pair, a

    def plan(self, pair, account):
        return renewal_plan(pair, account.owners['pair'].qty, self.i, self.d, self.de, self.cfg, self.p,
                            self.cfg['scenarios']['base'], account.physical.qty, 1e6, ledger=account.owners['pair'])

    def test_old_control_paths_reproduce_exactly(self):
        for arm in ['first_strategy', 'second_strategy', 'first_dense_ridge_24', 'second_dense_renew_bias_24']:
            a = self.run_arm(arm); b = self.run_arm(arm, previous_replay)
            for key in ['fills', 'snapshots', 'events', 'metrics']:
                self.assertEqual(a[key], b[key], (arm, key))

    def test_no_adjustment_needed_preserves_previous_renewal(self):
        self.de['beta'][:] = 1.
        a = self.run_arm(); b = self.run_arm('pair_dense_renew_ridge_24', previous_replay)
        for key in ['fills', 'snapshots', 'events']:
            self.assertEqual(a[key], b[key], key)
        self.assertEqual(a['metrics']['net_pnl'], b['metrics']['net_pnl'])
        self.assertEqual(a['metrics']['hedge_adjustment_intents'], 0)

    def test_adjustment_trades_only_deltas_and_waits_for_actual_fills(self):
        r = self.run_arm(); m = r['metrics']
        planned = next(e for e in r['events'] if e['event'] == 'hedge_adjustment_planned')
        done = next(e for e in r['events'] if e.get('outcome') == 'renewed')
        self.assertEqual(done['time'], planned['time']+300)
        self.assertEqual(done['new_maturity'], planned['time']+24*3600)
        self.assertLess(abs(planned['target_gross']/planned['current_gross']-1), .0001)
        fills = [f for f in r['owner_fills']['pair'] if f['time'] == done['time']]
        self.assertEqual(len(fills), 2)
        self.assertTrue(all(abs(f['signed_qty']) < 2 and f['fee'] > 0 for f in fills))
        self.assertEqual(m['hedge_adjustments_completed'], 1)
        self.assertEqual(m['completed_pairs'], 0)
        self.assertEqual(m['renewals'], 2)
        self.assertAlmostEqual(m['net_pnl'], -m['execution_cost']-m['fees'], places=7)

    def test_zero_fills_retain_old_maturity_without_fake_renewal(self):
        self.d['par_volume'][self.i:self.i+2, :2] = 0
        r = self.run_arm(); e = next(e for e in r['events'] if e['event'] == 'rehedge_unfilled')
        self.assertEqual(e['actual_delta'], [0., 0.])
        close = next(e for e in r['events'] if e['event'] == 'closed')
        self.assertEqual(close['time'], int(self.t[0])+24*3600)
        self.assertFalse(any(e['event'] == 'renewed' and e['time'] <= close['time'] for e in r['events']))

    def test_one_leg_fill_is_retained_and_paid_unwind_is_sticky(self):
        blocked = (self.t >= self.change) & (self.t < self.change+3*3600)
        self.d['par_volume'][blocked, 0] = 0
        r = self.run_arm(); e = next(e for e in r['events'] if e['event'] == 'rehedge_unwind')
        self.assertEqual(e['reason'], 'partial_adjustment_hedge_mismatch')
        self.assertNotEqual(e['actual_delta'][0], 0.)
        self.assertEqual(e['actual_delta'][1], 0.)
        close = next(e for e in r['events'] if e['event'] == 'closed')
        self.assertGreater(close['time'], self.change+3*3600)
        self.assertFalse(any(e['event'] == 'renewed' and e['time'] <= close['time'] for e in r['events']))
        self.assertGreater(r['metrics']['fees'], 0)

    def test_other_leg_future_price_cannot_retroactively_cancel_fill(self):
        self.d['par_close'][self.i+1, 0] = 80.
        r = self.run_arm(); e = next(e for e in r['events'] if e['event'] == 'rehedge_unwind')
        self.assertNotEqual(e['actual_delta'][0], 0.)
        self.assertEqual(e['actual_delta'][1], 0.)
        fs = [f for f in r['owner_fills']['pair'] if f['time'] == self.change+300]
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0]['symbol'], 'ETH')

    def test_added_exposure_uses_original_cumulative_loss_stop(self):
        at = int(self.t[0])+20*3600
        self.d['par_close'][self.t >= at, 0] = 105.
        r = self.run_arm(); e = next(e for e in r['events'] if e['event'] == 'exit_requested')
        self.assertEqual(e['reason'], 'pair_return_stop'); self.assertEqual(e['time'], at)
        self.assertLess(e['episode_price_pnl'], -30.)
        self.assertAlmostEqual(e['stop_initial_gross'], 1000.)
        self.assertAlmostEqual(e['return_stop_bound'], .002*3*np.sqrt(24))
        # Old fixed quantities would show only a 26-dollar marked price loss.
        self.assertLess(26/1000, e['return_stop_bound'])

    def test_stop_anchor_includes_realized_resize_losses_without_fees_or_funding_reset(self):
        pair, a = self.direct(); owner = a.owners['pair']; origin = pair['stop_price_origin']
        a.owner_fill('pair', 1, -2., 90., 9, self.change+300, 'external_fill')
        a.owner_fill('pair', 0, -2., 110., 9, self.change+300, 'external_fill')
        pair['has_rehedged'] = True
        s = held_state(pair, self.i+1, self.d, self.de, self.cfg, ledger=owner)
        expected = owner.equity(self.d['par_close'][self.i+1])+owner.fees-owner.funding-origin
        self.assertAlmostEqual(s['episode_price_pnl'], expected)
        anchor_stop(pair, owner, self.d['par_close'][self.i+1])
        self.assertEqual(origin, pair['stop_price_origin'])

    def test_economic_limits_budget_both_independent_legs(self):
        pair, a = self.direct(); plan = self.plan(pair, a)
        self.assertTrue(plan['requires_adjustment'])
        self.assertGreaterEqual(plan['minimum_advantage_at_limits'], plan['buffer_cash']-1e-8)
        self.assertLess(plan['incremental_advantage'], plan['expected_hold_cash']-plan['immediate_close_cash'])
        self.assertEqual(set(plan['entry_limits']), {'0', '1'})

    def test_independent_leg_risk_checked_even_when_final_pair_is_safe(self):
        pair, a = self.direct(); self.p['risk']['gross_fraction'] = .0011
        plan = self.plan(pair, a)
        self.assertFalse(plan['accepted'])
        self.assertEqual(plan['reason'], 'adjustment_independent_leg_risk')

    def book_adjustment(self, missing=False):
        pair, a = self.direct(); plan = self.plan(pair, a)
        pair.update(rehedging=True, rehedge_plan=plan)
        target = a.owners['pair'].qty.copy(); target[pair['ids']] = plan['target']
        sc = dict(self.cfg['scenarios']['base'], depth_fraction=1., latency_ms=1000)
        packet = a.seal(dict(decision_time=self.change, source_close=self.change, source_available=self.change,
                            source_prices=self.d['par_close'][self.i].tolist(), risk_forced=False,
                            owner_targets={'base':np.zeros(10).tolist(), 'pair':target.tolist()},
                            owner_limits={'pair':plan['entry_limits']}), self.p, sc)
        book = book_fixture.AccountTests().book
        streams = {'ETH':book(at=self.change*1000+1000, size=10.)}
        if not missing:
            streams['BTC'] = book(at=self.change*1000+1000, size=.5)
        cfg = dict(book_max_age_ms=15000, context_max_age_ms=3000, future_exit_observation_grace_ms=5000)
        a.execute_books(packet, streams, self.d['par_close'][self.i], self.d['sigma'][self.i], self.p, sc, cfg)
        outcome = reconcile_adjustment(pair, a.owners['pair'], self.i+1, self.d, self.de, self.cfg,
                                       self.p, sc, a.physical.qty, a.physical.equity(self.d['par_close'][self.i+1]),
                                       all_outcomes_observed=not a.unknown)
        return pair, a, outcome, packet, sc, cfg

    def test_received_book_depth_partial_adjustment_is_not_atomic(self):
        pair, a, outcome, _, _, _ = self.book_adjustment()
        self.assertEqual(outcome['outcome'], 'unwind')
        self.assertAlmostEqual(outcome['actual_delta'][1], -.5)
        self.assertTrue(pair['closing']); self.assertNotIn('renewal_maturity', pair)
        a.reconcile(self.d['par_close'][self.i+1])

    def test_unknown_book_outcome_keeps_pending_state_and_blocks_resubmission(self):
        pair, a, outcome, packet, sc, cfg = self.book_adjustment(missing=True)
        self.assertEqual(outcome['outcome'], 'awaiting_reconciliation')
        self.assertTrue(pair['rehedging']); self.assertIn('rehedge_plan', pair)
        self.assertNotIn('renewal_maturity', pair)
        with self.assertRaises(RuntimeError):
            a.execute_books(packet, {}, self.d['par_close'][self.i], self.d['sigma'][self.i], self.p, sc, cfg)

    def test_insufficient_value_does_not_pay_adjustment_cost(self):
        pair, a = self.direct(); self.de['mu'][self.i] = 0.
        plan = self.plan(pair, a)
        self.assertFalse(plan['accepted']); self.assertEqual(plan['reason'], 'adjustment_cost_exceeds_holding_advantage')

    def test_repeat_source_and_tiny_delta_do_not_make_orders(self):
        pair, a = self.direct(); pair['last_rehedge_attempt_source'] = self.change
        self.assertEqual(self.plan(pair, a)['reason'], 'adjustment_source_already_attempted')
        pair.pop('last_rehedge_attempt_source'); self.p['execution']['minimum_order_value'] = 200.
        self.assertEqual(self.plan(pair, a)['reason'], 'adjustment_below_minimum_order')

    def test_new_maturity_cannot_bypass_hard_holding_limit(self):
        self.cfg['renewal_policy']['maximum_holding_hours'] = 36
        r = self.run_arm()
        self.assertEqual(r['metrics']['hedge_adjustment_intents'], 0)
        self.assertEqual(next(e for e in r['events'] if e['event']=='closed')['time'], int(self.t[0])+24*3600)

    def test_future_changes_leave_earlier_fills_and_decisions_invariant(self):
        a = self.run_arm(); cut = self.change+3600
        self.d['par_close'][self.t > cut] *= 1.2
        self.de['mu'][self.t > cut] *= -2
        b = self.run_arm()
        for key in ['events', 'fills', 'snapshots']:
            self.assertEqual([e for e in a[key] if e['time'] <= cut], [e for e in b[key] if e['time'] <= cut], key)

    def test_actual_funding_and_cost_attribution_reconcile_after_resize(self):
        self.d['funding'][:, 1] = .0001
        self.d['known_hourly_funding'][:, 1] = .0001
        r = self.run_arm(); m = r['metrics']
        # 18 hourly debits on five ETH, then 30 on the resized actual quantity.
        self.assertAlmostEqual(m['funding'], -18*.05-30*3.3333*.01)
        self.assertAlmostEqual(m['net_pnl'], m['price_move_pnl']-m['execution_cost']-m['fees']+m['funding'], places=7)
        audit = episode_attribution(r, self.d, self.cfg['scenarios']['base'])
        self.assertLess(audit['accounting_error'], 1e-7)


if __name__ == '__main__':
    unittest.main()
