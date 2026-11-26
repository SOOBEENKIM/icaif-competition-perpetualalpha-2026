from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
from experiments.paired_venue_20261017.arrival import broker
from experiments.paired_venue_20261017.engine import PairExecution
from .engine import Hub, PairScope, Account, interval_risk

START = 1792238400000
COINS = ['par:BTC', 'par:EWY', 'par:SP500']
RISK = dict(gross_fraction=.6, net_fraction=.4, asset_fraction=.15, asset_notional=150000,
            daily_volatility=.005, drawdown_halt=.08, sizing_headroom=.9, normal_turnover_fraction=.15)


def fixture():
    cfg = json.loads(Path('experiments/paired_venue_20261017/protocol.json').read_text())
    p = dict(risk=RISK, execution=dict(decimals=[3] * 3))
    return dict(paper_start_unix=START / 1000, capture_end_unix=START / 1000 + 4 * 86400,
                end_liquidation_seconds=600, paired_protocol=cfg, trading_protocol=p, coins=COINS,
                overlay_coins=COINS[1:], metadata={c: dict(szDecimals=3) for c in COINS}, scale_control=1.04)


def snapshot(now, funding=.0004, quantities=None):
    books, ctx = {}, {}
    for i, c in enumerate(COINS):
        size = (quantities or {}).get(c, 10000)
        books[c] = dict(received_ms=now, data=dict(coin=c, time=now, levels=[
            [dict(px='99.99', sz=str(size), n=1)], [dict(px='100.01', sz=str(size), n=1)]]))
        ctx[c] = dict(received_ms=now, ctx=dict(markPx='100', oraclePx='100', funding=str(funding if i == 1 else 0)))
    signal = dict(incumbent=[.04, 0, 0], candidate=[.04, 0, 0],
        frame=dict(source_time=now // 3600000 * 3600, sigma=(np.eye(3) * 1e-6).tolist(),
                   retention=[1, 1, 1], volume=[1000000] * 3, trades=[1000] * 3,
                   paired_metrics=[dict(coins=COINS[1:], beta=1., correlation=.8)]))
    return books, ctx, signal


def drive(account, now, books, ctx, signal):
    for c, b in books.items(): account.paper.depth.update(c, b['data'])
    account.step(now, books, ctx, signal)
    account.paper.snapshot(now, ctx)


class EnsembleTests(unittest.TestCase):
    def hub(self):
        p = fixture(); h = Hub(broker(p['metadata'], p['paired_protocol']['scenarios']['base']), COINS, RISK)
        h.update_risk(np.ones(3) * 100, 1e6, np.eye(3) * 1e-6)
        return h

    def test_pair_cancellation_preserves_baseline_task_and_pending_order(self):
        h = self.hub(); h.request({'par:BTC': 10.}, START, START + 60000, 'base')
        b, c, _ = snapshot(START); h.tick(START, b, c)
        pending = deepcopy(h.paper.pending); task = deepcopy(h.paper.tasks['par:BTC'])
        scope = PairScope(h, COINS[1:]); scope.tasks = {}
        self.assertEqual(h.paper.pending, pending); self.assertEqual(h.paper.tasks['par:BTC'], task)
        self.assertFalse(scope.pending)

    def test_pair_can_enter_while_unrelated_order_is_in_flight(self):
        h = self.hub(); h.request({'par:BTC': 10.}, START, START + 60000, 'base')
        b, c, _ = snapshot(START); h.tick(START, b, c)
        manager = PairExecution(PairScope(h, COINS[1:]), fixture()['paired_protocol'])
        manager.enter(dict(coins=COINS[1:], target_qty=[-10., 10.], horizon_hours=72, predicted_net_bps=8), START)
        self.assertEqual(len(h.paper.pending), 1); self.assertEqual(len(h.paper.tasks), 3)

    def test_same_scope_pending_cannot_be_overwritten(self):
        h = self.hub(); h.request({'par:BTC': 10.}, START, START + 60000, 'base')
        b, c, _ = snapshot(START); h.tick(START, b, c)
        self.assertFalse(h.request({'par:BTC': 20.}, START, START + 60000, 'replace'))
        self.assertEqual(h.paper.tasks['par:BTC']['target'], 10.)

    def test_partial_fill_risk_rejected_even_when_final_pair_is_neutral(self):
        risk = dict(RISK, asset_fraction=1., asset_notional=1e6, gross_fraction=1., net_fraction=.2,
                    daily_volatility=1.)
        breaches = interval_risk({}, {'par:EWY': -3000., 'par:SP500': 3000.}, [], COINS,
                                  np.ones(3) * 100, 1e6, np.eye(3) * 1e-6, risk)
        self.assertIn('net', breaches)

    def test_other_scope_target_consumes_shared_risk_budget(self):
        h = self.hub(); h.risk = dict(RISK, net_fraction=.03)
        self.assertTrue(h.request({'par:BTC': 200.}, START, START + 60000, 'base'))
        self.assertFalse(h.allowed({'par:SP500': 200.}, START))

    def test_reduction_cannot_reverse_or_increase(self):
        h = self.hub(); h.paper.qty['par:BTC'] = 5
        for target in (-1., 6., float('nan')):
            with self.assertRaises(ValueError):
                h.request({'par:BTC': target}, START, START + 60000, 'risk', reduction=True)

    def test_price_change_is_checked_again_before_submission(self):
        h = self.hub()
        self.assertTrue(h.request({'par:BTC': 1400.}, START, START + 60000, 'base'))
        h.update_risk(np.ones(3) * 200, 1e6, np.eye(3) * 1e-6)
        b, c, _ = snapshot(START); h.tick(START, b, c)
        self.assertFalse(h.paper.pending)
        self.assertEqual(h.paper.tasks['par:BTC']['target'], 0)

    def test_no_carry_edge_reproduces_second_strategy_exactly(self):
        arms = [Account(n, 'base', fixture()) for n in ('second_strategy', 'second_plus_carry')]
        for sec in range(100):
            now = START + sec * 1000; b, c, s = snapshot(now, funding=0)
            if sec >= 60: s['frame']['source_time'] -= 7 * 3600
            for a in arms: drive(a, now, b, c, s)
        normalized = [[{k: v for k, v in f.items() if k != 'id'} for f in a.paper.fills] for a in arms]
        self.assertEqual(normalized[0], normalized[1])
        self.assertEqual(arms[0].paper.equity_series, arms[1].paper.equity_series)
        self.assertFalse(arms[0].paper.result(c)['fatal'])

    def test_one_ledger_baseline_plus_pair_paid_funding_and_complete_close(self):
        for scenario in ('base', 'stress'):
            a = Account('second_plus_carry', scenario, fixture())
            for sec in range(100):
                now = START + sec * 1000; b, c, s = snapshot(now)
                if sec == 30:
                    self.assertGreater(a.paper.qty['par:BTC'], 0)
                    self.assertLess(a.paper.qty['par:EWY'], 0)
                    self.assertGreater(a.paper.qty['par:SP500'], 0)
                    a.paper.charge_funding(now, dict(zip(COINS, [0, .0004, 0])), c)
                if sec >= 60: s['frame']['source_time'] -= 7 * 3600
                drive(a, now, b, c, s)
            r = a.paper.result(c)
            self.assertTrue(r['flat']); self.assertFalse(r['fatal']); self.assertGreater(r['funding'], 0)
            self.assertEqual(r['fill_count'], 6); self.assertEqual(r['pending_orders'], 0)
            self.assertEqual(sum(e['event'] == 'balanced_open' for e in a.pair.events), 1)
            self.assertEqual(sum(e['event'] == 'flat_confirmed' for e in a.pair.events), 1)

    def test_pair_partial_fill_is_closed_without_cancelling_baseline(self):
        a = Account('second_plus_carry', 'base', fixture())
        for sec in range(40):
            now = START + sec * 1000
            b, c, s = snapshot(now, quantities={'par:SP500': 0} if sec > 0 else None)
            drive(a, now, b, c, s)
        self.assertGreater(a.paper.qty['par:BTC'], 0)
        self.assertEqual(a.paper.qty.get('par:EWY', 0), 0)
        self.assertEqual(a.paper.qty.get('par:SP500', 0), 0)
        self.assertTrue(any(e.get('reason') == 'unhedged_abort' for e in a.pair.events))
        self.assertFalse(a.paper.fatal)

    def test_entry_without_any_submitted_order_has_a_deadline(self):
        a = Account('second_plus_carry', 'base', fixture())
        a.hub.update_risk(np.ones(3) * 100, 1e6, np.eye(3) * 1e-6)
        a.pair.enter(dict(coins=COINS[1:], target_qty=[-.01, .01], horizon_hours=72, predicted_net_bps=8), START)
        # Tiny synthetic lots fail the minimum executable order size. There is
        # no submitted order; the pair must nevertheless leave entering state.
        for sec in range(40):
            now = START + sec * 1000; b, c, s = snapshot(now, funding=0)
            drive(a, now, b, c, s)
        self.assertIsNone(a.pair.state)
        self.assertTrue(any(e.get('reason') == 'entry_deadline' for e in a.pair.events))

    def test_baseline_asset_claim_closes_pair_before_taking_ownership(self):
        a = Account('second_plus_carry', 'base', fixture())
        for sec in range(10):
            now = START + sec * 1000; b, c, s = snapshot(now); drive(a, now, b, c, s)
        for sec in range(6 * 3600, 6 * 3600 + 40):
            now = START + sec * 1000; b, c, s = snapshot(now)
            s['candidate'] = [.04, .01, 0.]
            drive(a, now, b, c, s)
        self.assertTrue(any(e.get('reason') == 'baseline_has_priority' for e in a.pair.events))
        self.assertIsNone(a.pair.state)
        self.assertGreater(a.paper.qty['par:EWY'], 0)
        self.assertFalse(a.paper.fatal)


if __name__ == '__main__': unittest.main()
