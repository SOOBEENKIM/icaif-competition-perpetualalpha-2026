import copy
import json
from pathlib import Path
import unittest

from .engine import Depth, Paper

P = json.loads(Path(__file__).with_name('protocol.json').read_text())
COIN = 'par:BTC'
META = {COIN: {'szDecimals': 4}}


def book(t, amount=.2, ask=100.01):
    return {'coin': COIN, 'time': t, 'levels': [[{'px': '99.99', 'sz': str(amount)}],
                                              [{'px': str(ask), 'sz': str(amount)}]]}


def frame(paper, t, amount=.2, ask=100.01):
    b = book(t, amount, ask)
    paper.depth.update(COIN, b)
    books = {COIN: {'received_ms': t, 'data': b}}
    contexts = {COIN: {'received_ms': t, 'ctx': {'markPx': '100', 'oraclePx': '100'}}}
    paper.tick(t, books, contexts)
    paper.snapshot(t, contexts)
    return contexts


class ExecutionReplayTests(unittest.TestCase):
    def test_funding_uses_actual_settlement_timestamp(self):
        from .development import funding_times
        self.assertEqual(funding_times({3600048: {COIN: .001}}, 3500000, 3700000), [3600048])
        with self.assertRaises(ValueError):
            funding_times({}, 3500000, 3700000)

    def test_unchanged_displayed_depth_cannot_be_spent_twice(self):
        d = Depth();d.update(COIN, book(1))
        self.assertEqual(d.consume(COIN, True, .2, 101), [(.2, 100.01)])
        d.update(COIN, book(2))
        self.assertEqual(d.consume(COIN, True, .2, 101), [])
        d.update(COIN, book(3, .3))
        self.assertAlmostEqual(sum(q for q, _ in d.consume(COIN, True, 1, 101)), .1, places=7)

    def test_same_timestamp_cannot_replenish_consumed_depth(self):
        d = Depth();d.update(COIN, book(1));d.consume(COIN, True, 1, 101)
        d.update(COIN, book(1, 5))
        self.assertEqual(d.consume(COIN, True, 1, 101), [])

    def test_fixed_retries_reduce_residual_only_when_new_liquidity_appears(self):
        arms = [Paper(n, P, P['scenarios']['base'], META) for n in ('baseline', 'candidate')]
        for a in arms:
            a.request_target({COIN: 1}, 100000, 1000000, 'r')
            frame(a, 100000);frame(a, 101000)
            frame(a, 130000, .4);frame(a, 131000, .4)
        self.assertAlmostEqual(arms[0].qty[COIN], .2)
        self.assertAlmostEqual(arms[1].qty[COIN], .4)
        self.assertEqual(len(arms[0].fills), 1)

    def test_arrival_price_cannot_cross_the_submitted_limit(self):
        a = Paper('candidate', P, P['scenarios']['base'], META)
        a.request_target({COIN: 1}, 100000, 1000000, 'r')
        frame(a, 100000);frame(a, 101000, ask=102)
        self.assertEqual(a.qty.get(COIN, 0), 0)
        self.assertEqual(a.attempts[-1]['filled'], 0)

    def test_candidate_does_not_chase_past_initial_price_budget(self):
        a = Paper('candidate', P, P['scenarios']['base'], META)
        a.request_target({COIN: 1}, 100000, 1000000, 'r')
        frame(a, 100000);frame(a, 101000)
        frame(a, 130000, ask=101);frame(a, 131000, ask=101)
        self.assertAlmostEqual(a.qty[COIN], .2)
        self.assertEqual(a.attempts[-1]['status'], 'no_executable_size')

    def test_cash_reconciles_after_entry_exit_fees_and_funding(self):
        a = Paper('candidate', P, P['scenarios']['base'], META)
        a.request_target({COIN: .2}, 100000, 1000000, 'r')
        frame(a, 100000);ctx=frame(a, 101000)
        a.charge_funding(101000, {COIN: .001}, ctx)
        a.request_target({COIN: 0}, 102000, 1000000, 'exit')
        frame(a, 102000);ctx=frame(a, 103000)
        result=a.result(ctx)
        self.assertTrue(result['flat'])
        self.assertAlmostEqual(result['fees'], .036)
        self.assertAlmostEqual(result['funding'], -.02)
        self.assertAlmostEqual(result['mark_net_pnl'], -.06)

    def test_missing_funding_invalidates_result_instead_of_becoming_zero(self):
        a = Paper('candidate', P, P['scenarios']['base'], META)
        a.request_target({COIN: .2}, 100000, 1000000, 'r')
        frame(a, 100000);ctx=frame(a, 101000)
        a.charge_funding(101000, {}, ctx)
        self.assertTrue(a.fatal)

    def test_stale_arrival_cannot_fill(self):
        a = Paper('baseline', P, P['scenarios']['base'], META)
        a.request_target({COIN: 1}, 100000, 1000000, 'r')
        frame(a, 100000)
        a.apply_pending(120000, {COIN: {'received_ms': 100000, 'data': book(100000)}})
        self.assertEqual(a.qty, {})
        self.assertEqual(a.attempts[-1]['status'], 'unfilled_stale_arrival_book')

    def test_reversal_cannot_open_new_side_before_flat(self):
        a = Paper('candidate', P, P['scenarios']['base'], META)
        a.qty[COIN] = .4;a.cash -= 40
        a.request_target({COIN: -.4}, 100000, 1000000, 'reverse')
        frame(a, 100000);frame(a, 101000)
        self.assertAlmostEqual(a.qty[COIN], .2)
        self.assertTrue(all(f['phase'] == 'reduce' for f in a.fills))

    def test_stress_fraction_respects_exchange_lot_size(self):
        d = Depth(.5);d.update(COIN, book(1, .0001))
        self.assertEqual(d.consume(COIN, True, .0001, 101, decimals=4), [])

    def test_future_dated_book_cannot_fill(self):
        a = Paper('baseline', P, P['scenarios']['base'], META)
        a.request_target({COIN: 1}, 100000, 1000000, 'r')
        frame(a, 100000)
        a.apply_pending(101000, {COIN: {'received_ms': 101000, 'data': book(120000)}})
        self.assertEqual(a.qty, {})

    def test_recorded_capacity_caps_new_exposure(self):
        a = Paper('candidate', P, P['scenarios']['base'], META)
        a.request_target({COIN: 1}, 100000, 1000000, 'r')
        b = book(100000, 2)
        a.depth.update(COIN, b)
        ctx = {COIN: {'received_ms': 100000, 'ctx': {'markPx': '100', 'oraclePx': '100'}}}
        a.tick(100000, {COIN: {'received_ms': 100000, 'data': b}}, ctx,
               {COIN: {'received_ms': 100000, 'data': {'maxTradeSzs': [.1, .1]}}})
        self.assertAlmostEqual(a.pending[0]['size'], .1)

    def test_stale_capacity_blocks_increase(self):
        a = Paper('candidate', P, P['scenarios']['base'], META)
        a.request_target({COIN: 1}, 100000, 1000000, 'r')
        b = book(100000, 2);a.depth.update(COIN, b)
        ctx = {COIN: {'received_ms': 100000, 'ctx': {'markPx': '100', 'oraclePx': '100'}}}
        a.tick(100000, {COIN: {'received_ms': 100000, 'data': b}}, ctx, {})
        self.assertEqual(a.pending, [])

    def test_daily_std_requires_whole_utc_days_not_initial_partial_day(self):
        from .forward import daily_metrics
        day = 86400000
        rows = [{'time': t, 'oracle_equity': v} for t, v in [(day-1000, 100), (2*day-1000, 101), (3*day-1000, 100)]]
        m = daily_metrics(rows, day//2, 3*day)
        self.assertEqual(m['complete_days'], 2)
        self.assertAlmostEqual(m['daily_volatility_percent'], np_std([.01, 100/101-1]))

    def test_stale_daily_boundary_does_not_create_return(self):
        from .forward import daily_metrics
        day = 86400000
        rows = [{'time': day-20000, 'oracle_equity': 100}, {'time': 2*day-1000, 'oracle_equity': 101}]
        self.assertEqual(daily_metrics(rows, day//2, 2*day)['complete_days'], 0)

    def test_completed_lifecycle_requires_actual_flattening(self):
        from .forward import completed_cycles
        fills = [{'coin': COIN, 'qty': q} for q in [.3, -.1, -.2, .3]]
        self.assertEqual(completed_cycles(fills), 1)

    def test_forward_pipeline_runs_controller_and_refuses_small_sample_promotion(self):
        from unittest.mock import patch
        from . import forward
        coins = forward.COINS
        root = Path(__file__).resolve().parents[2]
        protocol = json.loads((root / 'replay_lab/output/tail_calibration/20261014-v1/baseline.json').read_text())['protocol']
        def events():
            for second in range(21590, 21691):
                for coin in coins:
                    t = second * 1000
                    ctx = {'coin': coin, 'ctx': {'markPx': '100', 'oraclePx': '100'}}
                    b = book(t, .2);b['coin'] = coin
                    for kind, data in [('activeAssetCtx', ctx), ('l2Book', b),
                                       ('activeAssetData', {'coin': coin, 'maxTradeSzs': [100, 100]})]:
                        yield {'received_ms': t, 'channel': kind, 'data': data}
                if second == 21602:
                    yield {'received_ms': second*1000, 'channel': 'strategySignal', 'data': {
                        'hour': 21600, 'weights': [.001] + [0.]*9,
                        'sigma': [[.000001 if i == j else 0. for j in range(10)] for i in range(10)],
                        'volume': [10000]*10, 'trades': [100]*10, 'protocol': protocol, 'input_hash': 'synthetic'}}
        def funding(payload):
            return {'response': [{'coin': payload['coin'], 'time': 21600048, 'fundingRate': '0'}]}
        with patch.object(forward, 'recorded_events', events), patch.object(forward, 'api', funding), patch.object(forward, 'save'):
            result = forward.evaluate_data({'test_start': 21590, 'test_end': 21690})
        self.assertEqual(result['decision'], 'HOLD_EXISTING')
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['orders_sent'], 0)
        self.assertEqual(result['results']['base:baseline']['fill_count'], 1)
        self.assertLess(result['results']['base:baseline']['oracle_net_pnl'], 0)


def np_std(values):
    import numpy as np
    return float(np.std(values) * 100)


if __name__ == '__main__':
    unittest.main()
