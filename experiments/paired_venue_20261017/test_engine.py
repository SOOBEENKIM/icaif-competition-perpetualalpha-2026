import copy
import json
from pathlib import Path
import unittest
import numpy as np
from .engine import broker, quote_pair, PairExecution

CFG = json.loads(Path(__file__).with_name('protocol.json').read_text())
META = {c: {'name': c, 'szDecimals': 3} for c in ('par:ETH', 'par:BTC')}
COINS = list(META)


def market(now, sizes=(20., 20.)):
    books, ctx = {}, {}
    for coin, size in zip(COINS, sizes):
        books[coin] = dict(received_ms=now, data=dict(time=now, coin=coin,
            levels=[[dict(px='99', sz=str(size))], [dict(px='101', sz=str(size))]]))
        ctx[coin] = dict(received_ms=now, ctx=dict(markPx='100', oraclePx='100', funding='0'))
    return books, ctx


def intent():
    return dict(coins=COINS, target_qty=[2., -2.], predicted_net_bps=20., horizon_hours=24)


class ExecutionTests(unittest.TestCase):
    def step(self, execution, now, sizes=(20., 20.)):
        books, contexts = market(now, sizes)
        for c, b in books.items(): execution.paper.depth.update(c, b['data'])
        execution.tick(now, books, contexts)
        return books, contexts

    def make(self):
        return PairExecution(broker(META, CFG['scenarios']['base']), CFG)

    def test_partial_pair_repairs_only_overfilled_leg_and_charges_fee(self):
        e = self.make(); e.enter(intent(), 100000)
        self.step(e, 100000)
        self.step(e, 101000, (20., 1.))
        self.assertEqual(e.state['phase'], 'balancing')
        self.step(e, 102000, (20., 1.))
        self.assertEqual(e.state['phase'], 'holding')
        np.testing.assert_allclose([e.paper.qty[c] for c in COINS], [1., -1.])
        self.assertEqual(len(e.paper.fills), 3)
        self.assertAlmostEqual(e.paper.fees, (202 + 99 + 99) * .0009)
        self.assertTrue(e.paper.fills[-1]['phase'] == 'reduce')

    def test_balanced_partial_fills_stay_open(self):
        e = self.make(); e.enter(intent(), 100000)
        self.step(e, 100000); self.step(e, 101000, (1., 1.))
        self.assertEqual(e.state['phase'], 'holding')
        self.step(e, 102000)
        self.assertEqual(len(e.paper.fills), 2)
        self.assertFalse(e.paper.tasks)

    def test_unchanged_depth_does_not_replenish_and_exit_is_not_invented(self):
        e = self.make(); e.enter(intent(), 100000)
        self.step(e, 100000); self.step(e, 101000)
        e.close(102000, 'test')
        books, ctx = market(102000, (0., 0.))
        for c, b in books.items(): e.paper.depth.update(c, b['data'])
        e.tick(102000, books, ctx)
        books, ctx = market(403000, (0., 0.))
        for c, b in books.items(): e.paper.depth.update(c, b['data'])
        e.tick(403000, books, ctx)
        self.assertTrue(e.paper.fatal)
        self.assertTrue(any(e.paper.qty.values()))
        self.assertFalse(any(x['event'] == 'flat_confirmed' for x in e.events))

    def test_no_fill_before_latency_and_funding_uses_actual_inventory(self):
        e = self.make(); e.enter(intent(), 100000)
        self.step(e, 100000)
        _, ctx = self.step(e, 100999)
        self.assertFalse(e.paper.fills)
        _, ctx = self.step(e, 101000)
        e.paper.charge_funding(101000, {COINS[0]: .001, COINS[1]: .002}, ctx)
        self.assertAlmostEqual(e.paper.funding, .2)

    def test_capacity_jointly_scales_both_legs_and_includes_exit(self):
        books, ctx = market(100000, (2., 1.))
        snapshots = {c: dict(book=books[c]['data'], oracle=100., mid=100., ctx=ctx[c]['ctx']) for c in COINS}
        q = quote_pair(COINS, snapshots, META, 1., 1, 24, CFG, CFG['scenarios']['base'], gross=1000.)
        np.testing.assert_allclose(q['target_qty'], [1., -1.])
        self.assertAlmostEqual(q['capacity_scale'], .2)
        self.assertLess(q['predicted_net_pnl'], 0.)
        # Existing two-unit positions cannot be valued as one-unit exits.
        self.assertIsNone(quote_pair(COINS, snapshots, META, 1., 1, 24, CFG, CFG['scenarios']['base'], fixed_quantity=[2., -2.]))

    def test_order_book_is_not_replenished_by_identical_snapshots(self):
        e = self.make(); b, _ = market(100000, (1., 1.))
        e.paper.depth.update(COINS[0], b[COINS[0]]['data'])
        self.assertEqual(sum(x[0] for x in e.paper.depth.consume(COINS[0], True, 1, 101)), 1)
        b, _ = market(101000, (1., 1.)); e.paper.depth.update(COINS[0], b[COINS[0]]['data'])
        self.assertEqual(e.paper.depth.consume(COINS[0], True, 1, 101), [])


if __name__ == '__main__': unittest.main()
