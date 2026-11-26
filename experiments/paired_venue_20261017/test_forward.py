import copy
from types import SimpleNamespace
import unittest
import numpy as np
from .arrival import broker
from .engine import PairExecution
from .forward import current_snapshots, paper_metrics
from .test_engine import CFG, META, COINS, market, intent


class ForwardTests(unittest.TestCase):
    def test_arrival_needs_newly_received_book_and_missing_quote_expires_unfilled(self):
        p = broker(META, CFG['scenarios']['base']); manager = PairExecution(p, CFG)
        books, ctx = market(100000)
        for c, b in books.items(): p.depth.update(c, b['data'])
        manager.enter(intent(), 100000); manager.tick(100000, books, ctx)
        manager.tick(101000, books, ctx)
        self.assertFalse(p.fills); self.assertEqual(len(p.pending), 2)
        manager.tick(106001, books, ctx)
        self.assertFalse(p.fills); self.assertIsNone(manager.state)
        self.assertEqual(len(p.attempts), 2)
        self.assertTrue(all(a['status'] == 'unfilled_missing_arrival_observation' for a in p.attempts))

    def test_stale_or_crossed_book_cannot_propose(self):
        books, ctx = market(100000)
        self.assertEqual(len(current_snapshots(100000, books, ctx, CFG)), 2)
        self.assertFalse(current_snapshots(104000, books, ctx, CFG))
        books[COINS[0]]['data']['levels'][0][0]['px'] = '102'
        self.assertNotIn(COINS[0], current_snapshots(100000, books, ctx, CFG))

    def test_score_uses_12utc_completed_returns_and_detects_missing_coverage(self):
        start = 43200000; end = start + 2*86400000
        series = [dict(time=start, mark_equity=1e6, oracle_equity=1e6),
            dict(time=start+86400000, mark_equity=1001000., oracle_equity=1e6),
            dict(time=end, mark_equity=1000500., oracle_equity=1e6)]
        p = SimpleNamespace(equity_series=series, fills=[dict(time=start+1000)], fatal=[])
        out = paper_metrics(p, start, end)
        returns = np.array([.001, 1000500./1001000.-1])
        expected = (.0005+.005)/(.0005+1.5*np.std(returns, ddof=1)*np.sqrt(2)+.05)
        self.assertAlmostEqual(out['mark_equity']['financial_score'], expected)
        self.assertTrue(p.fatal)  # Sparse daily endpoints are not a valid full risk path.

    def test_close_waits_for_inflight_fill_and_then_reduces_it(self):
        p = broker(META, CFG['scenarios']['base']); manager = PairExecution(p, CFG)
        def step(now):
            b, c = market(now)
            for coin, book in b.items(): p.depth.update(coin, book['data'])
            manager.tick(now, b, c)
        manager.enter(intent(), 100000); step(100000)
        manager.close(100500, 'risk'); self.assertEqual(len(p.pending), 2)
        step(101000); step(102000)
        self.assertIsNone(manager.state)
        self.assertTrue(all(q == 0. for q in p.qty.values()))
        self.assertEqual(len(p.fills), 4)
        self.assertTrue(all(f['phase'] == 'reduce' for f in p.fills[-2:]))


if __name__ == '__main__': unittest.main()
