import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from candidate_handoff import test_risk_priority as fixtures
from competition_live.live_account import LiveAccount
from experiments.basis_carry_20261017.probe import asof
from production_replay_20261025.quote_guard import QuoteAccount
from production_replay_20261025.quote_history import quote_history, filter_target
from production_replay_20261025 import test_quote_history as history_fixtures
from .adapter import attach_history, prepare_unsigned, wire_staged


class StagingTests(unittest.TestCase):
    cfg = dict(book_max_age_ms=15000, context_max_age_ms=3000)

    def streams(self):
        archive = history_fixtures.QuoteHistoryTests().stream()
        now = 14410
        at = int(now * 1000)
        live = dict(BTC=dict(book=[(at, 0, dict(time=at, levels=[
            [dict(px='98', sz='10')], [dict(px='100', sz='10')]]))],
            book_times=[at], ctx=[(at, 0, dict(markPx='99', oraclePx='99'))],
            ctx_times=[at], resets=[]))
        return live, archive, now

    def test_same_feature_and_execution_quote_without_mutation(self):
        live, archive, now = self.streams()
        before = copy.deepcopy((live, archive))
        joined, _ = attach_history(live, archive, ['BTC'], 14400, now, self.cfg)
        p1, _ = quote_history(archive, ['BTC'], 14400, self.cfg)
        p2, _ = quote_history(joined, ['BTC'], 14400, self.cfg)
        np.testing.assert_array_equal(p1, p2)
        self.assertEqual(asof(live['BTC'], now * 1000, self.cfg),
                         asof(joined['BTC'], now * 1000, self.cfg))
        self.assertEqual((live, archive), before)

    def test_future_archive_cannot_change_historical_features(self):
        live, archive, now = self.streams()
        changed = copy.deepcopy(archive)
        for field in ('book', 'ctx'):
            for stamp in (14400001, 14420000):
                value = copy.deepcopy(changed['BTC'][field][-1][2])
                if field == 'book':
                    value['time'] = stamp
                    value['levels'][0][0]['px'] = '1000'
                    value['levels'][1][0]['px'] = '1001'
                changed['BTC'][field].append((stamp, 1, value))
                changed['BTC'][field + '_times'].append(stamp)
        a, _ = attach_history(live, archive, ['BTC'], 14400, now, self.cfg)
        b, _ = attach_history(live, changed, ['BTC'], 14400, now, self.cfg)
        self.assertEqual(a, b)

    def test_missing_archive_blocks_additions_but_not_reductions(self):
        live, _, now = self.streams()
        joined, evidence = attach_history(live, {}, ['BTC'], 14400, now, self.cfg)
        prices, _ = quote_history(joined, ['BTC'], 14400, self.cfg)
        self.assertTrue(np.isnan(prices).all())
        self.assertFalse(any(x['available'] for x in evidence))
        self.assertEqual(filter_target([2.], [1.], prices)[0].tolist(), [1.])
        self.assertEqual(filter_target([.5], [1.], prices)[0].tolist(), [.5])

    def test_reconnect_boundary_cannot_reuse_old_feature_book(self):
        live, archive, now = self.streams()
        archive['BTC']['resets'] = [14399500]
        joined, evidence = attach_history(live, archive, ['BTC'], 14400, now, self.cfg)
        self.assertFalse(evidence[-1]['available'])
        self.assertIsNone(asof(joined['BTC'], 14400000, self.cfg))

    def test_future_or_incomplete_current_universe_rejected(self):
        live, archive, now = self.streams()
        with self.assertRaises(ValueError):
            attach_history(live, archive, ['BTC'], 18000, now, self.cfg)
        with self.assertRaises(ValueError):
            attach_history(live, archive, ['BTC', 'ETH'], 14400, now, self.cfg)

    def unsigned(self):
        p, *_ = fixtures.RiskPriorityTests().setup_case()
        p.settlement = None
        for key in ('executor', 'checkpoint', 'collect_fills', 'refresh_market'):
            setattr(p.account, key, None)
        return p

    def test_preserve_inventory_and_state_but_drop_only_unsubmitted_task(self):
        seed = self.unsigned()
        seed.base_task = dict(target=np.ones(10), expires=100)
        p = prepare_unsigned(seed)
        self.assertIsNone(p.base_task)
        self.assertIsNotNone(seed.base_task)
        for name in ('physical', 'reserved'):
            np.testing.assert_array_equal(getattr(p.account, name).qty,
                                          getattr(seed.account, name).qty)
        self.assertEqual(p.last_decision, seed.last_decision)
        self.assertEqual(p.peak, seed.peak)
        self.assertEqual(p.used, seed.used)
        self.assertIsInstance(p.account, QuoteAccount)

    def test_connected_or_unresolved_or_active_pair_cannot_be_migrated(self):
        seed = self.unsigned()
        seed.account.executor = object()
        with self.assertRaisesRegex(ValueError, 'sanitized'):
            prepare_unsigned(seed)
        seed.account.executor = None
        seed.account.unknown.add('unresolved-order-1')
        with self.assertRaisesRegex(ValueError, 'Unresolved'):
            prepare_unsigned(seed)
        seed.account.unknown.clear()
        seed.pair = object()
        with self.assertRaisesRegex(ValueError, 'Active pair'):
            prepare_unsigned(seed)

    def test_reconnection_keeps_guard_and_first_decision_price(self):
        p = prepare_unsigned(self.unsigned())
        p.base_task = dict(frozen_side_prices=np.array([[99., 101.]] * 10),
                           intent_source=14400, expires=18000)
        before = copy.deepcopy(p.base_task)
        bridge = SimpleNamespace(settlement=lambda *args: None)
        for _ in range(2):
            wire_staged(p, bridge, object(), lambda *a: None,
                        lambda *a: [], lambda *a: {})
            self.assertIsInstance(p.account, QuoteAccount)
            np.testing.assert_array_equal(p.base_task['frozen_side_prices'], before['frozen_side_prices'])
            p.account.frozen_side_prices = p.base_task['frozen_side_prices']
            held = p.account.owners['base'].qty
            target = held.copy(); target[0] += 1.
            raw = dict(packet_id='base:1', owner_targets={'base': target})
            with patch.object(LiveAccount, 'seal', side_effect=lambda r, *a: r):
                intent = p.account.seal(raw, {}, {})
            self.assertAlmostEqual(intent['owner_limits']['base']['0'], 101.101)


if __name__ == '__main__':
    unittest.main()
