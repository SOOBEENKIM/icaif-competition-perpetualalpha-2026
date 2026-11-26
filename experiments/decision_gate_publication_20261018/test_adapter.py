import copy
import json
import time
import unittest
from pathlib import Path

import numpy as np

from .signals import DecisionGateSignals
from .portfolio import Portfolio
from experiments.published_pair_lifecycle_20261017.portfolio import Portfolio as OriginalPortfolio
from experiments.published_pair_lifecycle_20261017.test_signals import SignalTests, envelope
from experiments.published_pair_lifecycle_20261017.test_forward import ForwardTests, book_events
from experiments.published_pair_lifecycle_20261017.forward import TapeWindow
from experiments.published_pair_lifecycle_20261017.test_lifecycle import LifecycleTests

ROOT = Path(__file__).resolve().parents[2]


def plain(value):
    return json.loads(json.dumps(value, default=lambda a: a.tolist() if hasattr(a, 'tolist') else a.item()))


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SignalTests.setUpClass()
        cls.fixture = SignalTests()
        cls.spec = json.loads((ROOT / 'experiments/decision_gate_pairs_20261017/protocol.json').read_text())
        with np.load(ROOT / 'experiments/decision_gate_pairs_20261017/artifacts/reference_issued_forecasts.npz') as z:
            cls.batch = {key: z[key].copy() for key in z.files}
        cls.timings = []

    def instance(self):
        f = self.fixture
        return DecisionGateSignals(f.p['symbols'], f.bias['pairs'], f.models, f.spec, f.bias, self.spec)

    def test_received_prefix_reproduces_frozen_gate_and_means_without_score_substitution(self):
        model = self.instance(); f = self.fixture
        for source in [f.day, f.day + 3600, f.day + 7200]:
            start = time.perf_counter()
            rows = model.ingest(envelope(source), f.prefix(source))
            self.timings.append(time.perf_counter() - start)
            direct = {tuple(r['pair']): r for r in rows if r['model'] == 'direct'}
            gated = [r for r in rows if r['model'] == 'decision_gate']
            self.assertEqual(len(gated), 2)
            i = int(np.flatnonzero(self.batch['times'] == source)[0])
            for k, row in enumerate(gated):
                original = direct[tuple(row['pair'])]
                for key in original:
                    if key != 'model': self.assertEqual(row[key], original[key], key)
                self.assertEqual(row['economic_available'], bool(self.batch['economic_available'][i, k]))
                self.assertEqual(row['economic_action'], int(self.batch['economic_action'][i, k]))
                self.assertAlmostEqual(row['decision_score'], self.batch['decision_score'][i, k], places=11)
                self.assertEqual(row['gate_fit_source'], int(self.batch['fit_stamps'][i, k]))
                self.assertEqual(row['available_time'], source + 8)
        self.assertEqual(len(model.gate_audit), 1)
        self.assertTrue(all(v['last_maturity'] < f.day for v in model.gate_daily.values()))
        print('ADAPTER_INFERENCE_SECONDS', self.timings, flush=True)

    def test_missing_midnight_and_later_revision_cannot_backfill_gate(self):
        f = self.fixture; missing = self.instance()
        rows = missing.ingest(envelope(f.day + 3600), f.prefix(f.day + 3600))
        self.assertFalse(any(r['model'] == 'decision_gate' for r in rows))
        self.assertFalse(missing.gate_daily)
        model = self.instance(); model.ingest(envelope(f.day), f.prefix(f.day))
        before = copy.deepcopy(model.gate_daily)
        later = f.prefix(f.day + 3600); later['close'][-500:-300, 0] *= 1.2
        model.ingest(envelope(f.day + 3600, 'b' * 64), later)
        self.assertEqual(before, model.gate_daily)
        self.assertEqual(len(model.gate_audit), 1)

    def test_unavailable_gate_keeps_direct_holding_forecast(self):
        f = self.fixture; model = self.instance()
        model.gate_spec['minimum_training_rows'] = 100000
        rows = model.ingest(envelope(f.day), f.prefix(f.day))
        direct = {tuple(r['pair']): r for r in rows if r['model'] == 'direct'}
        gated = [r for r in rows if r['model'] == 'decision_gate']
        self.assertEqual(len(gated), 2)
        for row in gated:
            self.assertFalse(row['economic_available'])
            self.assertIsNone(row['decision_score'])
            self.assertEqual(row['terminal_prices'], direct[tuple(row['pair'])]['terminal_prices'])

    def test_duplicates_revisions_and_future_snapshots(self):
        f = self.fixture; model = self.instance(); event = envelope(f.day)
        rows = model.ingest(event, f.prefix(f.day))
        self.assertEqual(rows, model.ingest(event, f.prefix(f.day)))
        self.assertEqual(len(model.gate_audit), 1)
        with self.assertRaisesRegex(ValueError, 'revised'):
            model.ingest(envelope(f.day, 'b' * 64), f.prefix(f.day))
        with self.assertRaisesRegex(ValueError, 'Out-of-order'):
            model.ingest(envelope(f.day - 3600), f.prefix(f.day - 3600))
        with self.assertRaisesRegex(ValueError, 'end exactly'):
            self.instance().ingest(event, f.prefix(f.day + 3600))


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.f = ForwardTests(); self.f.setUp()
        self.zero = LifecycleTests(); self.zero.setUp()
        self.streams = TapeWindow(book_events(self.f.p['symbols'], range(900, 912))).ensure(900000)
        self.publication = self.f.publication(0, [0.] * 10)
        self.forecast = dict(model='direct', pair=['ETH', 'BTC'], ids=[1, 0],
            source_beta=1., source_prices=[100., 100.], terminal_prices=[102., 98.],
            source_vol=.01, forecast_source=0, available_time=8, horizon=24)

    def arm(self, cls, name):
        # This routing test's synthetic market has known zero hourly funding.
        settlement = LifecycleTests(); settlement.setUp()
        arm = cls(name, 'base', self.f.plan, settlement.funding)
        arm.last_decision = 0
        arm.base_task = dict(target=np.array([10.] + [0.] * 9), expires=1200)
        return arm

    def test_native_and_ungated_accounts_exactly_match_original_with_shared_base_fills(self):
        for name in ['first_strategy', 'second_strategy', 'first_plus_direct']:
            original = self.arm(OriginalPortfolio, name); adapted = self.arm(Portfolio, name)
            for arm in [original, adapted]:
                arm.step(900, self.streams, self.publication, [self.forecast])
            self.assertEqual(plain(original.events), plain(adapted.events))
            self.assertEqual(plain(original.snapshots), plain(adapted.snapshots))
            self.assertEqual(original.account.physical.fills, adapted.account.physical.fills)
            self.assertEqual(original.account.physical.cash, adapted.account.physical.cash)
            np.testing.assert_array_equal(original.account.physical.qty, adapted.account.physical.qty)

    def test_allowed_direction_preserves_actual_entries_and_opposite_or_missing_blocks(self):
        ungated = self.arm(OriginalPortfolio, 'first_plus_direct')
        ungated.step(900, self.streams, self.publication, [self.forecast])
        self.assertEqual(ungated.pair.state, 'open')
        for available, action, expect_entry in [(True, 1, True), (True, -1, False), (True, 0, False), (False, 1, False)]:
            arm = self.arm(Portfolio, 'first_plus_decision_gate')
            forecast = dict(self.forecast, model='decision_gate', economic_available=available, economic_action=action)
            arm.step(900, self.streams, self.publication, [forecast])
            self.assertEqual(arm.pair is not None, expect_entry)
            if expect_entry:
                self.assertEqual(ungated.account.physical.fills, arm.account.physical.fills)
                self.assertEqual(arm.pair.state, 'open')
            else:
                self.assertFalse(np.any(arm.account.owners['pair'].qty))
                self.assertGreater(len(arm.account.owners['base'].fills), 0)

    def test_flat_gate_does_not_cancel_current_forecast_renewal(self):
        arm = self.arm(Portfolio, 'first_plus_decision_gate')
        initial = dict(self.forecast, model='decision_gate', economic_available=True, economic_action=1)
        arm.step(900, self.streams, self.publication, [initial])
        self.assertEqual(arm.pair.state, 'open')
        stop = arm.pair.stop_bound; old_maturity = arm.pair.maturity
        now = 18 * 3600 + 30
        streams = TapeWindow(book_events(self.f.p['symbols'], range(now-1, now+12))).ensure(now*1000)
        updated = dict(initial, forecast_source=18*3600, available_time=18*3600+8,
                       economic_available=False, economic_action=0)
        publication = self.f.publication(18*3600, [0.]*10)
        arm.last_decision = 18*3600
        arm.step(now, streams, publication, [updated])
        self.assertEqual(arm.pair.state, 'open')
        self.assertGreater(arm.pair.maturity, old_maturity)
        self.assertEqual(arm.pair.maturity, 42*3600)
        self.assertEqual(arm.pair.stop_bound, stop)

    def test_flat_gate_does_not_block_paid_terminal_exit(self):
        arm = self.arm(Portfolio, 'first_plus_decision_gate')
        initial = dict(self.forecast, model='decision_gate', economic_available=True, economic_action=1)
        arm.step(900, self.streams, self.publication, [initial])
        self.assertEqual(arm.pair.state, 'open'); fees = arm.account.physical.fees
        now = 1800; arm.plan = dict(arm.plan, capture_end_unix=2400)
        streams = TapeWindow(book_events(self.f.p['symbols'], range(now-1, now+12))).ensure(now*1000)
        arm.step(now, streams, self.publication, [dict(initial, economic_action=0)])
        self.assertFalse(np.any(arm.account.physical.qty))
        self.assertGreater(arm.account.physical.fees, fees)
        self.assertEqual(arm.pair.state, 'closed')

    def test_forward_release_guard_precedes_file_or_network_access(self):
        from unittest.mock import patch
        from .forward import ready
        with patch('pathlib.Path.read_bytes', side_effect=AssertionError('must not read outcomes')):
            with self.assertRaisesRegex(ValueError, 'remain sealed'):
                ready({'outcome_release_unix': 1000}, 999)


if __name__ == '__main__':
    unittest.main()
