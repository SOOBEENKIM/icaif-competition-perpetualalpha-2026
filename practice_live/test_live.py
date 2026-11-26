"""Execution integration tests with a fake venue; never connect or sign."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from bot.execution import OutcomeUnknown, OrderRejected
from .live_logic import accounting, make_ioc, phase_delta
from .live_store import Store
from .runner import COINS, Runner, paged


def order_fixture(delta=1., reduce=False, slot="test"):
    market = {"szDecimals": 4}
    context = {"markPx": "100"}
    book = {"coin": "par:BTC", "time": 1000000,
            "levels": [[{"px": "99.99", "sz": "0.25"}],
                       [{"px": "100.01", "sz": "0.25"}, {"px": "101", "sz": "99"}]]}
    active = {"coin": "par:BTC", "maxTradeSzs": ["2", "2"]}
    return make_ioc("par:BTC", delta, reduce, market, context, book, active, 1000000, slot, 150000)


class LogicTests(unittest.TestCase):
    def test_actual_book_depth_and_limit_prevent_sweeping_far_levels(self):
        order = order_fixture()
        self.assertEqual(order.size, "0.2500")
        self.assertEqual(order.tif, "Ioc")
        self.assertLess(float(order.price), 101)

    def test_position_reversal_requires_verified_flat_before_increase(self):
        self.assertEqual(phase_delta(2, -3, "reduce"), -2)
        self.assertEqual(phase_delta(2, -3, "increase"), 0)
        self.assertEqual(phase_delta(.2, -3, "increase"), 0)
        self.assertEqual(phase_delta(0, -3, "increase"), -3)

    def test_reduction_cannot_grow_position(self):
        self.assertEqual(phase_delta(2, 3, "reduce"), 0)
        self.assertEqual(phase_delta(-2, -1, "reduce"), 1)
        self.assertTrue(order_fixture(-1, True).reduce_only)

    def test_cloid_is_stable_for_same_round_coin_phase(self):
        self.assertEqual(order_fixture(slot="a").cloid, order_fixture(slot="a").cloid)
        self.assertNotEqual(order_fixture(slot="a").cloid, order_fixture(slot="b").cloid)

    def test_flat_start_equity_includes_fees_funding_and_mark_without_double_pnl(self):
        fills = [{"coin": "par:BTC", "side": "B", "sz": "2", "px": "100", "fee": "1", "feeToken": "USDPR", "closedPnl": "0"},
                 {"coin": "par:BTC", "side": "A", "sz": "1", "px": "110", "fee": ".5", "feeToken": "USDPR", "closedPnl": "10"}]
        result = accounting({"initial_equity": 1000}, fills,
                 [{"delta": {"coin": "par:BTC", "usdc": "-.2"}}], [],
                 [{"position": {"coin": "par:BTC", "szi": "1"}}], {"par:BTC": 120.}, "team")
        self.assertAlmostEqual(result["equity"], 1028.3)
        self.assertEqual(result["qty"]["par:BTC"], 1)
        self.assertEqual(result["reported_closed_pnl"], 10)

    def test_missing_fill_stops_sizing_from_wrong_account(self):
        with self.assertRaisesRegex(ValueError, "reconciliation"):
            accounting({"initial_equity": 1000}, [], [], [],
                       [{"position": {"coin": "par:BTC", "szi": "1"}}], {"par:BTC": 100}, "team")

    def test_deposit_increases_equity_but_not_trading_return(self):
        result = accounting({"initial_equity": 1000}, [], [],
              [{"delta": {"token": "USDPR", "user": "other", "destination": "team", "amount": "500"}}],
              [], {"par:BTC": 100}, "team")
        self.assertEqual(result["equity"], 1500)
        self.assertEqual(result["return_on_initial_percent"], 0)

    def test_saturated_same_millisecond_pagination_fails_instead_of_dropping_events(self):
        info = Mock()
        info.post.return_value = [{"time": 100, "id": 1}, {"time": 100, "id": 2}]
        with self.assertRaisesRegex(ValueError, "saturated"):
            paged(info, {"type": "userFunding", "startTime": 100}, 200, 2)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.store = Store(self.path / "state.sqlite3")
        self.addCleanup(lambda: self.store.close())
        self.client = Mock()
        self.client.orders_enabled = True
        self.runner = Runner(self.client, {"team_wallet": "team"}, self.store, self.path, execute=True)

    def test_claim_survives_restart_and_blocks_duplicate_submission(self):
        order = asdict(order_fixture())
        self.assertTrue(self.store.claim("round", order))
        self.store.close()
        self.store = Store(self.path / "state.sqlite3")
        self.assertFalse(self.store.claim("round", order))
        self.assertEqual(self.store.orders()[0]["status"], "sending")

    def test_uncertain_submission_queries_existing_id_without_resending(self):
        order = asdict(order_fixture())
        self.store.claim("round", order)
        self.runner.executor.status = Mock(return_value={"status": "unknownOid"})
        self.runner.executor.submit = Mock()
        with self.assertRaises(OutcomeUnknown):
            self.runner.recover()
        self.runner.executor.submit.assert_not_called()
        self.assertEqual(self.store.orders()[0]["status"], "unknown")

    def test_recovered_partial_fill_remains_actual_venue_result(self):
        order = asdict(order_fixture())
        self.store.claim("round", order)
        result = {"status": "order", "order": {"status": "filled", "order": {
            "coin": "par:BTC", "oid": 12, "origSz": ".25", "sz": ".15"}}}
        self.runner.executor.status = Mock(return_value=result)
        self.runner.recover()
        self.assertEqual(self.store.orders()[0]["result"], result)

    def test_recovery_cancels_only_our_persisted_open_cloid(self):
        order = asdict(order_fixture())
        self.store.claim("round", order)
        self.runner.executor.status = Mock(return_value={"status": "order", "order": {
            "status": "open", "order": {"coin": "par:BTC", "oid": 12}}})
        self.runner.executor.cancel = Mock()
        with self.assertRaises(OutcomeUnknown):
            self.runner.recover()
        self.runner.executor.cancel.assert_called_once_with("par:BTC", order["cloid"])

    def test_history_repoll_does_not_duplicate_fees_or_fills(self):
        fill = {"coin": "par:BTC", "tid": 1, "fee": "3"}
        self.store.add_records("fills", [("par:BTC:1", fill)])
        self.store.add_records("fills", [("par:BTC:1", fill)])
        self.assertEqual(self.store.records("fills"), [fill])

    def fixtures(self, hour):
        from strategy.agent import TAIL, read
        protocol = read(TAIL / "baseline.json")["protocol"]
        inp = {"protocol": protocol, "weights": np.array([.04, 0, .005] + [0.]*7),
               "sigma": np.eye(10)*1e-6, "volume": np.ones(10)*1000, "trades": np.ones(10)*100,
               "hour": hour, "input_hash": "abc"}
        account = {"ledger": {"equity": 1e6, "external_capital": 0.}, "qty": np.zeros(10),
                   "prices": np.ones(10)*100,
                   "universe": {c: ({"szDecimals": 4}, {"markPx": "100"}) for c in COINS}}
        return account, inp

    def test_normal_clock_and_late_catchup_are_enforced(self):
        a, inputs = self.fixtures(8*3600)
        self.assertIsNone(self.runner.decide(a, inputs, 8*3600+120))
        a, inputs = self.fixtures(12*3600)
        self.assertIsNone(self.runner.decide(a, inputs, 12*3600+901))
        self.assertIsNotNone(self.runner.decide(a, inputs, 12*3600+120))

    def test_venue_price_used_for_quantity_without_rescaling_strategy(self):
        a, inputs = self.fixtures(12*3600)
        a["prices"][0] = 200
        plan = self.runner.decide(a, inputs, 12*3600+120)
        self.assertEqual(plan["target_qty"][0], 200.)  # 4% * 1M / actual par 200
        self.assertEqual(plan["target_qty"][2], 50.)

    def test_drawdown_halt_is_persistent_and_only_reduces(self):
        a, inputs = self.fixtures(8*3600)
        self.store.put("peak", 1e6)
        a["ledger"]["equity"] = 900000.
        a["qty"][0] = 100
        plan = self.runner.decide(a, inputs, 8*3600+120)
        self.assertTrue(plan["risk_forced"])
        self.assertTrue(self.store.get("halted"))
        self.assertEqual(plan["target_qty"], [0.]*10)

    def test_round_completion_is_not_replayed_on_restart(self):
        a, inputs = self.fixtures(12*3600)
        plan = self.runner.decide(a, inputs, 12*3600+120)
        plan["status"] = "complete"
        self.store.put(plan["key"], plan)
        self.assertIsNone(self.runner.decide(a, inputs, 12*3600+150))

    def execution_fixture(self):
        a, inputs = self.fixtures(12*3600)
        now = 12*3600+120
        plan = self.runner.decide(a, inputs, now)
        plan["target_qty"] = [1.] + [0.]*9
        self.runner.account = Mock(side_effect=lambda: a)
        self.client.info.post.return_value = {"coin": "par:BTC", "leverage": {"type": "cross", "value": 1},
                                             "maxTradeSzs": ["100", "100"]}
        self.client.info.l2_snapshot.return_value = {"coin": "par:BTC", "time": now*1000,
                "levels": [[{"px": "99.99", "sz": ".25"}], [{"px": "100.01", "sz": ".25"}]]}
        self.runner.executor.status = Mock(return_value={"status": "order", "order": {
            "status": "filled", "order": {"coin": "par:BTC", "oid": 42, "origSz": ".25", "sz": ".15"}}})
        return a, inputs, plan, now

    def test_complete_execution_path_records_partial_fill_without_resending_residual(self):
        a, inputs, plan, now = self.execution_fixture()
        def partial(order):
            a["qty"][0] = .1
            return {"filled": {"oid": 42, "totalSz": ".1", "avgPx": "100.01"}}
        self.runner.executor.submit = Mock(side_effect=partial)
        with patch("practice_live.runner.time.time", return_value=now):
            self.runner.execute_round(plan, inputs)
            self.runner.execute_round(plan, inputs)
        self.runner.executor.submit.assert_called_once()
        sent = self.runner.executor.submit.call_args.args[0]
        self.assertEqual(float(sent.size), .25)
        self.assertEqual(a["qty"][0], .1)
        self.assertEqual(self.store.orders()[0]["status"], "filled")
        self.assertEqual(self.store.get(plan["key"])["status"], "complete")
        self.client.set_expires_after.assert_called_once_with(now*1000+15000)

    def test_lost_submission_response_preserves_claim_and_stops_following_orders(self):
        a, inputs, plan, now = self.execution_fixture()
        self.runner.executor.submit = Mock(side_effect=OutcomeUnknown("lost response"))
        with patch("practice_live.runner.time.time", return_value=now):
            with self.assertRaises(OutcomeUnknown):
                self.runner.execute_round(plan, inputs)
        self.assertEqual(len(self.store.orders()), 1)
        self.assertEqual(self.store.orders()[0]["status"], "sending")
        self.runner.executor.submit.assert_called_once()

    def test_rejected_order_is_not_counted_as_fill_or_retried(self):
        a, inputs, plan, now = self.execution_fixture()
        self.runner.executor.submit = Mock(side_effect=OrderRejected("No liquidity"))
        with patch("practice_live.runner.time.time", return_value=now):
            self.runner.execute_round(plan, inputs)
            self.runner.execute_round(plan, inputs)
        self.runner.executor.submit.assert_called_once()
        self.assertEqual(self.store.orders()[0]["status"], "rejected")
        self.assertEqual(a["qty"][0], 0)


if __name__ == "__main__":
    unittest.main()
