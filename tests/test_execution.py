"""Order validation, exchange response handling and explicit submission gates."""

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

from requests import Timeout
from hyperliquid.utils.constants import TESTNET_API_URL

from bot.connection import SetupError, TradingExchange
from bot.execution import Executor, OrderRejected, OutcomeUnknown, prepare_limit


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client = Mock()
        self.client.orders_enabled = True
        self.client.info.coin_to_asset = {"par:ETH": 2670001}
        self.client.info.post.return_value = [
            {"universe": [{"name": "par:ETH", "szDecimals": 4, "maxLeverage": 25}]},
            [{"markPx": "3000"}],
        ]
        self.settings = {"team_wallet": "0x" + "2" * 40}
        self.executor = Executor(self.client, self.settings, journal=Path(self.temp.name) / "events.jsonl")
        self.order = self.executor.prepare("par:ETH", True, "0.0041", "2970")

    def test_price_size_rounding_and_integer_price_exception(self):
        order = prepare_limit("par:ETH", True, "0.004199", "2999.99", sz_decimals=4, mark_price="3000")
        self.assertEqual(order.size, "0.0041")
        self.assertEqual(order.price, "2999.9")
        sell = prepare_limit("par:ETH", False, "0.0041", "3000.01", sz_decimals=4, mark_price="3000")
        self.assertEqual(sell.price, "3000.1")
        btc = prepare_limit("par:BTC", True, "0.0001", "100001", sz_decimals=4, mark_price="100000")
        self.assertEqual(btc.price, "100001")
        self.assertEqual(btc.notional, Decimal("10.0001"))

    def test_invalid_inputs_and_small_or_large_orders_are_blocked(self):
        for coin, size, price in (("ETH", "0.004", "3000"), ("par:ETH", "NaN", "3000"),
                                 ("par:ETH", "0.0001", "3000"), ("par:ETH", "1", "3000"),
                                 ("par:ETH", "0.004", "2700")):
            with self.subTest(coin=coin, size=size, price=price):
                with self.assertRaises(SetupError):
                    prepare_limit(coin, True, size, price, sz_decimals=4, mark_price="3000")

    def test_read_mode_never_calls_order_or_cancel(self):
        self.client.orders_enabled = False
        with self.assertRaises(SetupError):
            self.executor.submit(self.order)
        with self.assertRaises(SetupError):
            self.executor.cancel(self.order.coin, self.order.cloid)
        self.client.order.assert_not_called()
        self.client.cancel_by_cloid.assert_not_called()

    def test_outer_ok_with_inner_error_is_rejected(self):
        self.client.order.return_value = {"status": "ok", "response": {
            "type": "order", "data": {"statuses": [{"error": "Insufficient margin"}]}}}
        with self.assertRaises(OrderRejected):
            self.executor.submit(self.order)

    def test_timeout_records_identity_and_never_retries_order(self):
        self.client.order.side_effect = Timeout("simulated lost response")
        with self.assertRaises(OutcomeUnknown) as result:
            self.executor.submit(self.order)
        self.assertIn(self.order.cloid, str(result.exception))
        self.assertEqual(self.client.order.call_count, 1)
        events = [json.loads(line) for line in self.executor.journal.read_text().splitlines()]
        self.assertEqual(events[0]["order"]["cloid"], self.order.cloid)
        self.assertEqual(events[-1]["event"], "submit_unknown")

    def test_partial_fill_is_preserved_as_reported(self):
        partial = {"filled": {"oid": 123, "totalSz": "0.002", "avgPx": "2970"}}
        self.client.order.return_value = {"status": "ok", "response": {
            "type": "order", "data": {"statuses": [partial]}}}
        self.assertEqual(self.executor.submit(self.order), partial)

    def test_cancel_targets_only_given_coin_and_client_order_id(self):
        self.client.cancel_by_cloid.return_value = {"status": "ok", "response": {
            "type": "cancel", "data": {"statuses": ["success"]}}}
        self.executor.cancel(self.order.coin, self.order.cloid)
        args = self.client.cancel_by_cloid.call_args.args
        self.assertEqual(args[0], "par:ETH")
        self.assertEqual(args[1].to_raw(), self.order.cloid)

    def test_reconciliation_uses_team_address_and_par_dex(self):
        self.executor.status(self.order.cloid)
        self.assertEqual(self.client.info.query_order_by_cloid.call_args.args[0], self.settings["team_wallet"])
        self.client.info.user_state.return_value = {"assetPositions": []}
        self.client.info.open_orders.return_value = []
        self.assertEqual(self.executor.account_state(), {"positions": [], "open_orders": []})
        self.client.info.user_state.assert_called_with(self.settings["team_wallet"], dex="par")
        self.client.info.open_orders.assert_called_with(self.settings["team_wallet"], dex="par")

    def test_signature_only_client_blocks_orders_and_account_administration(self):
        client = object.__new__(TradingExchange)
        client.base_url = TESTNET_API_URL
        client.orders_enabled = False
        with patch("hyperliquid.api.API.post") as network:
            client.post("/exchange", {"action": {"type": "noop"}})
            self.assertEqual(network.call_count, 1)
            for kind in ("order", "cancel", "updateLeverage", "usdSend", "approveAgent"):
                with self.assertRaises(SetupError):
                    client.post("/exchange", {"action": {"type": kind}})
            self.assertEqual(network.call_count, 1)
            client.orders_enabled = True
            client.post("/exchange", {"action": {"type": "order"}})
            self.assertEqual(network.call_count, 2)
            client.base_url = "https://api.hyperliquid.xyz"
            with self.assertRaises(SetupError):
                client.post("/exchange", {"action": {"type": "order"}})
            self.assertEqual(network.call_count, 2)


if __name__ == "__main__":
    unittest.main()
