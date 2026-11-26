"""Data-boundary regression tests, all synthetic and offline."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from .refresh import BAR_FIELDS, HOUR, REFERENCES, append_inputs, parse_symbol, public_info


class RefreshTests(unittest.TestCase):
    def setUp(self):
        self.previous, self.target = 200 * HOUR, 202 * HOUR
        known = np.arange(HOUR, self.previous + HOUR, HOUR, dtype=np.int64)
        shape = (len(known), 10)
        self.old = {key: np.full(shape, 100.) for key in BAR_FIELDS}
        self.old.update(times=known-HOUR, known=known, market_times=known,
                        returns=np.zeros(shape), sigma=np.full(shape, 1e-6),
                        funding=np.zeros(shape), observed=np.ones(shape, dtype=bool))
        self.receipts = {}
        for symbol, coin in REFERENCES.items():
            candles = [{"s": coin, "i": "1h", "t": (t-HOUR)*1000, "T": t*1000-1,
                        "o": "100", "h": "100", "l": "100", "c": "100", "v": "100", "n": 100}
                       for t in range(self.previous, self.target + HOUR, HOUR)]
            funding = [{"coin": coin, "time": t*1000, "fundingRate": "0"}
                       for t in range(self.previous, self.target + HOUR, HOUR)]
            self.receipts[symbol] = {"symbol": symbol, "reference_coin": coin,
                                     "candles": {"response": candles},
                                     "funding": [{"response": funding}]}

    def append(self):
        return append_inputs(self.old, self.receipts, self.target)

    def test_preserves_all_old_fields_and_observed_zero_funding(self):
        merged, audit = self.append()
        for key, value in self.old.items():
            np.testing.assert_array_equal(merged[key][:len(value)], value)
        self.assertEqual(len(merged["known"]), 202)
        self.assertTrue(merged["observed"].all())
        self.assertEqual(audit["BTC"]["new_hours"], 2)
        np.testing.assert_array_equal(merged["returns"], 0.)

    def test_missing_candle_is_rejected(self):
        self.receipts["BTC"]["candles"]["response"].pop(1)
        with self.assertRaisesRegex(ValueError, "Missing BTC candles"):
            self.append()

    def test_missing_funding_is_not_imputed_as_zero(self):
        self.receipts["ETH"]["funding"][0]["response"].pop()
        with self.assertRaisesRegex(ValueError, "Missing ETH funding"):
            self.append()

    def test_unfinished_candle_is_excluded(self):
        rows = self.receipts["BTC"]["candles"]["response"]
        future = dict(rows[-1], t=self.target*1000, T=(self.target+HOUR)*1000-1, c="999")
        rows.append(future)
        merged, audit = self.append()
        self.assertEqual(merged["close"][-1, 0], 100.)
        self.assertEqual(audit["BTC"]["excluded"]["unfinished_candles"], 1)

    def test_conflicting_duplicate_candle_rejected(self):
        rows = self.receipts["BTC"]["candles"]["response"]
        rows.append(dict(rows[-1], v="101"))
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            self.append()

    def test_identical_duplicate_and_subsecond_funding_are_deduplicated(self):
        rows = self.receipts["BTC"]["funding"][0]["response"]
        rows.append(dict(rows[-1], time=rows[-1]["time"] + 123))
        self.append()

    def test_late_funding_cannot_be_shifted_into_earlier_hour(self):
        self.receipts["BTC"]["funding"][0]["response"][-1]["time"] += 1000
        with self.assertRaisesRegex(ValueError, "subsecond tolerance"):
            self.append()

    def test_conflicting_duplicate_funding_rejected(self):
        rows = self.receipts["BTC"]["funding"][0]["response"]
        rows.append(dict(rows[-1], fundingRate="0.001"))
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            self.append()

    def test_overlap_revision_does_not_overwrite_history(self):
        original = copy.deepcopy(self.old)
        self.receipts["BTC"]["candles"]["response"][0]["v"] = "101"
        with self.assertRaisesRegex(ValueError, "Source overlap changed"):
            self.append()
        for key in original:
            np.testing.assert_array_equal(self.old[key], original[key])

    def test_wrong_market_rejected(self):
        self.receipts["BTC"]["candles"]["response"][-1]["s"] = "par:BTC"
        with self.assertRaisesRegex(ValueError, "Wrong reference candle"):
            self.append()

    def test_repeat_same_hour_has_no_duplicate_append(self):
        merged, _ = self.append()
        repeated, _ = append_inputs(merged, self.receipts, self.target)
        for key in merged:
            np.testing.assert_array_equal(repeated[key], merged[key])

    def test_invalid_ohlc_rejected(self):
        self.receipts["BTC"]["candles"]["response"][-1]["h"] = "99"
        with self.assertRaisesRegex(ValueError, "Invalid OHLC"):
            self.append()

    def test_misaligned_candle_timestamp_rejected(self):
        self.receipts["BTC"]["candles"]["response"][-1]["t"] += 1
        with self.assertRaisesRegex(ValueError, "Invalid candle timestamp"):
            self.append()

    def test_account_order_and_nonreference_queries_are_rejected_before_network(self):
        with patch("urllib.request.urlopen") as network:
            for payload in ({"type": "order"}, {"type": "clearinghouseState", "user": "0x0"},
                            {"type": "fundingHistory", "coin": "par:BTC", "startTime": 0, "endTime": 1}):
                with self.assertRaises(ValueError):
                    public_info(payload)
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
