"""Public market observations shared by strategy research and paper operation.

No account configuration, credentials, report paths or fitted parameters live here.
All timestamps are local receipt milliseconds; source clocks only check freshness.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path

SYMBOLS = ("BTC", "ETH", "HYPE", "SP500", "EWY", "EWJ", "BRENT", "GOLD", "USDEUR", "USDJPY")


@dataclass(frozen=True)
class Snapshot:
    time_ms: int
    coin: str
    book_ms: int
    context_ms: int
    bids: tuple
    asks: tuple
    oracle: float
    mark: float
    funding: float
    decimals: int

    @property
    def mid(self):
        return (self.bids[0][0] + self.asks[0][0]) / 2

    @property
    def spread_bps(self):
        return (self.asks[0][0] - self.bids[0][0]) / self.mid * 10000

    @property
    def basis_bps(self):
        return (self.oracle / self.mid - 1) * 10000

    def fresh(self, now, maximum_age_ms=5000):
        return (0 <= now - self.time_ms <= maximum_age_ms
                and 0 <= now - self.book_ms <= maximum_age_ms
                and 0 <= now - self.context_ms <= maximum_age_ms)


def snapshot(coin, book, context, receipt_ms, context_ms, decimals):
    if book.get("coin") != "par:" + coin:
        raise ValueError("Wrong venue or asset")
    levels = [tuple((float(x["px"]), float(x["sz"])) for x in side) for side in book["levels"]]
    if len(levels) != 2 or not all(levels):
        return None
    if any(not math.isfinite(v) or v <= 0 for side in levels for pair in side for v in pair):
        return None
    bids, asks = levels
    if bids[0][0] >= asks[0][0] or list(bids) != sorted(bids, reverse=True) or list(asks) != sorted(asks):
        return None
    oracle, mark, funding = (float(context[k]) for k in ("oraclePx", "markPx", "funding"))
    if not all(math.isfinite(v) for v in (oracle, mark, funding)) or min(oracle, mark) <= 0:
        return None
    result = Snapshot(receipt_ms, coin, int(book["time"]), context_ms, bids, asks,
                      oracle, mark, funding, decimals)
    return result if result.fresh(receipt_ms) else None


def lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def rest_snapshots(paths):
    for path in paths:
        for row in lines(path):
            if row["network"] != "testnet" or row["dex"] != "par":
                raise ValueError("Only competition venue recordings are supported")
            t = round(row["received_at"] * 1000)
            for coin in SYMBOLS:
                item = row["markets"][coin]
                raw = row.get("books", {}).get(coin, {}).get("book")
                if raw:
                    s = snapshot(coin, raw, item["context"], t, t, item["spec"]["szDecimals"])
                    if s:
                        yield s


def ws_snapshots(path, metadata):
    contexts = {}
    for event in lines(path):
        if event["venue"] != "par":
            continue
        t, channel, data = event["received_ms"], event["channel"], event.get("data", {})
        if channel in ("disconnected", "connected", "connection_error"):
            contexts.clear()
        elif channel == "activeAssetCtx":
            coin = data["coin"].removeprefix("par:")
            contexts[coin] = (t, data["ctx"])
        elif channel == "l2Book":
            coin = data["coin"].removeprefix("par:")
            if coin not in metadata or coin not in contexts:
                continue
            ct, ctx = contexts[coin]
            s = snapshot(coin, data, ctx, t, ct, metadata[coin])
            if s:
                yield s


class History:
    """The same past-only feature builder runs at both recording cadences."""
    def __init__(self):
        self.rows = []
        self.times = []

    def add(self, row):
        if self.times and row.time_ms < self.times[-1]:
            raise ValueError("Nonmonotonic history")
        self.rows.append(row)
        self.times.append(row.time_ms)

    def at(self, t, tolerance_ms=75000):
        i = bisect_right(self.times, t) - 1
        return self.rows[i] if i >= 0 and t - self.times[i] <= tolerance_ms else None

    def features(self, current, current_only=False):
        if current_only:
            return [current.basis_bps, 0., current.spread_bps] + [0.]*8 + [current.funding*10000]
        past = [self.at(current.time_ms - m * 60000) for m in (1, 5, 15, 60)]
        if any(r is None for r in past):
            return None  # readiness is required, never zero-imputed
        lookback = self.rows[bisect_right(self.times, current.time_ms - 3600000):]
        if len(lookback) < 30:
            return None
        # Sample each calendar minute once so REST and WS carry equal weight.
        minute = {}
        for r in lookback:
            minute[r.time_ms // 60000] = r.basis_bps
        baseline = sum(minute.values()) / len(minute)
        x = [current.basis_bps, current.basis_bps - baseline, current.spread_bps]
        x += [(current.oracle / r.oracle - 1) * 10000 for r in past]
        x += [(current.mid / r.mid - 1) * 10000 for r in past]
        x += [current.funding * 10000]
        return x

    def prune(self, now):
        i = bisect_right(self.times, now - 3700000)
        if i:
            del self.times[:i]
            del self.rows[:i]
