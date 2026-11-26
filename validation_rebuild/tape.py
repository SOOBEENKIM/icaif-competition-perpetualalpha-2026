"""Receipt-ordered tape with bounded fill-arrival lookahead and causal reads."""
import gzip
import json
from collections import deque
from pathlib import Path
import numpy as np
from experiments.basis_carry_20261017.probe import asof


def raw_events(directory):
    # Collector serializes timestamps and writes under a single lock.
    last = -1
    for path in sorted(Path(directory).glob("*.jsonl.gz")):
        with gzip.open(path, "rt") as f:
            for line in f:
                event = json.loads(line)
                if event["received_ms"] < last:
                    raise ValueError("Out-of-order capture")
                last = event["received_ms"]
                yield event


class TapeWindow:
    def __init__(self, events):
        self.events = iter(events); self.next = next(self.events, None)
        self.data = {}; self.signals = deque(); self.streams = {}
        self.resets = deque(); self.epoch = 0; self.last_seen = -1

    def ensure(self, now_ms, lookahead_ms=10000):
        until = now_ms+lookahead_ms
        while self.next is not None and self.next["received_ms"] <= until:
            e = self.next; at = e["received_ms"]
            if at < self.last_seen:
                raise ValueError("Out-of-order capture")
            self.last_seen = at
            if e["channel"] == "strategySignal":
                self.signals.append(e)
            elif e["channel"] == "connection":
                self.epoch += 1; self.resets.append(at)
            elif e["channel"] in ("l2Book", "activeAssetCtx"):
                symbol = e["data"]["coin"].split(":")[-1]
                fields = self.data.setdefault(symbol, {"book": deque(), "ctx": deque()})
                field = "book" if e["channel"] == "l2Book" else "ctx"
                fields[field].append((at, self.epoch,
                                      e["data"] if field == "book" else e["data"]["ctx"]))
                for rows in fields.values():
                    while len(rows) > 1 and rows[1][0] < at-30000:
                        rows.popleft()
            self.next = next(self.events, None)
        while len(self.resets) > 1 and self.resets[1] < now_ms-30000:
            self.resets.popleft()
        self.streams = {}
        for symbol, fields in self.data.items():
            result = {"resets": list(self.resets)}
            for key, rows in fields.items():
                while len(rows) > 1 and rows[1][0] < now_ms-20000:
                    rows.popleft()
                result[key] = list(rows)
                result[key+"_times"] = [r[0] for r in rows]
            self.streams[symbol] = result
        return self.streams

    def publications(self, now_ms):
        while self.signals and self.signals[0]["received_ms"] <= now_ms:
            yield self.signals.popleft()


class FundingClock:
    def __init__(self, start, tape, rates, symbols, cfg, receipt_multiplier):
        self.last = start*1000; self.next = (start//3600+1)*3600000
        self.tape, self.rates, self.symbols, self.cfg = tape, rates, symbols, cfg
        self.receipt_multiplier = receipt_multiplier

    def __call__(self, at, account):
        if at < self.last:
            raise ValueError("Settlement clock moved backwards")
        while self.next <= at:
            bucket = self.next
            prices = np.ones(len(self.symbols)); rates = np.zeros(len(prices))
            observed = np.zeros(len(prices), bool)
            for j, symbol in enumerate(self.symbols):
                stream = self.tape.streams.get(symbol)
                prior = asof(stream, bucket-1, self.cfg) if stream else None
                rate = self.rates.get(bucket, {}).get(symbol)
                if prior is not None and rate is not None:
                    prices[j], rates[j], observed[j] = prior["oracle"], rate, True
                if any(abs(o.qty[j]) > 1e-12 for o in account.owners.values()) and not observed[j]:
                    raise ValueError(f"Missing settled funding/oracle: {symbol} {bucket}")
            cash = -account.physical.qty*prices*rates
            rates = np.where(cash > 0, rates*self.receipt_multiplier, rates)
            account.charge_funding(bucket/1000, prices, rates, observed)
            self.next += 3600000
        self.last = at
