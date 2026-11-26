"""Append completed mainnet reference bars and settled funding, with receipts.

Existing research inputs are immutable. Missing observations fail the refresh;
zero funding, forward fills and currently forming candles are never substituted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import numpy as np

from replay_lab.common_comparison.models import funding_bucket
from replay_lab.v3.public import REFERENCES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "practice_live"
FROZEN = ROOT / "replay_lab/output/tail_calibration/20261014-v1/runtime/frozen_inputs.npz"
FROZEN_SHA = "c7f87bee20cf2fe54693f34fb175ff95935a8b807d1ba6cc0cae5f7dfa194324"
URL = "https://api.hyperliquid.xyz/info"
HOUR = 3600
BAR_FIELDS = ("open", "high", "low", "close", "volume", "trades")
RAW_FIELDS = ("o", "h", "l", "c", "v", "n")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def iso(seconds=None):
    return datetime.fromtimestamp(time.time() if seconds is None else seconds, timezone.utc).isoformat()


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_inputs(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def public_info(payload):
    """Two public read-only query types. No credentials or account queries."""
    if payload.get("type") == "candleSnapshot":
        req = payload.get("req", {})
        valid = (set(payload) == {"type", "req"}
                 and set(req) == {"coin", "interval", "startTime", "endTime"}
                 and req["coin"] in REFERENCES.values() and req["interval"] == "1h")
    else:
        valid = (payload.get("type") == "fundingHistory"
                 and set(payload) == {"type", "coin", "startTime", "endTime"}
                 and payload["coin"] in REFERENCES.values())
    if not valid:
        raise ValueError("Only reference candleSnapshot/fundingHistory queries are allowed")
    for attempt in range(4):
        try:
            request = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
            if not isinstance(result, list):
                raise ValueError("Public endpoint returned a non-list response")
            return {"endpoint": URL, "request": payload, "received_at_utc": iso(), "response": result}
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)


def fetch_symbol(symbol, previous, target):
    coin = REFERENCES[symbol]
    candles = public_info({"type": "candleSnapshot", "req": {
        "coin": coin, "interval": "1h", "startTime": (previous - HOUR) * 1000,
        "endTime": target * 1000 - 1}})
    funding, cursor = [], previous * 1000
    for _ in range(32):
        batch = public_info({"type": "fundingHistory", "coin": coin,
                             "startTime": cursor, "endTime": target * 1000 + 999})
        funding.append(batch)
        if len(batch["response"]) < 500:
            break
        next_cursor = max(int(row["time"]) for row in batch["response"]) + 1
        if next_cursor <= cursor:
            raise ValueError(f"Nonadvancing funding pagination: {symbol}")
        cursor = next_cursor
        if cursor > target * 1000 + 999:
            break
    else:
        raise ValueError(f"Funding pagination limit: {symbol}")
    return {"symbol": symbol, "reference_coin": coin, "candles": candles, "funding": funding}


def unique(rows, key, value, label):
    if key in rows and rows[key] != value:
        raise ValueError(f"Conflicting duplicate {label} at {iso(key)}")
    rows[key] = value


def parse_symbol(receipt, previous, target):
    symbol = receipt["symbol"]
    coin = REFERENCES[symbol]
    candles, rates = {}, {}
    ignored = {"unfinished_candles": 0, "outside_requested_window": 0}
    for row in receipt["candles"]["response"]:
        if row["s"] != coin or row["i"] != "1h":
            raise ValueError(f"Wrong reference candle: {symbol}")
        start_ms, end_ms = int(row["t"]), int(row["T"])
        if start_ms % (HOUR * 1000) or end_ms != start_ms + HOUR * 1000 - 1:
            raise ValueError(f"Invalid candle timestamp: {symbol}")
        known = start_ms // 1000 + HOUR
        if known > target:
            ignored["unfinished_candles"] += 1
            continue
        if known < previous:
            ignored["outside_requested_window"] += 1
            continue
        values = tuple(float(row[key]) for key in RAW_FIELDS)
        o, h, low, c, v, n = values
        if (not np.isfinite(values).all() or min(o, h, low, c) <= 0
                or low > min(o, c) or h < max(o, c) or h < low
                or v < 0 or n < 0 or n != int(n)):
            raise ValueError(f"Invalid OHLC/volume/trades: {symbol} at {iso(known)}")
        unique(candles, known, values, f"{symbol} candle")
    for batch in receipt["funding"]:
        for row in batch["response"]:
            if row["coin"] != coin:
                raise ValueError(f"Wrong funding market: {symbol}")
            known = funding_bucket(int(row["time"]))
            if known < previous or known > target:
                continue
            rate = float(row["fundingRate"])
            if not np.isfinite(rate):
                raise ValueError(f"Nonfinite funding: {symbol}")
            unique(rates, known, rate, f"{symbol} funding")
    required = set(range(previous, target + HOUR, HOUR))
    for label, rows in (("candles", candles), ("funding", rates)):
        missing = sorted(required - rows.keys())
        if missing:
            raise ValueError(f"Missing {symbol} {label}: {[iso(t) for t in missing[:8]]}")
    return candles, rates, ignored


def append_inputs(old, receipts, target):
    previous = int(old["market_times"][-1])
    if target < previous or target % HOUR:
        raise ValueError("Target must be a completed UTC hour at/after existing data")
    if list(receipts) != list(REFERENCES):
        raise ValueError("Wrong symbol order/set")
    if not np.array_equal(old["market_times"], old["known"]):
        raise ValueError("Existing data time mismatch")
    if np.any(np.diff(old["known"]) != HOUR) or not np.array_equal(old["known"], old["times"] + HOUR):
        raise ValueError("Existing data has noncontiguous or mislabelled hours")
    times = np.arange(previous + HOUR, target + HOUR, HOUR, dtype=np.int64)
    shape = (len(times), len(REFERENCES))
    new = {key: np.empty(shape, dtype=np.float64) for key in (*BAR_FIELDS, "funding")}
    audit = {}
    for j, symbol in enumerate(REFERENCES):
        if receipts[symbol]["symbol"] != symbol:
            raise ValueError("Receipt symbol mismatch")
        bars, rates, ignored = parse_symbol(receipts[symbol], previous, target)
        for k, field in enumerate(BAR_FIELDS):
            if bars[previous][k] != old[field][-1, j]:
                raise ValueError(f"Source overlap changed: {symbol} {field}; old data preserved")
            new[field][:, j] = [bars[int(t)][k] for t in times]
        if not old["observed"][-1, j] or rates[previous] != old["funding"][-1, j]:
            raise ValueError(f"Source overlap changed: {symbol} funding; old data preserved")
        new["funding"][:, j] = [rates[int(t)] for t in times]
        audit[symbol] = {"reference_coin": REFERENCES[symbol], "new_hours": len(times),
                         "overlap_exact": True, "missing_candles": 0, "missing_funding": 0,
                         "excluded": ignored}
    new.update(times=times - HOUR, known=times, market_times=times,
               observed=np.ones(shape, dtype=bool))
    close = np.concatenate((old["close"], new["close"]))
    returns = np.diff(np.log(close), axis=0, prepend=np.log(close[:1]))
    new["returns"] = returns[len(old["close"]):]
    new["sigma"] = np.empty(shape, dtype=np.float64)
    for k, i in enumerate(range(len(old["close"]), len(close))):
        new["sigma"][k] = np.maximum(returns[i-167:i+1].std(0, ddof=1), 1e-6)
    if set(new) != set(old):
        raise ValueError("Unexpected input schema; refusing to drop fields")
    merged = {key: np.concatenate((old[key], new[key])) for key in old}
    for key in old:
        np.testing.assert_array_equal(merged[key][:len(old[key])], old[key])
    return merged, audit


def base_path():
    if sha(FROZEN) != FROZEN_SHA:
        raise ValueError("Original frozen inputs changed")
    latest = OUT / "latest.json"
    if not latest.exists():
        return FROZEN
    pointer = json.loads(latest.read_text())
    path = ROOT / pointer["inputs"]
    if sha(path) != pointer["inputs_sha256"]:
        raise ValueError("Latest validated input hash changed")
    return path


def refresh():
    started = time.time()
    target = int(started // HOUR) * HOUR
    base = base_path()
    old = load_inputs(base)
    previous = int(old["market_times"][-1])
    if target < previous or (target - previous) // HOUR + 1 > 4999:
        raise ValueError("Clock is behind inputs or gap exceeds candle retention")
    run = OUT / "updates" / datetime.fromtimestamp(started, timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (run / "receipts").mkdir(parents=True)
    manifest = {"schema": 1, "started_at_utc": iso(started), "status": "collecting",
                "source": "mainnet_reference_settled", "endpoint": URL,
                "source_mappings": REFERENCES, "base": str(base.relative_to(ROOT)),
                "base_sha256": sha(base), "original_frozen_sha256": FROZEN_SHA,
                "previous_completed_bar_utc": iso(previous), "target_completed_bar_utc": iso(target),
                "orders_sent": 0, "filled_missing_observations": 0}
    save_json(run / "manifest.json", manifest)
    try:
        receipts = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {symbol: pool.submit(fetch_symbol, symbol, previous, target) for symbol in REFERENCES}
            errors = []
            for symbol, future in futures.items():
                try:
                    receipt = future.result()
                    save_json(run / "receipts" / f"{symbol}.json", receipt)
                    receipts[symbol] = receipt
                    print(f"Received {symbol} reference candles + funding", flush=True)
                except Exception as error:
                    errors.append(f"{symbol}: {type(error).__name__}: {error}")
            if errors:
                raise ValueError("; ".join(errors))
        merged, audit = append_inputs(old, receipts, target)
        with (run / "inputs.npz.tmp").open("wb") as stream:
            np.savez_compressed(stream, **merged)
        (run / "inputs.npz.tmp").replace(run / "inputs.npz")
        manifest.update(status="data_validated_pending_inference", completed_at_utc=iso(),
                        appended_hours=len(merged["known"]) - len(old["known"]),
                        total_hours=len(merged["known"]), old_prefix_exact=True,
                        inputs_sha256=sha(run / "inputs.npz"), assets=audit,
                        receipts_sha256={s: sha(run / "receipts" / f"{s}.json") for s in REFERENCES})
        save_json(run / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}", completed_at_utc=iso())
        save_json(run / "manifest.json", manifest)
        raise
    print(json.dumps({"run": str(run), "completed_bar_utc": iso(target),
                      "appended_hours": manifest["appended_hours"], "orders_sent": 0}), flush=True)
    return run


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    refresh()
