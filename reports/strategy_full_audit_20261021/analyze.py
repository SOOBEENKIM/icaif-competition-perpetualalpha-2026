"""Offline arithmetic from timestamped public receipts; never submits orders."""
import collections
import datetime as dt
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from replay_lab.v3.public import REFERENCES


def load(name):
    return json.loads((OUT / "raw" / (name + ".json")).read_text())


def response(name):
    return load(name)["response"]


def kst(t):
    return dt.datetime.fromtimestamp(t, dt.timezone(dt.timedelta(hours=9))).isoformat()


def walk(levels, units):
    left, value = units, 0.
    for level in levels:
        take = min(left, float(level["sz"]))
        left -= take
        value += take * float(level["px"])
        if left < 1e-10:
            return value / units
    return None


def main():
    profile = json.loads((ROOT / "competition_live/config.json").read_text())
    symbols = profile["trading"]["symbols"]
    meta, contexts = response("pas_market")
    reference = {}
    for name in ("main_market", "xyz_market"):
        m, c = response(name)
        reference.update({asset["name"]: context for asset, context in zip(m["universe"], c)})
    pointer = load("reference_pointer")
    run = ROOT / pointer["run"]
    z = np.load(run / "inputs.npz", allow_pickle=False)
    t = z["market_times"]
    prices = z["close"]
    source = int(t[-1])
    signal = load("runtime_signal")
    frame = signal["publication"]["data"]["frame"]
    features = json.loads((run / "signals.json").read_text())
    markets = []
    for j, (symbol, ctx) in enumerate(zip(symbols, contexts)):
        book_receipt = load("book_" + symbol)
        book = book_receipt["response"]
        bid, ask = (float(x[0]["px"]) if x else None for x in book["levels"])
        row = {
            "symbol": symbol, "mark": float(ctx["markPx"]), "oracle": float(ctx["oraclePx"]),
            "pas_mark_24h_pct": (float(ctx["markPx"]) / float(ctx["prevDayPx"]) - 1) * 100,
            "funding_bps_hour": float(ctx["funding"]) * 10000,
            "volume_24h_usdp": float(ctx["dayNtlVlm"]),
            "open_interest_usdp": float(ctx["openInterest"]) * max(float(ctx["markPx"]), float(ctx["oraclePx"])),
            "book_server_age_ms": book_receipt["received_unix"] * 1000 - book["time"],
            "bid": bid, "ask": ask,
            "reference_last_completed_close": float(prices[-1, j]),
            "raw_weight": frame["weights"][j],
        }
        rctx = reference[REFERENCES[symbol]]
        row["mainnet_mark"] = float(rctx["markPx"])
        row["basis_mark_bps"] = (row["mark"] / row["mainnet_mark"] - 1) * 10000
        row["mainnet_funding_bps_hour"] = float(rctx["funding"]) * 10000
        for hours in (6, 24, 72, 168):
            i = np.searchsorted(t, source - hours * 3600)
            row[f"reference_{hours}h_pct"] = (prices[-1, j] / prices[i, j] - 1) * 100
        i = np.searchsorted(t, source - 168 * 3600)
        row["reference_from_7d_high_pct"] = (prices[-1, j] / np.max(z["high"][i:, j]) - 1) * 100
        i = np.searchsorted(t, profile["start_unix"])
        row["reference_since_start_pct"] = (prices[-1, j] / prices[i, j] - 1) * 100
        row["reference_source_kst"] = kst(source)
        row["signal_diagnostics"] = features["assets"][j]
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2
            row["mid"] = mid
            row["spread_bps"] = (ask - bid) / mid * 10000
            for side, levels, best, sign in [
                    ("bid", book["levels"][0], bid, -1), ("ask", book["levels"][1], ask, 1)]:
                row[f"{side}_depth_10bp_from_best_usdp"] = sum(
                    float(level["sz"]) * float(level["px"]) for level in levels
                    if sign * (float(level["px"]) / best - 1) <= .001 + 1e-12)
            row["instant_roundtrip_cost_bps"] = {}
            for notional in (1000, 5000, 7500, 25000):
                buy, sell = walk(book["levels"][1], notional / mid), walk(book["levels"][0], notional / mid)
                row["instant_roundtrip_cost_bps"][str(notional)] = (
                    (buy - sell) / mid * 10000 + 18 if buy is not None and sell is not None else None)
        candles = response("candles_" + symbol)
        completed = [c for c in candles if c["t"] >= profile["start_unix"] * 1000 and c["T"] < source * 1000]
        row["pas_completed_hours_since_start"] = len(completed)
        row["pas_zero_trade_hours_since_start"] = sum(c["n"] == 0 for c in completed)
        row["pas_trades_since_start_completed_hours"] = sum(c["n"] for c in completed)
        markets.append(row)
    positions = response("clearinghouseState")["assetPositions"]
    funding = response("userFunding")
    fills = response("userFillsByTime")
    assert len(funding) < 500 and len(fills) < 2000, "Pagination required"
    pas_funding = [r for r in funding if r["delta"]["coin"].startswith("pas:")]
    pas_fills = [r for r in fills if r["coin"].startswith("pas:")]
    fees = sum(float(r["fee"]) for r in pas_fills)
    funding_cash = sum(float(r["delta"]["usdc"]) for r in pas_funding)
    unrealized = sum(float(p["position"]["unrealizedPnl"]) for p in positions)
    realized = sum(float(r["closedPnl"]) for r in pas_fills)
    balance = next(b for b in response("spotClearinghouseState")["balances"] if b["coin"] == "USDP")
    equity = float(balance["total"])
    gross = sum(float(p["position"]["positionValue"]) for p in positions)
    by_symbol = collections.defaultdict(float)
    for r in pas_funding:
        by_symbol[r["delta"]["coin"]] += float(r["delta"]["usdc"])
    weights = np.asarray(frame["weights"])
    sigma = np.asarray(frame["sigma"])
    gate = []
    for f in signal["forecasts"]:
        if f["model"] == "decision_gate":
            cost = .0038 / (f["source_vol"] * np.sqrt(24))
            gate.append({"pair": f["pair"], "action": f["economic_action"],
                         "decision_score": f["decision_score"], "threshold": cost,
                         "score_fraction_of_threshold": abs(f["decision_score"]) / cost,
                         "price_forecast_bps": f["mu"] * 10000})
    account = {
        "observed_kst": kst(response("clearinghouseState")["time"] / 1000),
        "equity": equity, "pnl": equity - 100000, "return_pct": (equity / 100000 - 1) * 100,
        "unrealized": unrealized, "realized": realized, "funding": funding_cash, "fees": fees,
        "arithmetic_pnl": unrealized + realized + funding_cash - fees,
        "reconciliation_error": equity - 100000 - (unrealized + realized + funding_cash - fees),
        "gross": gross, "gross_team_pct": gross / equity * 100,
        "gross_sleeve_pct": gross / (equity - 50000) * 100,
        "spot_hold": float(balance["hold"]),
        "spot_total_minus_hold_pct": (equity - float(balance["hold"])) / equity * 100,
        "fills": len(pas_fills), "orders": len({r["oid"] for r in pas_fills}),
        "open_orders": len(response("openOrders")), "positions": positions,
        "funding_by_symbol": dict(by_symbol),
        "raw_weight_gross_pct": abs(weights).sum() * 100,
        "raw_notional": abs(weights).sum() * (equity - 50000),
        "raw_daily_risk_pct": np.sqrt(24 * weights @ sigma @ weights) * 100,
        "gate": gate,
    }
    report = {
        "market_capture_kst": kst(load("pas_market")["received_unix"]),
        "reference_run": str(run), "account": account, "markets": markets,
        "limitations": [
            "Books are single snapshots, not historical or guaranteed future fills.",
            "24h PAS change uses current mark / API prevDayPx; reference horizons use completed hourly closes.",
            "All posted depth is visible, cancellable, and includes multiple price levels.",
            "Instant roundtrip previews include 9bp per side but no latency or adverse price/funding movement.",
        ],
    }
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(account, indent=2))
    for r in markets:
        print(r["symbol"], json.dumps({k: r[k] for k in [
            "reference_24h_pct", "reference_168h_pct", "reference_from_7d_high_pct",
            "pas_mark_24h_pct", "spread_bps", "bid_depth_10bp_from_best_usdp",
            "ask_depth_10bp_from_best_usdp", "funding_bps_hour", "mainnet_funding_bps_hour",
            "basis_mark_bps", "volume_24h_usdp", "pas_zero_trade_hours_since_start",
        ]}))


if __name__ == "__main__":
    main()
