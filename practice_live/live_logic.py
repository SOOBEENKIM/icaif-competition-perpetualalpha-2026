"""Pure account accounting and bounded IOC construction for the fixed strategy."""
from dataclasses import asdict, replace
from decimal import Decimal
import hashlib
import json

import numpy as np

from bot.execution import prepare_limit, rounded_price
from replay_lab.common_comparison.engine import rounded

D = Decimal


def identity(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def accounting(anchor, fills, funding, transfers, positions, marks, user):
    """Flat-start trade-cashflow ledger; never double-add unified-account PnL."""
    qty = {coin: D(0) for coin in marks}
    trade_cash = fees = closed = paid_funding = external = transfer_fees = D(0)
    for fill in fills:
        coin = fill["coin"]
        if coin not in qty or fill["feeToken"] != "USDPR" or fill["side"] not in ("A", "B"):
            raise ValueError("Unsupported concurrent trade or fee currency")
        amount = D(fill["sz"]) * (1 if fill["side"] == "B" else -1)
        qty[coin] += amount
        trade_cash -= amount * D(fill["px"])
        fees += D(fill["fee"])
        closed += D(fill["closedPnl"])
    for row in funding:
        delta = row["delta"]
        if delta.get("coin") in qty:
            paid_funding += D(delta["usdc"])
    for row in transfers:
        delta = row["delta"]
        if delta.get("token") != "USDPR":
            continue
        sender, receiver = delta.get("user", "").lower(), delta.get("destination", "").lower()
        amount = D(delta.get("amount", "0"))
        if sender == receiver == user.lower():
            pass
        elif receiver == user.lower():
            external += amount
        elif sender == user.lower():
            external -= amount
        else:
            raise ValueError("Unclassified USDPR capital flow")
        if sender == user.lower() and delta.get("feeToken") == "USDPR":
            transfer_fees += D(delta["fee"])
    actual = {p["position"]["coin"]: D(p["position"]["szi"]) for p in positions}
    if set(actual) - set(qty):
        raise ValueError("Unexpected par position")
    for coin, q in qty.items():
        if q != actual.get(coin, D(0)):
            raise ValueError(f"Fill/position reconciliation pending: {coin}")
    price_pnl = trade_cash + sum(qty[c] * D(str(marks[c])) for c in qty)
    net = price_pnl - fees + paid_funding - transfer_fees
    equity = D(str(anchor["initial_equity"])) + external + net
    if not equity.is_finite() or equity <= 0:
        raise ValueError("Invalid reconstructed equity")
    return {"equity": float(equity), "net_pnl": float(net), "external_capital": float(external),
            "fees": float(fees), "funding": float(paid_funding), "reported_closed_pnl": float(closed),
            "return_on_initial_percent": float(net / D(str(anchor["initial_equity"])) * 100),
            "qty": {c: float(q) for c, q in qty.items()}, "fills": len(fills)}


def phase_delta(current, target, phase):
    """Reversals close first. An unclosed residual cannot reverse or grow."""
    if phase == "reduce":
        if not current:
            return 0.
        if current * target <= 0:
            return -current
        return target - current if abs(target) < abs(current) else 0.
    if phase != "increase":
        raise ValueError("Unknown execution phase")
    if current * target < 0 or abs(target) <= abs(current):
        return 0.
    return target - current


def make_ioc(coin, delta, reduce_only, market, context, book, active, now_ms, slot, cap,
             slippage_bps=10., max_book_age_ms=15000):
    if book["coin"] != coin or active["coin"] != coin:
        raise ValueError("Wrong execution market")
    if not -2000 <= now_ms - int(book["time"]) <= max_book_age_ms:
        raise ValueError("Stale execution book")
    levels = book["levels"]
    if len(levels) != 2 or not levels[0] or not levels[1]:
        raise ValueError("Missing two-sided execution book")
    bid, ask = float(levels[0][0]["px"]), float(levels[1][0]["px"])
    if not np.isfinite([bid, ask]).all() or bid <= 0 or ask <= bid:
        raise ValueError("Crossed or invalid execution book")
    buy = bool(delta > 0)
    limit = (ask * (1 + slippage_bps / 10000)) if buy else (bid * (1 - slippage_bps / 10000))
    limit = float(rounded_price(D(str(limit)), int(market["szDecimals"]), buy))
    visible = 0.
    for row in levels[1 if buy else 0]:
        px, sz = float(row["px"]), float(row["sz"])
        if not np.isfinite([px, sz]).all() or px <= 0 or sz < 0:
            raise ValueError("Invalid execution depth")
        if (buy and px <= limit) or (not buy and px >= limit):
            visible += sz
    size = min(abs(delta), visible, cap / limit)
    if not reduce_only:
        size = min(size, float(active["maxTradeSzs"][0 if buy else 1]), cap / limit)
    size = abs(rounded(size, int(market["szDecimals"])))
    if not size or size * limit < 10:
        return None
    order = prepare_limit(coin, buy, size, limit, sz_decimals=int(market["szDecimals"]),
                          mark_price=context["markPx"], max_notional=str(cap),
                          tif="Ioc", reduce_only=reduce_only)
    cloid = "0x" + hashlib.sha256(slot.encode()).hexdigest()[:32]
    return replace(order, cloid=cloid)
