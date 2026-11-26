"""One offline event ledger, delayed intents, partial IOC fills and shared risk."""
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
import math

import numpy as np

from .models import HOUR, SYMBOLS, covariance, predict_ridge
from ..pullback.broker import execution_price


def rounded(value, decimals):
    amount = Decimal(str(abs(float(value)))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_FLOOR)
    return math.copysign(float(amount), value) if amount else 0.0


class Ledger:
    def __init__(self, initial, symbols=SYMBOLS):
        self.initial = self.cash = self.shadow_cash = float(initial)
        self.symbols = tuple(symbols)
        self.qty = np.zeros(len(self.symbols))
        self.entry = np.zeros(len(self.symbols))
        self.realized = self.fees = self.funding = self.turnover = 0.0
        self.signed_funding = self.adverse_funding = 0.0
        self.fills, self.funding_events = [], []
        self.max_error = 0.0

    def equity(self, prices):
        return float(self.cash + self.qty @ (prices - self.entry))

    def reconcile(self, prices):
        equity = self.equity(prices)
        shadow = float(self.shadow_cash + self.qty @ prices)
        components = self.initial + self.realized - self.fees + self.funding + self.qty @ (prices - self.entry)
        error = max(abs(equity - shadow), abs(equity - components))
        self.max_error = max(self.max_error, error)
        if not np.isfinite(equity) or error > 1e-6:
            raise AssertionError(f"Ledger conservation failed: {error}")
        return equity

    def fill(self, asset, signed_qty, price, fee_bps, time, **metadata):
        if not signed_qty or not np.isfinite(price) or price <= 0:
            raise ValueError("Invalid fill")
        old = self.qty[asset]
        reduction = old * signed_qty < 0
        if reduction and abs(signed_qty) > abs(old) + 1e-10:
            raise ValueError("A reduction must not reverse a residual position")
        realized = 0.0
        if reduction:
            realized = abs(signed_qty) * (price - self.entry[asset]) * np.sign(old)
        else:
            self.entry[asset] = (abs(old) * self.entry[asset] + abs(signed_qty) * price) / (abs(old) + abs(signed_qty))
        # Exchange quantities live on decimal lot grids. Binary accumulation
        # (77.8 + .01 -> 77.809999...) must never leave an untradeable phantom
        # lot after a full close and permanently block a later sign reversal.
        self.qty[asset] = float(Decimal(str(float(old))) + Decimal(str(float(signed_qty))))
        if abs(self.qty[asset]) < 1e-10:
            self.qty[asset] = self.entry[asset] = 0.0
        fee = abs(signed_qty) * price * fee_bps / 10000
        self.realized += realized; self.fees += fee
        self.cash += realized - fee
        self.shadow_cash -= signed_qty * price + fee
        self.turnover += abs(signed_qty) * price
        self.fills.append(dict(time=int(time), symbol=self.symbols[asset], asset=asset,
                               signed_qty=float(signed_qty), price=float(price), fee=float(fee),
                               realized=float(realized), reduce_only=bool(reduction),
                               position_before=float(old), position_after=float(self.qty[asset]), **metadata))

    def charge_funding(self, time, prices, rates, observed, adverse_bps_hour):
        signed = -float((self.qty * prices * rates).sum())
        adverse = -float((abs(self.qty) * prices).sum()) * adverse_bps_hour / 10000
        amount = signed + adverse
        self.cash += amount; self.shadow_cash += amount; self.funding += amount
        self.signed_funding += signed; self.adverse_funding += adverse
        if np.any(self.qty):
            self.funding_events.append(dict(time=int(time), signed_amount=signed, adverse_amount=adverse,
                qty=self.qty.tolist(), prices=prices.tolist(), rates=rates.tolist(), observed=observed.tolist()))


def risk_values(qty, equity, prices, sigma):
    if equity <= 0:
        raise ValueError("Nonpositive shared equity")
    notional = qty * prices
    weights = notional / equity
    return dict(gross=float(abs(weights).sum()), net=float(abs(weights.sum())),
                asset=float(abs(weights).max()), asset_notional=float(abs(notional).max()),
                daily_volatility=float(np.sqrt(max(0.0, 24 * weights @ sigma @ weights))))


def risk_breaches(qty, equity, prices, sigma, risk):
    values = risk_values(qty, equity, prices, sigma)
    limits = dict(gross=risk["gross_fraction"], net=risk["net_fraction"], asset=risk["asset_fraction"],
                  asset_notional=risk["asset_notional"], daily_volatility=risk["daily_volatility"])
    return [name for name, limit in limits.items() if values[name] > limit + 1e-10]


def project(weights, equity, sigma, risk):
    w = np.asarray(weights, dtype=float).copy()
    if not np.isfinite(w).all():
        raise ValueError("Nonfinite proposed weights")
    headroom = risk["sizing_headroom"]
    per_asset = min(risk["asset_fraction"], risk["asset_notional"] / equity) * headroom
    w = np.clip(w, -per_asset, per_asset)
    ratios = [1.0, abs(w).sum() / (risk["gross_fraction"] * headroom),
              abs(w.sum()) / (risk["net_fraction"] * headroom),
              np.sqrt(max(0.0, 24 * w @ sigma @ w)) / (risk["daily_volatility"] * headroom)]
    return w / max(ratios)


def incremental_utility(new, old, gain, cost, reserve, buffer):
    return (new - old) * gain - (cost + reserve) * (abs(new) - abs(old)) - (cost + buffer) * abs(new - old)


def economic_weights(gain, old, c, sigma, equity, protocol, scenario, equal=False, buffer_bps=None,
                     horizon_hours=4, gross_budget=None):
    risk = protocol["risk"]
    if equal:
        raw = np.sign(gain)
    else:
        raw = gain / np.maximum(np.diag(c), risk["hourly_sigma_floor"] ** 2)
    gross = float(abs(raw).sum())
    budget = risk["gross_fraction"] if gross_budget is None else min(gross_budget, risk["gross_fraction"])
    candidate = np.zeros(len(gain)) if gross == 0 else raw * budget / gross
    candidate = project(candidate, equity, sigma, risk)
    cost = (protocol["execution"]["fee_bps"] + scenario["impact_bps"]) / 10000
    reserve = horizon_hours * scenario["adverse_funding_bps_hour"] / 10000
    buffer = (protocol["cost_buffer_bps"] if buffer_bps is None else buffer_bps) / 10000
    chosen = old.copy()
    for j in range(len(gain)):
        best = 0.0
        # Current wins exact ties, then flat wins over a non-current candidate.
        for value in (0.0, candidate[j]):
            utility = incremental_utility(value, old[j], gain[j], cost, reserve, buffer)
            if utility > best + 1e-15:
                best, chosen[j] = utility, value
    return chosen


@dataclass
class Policy:
    key: str
    kind: str
    parameter: float = 0
    state: object = None
    equal: bool = False
    buffer_bps: object = None
    rebalance_fraction: float = .2
    gross_budget: object = None


class DecisionPolicy:
    def __init__(self, policy, assets=3):
        self.policy = policy
        self.target = np.zeros(assets)
        self.previous_signal = None
        self.previous_day = None

    def weights(self, market, x, source, ledger, protocol, scenario, c, sigma):
        p = self.policy
        prices = market.close[source]
        equity = ledger.equity(prices)
        current = ledger.qty * prices / equity
        gain = None
        assets = len(ledger.qty)
        if p.kind == "control":
            model = p.state
            signal = model["signal"][source, :assets]
            volatility = model["volatility"][source, :assets]
            day = int(market.times[source] // (24 * HOUR))
            changed = self.previous_signal is None or not np.array_equal(signal, self.previous_signal)
            if day != self.previous_day or changed:
                if "target_weight" in model:
                    raw = model["target_weight"][source, :assets].copy()
                else:
                    raw = np.where(np.isfinite(volatility), signal * np.minimum(.25 / np.maximum(volatility, .01), 2) * .01, 0)
                candidate = project(raw, equity, sigma, protocol["risk"]) * equity / prices
                for j in range(assets):
                    signal_changed = self.previous_signal is None or signal[j] != self.previous_signal[j]
                    relative = abs(candidate[j] - ledger.qty[j]) / max(abs(candidate[j]), abs(ledger.qty[j]), 1e-12)
                    if signal_changed or relative > p.rebalance_fraction:
                        self.target[j] = candidate[j]
                self.previous_day, self.previous_signal = day, signal.copy()
            proposed = self.target * prices / equity
        elif p.kind == "cash":
            proposed = np.zeros(assets)
        elif p.kind == "basket":
            # Fixed per-asset target weights, rebalanced only on the shared clock.
            proposed = np.full(assets, .03)
        else:
            if p.kind == "momentum":
                mu = 4 / p.parameter * np.log(prices / market.close[source - int(p.parameter)])
            elif p.kind == "ridge":
                mu = predict_ridge(p.state, x[source])
            elif p.kind == "forecast":
                mu = p.state['forecast'][source]
            else:
                raise ValueError(p.kind)
            gain = np.expm1(mu)
            if not np.isfinite(gain).all():
                raise ValueError("Nonfinite forecast")
            proposed = economic_weights(gain, current, c, sigma, equity, protocol, scenario, p.equal, p.buffer_bps,
                horizon_hours=p.parameter if p.kind == 'forecast' else 4, gross_budget=p.gross_budget)
        fresh = (market.volume[source] > 0) & (market.trades[source] > 0)
        for j in range(assets):
            if not fresh[j] and (proposed[j] * current[j] < 0 or abs(proposed[j]) > abs(current[j])):
                proposed[j] = current[j]
        projected = project(proposed, equity, sigma, protocol["risk"])
        if not risk_breaches(ledger.qty, equity, prices, sigma, protocol["risk"]):
            change = float(abs(projected - current).sum())
            if change > protocol["risk"]["normal_turnover_fraction"]:
                projected = current + (projected - current) * protocol["risk"]["normal_turnover_fraction"] / change
        return projected, gain


def execute_intent(ledger, intent, market, index, protocol, scenario):
    """Only quantity reductions are allowed after an intent is sealed."""
    time = int(market.times[index]); prices = market.close[index]
    sigma, risk = intent["sigma"], protocol["risk"]
    available = scenario["participation"] * np.minimum(intent["source_volume"], market.volume[index])
    available *= (intent["source_trades"] > 0) & (market.trades[index] > 0)
    orders = []
    target = np.array(intent["target_qty"])
    for reduction_pass in (True, False):
        for j in range(len(ledger.qty)):
            old = ledger.qty[j]
            delta = float(Decimal(str(float(target[j]))) - Decimal(str(float(old))))
            reducing = old != 0 and (target[j] * old <= 0 or abs(target[j]) < abs(old))
            if reducing != reduction_pass or not delta:
                continue
            if not reduction_pass and j in intent.get("reduced_assets", []):
                continue
            requested = min(abs(delta), abs(old)) if reducing else abs(delta)
            sign = -np.sign(old) if reducing else np.sign(delta)
            decimals = protocol["execution"]["decimals"][j]
            requested = abs(rounded(requested, decimals))
            price = execution_price(prices[j], sign, scenario["impact_bps"], decimals)
            amount = abs(rounded(min(requested, available[j]), decimals))
            if not reducing and amount:
                def feasible(size):
                    qty = ledger.qty.copy(); qty[j] += sign * size
                    fee = size * price * protocol["execution"]["fee_bps"] / 10000
                    equity = ledger.equity(prices) - sign * size * (price - prices[j]) - fee
                    return equity > 0 and not risk_breaches(qty, equity, prices, sigma, risk)
                if not feasible(0):
                    amount = 0.0
                elif not feasible(amount):
                    lo, hi = 0.0, amount
                    for _ in range(40):
                        mid = (lo + hi) / 2
                        if feasible(mid): lo = mid
                        else: hi = mid
                    amount = abs(rounded(lo, decimals))
            # The minimum applies to the submitted order, not a partial fill.
            # Organizer Q18 (Oct 13): a reduce-only order closing the entire
            # position is valid even when its notional is below $10.
            full_close = reducing and requested == abs(old)
            if requested * price < protocol["execution"]["minimum_order_value"] and not full_close:
                amount = 0.0
            order = dict(time=time, decision_time=int(intent["decision_time"]),
                         source_close=int(intent["source_close"]), source_available=int(intent["source_available"]),
                         symbol=ledger.symbols[j], requested_qty=requested, filled_qty=amount,
                         sign=float(sign), reduce_only=bool(reducing), capacity=float(available[j]),
                         risk_forced=bool(intent["risk_forced"]),
                         status="FILLED" if amount == requested and amount > 0 else "IOC_REMAINDER_CANCELLED")
            if amount:
                ledger.fill(j, sign * amount, price, protocol["execution"]["fee_bps"], time,
                            decision_time=order["decision_time"], source_close=order["source_close"],
                            source_available=order["source_available"], risk_forced=order["risk_forced"])
                available[j] -= amount
                if reducing:
                    intent.setdefault("reduced_assets", []).append(j)
            orders.append(order)
    return orders


def simulate(market, x, policy, protocol, scenario_name, start, end):
    scenario = protocol["scenarios"][scenario_name]
    risk = protocol["risk"]
    if len(market.symbols) != market.close.shape[1]:
        raise ValueError("Market symbols and price columns differ")
    ledger = Ledger(protocol["initial"], market.symbols)
    agent = DecisionPolicy(policy, len(market.symbols))
    first, last = market.exact(start), market.exact(end)
    decisions, orders, snapshots, days = [], [], [], {}
    peak = protocol["initial"]; drawdown = 0.0; halted = False
    pending = None; breaches = []; streak = 0; max_streak = 0
    observed_hours = missing_hours = 0
    equity = ledger.initial
    for i in range(first, last + 1):
        time = int(market.times[i]); prices = market.close[i]
        # Midnight boundary is valued before this timestamp's settlements and fills.
        if time % (24 * HOUR) == 0:
            days[time] = ledger.equity(prices)
        if i > first:
            held = ledger.qty != 0
            observed_hours += int(np.sum(held & market.observed[i]))
            missing_hours += int(np.sum(held & ~market.observed[i]))
            ledger.charge_funding(time, prices, market.funding[i], market.observed[i], scenario["adverse_funding_bps_hour"])
        if pending is not None and pending["execution_time"] == time:
            if time < end:
                orders.extend(execute_intent(ledger, pending, market, i, protocol, scenario))
            pending = None
        equity = ledger.reconcile(prices)
        if equity <= 0:
            raise ValueError("Insolvent historical ledger")
        peak = max(peak, equity); drawdown = min(drawdown, equity / peak - 1)
        source = market.source(time)
        c, sigma = covariance(market, source, risk)
        current_breaches = risk_breaches(ledger.qty, equity, prices, sigma, risk)
        streak = streak + 1 if current_breaches else 0
        max_streak = max(max_streak, streak)
        if current_breaches:
            breaches.append(dict(time=time, constraints=current_breaches, consecutive_hours=streak))
        # Risk decisions use the delayed observable price and current account balance,
        # never the just-completed close that is still unavailable to the policy.
        visible_prices = market.close[source]
        visible_equity = ledger.equity(visible_prices)
        visible_breach = risk_breaches(ledger.qty, visible_equity, visible_prices, sigma, risk)
        visible_peak = max([protocol["initial"]] + [s["visible_equity"] for s in snapshots[-1:]])
        if snapshots:
            visible_peak = max(visible_peak, snapshots[-1]["visible_peak"])
        visible_peak = max(visible_peak, visible_equity)
        if visible_equity / visible_peak - 1 <= -risk["drawdown_halt"]:
            halted = True
        force = halted or bool(visible_breach)
        if time < end and (time % (4 * HOUR) == 0 or force):
            if pending is not None:
                raise AssertionError("More than one unresolved delayed IOC intent")
            if force:
                weights = np.zeros(len(ledger.qty)) if halted else project(ledger.qty * visible_prices / visible_equity, visible_equity, sigma, risk)
                gain = None
            else:
                weights, gain = agent.weights(market, x, source, ledger, protocol, scenario, c, sigma)
            target = weights * visible_equity / visible_prices
            if force:
                # Risk actions can only reduce current exposure, never rebalance into a new position.
                target = np.sign(ledger.qty) * np.minimum(abs(target), abs(ledger.qty))
            target = np.array([rounded(q, d) for q, d in zip(target, protocol["execution"]["decimals"])])
            if not force:
                old_weights = ledger.qty * visible_prices / visible_equity
                new_weights = target * visible_prices / visible_equity
                # Rounding can add one lot of reduction. Record that tolerance explicitly.
                lot_notional = sum(10 ** -d * p for d, p in zip(protocol["execution"]["decimals"], visible_prices)) / visible_equity
                if abs(new_weights - old_weights).sum() > risk["normal_turnover_fraction"] + lot_notional + 1e-10:
                    raise AssertionError("Turnover projection failed")
            event = dict(decision_time=time, execution_time=time + HOUR,
                         source_close=int(market.times[source]), source_available=int(market.times[source] + HOUR),
                         source_index=source, target_qty=target.tolist(), source_prices=visible_prices.tolist(),
                         visible_equity=visible_equity, position_before=ledger.qty.tolist(),
                         gain=None if gain is None else gain.tolist(), risk_forced=force,
                         source_volume=market.volume[source].tolist(), source_trades=market.trades[source].tolist())
            assert event["source_available"] <= time < event["execution_time"]
            decisions.append(event)
            pending = {**event, "source_volume": market.volume[source].copy(),
                       "source_trades": market.trades[source].copy(), "sigma": sigma}
        rv = risk_values(ledger.qty, equity, prices, sigma)
        snapshots.append(dict(time=time, equity=equity, visible_equity=visible_equity,
                              visible_peak=visible_peak, cash=ledger.cash, qty=ledger.qty.tolist(),
                              unrealized=equity-ledger.cash, fees=ledger.fees, funding=ledger.funding,
                              realized=ledger.realized, **rv))
    daily = [dict(time=t, return_=days[t] / days[t - 24 * HOUR] - 1)
             for t in sorted(days) if t - 24 * HOUR in days]
    daily_vol = float(np.std([r["return_"] for r in daily], ddof=0)) if daily else None
    information_ok = all(f["source_available"] <= f["decision_time"] < f["time"] for f in ledger.fills)
    gates = dict(accounting=ledger.max_error < 1e-6, information_time=information_ok,
                 risk=max_streak < risk["persistent_breach_hours"], not_halted=not halted)
    metrics = dict(roi=equity / ledger.initial - 1, equity=equity, daily_vol=daily_vol,
                   complete_days=len(daily), max_drawdown=drawdown, turnover=ledger.turnover,
                   fees=ledger.fees, funding=ledger.funding, signed_funding=ledger.signed_funding,
                   adverse_funding=ledger.adverse_funding, realized=ledger.realized,
                   unrealized=equity-ledger.cash, trading_days=len({f["time"] // 86400 for f in ledger.fills}),
                   fills=len(ledger.fills), orders=len(orders),
                   partial_or_unfilled_orders=sum(o["status"] != "FILLED" for o in orders),
                   observed_position_hours=observed_hours, missing_position_hours=missing_hours,
                   mean_gross=float(np.mean([s["gross"] for s in snapshots])),
                   max_gross=max(s["gross"] for s in snapshots),
                   end_positions=int(np.count_nonzero(ledger.qty)), final_qty=ledger.qty.tolist(),
                   maximum_ledger_error=ledger.max_error, maximum_breach_streak=max_streak, halted=halted)
    return dict(policy=policy.key, kind=policy.kind, parameter=policy.parameter,
                scenario=scenario_name, start=start, end=end, metrics=metrics, gates=gates,
                decisions=decisions, orders=orders, fills=ledger.fills, funding_events=ledger.funding_events,
                snapshots=snapshots, daily=daily, risk_breaches=breaches,
                live_orders_sent=0, terminal_valuation="reference close; positions not liquidated")
