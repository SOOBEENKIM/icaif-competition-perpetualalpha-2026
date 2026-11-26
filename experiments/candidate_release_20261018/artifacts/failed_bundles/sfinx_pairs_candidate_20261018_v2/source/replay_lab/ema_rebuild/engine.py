"""Shared six-hour controller and hourly ledger with next-close proxy fills."""
from dataclasses import dataclass
import numpy as np
from ..common_comparison.engine import Ledger, execute_intent, project, risk_breaches, risk_values, rounded
from ..common_comparison.models import covariance, HOUR


@dataclass
class Controller:
    last_decision: int = -1

    def propose(self, time, prices, qty, equity, raw, sigma, volume, trades, protocol,
                buffer=.025, defensive=False, halted=False, valid_input=True):
        # Invalid inputs must not consume the decision cursor. Risk reductions
        # use the same controller and do not wait for a strategy boundary.
        if (not valid_input or not np.isfinite(prices).all() or np.any(prices <= 0)
                or not np.isfinite(raw).all() or not np.isfinite(sigma).all()
                or not np.isfinite(qty).all() or not np.isfinite(volume).all()
                or not np.isfinite(trades).all() or np.any(volume < 0) or np.any(trades < 0)
                or not np.isfinite(equity) or equity <= 0):
            return None
        risk = protocol['risk']
        forced = halted or bool(risk_breaches(qty, equity, prices, sigma, risk))
        if time <= self.last_decision or (time % 21600 and not forced):
            return None
        current = qty*prices/equity
        if forced:
            weights = np.zeros_like(current) if halted else project(current, equity, sigma, risk)
        else:
            weights = project(raw, equity, sigma, risk)
            candidate = weights*equity/prices
            relative = abs(candidate-qty)/np.maximum(np.maximum(abs(candidate), abs(qty)), 1e-12)
            same_sign = candidate*qty > 0
            weights = np.where(same_sign & (relative <= buffer), current, weights)
            if defensive:
                # A flat account stays flat; a residual cannot reverse or grow.
                same = weights*current > 0
                weights = np.where(same, np.sign(current)*np.minimum(abs(current), .5*abs(weights)), 0.)
            fresh = (volume > 0) & (trades > 0)
            increasing = (weights*current < 0) | (abs(weights) > abs(current))
            weights = np.where(~fresh & increasing, current, weights)
            weights = project(weights, equity, sigma, risk)
            change = abs(weights-current).sum()
            if change > risk['normal_turnover_fraction']:
                weights = current+(weights-current)*risk['normal_turnover_fraction']/change
        target = weights*equity/prices
        if forced:
            target = np.sign(qty)*np.minimum(abs(qty), abs(target))
        target = np.array([rounded(q, d) for q, d in zip(target, protocol['execution']['decimals'])])
        self.last_decision = int(time)
        return dict(target_qty=target.tolist(), risk_forced=bool(forced), defensive=bool(defensive))


def simulate(market, model, key, protocol, scenario_name, start, end, flags=None, buffer=.025):
    scenario, risk = protocol['scenarios'][scenario_name], protocol['risk']
    ledger = Ledger(protocol['initial'], market.symbols)
    agent = Controller()
    first, last = market.exact(start), market.exact(end)
    snapshots, decisions, orders, days, breaches = [], [], [], {}, []
    pending = None
    peak = ledger.initial; drawdown = 0.; halted = False
    streak = maximum_streak = observed_hours = missing_hours = defensive_decisions = 0
    for i in range(first, last+1):
        time, prices = int(market.times[i]), market.close[i]
        _, sigma = covariance(market, i, risk)
        if time % 86400 == 0:
            days[time] = ledger.equity(prices)
        if i > first:
            held = ledger.qty != 0
            observed_hours += int(np.sum(held & market.observed[i]))
            missing_hours += int(np.sum(held & ~market.observed[i]))
            ledger.charge_funding(time, prices, market.funding[i], market.observed[i], scenario['adverse_funding_bps_hour'])
        if pending is not None:
            assert pending['execution_time'] == time
            if time < end:
                # Safety at execution uses the completed execution-time inputs.
                # It may only shrink quantities sealed at the earlier decision.
                pending['sigma'] = sigma
                orders.extend(execute_intent(ledger, pending, market, i, protocol, scenario))
            pending = None
        equity = ledger.reconcile(prices)
        if equity <= 0:
            raise ValueError('Insolvent shared ledger')
        peak = max(peak, equity); drawdown = min(drawdown, equity/peak-1)
        if equity/peak-1 <= -risk['drawdown_halt']:
            halted = True
        limits = risk_breaches(ledger.qty, equity, prices, sigma, risk)
        streak = streak+1 if limits else 0
        maximum_streak = max(streak, maximum_streak)
        if limits:
            breaches.append(dict(time=time, constraints=limits, consecutive_hours=streak))
        if time < end:
            action = agent.propose(time, prices, ledger.qty, equity, model['target_weight'][i, :3],
                sigma, market.volume[i], market.trades[i], protocol, buffer=buffer,
                defensive=bool(flags[i]) if flags is not None else False, halted=halted)
            if action is not None:
                defensive_decisions += int(action['defensive'])
                event = dict(decision_time=time, execution_time=time+HOUR,
                    source_close=time, source_available=time, source_index=i,
                    source_prices=prices.tolist(), visible_equity=equity,
                    position_before=ledger.qty.tolist(), source_volume=market.volume[i].tolist(),
                    source_trades=market.trades[i].tolist(), **action)
                decisions.append(event)
                pending = {**event, 'source_volume': market.volume[i].copy(),
                           'source_trades': market.trades[i].copy(), 'sigma': sigma}
        snapshots.append(dict(time=time, equity=equity, cash=ledger.cash, qty=ledger.qty.tolist(),
            unrealized=equity-ledger.cash, fees=ledger.fees, funding=ledger.funding, realized=ledger.realized,
            **risk_values(ledger.qty, equity, prices, sigma)))
    daily = [dict(time=t, return_=days[t]/days[t-86400]-1) for t in sorted(days) if t-86400 in days]
    gates = dict(accounting=ledger.max_error < 1e-6,
        information_time=all(f['source_available'] <= f['decision_time'] < f['time'] for f in ledger.fills),
        risk=maximum_streak < 2, not_halted=not halted)
    metrics = dict(roi=equity/ledger.initial-1, equity=equity,
        daily_vol=float(np.std([d['return_'] for d in daily], ddof=0)), complete_days=len(daily),
        max_drawdown=drawdown, turnover=ledger.turnover, fees=ledger.fees, funding=ledger.funding,
        signed_funding=ledger.signed_funding, adverse_funding=ledger.adverse_funding,
        realized=ledger.realized, unrealized=equity-ledger.cash,
        fills=len(ledger.fills), orders=len(orders), trading_days=len({f['time']//86400 for f in ledger.fills}),
        partial_or_unfilled_orders=sum(o['status'] != 'FILLED' for o in orders),
        observed_position_hours=observed_hours, missing_position_hours=missing_hours,
        mean_gross=float(np.mean([s['gross'] for s in snapshots])), max_gross=max(s['gross'] for s in snapshots),
        final_qty=ledger.qty.tolist(), end_positions=int(np.count_nonzero(ledger.qty)),
        maximum_ledger_error=ledger.max_error, maximum_breach_streak=maximum_streak, halted=halted,
        defensive_decisions=defensive_decisions,
        short_opening_fills=sum(f['signed_qty'] < 0 and not f['reduce_only'] for f in ledger.fills))
    return dict(policy=key, kind='ema_rebuild', parameter=0, scenario=scenario_name,
        start=start, end=end, metrics=metrics, gates=gates, snapshots=snapshots, decisions=decisions,
        orders=orders, fills=ledger.fills, funding_events=ledger.funding_events, daily=daily,
        risk_breaches=breaches, live_orders_sent=0, terminal_valuation='reference close; no liquidation')


def select(rows):
    indexed = {(r['policy'], r['scenario']): r for r in rows}
    base = indexed['ema_base', 'base']
    eligible = []
    base_ok = all(base['gates'].values()) and all(indexed['ema_base', 'stress']['gates'].values())
    for name in ('ema_signed', 'ema_vae', 'ema_signed_vae'):
        r, stress = indexed[name, 'base'], indexed[name, 'stress']
        m = r['metrics']
        if (base_ok and all(r['gates'].values()) and all(stress['gates'].values())
                and m['roi'] > 0 and m['roi'] >= base['metrics']['roi']+0.00001
                and m['daily_vol'] <= base['metrics']['daily_vol']):
            eligible.append(r)
    eligible.sort(key=lambda r: (-r['metrics']['roi'], r['metrics']['daily_vol'], r['policy']))
    return dict(selected=eligible[0]['policy'] if eligible else 'ema_base',
                improvement_on_selection=bool(eligible), baseline_gates_pass=base_ok,
                eligible=[r['policy'] for r in eligible], independent_evaluation=False)
