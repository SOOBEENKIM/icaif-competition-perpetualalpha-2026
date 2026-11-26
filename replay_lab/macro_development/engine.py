"""Generic version of the audited EMA event loop, supporting all ten assets.

Only changes: target dimension, covariance cache, execute a sealed final-hour
intent. Reuses Controller, actual ledger, lot sizing, risk and partial fills.
"""
import numpy as np
from ..ema_rebuild.engine import Controller
from ..common_comparison.engine import Ledger, execute_intent, risk_breaches, risk_values
from ..common_comparison.models import covariance, HOUR

def simulate(market, model, key, protocol, scenario_name, start, end, flags=None, buffer=.025, sigmas=None):
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
        sigma = sigmas[i] if sigmas is not None else covariance(market, i, risk)[1]
        if time % 86400 == 0:
            days[time] = ledger.equity(prices)
        if i > first:
            held = ledger.qty != 0
            observed_hours += int(np.sum(held & market.observed[i]))
            missing_hours += int(np.sum(held & ~market.observed[i]))
            ledger.charge_funding(time, prices, market.funding[i], market.observed[i], scenario['adverse_funding_bps_hour'])
        if pending is not None:
            assert pending['execution_time'] == time
            if time <= end:
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
            action = agent.propose(time, prices, ledger.qty, equity, model['target_weight'][i, :len(market.symbols)],
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
    return dict(policy=key, kind='macro_development', parameter=0, scenario=scenario_name,
        start=start, end=end, metrics=metrics, gates=gates, snapshots=snapshots, decisions=decisions,
        orders=orders, fills=ledger.fills, funding_events=ledger.funding_events, daily=daily,
        risk_breaches=breaches, live_orders_sent=0, terminal_valuation='reference close; no liquidation')

