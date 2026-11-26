"""Conditional PAS replay using the deployed controller and account risk checks.

Minute trade candles cannot reconstruct historical executable books or marks.
Every result is explicitly a conditional simulation, not a live PnL forecast.
"""
import gzip
import json
from datetime import datetime
import numpy as np

from competition_live.profile import ROOT, load_profile
from competition_live.scoring import evaluate
from replay_lab.ema_rebuild.engine import Controller
from replay_lab.common_comparison.engine import risk_breaches, risk_values, rounded
from experiments.shared_execution_pairs_20261017.account_v2 import SharedAccount, difference
from pas_strategy_audit.data import load
from pas_adaptive_development.study import native


def simulate(cfg, d, targets, p, scenario, *, controller=None):
    symbols = cfg['symbols']
    account = SharedAccount(p['initial'], symbols)
    ledger = account.physical
    controller = Controller() if controller is None else controller
    sc = scenario
    pending = None
    cursor = 0
    funding_cursor = 0
    sigma = np.eye(len(symbols)) * p['risk']['hourly_sigma_floor'] ** 2
    snapshots, decisions, orders, breaches = [], [], [], []
    peak = p['initial']
    halted = False
    last = d['close'][0].copy()
    max_error = 0.
    reserved = 50000.
    for i, t in enumerate(d['times']):
        if t < cfg['start_ms']:
            last = d['close'][i].copy()
            continue
        observed_prices = np.where(np.isfinite(d['close'][i]), d['close'][i], last)
        priced = np.isfinite(observed_prices) & (observed_prices > 0)
        if np.any(~priced & (ledger.qty != 0)):
            raise ValueError('Cannot value held inventory without a prior price')
        # Missing, unheld assets have zero value and cannot acquire a target.
        # A finite placeholder serves array arithmetic only, never execution
        # or a future-price backfill. Keep the original missing mask below.
        prices = np.where(priced, observed_prices, 1.)
        while funding_cursor < len(d['funding_events']) and d['funding_events'][funding_cursor]['time'] <= t:
            f = d['funding_events'][funding_cursor]
            rates = np.zeros(len(symbols))
            rates[f['j']] = f['rate']
            funding_prices = np.nan_to_num(last, nan=0.)
            cash = -ledger.qty[f['j']] * funding_prices[f['j']] * f['rate']
            if cash > 0:
                rates[f['j']] *= sc['receipt_multiplier']
            account.charge_funding(f['time'] / 1000, funding_prices, rates, np.ones(len(symbols), bool))
            funding_cursor += 1
        if pending is not None and t > pending['time'] and t <= pending['expires']:
            target = pending['qty']
            ids = sorted(range(len(symbols)), key=lambda j: ledger.qty[j] *
                         difference(target[j], ledger.qty[j]) >= 0)
            for j in ids:
                old = ledger.qty[j]
                delta = difference(target[j], old)
                if not delta or not d['active'][i, j] or not priced[j]:
                    continue
                side = 1 if delta > 0 else -1
                reducing = old * delta < 0
                request = min(abs(old), abs(delta)) if reducing else abs(delta)
                price = (d['high'][i, j] if side > 0 else d['low'][i, j]) * (1 + side * sc['impact_bps'] / 10000)
                limit = pending['source_prices'][j] * (1 + side * sc['impact_bps'] / 10000) * (1 + side * .001)
                blocked = not reducing and side * (price - limit) > 1e-8
                available = min(pending['source_volume'][j], d['volume'][i, j]) * sc['participation']
                amount = abs(rounded(min(request, available, 7500 / price),
                                     p['execution']['decimals'][j]))
                if blocked:
                    amount = 0.
                if amount:
                    amount = account.feasible_size(j, side, amount, price, prices, sigma, p, sc['fee_bps'])
                if amount * price < 10 and not (reducing and amount == abs(old)):
                    amount = 0.
                orders.append(dict(time=int(t), symbol=symbols[j], requested=request, filled=amount,
                                   limit_blocked=blocked, volume_cap=available))
                if amount:
                    before = len(ledger.fills)
                    account.physical_fill(j, side * amount, price, sc['fee_bps'], int(t),
                                          decision_time=pending['time'])
                    for f in ledger.fills[before:]:
                        account.owner_fill('base', j, f['signed_qty'], price, sc['fee_bps'], int(t), 'external_fill')
                    assert old * ledger.qty[j] >= -1e-10
        equity = account.reconcile(prices)
        max_error = max(max_error, account.maximum_error, ledger.max_error)
        peak = max(peak, equity)
        halted |= equity / peak - 1 <= -p['risk']['drawdown_halt']
        breach = risk_breaches(ledger.qty, equity, prices, sigma, p['risk'])
        if breach:
            breaches.append(dict(time=int(t), limits=breach))
        if halted or breach:
            pending = dict(time=int(t), expires=cfg['cutoff_ms'], qty=np.zeros(len(symbols)),
                           source_prices=prices.copy(), source_volume=np.full(len(symbols), np.inf))
        while cursor < len(targets) and targets[cursor]['time'] <= t:
            event = targets[cursor]
            sigma = np.asarray(event['sigma'])
            action = controller.propose(event['source_time'], prices, ledger.qty, equity,
                                        np.where(priced, np.asarray(event['weights']), 0.), sigma,
                                        np.where(priced, np.asarray(event['volume']), 0.),
                                        np.where(priced, np.asarray(event['trades']), 0.),
                                        p, buffer=.2, halted=halted)
            if action is not None:
                target = np.asarray(action['target_qty'])
                pending = dict(time=int(t), expires=event['source_time'] * 1000 + 900000,
                               qty=target, source_prices=prices.copy(),
                               source_volume=d['volume'][max(0, i-59):i+1].sum(0))
                decisions.append(dict(time=int(t), source_time=event['source_time'],
                    raw_weights=event['weights'], target_qty=target.tolist(), equity=equity,
                    unpriced_assets=np.flatnonzero(~priced).tolist(),
                    raw_gross=sum(abs(x) for x in event['weights']) * equity,
                    target_gross=float(abs(target) @ prices),
                    held_gross=float(abs(ledger.qty) @ prices)))
            cursor += 1
        rv = risk_values(ledger.qty, equity, prices, sigma)
        snapshots.append(dict(time=int(t), equity=equity + reserved, user_equity=equity,
                              gross=rv['gross'] * equity, qty=ledger.qty.tolist(),
                              valuation_age_minutes=float(np.max(np.where(ledger.qty != 0, d['age_ms'][i] / 60000, 0)))))
        last = observed_prices
    daily = [dict(time=cfg['start_ms'] // 1000, equity=p['initial'] + reserved)]
    daily += [dict(time=s['time'] // 1000, equity=s['equity']) for s in snapshots
              if s['time'] > cfg['start_ms'] and s['time'] % 86400000 == 43200000]
    score = evaluate(daily, trading_days=len({f['time'] // 86400000 for f in ledger.fills}))
    final = snapshots[-1]
    reserve = final['gross'] * (sc['fee_bps'] + sc['impact_bps']) / 10000
    team_curve = np.array([s['equity'] for s in snapshots])
    team_peak = np.maximum.accumulate(np.r_[100000., team_curve])[1:]
    metrics = dict(net_pnl=final['equity'] - 100000., team_return=final['equity'] / 100000. - 1,
                   terminal_cost_reserve=reserve, pnl_after_liquidation_reserve=final['equity'] - 100000. - reserve,
                   fees=ledger.fees, funding_proxy=ledger.funding,
                   mean_notional=float(np.mean([s['gross'] for s in snapshots])),
                   maximum_notional=max(s['gross'] for s in snapshots),
                   final_notional=final['gross'], final_qty=ledger.qty.tolist(),
                   max_drawdown=float(-np.min(team_curve / team_peak - 1)),
                   fills=len(ledger.fills),
                   short_opening_fills=sum(f['signed_qty'] < 0 and not f['reduce_only'] for f in ledger.fills),
                   reduction_fills=sum(f['reduce_only'] for f in ledger.fills),
                   traded_symbols=sorted({f['symbol'] for f in ledger.fills}),
                   risk_breach_bars=len(breaches), maximum_accounting_error=max_error,
                   daily_returns=score['completed_returns'], volatility=score['volatility'], Q=score['Q'],
                   final_valuation_age_minutes=final['valuation_age_minutes'],
                   timing_pass=all(f['decision_time'] < f['time'] for f in ledger.fills),
                   pair_overlay='Base-owner conditional comparison only; existing pair overlay excluded from BOTH arms')
    assert max_error < 1e-6 and metrics['timing_pass']
    return dict(metrics=metrics, snapshots=snapshots, fills=ledger.fills, decisions=decisions,
                orders=orders, funding=ledger.funding_events, daily=daily, breaches=breaches)

