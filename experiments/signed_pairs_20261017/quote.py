"""Signed return-beta valuation, including negative-correlation FX contracts.

This is neither a cointegration test nor a guarantee of mean reversion. The
quote estimates a basis/funding hypothesis and never creates an order or fill.
"""
import math
import numpy as np
from experiments.execution_retry_20261016.engine import lot_floor
from experiments.paired_venue_20261017.engine import book_walk


def signed_beta(returns):
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError('Expected finite paired returns')
    covariance = np.cov(values.T, ddof=0)
    if min(covariance[0, 0], covariance[1, 1]) <= 1e-14:
        raise ValueError('Insufficient moving prices')
    beta = float(covariance[0, 1] / covariance[1, 1])
    correlation = float(covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1]))
    return dict(beta=beta, correlation=correlation, covariance=covariance.tolist())


def hedge_diagnostic(returns):
    fitted = signed_beta(returns); beta = fitted['beta']
    covariance = np.array(fitted['covariance'])
    correct = np.array([1., -beta]) / (1 + abs(beta))
    forced_opposite = np.array([1., -abs(beta)]) / (1 + abs(beta))
    fitted.update(correct_weights=correct.tolist(), forced_opposite_weights=forced_opposite.tolist(),
        correct_hourly_sigma=float(np.sqrt(max(0., correct @ covariance @ correct))),
        forced_opposite_hourly_sigma=float(np.sqrt(max(0., forced_opposite @ covariance @ forced_opposite))),
        net_weight=float(sum(correct)), gross_weight=float(sum(abs(correct))),
        dollar_neutral=bool(abs(sum(correct)) < 1e-10), mean_reversion_proven=False)
    return fitted


def signed_quote(coins, snapshots, metadata, beta, direction, horizon, cfg, scenario):
    if not math.isfinite(beta) or not .25 <= abs(beta) <= 4 or direction not in (-1, 1): return None
    if not isinstance(horizon, int) or horizon <= 0: raise ValueError('Positive whole funding hours required')
    oracle = np.array([snapshots[c]['oracle'] for c in coins])
    if not np.isfinite(oracle).all() or np.any(oracle <= 0): return None
    signs = np.array([direction, -direction * np.sign(beta)])
    planned = cfg['target_pair_gross'] * np.array([1., abs(beta)]) / (1 + abs(beta)) / oracle
    fraction = scenario['visible_depth_fraction']; scale = 1.
    for c, q, sign in zip(coins, planned, signs):
        for side in (sign, -sign):
            levels = snapshots[c]['book']['levels'][1 if side > 0 else 0]
            if not levels: return None
            best = float(levels[0]['px']); limit = best * (1 + side * cfg['limit_bps'] / 10000)
            capacity = sum(float(r['sz']) * fraction for r in levels if side * (float(r['px']) - limit) <= 1e-9)
            scale = min(scale, capacity / max(q, 1e-12))
    quantity = np.array([lot_floor(q * scale, metadata[c]['szDecimals']) for c, q in zip(coins, planned)])
    if np.any(quantity * oracle < cfg['minimum_leg_notional']): return None
    ratios = quantity / planned
    if min(ratios) / max(ratios) < 1 - cfg['balanced_fill_tolerance']: return None
    entry, exits, closing_now = [], [], []
    for c, q, side in zip(coins, quantity, signs):
        snap = snapshots[c]; levels = snap['book']['levels']
        en = book_walk(levels[1 if side > 0 else 0], q, fraction)
        ex = book_walk(levels[0 if side > 0 else 1], q, fraction)
        if en is None or ex is None: return None
        entry.append(en); closing_now.append(ex)
        exits.append(snap['mid'] + cfg['basis_convergence'] * (snap['oracle'] - snap['mid']) + ex - snap['mid'])
    entry, exits, closing_now = map(np.array, (entry, exits, closing_now))
    qty = quantity * signs
    rates = np.array([float(snapshots[c]['ctx']['funding']) for c in coins])
    if not np.isfinite(rates).all(): return None
    carry = float(-qty @ (oracle * rates) * horizon)
    if carry > 0: carry *= cfg['positive_funding_haircut']
    price_pnl = float(qty @ (exits - entry))
    fees = float(quantity @ (entry + exits) * scenario['fee_bps'] / 10000)
    gross = float(quantity @ entry)
    instant = float(qty @ (closing_now - entry) - quantity @ (entry + closing_now) * scenario['fee_bps'] / 10000)
    return dict(coins=list(coins), beta=beta, direction=direction, horizon_hours=horizon,
        target_qty=qty.tolist(), capacity_scale=scale, gross_notional=gross,
        entry_preview=entry.tolist(), exit_forecast=exits.tolist(), predicted_price_pnl=price_pnl,
        predicted_funding=carry, predicted_fees=fees, predicted_net_pnl=price_pnl + carry - fees,
        predicted_net_bps=(price_pnl + carry - fees) / gross * 10000,
        instant_roundtrip_bps=instant / gross * 10000,
        signed_oracle_notional=float(qty @ oracle), funding_is_forecast=True, future_exit_depth_is_unknown=True)
