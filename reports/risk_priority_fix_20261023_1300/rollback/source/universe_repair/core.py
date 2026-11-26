"""The core universe is a deployment contract, not an executor capability.

No new fit, winner selection, leverage increase or compulsory position count.
The frozen two-pair specialist does not veto any of these ten core markets.
"""
import numpy as np
from all_market_strategy.signals import SYMBOLS, Market
from strategy_repairs.model import build
from strategy_final_risk.model import final_mix

REVISION = 'all_market_signed_core_v1'
LEGACY = 'legacy_core_v1'
ARMS = {
    LEGACY: dict(scope='legacy', short=False),
    'all_market_scope_only': dict(scope='all', short=False),
    REVISION: dict(scope='all', short=True),
}


def calculate(data, forecast, probability, revision=REVISION):
    if revision not in ARMS:
        raise ValueError('Unknown core revision')
    n = len(data['market_times']); shape = (n, len(SYMBOLS))
    if not np.array_equal(data['known'], data['market_times']):
        raise ValueError('Core input must contain completed, aligned hours')
    for name in ('close', 'open', 'high', 'low', 'volume', 'trades', 'funding', 'observed'):
        if np.asarray(data[name]).shape != shape:
            raise ValueError('Missing market history: '+name)
    if forecast.shape != shape or probability.shape != shape:
        raise ValueError('Forecast universe does not match core universe')
    if not np.isfinite(forecast).all() or not np.isfinite(probability).all():
        raise ValueError('Nonfinite retained forecast')
    _, parts = build(data, forecast, probability,
        dict(**ARMS[revision], cross_exit=False, risk=True, original_size=False))
    m = Market(data['market_times'], data['close'], data['volume'], data['trades'],
               data['funding'], data['observed'], SYMBOLS)
    anchor, gate = final_mix(m, parts['ungated_sized_trend'], .5*parts['reversion_weight'],
                            float(parts['final_scale']), probability, forecast)
    return anchor, gate, parts


def audit(parts, gate, anchors, symbols=SYMBOLS):
    """Evidence for each market, including zero-signal markets; no forced trades."""
    if tuple(symbols) != SYMBOLS:
        raise ValueError('Changed symbol order')
    rows = []
    for j, symbol in enumerate(symbols):
        w = float(anchors[-1, j]); protected = bool(gate['protected'][-1, j])
        rows.append(dict(symbol=symbol, considered=bool(parts['allowed'][j]),
            ema_direction=float(parts['ema'][-1,j]), channel_weight=float(parts['channel'][-1,j]),
            reversion_direction=float(parts['reversion_state'][-1,j]),
            carry_weight=float(parts['carry'][-1,j]),
            pre_gate_weight=float(gate['pre_gate'][-1,j]), anchor_weight=w,
            risk_reduced=protected,
            core_decision=('long' if w>0 else 'short' if w<0 else 'no_core_signal'),
            note='Core target before current PAS quotes, quantity rounding, shared risk and execution filters'))
    return rows
