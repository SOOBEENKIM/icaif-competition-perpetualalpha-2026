"""Finite-grid inventory/funding HJB. Synthetic quote study, NEVER a fill/PnL model.

Price normalized to 1; q is units of one quote lot. Time is hours. Inventory
funding cost q*f assumes oracle equals the normalization price for this demo.
Fee is added to each Hamiltonian relative to the paper's displayed dynamics.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Parameters:
    inventory_bound: int = 3
    horizon_hours: float = .25
    arrival_per_hour: float = 6.
    k: float = 1250.  # inverse normalized price, characteristic distance 8bp
    maker_fee: float = .0003
    terminal_penalty: float = .00015
    running_penalty: float = .0001
    funding_kappa: float = 2.
    funding_sigma: float = .0001
    funding_bound: float = .001
    funding_nodes: int = 17


def solve(p=Parameters()):
    if (p.inventory_bound < 1 or p.horizon_hours <= 0 or p.arrival_per_hour <= 0
            or p.k <= 0 or p.funding_nodes < 3 or p.funding_nodes % 2 != 1
            or p.funding_bound <= 0 or min(p.maker_fee, p.terminal_penalty,
                                         p.running_penalty, p.funding_kappa, p.funding_sigma) < 0):
        raise ValueError('Invalid HJB parameters')
    q = np.arange(-p.inventory_bound, p.inventory_bound+1)
    f = np.linspace(-p.funding_bound, p.funding_bound, p.funding_nodes)
    df = f[1]-f[0]
    drift = -p.funding_kappa*f  # illustrative zero long-run funding mean
    up = p.funding_sigma**2/(2*df**2)+np.maximum(drift, 0)/df
    down = p.funding_sigma**2/(2*df**2)+np.maximum(-drift, 0)/df
    up[-1] = 0; down[0] = 0  # reflecting finite-grid boundaries
    steps = max(1, int(np.ceil(p.horizon_hours*(max(up+down)+2*p.arrival_per_hour)/.45)))
    dt = p.horizon_hours/steps
    theta = np.repeat((-p.terminal_penalty*q*q).astype(float)[:, None], len(f), axis=1)

    def quotes(value):
        bid = np.full_like(value, np.nan); ask = np.full_like(value, np.nan)
        # max_delta exp(-k delta)*(delta-fee + theta(q +/- 1)-theta(q))
        bid[:-1] = np.maximum(0., 1/p.k+p.maker_fee-(value[1:]-value[:-1]))
        ask[1:] = np.maximum(0., 1/p.k+p.maker_fee-(value[:-1]-value[1:]))
        return bid, ask

    for _ in range(steps):
        bid, ask = quotes(theta)
        derivative = -q[:, None]*f[None, :]-p.running_penalty*q[:, None]**2
        derivative[:, :-1] += up[:-1]*(theta[:, 1:]-theta[:, :-1])
        derivative[:, 1:] += down[1:]*(theta[:, :-1]-theta[:, 1:])
        derivative[:-1] += p.arrival_per_hour*np.exp(-p.k*bid[:-1])*(bid[:-1]-p.maker_fee+theta[1:]-theta[:-1])
        derivative[1:] += p.arrival_per_hour*np.exp(-p.k*ask[1:])*(ask[1:]-p.maker_fee+theta[:-1]-theta[1:])
        theta += dt*derivative
    bid, ask = quotes(theta)
    return dict(inventory=q, funding=f, value=theta, bid_distance=bid, ask_distance=ask,
                steps=steps, cfl=dt*(max(up+down)+2*p.arrival_per_hour))


def demonstration():
    p = Parameters(); result = solve(p)
    rows = []
    for q in (-3, -2, 0, 2, 3):
        qi = int(np.where(result['inventory']==q)[0][0])
        for f in (-.0005, 0., .0005):
            fi = int(np.argmin(abs(result['funding']-f)))
            b = result['bid_distance'][qi, fi]; a = result['ask_distance'][qi, fi]
            rows.append(dict(inventory_lots=q, funding_bps_per_hour=f*10000,
                             bid_offset_bps=-float(b)*10000 if np.isfinite(b) else None,
                             ask_offset_bps=float(a)*10000 if np.isfinite(a) else None))
    return dict(status='synthetic_quote_calculation_only', calibrated_to_par=False,
                empirical_pnl=None, parameters=asdict(p), rows=rows, steps=result['steps'],
                cfl=result['cfl'],
                missing=['par arrival intensity calibration', 'queue position and cancellation latency',
                         'conditional adverse-selection/markout model', 'actual order/price constraints adapter',
                         'oracle/mark basis and funding settlement timing in executable simulator'])
