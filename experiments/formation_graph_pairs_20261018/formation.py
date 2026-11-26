"""Fixed past-only pair selection, independent of forecast or PnL labels."""
from datetime import datetime
import numpy as np


def forest(edges, nodes):
    parent = {n: n for n in nodes}
    def root(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    chosen = []
    for edge in sorted(edges, key=lambda e: (e['distance'], tuple(e['pair']))):
        a, b = map(root, edge['pair'])
        if a != b:
            parent[a] = b
            chosen.append(edge)
    return chosen


def select(reference, symbols, spec):
    t = np.asarray(reference['market_times'])
    if t.ndim != 1 or np.any(np.diff(t) <= 0):
        raise ValueError('Expected strictly increasing observation timestamps')
    cutoff = int(datetime.fromisoformat(spec['formation_end_exclusive']).timestamp())
    first = cutoff - int(spec['formation_days']) * 86400
    # Slice before price operations, so future prices cannot enter statistics.
    rows = (t >= first) & (t < cutoff)
    t = t[rows]
    close = np.asarray(reference['close'])[rows]
    volume = np.asarray(reference['volume'])[rows]
    trades = np.asarray(reference['trades'])[rows]
    active = (np.isfinite(close) & (close > 0) & np.isfinite(volume)
              & (volume > 0) & np.isfinite(trades) & (trades > 0))
    prices = np.where(active, close, np.nan)
    ret = np.diff(np.log(prices), axis=0)
    step = np.diff(t) == 3600
    edges, groups = [], []
    for group in spec['groups']:
        candidates = []
        for pair in spec['pairs']:
            if not all(s in group for s in pair):
                continue
            ids = [symbols.index(s) for s in pair]
            ok = step & active[1:, ids].all(1) & active[:-1, ids].all(1)
            ok &= np.isfinite(ret[:, ids]).all(1)
            values = ret[ok][:, ids]
            n = len(values)
            rho = None
            if n >= spec['minimum_observations'] and np.all(values.std(0) > 0):
                rho = float(np.clip(np.corrcoef(values.T)[0, 1], -1, 1))
                if not np.isfinite(rho):
                    rho = None
            eligible = rho is not None and rho >= spec['minimum_correlation']
            edge = dict(pair=list(pair), observations=n, correlation=rho,
                        distance=float(np.sqrt(2 * (1-rho))) if rho is not None else None,
                        eligible=eligible,
                        last_observation=int(t[1:][ok][-1]) if ok.any() else None)
            edges.append(edge)
            if eligible:
                candidates.append(edge)
        chosen = forest(candidates, group)
        groups.append(dict(nodes=group, selected_pairs=[e['pair'] for e in chosen],
                           fully_connected=len(chosen) == len(group)-1))
    pairs = [pair for g in groups for pair in g['selected_pairs']]
    return dict(cutoff_exclusive=cutoff, formation_start=first, edges=edges,
                groups=groups, selected_pairs=pairs,
                mask=[list(pair) in pairs for pair in spec['pairs']],
                used_only_pre_evaluation_observations=True)


def mask_entries(state, allowed):
    allowed = np.asarray(allowed, dtype=bool)
    if allowed.shape != (len(state['pairs']),):
        raise ValueError('Pair selection must match forecast pair order')
    result = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state.items()}
    result['economic_available'][:, ~allowed] = False
    # Do not alter source directions: holding/renewal must retain its forecasts.
    return result
