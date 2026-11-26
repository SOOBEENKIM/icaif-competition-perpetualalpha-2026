"""Causal combinations of originally issued same-horizon expert forecasts."""
import itertools
import numpy as np
from experiments.adaptive_residual_pairs_20261017.model_v2 import build
from experiments.dense_history_pairs_20261017.model import labels, map_state


def simplex_fit(predictions, target, observation_weight, prior):
    """Exact small convex quadratic solve over every nonempty simplex face."""
    x = np.asarray(predictions, float)
    y = np.asarray(target, float)
    v = np.asarray(observation_weight, float)
    k = x.shape[1]
    uniform = np.full(k, 1/k)
    a = x.T @ (v[:, None]*x) + prior*np.eye(k)
    b = x.T @ (v*y) + prior*uniform
    best = None
    for size in range(1, k+1):
        for face in itertools.combinations(range(k), size):
            ii = np.array(face)
            block = a[np.ix_(ii, ii)]
            solution = np.linalg.solve(block, np.column_stack([b[ii], np.ones(size)]))
            multiplier = (solution[:, 0].sum()-1)/solution[:, 1].sum()
            active = solution[:, 0] - multiplier*solution[:, 1]
            if np.any(active < -1e-10):
                continue
            weights = np.zeros(k)
            weights[ii] = np.maximum(active, 0)
            weights /= weights.sum()
            objective = float(v @ (x@weights-y)**2 + prior*np.sum((weights-uniform)**2))
            if best is None or objective < best[0]:
                best = (objective, weights)
    if best is None:
        raise ArithmeticError('No feasible simplex face')
    return best[1]


def combine(times, issued, target, maturity, fresh, terminal_fresh, horizon, spec):
    times = np.asarray(times)
    if not np.all(np.diff(times) == 3600):
        raise ValueError('Contiguous completed hourly source bars required')
    if issued.ndim != 3 or target.shape != issued.shape[:2] or fresh.shape != target.shape or terminal_fresh.shape != target.shape:
        raise ValueError('Aligned time, pair and expert matrices required')
    if len(times) != len(issued) or not np.array_equal(maturity, times+horizon*3600):
        raise ValueError('Forecast horizon mismatch')
    if horizon != spec['horizon'] or issued.shape[-1] != len(spec['experts']):
        raise ValueError('Expert identity/horizon mismatch')
    n, pairs, experts = issued.shape
    forecasts = {kind: np.full((n, pairs), np.nan) for kind in spec['candidates']}
    weights = {kind: np.full((n, pairs, experts), np.nan) for kind in spec['candidates']}
    records = []
    available = np.isfinite(issued).all(-1) & fresh
    valid_label = available & terminal_fresh & np.isfinite(target)
    sample = times % (horizon*3600) == 0
    prior = np.full(experts, 1/experts)
    for i, now in enumerate(times):
        window = sample & (maturity < now) & (maturity >= now-spec['lookback_days']*86400)
        for k in np.flatnonzero(available[i]):
            ix = np.flatnonzero(window & valid_label[:, k])
            assert np.all(np.diff(times[ix]) >= horizon*3600)
            chosen = {kind: prior.copy() for kind in spec['candidates']}
            effective = 0.
            loss = None
            if len(ix) >= spec['minimum_rows']:
                decay = np.exp2(-(now-maturity[ix])/(spec['half_life_days']*86400))
                effective = float(decay.sum()**2/(decay@decay))
                v = decay/decay.sum()*effective
                y = np.clip(target[ix, k], -spec['normalized_target_clip'], spec['normalized_target_clip'])
                # These are the predictions issued at source ix, not refitted
                # historical predictions using the current calibration.
                x = issued[ix, k]
                loss = (v @ (x-y[:, None])**2 + spec['prior_rows']*spec['prior_mse'])/(effective+spec['prior_rows'])
                inverse = 1/np.maximum(loss, 1e-12)
                chosen['imse'] = inverse/inverse.sum()
                chosen['stack'] = simplex_fit(x, y, v, spec['prior_rows'])
            for kind in spec['candidates']:
                weights[kind][i, k] = chosen[kind]
                forecasts[kind][i, k] = chosen[kind] @ issued[i, k]
            records.append(dict(time=int(now), pair_index=int(k), rows=len(ix), effective_rows=effective,
                                first_source=int(times[ix[0]]) if len(ix) else None,
                                last_source=int(times[ix[-1]]) if len(ix) else None,
                                last_maturity=int(maturity[ix[-1]]) if len(ix) else None,
                                fallback=len(ix) < spec['minimum_rows'],
                                posterior_mse=loss.tolist() if loss is not None else None,
                                weights={kind: chosen[kind].tolist() for kind in spec['candidates']}))
    return forecasts, weights, records


def infer(times, reference, features, models, pairs, expert_spec, spec):
    h = spec['horizon']
    forecasts, _ = build(reference, features, models, pairs, h, expert_spec)
    unit = features['vol']*np.sqrt(h)
    issued = np.stack([forecasts[k]/unit for k in spec['experts']], axis=-1)
    target, maturity = labels(reference['market_times'], reference['close'], features, h)
    terminal_fresh = np.zeros_like(features['fresh'])
    terminal_fresh[:-h] = features['fresh'][h:]
    blended, weights, records = combine(reference['market_times'], issued, target, maturity,
                                        features['fresh'], terminal_fresh, h, spec)
    states = {kind: map_state(times, reference, features, value*unit, features['pairs'],
                             source_hours=1, lifetime_hours=1) for kind, value in blended.items()}
    controls = {kind: map_state(times, reference, features, forecasts[kind], features['pairs'],
                               source_hours=1, lifetime_hours=1) for kind in spec['experts']}
    return states, controls, records, dict(times=reference['market_times'], issued=issued, target=target,
                                          maturity=maturity, fresh=features['fresh'], terminal_fresh=terminal_fresh,
                                          **{f'prediction_{k}': v for k, v in blended.items()},
                                          **{f'weights_{k}': v for k, v in weights.items()})
