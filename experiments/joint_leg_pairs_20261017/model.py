"""Causal two-output ridge; preserve the spread forecast in a separate ablation."""
import numpy as np
import pandas as pd


def fit(x, y, maturity, now, spec):
    if len(x) < spec['minimum_training_rows']:
        return None
    if np.any(maturity >= now) or not all(np.isfinite(v).all() for v in [x, y]):
        raise ValueError('Training requires finite, strictly matured observations')
    w = np.exp2(-(now-maturity)/(86400*spec['half_life_days']))
    w /= w.sum()
    center = w@x
    scale = np.maximum(np.sqrt(w@((x-center)**2)), 1e-6)
    z = np.c_[np.ones(len(x)), (x-center)/scale]
    penalty = np.eye(z.shape[1])*spec['ridge_penalty']
    penalty[0, 0] = spec['intercept_prior_rows']/len(x)
    coefficient = np.linalg.solve(z.T@(w[:, None]*z)+penalty,
                                  z.T@(w[:, None]*np.clip(y, -spec['clip'], spec['clip'])))
    return dict(center=center.tolist(), scale=scale.tolist(), coefficient=coefficient.tolist(),
                rows=len(x), fit_time=int(now), last_maturity=int(maturity.max()))


def rolling(times, features, targets, maturity, units, terminal_fresh, spec):
    if np.any(np.diff(times) != 3600) or not np.array_equal(maturity, times+spec['horizon']*3600):
        raise ValueError('Incorrect source cadence or target maturity')
    n, pairs, _ = targets.shape
    assert targets.shape == units.shape == (n, pairs, 2)
    valid = np.isfinite(features['x']).all(-1) & features['fresh']
    valid &= np.isfinite(units).all(-1) & (units > 0).all(-1)
    train_ok = valid & terminal_fresh & np.isfinite(targets).all(-1)
    predictions = np.full_like(targets, np.nan)
    stamps = np.full((n, pairs), -1, np.int64)
    records, models = [], {}
    for i, now in enumerate(times):
        if now % (spec['refit_hours']*3600) == 0:
            models = {}
            window = (times >= now-spec['lookback_days']*86400) & (maturity < now)
            window &= times % (spec['sample_hours']*3600) == 0
            for k in range(pairs):
                ix = np.flatnonzero(window & train_ok[:, k])
                model = fit(features['x'][ix, k], targets[ix, k], maturity[ix], now, spec)
                if model is not None:
                    models[k] = model
                    records.append(dict(pair_index=k, first_source=int(times[ix[0]]), **model))
        for k, model in models.items():
            if not valid[i, k]:
                continue
            z = np.r_[1., (features['x'][i, k]-model['center'])/model['scale']]
            predictions[i, k] = np.clip(z@model['coefficient'], -spec['clip'], spec['clip'])*units[i, k]
            stamps[i, k] = model['fit_time']
    return predictions, stamps, records


def targets_and_units(reference, features, h):
    close = reference['close']; n = len(close)
    sigma = pd.DataFrame(np.log(close)).diff().rolling(168).std(ddof=0).to_numpy()
    units = np.stack([sigma[:, ids]*np.sqrt(h) for ids in features['pairs']], axis=1)
    target = np.full_like(units, np.nan)
    for k, ids in enumerate(features['pairs']):
        target[:-h, k] = (close[h:, ids]/close[:-h, ids]-1)/units[:-h, k]
    terminal_fresh = np.zeros_like(features['fresh'])
    terminal_fresh[:-h] = features['fresh'][h:]
    return target, reference['market_times']+h*3600, units, terminal_fresh


def common_projection(mu, beta, leg_returns):
    """Add a learned null-space component while preserving source hedge PnL."""
    common = leg_returns.mean(axis=-1)
    return np.stack([mu+2*beta*common/(1+beta), -mu+2*common/(1+beta)], axis=-1)


def map_forecasts(times, reference_times, source_predictions, baseline, kind):
    if kind not in ('joint', 'common'):
        raise ValueError('Unknown prespecified candidate')
    state = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in baseline.items()}
    source = state['signal_times']
    ix = np.searchsorted(reference_times, source)
    bounded = np.clip(ix, 0, len(reference_times)-1)
    available = (ix < len(reference_times)) & (reference_times[bounded] == source) & (source <= times[:, None])
    leg_returns = source_predictions[bounded, np.arange(source.shape[1])[None, :]].copy()
    leg_returns[~available] = np.nan
    beta = state['beta']
    if kind == 'common':
        leg_returns = common_projection(state['mu'], beta, leg_returns)
    else:
        state['mu'] = (leg_returns[..., 0]-beta*leg_returns[..., 1])/(1+beta)
    anchors = np.stack([state['source_prices'][:, ids] for ids in state['pairs']], axis=1)
    state['terminal_prices'] = anchors*(1+leg_returns)
    valid = available & np.isfinite(state['terminal_prices']).all(-1) & (state['terminal_prices'] > 0).all(-1)
    state['directions'] = np.where(valid, state['directions'], 0)
    return state


def terminal_prices(state, i, k, ids):
    if 'terminal_prices' in state:
        value = np.asarray(state['terminal_prices'][i, k], float)
        if value.shape != (2,):
            raise ValueError('A forecast must supply exactly two leg prices')
        return value.copy()
    mu = float(state['mu'][i, k])
    return state['source_prices'][i, ids]*np.array([1+mu, 1-mu])
