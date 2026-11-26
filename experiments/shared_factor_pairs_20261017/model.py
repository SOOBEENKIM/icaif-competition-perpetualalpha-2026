"""Shared BTC forecast plus source-loading pair residual; no execution changes."""
import numpy as np
import pandas as pd


def prepare(reference, features, horizon=24, factor_window=168):
    times = reference['market_times']
    if horizon <= 0 or factor_window < 2 or np.any(np.diff(times) != 3600):
        raise ValueError('Positive horizon and contiguous hourly source data required')
    close = reference['close']
    logs = pd.DataFrame(np.log(close))
    sigma = logs[0].diff().rolling(factor_window).std(ddof=0).to_numpy()
    sigma = np.where(sigma > 0, sigma, np.nan)
    columns, names = [], []
    for hours in (1, 6, 24, 72, 168, 336):
        columns.append(logs[0].diff(hours).to_numpy() / (sigma * np.sqrt(hours)))
        names.append(f'btc_return_{hours}')
    for hours in (24, 168, 720):
        columns.append((logs[0] - logs[0].rolling(hours).mean()).to_numpy() / (sigma * np.sqrt(hours)))
        names.append(f'btc_deviation_{hours}')
    columns.append(logs[0].diff().rolling(24).std(ddof=0).to_numpy() / sigma)
    names.append('btc_volatility_ratio')
    market_x = np.clip(np.column_stack(columns), -8, 8)
    simple = pd.DataFrame(close).pct_change(fill_method=None)
    variance = simple[0].rolling(factor_window).var(ddof=0)
    variance = variance.where(variance > 0)
    beta = np.column_stack([(simple[j].rolling(factor_window).cov(simple[0], ddof=0) / variance).to_numpy()
                            for j in simple])
    gamma = np.column_stack([(beta[:, a] - features['beta'][:, k] * beta[:, b]) / (1 + features['beta'][:, k])
                             for k, (a, b) in enumerate(features['pairs'])])
    n, count = gamma.shape
    actual = np.full((n, count), np.nan)
    market_return = np.full(n, np.nan)
    market_return[:-horizon] = close[horizon:, 0] / close[:-horizon, 0] - 1
    for k, (a, b) in enumerate(features['pairs']):
        leg = close[horizon:, [a, b]] / close[:-horizon, [a, b]] - 1
        hedge = features['beta'][:-horizon, k]
        actual[:-horizon, k] = (leg[:, 0] - hedge * leg[:, 1]) / (1 + hedge)
    pair_unit = features['vol'] * np.sqrt(horizon)
    market_unit = sigma * np.sqrt(horizon)
    market_fresh = (reference['volume'][:, 0] > 0) & (reference['trades'][:, 0] > 0)
    terminal_market = np.zeros(n, bool)
    terminal_market[:-horizon] = market_fresh[horizon:]
    terminal_pair = np.zeros((n, count), bool)
    terminal_pair[:-horizon] = features['fresh'][horizon:]
    residual = (actual - gamma * market_return[:, None]) / pair_unit
    return dict(x=market_x, names=names, unit=market_unit, gamma=gamma, pair_unit=pair_unit,
                market_return=market_return, actual=actual, market_target=market_return / market_unit,
                residual_target=residual, maturity=times + horizon * 3600, fresh=market_fresh,
                terminal_market=terminal_market, terminal_pair=terminal_pair, source_beta_to_btc=beta)


def fit(x, target, maturity, now, spec):
    if len(x) < spec['minimum_training_rows']:
        return None
    if np.any(maturity >= now) or not all(np.isfinite(v).all() for v in (x, target)):
        raise ValueError('Finite inputs and strictly matured labels required')
    weight = np.exp2(-(now - maturity) / (86400 * spec['half_life_days']))
    weight /= weight.sum()
    center = weight @ x
    scale = np.maximum(np.sqrt(weight @ ((x - center) ** 2)), 1e-6)
    z = np.c_[np.ones(len(x)), (x - center) / scale]
    penalty = np.eye(z.shape[1]) * spec['ridge_penalty']
    penalty[0, 0] = spec['intercept_prior_rows'] / len(x)
    y = np.clip(target, -spec['target_clip'], spec['target_clip'])
    coef = np.linalg.solve(z.T @ (weight[:, None] * z) + penalty, z.T @ (weight * y))
    return dict(center=center.tolist(), scale=scale.tolist(), coefficient=coef.tolist(), rows=len(x),
                first_maturity=int(maturity.min()), last_maturity=int(maturity.max()), fit_time=int(now),
                clipped_targets=int(np.sum(abs(target) > spec['target_clip'])))


def predict(model, x, unit, spec):
    z = np.r_[1., (x - model['center']) / model['scale']]
    return float(np.clip(z @ model['coefficient'], -spec['prediction_clip'], spec['prediction_clip']) * unit)


def rolling(times, features, factor, original_raw, spec):
    if np.any(np.diff(times) != 3600) or not np.array_equal(factor['maturity'], times + spec['horizon'] * 3600):
        raise ValueError('Incorrect source cadence or target maturity')
    shape = features['fresh'].shape
    if original_raw.shape != shape or factor['gamma'].shape != shape:
        raise ValueError('Unaligned pair inputs')
    market_ok = (np.isfinite(factor['x']).all(1) & factor['fresh'] & np.isfinite(factor['unit']) & (factor['unit'] > 0))
    pair_ok = (np.isfinite(features['x']).all(-1) & features['fresh'] & np.isfinite(original_raw)
               & np.isfinite(factor['gamma']) & np.isfinite(factor['pair_unit']) & (factor['pair_unit'] > 0)
               & market_ok[:, None])
    market_train = market_ok & factor['terminal_market'] & np.isfinite(factor['market_target'])
    pair_train = pair_ok & factor['terminal_pair'] & factor['terminal_market'][:, None] & np.isfinite(factor['residual_target'])
    forecasts = {kind: np.full(shape, np.nan) for kind in ('combined', 'market', 'residual')}
    components = {kind: np.full(shape, np.nan) for kind in ('market', 'residual')}
    btc = np.full(len(times), np.nan)
    stamps = np.full(shape, -1, np.int64)
    records, pair_models, market_model = [], {}, None
    for i, now in enumerate(times):
        if now % (spec['refit_hours'] * 3600) == 0:
            pair_models = {}
            window = ((times >= now - spec['lookback_days'] * 86400) & (factor['maturity'] < now)
                      & (times % (spec['sample_hours'] * 3600) == 0))
            ix = np.flatnonzero(window & market_train)
            market_model = fit(factor['x'][ix], factor['market_target'][ix], factor['maturity'][ix], now, spec)
            if market_model is not None:
                records.append(dict(role='market', first_source=int(times[ix[0]]), names=factor['names'], **market_model))
            for k in range(shape[1]):
                ix = np.flatnonzero(window & pair_train[:, k])
                model = fit(features['x'][ix, k], factor['residual_target'][ix, k], factor['maturity'][ix], now, spec)
                if model is not None:
                    pair_models[k] = model
                    records.append(dict(role='residual', pair_index=k, first_source=int(times[ix[0]]), names=features['names'], **model))
        if market_model is None or not market_ok[i]:
            continue
        btc[i] = predict(market_model, factor['x'][i], factor['unit'][i], spec)
        for k, model in pair_models.items():
            if not pair_ok[i, k]:
                continue
            market = factor['gamma'][i, k] * btc[i]
            residual = predict(model, features['x'][i, k], factor['pair_unit'][i, k], spec)
            cap = spec['prediction_clip'] * factor['pair_unit'][i, k]
            components['market'][i, k] = market
            components['residual'][i, k] = residual
            for kind, value in [('combined', market + residual), ('market', market), ('residual', residual)]:
                forecasts[kind][i, k] = np.clip(value, -cap, cap)
            stamps[i, k] = model['fit_time']
    return forecasts, btc, components, stamps, records
