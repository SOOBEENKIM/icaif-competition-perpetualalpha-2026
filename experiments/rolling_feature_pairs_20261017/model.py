"""Daily rolling per-pair feature fits with strictly matured price labels."""
import numpy as np
from experiments.adaptive_residual_pairs_20261017.model import frozen_predictions
from experiments.adaptive_residual_pairs_20261017.model_v2 import build
from experiments.dense_history_pairs_20261017.model import labels, map_state


def fit(x, target, frozen, maturity, now, spec):
    if len(x) < spec['minimum_training_rows']:
        return None
    if np.any(maturity >= now):
        raise ValueError('Unmatured label in rolling fit')
    if not all(np.isfinite(v).all() for v in [x, target, frozen]):
        raise ValueError('Nonfinite training row')
    weight = np.exp2(-(now-maturity)/(spec['half_life_days']*86400))
    weight /= weight.sum()
    center = weight@x
    scale = np.maximum(np.sqrt(weight@((x-center)**2)), 1e-6)
    z = np.c_[np.ones(len(x)), (x-center)/scale]
    penalty = np.eye(z.shape[1])*spec['ridge_penalty']
    penalty[0, 0] = spec['intercept_prior_rows']/len(x)
    y = np.clip(target, -spec['target_clip'], spec['target_clip'])
    targets = np.column_stack([y, y-frozen])
    coefficient = np.linalg.solve(z.T@(weight[:, None]*z)+penalty, z.T@(weight[:, None]*targets))
    scalar = float(weight@(y-frozen)/(1+penalty[0, 0]))
    return dict(center=center.tolist(), scale=scale.tolist(), direct=coefficient[:, 0].tolist(),
                residual=coefficient[:, 1].tolist(), scalar=scalar, rows=len(x),
                first_maturity=int(maturity.min()), last_maturity=int(maturity.max()),
                fit_time=int(now), clipped_targets=int(np.sum(abs(target) > spec['target_clip'])))


def rolling(times, features, target, maturity, raw, terminal_fresh, spec):
    if not np.all(np.diff(times) == 3600):
        raise ValueError('Contiguous hourly sources required')
    if not np.array_equal(maturity, times+spec['horizon']*3600):
        raise ValueError('Wrong source label horizon')
    if target.shape != raw.shape or terminal_fresh.shape != raw.shape or features['x'].shape[:2] != raw.shape:
        raise ValueError('Unaligned pair data')
    kinds = [*spec['candidates'], *spec['diagnostic_variants']]
    forecasts = {kind: np.full(raw.shape, np.nan) for kind in kinds}
    stamps = np.full(raw.shape, -1, np.int64)
    records = []
    feature_ok = np.isfinite(features['x']).all(-1) & features['fresh'] & np.isfinite(raw)
    train_ok = feature_ok & terminal_fresh & np.isfinite(target)
    models = {}
    for i, now in enumerate(times):
        if now % (spec['refit_hours']*3600) == 0:
            models = {}
            window = (times >= now-spec['lookback_days']*86400) & (maturity < now)
            window &= times % (spec['sample_hours']*3600) == 0
            for k in range(raw.shape[1]):
                ix = np.flatnonzero(window & train_ok[:, k])
                model = fit(features['x'][ix, k], target[ix, k], raw[ix, k], maturity[ix], now, spec)
                if model is not None:
                    models[k] = model
                    records.append(dict(pair_index=k, first_source=int(times[ix[0]]), last_source=int(times[ix[-1]]),
                                        sample_hours=spec['sample_hours'], labels_overlap=spec['sample_hours'] < spec['horizon'],
                                        names=features['names'], **model))
        for k, model in models.items():
            if not feature_ok[i, k] or not np.isfinite(features['vol'][i, k]) or features['vol'][i, k] <= 0:
                continue
            z = np.r_[1., (features['x'][i, k]-model['center'])/model['scale']]
            values = dict(direct=float(z@model['direct']), residual=float(raw[i, k]+z@model['residual']),
                          scalar=float(raw[i, k]+model['scalar']))
            for kind in kinds:
                forecasts[kind][i, k] = np.clip(values[kind], -spec['prediction_clip'], spec['prediction_clip'])*features['vol'][i, k]*np.sqrt(spec['horizon'])
            stamps[i, k] = model['fit_time']
    return forecasts, stamps, records


def infer(times, reference, features, models, pairs, expert_spec, spec):
    h = spec['horizon']
    original, _ = build(reference, features, models, pairs, h, expert_spec)
    raw = frozen_predictions(features, models, pairs, h)
    target, maturity = labels(reference['market_times'], reference['close'], features, h)
    terminal_fresh = np.zeros_like(features['fresh'])
    terminal_fresh[:-h] = features['fresh'][h:]
    predictions, stamps, records = rolling(reference['market_times'], features, target, maturity,
                                            raw, terminal_fresh, spec)
    states = {kind: map_state(times, reference, features, value, features['pairs'], source_hours=1, lifetime_hours=1)
              for kind, value in predictions.items()}
    controls = {kind: map_state(times, reference, features, original[kind], features['pairs'], source_hours=1, lifetime_hours=1)
                for kind in spec['experts']}
    return states, controls, records, dict(times=reference['market_times'], target=target, maturity=maturity,
        raw_normalized=raw, unit=features['vol']*np.sqrt(h), fit_stamps=stamps,
        fresh=features['fresh'], terminal_fresh=terminal_fresh,
        **{f'prediction_{kind}': value for kind, value in predictions.items()},
        **{f'prediction_{kind}': original[kind] for kind in spec['experts']})
