"""Daily direct paired PRICE forecasts, augmented by strictly past funding."""
import copy
import numpy as np
import pandas as pd
from experiments.dense_history_pairs_20261017.model import design as price_design, labels, map_state
from experiments.rolling_feature_pairs_20261017.model import rolling


def design(reference, symbols, pairs, windows=(24, 168)):
    d = price_design(reference['market_times'], reference['close'], reference['volume'],
                     reference['trades'], symbols, pairs)
    funding = np.asarray(reference['funding'])
    observed = np.asarray(reference['observed'])
    if funding.shape != reference['close'].shape or observed.shape != funding.shape:
        raise ValueError('Funding and observation mask must align with price rows')
    if observed.dtype != np.dtype(bool):
        raise ValueError('Funding observation mask must be boolean')
    # Missing settlements are unknown, including when the serialized rate is 0.
    known = observed & np.isfinite(funding)
    settled = pd.DataFrame(np.where(known, funding, np.nan)).shift(1)
    columns = []
    for h in windows:
        if not isinstance(h, int) or h <= 0:
            raise ValueError('Funding windows must be positive integer hours')
        # min_periods=h requires every historical hourly settlement to exist.
        rates = settled.rolling(h, min_periods=h).mean().to_numpy()
        values = []
        for k, (a, b) in enumerate(d['pairs']):
            beta, vol = d['beta'][:, k], d['vol'][:, k]
            values.append((rates[:, a] - beta * rates[:, b]) / (1 + beta) / vol)
        columns.append(np.stack(values, axis=1))
    features = np.clip(np.stack(columns, axis=-1), -8, 8)
    return {**d, 'x': np.concatenate([d['x'], features], axis=-1),
            'names': [*d['names'], *(f'funding_{h}' for h in windows)]}


def infer(times, reference, symbols, pairs, direct_spec, windows=(24, 168)):
    features = design(reference, symbols, pairs, windows)
    spec = copy.deepcopy(direct_spec)
    spec['candidates'], spec['diagnostic_variants'] = ['direct'], []
    h = spec['horizon']
    target, maturity = labels(reference['market_times'], reference['close'], features, h)
    terminal = np.zeros_like(features['fresh'])
    terminal[:-h] = features['fresh'][h:]
    unused = np.where(np.isfinite(features['x']).all(-1), 0., np.nan)
    forecasts, stamps, records = rolling(reference['market_times'], features, target, maturity,
                                        unused, terminal, spec)
    state = map_state(times, reference, features, forecasts['direct'], features['pairs'],
                      source_hours=1, lifetime_hours=1)
    return state, records, dict(times=reference['market_times'], target=target, maturity=maturity,
        predictions=forecasts['direct'], fit_stamps=stamps, unit=features['vol']*np.sqrt(h),
        fresh=features['fresh'], terminal_fresh=terminal)
