"""Issued leg forecasts indexed by an exact, strictly matured source horizon."""
import numpy as np
from experiments.joint_leg_pairs_20261017.model import rolling, targets_and_units


def prepare(reference, features, spec):
    predictions, stamps, records = [], [], []
    for h in spec['horizons']:
        y, maturity, units, fresh = targets_and_units(reference, features, h)
        pred, fit_times, models = rolling(reference['market_times'], features, y,
                                         maturity, units, fresh, {**spec, 'horizon': h})
        predictions.append(pred)
        stamps.append(fit_times)
        records.extend(dict(horizon=h, **m) for m in models)
    return np.stack(predictions, axis=2), np.stack(stamps, axis=2), records


def map_remaining(times, reference_times, predictions, stamps, baseline, horizons):
    state = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in baseline.items()}
    source = state['signal_times']
    ix = np.searchsorted(reference_times, source)
    bounded = np.clip(ix, 0, len(reference_times)-1)
    available = (ix < len(reference_times)) & (reference_times[bounded] == source)
    available &= (source <= times[:, None]) & ((times[:, None]-source) < 3600)
    values = predictions[bounded, np.arange(source.shape[1])[None, :]].copy()
    fit_times = stamps[bounded, np.arange(source.shape[1])[None, :]].copy()
    anchors = np.stack([state['source_prices'][:, ids] for ids in state['pairs']], axis=1)
    terminal = anchors[:, :, None, :] * (1+values)
    valid = available[:, :, None] & np.isfinite(terminal).all(-1) & (terminal > 0).all(-1)
    valid &= (fit_times >= 0) & (fit_times <= source[:, :, None])
    terminal[~valid] = np.nan
    state.update(holding_terminal=terminal, holding_available=valid, holding_fit_times=fit_times,
                 holding_horizons=np.array(horizons, dtype=int))
    return state


def terminal_for_maturity(de, i, k, now, maturity):
    if 'holding_terminal' not in de:
        return None
    source = int(de['signal_times'][i, k])
    if not source <= now < source+3600 or maturity <= now:
        return None
    span = maturity-source
    if span % 3600:
        return None
    matches = np.flatnonzero(de['holding_horizons'] == span//3600)
    if len(matches) != 1:
        return None
    j = int(matches[0])
    if not de['holding_available'][i, k, j]:
        return None
    stamp = int(de['holding_fit_times'][i, k, j])
    terminal = np.asarray(de['holding_terminal'][i, k, j], float)
    if stamp < 0 or stamp > source or not np.isfinite(terminal).all() or np.any(terminal <= 0):
        return None
    return dict(terminal_prices=terminal.copy(), source=source, horizon_hours=span//3600, fit_time=stamp)
