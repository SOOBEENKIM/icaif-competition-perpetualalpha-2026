"""Extend only the entry validity of an already observed fixed-source forecast."""
import copy
import numpy as np
from experiments.paid_rehedge_pairs_20261017.portfolio import replay as paid_replay


def extend_validity(state, times, hours=6):
    if not 1 <= hours <= 6:
        raise ValueError('Entry validity cannot exceed the unchanged forecast refresh interval')
    out = {k:v.copy() if isinstance(v, np.ndarray) else copy.deepcopy(v) for k,v in state.items()}
    if len(times) != len(state['directions']) or np.any(np.diff(times) <= 0):
        raise ValueError('State must align to strictly increasing observation times')
    active = np.full(state['directions'].shape[1], -1, dtype=np.int64)
    directions = np.zeros_like(state['directions'])
    for i, now in enumerate(times):
        sources = state['signal_times'][i]
        if np.any(sources > now):
            raise ValueError('A future forecast source cannot be made available early')
        original = state['directions'][i] != 0
        if np.any(original & ((now-sources) >= 3600)):
            raise ValueError('Expected the original one-hour validity mapping')
        active[original] = sources[original]
        valid = (active == sources) & ((now-sources) < hours*3600)
        valid &= np.isfinite(state['mu'][i]) & np.isfinite(state['beta'][i]) & np.isfinite(state['vol'][i])
        directions[i] = valid.astype(directions.dtype)
    out['directions'] = directions
    return out


def replay(d, de, pred, cfg, protocol, arm, scenario, start, end, **kwargs):
    effective = copy.deepcopy(cfg)
    if '_refreshhedge_' in arm:
        effective['dense_policy']['source_lifetime_hours'] = cfg['source_validity_policy']['candidate_hours']
    run = paid_replay(d, de, pred, effective, protocol, arm.replace('_refreshhedge_', '_renewhedge_'),
                      scenario, start, end, **kwargs)
    run['arm'] = arm
    return run
