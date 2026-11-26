"""Frozen long EMA and a symmetric short lifecycle; no fitted return predictor."""
import numpy as np
from ..ema_development.lifecycle import Policy, features, fit
from ..common_comparison.models import retained_models


def add_shorts(original, long_model, p, f):
    out = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in long_model.items()}
    state = np.zeros(3, dtype=bool)
    stops = np.full(3, np.inf)
    prior_exit = np.zeros(3, dtype=np.int64)
    fast, slow = f['ema'][p.frame][p.fast], f['ema'][p.frame][p.slow]
    for i, time in enumerate(original['known']):
        if time % (p.decision_hours*3600) == 0:
            price, atr = original['close'][i, :3], f['atr'][i]
            ready = np.isfinite(atr) & np.isfinite(slow[i])
            stops = np.where(state, np.minimum(stops, price+p.atr_multiple*atr), stops)
            exits = state & ((price >= stops) | (long_model['signal'][i, :3] > 0))
            state[exits] = False; prior_exit[exits] = time
            down = ready & (fast[i] < slow[i]) & (price < fast[i]/1.0038) & (f['eff'][i] <= -p.confidence)
            enter = down & ~state & ~exits & (long_model['signal'][i, :3] == 0) & (time-prior_exit >= 86400)
            state[enter] = True; stops[enter] = price[enter]+p.atr_multiple*atr[enter]
        out['signal'][i, :3] = np.where(long_model['signal'][i, :3] > 0, 1, -state.astype(float))
    vol = out['volatility']
    out['target_weight'] = out['signal']*np.where(np.isfinite(vol), np.minimum(.25/np.maximum(vol, .01), 2)*.01, 0)*p.scale
    return out


def build(original, manifest, selected):
    controls, _ = retained_models(original, manifest)
    vol = controls['adaptive-horizons-lifecycle-d14-s1']['volatility']
    p = Policy(**selected['policy'])
    f = features(original)
    long, _ = fit(original, p, vol, detail=False, f=f)
    models = {**controls, 'ema_base': long, 'ema_signed': add_shorts(original, long, p, f)}
    for key in controls:
        if 'target_weight' not in models[key]:
            m = models[key]
            m['target_weight'] = m['signal']*np.where(np.isfinite(m['volatility']),
                np.minimum(.25/np.maximum(m['volatility'], .01), 2)*.01, 0)
    models['ema_vae'] = models['ema_base']
    models['ema_signed_vae'] = models['ema_signed']
    models['cash'] = {'target_weight': np.zeros_like(long['target_weight'])}
    return models
