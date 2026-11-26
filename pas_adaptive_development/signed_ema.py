"""Isolated ablation of the existing, previously unselected short EMA branch.

No live modules are written. The original full pipeline is reproduced first;
the macro builder is replaced only inside this research process.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from .study import ROOT, OUT as PARENT, save, sha

OUT = ROOT / 'reports/pas_signed_ema_20261022'
ARMS = ['incumbent', 'size_only', 'signed_ema', 'signed_ema_restored',
        'signed_ema_all_reversion', 'signed_ema_all_reversion_restored']


def prepare():
    from strategy.agent import load_agent, profiles
    from replay_lab.common_comparison.models import Market
    from replay_lab.common_comparison.engine import project
    from replay_lab.reversion_blend.model import build
    from replay_lab.macro_development import candidate_agent as macro
    from competition_live.profile import load_profile, verify_deployment
    OUT.mkdir(exist_ok=True)
    if (OUT / 'protocol.json').exists():
        raise FileExistsError('Already frozen')
    parent = json.loads((PARENT / 'protocol.json').read_text())
    trading = load_profile()['trading']; cfg = profiles()
    p = dict(created_utc=datetime.now(timezone.utc).isoformat(), arms=ARMS,
        nominee='signed_ema_restored',
        hypothesis='Selected macro price expert uses ema_base although a symmetric short lifecycle already exists as ema_signed. Test this exact existing branch with unchanged gates, pair expert, costs, risk caps and downstream sizing.',
        universe='EMA implementation remains crypto-three. Separate all-reversion arms extend the existing seven-day reversion rule to ten assets; never claim EMA itself covers ten.',
        fitting='No parameter search or retraining. Reuse exact existing Chronos input-key forecasts. Existing frozen regime/calibration states unchanged.',
        controls='Original-size control changes only base size by 1.44117372816997; pair capital stays fixed in every arm.',
        selection='Nominee fixed before outcomes. All-reversion arms are diagnostics, not fallback winner selection.',
        promotion='Beat incumbent AND size_only PnL/Q in continuous and later windows, base and stress, with zero hard-risk/accounting/timing breaches. PAS execution needs separate verification.',
        independent=False, history_reused=True, orders_sent=0, live_changed=False,
        pointer=parent['pointer'], deployment=verify_deployment()['files'])
    save(OUT / 'protocol.json', p)
    d = dict(np.load(ROOT / p['pointer']['inputs'], allow_pickle=False))
    old = dict(np.load(ROOT / p['pointer']['run'] / 'targets.npz'))
    m = Market(d['market_times'], d['close'], d['volume'], d['trades'],
               d['funding'], d['observed'], tuple(trading['symbols']))
    original = {k:v for k,v in d.items() if k not in ['market_times','funding','observed']}
    def agent():
        a = load_agent('default')
        for i,t in enumerate(m.times):
            if i >= 1008 and t % 21600 == 0:
                key = int(t), np.log(m.close[i-167:i+1].T).astype('float32').tobytes()
                a.agent.signal.cached[key] = old['chronos_forecast'][i].copy()
        return a
    baseline = agent()
    baseline.prepare_history(m, original, funding_source='mainnet_reference_settled')
    np.testing.assert_array_equal(baseline.weights, old['default'])
    original_builder = macro.build
    def signed_builder(*args, **kwargs):
        result = original_builder(*args, **kwargs)
        result['ema_base'] = result['ema_signed']
        return result
    changed = agent()
    with patch.object(macro, 'build', signed_builder):
        changed.prepare_history(m, original, funding_source='mainnet_reference_settled')
    z = dict(np.load(PARENT / 'targets.npz'))
    w = {k:z[k] for k in ['incumbent','size_only']}
    w['signed_ema'] = changed.weights.copy()
    ratio = cfg['original']['total_scale'] / cfg['default']['total_scale']
    w['signed_ema_restored'] = w['signed_ema'] * ratio
    candidates,_ = build(m, changed.agent.sigma, changed.agent.frozen_baseline_weights)
    w['signed_ema_all_reversion'] = candidates['all_d7_aligned_add50'] * cfg['default']['total_scale']
    w['signed_ema_all_reversion_restored'] = w['signed_ema_all_reversion'] * ratio
    for name in ARMS[2:]:
        for i in range(168, len(m.times)):
            w[name][i] = project(w[name][i], trading['initial'], z['sigma'][i], trading['risk'])
    np.savez_compressed(OUT / 'targets.npz', times=m.times, sigma=z['sigma'], **w)
    save(OUT / 'reproduction.json', dict(original_bitwise_exact=True, original_builder_restored=macro.build is original_builder,
        forecast_key_cache_reused=True, counts={name:dict(positive=(x>1e-12).sum(0).tolist(),negative=(x< -1e-12).sum(0).tolist()) for name,x in w.items()}))
    sources = [Path(__file__), OUT/'protocol.json', OUT/'targets.npz', OUT/'reproduction.json',
               ROOT/'replay_lab/ema_rebuild/models.py', ROOT/'replay_lab/macro_development/candidate_agent.py']
    save(OUT/'source_freeze.json', {str(f.relative_to(ROOT)):sha(f) for f in sources})
    print('Original reproduced; existing short branch activated in isolated process; targets frozen', flush=True)


def run():
    from . import covariance_v2 as runner
    runner.OUT = OUT; runner.ARMS = ARMS; runner.run()


if __name__ == '__main__':
    import sys
    globals()[sys.argv[1]]()
