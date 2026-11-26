"""Generalize existing independent EMA rules without changing their parameters.

The legacy implementation reads three columns. Feed each asset through those
same slots, then restore its identity; no new EMA/ATR/exit logic is introduced.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .study import ROOT, OUT as PARENT, save, sha

OUT = ROOT / 'reports/pas_all_market_ema_20261022'
ARMS = ['incumbent', 'size_only', 'all_long', 'all_signed', 'all_signed_restored']


def extend(original, selected, volatility, signed):
    from replay_lab.ema_development.lifecycle import features, fit, Policy
    from replay_lab.ema_rebuild.models import add_shorts
    n = original['close'].shape[1]
    result = np.zeros_like(original['close']); signal = result.copy()
    p = Policy(**selected['policy'])
    for begin in range(0, n, 3):
        ids = list(range(begin, min(begin+3, n)))
        order = ids + [j for j in range(n) if j not in ids]
        data = {k:(v[:,order].copy() if isinstance(v,np.ndarray) and v.ndim==2 and v.shape[1]==n else v)
                for k,v in original.items()}
        f = features(data)
        model,_ = fit(data, p, volatility[:,order], detail=False, f=f)
        if signed:
            model = add_shorts(data, model, p, f)
        result[:,ids] = model['target_weight'][:,:len(ids)]
        signal[:,ids] = model['signal'][:,:len(ids)]
    return dict(target_weight=result, signal=signal, volatility=volatility.copy())


def prepare():
    from strategy.agent import load_agent, profiles
    from replay_lab.common_comparison.models import Market
    from replay_lab.common_comparison.engine import project
    from replay_lab.macro_development import candidate_agent as macro
    from competition_live.profile import load_profile, verify_deployment
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists(): raise FileExistsError('Already frozen')
    parent=json.loads((PARENT/'protocol.json').read_text()); trading=load_profile()['trading'];cfg=profiles()
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(), arms=ARMS,nominee='all_signed_restored',
        hypothesis='Both legacy long EMA and its short extension hard-code three columns. Apply the identical independent asset rule to every authorized symbol, preserving parameters and downstream models.',
        changes='Only macro EMA output is broadened. Existing crypto-reversion and pair expert retained. No forced trading, no minimum invested capital, no new fit.',
        controls='all_long separates universe scope from enabling the existing short branch; size_only separates sizing from signal changes.',
        promotion='Beat incumbent AND size_only PnL/Q in continuous and later windows, base/stress, zero risk/timing/accounting violations; no PAS claim from reference prices.',
        independent=False,history_reused=True,orders_sent=0,live_changed=False,
        pointer=parent['pointer'],deployment=verify_deployment()['files'])
    save(OUT/'protocol.json',p)
    d=dict(np.load(ROOT/p['pointer']['inputs'],allow_pickle=False));old=dict(np.load(ROOT/p['pointer']['run']/'targets.npz'))
    m=Market(d['market_times'],d['close'],d['volume'],d['trades'],d['funding'],d['observed'],tuple(trading['symbols']))
    original={k:v for k,v in d.items() if k not in ['market_times','funding','observed']}
    def agent():
        a=load_agent('default')
        for i,t in enumerate(m.times):
            if i>=1008 and t%21600==0:
                key=int(t),np.log(m.close[i-167:i+1].T).astype('float32').tobytes()
                a.agent.signal.cached[key]=old['chronos_forecast'][i].copy()
        return a
    a=agent();a.prepare_history(m,original,funding_source='mainnet_reference_settled')
    np.testing.assert_array_equal(a.weights,old['default'])
    z=dict(np.load(PARENT/'targets.npz'));w={k:z[k] for k in ['incumbent','size_only']};builder=macro.build
    checks=[]
    for name,signed in [('all_long',False),('all_signed',True)]:
        def broadened(data,manifest,selected):
            b=builder(data,manifest,selected)
            expanded=extend(data,selected,b['ema_base']['volatility'],signed)
            legacy=b['ema_signed' if signed else 'ema_base']
            np.testing.assert_array_equal(expanded['target_weight'][:,:3],legacy['target_weight'][:,:3])
            b['ema_base']=expanded
            return b
        a=agent()
        with patch.object(macro,'build',broadened):
            a.prepare_history(m,original,funding_source='mainnet_reference_settled')
        w[name]=a.weights.copy();checks.append(name)
    w['all_signed_restored']=w['all_signed']*cfg['original']['total_scale']/cfg['default']['total_scale']
    for name in ARMS[2:]:
        for i in range(168,len(m.times)):
            w[name][i]=project(w[name][i],trading['initial'],z['sigma'][i],trading['risk'])
    np.savez_compressed(OUT/'targets.npz',times=m.times,sigma=z['sigma'],**w)
    save(OUT/'reproduction.json',dict(original_exact=True,legacy_crypto_exact=checks,
        counts={name:dict(positive=(x>1e-12).sum(0).tolist(),negative=(x< -1e-12).sum(0).tolist()) for name,x in w.items()}))
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in
        [Path(__file__),OUT/'protocol.json',OUT/'targets.npz',OUT/'reproduction.json',
         ROOT/'replay_lab/ema_development/lifecycle.py',ROOT/'replay_lab/ema_rebuild/models.py']})
    print('All-market EMA prepared; legacy crypto output reproduced exactly',flush=True)


def run():
    from . import covariance_v2 as runner
    runner.OUT=OUT;runner.ARMS=ARMS;runner.run()


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
