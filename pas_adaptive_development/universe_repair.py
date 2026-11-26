"""Minimal scope experiment: same seven-day reversion rule on all ten markets."""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .study import ROOT,OUT as PARENT,save,sha

OUT=ROOT/'reports/pas_universe_repair_20261021'
ARMS=['incumbent','size_only','all_reversion','all_reversion_restored']


def prepare():
    # Import the original strategy path before the live bundle alters sys.path.
    from strategy.agent import load_agent,profiles
    from replay_lab.common_comparison.models import Market
    from replay_lab.reversion_blend.model import build
    from replay_lab.common_comparison.engine import project
    from competition_live.profile import load_profile,verify_deployment
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    parent=json.loads((PARENT/'protocol.json').read_text());trading=load_profile()['trading'];cfg=profiles()
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,nominee='all_reversion_restored',
        hypothesis='The selected crypto_d7_aligned_add50 rule excludes seven authorized markets. Apply the EXACT same7day signal/cost/entry/exit rules to all ten, preserving trained base models and existing pair expert.',
        changes='crypto_d7_aligned_add50 -> all_d7_aligned_add50; same original size vs downscaled size ablation. No fitting or threshold changes.',
        model_reuse='Reuse existing Chronos forecasts with exact timestamp AND input-context keys. Reproduce default weights before accepting broadened weights.',
        selection='all_reversion_restored fixed nominee; original six windows and base/stress; all failed model families retained.',
        promotion='PnL and Q exceed incumbent AND size_only in continuous and later windows under both costs; no timing/accounting/hard-risk breaches; PAS separately.',
        independence=False,history_reused=True,orders_sent=0,live_changed=False,pointer=parent['pointer'],
        deployment=verify_deployment()['files'])
    save(OUT/'protocol.json',p)
    d=dict(np.load(ROOT/p['pointer']['inputs'],allow_pickle=False));old=dict(np.load(ROOT/p['pointer']['run']/'targets.npz'))
    m=Market(d['market_times'],d['close'],d['volume'],d['trades'],d['funding'],d['observed'],tuple(trading['symbols']))
    agent=load_agent('default');signal=agent.agent.signal
    for i,t in enumerate(m.times):
        if i>=1008 and t%21600==0:
            context=np.log(m.close[i-167:i+1].T).astype('float32')
            signal.cached[int(t),context.tobytes()]=old['chronos_forecast'][i].copy()
    original={k:v for k,v in d.items() if k not in ['market_times','funding','observed']}
    agent.prepare_history(m,original,funding_source='mainnet_reference_settled')
    np.testing.assert_array_equal(agent.weights,old['default'])
    candidates,_=build(m,agent.agent.sigma,agent.agent.frozen_baseline_weights)
    z=dict(np.load(PARENT/'targets.npz'));w={k:z[k] for k in ['incumbent','size_only']}
    w['all_reversion']=candidates['all_d7_aligned_add50']*cfg['default']['total_scale']
    w['all_reversion_restored']=candidates['all_d7_aligned_add50']*cfg['original']['total_scale']
    for name in ['all_reversion','all_reversion_restored']:
        for i in range(168,len(m.times)):w[name][i]=project(w[name][i],trading['initial'],z['sigma'][i],trading['risk'])
    np.savez_compressed(OUT/'targets.npz',times=m.times,sigma=z['sigma'],**w)
    save(OUT/'reproduction.json',dict(default_exact=True,chronos_recomputed=False,forecasts_reused=len(signal.cached),
        asset_counts={name:{'positive':(a>1e-12).sum(0).tolist(),'negative':(a< -1e-12).sum(0).tolist()} for name,a in w.items()}))
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in
        [Path(__file__),OUT/'protocol.json',OUT/'targets.npz',OUT/'reproduction.json']})
    print('Exact original reproduced; all-market reversion targets frozen',flush=True)


def run():
    from . import covariance_v2 as runner
    runner.OUT=OUT;runner.ARMS=ARMS;runner.run()


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
