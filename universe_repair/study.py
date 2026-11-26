"""Matched, predeclared scope repair comparison; all history is development."""
import copy, gzip, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .core import calculate, audit, ARMS, REVISION, LEGACY
from pas_adaptive_development.study import ROOT, save, sha, native
from pas_economic_allocation.anchored import RATIO

OUT=ROOT/'reports/universe_repair_20261022'


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists(): raise FileExistsError('Frozen comparison exists')
    prior=json.loads((ROOT/'reports/pas_adaptive_development_20261021/protocol.json').read_text())
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(), primary=REVISION, arms=list(ARMS),
        input=prior['pointer'], independent=False, history_reused=True, model_refit=False,
        purpose='Repair the main EMA/channel/reversion universe and symmetric entry logic; no outcome-based selection.',
        changes='All ten core assets, symmetric trend. Existing stops and cross-exit setting retained. Frozen two-pair specialist retained; it does not gate core assets.',
        fixed='Same 50000 budget plus50000 reserve, scale2*RATIO, final risk gates, cost model, forecasts, pair model, funding and lifecycle.',
        validity='All ten markets reach target and executor; prefix causality, permutation symmetry, no forced holdings, no accounting/timing/risk errors.',
        performance='Report all three predetermined arms over all six reference windows and three PAS execution scenarios. No winner substitution or further tuning on these outcomes.',
        limitations=['Reused development history, not independent validation.',
            'Reference profitability does not equal PAS profitability. PAS48h conditional trade-tape replay lacks historical L2/mark/oracle.',
            'Pair expert remains BTC/ETH and BTC/HYPE; core eligibility now covers every listed market.'],
        orders_sent=0)
    save(OUT/'protocol.json',p)
    d=dict(np.load(ROOT/p['input']['inputs']));z=dict(np.load(ROOT/p['input']['run']/'targets.npz'))
    result={'times':d['market_times']};latest={}
    for arm in ARMS:
        anchor,gate,parts=calculate(d,z['chronos_forecast'],z['frozen_regime_probability'],arm)
        result[arm]=anchor*RATIO*2;result[arm+'_protected']=gate['protected']
        latest[arm]=audit(parts,gate,anchor)
        if arm==LEGACY:
            frozen=dict(np.load(ROOT/'reports/strategy_repairs_20261021/final_risk_targets.npz'))
            ix=np.searchsorted(d['market_times'],frozen['times'])
            np.testing.assert_allclose(anchor[ix],frozen['final_mix_risk'],atol=1e-10,rtol=0)
        print('prepared',arm,flush=True)
    np.savez_compressed(OUT/'targets.npz',**result);save(OUT/'coverage.json',latest)
    files=[*Path(__file__).parent.glob('*.py'),OUT/'protocol.json',OUT/'targets.npz',ROOT/p['input']['inputs'],ROOT/p['input']['run']/'targets.npz']
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in files})


def reference():
    from competition_live.profile import load_profile,BUNDLE
    from pas_economic_allocation.replay import replay
    from pas_economic_allocation.lifecycle import LifecycleController
    from pas_allocation_study.study import summarize
    p=load_profile();d=dict(np.load(BUNDLE/'data/reference.npz'));states=dict(np.load(BUNDLE/'models/reference_gate_state.npz'))
    z=dict(np.load(OUT/'targets.npz'));ix=np.searchsorted(z['times'],d['times'])
    f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz'));fi=np.searchsorted(f['times'],d['times'])
    np.testing.assert_array_equal(z['times'][ix],d['times']);np.testing.assert_array_equal(f['times'][fi],d['times'])
    cfg=copy.deepcopy(p['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
    windows=[w for w in json.loads((BUNDLE/'expected.json').read_text())['runs'] if w['venue']=='reference' and w['arm']==p['candidate']]
    rows=[]
    for w in windows:
        for arm in ARMS:
            c=LifecycleController(d['times'],f['forecast'][fi],f['carry'][fi],z[arm][ix],z[arm+'_protected'][ix],5.)
            r=replay(d,states,None,cfg,p['trading'],p['candidate'],w['scenario'],w['start'],w['end'],base_policy=c)
            row=summarize(r,arm,w['phase'],w['scenario']);rows.append(row)
            assert row['timing_pass'] and row['accounting_error']<1e-6
            (OUT/'reference').mkdir(exist_ok=True)
            with gzip.open(OUT/'reference'/f"{arm}_{w['phase']}_{w['scenario']}.json.gz",'wt') as file:
                json.dump(r,file,default=native,allow_nan=False)
            save(OUT/'reference_results.json',rows);print(json.dumps(row),flush=True)


def pas():
    from competition_live.profile import load_profile
    from pas_counterfactual_20261022.engine import simulate
    from pas_counterfactual_20261022.prepare import END
    from pas_strategy_audit.data import load
    from pas_economic_allocation.lifecycle import LifecycleController
    old=ROOT/'reports/pas_counterfactual_20261022'
    read=lambda n:json.loads((old/n).read_text())
    pairs=read('pair_forecasts.json')
    assert all(f['economic_available'] and f['economic_action']==0 for row in pairs for f in row['forecasts'] if f['model']=='decision_gate')
    cfg,d=load(ROOT/'reports/pas_allocation_study_20261021/pas');cfg['cutoff_ms']=END*1000
    p=load_profile();events=read('events.json');tape=read('trade_tape.json');funding=read('funding_tape.json')
    known={(r['coin'],r['time']//3600000) for r in funding}
    for r in d['funding_events']:
        coin='pas:'+cfg['symbols'][r['j']]
        if (coin,r['time']//3600000) not in known:funding.append(dict(coin=coin,time=r['time'],rate=r['rate'],cash_per_unit=None))
    f=dict(np.load(old/'candidate_inputs.npz'));z=dict(np.load(OUT/'targets.npz'));ix=np.searchsorted(z['times'],f['times'])
    np.testing.assert_array_equal(z['times'][ix],f['times']);rows=[]
    for name,sc in read('protocol.json')['scenarios'].items():
        for arm in ARMS:
            c=LifecycleController(f['times'],f['forecast'],f['carry'],z[arm][ix],z[arm+'_protected'][ix],5.)
            r=simulate(cfg,d,copy.deepcopy(events),tape,funding,p['trading'],sc,controller=c)
            row=dict(arm=arm,scenario=name,**r['metrics']);rows.append(row)
            assert row['timing_pass'] and row['maximum_accounting_error']<1e-6
            (OUT/'pas').mkdir(exist_ok=True)
            with gzip.open(OUT/'pas'/f'{arm}_{name}.json.gz','wt') as file:json.dump(r,file,default=native,allow_nan=False)
            save(OUT/'pas_results.json',rows);print(json.dumps(row),flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1]!='prepare':
        for name,digest in json.loads((OUT/'source_freeze.json').read_text()).items():assert sha(ROOT/name)==digest,name
    globals()[sys.argv[1]]()
