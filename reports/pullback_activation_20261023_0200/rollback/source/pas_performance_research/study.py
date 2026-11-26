"""Reproducible comparisons against the actual all-market deployed core.

Every interval here was already inspected. These are development experiments,
not an independent test or a claim to exact historical PAS executions.
"""
import copy, gzip, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/pas_performance_research_20261022'
BASE = 'all_market_signed_core_v1'
ARMS = [BASE, 'momentum_agreement', 'trend_agreement', 'forecast_agreement', 'cost_positive_core']

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x:x.tolist(), allow_nan=False))

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def live_files():
    files = list((ROOT/'candidate_handoff').glob('*.py')) + list((ROOT/'universe_repair').glob('*.py'))
    files += list((ROOT/'pas_economic_allocation').glob('*.py'))
    files += [ROOT/'.runtime/pas_budget_lifecycle/strategy_revision.json', ROOT/'.runtime/pas_budget_lifecycle/team_execution_contract.json']
    return {str(f.relative_to(ROOT)):sha(f) for f in files if f.exists()}

def masks(d, forecast, carry, anchor):
    lp=pd.DataFrame(np.log(d['close']))
    a=(lp-lp.shift(168)).to_numpy(); b=(lp-lp.shift(672)).to_numpy()
    # Completed information only; no labels, named assets, or date rules.
    s=np.sign(anchor)
    m=.5*(s*np.sign(a)>0)+.5*(s*np.sign(b)>0)
    trend=(s*np.sign(b)>0).astype(float)
    edge=s*forecast-np.where(s*carry>0,s*carry,.5*s*carry)
    agreement=(edge>0).astype(float)
    economic=np.clip((edge-.0043)/.0043,0,1)
    return dict(momentum_agreement=m, trend_agreement=trend,
                forecast_agreement=agreement, cost_positive_core=economic)

def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Experiment already sealed')
    p=json.loads((ROOT/'reports/universe_repair_20261022/protocol.json').read_text())
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),baseline=BASE,arms=ARMS,
        input=p['input'],live_files=live_files(),orders_sent=0,live_changed=False,
        history_reused=True,independent_test=False,
        hypotheses=['The signed core combines contradictory horizons without requiring directional agreement.',
          'A protected risk reduction or new directional veto must not be refilled by the auxiliary allocator.',
          'Core targets lack a cost-positive check despite a retained, purged weekly price forecast.'],
        definitions=dict(momentum_agreement='Retain 0, half, or all of the core when 7d and 28d completed-price directions agree.',
          trend_agreement='Retain the core only when its side agrees with 28d completed-price direction.',
          forecast_agreement='Retain the core only when the frozen weekly forecast, less conservatively estimated funding, has positive signed edge.',
          cost_positive_core='Retain a smooth fraction when that edge exceeds 43bp roundtrip costs; full fraction at86bp.'),
        fixed='All10 eligible, symmetric sides, same50k sleeve+50k reserve, same risk caps, pair expert, forecasts, lot sizes, costs and delayed partial-fill engines. No leverage or capital increase.',
        evaluation='All six reference windows and all three existing PAS scenarios. Show failures, including Aug and Sep-Oct separately. PnL, Q, drawdown, turnover, exposure, terminal reserve and risk violations.',
        success='A candidate improves PnL and Q versus actual baseline in continuous and later reference base/stress and PAS1%/0.25%; accounting/timing/risk checks pass. Robustness and additional interval tests precede any recommendation.',
        selection='These are five declared development arms. Any later variation is a separate recorded wave, never retrospectively described as an untouched test.',
        venue_limit='PAS trade-tape conditional replay lacks historical queues/L2/marks/oracles; no claim of exact realized performance.')
    save(OUT/'protocol.json',protocol)
    d=dict(np.load(ROOT/p['input']['inputs']));z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz'))
    f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz'))
    np.testing.assert_array_equal(d['market_times'],z['times']);np.testing.assert_array_equal(f['times'],z['times'])
    anchor=z[BASE];protected=z[BASE+'_protected'];result={'times':z['times'],BASE:anchor,BASE+'_protected':protected}
    gates=masks(d,f['forecast'],f['carry'],anchor)
    for arm,g in gates.items():
        g=np.nan_to_num(g,nan=0.)
        result[arm]=anchor*g
        result[arm+'_protected']=protected|((anchor!=0)&(g<1))
    np.savez_compressed(OUT/'targets.npz',**result)
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'protocol.json',OUT/'targets.npz']})
    print('Research frozen',flush=True)

def verify():
    for f,h in json.loads((OUT/'freeze.json').read_text()).items():assert sha(ROOT/f)==h,f
    assert live_files()==json.loads((OUT/'protocol.json').read_text())['live_files'],'Live source/config changed concurrently'

def reference():
    from competition_live.profile import load_profile,BUNDLE
    from pas_economic_allocation.replay import replay
    from pas_economic_allocation.lifecycle import LifecycleController
    from pas_allocation_study.study import summarize
    verify();p=load_profile();d=dict(np.load(BUNDLE/'data/reference.npz'));states=dict(np.load(BUNDLE/'models/reference_gate_state.npz'))
    z=dict(np.load(OUT/'targets.npz'));ix=np.searchsorted(z['times'],d['times'])
    f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz'));fi=np.searchsorted(f['times'],d['times'])
    np.testing.assert_array_equal(z['times'][ix],d['times']);np.testing.assert_array_equal(f['times'][fi],d['times'])
    cfg=copy.deepcopy(p['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
    windows=[w for w in json.loads((BUNDLE/'expected.json').read_text())['runs'] if w['venue']=='reference' and w['arm']==p['candidate']]
    rows=[]
    for w in windows:
        for arm in ARMS:
            file=OUT/'reference'/f"{arm}_{w['phase']}_{w['scenario']}.json.gz"
            if file.exists():
                with gzip.open(file,'rt') as fp:r=json.load(fp)
            else:
                c=LifecycleController(d['times'],f['forecast'][fi],f['carry'][fi],z[arm][ix],z[arm+'_protected'][ix],5.)
                r=replay(d,states,None,cfg,p['trading'],p['candidate'],w['scenario'],w['start'],w['end'],base_policy=c)
                file.parent.mkdir(exist_ok=True)
                with gzip.open(file,'wt') as fp:json.dump(r,fp,default=lambda x:x.tolist(),allow_nan=False)
            row=summarize(r,arm,w['phase'],w['scenario']);rows.append(row)
            assert row['timing_pass'] and row['accounting_error']<1e-6
            if arm==BASE:
                original=next(o for o in json.loads((ROOT/'reports/universe_repair_20261022/reference_results.json').read_text()) if o['arm']==BASE and o['phase']==w['phase'] and o['scenario']==w['scenario'])
                assert abs(row['net_pnl']-original['net_pnl'])<1e-7
            save(OUT/'reference_results.json',rows)
            print(json.dumps({k:row[k] for k in ['arm','phase','scenario','net_pnl','Q','risk_breach_bars']}),flush=True)
    verify()

def pas():
    from competition_live.profile import load_profile
    from pas_counterfactual_20261022.engine import simulate
    from pas_counterfactual_20261022.prepare import END
    from pas_strategy_audit.data import load
    from pas_economic_allocation.lifecycle import LifecycleController
    verify();old=ROOT/'reports/pas_counterfactual_20261022';read=lambda n:json.loads((old/n).read_text())
    assert all(f['economic_available'] and f['economic_action']==0 for row in read('pair_forecasts.json') for f in row['forecasts'] if f['model']=='decision_gate')
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
            file=OUT/'pas'/f'{arm}_{name}.json.gz';file.parent.mkdir(exist_ok=True)
            with gzip.open(file,'wt') as fp:json.dump(r,fp,default=lambda x:x.tolist(),allow_nan=False)
            save(OUT/'pas_results.json',rows)
            print(json.dumps({k:row[k] for k in ['arm','scenario','net_pnl','Q','risk_breach_events']}),flush=True)
    verify()

if __name__=='__main__':globals()[sys.argv[1]]()
