"""Frozen, matched research comparison. Never follows live pointer after freeze."""
import copy
import gzip
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/auto_pair_selection_20261023'
from .selection import PAIRS, IDS, SYMBOLS, SelectionSpec
from .signals import prepare_features, fit_day, predict_hour, state_for_arm
from competition_live.profile import load_profile, BUNDLE, verify_deployment


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False,
                           default=lambda x:x.tolist())+'\n')


def freeze():
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    pointer=read(ROOT/'practice_live/latest.json')
    policy=read(ROOT/'.runtime/pas_budget_lifecycle/execution_policy.json')
    profile=load_profile();source=ROOT/pointer['run']/'inputs.npz'
    record=dict(created_utc=datetime.now(timezone.utc).isoformat(),
      primary='automatic_45',arms=['incumbent_fixed_two','fixed_two_rebuilt','automatic_45'],
      purpose='Replace the fixed pair allowlist with causal all-universe formation screening and the retained forecasting/execution policy.',
      universe=[list(x) for x in PAIRS],selection_spec=SelectionSpec().__dict__,
      selection='At00UTC, all45 canonical unordered pairs receive three30day return-correlation, raw-beta, beta-stability, coverage and reference-liquidity diagnostics. All eligible pairs compete; no forced top-K or named-asset quota. The cost-sensitive model and current venue execution planner decide whether and which one to enter.',
      reasons='Positive beta within retained hedge bounds; relationship must survive all three subwindows; median4000 notional gives10 units of quote at0.25% participation. This is a heuristic formation screen, not a statistical-profit certificate or cointegration claim.',
      forecasts='Identical existing24h direct rolling ridge and SPO+ gate,90d window,6h sampling, strictly mature labels. No original pretraining contribution to the direct column. Fit additional legacy gates solely for the fixed-two parity control.',
      lifecycle='45 IDs are stable; selection applies only to NEW entries. Direct forecasts and all held-pair exits/stops/paid hedges remain available after deselection. One active pair and original capital/risk caps retained.',
      fixed='Current pullback_any3 base strategy,50k sleeve+50k cash reserve,fees,funding,lot sizes,risk limits,delayed independent-leg execution unchanged.',
      input=str(source.relative_to(ROOT)),input_sha256=sha(source),
      source_cutoff_utc=pointer.get('completed_bar_utc'),
      direct_spec=read(ROOT/'competition_live/strategy_plan.json')['direct_spec'],gate_spec=profile['gate_spec'],
      start=1785542400, # August1 00UTC, before first evaluation start at12UTC.
      history_reused=True,independent_test=False,no_parameter_search=True,
      evaluation='All original August,later,and continuous reference windows base/stress; report failures. Fixed-two rebuilt must reproduce incumbent forecasts and PnL to isolate selection changes. PAS uses available public observations separately; never label reference returns PAS.',
      promotion='No automatic deployment. Require functional invariants, existing-path parity, zero accounting/timing/hard-risk violations and demonstrated benefit across costs before recommendation; no profit guarantee.',
      live_source_hashes={k:sha(ROOT/k) for k in policy['source_sha256']},
      live_configs={k:sha(ROOT/k) for k in ['.runtime/pas_budget_lifecycle/execution_policy.json','.runtime/pas_budget_lifecycle/strategy_revision.json','competition_live/config.json']},
      orders_sent=0,live_changed=False)
    save(OUT/'protocol.json',record)
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in [*Path(__file__).parent.glob('*.py'),OUT/'protocol.json']})
    print('FROZEN',record['input'],flush=True)


def verify():
    for path,digest in read(OUT/'freeze.json').items():
        assert sha(ROOT/path)==digest,path
    p=read(OUT/'protocol.json')
    for path,digest in {**p['live_source_hashes'],**p['live_configs']}.items():
        assert sha(ROOT/path)==digest,'Active code/config changed: '+path
    assert sha(ROOT/p['input'])==p['input_sha256']
    verify_deployment()
    return p


def build():
    p=verify();d=dict(np.load(ROOT/p['input'],allow_pickle=False));t=d['market_times']
    d={k:d[k] for k in ('market_times','close','volume','trades')}
    f,y,maturity,terminal=prepare_features(d)
    n=len(t);q=len(PAIRS)
    state=dict(dense=np.array(True),pairs=np.asarray(IDS),
      signal_times=np.broadcast_to(t[:,None],(n,q)).copy(),
      source_prices=d['close'],beta=f['beta'],vol=f['vol'],
      forecast_times=t.copy(),mu=np.full((n,q),np.nan),directions=np.zeros((n,q),int),
      selection_eligible=np.zeros((n,q),bool),gate_available=np.zeros((n,q),bool),
      gate_action=np.zeros((n,q),int),gate_score=np.full((n,q),np.nan),
      fit_source=np.full((n,q),-1,np.int64))
    controls=[PAIRS.index(('BTC','ETH')),PAIRS.index(('BTC','HYPE'))]
    summaries=[];spec=SelectionSpec(**p['selection_spec']);model=None
    days=np.unique(t[t>=p['start']]//86400*86400)
    for day in days:
        if day not in t:continue
        cache=OUT/'models'/f'{day}.json'
        begun=time.monotonic()
        if cache.exists():
            model=read(cache)
            for key in ('direct','gate'):model[key]={int(k):v for k,v in model[key].items()}
        else:
            model=fit_day(d,f,y,maturity,terminal,int(day),p['direct_spec'],p['gate_spec'],spec,controls)
            save(cache,model)
        eligible=np.asarray(model['selection']['selected'])
        for i in np.flatnonzero((t>=day)&(t<day+86400)):
            a=predict_hour(f,i,model,p['direct_spec'],p['gate_spec'])
            for key in ('mu','directions','gate_available','gate_action','gate_score'):state[key][i]=a[key]
            state['selection_eligible'][i]=eligible
            state['fit_source'][i]=np.where(a['directions'],day,-1)
        summary=dict(source=int(day),selected_pairs=[list(PAIRS[k]) for k in np.flatnonzero(eligible)],
            direct_models=len(model['direct']),gate_models=len(model['gate']),failures=model['failures'],
            max_training_maturity=max([v['last_maturity'] for v in model['direct'].values()],default=None))
        if summary['max_training_maturity'] is not None:assert summary['max_training_maturity']<day
        summaries.append(summary)
        print(datetime.fromtimestamp(int(day),timezone.utc).date(), 'eligible',int(eligible.sum()),
              'gates',len(model['gate']),'seconds',round(time.monotonic()-begun,2),flush=True)
    np.savez_compressed(OUT/'states.npz',times=t,**state)
    save(OUT/'selection_history.json',summaries)
    save(OUT/'feature_metadata.json',dict(names=f['names'],pairs=[list(x) for x in PAIRS],symbols=list(SYMBOLS)))
    # The same old two pair forecasts must be reproduced before any candidate
    # performance claim. Scalar floating-point tolerances are explicit.
    old=dict(np.load(BUNDLE/'models/reference_gate_state.npz',allow_pickle=False))
    reference=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False));ix=np.searchsorted(t,reference['times'])
    np.testing.assert_array_equal(t[ix],reference['times'])
    np.testing.assert_allclose(d['close'][ix],reference['reference_close'],rtol=0,atol=1e-10)
    mask=reference['times']>=p['start'];errors=[]
    for old_k,pair in enumerate(load_profile()['pairs']):
        k=PAIRS.index(tuple(pair));a=state['mu'][ix,k][mask];b=old['mu'][mask,old_k]
        valid=np.isfinite(a)&np.isfinite(b)
        error=float(np.max(abs(a[valid]-b[valid]))) if valid.any() else None
        availability_equal=np.array_equal(state['directions'][ix,k][mask].astype(bool),old['directions'][mask,old_k].astype(bool))
        action_equal=np.array_equal(state['gate_action'][ix,k][mask],old['economic_action'][mask,old_k])
        errors.append(dict(pair=pair,max_forecast_difference=error,direction_availability_equal=availability_equal,gate_actions_equal=action_equal))
        assert error is not None and error<1e-10 and availability_equal and action_equal,errors[-1]
    save(OUT/'incumbent_parity.json',dict(passed=True,pairs=errors))
    verify()


def reference():
    from pas_economic_allocation.replay import replay
    from pas_performance_research.selected import controller
    from pas_allocation_study.study import summarize
    p=verify();assert read(OUT/'incumbent_parity.json')['passed']
    profile=load_profile();d=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False))
    incumbent=dict(np.load(BUNDLE/'models/reference_gate_state.npz',allow_pickle=False))
    state=dict(np.load(OUT/'states.npz',allow_pickle=False));st=state.pop('times');ii=np.searchsorted(st,d['times'])
    mapped={k:(v[ii] if isinstance(v,np.ndarray) and v.ndim>0 and len(v)==len(st) else v) for k,v in state.items()}
    raw=dict(np.load(ROOT/p['input'],allow_pickle=False))
    z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz',allow_pickle=False))
    ix=np.searchsorted(z['times'],d['times'])
    f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz',allow_pickle=False));fi=np.searchsorted(f['times'],d['times'])
    windows=[w for w in read(BUNDLE/'expected.json')['runs'] if w['venue']=='reference' and w['arm']==profile['candidate']]
    rows=[]
    prior=read(ROOT/'reports/pas_performance_research_20261022/all_reference_results.json')
    for w in windows:
        for arm in p['arms']:
            cfg=copy.deepcopy(profile['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
            signals=incumbent if arm=='incumbent_fixed_two' else state_for_arm(mapped,arm)
            if arm!='incumbent_fixed_two':cfg['dense_pairs']=[list(x) for x in PAIRS]
            path=OUT/'reference'/f"{arm}_{w['phase']}_{w['scenario']}.json.gz"
            if path.exists():
                with gzip.open(path,'rt') as fp:r=json.load(fp)
            else:
                c=controller(d['times'],f['forecast'][fi],f['carry'][fi],z['all_market_signed_core_v1'][ix],
                    z['all_market_signed_core_v1_protected'][ix],price_times=raw['market_times'],prices=raw['close'])
                r=replay(d,signals,None,cfg,profile['trading'],profile['candidate'],w['scenario'],w['start'],w['end'],base_policy=c)
                path.parent.mkdir(exist_ok=True)
                with gzip.open(path,'wt') as fp:json.dump(r,fp,default=lambda x:x.tolist(),allow_nan=False)
            row=summarize(r,arm,w['phase'],w['scenario'])
            row['entered_pairs']=sorted({tuple(x['pair']) for x in r['events'] if x['event']=='pair_intent'})
            rows.append(row)
            assert row['timing_pass'] and row['accounting_error']<1e-6
            if arm!='automatic_45':
                previous=next(x for x in prior if x['arm']=='pullback_any3' and x['phase']==w['phase'] and x['scenario']==w['scenario'])
                assert abs(row['net_pnl']-previous['net_pnl'])<1e-6,(arm,row['net_pnl'],previous['net_pnl'])
            save(OUT/'reference_results.json',rows)
            print(json.dumps({k:row[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs','risk_breach_bars']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
