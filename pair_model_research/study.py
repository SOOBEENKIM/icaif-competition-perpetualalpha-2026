import copy
import gzip
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import xgboost as xgb
from auto_pair_selection.study import ROOT,OUT as PRIOR,read,save,sha,verify as verify_prior
from auto_pair_selection.selection import PAIRS,IDS
from auto_pair_selection.signals import prepare_features
from experiments.dense_history_pairs_20261017.model import labels
from competition_live.profile import BUNDLE,load_profile
from .models import fit,predict

OUT=ROOT/'reports/pair_model_research_20261023'
SPEC=dict(lookback_days=90,half_life_days=30,sample_hours=6,minimum_source_times=100,
          ridge_penalty=1.,intercept_prior_sources=12,target_clip=4.,prediction_clip=4.,
          tree=dict(max_depth=3,eta=.05,min_child_weight=8,reg_lambda=3.,subsample=1.,colsample_bytree=1.,rounds=128))
ARMS=[f'{kind}_{h}' for h in [24,72,168] for kind in ['shared_ridge','shared_ensemble']]


def freeze():
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    old=verify_prior()
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),input=old['input'],input_sha256=old['input_sha256'],
      wave=1,arms=ARMS,spec=SPEC,start=old['start'],horizons=[24,72,168],all_pairs=[list(x) for x in PAIRS],
      rationale='Independent per-pair prediction and over-selective after-the-fact monitors did not produce gains. Share statistical patterns across all45 source-normalized relative-return series; compare a regularized linear model against an equal linear/boosted-tree ensemble and economically longer horizons.',
      features='Existing20 causal price-only spread/common returns, deviations,beta,correlation and volatility features. No names, pair IDs or winner labels enter model features.',
      train='Daily90d rolling fits,6h sampled sources; source+h label strictly before fit. All45 targets from a source stay in the same past split, total source weight equal before30d recency decay. They are correlated targets, not independent observations.',
      action='All45 hourly forecasts compete in the existing net-cash planner. Directional entry needs price forecast magnitude>38bp proxy, then current funding/fees/depth/rounding/risk veto. One active pair, original6000 gross cap and50k own+50k reserve.',
      selection='StageA: run August base+stress for all6 candidates; choose maximal minimum Q advantage over incumbent, then minimum PnL advantage, on August only. Save choice before later replay. StageB: evaluate the chosen candidate and retain all attempted results. Later results are reused development evidence, not independent test.',
      target='Higher PnL and Q than current incumbent on full and later histories, both costs, with actual simulated pair trades and zero accounting/timing/risk violations. Also report August failures; no universal-profit claim.',
      scientific_limit='Repeated development on these histories creates selection bias even without temporal leakage. Report wave count and failures. No live installation in this experiment.',
      research=[dict(url='https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2026.1749337/full',
                     use='2026 work motivates dynamic pair relationships and ensemble forecasting; this is a different ridge/tree model and not a reproduction of their DNN/LSTM or reported results.'),
                dict(url='https://www.sciencedirect.com/science/article/pii/S0957417425021153',
                     use='Publisher abstract only: separates learned pair selection from execution. We do not claim a GAT/DRL implementation.')],
      history_reused=True,independent_test=False,orders_sent=0,live_changed=False)
    save(OUT/'protocol.json',p)
    save(OUT/'freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in [*Path(__file__).parent.glob('*.py'),OUT/'protocol.json']})
    print('Frozen six models/horizons before outcomes',flush=True)


def verify():
    verify_prior()
    for path,digest in read(OUT/'freeze.json').items():assert sha(ROOT/path)==digest,path
    p=read(OUT/'protocol.json');assert sha(ROOT/p['input'])==p['input_sha256']
    return p


def build():
    p=verify();d=dict(np.load(ROOT/p['input'],allow_pickle=False));t=d['market_times'];f,_,_,_=prepare_features(d)
    np.savez_compressed(OUT/'features.npz',times=t,x=f['x'],vol=f['vol'],fresh=f['fresh'],beta=f['beta'])
    for h in p['horizons']:
        y,maturity=labels(t,d['close'],f,h);terminal=np.zeros_like(f['fresh']);terminal[:-h]=f['fresh'][h:]
        means={k:np.full(y.shape,np.nan) for k in ['shared_ridge','shared_ensemble']};records=[]
        for day in np.unique(t[t>=p['start']]//86400*86400):
            begun=time.monotonic();stem=OUT/'models'/f'{h}_{day}';meta=stem.with_suffix('.json');weights=stem.with_suffix('.ubj')
            if meta.exists():
                model=read(meta);tree=xgb.Booster();tree.load_model(weights)
            else:
                model,tree=fit(t,f,y,maturity,terminal,int(day),p['spec'])
                if model is None:continue
                save(meta,model);tree.save_model(weights)
            ix=np.flatnonzero((t>=day)&(t<day+86400));a,b=predict(f,ix,model,tree,h,p['spec'])
            means['shared_ridge'][ix]=a;means['shared_ensemble'][ix]=b;records.append(model)
            assert model['last_maturity']<day
            print(h,datetime.fromtimestamp(int(day),timezone.utc).date(),model['rows'],round(time.monotonic()-begun,2),flush=True)
        for kind,mu in means.items():
            available=np.isfinite(mu)&f['fresh'];actions=np.where(available&(abs(mu)>.0038),np.sign(np.nan_to_num(mu)),0).astype(int)
            np.savez_compressed(OUT/f'{kind}_{h}.npz',times=t,dense=True,pairs=np.asarray(IDS),
              signal_times=np.broadcast_to(t[:,None],mu.shape),source_prices=d['close'],beta=f['beta'],vol=f['vol'],
              forecast_times=t,mu=mu,directions=available.astype(int),economic_available=available,economic_action=actions)
        save(OUT/f'fits_{h}.json',records)
    verify()


def job(arm,phase,scenario):
    from pas_economic_allocation.replay import replay
    from pas_performance_research.selected import controller
    from pas_allocation_study.study import summarize
    profile=load_profile();p=read(OUT/'protocol.json')
    w=next(w for w in read(BUNDLE/'expected.json')['runs'] if w['venue']=='reference' and w['arm']==profile['candidate'] and w['phase']==phase and w['scenario']==scenario)
    dest=OUT/'reference'/f'{arm}_{phase}_{scenario}.json.gz'
    if dest.exists():
        with gzip.open(dest,'rt') as fp:r=json.load(fp)
    else:
        d=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False))
        state=dict(np.load(OUT/f'{arm}.npz',allow_pickle=False));t=state.pop('times');ii=np.searchsorted(t,d['times'])
        assert np.array_equal(t[ii],d['times'])
        state={k:(v[ii] if v.ndim and len(v)==len(t) else v) for k,v in state.items()}
        z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz',allow_pickle=False));zi=np.searchsorted(z['times'],d['times'])
        f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz',allow_pickle=False));fi=np.searchsorted(f['times'],d['times'])
        raw=dict(np.load(ROOT/p['input'],allow_pickle=False))
        cfg=copy.deepcopy(profile['configs']['reference']);cfg['dense_pairs']=[list(x) for x in PAIRS];cfg['conditional_completion']={'risk_aversion':0.}
        c=controller(d['times'],f['forecast'][fi],f['carry'][fi],z['all_market_signed_core_v1'][zi],
                     z['all_market_signed_core_v1_protected'][zi],price_times=raw['market_times'],prices=raw['close'])
        h=int(arm.rsplit('_',1)[1]);run_arm=f'first_dense_hourlyhedge_shared_{h}'
        r=replay(d,state,None,cfg,profile['trading'],run_arm,scenario,w['start'],w['end'],base_policy=c)
        dest.parent.mkdir(exist_ok=True)
        with gzip.open(dest,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
    row=summarize(r,arm,phase,scenario)
    row['entered_pairs']=sorted({tuple(e['pair']) for e in r['events'] if e['event']=='pair_intent'})
    assert row['timing_pass'] and row['accounting_error']<1e-6 and row['risk_breach_bars']==0
    save(OUT/'rows'/f'{arm}_{phase}_{scenario}.json',row)
    return row


def run_jobs(jobs):
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(job,*args) for args in jobs]
        for done in as_completed(futures):
            r=done.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs']}),flush=True)


def stage_a():
    p=verify();run_jobs([(a,'validation',s) for a in p['arms'] for s in ['base','stress']])
    baseline=[r for r in read(PRIOR/'reference_results.json') if r['arm']=='incumbent_fixed_two' and r['phase']=='validation']
    ranking=[]
    for a in p['arms']:
        rs=[read(OUT/'rows'/f'{a}_validation_{s}.json') for s in ['base','stress']]
        ds=[(r['Q']-b['Q'],r['net_pnl']-b['net_pnl']) for r in rs for b in baseline if r['scenario']==b['scenario']]
        ranking.append(dict(arm=a,min_Q_advantage=min(x[0] for x in ds),min_pnl_advantage=min(x[1] for x in ds),
                            min_completed=min(r['completed_pairs'] for r in rs)))
    ranking.sort(key=lambda r:(r['min_Q_advantage'],r['min_pnl_advantage']),reverse=True)
    save(OUT/'stage_a_selection.json',dict(selected=ranking[0]['arm'],ranking=ranking,
                                         criterion='August worst-cost Q advantage, then PnL; fixed before later runs'))
    print('SELECTED',ranking[0],flush=True);verify()


def stage_b():
    verify();a=read(OUT/'stage_a_selection.json')['selected']
    run_jobs([(a,phase,s) for phase in ['comparison','continuous'] for s in ['base','stress']]);verify()


if __name__=='__main__':globals()[sys.argv[1]]()
