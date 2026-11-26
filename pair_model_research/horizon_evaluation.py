"""Offline-only matched replay for new, frozen model-research waves."""
import copy,gzip,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .study import ROOT,PRIOR,read,save
from auto_pair_selection.selection import PAIRS
from competition_live.profile import BUNDLE,load_profile


def job(out,arm,phase,scenario):
    from pas_economic_allocation.replay import replay
    from pas_performance_research.selected import controller
    from pas_allocation_study.study import summarize
    out=Path(out);p=read(out/'protocol.json');profile=load_profile();dest=out/'reference'/f'{arm}_{phase}_{scenario}.json.gz'
    w=next(x for x in read(BUNDLE/'expected.json')['runs'] if x['venue']=='reference' and x['arm']==profile['candidate'] and x['phase']==phase and x['scenario']==scenario)
    if dest.exists():
        with gzip.open(dest,'rt') as fp:r=json.load(fp)
    else:
        d=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False));s=dict(np.load(out/f'{arm}.npz',allow_pickle=False));t=s.pop('times');ii=np.searchsorted(t,d['times'])
        assert np.array_equal(t[ii],d['times']);s={k:(v[ii] if v.ndim and len(v)==len(t) else v) for k,v in s.items()}
        z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz',allow_pickle=False));zi=np.searchsorted(z['times'],d['times'])
        f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz',allow_pickle=False));fi=np.searchsorted(f['times'],d['times'])
        raw=dict(np.load(ROOT/p['input'],allow_pickle=False));cfg=copy.deepcopy(profile['configs']['reference']);cfg['dense_pairs']=[list(x) for x in PAIRS];cfg['conditional_completion']={'risk_aversion':0.}
        c=controller(d['times'],f['forecast'][fi],f['carry'][fi],z['all_market_signed_core_v1'][zi],z['all_market_signed_core_v1_protected'][zi],price_times=raw['market_times'],prices=raw['close'])
        r=replay(d,s,None,cfg,profile['trading'],f"first_dense_hourlyhedge_individual_{int(arm.rsplit('_',1)[1])}",scenario,w['start'],w['end'],base_policy=c)
        dest.parent.mkdir(exist_ok=True)
        with gzip.open(dest,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
    row=summarize(r,arm,phase,scenario);row['entered_pairs']=sorted({tuple(e['pair']) for e in r['events'] if e['event']=='pair_intent'})
    assert row['timing_pass'] and row['accounting_error']<1e-6
    row['eligible']=row['risk_breach_bars']==0;save(out/'rows'/f'{arm}_{phase}_{scenario}.json',row);return row


def run_jobs(out,jobs):
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(job,out,*j) for j in jobs]):
            r=future.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs','risk_breach_bars']}),flush=True)


def stage_a(out,arms):
    run_jobs(out,[(a,'validation',s) for a in arms for s in ['base','stress']])
    b=[r for r in read(PRIOR/'reference_results.json') if r['arm']=='incumbent_fixed_two' and r['phase']=='validation'];ranking=[]
    for a in arms:
        rs=[read(out/'rows'/f'{a}_validation_{s}.json') for s in ['base','stress']]
        ds=[(r['Q']-x['Q'],r['net_pnl']-x['net_pnl']) for r in rs for x in b if r['scenario']==x['scenario']]
        ranking.append(dict(arm=a,eligible=all(r['eligible'] for r in rs),min_Q_advantage=min(x[0] for x in ds),min_pnl_advantage=min(x[1] for x in ds)))
    ranking.sort(key=lambda r:(r['eligible'],r['min_Q_advantage'],r['min_pnl_advantage']),reverse=True)
    save(out/'selection.json',dict(selected=ranking[0]['arm'] if ranking[0]['eligible'] else None,ranking=ranking));print('SELECTED',ranking[0],flush=True)


def stage_b(out):
    a=read(out/'selection.json')['selected'];assert a is not None
    run_jobs(out,[(a,p,s) for p in ['comparison','continuous'] for s in ['base','stress']])
