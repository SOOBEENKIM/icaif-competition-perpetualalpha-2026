import json,sys,copy,gzip
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from auto_pair_validation.study import OUT as V2,verify as verify_v2
from auto_pair_selection.signals import prepare_features
from auto_pair_selection.selection import PAIRS
from competition_live.profile import BUNDLE,load_profile
from .study import ROOT,PRIOR,read,save,sha,verify as verify_wave1
from .uncertainty import covariance,conservative

OUT=ROOT/'reports/pair_model_research_20261023/wave3'
ARMS={'uncertainty_half':.5,'uncertainty_one':1.,'uncertainty_two':2.}


def freeze():
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=verify_v2();verify_wave1()
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),wave=3,arms=ARMS,
      input=p['input'],input_sha256=p['input_sha256'],
      hypothesis='Ranking the largest point forecasts across45 can select high-uncertainty extrapolations. Deduct estimated mean uncertainty from every forecast rather than apply name/correlation restrictions or require nonexistent daily profitable-trade evidence.',
      covariance='Retained per-pair ridge bread; source-day clustered score residuals and two-day Bartlett HAC; raw mature target residuals, positive semidefinite projection. No45-way significance or future-profit guarantee.',
      valuation='sign(mu)*max(abs(mu)-k*SE,0), k=.5,1,2 fixed before results. This is a conservative price hypothesis, NOT claimed unbiased mean or calibrated quantile. Used consistently for entry and later holding decisions. Existing SPO+ gate remains, but all45 fit and compete.',
      fixed='Same6000 pair gross,50k own+50k reserve,existing beta and24h horizon,fees,funding,base model,delays,partial fills and risk limits.',
      selection='Select August worst Q improvement then PnL, reject any risk violation. Lock selection before later development comparison. Keep every failed wave.',
      history_reused=True,independent_test=False,orders_sent=0,live_changed=False))
    save(OUT/'freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),Path(__file__).with_name('uncertainty.py'),OUT/'protocol.json']})


def verify():
    verify_v2();verify_wave1()
    for path,digest in read(OUT/'freeze.json').items():assert sha(ROOT/path)==digest,path
    return read(OUT/'protocol.json')


def build():
    p=verify();v=read(V2/'protocol.json');d=dict(np.load(ROOT/p['input'],allow_pickle=False));t=d['market_times']
    f,y,maturity,terminal=prepare_features(d);unit=f['vol']*np.sqrt(24);ds=v['direct_spec']
    s=dict(np.load(V2/'states.npz',allow_pickle=False));std=np.full_like(s['raw_mu'],np.nan);records=[]
    for day in np.unique(t[t>=v['evaluation_start']]//86400*86400):
        models=read(V2/'models'/f'{day}.json')['direct'];day_ix=np.flatnonzero((t>=day)&(t<day+86400))
        window=(t>=day-ds['lookback_days']*86400)&(maturity<day)&(t%(ds['sample_hours']*3600)==0)
        for k in range(len(PAIRS)):
            if str(k) not in models:continue
            model=models[str(k)]
            valid=np.isfinite(f['x'][:,k]).all(1)&f['fresh'][:,k]&terminal[:,k]&np.isfinite(y[:,k])&np.isfinite(unit[:,k])&(unit[:,k]>0)
            ii=np.flatnonzero(window&valid)
            c=covariance(f['x'][ii,k],y[ii,k],t[ii],maturity[ii],int(day),model,ds)
            if c is None:continue
            c['pair_index']=k;records.append(c)
            z=np.c_[np.ones(len(day_ix)),(f['x'][day_ix,k]-model['center'])/model['scale']]
            variance=np.einsum('ij,jk,ik->i',z,np.asarray(c['matrix']),z)
            std[day_ix,k]=np.sqrt(np.maximum(0,variance))*unit[day_ix,k]
        print(int(day),flush=True)
    for arm,multiplier in p['arms'].items():
        out={k:s[k] for k in ['times','dense','pairs','signal_times','source_prices','beta','vol','forecast_times','directions']}
        out['mu']=conservative(s['raw_mu'],std,multiplier)
        available=s['gate_available']&np.isfinite(out['mu'])&s['directions'].astype(bool)
        out['economic_available']=available;out['economic_action']=np.where(available,s['gate_action'],0)
        out['raw_mu']=s['raw_mu'];out['estimation_se']=std
        np.savez_compressed(OUT/f'{arm}.npz',**out)
    save(OUT/'uncertainty_models.json',records);verify()


def job(arm,phase,scenario):
    from pas_economic_allocation.replay import replay
    from pas_performance_research.selected import controller
    from pas_allocation_study.study import summarize
    p=read(OUT/'protocol.json');profile=load_profile();dest=OUT/'reference'/f'{arm}_{phase}_{scenario}.json.gz'
    w=next(x for x in read(BUNDLE/'expected.json')['runs'] if x['venue']=='reference' and x['arm']==profile['candidate'] and x['phase']==phase and x['scenario']==scenario)
    if dest.exists():
        with gzip.open(dest,'rt') as fp:r=json.load(fp)
    else:
        d=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False));s=dict(np.load(OUT/f'{arm}.npz',allow_pickle=False));t=s.pop('times');ii=np.searchsorted(t,d['times'])
        assert np.array_equal(t[ii],d['times']);s={k:(v[ii] if v.ndim and len(v)==len(t) else v) for k,v in s.items()}
        z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz',allow_pickle=False));zi=np.searchsorted(z['times'],d['times'])
        f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz',allow_pickle=False));fi=np.searchsorted(f['times'],d['times'])
        raw=dict(np.load(ROOT/p['input'],allow_pickle=False));cfg=copy.deepcopy(profile['configs']['reference']);cfg['dense_pairs']=[list(x) for x in PAIRS];cfg['conditional_completion']={'risk_aversion':0.}
        c=controller(d['times'],f['forecast'][fi],f['carry'][fi],z['all_market_signed_core_v1'][zi],z['all_market_signed_core_v1_protected'][zi],price_times=raw['market_times'],prices=raw['close'])
        r=replay(d,s,None,cfg,profile['trading'],profile['candidate'],scenario,w['start'],w['end'],base_policy=c)
        dest.parent.mkdir(exist_ok=True)
        with gzip.open(dest,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
    row=summarize(r,arm,phase,scenario);row['entered_pairs']=sorted({tuple(e['pair']) for e in r['events'] if e['event']=='pair_intent'})
    assert row['timing_pass'] and row['accounting_error']<1e-6
    row['eligible']=row['risk_breach_bars']==0;save(OUT/'rows'/f'{arm}_{phase}_{scenario}.json',row);return row


def run_jobs(jobs):
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(job,*j) for j in jobs]):
            r=future.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs','risk_breach_bars']}),flush=True)


def stage_a():
    p=verify();run_jobs([(a,'validation',s) for a in p['arms'] for s in ['base','stress']])
    b=[r for r in read(PRIOR/'reference_results.json') if r['arm']=='incumbent_fixed_two' and r['phase']=='validation'];ranking=[]
    for a in p['arms']:
        rs=[read(OUT/'rows'/f'{a}_validation_{s}.json') for s in ['base','stress']]
        ds=[(r['Q']-x['Q'],r['net_pnl']-x['net_pnl']) for r in rs for x in b if r['scenario']==x['scenario']]
        ranking.append(dict(arm=a,eligible=all(r['eligible'] for r in rs),min_Q_advantage=min(x[0] for x in ds),min_pnl_advantage=min(x[1] for x in ds)))
    ranking.sort(key=lambda r:(r['eligible'],r['min_Q_advantage'],r['min_pnl_advantage']),reverse=True)
    save(OUT/'selection.json',dict(selected=ranking[0]['arm'] if ranking[0]['eligible'] else None,ranking=ranking));print('SELECTED',ranking[0],flush=True)


def stage_b():
    verify();a=read(OUT/'selection.json')['selected'];assert a is not None
    run_jobs([(a,p,s) for p in ['comparison','continuous'] for s in ['base','stress']]);verify()


if __name__=='__main__':globals()[sys.argv[1]]()
