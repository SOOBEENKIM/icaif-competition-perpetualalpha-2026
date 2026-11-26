"""All-market relative momentum with explicit two-leg execution and costs."""
import copy,gzip,itertools,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from .study import ROOT,OUT as PARENT,sha,save,native
from . import sleeve_budget

OUT=ROOT/'reports/pas_relative_trend_20261021'
ARMS=['incumbent','base_restore','relative24','relative72','relative72_restored','guarded72']


def build_state(d,horizon):
    prices=d['reference_close'];t=d['times'];lp=pd.DataFrame(np.log(prices));r=lp.diff()
    pairs=np.array(list(itertools.combinations(range(10),2)));n=len(t);k=len(pairs)
    beta=np.full((n,k),np.nan);vol=np.full((n,k),np.nan);mu=np.full((n,k),np.nan)
    good=np.zeros((n,k),bool);correlations=np.full((n,k),np.nan)
    for a,(j,l) in enumerate(pairs):
        covariance=r[j].rolling(720,min_periods=336).cov(r[l]).to_numpy()
        va=r[j].rolling(720,min_periods=336).var().to_numpy()
        vb=r[l].rolling(720,min_periods=336).var().to_numpy()
        b=covariance/np.maximum(vb,.0005**2)
        corr=covariance/np.sqrt(np.maximum(va*vb,1e-30));correlations[:,a]=corr
        beta[:,a]=b
        vol[:,a]=np.sqrt(np.maximum(va+b*b*vb-2*b*covariance,1e-12))/(1+b)
        forecasts=[]
        for lookback in [168,672]:
            spread=((lp[j]-lp[j].shift(lookback))-b*(lp[l]-lp[l].shift(lookback))).to_numpy()/(1+b)
            forecasts.append(spread*horizon/lookback)
        mu[:,a]=np.clip(np.mean(forecasts,axis=0),-.2,.2)
        good[:,a]=(np.isfinite(mu[:,a])&np.isfinite(vol[:,a])&(corr>=.5)&(b>=.1)&(b<=10)
                    &(d['par_volume'][:,j]>0)&(d['par_volume'][:,l]>0))
    # The forecast refresh is known only at the completed six-hour source bar.
    source=np.full(n,-1,dtype=np.int64);last=-1
    for i,now in enumerate(t):
        if now%21600==0:last=i
        source[i]=last
    safe=np.maximum(source,0);valid=(source>=0)&(t-t[safe]<3600)
    available=good[safe]&valid[:,None]
    direction=np.where(available,np.sign(mu[safe]),0).astype(int)
    return dict(dense=np.array(True),pairs=pairs,
        signal_times=np.tile(t[safe,None],(1,k)),directions=direction,
        source_prices=prices[safe],beta=beta[safe],vol=vol[safe],mu=mu[safe],
        forecast_times=t[safe],economic_action=direction,economic_available=available,
        source_correlations=correlations[safe])


def prepare():
    from competition_live.profile import verify_deployment,BUNDLE,load_profile
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,nominee='guarded72',
        hypothesis='The unconditional all-market directional sleeve was weak. Trade RELATIVE weekly/monthly momentum within causally correlated pairs, using the existing actual two-leg lifecycle. A 72h hypothesis can amortize roundtrip costs.',
        universe='Every unordered pair among all ten authorized markets (45); no asset-specific exclusion or best-pair selection from evaluation outcomes.',
        model='720h rolling covariance/variance hedge ratio, at least336 observations, correlation >=0.5 and beta in existing [0.1,10]. Equal 7d/28d hedged-return drift projected to24/72h, bounded at20%. Six-hour source cadence, one-hour entry validity.',
        coefficients='No supervised fitted labels; fixed linear drift extrapolation rule. Does not reproduce an empirical result from a paper.',
        execution='Same independently delayed two-leg fills, economic price limits, fees, funding, one active pair, paid completion/unwind and renewal. All45 pairs compete on current expected cash, never future profit.',
        risk='Same 6000 gross pair budget and 50k strategy capital. guarded72 uses paid proactive risk controls; other arms isolate forecast/horizon/size.',
        selection='guarded72 fixed nominee before results; no horizon/beta/correlation threshold search. Full six existing windows retained.',
        promotion='Higher PnL and Q than incumbent and base_restore in continuous and later windows, base/stress; zero accounting/timing/risk violations. Then PAS-specific applicability.',
        independent=False,history_reused=True,orders_sent=0,live_changed=False,
        deployment=verify_deployment()['files'])
    save(OUT/'protocol.json',p)
    d=dict(np.load(BUNDLE/'data/reference.npz'))
    for h in [24,72]:np.savez_compressed(OUT/f'state_{h}.npz',**build_state(d,h))
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in
        [Path(__file__),Path(__file__).with_name('guarded_portfolio.py'),Path(__file__).with_name('risk_trim.py'),
         OUT/'protocol.json',OUT/'state_24.npz',OUT/'state_72.npz',PARENT/'targets.npz']})


def run():
    from competition_live.profile import load_profile,BUNDLE,verify_deployment
    from experiments.decision_gate_pairs_20261017.portfolio import replay
    from .guarded_portfolio import replay as guarded
    from pas_allocation_study.study import summarize
    p=json.loads((OUT/'protocol.json').read_text())
    for rel,h in json.loads((OUT/'source_freeze.json').read_text()).items():assert sha(ROOT/rel)==h
    profile=load_profile();d0=dict(np.load(BUNDLE/'data/reference.npz'));z=dict(np.load(PARENT/'targets.npz'));ix=np.searchsorted(z['times'],d0['times'])
    states={h:dict(np.load(OUT/f'state_{h}.npz')) for h in [24,72]};rows=[]
    windows=[r for r in json.loads((BUNDLE/'expected.json').read_text())['runs'] if r['venue']=='reference' and r['arm']==profile['candidate']]
    for win in windows:
        for arm in ARMS:
            dest=OUT/'reference'/f"{arm}_{win['phase']}_{win['scenario']}.json.gz"
            if dest.exists():
                with gzip.open(dest,'rt') as f:r=json.load(f)
            elif arm in ['incumbent','base_restore']:
                with gzip.open(sleeve_budget.OUT/'reference'/f"{arm}_{win['phase']}_{win['scenario']}.json.gz",'rt') as f:r=json.load(f)
                dest.parent.mkdir(exist_ok=True)
                with gzip.open(dest,'wt') as f:json.dump(r,f,default=native)
            else:
                h=24 if arm=='relative24' else 72;d=dict(d0)
                d['first']=z['size_only' if arm=='relative72_restored' else 'incumbent'][ix]
                cfg=copy.deepcopy(profile['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
                cfg['dense_pairs']=[[profile['trading']['symbols'][i] for i in pair] for pair in states[h]['pairs']]
                engine=guarded if arm=='guarded72' else replay
                r=engine(d,states[h],None,cfg,profile['trading'],f'first_dense_hourlyhedge_decisiongate_{h}',win['scenario'],win['start'],win['end'])
                dest.parent.mkdir(exist_ok=True)
                with gzip.open(dest,'wt') as f:json.dump(r,f,default=native)
            row=summarize(r,arm,win['phase'],win['scenario']);assert row['timing_pass'] and row['accounting_error']<1e-6
            rows.append(row);save(OUT/'reference_results.json',rows);print(json.dumps(row),flush=True)
    assert verify_deployment()['files']==p['deployment']


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
