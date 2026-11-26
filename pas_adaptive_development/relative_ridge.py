"""Purged, monthly pooled relative-return regression; retain incumbent expert."""
import copy,gzip,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from .study import ROOT,OUT as PARENT,sha,save,native,ts
from .relative_trend import build_state
from . import sleeve_budget

OUT=ROOT/'reports/pas_relative_ridge_20261021'
ARMS=['incumbent','base_restore','ridge24_add','ridge72_add','ridge72_only','ridge72_restored']


def features(d,h):
    # Build causal covariance/eligibility on every completed hour, not a held
    # six-hour source. build_state's vectorized covariance uses past rows only.
    raw=build_state(d,h);lp=np.log(d['reference_close']);pairs=raw['pairs'];t=d['times']
    # Only six-hour sources are used for train/predict, so these are their exact
    # source beta and volatility, not interpolation from the future.
    b=raw['beta'];v=raw['vol'];n,k=b.shape;features=[]
    for lag in [6,24,72,168,672]:
        r=np.full_like(lp,np.nan);r[lag:]=lp[lag:]-lp[:-lag]
        x=(r[:,pairs[:,0]]-b*r[:,pairs[:,1]])/(1+b)
        features.append(np.clip(x/np.maximum(v*np.sqrt(lag),1e-8),-5,5))
    x=np.stack(features,-1)
    y=np.full((n,k),np.nan)
    forward=lp[h:]-lp[:-h]
    y[:-h]=(forward[:,pairs[:,0]]-b[:-h]*forward[:,pairs[:,1]])/(1+b[:-h])/np.maximum(v[:-h]*np.sqrt(h),1e-8)
    good=raw['economic_available']&np.isfinite(x).all(-1)
    return raw,x,y,good


def fit_states(d,h):
    raw,x,y,good=features(d,h);t=d['times'];maturity=t+h*3600
    prediction=np.full(y.shape,np.nan);records=[]
    fits=[ts(f'2026-{m:02d}-01T00:00:00Z') for m in [7,8,9,10]]
    for n,now in enumerate(fits):
        cut=now-30*86400
        mask=(t>=max(ts('2026-04-01T00:00:00Z'),now-120*86400))&(maturity<cut)&(t%86400==0)
        ii,jj=np.where(good&mask[:,None]&np.isfinite(y))
        if len(ii)<100:raise ValueError('Too few purged relative labels')
        xx=x[ii,jj];center=np.median(xx,0);scale=np.maximum(np.std(xx,axis=0),.5)
        design=np.column_stack([np.ones(len(xx)),(xx-center)/scale]);labels=np.clip(y[ii,jj],-4,4)
        weights=np.exp2(-(cut-t[ii])/(60*86400));weights*=len(weights)/weights.sum()
        penalty=np.eye(design.shape[1])*20;penalty[0,0]=2
        beta=np.linalg.solve(design.T@(weights[:,None]*design)+penalty,design.T@(weights*labels))
        cal=(t>=cut)&(maturity<now)&(t%86400==0)
        ci,cj=np.where(good&cal[:,None]&np.isfinite(y))
        cr=np.column_stack([np.ones(len(ci)),(x[ci,cj]-center)/scale])@beta
        attenuation=float(np.clip(cr@y[ci,cj]/(cr@cr+10),0,1))
        end=fits[n+1] if n+1<len(fits) else t[-1]+1
        mask=(t>=now)&(t<end)
        all_x=np.concatenate([np.ones((*x[mask].shape[:2],1)),(x[mask]-center)/scale],axis=2)
        prediction[mask]=np.clip(all_x@beta*attenuation,-2,2)*raw['vol'][mask]*np.sqrt(h)
        records.append(dict(fit=now,train_rows=len(ii),calibration_rows=len(ci),train_last_maturity=int(maturity[ii].max()),
            training_cutoff=cut,calibration_last_maturity=int(maturity[ci].max()),attenuation=attenuation,
            coefficients=beta.tolist(),center=center.tolist(),scale=scale.tolist(),
            calibration_mae=float(np.mean(abs(cr-y[ci,cj]))),zero_mae=float(np.mean(abs(y[ci,cj])))))
    available=good&np.isfinite(prediction)&(t[:,None]>=fits[0])
    raw['mu']=prediction
    raw['economic_available']=available
    raw['economic_action']=np.where(available,np.sign(prediction),0).astype(int)
    raw['directions']=raw['economic_action'].copy()
    return raw,records


def ensemble(old,new):
    # Source anchors and forecast maturities agree for sources accepted by both.
    # The old expert remains available even if new attenuation is zero.
    result={k:v.copy() for k,v in new.items()}
    for key in ['pairs','signal_times','directions','beta','vol','mu','economic_action','economic_available']:
        axis=0 if key=='pairs' else 1
        result[key]=np.concatenate([old[key],new[key]],axis=axis)
    return result


def prepare():
    from competition_live.profile import verify_deployment,BUNDLE
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,nominee='ridge24_add',
        hypothesis='Learn the sign and strength of relative trends on past labels instead of assuming continuation. Retain incumbent expert so abstention cannot erase established opportunities.',
        model='Pooled ridge on 6/24/72/168/672h standardized hedge-adjusted past returns, intercept. All45 correlated pairs, no named pair coefficient or exclusion. Ridge20, intercept2; clip labels4 sigma and forecasts2 sigma.',
        fitting='Monthly July/Aug/Sep/Oct1; last120 days, last30 days reserved for calibration. DAILY source sampling. Purge every training label crossing calibration boundary; calibration labels strictly mature before refit. Nonnegative attenuation [0,1] with prior10.',
        horizon='24h/72h diagnostics. Nominee24h additive uses the SAME maturity as existing24h expert; 72h additive uses only its own expert to avoid mixing mismatched maturities.',
        selection='Nominee ridge24_add pre-fixed; all failures and full six windows retained, no choosing a later test winner.',
        promotion='Higher PnL and Q than incumbent and base_restore in continuous and later windows, base/stress, with zero accounting/timing/risk violations; PAS separately assessed.',
        independent=False,history_reused=True,orders_sent=0,live_changed=False,
        deployment=verify_deployment()['files'])
    save(OUT/'protocol.json',p);d=dict(np.load(BUNDLE/'data/reference.npz'));old=dict(np.load(BUNDLE/'models/reference_gate_state.npz'));models={}
    for h in [24,72]:
        state,records=fit_states(d,h);models[str(h)]=records
        np.savez_compressed(OUT/f'state_{h}.npz',**state)
        if h==24:
            # Both experts' source price vectors are actual same-hour closes.
            state['source_prices']=d['reference_close'].copy()
            state['forecast_times']=d['times'].copy()
            np.savez_compressed(OUT/'ensemble_24.npz',**ensemble(old,state))
    save(OUT/'models.json',models)
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in
        [Path(__file__),Path(__file__).with_name('relative_trend.py'),Path(__file__).with_name('guarded_portfolio.py'),Path(__file__).with_name('risk_trim.py'),
         OUT/'protocol.json',OUT/'state_24.npz',OUT/'state_72.npz',OUT/'ensemble_24.npz',OUT/'models.json',PARENT/'targets.npz']})
    print(json.dumps(models),flush=True)


def run():
    from competition_live.profile import load_profile,BUNDLE,verify_deployment
    from .guarded_portfolio import replay
    from pas_allocation_study.study import summarize
    p=json.loads((OUT/'protocol.json').read_text())
    for rel,h in json.loads((OUT/'source_freeze.json').read_text()).items():assert sha(ROOT/rel)==h
    profile=load_profile();d0=dict(np.load(BUNDLE/'data/reference.npz'));z=dict(np.load(PARENT/'targets.npz'));ix=np.searchsorted(z['times'],d0['times']);rows=[]
    states={h:dict(np.load(OUT/f'state_{h}.npz')) for h in [24,72]};ens=dict(np.load(OUT/'ensemble_24.npz'))
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
                h=24 if arm=='ridge24_add' else 72;de=ens if h==24 else states[h];d=dict(d0)
                d['first']=z['size_only' if arm=='ridge72_restored' else 'incumbent'][ix]
                cfg=copy.deepcopy(profile['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
                cfg['dense_pairs']=[[profile['trading']['symbols'][i] for i in pair] for pair in de['pairs']]
                # ridge72_add is explicitly duplicate diagnostic; a 24h expert
                # is never silently evaluated as though its forecast is72h.
                r=replay(d,de,None,cfg,profile['trading'],f'first_dense_hourlyhedge_decisiongate_{h}',win['scenario'],win['start'],win['end'])
                dest.parent.mkdir(exist_ok=True)
                with gzip.open(dest,'wt') as f:json.dump(r,f,default=native)
            row=summarize(r,arm,win['phase'],win['scenario']);assert row['timing_pass'] and row['accounting_error']<1e-6
            rows.append(row);save(OUT/'reference_results.json',rows);print(json.dumps(row),flush=True)
    assert verify_deployment()['files']==p['deployment']


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
