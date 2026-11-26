"""Past-only pair formation and trend-continuation guard.

Holm correction is per formation date, not a guarantee over repeated daily
selection. Cointegration tests assume I(1) inputs; these diagnostics are not a
proof of stationarity, future convergence, or trading profitability.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.tools/pair_stats_20261023'))
from statsmodels.tsa.stattools import coint
from statsmodels.stats.multitest import multipletests
from auto_pair_selection.selection import IDS,PAIRS,validate_snapshot
from pair_model_research.adaptive_hedge import fit_pair as legacy_fit

@dataclass(frozen=True)
class Spec:
    window: int=1440
    coverage: float=.8
    blocks: int=3
    minimum_level_correlation: float=.3
    maximum_beta_ratio: float=2.
    maximum_equilibrium_shift_sigmas: float=2.
    family_alpha: float=.05
    adf_lags: int=6
    horizon: int=72
    proxy_cost: float=.0038


def regression(x,y):
    dx=x-x.mean();dy=y-y.mean();vx=float(dx@dx);vy=float(dy@dy);cov=float(dx@dy)
    if min(vx,vy)<1e-16:return None
    return dict(beta=cov/vy,correlation=cov/np.sqrt(vx*vy),intercept=float(x.mean()-cov/vy*y.mean()))


def fit_ols(logs,active,index,ids,spec=Spec()):
    """OLS hedge and AR on its residual. Missing time rows are not compressed."""
    a,b=ids;start=index-spec.window+1
    if start<0:return None,dict(reason='insufficient_history')
    ix=np.arange(start,index+1);good=active[ix,a]&active[ix,b]&np.isfinite(logs[ix,a])&np.isfinite(logs[ix,b]);ii=ix[good]
    if len(ii)<spec.coverage*spec.window:return None,dict(reason='insufficient_fresh_history')
    reg=regression(logs[ii,a],logs[ii,b])
    if reg is None or not .1<=reg['beta']<=10:return None,dict(reason='unsupported_ols_hedge',regression=reg)
    beta=reg['beta'];spread=logs[:index+1,a]-beta*logs[:index+1,b];rows=ii[ii>start];rows=rows[active[rows-1,a]&active[rows-1,b]]
    if len(rows)<spec.coverage*spec.window-1:return None,dict(reason='insufficient_contiguous_history')
    x,y=spread[rows-1],spread[rows];dx=x-x.mean();den=float(dx@dx)
    if den<1e-16:return None,dict(reason='degenerate_residual')
    rho=float(dx@(y-y.mean())/den)
    if not 0<rho<1:return None,dict(reason='nonreverting_point_estimate',rho=rho)
    half=float(-np.log(2)/np.log(rho))
    if not 6<=half<=336:return None,dict(reason='unsupported_half_life',half_life=half)
    intercept=float(y.mean()-rho*x.mean());theta=intercept/(1-rho)
    steps=np.arange(max(1,index-167),index+1);valid=active[steps,a]&active[steps,b]&active[steps-1,a]&active[steps-1,b]
    if valid.sum()<.8*168:return None,dict(reason='insufficient_recent_volatility')
    vol=float(np.std(np.diff(spread[max(0,index-168):index+1])[valid],ddof=0))/(1+beta)
    if not np.isfinite(vol) or vol<=1e-6:return None,dict(reason='degenerate_volatility')
    # Engle-Granger assumes equally spaced inputs. Use the longest contiguous
    # fresh run; never pretend gaps are adjacent hourly observations.
    split=np.split(ii,np.flatnonzero(np.diff(ii)>1)+1);run=max(split,key=len)
    pvalue=1.;tstat=None
    if len(run)>=spec.coverage*spec.window:
        tstat,pvalue,_=coint(logs[run,a],logs[run,b],trend='c',maxlag=spec.adf_lags,autolag=None)
        pvalue=float(pvalue);tstat=float(tstat)
    blocks=[];width=spec.window//spec.blocks
    for block in range(spec.blocks):
        j=ii[(ii>=start+block*width)&(ii<start+(block+1)*width)]
        r=regression(logs[j,a],logs[j,b]) if len(j)>=spec.coverage*width else None
        blocks.append(dict(rows=len(j),**(r or dict(beta=None,correlation=None,intercept=None)),spread_mean=float(np.mean(spread[j])) if len(j) else None))
    betas=[b['beta'] for b in blocks];ratio=max(betas)/min(betas) if all(b is not None and b>0 for b in betas) else None
    sd=float(np.std(spread[ii],ddof=1));centers=[b['spread_mean'] for b in blocks];shift=(max(centers)-min(centers))/sd if sd>1e-12 and all(x is not None for x in centers) else None
    reasons=[]
    if reg['correlation']<spec.minimum_level_correlation:reasons.append('weak_level_relationship')
    if any(b['correlation'] is None or b['correlation']<spec.minimum_level_correlation for b in blocks):reasons.append('relationship_not_stable_across_blocks')
    if ratio is None or ratio>spec.maximum_beta_ratio:reasons.append('hedge_coefficient_unstable')
    if shift is None or shift>spec.maximum_equilibrium_shift_sigmas:reasons.append('equilibrium_shift')
    if len(run)<spec.coverage*spec.window:reasons.append('insufficient_gap_free_cointegration_history')
    model=dict(beta=beta,rho=rho,equilibrium=theta,half_life=half,vol=vol,first_price_index=start,last_price_index=int(rows[-1]),formation_source_index=index)
    audit=dict(reason='estimated',regression=reg,blocks=blocks,beta_ratio=ratio,equilibrium_shift_sigmas=shift,eg_pvalue=pvalue,eg_tstat=tstat,eg_rows=len(run),reasons=reasons)
    return model,audit


def formation(logs,active,index,spec=Spec()):
    models={};audit=[];pvalues=np.ones(len(IDS))
    for k,ids in enumerate(IDS):
        model,r=fit_ols(logs,active,index,ids,spec);r.update(pair=list(PAIRS[k]),pair_index=k,formation_source_index=int(index))
        if model is not None:models[k]=model;pvalues[k]=r['eg_pvalue']
        audit.append(r)
    rejected,adjusted,_,_=multipletests(pvalues,alpha=spec.family_alpha,method='holm')
    for k,r in enumerate(audit):
        reasons=list(r.get('reasons',[]))
        if k not in models:reasons.append(r['reason'])
        if not rejected[k]:reasons.append('cointegration_not_confirmed_after_multiple_testing')
        r.update(holm_pvalue=float(adjusted[k]),eligible=not reasons,reasons=reasons)
    return models,audit


def trend_guard(logs,active,index,ids,model):
    """Abstain when multi-horizon residual drift overwhelms fitted correction.

This is a predeclared economic guard, not a regime probability or a short signal.
It never uses price observations after index and never flips a rejected entry.
"""
    a,b=ids
    if index<72:return dict(eligible=False,reason='regime_history_missing')
    j=np.array([index,index-6,index-24,index-72]);fresh=active[j,a]&active[j,b]
    if not fresh.all():return dict(eligible=False,reason='regime_prices_not_fresh')
    s=logs[j,a]-model['beta']*logs[j,b];deviation=float(s[0]-model['equilibrium']);side=np.sign(deviation)
    change6=float(s[0]-s[1]);change24=float(s[0]-s[2]);change72=float(s[0]-s[3]);correction24=abs(deviation)*(1-model['rho']**24)
    away=side*change6>0 and side*change24>correction24 and side*change72>correction24*3
    return dict(eligible=not away,reason='persistent_divergence' if away else 'no_persistent_divergence',deviation=deviation,change6=change6,change24=change24,change72=change72,expected_24h_reversion=correction24)


ARMS=('ols_72','relationship_72','persistence_72','stable_regime_72')
def build(data,start,spec=Spec()):
    t=validate_snapshot(data);logs=np.log(data['close']);active=(data['volume']>0)&(data['trades']>0);shape=(len(t),len(IDS))
    states={a:dict(mu=np.full(shape,np.nan),beta=np.full(shape,np.nan),vol=np.full(shape,np.nan),economic_action=np.zeros(shape,int),relationship_allowed=np.zeros(shape,bool),regime_allowed=np.zeros(shape,bool)) for a in ARMS}
    audits=[];models={};old={};elig={};counters={a:dict(available=0,cost_gate_pass=0,relationship_reject=0,regime_reject=0,entry_pass=0) for a in ARMS}
    for i,now in enumerate(t):
        if now<start:continue
        if now%86400==0:
            models,rs=formation(logs,active,i,spec);elig={r['pair_index']:r['eligible'] for r in rs};old={k:legacy_fit(logs,active,i,ids) for k,ids in enumerate(IDS)}
            audits.append(dict(source=int(now),pairs=rs))
        for arm in ARMS:
            for k,ids in enumerate(IDS):
                m=old.get(k) if arm=='persistence_72' else models.get(k)
                if m is None or not active[i,list(ids)].all():continue
                x=states[arm];a,b=ids;s=logs[i,a]-m['beta']*logs[i,b];mu=float(np.clip((m['equilibrium']-s)*(1-m['rho']**spec.horizon)/(1+m['beta']),-4*m['vol']*np.sqrt(spec.horizon),4*m['vol']*np.sqrt(spec.horizon)))
                x['mu'][i,k]=mu;x['beta'][i,k]=m['beta'];x['vol'][i,k]=m['vol'];counters[arm]['available']+=1
                relationship=elig.get(k,False) if arm in ('relationship_72','stable_regime_72') else True
                regime=trend_guard(logs,active,i,ids,m)['eligible'] if arm in ('persistence_72','stable_regime_72') else True
                x['relationship_allowed'][i,k]=relationship;x['regime_allowed'][i,k]=regime
                if abs(mu)>spec.proxy_cost:
                    counters[arm]['cost_gate_pass']+=1;counters[arm]['relationship_reject']+=not relationship;counters[arm]['regime_reject']+=not regime
                    if relationship and regime:x['economic_action'][i,k]=int(np.sign(mu));counters[arm]['entry_pass']+=1
    for arm,x in states.items():
        x.update(times=t,dense=True,pairs=np.asarray(IDS),signal_times=np.broadcast_to(t[:,None],shape),source_prices=data['close'],forecast_times=t,directions=np.isfinite(x['mu']).astype(int),economic_available=np.isfinite(x['mu']))
    return states,audits,counters
