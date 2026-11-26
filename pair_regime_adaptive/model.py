"""Select a predictive regime from matured, prequential expert forecasts.

The hedge reduces return variance; this is not a claim of cointegration. No
asset identifiers, PAS future labels, or retrospective winner list are fitted.
"""
from dataclasses import dataclass
import numpy as np
from auto_pair_selection.selection import IDS,PAIRS,validate_snapshot

@dataclass(frozen=True)
class Spec:
    window: int=1440
    coverage: float=.8
    minimum_hedged_variance_reduction: float=.05
    maximum_block_beta_ratio: float=3.
    horizon: int=72
    evaluation_days: int=60
    minimum_matured_origins: int=10
    half_life_days: float=30.
    proxy_cost: float=.0038


def stats(x,y):
    x=x-x.mean();y=y-y.mean();xx=float(x@x);yy=float(y@y);xy=float(x@y)
    if min(xx,yy)<=1e-18:return None
    return dict(beta=xy/yy,correlation=xy/np.sqrt(xx*yy),hedged_variance_reduction=xy*xy/(xx*yy))


def fit(logs,active,index,ids,spec=Spec()):
    a,b=ids;start=index-spec.window
    if start<0:return None,dict(eligible=False,reasons=['insufficient_history'])
    ix=np.arange(start+1,index+1);valid=active[ix,a]&active[ix,b]&active[ix-1,a]&active[ix-1,b];ii=ix[valid]
    if len(ii)<spec.coverage*spec.window:return None,dict(eligible=False,reasons=['insufficient_fresh_returns'])
    dr=np.diff(logs[start:index+1],axis=0);s=stats(dr[valid,a],dr[valid,b])
    if s is None or not .1<=s['beta']<=10:return None,dict(eligible=False,reasons=['unsupported_return_hedge'],return_statistics=s)
    reasons=[]
    if s['correlation']<=0 or s['hedged_variance_reduction']<spec.minimum_hedged_variance_reduction:reasons.append('insufficient_risk_reduction')
    blocks=[]
    for n in range(3):
        use=(ix>start+n*480)&(ix<=start+(n+1)*480)&valid;bstat=stats(dr[use,a],dr[use,b]) if use.sum()>=spec.coverage*480 else None;blocks.append(bstat)
    betas=[r['beta'] for r in blocks if r is not None]
    ratio=max(betas)/min(betas) if len(betas)==3 and min(betas)>0 else None
    if ratio is None or ratio>spec.maximum_block_beta_ratio:reasons.append('unstable_return_hedge')
    beta=s['beta'];spread=logs[:index+1,a]-beta*logs[:index+1,b];x,y=spread[ii-1],spread[ii];xx=x-x.mean();den=float(xx@xx);rho=float(xx@(y-y.mean())/den) if den>1e-18 else None
    mr=None
    if rho is not None and 0<rho<1:
        half=float(-np.log(2)/np.log(rho))
        if 6<=half<=336:mr=dict(rho=rho,equilibrium=float((y.mean()-rho*x.mean())/(1-rho)),half_life=half)
    recent=ii[ii>index-168];vol=float(np.std((np.diff(spread)[recent-1])/(1+beta))) if len(recent)>=spec.coverage*168 else None
    if vol is None or vol<=1e-6:return None,dict(eligible=False,reasons=['insufficient_volatility'])
    return dict(beta=beta,vol=vol,mr=mr),dict(eligible=not reasons,reasons=reasons,return_statistics=s,blocks=blocks,beta_ratio=ratio)


def experts(logs,active,index,ids,model,spec=Spec()):
    a,b=ids
    if index<spec.horizon or not (active[index,[a,b]].all() and active[index-spec.horizon,[a,b]].all()):return np.array([np.nan,np.nan])
    beta=model['beta'];s=logs[index,a]-beta*logs[index,b];prior=logs[index-spec.horizon,a]-beta*logs[index-spec.horizon,b];mr=model['mr'];v=np.nan
    if mr is not None:v=(mr['equilibrium']-s)*(1-mr['rho']**spec.horizon)/(1+beta)
    return np.clip([v,(s-prior)/(1+beta)],-4*model['vol']*np.sqrt(spec.horizon),4*model['vol']*np.sqrt(spec.horizon))


def select(times,pred,beta,prices,fresh,source,spec=Spec()):
    # Sources spaced72h apart: do not treat overlapping targets as independent
    # hourly validation observations. Maturity is STRICTLY before fit time.
    ix=np.flatnonzero((times>=source-spec.evaluation_days*86400)&(times+spec.horizon*3600<source)&(times%(spec.horizon*3600)==0))
    result=[]
    for kind in range(2):
        ii=ix[np.isfinite(pred[ix,kind])&np.isfinite(beta[ix])&fresh[ix]];end=np.searchsorted(times,times[ii]+spec.horizon*3600);valid=(end<len(times));ii,end=ii[valid],end[valid];valid=fresh[end];ii,end=ii[valid],end[valid]
        if len(ii)<spec.minimum_matured_origins:result.append(dict(eligible=False,reason='insufficient_matured_predictions',rows=len(ii)));continue
        b=beta[ii];rr=prices[end]/prices[ii]-1;y=(rr[:,0]-b*rr[:,1])/(1+b);p=pred[ii,kind];w=np.exp2(-(source-(times[ii]+spec.horizon*3600))/(86400*spec.half_life_days));w/=w.sum()
        error=float(w@abs(y-p));zero=float(w@abs(y));signal=abs(p)>spec.proxy_cost;net=np.sign(p)*y-spec.proxy_cost;mean_net=float(w@(net*signal));active=int(signal.sum())
        result.append(dict(eligible=error<zero and mean_net>0 and active>=3,reason='past_prediction_advantage' if error<zero and mean_net>0 and active>=3 else 'no_past_prediction_advantage',rows=len(ii),active=active,mae=error,zero_mae=zero,proxy_net=mean_net,skill=max(0.,1-error/max(zero,1e-12)),last_maturity=int(max(times[ii])+spec.horizon*3600)))
    return result


ARMS=('regime_choice_72','regime_ensemble_72')
def build(d,start,spec=Spec()):
    t=validate_snapshot(d);logs=np.log(d['close']);active=(d['volume']>0)&(d['trades']>0);n=len(t);shape=(n,45);pred=np.full((n,45,2),np.nan);beta=np.full(shape,np.nan);vol=beta.copy();fresh=np.array([active[:,a]&active[:,b] for a,b in IDS]).T
    states={a:dict(mu=np.full(shape,np.nan),economic_action=np.zeros(shape,int),regime=np.zeros(shape,int),relationship_allowed=np.zeros(shape,bool),regime_allowed=np.ones(shape,bool)) for a in ARMS};models={};audits=[];eligible={};selection={}
    for i,now in enumerate(t):
        if now%86400==0:
            models={};eligible={};selection={};rows=[]
            for k,ids in enumerate(IDS):
                model,audit=fit(logs,active,i,ids,spec)
                if model is not None:models[k]=model
                eligible[k]=audit['eligible'];sel=select(t,pred[:,k],beta[:,k],d['close'][:,list(ids)],fresh[:,k],int(now),spec);selection[k]=sel
                rows.append(dict(pair=list(PAIRS[k]),pair_index=k,formation=audit,experts=sel))
            if now>=start:audits.append(dict(source=int(now),pairs=rows))
        for k,m in models.items():
            vs=experts(logs,active,i,IDS[k],m,spec);pred[i,k]=vs;beta[i,k]=m['beta'];vol[i,k]=m['vol']
            if now<start:continue
            choices=[e for e in range(2) if selection[k][e]['eligible'] and np.isfinite(vs[e])]
            # Fallback is for existing-position valuation only; no entry without
            # a past prediction advantage. It never borrows current labels.
            fallback=vs[0] if np.isfinite(vs[0]) else vs[1]
            for arm,s in states.items():
                s['mu'][i,k]=fallback;s['relationship_allowed'][i,k]=eligible[k]
                if not choices:continue
                best=max(choices,key=lambda e:(selection[k][e]['proxy_net'],selection[k][e]['skill']))
                value=vs[best];regime=best+1
                if arm=='regime_ensemble_72' and len(choices)==2:
                    w=np.array([selection[k][e]['skill'] for e in choices]);w/=w.sum();value=float(w@vs[choices]);regime=3
                s['mu'][i,k]=value;s['regime'][i,k]=regime
                if eligible[k] and abs(value)>spec.proxy_cost:s['economic_action'][i,k]=int(np.sign(value))
    for arm,s in states.items():s.update(times=t,dense=True,pairs=np.asarray(IDS),signal_times=np.broadcast_to(t[:,None],shape),source_prices=d['close'],forecast_times=t,beta=beta,vol=vol,directions=np.isfinite(s['mu']).astype(int),economic_available=np.isfinite(s['mu']))
    return states,audits,dict(pred=pred,beta=beta,vol=vol)
