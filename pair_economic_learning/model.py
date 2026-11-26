"""Prequential expert calibration and shallow nonlinear return prediction.

Source hedges remain fixed when labels are formed. PAS outcomes are never
training labels. Pair identity is not a feature. No future target participates
in a daily fit, including the target which matures exactly at the fit boundary.
"""
from dataclasses import dataclass
import numpy as np
from auto_pair_selection.selection import IDS,validate_snapshot
from pair_model_research.adaptive_hedge import forecasts
from pair_value_learning.scaled_coherence import project


@dataclass(frozen=True)
class Spec:
    horizon:int=72
    history_days:int=90
    decay_days:int=30
    stride_hours:int=6
    min_origins:int=80
    mixture_prior:float=6.
    tree_rounds:int=64
    max_depth:int=2
    eta:float=.04
    min_child_weight:float=8.
    ridge:float=5.


def prepare(d):
    t=validate_snapshot(d);prices=d['close'];active=(d['volume']>0)&(d['trades']>0)
    mu,beta,vol,_,formations=forecasts(d,int(t[0]+1440*3600),72)
    unit=vol*np.sqrt(72);fresh=np.array([active[:,a]&active[:,b] for a,b in IDS]).T
    x=np.full((*mu.shape,17),np.nan);ar=np.full_like(mu,np.nan);trend=ar.copy();y=ar.copy()
    daily={};byday={}
    for r in formations:
        if r['model'] is not None:byday.setdefault(r['source'],{})[r['pair_index']]=r['model']
    for i,now in enumerate(t):
        if now%86400==0:daily=byday.get(int(now),{})
        valid=np.isfinite(mu[i])&fresh[i]
        if not valid.any() or i<168:continue
        ar[i],_=project(mu[i],beta[i],IDS,valid,unit[i])
        ret={h:prices[i]/prices[i-h]-1 for h in [6,24,72,168]}
        future_asset=np.expm1(np.log(prices[i]/prices[i-168])*72/168)
        rawtrend=np.array([(future_asset[a]-b*future_asset[c])/(1+b) for (a,c),b in zip(IDS,beta[i])])
        trend[i],_=project(np.clip(rawtrend,-4*unit[i],4*unit[i]),beta[i],IDS,valid,unit[i])
        for k in np.flatnonzero(valid):
            a,b=IDS[k];bt=beta[i,k];u=unit[i,k];m=daily[k]
            mom=[(ret[h][a]-bt*ret[h][b])/(1+bt)/u for h in [6,24,72,168]]
            lr=np.diff(np.log(prices[i-168:i+1][:,[a,b]]),axis=0)
            rel=(lr[:,0]-bt*lr[:,1])/(1+bt)
            corr=float(np.corrcoef(lr.T)[0,1]) if np.min(np.std(lr,axis=0))>1e-12 else 0.
            recent_vol=float(np.std(rel[-24:])/max(np.std(rel),1e-10))
            x[i,k]=np.clip([ar[i,k]/u,trend[i,k]/u,*mom,mom[1]-mom[2]/3,
                            recent_vol,corr,m['level_correlation'],np.log(bt),m['half_life']/72,
                            m['ar_tstat'],.0038/u,abs(mom[2]),mom[1]*mom[3],(ar[i,k]-trend[i,k])/u],-12,12)
    terminal=np.zeros_like(fresh);terminal[:-72]=fresh[72:]
    for k,(a,b) in enumerate(IDS):
        rr=prices[72:,[a,b]]/prices[:-72,[a,b]]-1;bt=beta[:-72,k]
        y[:-72,k]=(rr[:,0]-bt*rr[:,1])/(1+bt)
    return dict(times=t,x=x,y=y,ar=ar,trend=trend,beta=beta,vol=vol,unit=unit,
                valid=np.isfinite(x).all(-1)&fresh,terminal=terminal,maturity=t+72*3600),formations


def training_rows(f,source,spec=Spec(),*,stride=None):
    t=f['times'];h=spec.stride_hours if stride is None else stride
    mask=(t>=source-spec.history_days*86400)&(f['maturity']<source)&(t%(h*3600)==0)
    valid=mask[:,None]&f['valid']&f['terminal']&np.isfinite(f['y'])&(f['unit']>0)
    ii,kk=np.where(valid)
    if len(np.unique(ii))<spec.min_origins*(h==spec.stride_hours):return None
    count=np.bincount(ii,minlength=len(t))
    w=np.exp2(-(source-f['maturity'][ii])/(86400*spec.decay_days))/np.maximum(1,count[ii])
    return ii,kk,w


def fit_mixture(f,source,spec=Spec()):
    rows=training_rows(f,source,spec,stride=72)
    if rows is None or not len(rows[0]):return dict(weight=.5,sources=0,last_maturity=None)
    ii,kk,w=rows;unit=f['unit'][ii,kk]
    a=f['ar'][ii,kk]/unit;b=f['trend'][ii,kk]/unit;y=np.clip(f['y'][ii,kk]/unit,-4,4)
    z=a-b;target=y-b
    value=(np.sum(w*z*target)+.5*spec.mixture_prior)/(np.sum(w*z*z)+spec.mixture_prior)
    return dict(weight=float(np.clip(value,0,1)),sources=len(np.unique(ii)),last_maturity=int(f['maturity'][ii].max()))


def fit_trees(f,source,spec=Spec()):
    import xgboost as xgb
    rows=training_rows(f,source,spec)
    if rows is None:return None
    ii,kk,w=rows;x=f['x'][ii,kk];y=np.clip(f['y'][ii,kk]/f['unit'][ii,kk],-4,4)
    params=dict(objective='reg:squarederror',max_depth=spec.max_depth,eta=spec.eta,
        min_child_weight=spec.min_child_weight,reg_lambda=spec.ridge,tree_method='hist',max_bin=64,
        nthread=1,seed=20261023,subsample=1.,colsample_bytree=1.,base_score=0.,verbosity=0)
    models={}
    for kind,target in [('direct',y),('residual',y-f['ar'][ii,kk]/f['unit'][ii,kk])]:
        mat=xgb.DMatrix(x,label=target,weight=w,nthread=1)
        models[kind]=xgb.train(params,mat,num_boost_round=spec.tree_rounds)
    audit=dict(source=int(source),last_maturity=int(f['maturity'][ii].max()),rows=len(ii),sources=len(np.unique(ii)),weight_sum=float(w.sum()))
    assert audit['last_maturity']<source
    return models,audit


def build(d,f,start,spec=Spec(),callback=None):
    import xgboost as xgb
    arms=['online_mixture_72','tree_direct_72','tree_coherent_72','tree_residual_72'];t=f['times']
    mus={k:np.full_like(f['ar'],np.nan) for k in arms};fits=[];models=[]
    for source in np.unique(t[t>=start]//86400*86400):
        ii=np.flatnonzero((t>=source)&(t<source+86400));mix=fit_mixture(f,int(source),spec)
        mus['online_mixture_72'][ii]=mix['weight']*f['ar'][ii]+(1-mix['weight'])*f['trend'][ii]
        fitted=fit_trees(f,int(source),spec);audit=dict(source=int(source),mixture=mix,tree=None)
        if fitted:
            mm,ta=fitted;audit['tree']=ta
            for row in ii:
                good=f['valid'][row]
                if not good.any():continue
                dm=xgb.DMatrix(f['x'][row,good],nthread=1);unit=f['unit'][row,good]
                direct=np.full(45,np.nan);residual=direct.copy()
                direct[good]=np.clip(mm['direct'].predict(dm),-4,4)*unit
                residual[good]=np.clip(mm['residual'].predict(dm)+f['ar'][row,good]/unit,-4,4)*unit
                mus['tree_direct_72'][row]=direct
                mus['tree_coherent_72'][row],_=project(direct,f['beta'][row],IDS,good,f['unit'][row])
                mus['tree_residual_72'][row],_=project(residual,f['beta'][row],IDS,good,f['unit'][row])
            models.append((int(source),mm))
        fits.append(audit)
        if callback:callback(audit)
    states={}
    for arm,mu in mus.items():
        valid=f['valid']&np.isfinite(mu)
        states[arm]=dict(times=t,dense=True,pairs=np.array(IDS),signal_times=np.broadcast_to(t[:,None],mu.shape),source_prices=d['close'],forecast_times=t,
            beta=f['beta'],vol=f['vol'],mu=mu,directions=valid.astype(int),economic_available=valid,
            economic_action=np.where(valid&(abs(mu)>.0038),np.sign(np.nan_to_num(mu)),0).astype(int))
    return states,fits,models
