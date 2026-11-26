"""Hierarchical return calibration with source-frozen physical hedge labels.

Relations are fitted using only past hourly returns. Supervised targets hold
that source's beta fixed; a later refitted residual is never a return label.
"""
from dataclasses import dataclass
import numpy as np
from auto_pair_selection.selection import IDS, PAIRS, validate_snapshot
from pair_regime_adaptive.model import Spec as RelationSpec, fit as fit_relation


@dataclass(frozen=True)
class Spec:
    history_days: int=90
    decay_days: int=30
    source_stride_hours: int=6
    minimum_global_sources: int=80
    minimum_local_sources: int=40
    ridge: float=1.
    local_prior: float=2.
    clip: float=4.
    proxy_cost: float=.0038


def features(d,horizon):
    times=validate_snapshot(d);logs=np.log(d['close']);active=(d['volume']>0)&(d['trades']>0)
    n=len(times);q=len(IDS);x=np.full((n,q,8),np.nan);beta=np.full((n,q),np.nan);vol=beta.copy()
    allowed=np.zeros((n,q),bool);fresh=np.zeros_like(allowed);audits=[];models={};eligible={}
    for i,source in enumerate(times):
        if source%86400==0:
            models={};eligible={};rows=[]
            for k,ids in enumerate(IDS):
                model,audit=fit_relation(logs,active,i,ids,RelationSpec(horizon=horizon))
                if model is not None:models[k]=model
                eligible[k]=audit['eligible'];rows.append(dict(pair=list(PAIRS[k]),**audit))
            audits.append(dict(source=int(source),pairs=rows))
        for k,m in models.items():
            a,b=IDS[k];bt=m['beta'];u=m['vol']*np.sqrt(horizon)
            beta[i,k]=bt;vol[i,k]=m['vol'];allowed[i,k]=eligible[k]
            if i<168 or not active[[i,i-24,i-72,i-168]][:,[a,b]].all():continue
            fresh[i,k]=True
            s=logs[i,a]-bt*logs[i,b]
            mom=[(s-(logs[i-h,a]-bt*logs[i-h,b]))/(1+bt)/u for h in [24,72,168]]
            mr=m['mr'];rev=(mr['equilibrium']-s)*(1-mr['rho']**horizon)/(1+bt)/u if mr else 0.
            # These interactions distinguish persistent drift from an isolated
            # dislocation without forcing either regime to be profitable.
            x[i,k]=np.clip([rev,*mom,rev*np.sign(mom[1])*np.sign(mom[2]),
                            mom[0]-mom[1]/3,float(mr is not None),np.log(bt)],-8,8)
    y=np.full((n,q),np.nan);terminal=np.zeros_like(allowed)
    for k,(a,b) in enumerate(IDS):
        ret=d['close'][horizon:,[a,b]]/d['close'][:-horizon,[a,b]]-1
        bt=beta[:-horizon,k];u=vol[:-horizon,k]*np.sqrt(horizon)
        y[:-horizon,k]=(ret[:,0]-bt*ret[:,1])/(1+bt)/u
        terminal[:-horizon,k]=active[horizon:,a]&active[horizon:,b]
    return dict(times=times,x=x,y=y,maturity=times+horizon*3600,beta=beta,vol=vol,
                allowed=allowed,fresh=fresh,terminal=terminal),audits


def fit_day(f,source,spec=Spec()):
    t=f['times'];mask=(t>=source-spec.history_days*86400)&(f['maturity']<source)&(t%(spec.source_stride_hours*3600)==0)
    valid=mask[:,None]&f['fresh']&f['terminal']&f['allowed']&np.isfinite(f['y'])&np.isfinite(f['x']).all(-1)
    ii,kk=np.where(valid);unique=np.unique(ii)
    if len(unique)<spec.minimum_global_sources:return None
    assert (f['maturity'][ii]<source).all()
    x=f['x'][ii,kk];y=np.clip(f['y'][ii,kk],-spec.clip,spec.clip)
    counts=np.bincount(ii,minlength=len(t));w=np.exp2(-(source-f['maturity'][ii])/(spec.decay_days*86400))/counts[ii];w/=w.sum()
    center=w@x;scale=np.maximum(np.sqrt(w@((x-center)**2)),.01);z=np.c_[np.ones(len(x)),(x-center)/scale]
    penalty=np.eye(z.shape[1])*spec.ridge;penalty[0,0]=.1
    coef=np.linalg.solve(z.T@(w[:,None]*z)+penalty,z.T@(w*y))
    locals_={}
    for k in np.unique(kk):
        use=kk==k
        if use.sum()<spec.minimum_local_sources:continue
        # Shrink local slopes and bias toward shared information, not a list
        # of retrospectively profitable pair names.
        ww=np.exp2(-(source-f['maturity'][ii[use]])/(spec.decay_days*86400));ww/=ww.sum()
        zz=z[use];yy=y[use];prior=np.eye(z.shape[1])*spec.local_prior
        locals_[int(k)]=np.linalg.solve(zz.T@(ww[:,None]*zz)+prior,zz.T@(ww*yy)+prior@coef)
    return dict(source=int(source),last_maturity=int(f['maturity'][ii].max()),
                sources=len(unique),rows=len(ii),pairs=len(np.unique(kk)),center=center,scale=scale,
                coefficient=coef,local=locals_)


def predict(f,ii,m,kind,horizon,spec=Spec()):
    x=f['x'][ii];good=f['fresh'][ii]&np.isfinite(x).all(-1);out=np.full(good.shape,np.nan)
    for k in range(len(IDS)):
        g=good[:,k]
        if not g.any():continue
        z=np.c_[np.ones(g.sum()),(x[g,k]-m['center'])/m['scale']]
        coef=m['local'].get(k,m['coefficient']) if kind=='hierarchical' else m['coefficient']
        out[g,k]=np.clip(z@coef,-spec.clip,spec.clip)*f['vol'][ii[g],k]*np.sqrt(horizon)
    return out


def build(d,start,horizon,spec=Spec()):
    f,audit=features(d,horizon);t=f['times'];mu={a:np.full(f['vol'].shape,np.nan) for a in ['pooled','hierarchical']};fits=[]
    for source in np.unique(t[t>=start]//86400*86400):
        m=fit_day(f,int(source),spec)
        if m is None:continue
        ii=np.flatnonzero((t>=source)&(t<source+86400))
        for kind in mu:mu[kind][ii]=predict(f,ii,m,kind,horizon,spec)
        fits.append(m)
    states={}
    for kind,mean in mu.items():
        available=np.isfinite(mean)&f['fresh'];action=np.where(available&f['allowed']&(abs(mean)>spec.proxy_cost),np.sign(np.nan_to_num(mean)),0).astype(int)
        states[f'{kind}_{horizon}']=dict(times=t,dense=True,pairs=np.asarray(IDS),signal_times=np.broadcast_to(t[:,None],mean.shape),source_prices=d['close'],forecast_times=t,
            beta=f['beta'],vol=f['vol'],mu=mean,economic_available=available,economic_action=action,directions=available.astype(int),relationship_allowed=f['allowed'],regime_allowed=np.ones_like(available))
    return states,fits,audit,f
