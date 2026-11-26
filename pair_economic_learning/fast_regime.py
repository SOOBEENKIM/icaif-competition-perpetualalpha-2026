"""Use matured short-horizon forecast errors to identify the current regime.

This selects the unchanged72h AR or trend forecast; it does not train on PAS
targets, refit the source hedge from the future, or choose retrospective pairs.
"""
import numpy as np
from auto_pair_selection.selection import IDS
from pair_value_learning.scaled_coherence import project


def short_experts(d,f,formation,h):
    t=f['times'];p=d['close'];logs=np.log(p);n=len(t);shape=f['ar'].shape
    ar=np.full(shape,np.nan);trend=ar.copy();y=ar.copy();byday={}
    for row in formation:
        if row['model'] is not None:byday.setdefault(row['source'],{})[row['pair_index']]=row['model']
    models={}
    for i,now in enumerate(t):
        if now%86400==0:models=byday.get(int(now),{})
        if i<168:continue
        good=f['valid'][i];unit=f['vol'][i]*np.sqrt(h);raw=ar[i].copy();drift=trend[i].copy()
        asset=np.expm1(np.log(p[i]/p[i-168])*h/168)
        for k in np.flatnonzero(good):
            a,b=IDS[k];m=models[k];beta=f['beta'][i,k];spread=logs[i,a]-beta*logs[i,b]
            raw[k]=(m['equilibrium']-spread)*(1-m['rho']**h)/(1+beta)
            drift[k]=(asset[a]-beta*asset[b])/(1+beta)
        ar[i],_=project(np.clip(raw,-4*unit,4*unit),f['beta'][i],IDS,good,unit)
        trend[i],_=project(np.clip(drift,-4*unit,4*unit),f['beta'][i],IDS,good,unit)
    for k,(a,b) in enumerate(IDS):
        ret=p[h:,[a,b]]/p[:-h,[a,b]]-1;beta=f['beta'][:-h,k]
        y[:-h,k]=(ret[:,0]-beta*ret[:,1])/(1+beta)
    return ar,trend,y


def fit(t,ar,trend,y,vol,valid,source,h):
    mask=(t>=source-28*86400)&(t+h*3600<source)&(t%(h*3600)==0)
    ii=np.flatnonzero(mask);u=vol[ii]*np.sqrt(h)
    good=valid[ii]&np.isfinite(ar[ii])&np.isfinite(trend[ii])&np.isfinite(y[ii])&(u>0)
    ay=np.divide(y[ii],u,out=np.zeros_like(y[ii]),where=u>0);ay=np.clip(ay,-4,4)
    aa=np.divide(ar[ii],u,out=np.zeros_like(ar[ii]),where=u>0);bb=np.divide(trend[ii],u,out=np.zeros_like(trend[ii]),where=u>0)
    diff=np.where(good,(bb-ay)**2-(aa-ay)**2,0.)
    decay=np.exp2(-(source-(t[ii]+h*3600))/(7*86400))
    wg=good*decay[:,None]/np.maximum(1,good.sum(1))[:,None]
    prior=float((wg*diff).sum()/max(wg.sum(),1e-12));w=good*decay[:,None]
    evidence=(np.sum(w*diff,axis=0)+4*prior)/(w.sum(0)+4)
    counts=good.sum(0);evidence=np.where(counts>=8,evidence,prior)
    return evidence,dict(source=int(source),horizon=h,last_maturity=int((t[ii]+h*3600)[good.any(1)].max()) if good.any() else None,
                         global_loss_advantage=prior,mean_reversion_pairs=int((evidence>=0).sum()),trend_pairs=int((evidence<0).sum()),pair_counts=counts.tolist())


def build(d,f,formation,start):
    t=f['times'];states={};records=[]
    for h in [24,6]:
        ar,tr,y=short_experts(d,f,formation,h);valid=f['valid'].copy();valid[:-h]&=f['valid'][h:];mu=np.full_like(f['ar'],np.nan)
        for source in np.unique(t[t>=start]//21600*21600):
            ev,audit=fit(t,ar,tr,y,f['vol'],valid,int(source),h);records.append(audit)
            ii=np.flatnonzero((t>=source)&(t<source+21600))
            for i in ii:
                raw=np.where(ev>=0,f['ar'][i],f['trend'][i]);good=f['valid'][i]
                mu[i],_=project(raw,f['beta'][i],IDS,good,f['unit'][i])
        allowed=f['valid']&np.isfinite(mu)
        states[f'regime{h}_72']=dict(times=t,dense=True,pairs=np.array(IDS),signal_times=np.broadcast_to(t[:,None],mu.shape),source_prices=d['close'],forecast_times=t,
           beta=f['beta'],vol=f['vol'],mu=mu,directions=allowed.astype(int),economic_available=allowed,
           economic_action=np.where(allowed&(abs(mu)>.0038),np.sign(np.nan_to_num(mu)),0).astype(int))
    return states,records
