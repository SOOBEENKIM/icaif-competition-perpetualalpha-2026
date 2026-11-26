"""PAS-only pooled regression; source/terminal freshness and strict maturity."""
import numpy as np
import pandas as pd

def features(times,prices,fresh,carry,basis,horizon):
    lp=pd.DataFrame(np.log(prices));ret=lp.diff()
    vol=ret.ewm(span=24,min_periods=6,adjust=False).std().clip(lower=.0005).to_numpy()
    unit=vol*np.sqrt(horizon)
    parts=[(lp-lp.shift(h)).to_numpy()/(vol*np.sqrt(h)) for h in [1,3,6,12]]
    parts += [(lp-lp.shift(1).rolling(12,min_periods=6).mean()).to_numpy()/(vol*np.sqrt(12)),
              carry/unit,basis/unit]
    x=np.clip(np.stack(parts,-1),-4.,4.)
    return x,unit

def fit_predict(times,prices,fresh,carry,basis,horizon):
    t=np.asarray(times);x,u=features(t,prices,fresh,carry,basis,horizon)
    out=np.full(prices.shape,np.nan);records=[];fit=None
    y=np.full(prices.shape,np.nan);y[:-horizon]=(prices[horizon:]/prices[:-horizon]-1)/u[:-horizon]
    good=np.isfinite(x).all(-1)&np.isfinite(y)&fresh
    terminal=np.zeros_like(fresh);terminal[:-horizon]=fresh[horizon:];good&=terminal
    maturity=t+horizon*3600
    for i,now in enumerate(t):
        if now%21600==0:
            valid=good&(maturity<now)[:,None]&(t>=now-72*3600)[:,None]
            a,b=np.where(valid)
            if len(a)>=60 and len(np.unique(a))>=6:
                w=2.**(-(now-maturity[a])/86400)/np.bincount(a,minlength=len(t))[a];w/=w.sum()
                xx=x[a,b];yy=np.clip(y[a,b],-4,4)
                center=(w[:,None]*xx).sum(0);scale=np.maximum(np.sqrt((w[:,None]*(xx-center)**2).sum(0)),.25)
                z=np.c_[np.ones(len(a)),(xx-center)/scale]
                coeff=np.linalg.solve(z.T@(w[:,None]*z)+np.diag([.25]+[1.]*xx.shape[1]),z.T@(w*yy))
                fit=(center,scale,coeff)
                records.append(dict(fit_time=int(now),rows=len(a),unique_sources=len(set(a)),last_maturity=int(maturity[a].max()),
                                    center=center.tolist(),scale=scale.tolist(),coef=coeff.tolist()))
                assert maturity[a].max()<now
        if fit is not None:
            center,scale,coeff=fit
            out[i]=np.clip(coeff[0]+((x[i]-center)/scale)@coeff[1:],-4.,4.)*u[i]
            out[i,~fresh[i]]=np.nan
    return out,records
