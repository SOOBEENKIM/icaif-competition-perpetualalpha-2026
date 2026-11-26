"""Purged, pooled weekly price prediction and lagged funding forecasts."""
import numpy as np
import pandas as pd

HORIZON = 168


def features(d):
    lp = pd.DataFrame(np.log(d['close']))
    r = lp.diff()
    vol = r.ewm(span=168, min_periods=168, adjust=False).std().clip(lower=.0005)
    x = [(lp-lp.shift(h))/(vol*np.sqrt(h)) for h in (24,168,672)]
    x += [(lp-lp.shift(1).rolling(168).mean())/(vol*np.sqrt(168)),
          r.ewm(span=24, min_periods=24, adjust=False).std()/vol-1]
    # No asset identifier, asset-specific intercept, or subsequent return.
    x = np.clip(np.stack([v.to_numpy() for v in x], -1), -4., 4.)
    rates = pd.DataFrame(np.where(d['observed'],d['funding'],np.nan)).shift(1)
    fast = rates.ewm(span=24, min_periods=24, adjust=False).mean().to_numpy()
    slow = rates.ewm(span=168, min_periods=168, adjust=False).mean().to_numpy()
    funding = np.where(fast*slow>0,np.sign(slow)*np.minimum(abs(fast),abs(slow)),0.)
    funding = np.nan_to_num(funding,nan=0.)
    return x,vol.to_numpy()*np.sqrt(HORIZON),funding*HORIZON


def rows(d,x,unit,start,end):
    t=d['market_times'];n=len(t)
    # Non-overlapping weekly observations per asset. Strict label maturity.
    clock=(t>=start)&(t+HORIZON*3600<end)&(t%(HORIZON*3600)==0)
    i,j=np.where(clock[:,None]&np.isfinite(x).all(-1))
    inside=i+HORIZON<n;i,j=i[inside],j[inside]
    fresh=(d['volume'][i,j]>0)&(d['trades'][i,j]>0)&(d['volume'][i+HORIZON,j]>0)&(d['trades'][i+HORIZON,j]>0)
    i,j=i[fresh],j[fresh]
    y=np.log(d['close'][i+HORIZON,j]/d['close'][i,j])/unit[i,j]
    return i,j,y


def predict(x,unit,fit):
    z=(x-np.asarray(fit['center']))/np.asarray(fit['scale'])
    # Signed model, no long-only transformation. Bias is pooled and regularized.
    y=np.asarray(fit['beta'])[0]+z@np.asarray(fit['beta'])[1:]
    return np.clip(y,-2,2)*unit


def train(d,p):
    train={k:v[d['market_times']<p['train_end']] for k,v in d.items()}
    x,unit,_=features(train);i,j,y=rows(train,x,unit,p['train_start'],p['train_end'])
    if len(y)<40:raise ValueError('Insufficient weekly training rows')
    a=x[i,j];center=a.mean(0);scale=np.maximum(a.std(0),.1)
    z=np.c_[np.ones(len(y)),(a-center)/scale]
    penalty=np.diag([.25]+[1.]*a.shape[1])
    beta=np.linalg.solve(z.T@z/len(y)+penalty,z.T@np.clip(y,-4,4)/len(y))
    fit=dict(center=center.tolist(),scale=scale.tolist(),beta=beta.tolist(),rows=len(y),
             asset_rows=np.bincount(j,minlength=10).tolist(),
             last_training_maturity=int((train['market_times'][i]+HORIZON*3600).max()))
    cal={k:v[d['market_times']<p['calibration_end']] for k,v in d.items()}
    x,u,_=features(cal);ci,cj,cy=rows(cal,x,u,p['calibration_start'],p['calibration_end'])
    if len(cy)<10:raise ValueError('Insufficient calibration rows')
    cyhat=predict(x,u,fit)[ci,cj]/u[ci,cj]
    attenuation=float(np.clip(cyhat@cy/(cyhat@cyhat+1.),0.,1.))
    fit.update(attenuation=attenuation,calibration_rows=len(cy),
        last_calibration_maturity=int((cal['market_times'][ci]+HORIZON*3600).max()),
        calibration_mae=float(np.mean(abs(cyhat-cy))),zero_mae=float(np.mean(abs(cy))))
    return fit
