"""Frozen 24h reference predictor, with chronological calibration.

PAS basis adjustment is a separate causal, shrinkage AR estimate. Its one-hour
observations must have matured at decision time; no 24h evaluation labels are
passed to this estimator. This is an explicitly online component, not a claim
of a completely frozen PAS fit or proven convergence to the reference market.
"""
import numpy as np
import pandas as pd

H = 24


def features(d):
    logp = pd.DataFrame(np.log(d['close']))
    r = logp.diff()
    vol = r.ewm(span=168, min_periods=168, adjust=False).std().clip(lower=.0005)
    x = [(logp-logp.shift(h))/(vol*np.sqrt(h)) for h in (3, 24, 168)]
    x += [(logp-logp.shift(1).rolling(24).mean())/(vol*np.sqrt(24)),
          r.ewm(span=24, min_periods=24, adjust=False).std()/vol-1]
    return np.clip(np.stack([v.to_numpy() for v in x],-1),-4,4),vol.to_numpy()*np.sqrt(H)


def rows(d, x, unit, start, end):
    t=d['market_times']
    i,j=np.where(((t>=start)&(t+H*3600<end)&(t%86400==0))[:,None]&np.isfinite(x).all(-1))
    ok=i+H<len(t);i,j=i[ok],j[ok]
    ok=(t[i+H]==t[i]+H*3600)&(d['volume'][i,j]>0)&(d['volume'][i+H,j]>0)
    i,j=i[ok],j[ok]
    y=np.log(d['close'][i+H,j]/d['close'][i,j])/unit[i,j]
    return i,j,y


def train(d, train_start, train_end, calibration_end):
    # Truncation is enforced before computing features or targets.
    tr={k:v[d['market_times']<train_end] for k,v in d.items()}
    x,u=features(tr);i,j,y=rows(tr,x,u,train_start,train_end)
    if len(y)<40: raise ValueError('Insufficient matured training observations')
    a=x[i,j];center=a.mean(0);scale=np.maximum(a.std(0),.1)
    z=np.c_[np.ones(len(y)),(a-center)/scale];target=np.clip(y,-4,4)
    penalty=np.diag([.25]+[1.]*a.shape[1]);inv=np.linalg.inv(z.T@z+len(y)*penalty)
    beta=inv@(z.T@target);err=target-z@beta
    # Cluster coefficient uncertainty by entry day, not by asset-row count.
    groups=np.unique(tr['market_times'][i]);meat=np.zeros((len(beta),len(beta)))
    for g in groups:
        ids=tr['market_times'][i]==g;v=z[ids].T@err[ids];meat+=np.outer(v,v)
    covariance=inv@meat@inv*len(groups)/max(1,len(groups)-1)
    fit=dict(center=center.tolist(),scale=scale.tolist(),beta=beta.tolist(),covariance=covariance.tolist(),
             training_rows=len(y),training_days=len(groups),training_end=train_end,
             last_training_maturity=int((tr['market_times'][i]+H*3600).max()),horizon_hours=H)
    cal={k:v[d['market_times']<calibration_end] for k,v in d.items()}
    cx,cu=features(cal);ci,cj,cy=rows(cal,cx,cu,train_end,calibration_end)
    cz=np.c_[np.ones(len(cy)),(cx[ci,cj]-center)/scale];raw=np.clip(cz@beta,-2,2)
    attenuation=float(np.clip(raw@cy/(raw@raw+1),0,1))
    fit.update(attenuation=attenuation,calibration_rows=len(cy),calibration_days=len(np.unique(cal['market_times'][ci])),
               last_calibration_maturity=int((cal['market_times'][ci]+H*3600).max()),
               calibration_mae=float(np.mean(abs(raw*attenuation-cy))),zero_mae=float(np.mean(abs(cy))))
    return fit


def predict(d,fit):
    x,u=features(d);z=np.concatenate([np.ones((*u.shape,1)),(x-fit['center'])/fit['scale']],axis=-1)
    raw=z@np.array(fit['beta']);logmu=np.clip(raw,-2,2)*u*fit['attenuation']
    se=np.sqrt(np.maximum(0,np.einsum('...i,ij,...j->...',z,np.array(fit['covariance']),z)))
    uncertainty=u*se*fit['attenuation']
    return logmu,uncertainty


def basis_adjustment(basis):
    """Past hourly log PAS/reference basis, NaNs retained for stale observations."""
    b=np.asarray(basis,float)
    if b.ndim!=1: raise ValueError('Basis must be a vector')
    good=np.isfinite(b[:-1])&np.isfinite(b[1:])
    if good.sum()<24 or not np.isfinite(b[-1]):return 0.,0.,int(good.sum()),1.
    center=float(np.nanmean(b));x=b[:-1][good]-center;dy=np.diff(b)[good]
    scale=max(float(np.std(x)),.0005);z=x/scale
    # Prior is zero drift (rho=1), not guaranteed convergence (rho=0).
    kappa=float((z@(dy/scale))/(z@z+24))
    rho=float(np.clip(1+kappa,.8,1.02));drift=float((rho**H-1)*(b[-1]-center))
    residual=dy-(rho-1)*x
    # Conservative residual accumulated over the forecast horizon, discounted
    # only by the number of observed days, not number of minute/asset rows.
    se=float(np.std(residual,ddof=1)*np.sqrt(H)/np.sqrt(max(1,good.sum()/24)))
    return drift,se,int(good.sum()),rho
