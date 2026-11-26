import numpy as np
from experiments.rolling_feature_pairs_20261017.model import fit


def fit_day(times,x,target,maturity,fresh,terminal,source,spec):
    window=(times>=source-spec['lookback_days']*86400)&(maturity<source)&(times%(spec['sample_hours']*3600)==0)
    models={}
    for k in range(x.shape[1]):
        ix=np.flatnonzero(window&fresh[:,k]&terminal[:,k]&np.isfinite(x[:,k]).all(-1)&np.isfinite(target[:,k]))
        m=fit(x[ix,k],target[ix,k],np.zeros(len(ix)),maturity[ix],source,spec)
        if m is not None:models[k]=m
    return models


def predict(x,unit,fresh,models,spec):
    out=np.full(unit.shape,np.nan)
    for k,m in models.items():
        good=fresh[:,k]&np.isfinite(x[:,k]).all(-1)&np.isfinite(unit[:,k])&(unit[:,k]>0)
        z=np.c_[np.ones(good.sum()),(x[good,k]-m['center'])/m['scale']]
        out[good,k]=np.clip(z@m['direct'],-spec['prediction_clip'],spec['prediction_clip'])*unit[good,k]
    return out


def consensus(learned,ar):
    # Same horizon and equal-dollar target units; neither model is treated as
    # ground truth. Disagreement blocks entries but preserves a holding mean.
    both=np.isfinite(learned)&np.isfinite(ar)
    mean=np.where(both,(learned+ar)/2,learned)
    entry=both&(np.sign(learned)==np.sign(ar))&(abs(mean)>.0043)
    return mean,entry
