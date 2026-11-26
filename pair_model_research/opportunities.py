"""Second-stage conditional return estimates from published first-stage forecasts.

Labels describe future relative price returns, not synthetic executable fills.
Every training forecast must have been published out of time and every target
must have matured before the fit. Exact fills/costs are checked by replay.
"""
import numpy as np
import xgboost as xgb


def features(x,raw,unit):
    sign=np.sign(raw)
    return np.concatenate([x,x*sign[...,None],(abs(raw)/unit)[...,None],(.0038/unit)[...,None]],axis=-1)


def fit(times,x,raw,unit,actual,maturity,source,spec):
    temporal=(times>=source-spec['lookback_days']*86400)&(maturity<source)&(times%21600==0)
    ix=np.flatnonzero(temporal)
    # Crop to mature history BEFORE numerical transforms; future poison ignored.
    xx=features(x[ix],raw[ix],unit[ix]);yy=np.sign(raw[ix])*actual[ix]/unit[ix]
    good=np.isfinite(xx).all(-1)&np.isfinite(yy)&(unit[ix]>0)&(abs(raw[ix])>.0038)
    ii,kk=np.where(good)
    if len(ii)<spec['minimum_rows'] or len(np.unique(ii))<spec['minimum_sources']:return None,None
    xx=xx[ii,kk];yy=np.clip(yy[ii,kk],-4,4);stamps=times[ix[ii]]
    unique,inv,counts=np.unique(stamps,return_inverse=True,return_counts=True)
    weights=np.exp2(-(source-maturity[ix[ii]])/(30*86400))/counts[inv]
    weights*=len(unique)/weights.sum();w=weights/weights.sum()
    center=w@xx;scale=np.maximum(np.sqrt(w@((xx-center)**2)),1e-6);z=np.c_[np.ones(len(xx)),(xx-center)/scale]
    penalty=np.eye(z.shape[1])*spec['ridge_penalty'];penalty[0,0]=12/len(unique)
    coefficient=np.linalg.solve(z.T@(w[:,None]*z)+penalty,z.T@(w*yy))
    params=dict(max_depth=2,eta=.05,min_child_weight=8,reg_lambda=3.,subsample=1.,colsample_bytree=1.,objective='reg:squarederror',tree_method='hist',nthread=1,seed=20261023,base_score=float(w@yy),verbosity=0)
    tree=xgb.train(params,xgb.DMatrix(xx,label=yy,weight=weights),num_boost_round=64)
    model=dict(source=int(source),rows=len(xx),source_count=len(unique),last_maturity=int(maturity[ix[ii]].max()),first_maturity=int(maturity[ix[ii]].min()),represented_pairs=np.unique(kk).tolist(),center=center.tolist(),scale=scale.tolist(),coefficient=coefficient.tolist())
    return model,tree


def predict(x,raw,unit,model,tree):
    xx=features(x,raw,unit);valid=np.isfinite(xx).all(-1)&np.isfinite(raw)&(unit>0)
    a=np.full(raw.shape,np.nan);b=a.copy()
    if valid.any():
        z=np.c_[np.ones(valid.sum()),(xx[valid]-model['center'])/model['scale']]
        linear=z@model['coefficient'];nonlinear=tree.predict(xgb.DMatrix(xx[valid]))
        # Keep original direction and cap at the original magnitude. A negative
        # estimated signed payoff vetoes entry; it is not a new reverse signal.
        a[valid]=np.sign(raw[valid])*np.minimum(abs(raw[valid]),np.maximum(0,linear)*unit[valid])
        b[valid]=np.sign(raw[valid])*np.minimum(abs(raw[valid]),np.maximum(0,.5*linear+.5*nonlinear)*unit[valid])
    return a,b


def slope(times,raw,unit,actual,maturity,source):
    """Per-pair no-intercept calibration on mature, published opportunities."""
    temporal=(times>=source-90*86400)&(maturity<source)&(times%21600==0)
    ii=np.flatnonzero(temporal);coefficients=np.zeros(raw.shape[1]);counts=[]
    for k in range(raw.shape[1]):
        use=np.isfinite(raw[ii,k])&np.isfinite(actual[ii,k])&(unit[ii,k]>0)&(abs(raw[ii,k])>.0038)
        rows=ii[use];counts.append(len(rows))
        if len(rows)<12:continue
        x=raw[rows,k]/unit[rows,k];y=actual[rows,k]/unit[rows,k]
        w=np.exp2(-(source-maturity[rows])/(30*86400));w/=4 # overlapping6h forecasts carry quarter-day weight
        coefficients[k]=np.clip((w*x)@y/((w*x)@x+6),0,1)
    return coefficients,counts
