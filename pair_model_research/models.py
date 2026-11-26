"""Shared statistical learning over source-normalized pair returns.

All pairs enter the same temporal training split; no random pair-level split,
no asset identity feature, no fitted/hindsight profitability whitelist.
"""
import numpy as np
import xgboost as xgb


def training_rows(times, features, target, maturity, terminal, source, spec):
    window=(times>=source-spec['lookback_days']*86400)&(maturity<source)&(times%(spec['sample_hours']*3600)==0)
    mask=window[:,None]&features['fresh']&terminal&np.isfinite(target)&np.isfinite(features['x']).all(-1)
    ii,kk=np.where(mask)
    if not len(ii):return None
    # Each source timestamp has the same total influence before recency decay;
    # its45 overlapping pair targets are NOT counted as45 independent dates.
    _,counts=np.unique(ii,return_counts=True)
    distinct=np.unique(ii)
    per_time=dict(zip(distinct,counts))
    w=np.exp2(-(source-maturity[ii])/(spec['half_life_days']*86400))/np.array([per_time[i] for i in ii])
    w*=len(distinct)/w.sum()
    return features['x'][ii,kk],np.clip(target[ii,kk],-spec['target_clip'],spec['target_clip']),w,ii,kk


def fit(times,features,target,maturity,terminal,source,spec):
    rows=training_rows(times,features,target,maturity,terminal,source,spec)
    if rows is None:return None,None
    x,y,w,ii,kk=rows
    if len(np.unique(ii))<spec['minimum_source_times']:return None,None
    assert np.all(maturity[ii]<source)
    normalized=w/w.sum();center=normalized@x
    scale=np.maximum(np.sqrt(normalized@((x-center)**2)),1e-6)
    z=np.c_[np.ones(len(x)),(x-center)/scale]
    penalty=np.eye(z.shape[1])*spec['ridge_penalty'];penalty[0]=0;penalty[0,0]=spec['intercept_prior_sources']/len(np.unique(ii))
    coefficient=np.linalg.solve(z.T@(normalized[:,None]*z)+penalty,z.T@(normalized*y))
    params=dict(spec['tree']);rounds=params.pop('rounds')
    params.update(objective='reg:squarederror',tree_method='hist',nthread=1,seed=20261023,
                  base_score=float(normalized@y),verbosity=0)
    tree=xgb.train(params,xgb.DMatrix(x,label=y,weight=w),num_boost_round=rounds)
    model=dict(source=int(source),first_maturity=int(maturity[ii].min()),last_maturity=int(maturity[ii].max()),
      rows=len(ii),distinct_source_times=len(np.unique(ii)),represented_pairs=np.unique(kk).tolist(),
      center=center.tolist(),scale=scale.tolist(),coefficient=coefficient.tolist(),names=features['names'])
    return model,tree


def predict(features,indexes,model,tree,horizon,spec):
    shape=(len(indexes),features['x'].shape[1]);ridge=np.full(shape,np.nan);ensemble=ridge.copy()
    x=features['x'][indexes];valid=np.isfinite(x).all(-1)&features['fresh'][indexes]
    unit=features['vol'][indexes]*np.sqrt(horizon)
    valid&=np.isfinite(unit)&(unit>0)
    if not valid.any():return ridge,ensemble
    z=np.c_[np.ones(valid.sum()),(x[valid]-model['center'])/model['scale']]
    linear=z@model['coefficient'];nonlinear=tree.predict(xgb.DMatrix(x[valid]))
    ridge[valid]=np.clip(linear,-spec['prediction_clip'],spec['prediction_clip'])*unit[valid]
    ensemble[valid]=np.clip(.5*linear+.5*nonlinear,-spec['prediction_clip'],spec['prediction_clip'])*unit[valid]
    return ridge,ensemble
