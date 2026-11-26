"""Recent-domain calibration of frozen source-anchored price predictions."""
import numpy as np
from experiments.dense_history_pairs_20261017.model import labels,linear_predict
from experiments.dense_history_pairs_20261017.research import one_pair


def frozen_predictions(d,models,pairs,h):
    result=np.full(d['beta'].shape,np.nan)
    for k,ids in enumerate(d['pairs']):
        record=models[tuple(pairs[k]),h];single=one_pair(d,ids)
        if record['names']!=single['names']:raise ValueError('Mismatched model feature identities')
        x=single['x'][:,0];valid=np.isfinite(x).all(1)
        result[valid,k]=linear_predict(record,x[valid])
    return result


def fit_correction(prediction,target,maturities,now,spec):
    if len(prediction)<spec['minimum_calibration_rows']:return None
    if np.any(maturities>=now):raise ValueError('Unmatured observation in calibration')
    weight=np.exp2(-(now-maturities)/(spec['half_life_days']*86400))
    effective=float(weight.sum()**2/(weight@weight));weight=weight/weight.sum()*effective
    prior=spec['prior_rows'];p=np.asarray(prediction);y=np.clip(target,-spec['normalized_target_clip'],spec['normalized_target_clip'])
    bias=float(weight@(y-p)/(effective+prior));recent=float(weight@y/(effective+prior))
    z=np.column_stack([np.ones(len(p)),p]);penalty=np.eye(2)*prior
    coefficient=np.linalg.solve(z.T@(weight[:,None]*z)+penalty,z.T@(weight*y)+np.array([0.,prior]))
    slope=float(np.clip(coefficient[1],*spec['affine_slope_bounds']))
    intercept=float(weight@(y-slope*p)/(effective+prior))
    return dict(rows=len(p),effective_rows=effective,bias=bias,recent=recent,
        affine_intercept=intercept,affine_slope=slope,
        first_maturity=int(min(maturities)),last_maturity=int(max(maturities)),
        clipped_targets=int(np.sum(abs(target)>spec['normalized_target_clip'])))


def adapt(times,target,maturity,raw,fresh,unit,h,spec):
    if not np.all(np.diff(times)==3600):raise ValueError('Hourly contiguous sources required')
    if target.shape!=raw.shape or fresh.shape!=raw.shape or unit.shape!=raw.shape:raise ValueError('Aligned pair matrices required')
    forecasts={name:np.full(raw.shape,np.nan) for name in ['bias','affine','recent']};records=[]
    sample=(times%(h*3600)==0);valid=np.isfinite(target)&np.isfinite(raw)&fresh
    for i in np.flatnonzero(times%(spec['decision_hours']*3600)==0):
        now=int(times[i]);window=sample&(maturity<now)&(maturity>=now-spec['lookback_days']*86400)
        for k in range(raw.shape[1]):
            if not np.isfinite(raw[i,k]) or not np.isfinite(unit[i,k]) or not fresh[i,k]:continue
            ix=np.flatnonzero(window&valid[:,k])
            record=fit_correction(raw[ix,k],target[ix,k],maturity[ix],now,spec)
            if record is None:continue
            assert np.all(np.diff(times[ix])>=h*3600)
            values=dict(bias=raw[i,k]+record['bias'],affine=record['affine_intercept']+record['affine_slope']*raw[i,k],recent=record['recent'])
            for name,value in values.items():forecasts[name][i,k]=float(np.clip(value,-spec['normalized_target_clip'],spec['normalized_target_clip'])*unit[i,k])
            records.append(dict(time=now,pair_index=k,first_source=int(times[ix[0]]),last_source=int(times[ix[-1]]),
                frozen_normalized_prediction=float(raw[i,k]),**record))
    return forecasts,records


def build(reference,d,models,pairs,h,spec):
    times=reference['market_times'];target,maturity=labels(times,reference['close'],d,h)
    raw=frozen_predictions(d,models,pairs,h);unit=d['vol']*np.sqrt(h)
    corrected,records=adapt(times,target,maturity,raw,d['fresh'],unit,h,spec)
    return dict(ridge=raw*unit,zero=np.where(np.isfinite(raw),0.,np.nan),**corrected),records
