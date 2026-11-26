"""Change only target normalization in the retained daily pair regression."""
import numpy as np
from experiments.rolling_feature_pairs_20261017.model import fit as normalized_fit


def fit(x, normalized_target, units, maturity, now, spec):
    if len(x) < spec['minimum_training_rows']:
        return None
    if np.any(maturity >= now) or np.any(units <= 0):
        raise ValueError('Strictly matured labels and positive source units required')
    if not all(np.isfinite(v).all() for v in [x,normalized_target,units]):
        raise ValueError('Finite training rows required')
    weight=np.exp2(-(now-maturity)/(spec['half_life_days']*86400));weight/=weight.sum()
    center=weight@x
    scale=np.maximum(np.sqrt(weight@((x-center)**2)),1e-6)
    z=np.c_[np.ones(len(x)),(x-center)/scale]
    penalty=np.eye(z.shape[1])*spec['ridge_penalty'];penalty[0,0]=spec['intercept_prior_rows']/len(x)
    # Keep precisely the parent's per-source target clipping; do not add tail
    # winsorization or relax the prediction cap to make larger signals appear.
    target=np.clip(normalized_target,-spec['target_clip'],spec['target_clip'])*units
    coefficient=np.linalg.solve(z.T@(weight[:,None]*z)+penalty,z.T@(weight*target))
    return dict(center=center.tolist(),scale=scale.tolist(),coefficient=coefficient.tolist(),rows=len(x),
                first_maturity=int(maturity.min()),last_maturity=int(maturity.max()),fit_time=int(now),
                clipped_targets=int(np.sum(abs(normalized_target)>spec['target_clip'])))


def predict(model,x,unit,spec):
    if not np.isfinite(unit) or unit<=0:return np.nan
    z=np.r_[1.,(x-model['center'])/model['scale']]
    return float(np.clip(z@model['coefficient'],-spec['prediction_clip']*unit,spec['prediction_clip']*unit))


def rolling(times,features,target,maturity,original_raw,terminal_fresh,spec):
    if not np.all(np.diff(times)==3600) or not np.array_equal(maturity,times+spec['horizon']*3600):
        raise ValueError('Wrong source cadence or label horizon')
    if target.shape!=original_raw.shape or terminal_fresh.shape!=target.shape or features['x'].shape[:2]!=target.shape:
        raise ValueError('Unaligned features and targets')
    available=np.isfinite(features['x']).all(-1)&features['fresh']&np.isfinite(original_raw)
    train_ok=available&terminal_fresh&np.isfinite(target)
    units=features['vol']*np.sqrt(spec['horizon'])
    cash=np.full_like(target,np.nan);control=np.full_like(target,np.nan)
    stamps=np.full(target.shape,-1,np.int64);records=[];models={}
    for i,now in enumerate(times):
        if now%(spec['refit_hours']*3600)==0:
            models={}
            window=(times>=now-spec['lookback_days']*86400)&(maturity<now)&(times%(spec['sample_hours']*3600)==0)
            for k in range(target.shape[1]):
                ix=np.flatnonzero(window&train_ok[:,k])
                base=normalized_fit(features['x'][ix,k],target[ix,k],original_raw[ix,k],maturity[ix],now,spec)
                candidate=fit(features['x'][ix,k],target[ix,k],units[ix,k],maturity[ix],now,spec)
                assert (candidate is None)==(base is None)
                if candidate is not None:
                    np.testing.assert_array_equal(candidate['center'],base['center'])
                    np.testing.assert_array_equal(candidate['scale'],base['scale'])
                    models[k]=(base,candidate)
                    records.append(dict(pair_index=k,first_source=int(times[ix[0]]),last_source=int(times[ix[-1]]),
                                        names=features['names'],**candidate))
        for k,(base,candidate) in models.items():
            if not available[i,k] or not np.isfinite(features['vol'][i,k]) or features['vol'][i,k]<=0:continue
            z=np.r_[1.,(features['x'][i,k]-base['center'])/base['scale']]
            value=float(z@base['direct'])
            # Exact operation order of the frozen control, not just close numerically.
            control[i,k]=np.clip(value,-spec['prediction_clip'],spec['prediction_clip'])*features['vol'][i,k]*np.sqrt(spec['horizon'])
            cash[i,k]=predict(candidate,features['x'][i,k],units[i,k],spec)
            stamps[i,k]=int(now//(spec['refit_hours']*3600)*(spec['refit_hours']*3600))
    return cash,control,stamps,records
