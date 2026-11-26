"""Causal daily per-pair shallow boosted tree and fixed ridge/tree mixture."""
import hashlib
from pathlib import Path
import numpy as np
import xgboost as xgb
from experiments.dense_history_pairs_20261017.model import design,labels,map_state
from experiments.rolling_feature_pairs_20261017.model import fit as ridge_fit


def fit(x,target,maturity,now,direct_spec,tree_spec):
    # Reuse the unchanged weighted linear fit to ensure the direct control has
    # exactly the same rows, recency weights, centers, penalty and intercept.
    record=ridge_fit(x,target,np.zeros(len(target)),maturity,now,direct_spec)
    if record is None:return None
    weights=np.exp2(-(now-maturity)/(direct_spec['half_life_days']*86400))
    weights*=len(weights)/weights.sum()
    y=np.clip(target,-direct_spec['target_clip'],direct_spec['target_clip'])
    params={k:v for k,v in tree_spec.items() if k!='num_boost_round'}
    params.update(objective='reg:squarederror',tree_method='hist',verbosity=0,
        subsample=1.,colsample_bytree=1.,base_score=float(record['direct'][0]),validate_parameters=True)
    matrix=xgb.DMatrix(x,label=y,weight=weights,nthread=1)
    booster=xgb.train(params,matrix,num_boost_round=tree_spec['num_boost_round'])
    blob=booster.save_raw(raw_format='ubj')
    record.update(tree_sha256=hashlib.sha256(blob).hexdigest(),tree_weight_sum=float(weights.sum()),
        tree_initial_intercept=float(record['direct'][0]),tree_rounds=booster.num_boosted_rounds(),
        tree_parameters=params,tree_runtime=xgb.__version__)
    return record,booster,blob


def infer(times,reference,symbols,pairs,direct_spec,tree_spec,model_directory=None):
    rt=reference['market_times'];h=direct_spec['horizon']
    if h!=24:raise ValueError('This frozen candidate family has only the 24h label')
    features=design(rt,reference['close'],reference['volume'],reference['trades'],symbols,pairs)
    target,maturity=labels(rt,reference['close'],features,h)
    terminal=np.zeros_like(features['fresh']);terminal[:-h]=features['fresh'][h:]
    feature_ok=np.isfinite(features['x']).all(-1)&features['fresh']
    train_ok=feature_ok&terminal&np.isfinite(target)
    predictions={kind:np.full(target.shape,np.nan) for kind in ['direct','tree','mix']}
    stamps=np.full(target.shape,-1,np.int64);records=[];models={}
    if model_directory is not None:Path(model_directory).mkdir(parents=True,exist_ok=False)
    for i,now in enumerate(rt):
        if now%(direct_spec['refit_hours']*3600)==0:
            models={}
            window=(rt>=now-direct_spec['lookback_days']*86400)&(maturity<now)
            window &= rt%(direct_spec['sample_hours']*3600)==0
            for k in range(len(pairs)):
                ix=np.flatnonzero(window&train_ok[:,k])
                fitted=fit(features['x'][ix,k],target[ix,k],maturity[ix],now,direct_spec,tree_spec)
                if fitted is None:continue
                record,booster,blob=fitted;models[k]=(record,booster)
                filename=f'pair{k}-{int(now)}.ubj'
                if model_directory is not None:(Path(model_directory)/filename).write_bytes(blob)
                records.append(dict(pair_index=k,pair=pairs[k],first_source=int(rt[ix[0]]),
                    last_source=int(rt[ix[-1]]),sample_hours=direct_spec['sample_hours'],labels_overlap=True,
                    names=features['names'],model_file=filename,**record))
        for k,(record,booster) in models.items():
            if not feature_ok[i,k] or not np.isfinite(features['vol'][i,k]) or features['vol'][i,k]<=0:continue
            x=features['x'][i,k]
            z=np.r_[1.,(x-record['center'])/record['scale']]
            direct=float(np.clip(z@record['direct'],-direct_spec['prediction_clip'],direct_spec['prediction_clip']))
            tree=float(np.clip(booster.predict(xgb.DMatrix(x[None,:],nthread=1))[0],
                -direct_spec['prediction_clip'],direct_spec['prediction_clip']))
            values={'direct':direct,'tree':tree,'mix':(direct+tree)/2}
            for kind,value in values.items():predictions[kind][i,k]=value*features['vol'][i,k]*np.sqrt(h)
            stamps[i,k]=record['fit_time']
    states={kind:map_state(times,reference,features,mu,features['pairs'],source_hours=1,lifetime_hours=1)
            for kind,mu in predictions.items()}
    return states,records,dict(times=rt,target=target,maturity=maturity,fit_stamps=stamps,
        unit=features['vol']*np.sqrt(h),fresh=features['fresh'],terminal_fresh=terminal,
        **{f'prediction_{k}':v for k,v in predictions.items()})
