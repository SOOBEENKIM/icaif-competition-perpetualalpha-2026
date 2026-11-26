"""Predict paid execution utility; unlike ranking, allow all pairs to be rejected."""
import numpy as np
from .ranker import prepare
from .realized_ranker import training_mask
from .realized_labels import HOURS


def build(d,f,labels,start,callback=None):
    import xgboost as xgb
    blank=np.full_like(f['ar'],np.nan)
    x,_,capacity=prepare(d,f,blank,np.zeros_like(blank,bool))
    scale=np.maximum(capacity*f['unit'],1.)
    t=f['times'];yy={k:blank.copy() for k in ['paid_value','robust_value']}
    groups={}
    for r in labels:groups.setdefault((r['index'],r['pair']),{})[r['scenario']]=r
    for (i,k),rr in groups.items():
        assert set(rr)=={'base','stress'}
        yy['paid_value'][i,k]=rr['base']['utility']/scale[i,k]
        yy['robust_value'][i,k]=min(rr[s]['utility'] for s in rr)/scale[i,k]
    scores={k:blank.copy() for k in yy};audit=[];models=[]
    for source in np.unique(t[t>=start]//86400*86400):
        for kind,y in yy.items():
            good=training_mask(t,source)[:,None]&np.isfinite(x).all(-1)&np.isfinite(y)
            ii,kk=np.where(good);origins,counts=np.unique(ii,return_counts=True)
            if len(origins)<30:continue
            count=np.zeros(len(t));count[origins]=counts
            weights=np.exp2(-(source-t[ii]-HOURS*3600)/(30*86400))/count[ii]
            dm=xgb.DMatrix(x[ii,kk],label=y[ii,kk],weight=weights,nthread=1)
            params=dict(objective='reg:pseudohubererror',huber_slope=1.,base_score=0.,
                tree_method='hist',max_bin=64,max_depth=2,eta=.04,reg_lambda=5.,
                min_child_weight=1.,nthread=1,seed=20261023,subsample=1.,colsample_bytree=1.,verbosity=0)
            model=xgb.train(params,dm,num_boost_round=64)
            for i in np.flatnonzero((t>=source)&(t<source+86400)):
                valid=np.isfinite(x[i]).all(-1)
                if valid.any():scores[kind][i,valid]=model.predict(xgb.DMatrix(x[i,valid],nthread=1))*scale[i,valid]
            row=dict(kind=kind,source=int(source),last_maturity=int((t[ii]+HOURS*3600).max()),origins=len(origins),rows=len(ii))
            assert row['last_maturity']<source
            audit.append(row);models.append((int(source),kind,model))
            if callback:callback(row)
    return scores,audit,models
