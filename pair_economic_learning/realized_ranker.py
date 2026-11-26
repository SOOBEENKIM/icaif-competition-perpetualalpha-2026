"""Prequential opportunity ranking against exact conditional execution labels."""
import numpy as np
from .ranker import prepare
from .realized_labels import HOURS


def training_mask(times, source):
    maturity=times+HOURS*3600
    return (times>=source-90*86400)&(maturity<source)


def build(d,f,labels,start,callback=None):
    import xgboost as xgb
    empty=np.full_like(f['ar'],np.nan)
    x,_,_=prepare(d,f,empty,np.zeros_like(empty,bool))
    t=f['times']; y={k:np.full_like(empty,np.nan) for k in ['realized','robust']}
    grouped={}
    for r in labels:
        grouped.setdefault((r['index'],r['pair']),{})[r['scenario']]=r
    for (i,k),rr in grouped.items():
        if set(rr)!=set(['base','stress']):raise ValueError('Incomplete cost pair')
        assert rr['base']['maturity']==int(t[i]+HOURS*3600)
        y['realized'][i,k]=rr['base']['utility']
        y['robust'][i,k]=min(rr[s]['utility'] for s in ['base','stress'])
    scores={k:np.full_like(empty,np.nan) for k in y};audit=[];models=[]
    for source in np.unique(t[t>=start]//86400*86400):
        for kind,target in y.items():
            good=training_mask(t,source)[:,None]&np.isfinite(x).all(-1)&np.isfinite(target)
            good[good.sum(1)<2]=False
            ii,kk=np.where(good);origins,counts=np.unique(ii,return_counts=True)
            if len(origins)<30:continue
            weights=np.exp2(-(source-(t[origins]+HOURS*3600))/(30*86400))
            dm=xgb.DMatrix(x[ii,kk],label=target[ii,kk],group=counts,weight=weights,nthread=1)
            params=dict(objective='rank:pairwise',tree_method='hist',max_bin=64,max_depth=2,
                eta=.04,reg_lambda=5.,min_child_weight=1.,nthread=1,seed=20261023,
                subsample=1.,colsample_bytree=1.,lambdarank_pair_method='topk',
                lambdarank_num_pair_per_sample=8,verbosity=0)
            model=xgb.train(params,dm,num_boost_round=64)
            for i in np.flatnonzero((t>=source)&(t<source+86400)):
                valid=np.isfinite(x[i]).all(-1)
                if valid.any():scores[kind][i,valid]=model.predict(xgb.DMatrix(x[i,valid],nthread=1))
            row=dict(kind=kind,source=int(source),last_maturity=int((t[ii]+HOURS*3600).max()),origins=len(origins),rows=len(ii))
            assert row['last_maturity']<source
            audit.append(row);models.append((int(source),kind,model))
            if callback:callback(row)
    return scores,audit,models
