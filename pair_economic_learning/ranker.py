"""Learn relative economic opportunity order, without treating scores as returns."""
import numpy as np
from auto_pair_selection.selection import IDS


def prepare(d,f,net,both):
    shape=f['ar'].shape;cash=np.full(shape,np.nan);gross=np.zeros(shape);x=np.full((*shape,19),np.nan)
    price=d['close'];vol=d['volume'];n=len(price)
    for k,(a,b) in enumerate(IDS):
        good=f['valid'][:,k]&(abs(f['ar'][:,k])>.0038)
        for i in np.flatnonzero(good):
            beta=f['beta'][i,k];side=np.sign(f['ar'][i,k]);w=side*np.array([1.,-beta])/(1+beta)
            capacity=np.minimum(6000.,np.min(.01*vol[i,[a,b]]*price[i,[a,b]]/abs(w)))
            if np.min(capacity*abs(w))<10:continue
            gross[i,k]=capacity
            x[i,k]=np.r_[f['x'][i,k],np.log1p(capacity),np.sign(f['ar'][i,k])]
            if i+72>=n or not np.isfinite(net[i,k]):continue
            # Ranking target is the costed72h two-leg markout times sourceknown
            # capacity. Nonjoint fills are given zero utility in this auxiliary
            # label; full replay, not this label, evaluates one-leg repair losses.
            # This explicit approximation cannot certify execution profitability.
            cash[i,k]=net[i,k]*capacity if both[i,k] else 0.
    return x,cash,gross


def build(d,f,net,both,start,callback=None):
    import xgboost as xgb
    x,y,gross=prepare(d,f,net,both);t=f['times'];scores=np.full_like(f['ar'],np.nan);audit=[];models=[]
    for source in np.unique(t[t>=start]//86400*86400):
        mask=(t>=source-90*86400)&(f['maturity']<source)&(t%(6*3600)==0)
        good=mask[:,None]&f['terminal']&np.isfinite(x).all(-1)&np.isfinite(y)
        good[good.sum(1)<2]=False;ii,kk=np.where(good);origins,counts=np.unique(ii,return_counts=True)
        if len(origins)<80:continue
        weights=np.exp2(-(source-f['maturity'][origins])/(30*86400))
        dm=xgb.DMatrix(x[ii,kk],label=y[ii,kk],group=counts,weight=weights,nthread=1)
        params=dict(objective='rank:pairwise',tree_method='hist',max_bin=64,max_depth=2,eta=.04,reg_lambda=5.,min_child_weight=1.,
             nthread=1,seed=20261023,subsample=1.,colsample_bytree=1.,lambdarank_pair_method='topk',lambdarank_num_pair_per_sample=8,verbosity=0)
        model=xgb.train(params,dm,num_boost_round=64);models.append((int(source),model))
        for i in np.flatnonzero((t>=source)&(t<source+86400)):
            valid=np.isfinite(x[i]).all(-1)
            if valid.any():scores[i,valid]=model.predict(xgb.DMatrix(x[i,valid],nthread=1))
        row=dict(source=int(source),last_maturity=int(f['maturity'][ii].max()),origins=len(origins),rows=len(ii));audit.append(row)
        assert row['last_maturity']<source
        if callback:callback(row)
    return scores,audit,models
