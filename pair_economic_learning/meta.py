"""Entry selection calibrated on historical fills rather than all forecasts."""
import numpy as np
from .model import Spec
from auto_pair_selection.selection import IDS
from experiments.quote_anchored_pairs_20261017.pricing import reservation_limits


def outcome(q,source,arrival,terminal,mu,known_carry,funding_cash):
    """Same source cash budget as the replay, normalized gross1; no lot/queue claim."""
    sides=np.sign(q);preview=source*(1+sides*.001)
    edge=float(q@(source*np.array([1+mu,1-mu])-source))
    budget=reservation_limits(q,source,preview,edge,known_carry,9.,10.,5.,10000.)
    if budget is None:return np.nan,False
    entry=arrival*(1+sides*.001);exit_=terminal*(1-sides*.001)
    filled=bool(np.all(sides*(entry-np.array(budget['limits']))<=1e-12))
    net=float(q@(exit_-entry)-abs(q)@(entry+exit_)*.0009+funding_cash)
    return net,filled


def labels(d,f):
    t=f['times'];p=d['close'];y=np.full_like(f['ar'],np.nan);both=np.zeros_like(f['valid'])
    cp=np.vstack([np.zeros(10),np.cumsum(p*d['funding'],axis=0)])
    for k,(a,b) in enumerate(IDS):
        for i in np.flatnonzero(f['valid'][:,k]&(abs(f['ar'][:,k])>.0038)):
            if i+72>=len(t):continue
            ids=[a,b];bt=f['beta'][i,k];side=np.sign(f['ar'][i,k]);source=p[i,ids]
            q=side*np.array([1.,-bt])/(1+bt)/source
            if not d['observed'][i+2:i+73][:,ids].all():continue
            cash=-q*source*d['funding'][i,ids]*72;carry=float(np.where(cash>0,cash*.5,cash).sum())
            # Settlement at arrival precedes order execution, so accrue only
            # i+2,...,i+72; settlement at the close precedes the closing fill.
            paid=float(-q@(cp[i+73,ids]-cp[i+2,ids]))
            y[i,k],both[i,k]=outcome(q,source,p[i+1,ids],p[i+72,ids],f['ar'][i,k],carry,paid)
            both[i,k] &= bool((d['volume'][i+1,ids]>0).all() and (d['trades'][i+1,ids]>0).all())
    return y,both


def fit(f,net,both,source,spec=Spec()):
    import xgboost as xgb
    t=f['times'];valid=(t>=source-90*86400)&(f['maturity']<source)&(t%(6*3600)==0)
    mask=valid[:,None]&f['valid']&f['terminal']&both&np.isfinite(net)&(abs(f['ar'])>.0038)
    ii,kk=np.where(mask)
    if len(np.unique(ii))<spec.min_origins:return None
    w=np.exp2(-(source-f['maturity'][ii])/(30*86400))/np.bincount(ii,minlength=len(t))[ii]
    x=np.c_[f['x'][ii,kk],np.sign(f['ar'][ii,kk])];unit=f['unit'][ii,kk]
    common=dict(max_depth=2,eta=.04,reg_lambda=5.,tree_method='hist',max_bin=64,nthread=1,seed=20261023,verbosity=0,subsample=1.,colsample_bytree=1.)
    result={}
    for kind,y in [('classifier',(net[ii,kk]>0).astype(float)),('value',np.clip(net[ii,kk]/unit,-4,4))]:
        params=dict(common,objective='binary:logistic' if kind=='classifier' else 'reg:squarederror',base_score=.5 if kind=='classifier' else 0.,min_child_weight=2. if kind=='classifier' else 8.)
        result[kind]=xgb.train(params,xgb.DMatrix(x,label=y,weight=w,nthread=1),num_boost_round=64)
    return result,dict(source=int(source),last_maturity=int(f['maturity'][ii].max()),origins=len(np.unique(ii)),rows=len(ii),weight=float(w.sum()),historical_positive_fraction=float(w@(net[ii,kk]>0)/w.sum()))


def build(d,f,start,callback=None):
    import xgboost as xgb
    net,both=labels(d,f);t=f['times'];allow={k:np.zeros_like(f['valid']) for k in ['cost_meta_72','cost_value_72']};pred={k:np.full_like(f['ar'],np.nan) for k in allow};audit=[];models=[]
    for source in np.unique(t[t>=start]//86400*86400):
        fitted=fit(f,net,both,int(source))
        if fitted is None:continue
        mm,record=fitted;audit.append(record);models.append((int(source),mm));ii=np.flatnonzero((t>=source)&(t<source+86400))
        for i in ii:
            g=f['valid'][i]&np.isfinite(f['ar'][i])
            if not g.any():continue
            dm=xgb.DMatrix(np.c_[f['x'][i,g],np.sign(f['ar'][i,g])],nthread=1)
            for arm,kind,threshold in [('cost_meta_72','classifier',.5),('cost_value_72','value',0.)]:
                pr=mm[kind].predict(dm);pred[arm][i,g]=pr;allow[arm][i,g]=pr>threshold
        if callback:callback(record)
    states={}
    for arm,permission in allow.items():
        mu=f['ar'];valid=f['valid']&np.isfinite(mu)
        states[arm]=dict(times=t,dense=True,pairs=np.array(IDS),signal_times=np.broadcast_to(t[:,None],mu.shape),source_prices=d['close'],forecast_times=t,
            beta=f['beta'],vol=f['vol'],mu=mu,directions=valid.astype(int),economic_available=valid,
            economic_action=np.where(valid&permission&(abs(mu)>.0038),np.sign(np.nan_to_num(mu)),0).astype(int))
    return states,audit,models,pred,net,both
