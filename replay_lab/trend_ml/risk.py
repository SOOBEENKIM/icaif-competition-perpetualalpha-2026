"""Causal learned risk forecasts and buffered Donchian positions.

The trained model predicts next-day realized variance, not tomorrow's direction.
Past weekly checkpoints are immutable; size forecasts cannot increase the original
20-day volatility budget. This isolates risk timing from extra leverage.
"""
from dataclasses import dataclass
import numpy as np
import xgboost as xgb
from .features import build, NAMES
from ..channel_crypto.model import fit as channel_fit
from ..channel_revision.diagnosis import write

RISK_NAMES=NAMES+['log_vol24','log_vol168','log_vol480','log_downside24']
PARAMS=dict(objective='reg:squarederror',tree_method='hist',max_depth=2,eta=.035,
            min_child_weight=40,reg_lambda=30,nthread=2,seed=20261007,verbosity=0)


@dataclass(frozen=True)
class RiskPolicy:
    lookbacks:tuple=(5,10,20,30)
    risk_model:str='original'
    buffer:float=.2
    @property
    def key(self):return 'channel'+'-'.join(map(str,self.lookbacks))+'-'+self.risk_model


def risk_inputs(a):
    f=build(a);n=len(a['times']);close=a['close'][:,:3]
    r=np.diff(np.log(close),axis=0,prepend=np.log(close[:1]))
    variance=np.full((n,3,4),np.nan)
    target=np.full((n,3),np.nan)
    for i in range(480,n):
        for k,h in enumerate((24,168,480)):
            variance[i,:,k]=np.maximum(np.mean(r[i-h+1:i+1]**2,axis=0)*8760,1e-4)
        variance[i,:,3]=np.maximum(np.mean(np.minimum(r[i-23:i+1],0)**2,axis=0)*8760,1e-4)
        # All 24 future returns are complete at known[i]+24h.
        if i+24<n:target[i]=np.maximum(np.mean(r[i+1:i+25]**2,axis=0)*8760,1e-4)
    x=np.concatenate([f['x'],.5*np.log(variance)],axis=2)
    return dict(x=x,target=target,maturity=a['known']+24*3600,
                trailing=np.sqrt(variance[:,:,:3]),ratio=np.log(target/variance[:,:,1]))


def train(a,f,cutoff):
    ids=np.flatnonzero((a['known']>=cutoff-90*86400)&(f['maturity']<cutoff)&
                      (a['known']%(6*3600)==0)&np.isfinite(f['ratio']).all(1)&np.isfinite(f['x']).all((1,2)))
    if len(ids)<72:return None
    x=f['x'][ids].reshape(-1,len(RISK_NAMES));y=np.clip(f['ratio'][ids].ravel(),-5,5)
    dm=xgb.DMatrix(x,label=y,feature_names=RISK_NAMES,nthread=2)
    b=xgb.train(PARAMS,dm,num_boost_round=100)
    # Squared-log target needs a training-only retransformation correction.
    residual=y-b.predict(dm)
    correction=float(np.clip(np.mean(np.exp(residual)),.5,3))
    return dict(cutoff=int(cutoff),last_maturity=int(f['maturity'][ids[-1]]),samples=len(x),
                smearing=correction,booster=bytes(b.save_raw(raw_format='json')).decode())


def predict(state,x,trailing):
    b=xgb.Booster({'nthread':2});b.load_model(bytearray(state['booster'].encode()))
    y=b.predict(xgb.DMatrix(x.reshape(-1,len(RISK_NAMES)),feature_names=RISK_NAMES,nthread=2)).reshape(x.shape[:2])
    return trailing[:,:,1]*np.sqrt(np.exp(np.clip(y,-5,5))*state['smearing'])


def walk(a,f,begin,dest):
    out=np.full((len(a['times']),3),np.nan);states=[]
    for cutoff in range(begin,int(a['known'][-1])+1,7*86400):
        state=train(a,f,cutoff)
        if state is None:continue
        assert state['last_maturity']<cutoff
        write(dest/f'risk-{cutoff}.json',state)
        ids=np.flatnonzero((a['known']>=cutoff)&(a['known']<cutoff+7*86400))
        out[ids]=predict(state,f['x'][ids],f['trailing'][ids])
        states.append({k:v for k,v in state.items() if k!='booster'})
    return out,states


def signals(a,p,f,pred):
    m,states=channel_fit(a,'long_only',tuple(p.lookbacks))
    if p.risk_model=='original':return m,states
    original=m['volatility'][:,:3]
    if p.risk_model=='recent':forecast=f['trailing'][:,:,0]
    elif p.risk_model=='learned':forecast=pred
    else:raise ValueError('Unknown risk model')
    # Half shrinkage to the original daily-vol estimate; never increase sizing.
    m['volatility'][:,:3]=np.where(np.isfinite(forecast),np.maximum(original,np.sqrt(original*forecast)),original)
    return m,states


def quality(a,f,pred,start,end):
    mask=(a['known']>=start)&(f['maturity']<end)&np.isfinite(f['target']).all(1)&np.isfinite(pred).all(1)
    y=f['target'][mask];scores={}
    for name,p in [('learned',pred[mask]**2),('recent24',f['trailing'][mask,:,0]**2),('recent168',f['trailing'][mask,:,1]**2)]:
        ratio=y/np.maximum(p,1e-8)
        scores[name]=dict(qlike=float(np.mean(ratio-np.log(ratio)-1)),log_mae=float(np.mean(abs(np.log(y)-np.log(p)))))
    return dict(samples=int(mask.sum()*3),scores=scores)
