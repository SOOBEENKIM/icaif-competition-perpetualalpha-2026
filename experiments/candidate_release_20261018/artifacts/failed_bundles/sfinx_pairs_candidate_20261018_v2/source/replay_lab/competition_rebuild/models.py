"""Published ideas adapted to actual one-hour observations and a common ledger."""
import numpy as np
import pandas as pd
from .common import GROUPS

HORIZON=24
WINDOWS=(24,72,168)

def rolling(x,window,method):
    return getattr(pd.DataFrame(x).rolling(window,min_periods=window),method)().to_numpy()

def features(m,window):
    logp=np.log(m.close);r=np.diff(logp,axis=0,prepend=logp[:1])
    sigma=np.maximum(rolling(r,168,'std'),.0005)
    cols=[]
    for lag in (1,3,6,12,24):
        diff=np.full_like(r,np.nan);diff[lag:]=logp[lag:]-logp[:-lag]
        cols.append(diff/(sigma*np.sqrt(lag)))
    mean=rolling(r,window,'mean');vol=rolling(r,window,'std')
    cols.extend([mean*np.sqrt(window)/sigma,vol/sigma,
        (logp-rolling(logp,window,'mean'))/(sigma*np.sqrt(window)),
        (logp-rolling(logp,window,'max'))/(sigma*np.sqrt(window))])
    for lag in (6,24):
        d=np.zeros_like(r);d[lag:]=logp[lag:]-logp[:-lag]
        g=np.zeros_like(r)
        for group in GROUPS:
            for j in group:
                peers=[k for k in group if k!=j]
                g[:,j]=np.mean(d[:,peers]/np.maximum(sigma[:,peers],.0005),axis=1)/np.sqrt(lag) if peers else 0
        cols.append(g)
    logged=np.log1p(m.volume)
    cols.extend([(logged-rolling(logged,window,'mean'))/np.maximum(rolling(logged,window,'std'),.1),
        rolling((m.trades>0).astype(float),24,'mean'),
        np.vstack([np.zeros((1,10)),rolling(m.funding,24,'mean')[:-1]])*HORIZON/.0038])
    t=m.times[:,None]
    cols.extend([np.broadcast_to(np.sin(2*np.pi*t/86400),r.shape),np.broadcast_to(np.cos(2*np.pi*t/86400),r.shape)])
    return np.clip(np.stack(cols,-1),-8,8)

def labels(m,horizon=HORIZON):
    y=np.full_like(m.close,np.nan)
    y[:-horizon-1]=np.log(m.close[horizon+1:]/m.close[1:-horizon])
    return y

def fit_predict(m,x,y,meta_start,end,train_hours=1008):
    """Strict labels-through boundary and train-only feature standardization."""
    ids=np.arange(max(168,meta_start-train_hours),meta_start-HORIZON-1)
    ids=ids[m.times[ids]%21600==0]
    pred=np.zeros((len(m.times),10));states=[]
    for j in range(10):
        selected=ids[np.isfinite(x[ids,j]).all(1)&np.isfinite(y[ids,j])&(m.trades[ids,j]>0)]
        if len(selected)<50:
            states.append(dict(asset=j,count=len(selected),disabled=True));continue
        assert selected[-1]+HORIZON+1<meta_start
        xx=x[selected,j];mean=xx.mean(0);std=np.maximum(xx.std(0),.1)
        z=(xx-mean)/std;yy=y[selected,j]
        # Mean squared error plus fixed unit ridge penalty: no test tuning.
        yc=float(yy.mean());beta=np.linalg.solve(z.T@z+len(z)*np.eye(z.shape[1]),z.T@(yy-yc))
        pred[meta_start:end+1,j]=np.clip(((x[meta_start:end+1,j]-mean)/std)@beta+yc,-.1,.1)
        states.append(dict(asset=j,count=len(selected),mean=mean.tolist(),std=std.tolist(),coefficient=beta.tolist(),intercept=yc,
            first_training_time=int(m.times[selected[0]]),last_training_time=int(m.times[selected[-1]]),
            last_label_maturity=int(m.times[selected[-1]+HORIZON+1]),strict_cutoff=int(m.times[meta_start])))
    return pred,states

def target_path(m,pred,start,end,entry=.0038):
    """Signals set model targets; the separate shared broker determines fills.

    A 24h model holding rule is not a claim of an actual minimum fill holding
    period: risk reductions and partial fills belong to the common controller.
    """
    ret=np.diff(np.log(m.close),axis=0,prepend=np.log(m.close[:1]))
    daily_sigma=np.maximum(rolling(ret,168,'std')*np.sqrt(24),.003)
    funding=np.vstack([np.zeros((1,10)),rolling(m.funding,24,'mean')[:-1]])*HORIZON
    weights=np.zeros_like(m.close);state=np.zeros(10);entry_index=np.full(10,-10000)
    for i in range(start,end+1):
        if m.times[i]%21600==0:
            score=pred[i]-funding[i]
            fresh=(m.volume[i]>0)&(m.trades[i]>0)&np.isfinite(score)
            for j in range(10):
                if state[j] and i-entry_index[j]>=HORIZON:
                    if np.sign(state[j])*score[j]<entry/2:state[j]=0
                if state[j]==0 and fresh[j] and abs(score[j])>entry:
                    state[j]=np.sign(score[j])*min(.015,.00012/daily_sigma[i,j]);entry_index[j]=i
        weights[i]=state
    return weights

def stabilized_softmax(values):
    q=np.asarray(values,dtype=float);q=(q-q.mean())/max(float(q.std()),1e-8)
    w=np.exp(q-q.max());w/=w.sum();w=.75*w+.25/len(w)
    w=np.clip(w,.01,.55);return w/w.sum()

def conflict_gate(m,base,pred,start=168):
    """Reduce a conflicting position by half; no hindsight risk inflation."""
    funding=np.vstack([np.zeros((1,10)),rolling(m.funding,24,'mean')[:-1]])*HORIZON
    valid=np.isfinite(pred)&np.isfinite(funding)
    conflict=valid&(np.sign(base)*(pred-funding)<-.0038)
    out=base.copy();out[start:]*=np.where(conflict[start:],.5,1.)
    return out
