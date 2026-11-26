"""Frozen return calibration and cost-aware increases; no API or order sender."""
import numpy as np
import pandas as pd
from replay_lab.ema_rebuild.engine import Controller
from replay_lab.common_comparison.engine import project,rounded,risk_breaches

def features(m,forecast):
    p=pd.DataFrame(np.log(m.close));r=p.diff()
    vol=r.ewm(span=168,adjust=False,min_periods=168).std().clip(lower=.0005).to_numpy()*np.sqrt(24)
    mean=p.shift(1).rolling(168).mean().to_numpy()
    x=np.stack([forecast,p.diff(6).to_numpy(),p.diff(24).to_numpy(),p.diff(168).to_numpy(),p.to_numpy()-mean],axis=2)/vol[:,:,None]
    return x,vol

def fit(m,x,vol,cutoff,alpha=1.):
    rows=[]
    for i,t in enumerate(m.times):
        if i<1440 or t%21600 or i+25>=len(m.times) or m.times[i+25]>=cutoff:continue
        for j in range(3):
            if not np.isfinite(x[i,j]).all():continue
            if min(m.volume[i+1,j],m.volume[i+25,j],m.trades[i+1,j],m.trades[i+25,j])<=0:continue
            rows.append((i,j,np.log(m.close[i+25,j]/m.close[i+1,j])/vol[i,j]))
    if len(rows)<100:raise ValueError('Insufficient calibration data')
    means=[];scales=[];ys=[];zs=[]
    for j in range(3):
        subset=[r for r in rows if r[1]==j]
        v=np.array([x[i,k] for i,k,y in subset]);y=np.array([y for i,k,y in subset])
        mu=v.mean(0);sd=np.maximum(v.std(0),1e-8)
        means.append(mu);scales.append(sd);zs.append((v-mu)/sd);ys.append(y)
    z=np.column_stack([np.ones(len(rows)),np.concatenate(zs)]);y=np.concatenate(ys)
    penalty=np.eye(z.shape[1])*alpha;penalty[0,0]=0
    beta=np.linalg.solve(z.T@z/len(z)+penalty,z.T@y/len(z))
    return {'means':np.array(means).tolist(),'scales':np.array(scales).tolist(),'beta':beta.tolist(),
       'rows':len(rows),'fit_cutoff':int(cutoff),'last_label_time':int(max(m.times[i+25] for i,j,y in rows)),
       'last_input_time':int(max(m.times[i] for i,j,y in rows)),'alpha':alpha}

def predict(state,x,vol):
    z=(x[:,:3]-np.array(state['means']))/np.array(state['scales'])
    beta=np.array(state['beta']);y=(beta[0]+np.nan_to_num(z,nan=0,posinf=0,neginf=0)@beta[1:])*vol[:,:3]
    out=np.zeros(vol.shape);out[:,:3]=np.nan_to_num(y,nan=0,posinf=0,neginf=0)
    return out

def economic_target(raw,current,gain,funding,round_trip_cost):
    raw=np.asarray(raw,float);current=np.asarray(current,float)
    if any(not np.isfinite(a).all() for a in (raw,current,gain,funding)) or not np.isfinite(round_trip_cost) or round_trip_cost<0:
        raise ValueError('Invalid economic input')
    # All incumbent reductions remain allowed. Reversals first close the old leg.
    side=np.sign(raw);same=raw*current>0
    remaining=side*gain-round_trip_cost-np.maximum(side*funding,0)*24
    increases=(abs(raw)>abs(current))|(~same&(raw!=0))
    reject=increases&(remaining<=0)
    kept=np.where(same,np.sign(current)*np.minimum(abs(current),abs(raw)),0.)
    return np.where(reject,kept,raw),reject,remaining

class CalibratedController:
    def __init__(self,times,gain,funding,base_round_trip_cost):
        self.base=Controller();self.index={int(t):i for i,t in enumerate(times)}
        self.gain=gain;self.funding=funding;self.cost=base_round_trip_cost
    def propose(self,time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs):
        i=self.index[time];current=qty*prices/equity
        desired,blocked,edge=economic_target(raw,current,self.gain[i],self.funding[i],self.cost)
        action=self.base.propose(time,prices,qty,equity,desired,sigma,volume,trades,protocol,**kwargs)
        if action is not None:
            action.update(calibrated_return=self.gain[i].tolist(),entry_net_edge=edge.tolist(),entry_blocked=blocked.tolist())
        return action
