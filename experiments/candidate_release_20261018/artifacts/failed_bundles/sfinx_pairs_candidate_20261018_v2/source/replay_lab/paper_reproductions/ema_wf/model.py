"""Causal port of the author's EMA selection; no wallet/network imports."""
from dataclasses import dataclass
import numpy as np

PERIODS=(5,7,10,15,20,30,40,50,100,150,200)
PAIRS=tuple((f,s) for s in PERIODS if s>35 for f in PERIODS if f<=35)
NAMES=tuple(f'{f}_{s}' for f,s in PAIRS)
WARMUP=max(PERIODS)-1


@dataclass(frozen=True)
class Policy:
    train_days:int=7
    deploy_days:int=28
    scale:float=1.
    buffer:float=.2
    decision_hours:int=1

    @property
    def key(self):return f'ema-wf-{self.train_days}-{self.deploy_days}-s{self.scale:.8g}'.replace('.','p')


def ema(x,period):
    """TTR::EMA uses an initial period-length SMA, then recursive EMA."""
    x=np.asarray(x,float);y=np.full(len(x),np.nan)
    if len(x)<period:return y
    y[period-1]=np.mean(x[:period]);alpha=2/(period+1)
    for i in range(period,len(x)):y[i]=alpha*x[i]+(1-alpha)*y[i-1]
    return y


def experts(close,fee=.001):
    x=np.asarray(close,float)
    if np.any(~np.isfinite(x)) or np.any(x<=0):raise ValueError('Finite positive closes required')
    e={k:ema(x,k) for k in PERIODS}
    direction=np.column_stack([np.where(e[f][WARMUP:]>e[s][WARMUP:],1.,-1.) for f,s in PAIRS])
    position=np.vstack([np.full((1,len(PAIRS)),np.nan),direction[:-1]])
    r=np.r_[0.,np.diff(np.log(x[WARMUP:]))]
    turnover=np.vstack([np.zeros((1,len(PAIRS))),np.abs(np.diff(position,axis=0))])
    turnover=np.nan_to_num(turnover)
    pnl=np.nan_to_num(position*r[:,None])-fee*turnover
    return dict(direction=direction,position=position,pnl=pnl,return_=r)


def select(pnl):
    sd=np.std(pnl,axis=0,ddof=1)
    score=np.divide(np.mean(pnl,axis=0),sd,out=np.full(len(sd),-np.inf),where=sd>0)
    return int(np.argmax(score))


def paper_walk(close,train_days,deploy_days):
    """Literal published pipeline, including its stitched-cost convention."""
    e=experts(close);n=len(e['pnl']);tr=train_days*24;te=deploy_days*24
    result=[];selection=[]
    for start in range(0,n-tr-te+1,te):
        begin=start+tr;end=begin+te;k=select(e['pnl'][start:begin])
        selection.append(dict(start=start,begin=begin,end=end,expert=k))
        for i in range(begin,end):result.append((i+WARMUP,e['pnl'][i,k],e['position'][i,k],k))
    return np.asarray(result),selection


def fit(a,p,volatility,assets=(0,1)):
    """Refit at published cadence; only returns known at cutoff may select.

    A signal known at close[i] targets the next bar's execution, not close[i].
    Preserve the entire last deployment interval even when local data end;
    unlike the offline paper table, live decisions cannot know dataset length.
    """
    n=len(a['times']);signal=np.zeros((n,10));states=[]
    tr=p.train_days*24;te=p.deploy_days*24
    for j in assets:
        e=experts(a['close'][:,j]);length=len(e['pnl'])
        for start in range(0,max(0,length-tr),te):
            begin=start+tr;end=min(begin+te,length+1)
            k=select(e['pnl'][start:begin])
            # Return at begin is next-hour return; its signal uses begin-1.
            lo=begin-1+WARMUP;hi=min(end-1+WARMUP,n)
            signal[lo:hi,j]=e['direction'][begin-1:end-1,k]
            states.append(dict(asset=j,cutoff=int(a['known'][lo]),
                training_start=int(a['known'][start+WARMUP]),training_last=int(a['known'][begin-1+WARMUP]),
                next_refit=int(a['known'][lo])+te*3600,fast=PAIRS[k][0],slow=PAIRS[k][1],
                training_sharpe=float(np.mean(e['pnl'][start:begin,k])/np.std(e['pnl'][start:begin,k],ddof=1)*np.sqrt(8760))))
    sigma=np.array(volatility,copy=True)
    budget=np.where(np.isfinite(sigma),np.minimum(.25/np.maximum(sigma,.01),2.)*.01,0.)
    weights=signal*budget*p.scale
    return dict(signal=signal,target_weight=weights,volatility=sigma,decision=a['known'].copy()),states
