"""Low-capacity adaptation of DeePM's features, net utility and SoftMin.

This is not the paper's LSTM/attention/graph architecture. Six shared input
coefficients and one bias replace it because the ten-asset history is short.
Every portfolio label starts at the next close, after the decision.
"""
import math
import numpy as np
import pandas as pd
import torch
from torch import nn

HORIZONS = (24, 168, 504)

def inputs(m):
    p = pd.DataFrame(m.close)
    r = np.log(p).diff().fillna(0.)
    # Conservative max of hourly and daily-return variance prevents a run of
    # flat hours from making a macro market's position arbitrarily large.
    hourly = r.ewm(span=504,adjust=False,min_periods=168).std().to_numpy()
    daily = np.log(p).diff(24).ewm(span=504,adjust=False,min_periods=168).std().to_numpy()/np.sqrt(24)
    sigma = np.maximum(np.maximum(hourly,daily),.0005)
    fs=[]
    for h in HORIZONS:
        f=p.pct_change(h).to_numpy()/(sigma*np.sqrt(h))
        fs.append(np.clip(np.nan_to_num(f,nan=0),-4,4))
    for short,long in ((48,144),(96,288),(192,576)):
        fast=p.ewm(span=short,adjust=False).mean().to_numpy()
        slow=p.ewm(span=long,adjust=False).mean().to_numpy()
        f=(fast-slow)/m.close/(sigma*np.sqrt(long-short))
        fs.append(np.clip(np.nan_to_num(f,nan=0),-4,4))
    x=np.stack(fs,axis=-1)
    valid=np.arange(len(p))>=max(HORIZONS)
    x[~valid]=0
    # Fixed risk budget, not fitted to any evaluation period.
    sleeve=np.nan_to_num(np.minimum(.0009/np.maximum(sigma*np.sqrt(24*365),.05),.015),nan=0)
    sleeve[~valid]=0
    return x,sleeve

def targets(m,signal,sleeve):
    raw=signal*sleeve
    # Keep signals between six-hour boundaries. Current zero-volume markets
    # cannot increase through the audited controller; no future volume filter.
    held=np.zeros(10); result=np.zeros_like(raw)
    for i,t in enumerate(m.times):
        if t%21600==0: held=raw[i].copy()
        result[i]=held
    return result

class SmallPolicy(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(seed)
        self.coefficients=nn.Parameter(torch.full((6,),1/6)+.025*torch.randn(6))
        self.bias=nn.Parameter(torch.zeros(()))
    def forward(self,x):return torch.tanh(x@self.coefficients+self.bias)

def labels(m,sleeve,first,last):
    ids=np.flatnonzero((m.times%21600==0)&(np.arange(len(m.times))>=first)&(np.arange(len(m.times))+7<=last))
    entry=ids+1; end=ids+7
    future=m.close[end]/m.close[entry]-1
    # Weights sealed with decision price, hence adjust next-close entry value.
    entry_ratio=m.close[entry]/m.close[ids]
    future*=entry_ratio
    funding=np.array([(m.funding[i+2:i+8]*m.close[i+2:i+8]/m.close[i]).sum(0) for i in ids])
    # Funding before entry at i+1 belongs to preceding holdings, not this target.
    return ids,future,funding,entry_ratio

def train(m,x,sleeve,first,last,seed,robust=True,steps=400):
    ids,ret,fund,entry_ratio=labels(m,sleeve,first,last)
    as_tensor=lambda z:torch.as_tensor(z,dtype=torch.float64)
    xx,ss,rr,ff,ee=map(as_tensor,(x[ids],sleeve[ids],ret,fund,entry_ratio))
    model=SmallPolicy(seed).double();opt=torch.optim.Adam(model.parameters(),lr=.01)
    initial=model.coefficients.detach().clone();history=[]
    for step in range(steps):
        w=model(xx)*ss
        # Approximate target-to-target turnover including price drift. This
        # differentiable training proxy is never reported as broker performance.
        drift=torch.cat((torch.zeros_like(w[:1]),w[:-1]*(ee[:-1]+rr[:-1])),0)
        cost=(w*ee-drift).abs().sum(-1)*.0019
        net=(w*(rr-ff)).sum(-1)-cost
        # Penalize terminal exit in training, although reporting marks inventory.
        net=net.clone();net[-1]-=w[-1].abs().sum()*.0019
        blocks=torch.tensor_split(net,4)
        sharpe=lambda v:v.mean()/torch.sqrt(v.var(unbiased=False)+1e-10)
        pooled=sharpe(net);each=torch.stack([sharpe(v) for v in blocks])
        soft=-.2*(torch.logsumexp(-each/.2,0)-math.log(len(blocks)))
        reg=.03*((model.coefficients-initial)**2).sum()+.03*model.bias**2
        loss=-pooled-(.1*soft if robust else 0)+reg
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%50==0 or step==steps-1:
            history.append(dict(step=step+1,loss=float(loss.detach()),pooled=float(pooled.detach()),softmin=float(soft.detach())))
    with torch.no_grad(): signal=model(as_tensor(x)).numpy()
    state=dict(seed=seed,robust=robust,steps=steps,coefficients=model.coefficients.detach().tolist(),bias=float(model.bias.detach()),
        input_names=['return_24h','return_168h','return_504h','macd_48_144h','macd_96_288h','macd_192_576h'],
        first_decision=int(m.times[ids[0]]),last_decision=int(m.times[ids[-1]]),last_label_end=int(m.times[ids[-1]+7]),
        train_labels=len(ids),history=history,model='7-parameter shared policy; inspired by DeePM objective, not full DeePM reproduction')
    return targets(m,signal,sleeve),state
