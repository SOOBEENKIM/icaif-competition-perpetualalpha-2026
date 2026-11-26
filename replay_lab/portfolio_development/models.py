"""Causal market-risk hedges, residual momentum and nonlinear exposure."""
import numpy as np
import pandas as pd

def daily_features(m):
    ids=np.flatnonzero(m.times%86400==0);p=m.close[ids,:3];r=np.diff(np.log(p),axis=0,prepend=np.log(p[:1]))
    n=len(ids);beta=np.ones((n,3));residual=np.zeros_like(r)
    for k in range(28,n):
        window=r[k-27:k+1];center=window-window.mean(0)
        raw=(center[:,0,None]*center).mean(0)/max(float((center[:,0]**2).mean()),1e-8)
        beta[k]=np.clip(.5+.5*raw,.25,2.5);beta[k,0]=1
        # Ex-post residual for day k uses beta estimated on day k-1.
        residual[k]=r[k]-beta[k-1]*r[k,0]
    sigma=pd.DataFrame(r).ewm(span=63,adjust=False,min_periods=28).std().to_numpy()
    sigma=np.maximum(np.nan_to_num(sigma,nan=.01),.0025)
    residual_sigma=pd.DataFrame(residual).rolling(28,min_periods=28).std().to_numpy()
    residual_sigma=np.maximum(np.nan_to_num(residual_sigma,nan=.01),.0025)
    mom=pd.DataFrame(np.log(p)).diff(21).to_numpy()/(sigma*np.sqrt(21));mom=np.nan_to_num(mom,nan=0)
    fast=pd.DataFrame(np.log(p)).diff(7).to_numpy()/(sigma*np.sqrt(7));fast=np.nan_to_num(fast,nan=0)
    rs=pd.DataFrame(residual).rolling(14,min_periods=14).sum().to_numpy();rs=np.nan_to_num(rs,nan=0)
    z=rs/(residual_sigma*np.sqrt(14));peak=pd.DataFrame(p).rolling(63,min_periods=28).max().to_numpy();dd=np.nan_to_num(p/peak-1,nan=0)
    src=np.searchsorted(m.times[ids],m.times,side='right')-1;src=np.maximum(src,0)
    return {key:value[src] for key,value in dict(beta=beta,sigma=sigma,momentum=mom,fast=fast,residual_momentum=z,residual_return=rs,residual_sigma=residual_sigma,drawdown=dd).items()}

def make(m,baseline):
    f=daily_features(m);result={};hedge=False;market=np.zeros_like(baseline);unit=np.zeros_like(baseline);res=np.zeros_like(baseline);state=np.zeros(2)
    for i,t in enumerate(m.times):
        if t%21600==0:
            if f['fast'][i,0]<-.5:hedge=True
            elif f['fast'][i,0]>0:hedge=False
        market[i]=baseline[i];unit[i]=baseline[i]
        if hedge:
            market[i,0]-=.5*float(np.maximum(baseline[i,:3],0)@f['beta'][i])
            unit[i,0]-=.5*float(np.maximum(baseline[i,:3],0).sum())
        if t%86400==0 and i>=1008:
            for j in (1,2):
                z=f['residual_momentum'][i,j];d=np.sign(z)
                if abs(z)<.25 or d!=np.sign(state[j-1]):state[j-1]=0
                # A two-leg round trip costs 38bp per notional per leg.
                carry=float(np.mean(m.funding[max(0,i-168):i,j]-f['beta'][i,j]*m.funding[max(0,i-168):i,0]))*168*d
                expected=abs(f['residual_return'][i,j])*.5-carry
                if abs(z)>.5 and expected>.0038*(1+f['beta'][i,j]):
                    state[j-1]=d*min(.0018/(f['residual_sigma'][i,j]*np.sqrt(365)),.015)
        res[i,1:3]=state;res[i,0]=-float(state@f['beta'][i,1:3])
    result['market_half_hedge']=market;result['unit_half_hedge']=unit
    result['residual_momentum']=res;result['residual_overlay']=baseline+res
    for name,response in [('linear_gate',np.clip(f['momentum'],0,1)),('nonlinear_gate',np.maximum(2*f['momentum']/(1+f['momentum']**2),0))]:
        w=baseline.copy();w[:,:3]*=response;result[name]=w
    guard=baseline.copy();guard[:,:3]*=np.where(f['drawdown'][:,0,None]<-.15,.5,1.);result['market_drawdown_guard']=guard
    return result,f
