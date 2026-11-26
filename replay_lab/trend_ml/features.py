"""Causal trend, regime and cross-asset inputs; next-open aligned mature labels."""
import numpy as np

LOOKBACKS=(24,72,168,336)
NAMES=(['return_z_'+str(h) for h in (6,24,72,168,336)] +
       ['efficiency_'+str(h) for h in (24,72,168)] +
       ['channel_'+str(h) for h in LOOKBACKS] +
       ['vol_ratio','downside_ratio','range_z','volume_ratio','btc_24_z','market_72_z',
        'residual_24_z','asset_btc','asset_eth','asset_hype'])


def build(a):
    n=len(a['times'])
    close=a['close'][:,:3]
    ret=np.diff(np.log(close),axis=0,prepend=np.log(close[:1]))
    x=np.full((n,3,len(NAMES)),np.nan)
    sigma=np.full((n,3),np.nan)
    high=np.full((n,3,4),np.nan)
    low=high.copy()
    efficiency=np.zeros((n,3))
    for i in range(336,n):
        vol=np.maximum(ret[i-167:i+1].std(0,ddof=1),.0001)
        sigma[i]=vol
        values=[]
        for h in (6,24,72,168,336):
            values.append(np.log(close[i]/close[i-h])/(vol*np.sqrt(h)))
        for h in (24,72,168):
            e=abs(np.log(close[i]/close[i-h]))/np.maximum(abs(ret[i-h+1:i+1]).sum(0),1e-9)
            values.append(e)
            if h==72:efficiency[i]=e
        for k,h in enumerate(LOOKBACKS):
            hi=close[i-h+1:i+1].max(0)
            lo=close[i-h+1:i+1].min(0)
            high[i,:,k],low[i,:,k]=hi,lo
            values.append((close[i]-lo)/np.maximum(hi-lo,1e-8)*2-1)
        values.extend([ret[i-23:i+1].std(0,ddof=1)/vol,
                       np.sqrt(np.mean(np.minimum(ret[i-71:i+1],0)**2,0))/vol,
                       (a['high'][i,:3]-a['low'][i,:3])/close[i]/vol,
                       np.log((a['volume'][i,:3]+1)/(a['volume'][i-167:i+1,:3].mean(0)+1)),
                       np.full(3,np.log(close[i,0]/close[i-24,0])/(vol[0]*np.sqrt(24))),
                       np.full(3,np.mean(np.log(close[i]/close[i-72])/(vol*np.sqrt(72)))),
                       (np.log(close[i]/close[i-24])-np.log(close[i,0]/close[i-24,0]))/(vol*np.sqrt(24))])
        values.extend(np.eye(3))
        x[i]=np.clip(np.array(values).T,-10,10)
    return dict(x=x,sigma=sigma,upper=high,lower=low,efficiency=efficiency)


def labels(a,features,horizon):
    n=len(a['times'])
    y=np.full((n,3),np.nan)
    risk=features['sigma']*np.sqrt(horizon)
    # Prediction after candle i closes -> buy at next open -> exit h hours later.
    for i in range(336,n-horizon-1):
        y[i]=(a['open'][i+horizon+1,:3]/a['open'][i+1,:3]-1)/risk[i]
    # label endpoint is an open, only used once that timestamp is strictly past.
    mature=a['known']+horizon*3600
    return y,risk,mature
