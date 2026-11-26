"""Catching Crypto Trends (2025), adapted to fixed contest assets and shorter history.

Daily closes, 5/10/20/30-day channels, 20-day sample volatility, max/min
monotonic channel midpoint stops. No supervised prediction or future fitting.
"""
from dataclasses import dataclass
import numpy as np
from ..maker.model import arrays

@dataclass(frozen=True)
class Policy:
    mode: str = 'long_only'
    buffer: float = .2
    @property
    def key(self):return self.mode+'-buffer'+str(self.buffer).replace('.','p')

def fit(a,mode,lookbacks=(5,10,20,30)):
    n=len(a['times']);signal=np.zeros((n,10));vol=np.full((n,10),np.nan);channel_positions=np.zeros((len(lookbacks),10));stops=np.zeros_like(channel_positions)
    daily=[];states=[];latest=np.zeros(10);sigma=np.full(10,np.nan)
    for i in range(n):
        if a['known'][i]%86400==0:
            daily.append(a['close'][i].copy());d=np.array(daily)
            if len(d)>=max(lookbacks)+1:
                sigma=np.maximum(np.diff(np.log(d[-21:]),axis=0).std(0,ddof=1)*np.sqrt(365),.01)
                for h,L in enumerate(lookbacks):
                    upper=d[-L:].max(0);lower=d[-L:].min(0);middle=(upper+lower)/2;current=d[-1]
                    for j in range(10):
                        old=channel_positions[h,j]
                        if old==1:
                            if current[j]<=stops[h,j]:channel_positions[h,j]=0
                            else:stops[h,j]=max(stops[h,j],middle[j])
                        elif old==-1:
                            if current[j]>=stops[h,j]:channel_positions[h,j]=0
                            else:stops[h,j]=min(stops[h,j],middle[j])
                        elif upper[j]>lower[j]:
                            if current[j]>=upper[j]:channel_positions[h,j]=1;stops[h,j]=middle[j]
                            elif mode=='long_short' and current[j]<=lower[j]:channel_positions[h,j]=-1;stops[h,j]=middle[j]
                # The paper's inactivity filter: median absolute daily move >= 0.5%.
                active=(np.median(abs(d[-30:]/d[-31:-1]-1),axis=0)>=.005)&(np.arange(10)<3)
                latest=channel_positions.mean(0)*active
                states.append(dict(known=int(a['known'][i]),positions=channel_positions.tolist(),stops=stops.tolist(),signal=latest.tolist(),volatility=sigma.tolist(),daily_samples=len(d)))
        signal[i]=latest;vol[i]=sigma
    return dict(signal=signal,volatility=vol),states
