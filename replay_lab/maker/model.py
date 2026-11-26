"""Past-only weekly calibration. OHLC penetration is a fill proxy, never L2 data."""
from dataclasses import dataclass, asdict
import numpy as np
from ..model import SYMBOLS
from ..liquidity.market_making import Parameters, solve

@dataclass(frozen=True)
class Policy:
    horizon: int = 6
    risk: float = 1.
    width: float = 1.
    @property
    def key(self):return f'h{self.horizon}-r{self.risk:g}-w{self.width:g}'.replace('.','p')

def arrays(data):
    a={k:np.array([[getattr(c,k) for c in data[s]] for s in SYMBOLS]).T for k in ('open','high','low','close','volume','trades')}
    a['times']=np.array([c.time for c in data['BTC']]);a['known']=a['times']+3600
    r=np.diff(np.log(a['close']),axis=0,prepend=np.log(a['close'][:1]));a['returns']=r
    a['sigma']=np.zeros_like(r)
    for i in range(168,len(r)):a['sigma'][i]=np.maximum(r[i-168+1:i+1].std(0,ddof=1),1e-6)
    return a

def calibrate(a,rates,observed,cutoff):
    ids=np.flatnonzero((a['known']<cutoff)&(a['known']>=cutoff-28*86400))
    if len(ids)<168:return None
    states=[]
    distances=np.array([.001,.002,.004,.008,.016,.032])
    for j,s in enumerate(SYMBOLS):
        valid=ids[a['trades'][ids,j]>0];o=a['open'][valid,j]
        # A crossing probability of historical bars, not a measured queue-fill intensity.
        penetration=np.concatenate((1-a['low'][valid,j]/o,a['high'][valid,j]/o-1))
        hit=np.array([(np.sum(penetration>d)+.5)/(len(penetration)+1) for d in distances])
        slope,intercept=np.polyfit(distances,np.log(hit),1)
        k=float(np.clip(-slope,20,4000));arrival=float(np.clip(np.exp(intercept),.01,1.))
        fids=ids[observed[ids,j]]
        funding_sigma=float(np.std(rates[fids,j])) if len(fids)>24 else 0.
        states.append(dict(symbol=s,k=k,arrival=arrival,hit_probabilities=hit.tolist(),distances=distances.tolist(),
                           sigma=float(max(a['returns'][ids,j].std(ddof=1),1e-6)),funding_observations=len(fids),
                           funding_sigma=funding_sigma,calibration_rows=len(valid)))
    return dict(cutoff=int(cutoff),latest_input=int(a['known'][ids[-1]]),assets=states)

def fit(a,rates,observed,p,begin):
    n=len(a['times']);bid=np.full((n,10,5),np.nan);ask=bid.copy();funding_state=np.zeros((n,10));checkpoints=[]
    for cutoff in range(begin,int(a['known'][-1])+1,168*3600):
        state=calibrate(a,rates,observed,cutoff)
        if state is None:continue
        ids=np.flatnonzero((a['known']>=cutoff)&(a['known']<cutoff+168*3600))
        tables=[]
        for j,asset in enumerate(state['assets']):
            penalty=p.risk*asset['sigma']*.1
            params=Parameters(inventory_bound=2,horizon_hours=p.horizon,arrival_per_hour=asset['arrival'],
                              k=asset['k']/p.width,maker_fee=.0003,terminal_penalty=penalty,
                              running_penalty=penalty/p.horizon,funding_kappa=0.,funding_sigma=asset['funding_sigma'],
                              funding_bound=max(.001,asset['funding_sigma']*4),funding_nodes=9)
            table=solve(params)
            for i in ids:
                past=np.flatnonzero(observed[max(0,i-24):i+1,j])+max(0,i-24)
                f=float(rates[past[-1],j]) if len(past) else 0.
                funding_state[i,j]=f
                for q in range(5):
                    bid[i,j,q]=np.interp(f,table['funding'],table['bid_distance'][q])
                    ask[i,j,q]=np.interp(f,table['funding'],table['ask_distance'][q])
            tables.append(dict(parameters=asdict(params),bid=np.where(np.isfinite(table['bid_distance']),table['bid_distance'],0).tolist(),
                               ask=np.where(np.isfinite(table['ask_distance']),table['ask_distance'],0).tolist(),funding_grid=table['funding'].tolist(),cfl=table['cfl']))
        state.update(tables=tables,policy=asdict(p));checkpoints.append(state)
    return dict(bid=bid,ask=ask,funding_state=funding_state),checkpoints
