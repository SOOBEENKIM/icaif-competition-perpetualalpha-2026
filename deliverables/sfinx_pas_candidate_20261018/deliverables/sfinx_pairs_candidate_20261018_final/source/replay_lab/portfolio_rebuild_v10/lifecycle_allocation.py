"""Keep original six-hour lifecycle changes; vary daily quantity adjustments."""
import numpy as np
import pandas as pd
from .common import *

def targets(raw,sigma,times,days=1,conviction=False,risk=.001,prices=None):
    if conviction:
        r=pd.DataFrame(np.log(prices)).diff()
        efficiency=(r.rolling(168).sum()/r.abs().rolling(168).sum()).fillna(0.).to_numpy()
        raw=raw*(.5+np.clip(2*efficiency,0,1.5))
    out=np.zeros_like(raw);held=np.zeros(10);prior_sign=np.zeros(10)
    for i,t in enumerate(times):
        if i<672 or t%21600:out[i]=held;continue
        current_sign=np.sign(raw[i]);change=current_sign!=prior_sign
        scale=risk/max(np.sqrt(max(0.,24*raw[i]@sigma[i]@raw[i])),1e-12)
        desired=raw[i]*scale
        desired=np.clip(desired,-.135,.135)
        desired/=max(1.,abs(desired).sum()/.54,abs(desired.sum())/.36)
        if t%86400==0:
            same=held*desired>0
            held=np.where(same,held+(desired-held)/days,desired)
        # Every original six-hour open/close/sign change takes effect now.
        held=np.where(change,desired,held)
        out[i]=held;prior_sign=current_sign
    return out

def main():
    stage=OUT/'lifecycle_allocation';stage.mkdir(parents=True,exist_ok=True)
    if (stage/'plan.json').exists():raise RuntimeError('No overwrite')
    write(stage/'plan.json',dict(created=stamp(),
      reason='Daily-sizing studies implicitly defer intraday lifecycle changes. Isolate size inertia while preserving all original6hour entry/exit/sign changes.',
      configurations=['v7_daily_size','v9_daily_size','v9_size3','v9_size7','v9_efficiency_size3'],
      hypothesis='Slower size adjustment preserves trend profits, reduces costs and drawdown without changing original lifecycle signals. Compare same timing/risk and all failures.',
      lifecycle='At every6h boundary detect raw sign/flat transition; changedassets immediately get current risk-sized target, including zero for exit. Same-sign size changes occur at midnight, using1/3/7day relaxation.',
      all_history_reused=True,live_orders_sent=0))
    m,z,b,p=protocol();sigma=np.load(OUT/'covariance.npy');raw=dict(np.load(OUT/'baselines/raw.npz'))
    specs={'v7':('v7',1,False),'v9':('v9',1,False),'v9_size3':('v9',3,False),'v9_size7':('v9',7,False),'v9_efficiency_size3':('v9',3,True)}
    models={name:targets(raw[k],sigma,m.times,days=d,conviction=c,prices=m.close) for name,(k,d,c) in specs.items()}
    np.savez_compressed(stage/'targets.npz',**models);rows=[]
    for phase,(a,b) in phases(m).items():
        for name,w in models.items():
            for scenario in ('base','stress'):
                rows.append(evaluate(m,p,sigma,w,name,a,b,'lifecycle_allocation',phase,scenario));write(stage/'running_results.json',rows)
    write(stage/'results.json',rows)

if __name__=='__main__':main()
