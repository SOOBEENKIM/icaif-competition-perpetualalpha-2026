"""12:00 UTC metrics from full hourly equity paths, never midnight proxies."""
import numpy as np
from experiments.basis_carry_20261017.score import score


def official_metrics(run):
    snapshots=run['snapshots']
    if snapshots[0]['time']%86400!=43200 or snapshots[-1]['time']%86400!=43200:
        raise ValueError('Comparison endpoints must be official daily boundaries')
    daily=[s for s in snapshots if s['time']%86400==43200]
    times=np.array([s['time'] for s in daily]);values=np.array([s['equity'] for s in daily])
    returns=values[1:]/values[:-1]-1
    spans=np.diff(times)/86400
    if not np.isfinite(values).all() or np.any(values<=0):raise ValueError('Invalid equity')
    R=values[-1]/values[0]-1
    td=len({(f['time']-43200)//86400 for f in run['fills']})
    out=score(float(R),returns,spans,trading_days=td,start_sample=True)
    out.update(daily=[dict(time=int(t),return_=float(r),days_covered=float(span)) for t,r,span in zip(times[1:],returns,spans)],
        trading_days=td,valuation='Reference hourly close at 12:00 UTC; not actual par or official submitted score')
    assert abs(np.prod(1+returns)-1-R)<1e-10
    return out


def paired_uncertainty(candidate,baseline,seed=20261017,draws=5000):
    x=np.array([r['return_'] for r in candidate['daily']]);y=np.array([r['return_'] for r in baseline['daily']]);n=len(x)
    if n!=len(y) or n<6:return None
    if [d['time'] for d in candidate['daily']]!=[d['time'] for d in baseline['daily']]:raise ValueError('Unpaired days')
    rng=np.random.default_rng(seed);starts=rng.integers(n,size=(draws,(n+2)//3))
    idx=np.stack([(starts+j)%n for j in range(3)],axis=-1).reshape(draws,-1)[:,:n]
    def q(a):
        R=np.prod(1+a,axis=1)-1;s=np.std(a,axis=1,ddof=1)*np.sqrt(n)
        return (np.maximum(R,0)+.005)/(abs(R)+1.5*s+.05)
    delta=q(x[idx])-q(y[idx])
    return dict(method='Paired circular 3-day block bootstrap, descriptive after repeated historical reuse',
        Q_difference_interval95=np.quantile(delta,[.025,.975]).tolist(),
        mean_daily_return_difference_interval95=np.quantile((x[idx]-y[idx]).mean(1),[.025,.975]).tolist(),
        probability_bootstrap_Q_difference_positive=float(np.mean(delta>0)),confirmatory=False)
