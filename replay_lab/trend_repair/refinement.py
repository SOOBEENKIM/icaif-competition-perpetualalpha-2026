"""Offline refinement of the FIXED seven-day/.4 confidence breakout strategy.

Only two components vary: quicker invalidation of the shortest channels and
past-outcome calibration of the existing variance forecaster. The long-term
channels, entry confidence, capital and execution assumptions remain fixed.
"""
from dataclasses import dataclass
import numpy as np
from ..channel_crypto.model import fit as channel_fit
from ..trend_ml.risk import signals as risk_signals, RiskPolicy


@dataclass(frozen=True)
class RefinementPolicy:
    fast_channels: int = 0
    risk_model: str = 'original'
    buffer: float = .2

    @property
    def key(self):return f'refine-fast{self.fast_channels}-{self.risk_model}'


def calibrate(a, f, predictions, checkpoints, mode):
    """Fit variance scale using ONLY matured, previously emitted predictions.

    For QLIKE loss y/p-log(y/p)-1, the optimum variance multiplier on fixed p
    is mean(y/p). Shrink that estimate toward one (20 prior observations).
    No fitted-on-the-same-sample residuals or future model outputs are used.
    """
    if mode not in ('pooled','asset'):raise ValueError('Unknown calibration')
    out=predictions.copy();states=[]
    for state in checkpoints:
        cutoff=state['cutoff']
        ids=np.flatnonzero((a['known']>=cutoff-28*86400)&(f['maturity']<cutoff)&
            (a['known']%(6*3600)==0)&np.isfinite(predictions).all(1)&np.isfinite(f['target']).all(1))
        ratio=f['target'][ids]/np.maximum(predictions[ids]**2,1e-8)
        if mode=='pooled':
            multiplier=np.full(3,(ratio.sum()+20)/(ratio.size+20))
        else:multiplier=(ratio.sum(0)+20)/(len(ids)+20)
        multiplier=np.clip(multiplier,.5,2.)
        mask=(a['known']>=cutoff)&(a['known']<cutoff+7*86400)
        out[mask]*=np.sqrt(multiplier)
        states.append(dict(cutoff=int(cutoff),mode=mode,samples=int(ratio.size),
            last_maturity=int(f['maturity'][ids[-1]]) if len(ids) else None,
            variance_multiplier=multiplier.tolist(),lookback_days=28,prior_observations=20))
    return out,states


def fit(a,p,f,pred):
    if p.fast_channels not in (0,1,2):raise ValueError('Only 0/1/2 short channels may have hourly exits')
    model_name='original' if p.risk_model=='original' else 'learned'
    m,_=risk_signals(a,RiskPolicy(risk_model=model_name),f,pred)
    _,shadow=channel_fit(a,'long_only');by_time={s['known']:s for s in shadow}
    enabled=np.zeros((4,10),dtype=bool);blocked=np.zeros_like(enabled)
    reentry=np.zeros((4,10));stops=np.zeros((4,10));upper=np.zeros_like(stops)
    daily=[];active=np.zeros(10,dtype=bool);out=np.zeros_like(m['signal']);rows=[]
    for i,known in enumerate(a['known']):
        close=a['close'][i]
        if known%86400==0:
            daily.append(close.copy())
            if int(known) in by_time:
                state=by_time[int(known)];raw=np.array(state['positions']);stops=np.array(state['stops'])
                d=np.log(np.array(daily[-8:]));r=np.diff(d,axis=0)
                confidence=(d[-1]-d[0])/np.maximum(abs(r).sum(0),1e-12)
                active=m['signal'][i]>0
                enabled&=raw>0
                blocked&=raw>0
                blocked&=close[None,:]<=reentry
                enabled|=(raw>0)&(confidence>=.4)[None,:]&~blocked
                upper=np.array([np.array(daily[-h:]).max(0) for h in (5,10,20,30)])
        # Never turn a stale daily channel straight back on after an hourly
        # stop: require a new higher daily close or a reset channel lifecycle.
        fast=np.arange(4)[:,None]<p.fast_channels
        failed=enabled&fast&(close[None,:]<=stops)
        blocked|=failed;reentry=np.where(failed,upper,reentry);enabled&=~failed
        out[i]=enabled.mean(0)*active
        if np.any(failed) or known%86400==0:
            rows.append(dict(known=int(known),signal=out[i,:3].tolist(),
                early_exits=int(failed.sum()),enabled=enabled[:,:3].tolist(),blocked=blocked[:,:3].tolist()))
    m['signal']=out
    return m,rows
