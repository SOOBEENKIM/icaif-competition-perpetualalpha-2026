"""Two-coefficient pooled funding forecast, fitted before trading begins."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from . import experiment as e
from ..macro_development.data import load
from ..macro_development.model import inputs
from ..macro_development.carry import signals
from ..common_comparison.models import timestamp
from ..pairs_development.experiment import write,stamp

DEST=e.OUT/'funding_forecast'
CUTOFF=timestamp('2026-05-17T00:00:00Z')

def features(m):
    f=pd.DataFrame(m.funding).shift(1)
    return np.stack([np.nan_to_num(f.ewm(span=h,adjust=False,min_periods=h).mean().to_numpy(),nan=0) for h in (24,168)],-1)

def fit(m):
    x=features(m)
    ids=np.flatnonzero((m.times%86400==0)&(np.arange(len(m.times))>=168)&(m.times+169*3600<CUTOFF))
    y=np.array([m.funding[i+2:i+170].mean(0) for i in ids])
    scale=np.maximum(np.median(abs(x[ids,:,1]),axis=0),.0000125)
    xx=(x[ids]/scale[None,:,None]).reshape(-1,2);yy=(y/scale).ravel()
    linear=nnls(xx,yy)[0];robust=linear.copy()
    for _ in range(3):
        residual=xx@robust-yy;mad=max(float(np.median(abs(residual))),.1)
        w=np.sqrt(np.minimum(1,1.5*mad/np.maximum(abs(residual),1e-12)))
        robust=nnls(xx*w[:,None],yy*w)[0]
    return dict(cutoff=CUTOFF,last_training_decision=int(m.times[ids[-1]]),last_training_label_end=int(m.times[ids[-1]]+169*3600),
        daily_cross_sections=len(ids),asset_labels=len(yy),scale=scale.tolist(),nnls=linear.tolist(),robust=robust.tolist())

def weights(m,forecast):
    x,sleeve=inputs(m);trend=x[:,:,[1,2]].mean(-1);state=np.zeros(10);w=np.zeros_like(m.close)
    for i,t in enumerate(m.times):
        if t>=CUTOFF and t%21600==0:
            direction=-np.sign(forecast[i]);carry=abs(forecast[i])*168;agreement=direction*trend[i]>0
            leave=(direction!=np.sign(state))|(carry<.0019)|~agreement
            state[leave]=0;enter=(carry>.0038)&agreement;state[enter]=direction[enter]*sleeve[i,enter]
        w[i]=state
    return w

def make(m,baseline,state=None):
    state=state or json.loads((DEST/'fit.json').read_text())
    x=features(m);old=signals(m)[1]['forecast'];fitted={k:x@np.array(state[k]) for k in ('nnls','robust')}
    fitted['robust_blend']=(fitted['robust']+old)/2
    scale=json.loads((e.PREVIOUS/'consensus/scales.json').read_text())['consensus_carry']
    price=baseline-signals(m)[0]['carry_trend']*scale
    return {k:price+weights(m,v)*scale for k,v in fitted.items()},dict(features=x,**fitted)

def main():
    m,_,_=load();state=fit(m);write(DEST/'fit.json',state)
    write(DEST/'model_plan.json',dict(time=stamp(),hypothesis='A two-coefficient forecast estimates funding persistence instead of extrapolating the conservative minimum for seven days.',
        fit='Pooled nonnegative least squares, two coefficients, no intercept; per-market input/label scales from training only. Second model adds three fixed Huber IRLS steps. Fixed half blend with old forecast.',
        cutoff=CUTOFF,inference='Model frozen before May17 evaluation; only settled rates lagged one hour. Position lifecycle, thresholds, price trend agreement and risk sleeves unchanged.',
        caveat='Predicts external mainnet funding, not unknown pa funding. Directional price risk remains. New engineering model, not a published NNLS trading agent.'))
    e.OUT=DEST;e.make=make
    e.main(dict(hypotheses=['Learn funding persistence with two shared nonnegative coefficients; compare ordinary and robust fit and fixed forecast blend.'],
        references=['https://link.springer.com/article/10.1007/s42521-026-00213-3'],reference_access='Risk/persistence motivation only; our NNLS forecast is not the paper model.'))

if __name__=='__main__':main()
