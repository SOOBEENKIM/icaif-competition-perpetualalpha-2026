"""Net execution-value learning; a zero fill is an observed zero, not a price label."""
import numpy as np
from experiments.lifecycle_value_pairs_20261018.model import uniqueness


def features(proposal,d,de,i,initial):
    k=proposal['pair_index'];direction=proposal['direction'];x=de['x'][i,k]
    gross=float(abs(np.asarray(proposal['target']))@d['par_close'][i,proposal['ids']])
    if gross<=0 or not np.isfinite(x).all():raise ValueError('Finite source features and positive gross required')
    return np.r_[x,direction*x,direction,np.log(gross/initial)],gross


def fit(rows,now,h,spec):
    past=[r for r in rows if r['horizon']==h and r['status'] in ('closed','unfilled')
          and now-spec['training_window_days']*86400<=r['source']<now and r['outcome_time']<now]
    if len(past)<spec['minimum_rows'] or len({r['source'] for r in past})<spec['minimum_distinct_sources']:return None
    x=np.asarray([r['features'] for r in past]);y=np.asarray([r['net_pnl']/r['intended_gross']/spec['target_scale'] for r in past])
    ends=np.asarray([r['outcome_time'] for r in past])
    w=uniqueness(past)*np.exp2(-(now-ends)/(86400*spec['half_life_days']));effective=float(w.sum()**2/(w@w));w/=w.sum()
    center=w@x;scale=np.maximum(np.sqrt(w@((x-center)**2)),1e-6);z=np.c_[np.ones(len(x)),(x-center)/scale]
    penalty=np.full(z.shape[1],spec['ridge_penalty']);penalty[0]=spec['intercept_prior_rows']/effective
    coefficient=np.linalg.solve(z.T@(z*w[:,None])+np.diag(penalty),z.T@(w*np.clip(y,-spec['target_clip'],spec['target_clip'])))
    assert np.isfinite(coefficient).all() and ends.max()<now
    return dict(fit_time=now,horizon=h,rows=len(past),effective_rows=effective,distinct_sources=len({r['source'] for r in past}),
        last_outcome=int(ends.max()),center=center.tolist(),scale=scale.tolist(),coefficient=coefficient.tolist(),
        target_scale=spec['target_scale'],training_ids=[r['sample_id'] for r in past],
        zero_rows=sum(r['status']=='unfilled' for r in past),positive_rows=int(np.sum(y>0)))


def predict(model,x,now):
    if model['last_outcome']>=model['fit_time'] or model['fit_time']>now:raise ValueError('Future or incomplete fit')
    return float(np.r_[1.,(x-model['center'])/model['scale']]@model['coefficient'])*model['target_scale']


def adjusted_net(prediction,base,scenario):
    additional=max(0.,scenario['fee_bps']+scenario['impact_bps']-base['fee_bps']-base['impact_bps'])
    return prediction-2*additional/10000
