"""Causal pooled regression of fully observed episodic net cash outcomes."""
import numpy as np


def feature(proposal, d, state, i, initial):
    k = proposal['pair_index']; ids = proposal['ids']
    unit = float(state['vol'][i, k]) * np.sqrt(proposal['horizon'])
    gross = float(abs(np.asarray(proposal['target'])) @ d['par_close'][i, ids])
    if not np.isfinite(unit) or unit <= 0 or gross <= 0:
        raise ValueError('Positive finite source unit and proposed gross required')
    x = np.r_[state['lifecycle_features'][i, k], np.eye(6)[k], proposal['direction'],
              proposal['expected_net_bps']/10000/unit, proposal['expected_funding_bps']/10000/unit,
              np.log(gross/initial), proposal['source_beta']]
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite entry features')
    return x, unit, gross


def eligible(rows, now, spec):
    return [r for r in rows if r['status'] in ('closed','unfilled')
            and now-spec['lookback_days']*86400 <= r['source'] < now
            and r['outcome_time'] < now]


def uniqueness(rows):
    """Average inverse concurrency on actual source-to-outcome intervals."""
    value = np.ones(len(rows))
    for pair in sorted({r['pair_index'] for r in rows}):
        indexes = [i for i,r in enumerate(rows) if r['pair_index'] == pair]
        boundaries = sorted({t for i in indexes for t in (rows[i]['source'],rows[i]['outcome_time'])})
        for i in indexes:
            begin,end = rows[i]['source'],rows[i]['outcome_time']
            if end <= begin: raise ValueError('Outcome must follow the source')
            mass = 0.
            for left,right in zip(boundaries,boundaries[1:]):
                if left >= begin and right <= end:
                    count = sum(rows[j]['source'] <= left and rows[j]['outcome_time'] >= right for j in indexes)
                    mass += (right-left)/count
            value[i] = mass/(end-begin)
    return value


def fit(rows, now, spec):
    rows = eligible(rows, now, spec)
    if len(rows) < spec['minimum_rows']: return None
    x=np.asarray([r['features'] for r in rows]); y=np.asarray([r['target'] for r in rows])
    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('Invalid complete labels')
    ends=np.asarray([r['outcome_time'] for r in rows]); sources=np.asarray([r['source'] for r in rows])
    weight=uniqueness(rows)*np.exp2(-(now-ends)/(86400*spec['half_life_days']))
    effective=float(weight.sum()**2/(weight@weight)); weight/=weight.sum()
    center=weight@x; scale=np.maximum(np.sqrt(weight@((x-center)**2)),1e-6)
    z=np.c_[np.ones(len(rows)),(x-center)/scale]
    penalty=np.full(z.shape[1],spec['ridge_penalty']);penalty[0]=spec['intercept_prior_rows']/effective
    clipped=np.clip(y,-spec['target_clip'],spec['target_clip'])
    coef=np.linalg.solve(z.T@(z*weight[:,None])+np.diag(penalty),z.T@(weight*clipped))
    assert ends.max() < now
    return dict(fit_time=int(now),rows=len(rows),effective_rows=effective,
                first_source=int(sources.min()),last_source=int(sources.max()),last_outcome=int(ends.max()),
                center=center.tolist(),scale=scale.tolist(),coefficient=coef.tolist(),
                positive_rows=int(np.sum(y>0)),zero_rows=int(np.sum(y==0)),
                training_ids=[r['sample_id'] for r in rows])


def predict(model,x):
    if len(x)!=len(model['center']): raise ValueError('Feature dimension mismatch')
    return float(np.r_[1.,(x-model['center'])/model['scale']]@model['coefficient'])


def choose(choices,d,state,i,initial,model,kind):
    choices=[c for c in choices if kind!='filter' or c['pair_index']<2]
    if model is None:return choices
    if model['fit_time']>d['times'][i]: raise ValueError('Future fitted model')
    kept=[]
    for proposal in choices:
        x,unit,gross=feature(proposal,d,state,i,initial)
        estimate=predict(model,x)
        if estimate>0:
            kept.append(dict(proposal,lifecycle_value=estimate*unit*gross,
                             lifecycle_model_time=model['fit_time']))
    if kind=='rank':kept.sort(key=lambda r:(-r['lifecycle_value'],r['pair']))
    return kept
