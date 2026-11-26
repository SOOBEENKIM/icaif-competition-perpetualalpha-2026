"""Fixed small tree execution classifier and conditional net-cash heads."""
from functools import lru_cache
import numpy as np
import xgboost as xgb
from experiments.lifecycle_value_pairs_20261018.model import uniqueness


def weights(rows,now,spec,key):
    adapted=[dict(r,outcome_time=r[key]) for r in rows]
    w=uniqueness(adapted)*np.exp2(-(now-np.array([r[key] for r in rows]))/(86400*spec['half_life_days']))
    effective=float(w.sum()**2/(w@w));return w/w.sum(),effective


def train_tree(x,y,w,spec,classification=False):
    parameters={k:v for k,v in spec['tree'].items() if k!='num_boost_round'}
    parameters.update(objective='multi:softprob' if classification else 'reg:squarederror',verbosity=0)
    if classification:parameters['num_class']=3
    tree=xgb.train(parameters,xgb.DMatrix(x,label=y,weight=w*len(w)),num_boost_round=spec['tree']['num_boost_round'])
    return tree.save_raw(raw_format='json').decode()


def ridge_head(rows,now,spec):
    if not rows:return dict(kind='constant',value=0.,rows=0,effective_rows=0.)
    x=np.asarray([r['features'] for r in rows])
    y=np.clip([r['net_pnl']/r['intended_gross']/spec['target_scale'] for r in rows],-spec['target_clip'],spec['target_clip'])
    w,eff=weights(rows,now,spec,'outcome_time')
    if len(rows)<spec['minimum_head_rows']:
        return dict(kind='constant',value=float(w@y*eff/(eff+spec['intercept_prior_rows'])),rows=len(rows),effective_rows=eff)
    center=w@x;scale=np.maximum(np.sqrt(w@((x-center)**2)),1e-6)
    z=np.c_[np.ones(len(x)),(x-center)/scale]
    penalty=np.full(z.shape[1],spec['ridge_penalty']);penalty[0]=spec['intercept_prior_rows']/eff
    coefficient=np.linalg.solve(z.T@(z*w[:,None])+np.diag(penalty),z.T@(w*y))
    return dict(kind='ridge',center=center.tolist(),scale=scale.tolist(),coefficient=coefficient.tolist(),rows=len(rows),effective_rows=eff)


def fit(rows,now,horizon,spec):
    initial=[r for r in rows if r['horizon']==horizon and now-spec['training_window_days']*86400<=r['source']<now and r['initial_time']<now]
    mature=[r for r in initial if r['status'] in ('closed','unfilled') and r['outcome_time']<now]
    if min(len(initial),len(mature))<spec['minimum_rows'] or len({r['source'] for r in mature})<spec['minimum_distinct_sources']:return None
    x=np.asarray([r['features'] for r in initial]);classes=np.asarray([r['initial_class'] for r in initial])
    w,eff=weights(initial,now,spec,'initial_time')
    prior=np.bincount(classes,weights=w*eff,minlength=3)+1.;prior/=prior.sum()
    classifier=train_tree(x,classes,w,spec,True)
    xm=np.asarray([r['features'] for r in mature])
    y=np.clip([r['net_pnl']/r['intended_gross']/spec['target_scale'] for r in mature],-spec['target_clip'],spec['target_clip'])
    wm,em=weights(mature,now,spec,'outcome_time');direct=train_tree(xm,y,wm,spec)
    heads={str(k):ridge_head([r for r in mature if r['initial_class']==k],now,spec) for k in [1,2]}
    assert all(r['net_pnl']==0 for r in mature if r['initial_class']==0)
    return dict(fit_time=now,horizon=horizon,classifier=classifier,direct=direct,heads=heads,
        prior=prior.tolist(),target_scale=spec['target_scale'],initial_rows=len(initial),mature_rows=len(mature),
        initial_effective_rows=eff,mature_effective_rows=em,
        initial_ids=[r['sample_id'] for r in initial],cash_ids=[r['sample_id'] for r in mature],
        class_counts=np.bincount(classes,minlength=3).tolist(),
        last_initial=max(r['initial_time'] for r in initial),last_cash=max(r['outcome_time'] for r in mature))


@lru_cache(maxsize=64)
def booster(raw):
    tree=xgb.Booster(params={'nthread':1});tree.load_model(bytearray(raw,'utf-8'));return tree


def predict(model,x,now,kind):
    if model['fit_time']>now or max(model['last_initial'],model['last_cash'])>=model['fit_time']:raise ValueError('Future information')
    x=np.asarray(x,dtype=float)
    if kind=='tree':return float(booster(model['direct']).inplace_predict(x[None,:])[0])*model['target_scale'],{}
    probability=booster(model['classifier']).inplace_predict(x[None,:])[0]
    values=[0.]
    for k in [1,2]:
        head=model['heads'][str(k)]
        v=head['value'] if head['kind']=='constant' else float(np.r_[1.,(x-head['center'])/head['scale']]@head['coefficient'])
        values.append(v*model['target_scale'])
    return float(probability@values),dict(initial_probabilities=probability.tolist(),conditional_net_returns=values)
