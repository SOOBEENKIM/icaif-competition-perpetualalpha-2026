"""Execution models fitted only to received-book outcomes already observed."""
from collections import Counter
from functools import lru_cache
import numpy as np
import xgboost as xgb


def weights(rows):
    count=Counter(r['source'] for r in rows)
    return np.asarray([1./count[r['source']] for r in rows])


def tree(x,y,w,spec,classes=False):
    parameters={k:v for k,v in spec['tree'].items() if k!='num_boost_round'}
    parameters.update(objective='multi:softprob' if classes else 'reg:squarederror',verbosity=0)
    if classes:parameters['num_class']=4
    trained=xgb.train(parameters,xgb.DMatrix(x,label=y,weight=w),num_boost_round=spec['tree']['num_boost_round'])
    return trained.save_raw(raw_format='json').decode()


def fit(rows,tapes,at,spec):
    past=[r for r in rows if r['tape'] in tapes and r['available_at']<at]
    if len(past)<spec['minimum_fit_rows']:return None
    x=np.asarray([r['features'] for r in past]);y=np.asarray([r['outcome'] for r in past]);w=weights(past)
    prior=np.bincount(y,weights=w,minlength=4)+1.;prior/=prior.sum()
    balanced=[r for r in past if r['outcome']==2]
    reconciled=[r for r in past if r['outcome']!=3]
    fraction=None;fraction_mean=0.
    if balanced:
        xb=np.asarray([r['features'] for r in balanced]);yb=np.asarray([r['held_fraction'] for r in balanced]);wb=weights(balanced)
        fraction=tree(xb,yb,wb,spec);fraction_mean=float(wb@yb/wb.sum())
    costs=None;cost_mean=0.
    if reconciled:
        xr=np.asarray([r['features'] for r in reconciled]);yr=np.asarray([r['entry_cost_return']*10000 for r in reconciled]);wr=weights(reconciled)
        costs=tree(xr,yr,wr,spec);cost_mean=float(wr@yr/wr.sum())
    return dict(fit_time=at,tapes=tapes,rows=len(past),unique_source_times=len({r['source'] for r in past}),
        classifier=tree(x,y,w,spec,True),fraction=fraction,costs=costs,
        prior=prior.tolist(),fraction_mean=fraction_mean,cost_mean_bps=cost_mean,
        class_counts=np.bincount(y,minlength=4).tolist(),balanced_rows=len(balanced),cost_rows=len(reconciled),
        last_observed=max(r['available_at'] for r in past),training_ids=[r['sample_id'] for r in past])


@lru_cache(maxsize=12)
def booster(raw):
    model=xgb.Booster(params={'nthread':1});model.load_model(bytearray(raw,'utf-8'));return model


def predict(model,x,now,constant=False):
    if model['fit_time']>now or model['last_observed']>=model['fit_time']:raise ValueError('Future fit or labels')
    xx=np.asarray(x)[None,:]
    if constant:prob=np.asarray(model['prior']);fraction=model['fraction_mean'];cost=model['cost_mean_bps']
    else:
        prob=booster(model['classifier']).inplace_predict(xx)[0]
        fraction=float(booster(model['fraction']).inplace_predict(xx)[0]) if model['fraction'] else 0.
        cost=float(booster(model['costs']).inplace_predict(xx)[0]) if model['costs'] else 0.
    return dict(probabilities=np.asarray(prob).tolist(),held_fraction_if_balanced=float(np.clip(fraction,0,1)),
        extra_cost_bps_if_reconciled=max(0.,cost))


def score_plan(plan,forecast):
    """A source-time utility proxy, never a measured full-horizon profit."""
    probability=np.asarray(forecast['probabilities']);gross=plan['intended_gross']
    held=probability[2]*forecast['held_fraction_if_balanced']
    repair=(1-probability[3])*forecast['extra_cost_bps_if_reconciled']/10000*gross
    unresolved=probability[3]*max(0.,plan['exit_cost_reserve'])
    return dict(value=held*plan['expected_net_cash']-repair-unresolved,
                expected_held_fraction=held,repair_cost=repair,unresolved_reserve=unresolved)
