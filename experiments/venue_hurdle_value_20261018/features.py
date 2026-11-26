"""Execution information available at the decision, with no forward activity."""
import numpy as np
from experiments.venue_execution_value_20261018.model import features as old_features


def activity(d, pairs):
    active=(d['par_volume']>0)&(d['par_trades']>0)
    n=len(active);fields=[];names=[]
    for a,b in pairs:
        x=[];columns=[]
        def add(name,v):columns.append(name);x.append(v)
        joint=active[:,a]&active[:,b]
        one=active[:,a]^active[:,b]
        for window in [12,72,288]:
            def mean(v):
                cumulative=np.r_[0.,np.cumsum(v)]
                lo=np.maximum(np.arange(n)+1-window,0)
                return (cumulative[1:]-cumulative[lo])/(np.arange(n)+1-lo)
            add(f'active_a_{window}',mean(active[:,a]))
            add(f'active_b_{window}',mean(active[:,b]))
            add(f'joint_active_{window}',mean(joint))
            previous=np.r_[False,joint[:-1]]
            denominator=mean(previous)
            add(f'past_joint_to_joint_{window}',np.divide(mean(previous&joint),denominator,out=np.zeros(n),where=denominator>0))
            add(f'past_joint_to_one_{window}',np.divide(mean(previous&one),denominator,out=np.zeros(n),where=denominator>0))
        for symbol,j in [('a',a),('b',b)]:
            streak=np.zeros(n)
            for i in range(n):streak[i]=min(288,(streak[i-1] if i else 0)+1) if active[i,j] else 0
            add(f'consecutive_active_{symbol}',np.log1p(streak))
            add(f'current_trades_{symbol}',np.log1p(d['par_trades'][:,j]))
        if names:assert names==columns
        names=columns;fields.append(np.stack(x,axis=-1))
    return np.stack(fields,axis=1),names


def features(proposal,d,de,i,p):
    original,gross=old_features(proposal,d,de,i,p['initial'])
    ids=proposal['ids'];q=np.abs(proposal['target']);prices=d['par_close'][i,ids]
    lots=10.**(-np.asarray(p['execution']['decimals'])[ids])
    shares=q/d['par_volume'][i,ids]
    geometry=np.r_[np.log1p(q/lots),shares,np.log1p(q*prices/p['execution']['minimum_order_value']),
                   (q[1]*prices[1]/(q[0]*prices[0]))/proposal['beta']-1]
    x=np.r_[original,de['execution_x'][i,proposal['pair_index']],geometry]
    if not np.isfinite(x).all():raise ValueError('Invalid source execution features')
    return x,gross


def initial_state(quantities,target,tolerance):
    q=np.asarray(quantities);target=np.asarray(target)
    if not np.isfinite(q).all() or not np.isfinite(target).all() or np.any(target==0):raise ValueError('Unknown initial state')
    if not np.any(q):return 0
    ratio=q/target
    return 2 if min(ratio)>0 and abs(ratio[0]-ratio[1])/max(ratio)<=tolerance else 1
