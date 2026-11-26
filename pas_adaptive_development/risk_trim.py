"""Soft exposure bands, paid proportional reductions, and addition reservation."""
import numpy as np


def soft_limits(risk):
    result=dict(risk)
    for key in ['gross_fraction','net_fraction','asset_fraction','asset_notional','daily_volatility']:
        result[key]*=risk['sizing_headroom']
    return result


def trim_targets(base,pair,prices,equity,sigma,p,cfg):
    from replay_lab.common_comparison.engine import risk_values,rounded
    v=risk_values(base+pair,equity,prices,sigma)
    # Trigger at 90% of hard caps, return to 81%: an explicit hysteresis band
    # built from the EXISTING 0.9 headroom, with no performance-tuned threshold.
    desired=soft_limits(soft_limits(p['risk']))
    mapping={'gross':'gross_fraction','net':'net_fraction','asset':'asset_fraction',
             'asset_notional':'asset_notional','daily_volatility':'daily_volatility'}
    factor=min([1.]+[desired[key]/v[name] for name,key in mapping.items() if v[name]>0])
    targets=[]
    for owner in [base,pair]:
        q=np.array([rounded(x*factor,d) for x,d in zip(owner,p['execution']['decimals'])])
        # Do not manufacture a below-minimum partial reduction.
        small=(abs(q-owner)*prices<p['execution']['minimum_order_value'])&(q!=0)
        q[small]=owner[small]
        targets.append(q)
    ids=np.flatnonzero(pair)
    if len(ids):
        ratios=targets[1][ids]/pair[ids]
        if min(ratios)<=0 or abs(ratios.max()-ratios.min())/ratios.max()>cfg['pair_balance_tolerance']:
            return None
    if all(np.array_equal(old,new) for old,new in zip([base,pair],targets)):return None
    return targets


def cap_base_addition(current,proposed,pair,prices,equity,sigma,p):
    from replay_lab.common_comparison.engine import risk_breaches,rounded
    risk=soft_limits(p['risk'])
    # Keep reductions. Only additions consume shared capacity occupied by pairs.
    reduced=current.copy();extra=np.zeros_like(current)
    for j in range(len(current)):
        if current[j]*proposed[j]<0:
            reduced[j]=0;extra[j]=proposed[j]
        elif abs(proposed[j])<abs(current[j]):reduced[j]=proposed[j]
        else:extra[j]=proposed[j]-current[j]
    if not np.any(extra):return proposed.copy()
    if risk_breaches(reduced+pair,equity,prices,sigma,risk):return reduced
    def feasible(f):return not risk_breaches(reduced+pair+f*extra,equity,prices,sigma,risk)
    if feasible(1):return proposed.copy()
    lo,hi=0.,1.
    for _ in range(40):
        mid=(lo+hi)/2
        if feasible(mid):lo=mid
        else:hi=mid
    return np.array([rounded(x,d) for x,d in zip(reduced+lo*extra,p['execution']['decimals'])])
