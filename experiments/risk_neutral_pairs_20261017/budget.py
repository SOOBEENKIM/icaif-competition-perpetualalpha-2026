"""Closed-form pair notional cap from information available at decision time."""
import numpy as np


def variance(notional,sigma,equity):
    q=np.asarray(notional,float)
    if equity<=0:return float('inf')
    return float(q@sigma@q/equity**2)


def maximum_gross(base_notional,unit_pair,sigma,equity,cost_per_gross):
    base=np.asarray(base_notional,float);u=np.asarray(unit_pair,float);s=np.asarray(sigma,float)
    if not np.isfinite([equity,cost_per_gross]).all() or equity<=0 or cost_per_gross<0 or not (np.isfinite(base).all() and np.isfinite(u).all() and np.isfinite(s).all()):
        raise ValueError('Invalid risk budget inputs')
    v=float(base@s@base);a=float(u@s@u)-v*(cost_per_gross/equity)**2
    b=2*float(base@s@u)+2*v*cost_per_gross/equity
    # (base+g*u)'S(base+g*u)/(E-cost*g)^2 <= base'Sbase/E^2.
    # For a positive quadratic coefficient, the nonzero feasible interval is
    # g in [0,-b/a], only when the initial marginal variance is negative.
    if v<=0 or a<=1e-20 or b>=0:return 0.
    cap=-b/a
    if cost_per_gross>0:cap=min(cap,equity/cost_per_gross*(1-1e-12))
    return max(0.,float(cap))


def budget(current_base,target_base,prices,ids,unit_weights,sigma,equity,cost_per_gross,margin):
    unit=np.zeros(len(prices));unit[ids]=unit_weights
    amounts=[np.asarray(q)*prices for q in (current_base,target_base)]
    caps=[maximum_gross(q,unit,sigma,equity,cost_per_gross) for q in amounts]
    return dict(maximum_gross=min(caps)*margin,current_baseline_cap=caps[0],pending_baseline_cap=caps[1],
        baseline_variances=[variance(q,sigma,equity) for q in amounts])


def rounded_risk_check(current_base,target_base,pair_qty,prices,ids,sigma,equity,cost_per_gross):
    addition=np.zeros(len(prices));addition[ids]=np.asarray(pair_qty)*prices[ids]
    gross=float(abs(addition).sum());after_equity=equity-gross*cost_per_gross
    before=[variance(np.asarray(q)*prices,sigma,equity) for q in (current_base,target_base)]
    after=[variance(np.asarray(q)*prices+addition,sigma,after_equity) for q in (current_base,target_base)]
    return all(b<=a*(1+1e-12)+1e-22 for a,b in zip(before,after)),dict(before=before,after=after,
        estimated_cost_cash=gross*cost_per_gross,conservative_after_equity=after_equity)
