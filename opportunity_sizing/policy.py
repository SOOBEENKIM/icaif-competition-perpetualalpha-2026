"""Incremental, per-asset allocation. Base targets are immutable constraints."""
import math
import numpy as np
from replay_lab.common_comparison.engine import risk_breaches,rounded


def allocate(base_qty, old_extra, prices, equity, covariance, risk, decimals,
             expected_return, uncertainty, entry_cost, exit_cost, capacity,
             allowed, hold_allowed, *, uniform=False):
    base=np.asarray(base_qty,float);px=np.asarray(prices,float);cov=np.asarray(covariance,float)
    n=len(base);direction=np.sign(base);w=base*px/equity
    if (not np.isfinite(cov).all() or not np.allclose(cov,cov.T) or np.linalg.eigvalsh(cov).min() < -1e-12):
        raise ValueError('Invalid covariance')
    if risk_breaches(base,equity,px,cov,risk):raise ValueError('Base target requires risk repair')
    for a in [base,old_extra,px,expected_return,uncertainty,entry_cost,exit_cost,capacity]:
        if np.shape(a)!=(n,) or not np.isfinite(a).all():raise ValueError('Invalid vector')
    if equity<=0 or np.any(px<=0) or np.any(np.asarray(capacity)<0):raise ValueError('Invalid market/account')
    if any(np.any(np.asarray(a)<0) for a in [uncertainty,entry_cost,exit_cost]):raise ValueError('Negative uncertainty/cost')
    old=np.where(direction*np.asarray(old_extra)>0,abs(np.asarray(old_extra))*px/equity,0.)
    cap=min(risk['asset_fraction'],risk['asset_notional']/equity)*risk['sizing_headroom']
    max_extra=np.maximum(0,np.minimum(2*abs(w),cap-abs(w)))
    max_extra=np.minimum(max_extra,old+np.asarray(capacity)/equity)
    signed_edge=direction*np.asarray(expected_return)-np.asarray(uncertainty)
    can_add=np.asarray(allowed,bool)&(signed_edge>np.asarray(entry_cost)+np.asarray(exit_cost))&(direction!=0)&(np.asarray(capacity)>0)
    can_hold=np.asarray(hold_allowed,bool)&(signed_edge>0)&(direction!=0)
    upper=np.where(can_add,max_extra,np.where(can_hold,np.minimum(old,max_extra),0.))
    if uniform:
        # Fixed 1.5x size control: deliberately ignores estimated opportunity.
        upper=np.where(np.asarray(allowed,bool)&(direction!=0),np.minimum(.5*abs(w),max_extra),0.)
    limits=dict(gross=max(abs(w).sum(),risk['gross_fraction']*risk['sizing_headroom']),
                net=max(abs(w.sum()),risk['net_fraction']*risk['sizing_headroom']),
                variance=max(float(24*w@cov@w),(risk['daily_volatility']*risk['sizing_headroom'])**2))
    lam=5.;horizon=24.
    flow=np.maximum(np.asarray(capacity)/equity,1e-12)
    impact=.001/flow # A declared convex capacity proxy, not measured historical L2 impact.
    gain=direction*np.asarray(expected_return)-np.asarray(uncertainty)-np.asarray(exit_cost)

    def utility(x):
        target=w+direction*x
        return float(gain@x-.5*lam*horizon*(target@cov@target)
                     -np.asarray(entry_cost)@abs(x-old)-np.sum(impact*((x-old)**2+x*x)))

    def interval(x,j):
        other=w+direction*x;other[j]=w[j]
        lo,hi=0.,float(upper[j]);s=direction[j]
        if not s:return 0.,0.
        hi=min(hi,limits['gross']-(abs(other).sum()-abs(w[j]))-abs(w[j]))
        total=other.sum()
        low_net,high_net=sorted([(-limits['net']-total)/s,(limits['net']-total)/s])
        lo=max(lo,low_net);hi=min(hi,high_net)
        A=24*cov[j,j];B=48*s*(cov[j]@other);C=24*other@cov@other-limits['variance']
        if A>1e-18:
            disc=B*B-4*A*C
            if disc < -1e-14:raise AssertionError('No feasible risk interval')
            root=math.sqrt(max(0.,disc));lo=max(lo,(-B-root)/(2*A));hi=min(hi,(-B+root)/(2*A))
        elif abs(B)>1e-18:
            if B>0:hi=min(hi,-C/B)
            else:lo=max(lo,-C/B)
        elif C>1e-12:raise AssertionError('No feasible variance')
        if hi<lo-1e-10:raise AssertionError('Empty feasible interval')
        return max(0.,lo),max(0.,hi)

    x=np.zeros(n)
    if uniform:
        # Scale the uniform extra sleeve downward as a whole to preserve ratios.
        def feasible(v):
            target=w+direction*v
            return (abs(target).sum()<=limits['gross']+1e-12 and abs(target.sum())<=limits['net']+1e-12
                    and 24*target@cov@target<=limits['variance']+1e-12)
        low,high=0.,1.
        for _ in range(50):
            mid=(low+high)/2
            if feasible(upper*mid):low=mid
            else:high=mid
        x=upper*low
    else:
        for _ in range(50):
            prior=x.copy()
            for j in range(n):
                if upper[j]<=0:continue
                lo,hi=interval(x,j)
                other=w+direction*x;other[j]=w[j]
                quadratic=lam*horizon*cov[j,j]+4*impact[j]
                linear=gain[j]-lam*horizon*direction[j]*(cov[j]@other)+2*impact[j]*old[j]
                candidates=[lo,hi,float(np.clip(old[j],lo,hi))]
                for sign in [-1,1]:
                    value=float(np.clip((linear-entry_cost[j]*sign)/max(quadratic,1e-18),lo,hi))
                    if (value-old[j])*sign>=-1e-12:candidates.append(value)
                best=x[j];score=utility(x)
                for value in candidates:
                    trial=x.copy();trial[j]=value;v=utility(trial)
                    if v>score+1e-14:best,score=value,v
                x[j]=best
            if max(abs(x-prior),default=0.)<1e-10:break
    extra=np.array([rounded(s*v*equity/p,int(dp)) for s,v,p,dp in zip(direction,x,px,decimals)])
    target=base+extra
    # Rounding one hedge leg can increase risk: reject the optional sleeve, do
    # not repair it by shrinking or changing a protected base leg.
    fallback=bool(risk_breaches(target,equity,px,cov,risk))
    if fallback:extra[:]=0.;target=base.copy()
    final_x=abs(extra)*px/equity;increment=utility(final_x)-utility(np.zeros(n))
    if not uniform and increment < -1e-12:extra[:]=0.;target=base.copy();increment=0.
    assert not risk_breaches(target,equity,px,cov,risk)
    assert np.all(direction*extra>=-1e-12)
    assert not np.any(extra[~(np.asarray(allowed,bool)|np.asarray(hold_allowed,bool))])
    return dict(target_qty=target.tolist(),extra_qty=extra.tolist(),eligible=can_add.tolist(),
                signed_edge=signed_edge.tolist(),utility_gain=float(increment),rounding_fallback=fallback)
