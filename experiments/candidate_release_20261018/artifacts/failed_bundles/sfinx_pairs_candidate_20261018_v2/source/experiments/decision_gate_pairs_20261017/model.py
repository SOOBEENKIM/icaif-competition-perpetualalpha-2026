"""Convex three-action cost-sensitive learning; outputs decisions, not means."""
import numpy as np


def action(prediction, cost):
    return np.where(abs(prediction) > cost, np.sign(prediction), 0).astype(int)


def loss(prediction, target, cost):
    optimal = action(target, cost)
    shifted = 2*prediction-target
    return np.maximum(0, abs(shifted)-cost)-shifted*optimal+cost*abs(optimal)


def solve(z, y, cost, weight, penalty, tolerance=1e-6, maximum=100000):
    """Fenchel dual proximal acceleration, bounded dual and certified gap.

    max(0,|s|-C)=sup_{|u|<=1} u*s-C*|u|. Write v=w*u,
    q=Z'(v-w*a*); theta=-P^-1*q. Minimize q'P^-1*q+y'v+C'|v|
    subject to |v|<=w. The reported gap uses the original nonsmooth primal.
    """
    if (np.any(cost < 0) or np.any(weight <= 0) or np.any(penalty <= 0)
            or not all(np.isfinite(a).all() for a in (z, y, cost, weight, penalty))):
        raise ValueError('Finite data, nonnegative cost and positive weights/penalty required')
    optimal = action(y, cost); wa = weight*optimal
    inv = 1/penalty
    scaled = z*np.sqrt(inv)
    lipschitz = 2*np.linalg.eigvalsh(scaled.T@scaled)[-1]
    if lipschitz <= 0: raise ValueError('Nondegenerate design required')
    step = 1/lipschitz
    v = wa.copy(); extrapolated = v.copy(); acceleration = 1.
    constant = y@wa + cost@(weight*abs(optimal))
    for iteration in range(maximum+1):
        if iteration % 25 == 0 or iteration == maximum:
            q = z.T@(v-wa); theta = -inv*q
            primal = float(penalty@(theta*theta)+weight@loss(z@theta, y, cost))
            dual = float(-q@(inv*q)-y@v-cost@abs(v)+constant)
            gap = primal-dual
            if gap < -1e-9: raise ArithmeticError('Negative primal-dual gap')
            if gap <= tolerance:
                return theta, dict(iterations=iteration, primal=primal, dual=dual, gap=max(0.,gap))
        gradient = 2*z@(inv*(z.T@(extrapolated-wa)))+y
        trial = extrapolated-step*gradient
        next_v = np.clip(np.sign(trial)*np.maximum(abs(trial)-step*cost, 0), -weight, weight)
        next_acceleration = (1+np.sqrt(1+4*acceleration**2))/2
        extrapolated = next_v+(acceleration-1)/next_acceleration*(next_v-v)
        v, acceleration = next_v, next_acceleration
    raise ArithmeticError(f'Decision fit did not certify convergence: gap={gap}, iterations={maximum}')


def fit(x, target, unit, maturity, now, spec):
    if len(x) < spec['minimum_training_rows']: return None
    if np.any(maturity >= now) or np.any(unit <= 0) or not all(np.isfinite(v).all() for v in (x,target,unit)):
        raise ValueError('Strictly mature labels and finite positive source units required')
    weight = np.exp2(-(now-maturity)/(86400*spec['half_life_days'])); weight /= weight.sum()
    center = weight@x; scale = np.maximum(np.sqrt(weight@((x-center)**2)), 1e-6)
    z = np.c_[np.ones(len(x)), (x-center)/scale]
    penalty = np.full(z.shape[1], spec['ridge_penalty']); penalty[0] = spec['intercept_prior_rows']/len(x)
    y = np.clip(target,-spec['target_clip'],spec['target_clip'])
    cost = 2*(spec['fee_bps']+spec['impact_bps'])/10000/unit
    theta, diagnostics = solve(z,y,cost,weight,penalty,spec['solver_gap_tolerance'],spec['solver_max_iterations'])
    return dict(center=center.tolist(),scale=scale.tolist(),coefficient=theta.tolist(),rows=len(x),
        first_maturity=int(maturity.min()),last_maturity=int(maturity.max()),fit_time=int(now),
        training_actions={str(k):int(np.sum(action(y,cost)==k)) for k in [-1,0,1]},solver=diagnostics)


def rolling(times,features,target,maturity,original_raw,terminal_fresh,spec):
    if np.any(np.diff(times)!=3600) or not np.array_equal(maturity,times+spec['horizon']*3600):
        raise ValueError('Incorrect source cadence or maturity')
    if target.shape!=original_raw.shape or target.shape!=features['fresh'].shape: raise ValueError('Unaligned inputs')
    unit=features['vol']*np.sqrt(spec['horizon'])
    valid=np.isfinite(features['x']).all(-1)&features['fresh']&np.isfinite(original_raw)&np.isfinite(unit)&(unit>0)
    train=valid&terminal_fresh&np.isfinite(target)
    actions=np.zeros(target.shape,int);scores=np.full(target.shape,np.nan);available=np.zeros(target.shape,bool)
    stamps=np.full(target.shape,-1,np.int64);records=[];models={}
    for i,now in enumerate(times):
        if now%(spec['refit_hours']*3600)==0:
            models={}
            window=(times>=now-spec['lookback_days']*86400)&(maturity<now)&(times%(spec['sample_hours']*3600)==0)
            for k in range(target.shape[1]):
                ix=np.flatnonzero(window&train[:,k])
                model=fit(features['x'][ix,k],target[ix,k],unit[ix,k],maturity[ix],now,spec)
                if model is not None:
                    models[k]=model;records.append(dict(pair_index=k,first_source=int(times[ix[0]]),names=features['names'],**model))
        for k,model in models.items():
            if not valid[i,k]:continue
            z=np.r_[1.,(features['x'][i,k]-model['center'])/model['scale']]
            # This score has no expected-return interpretation and is never
            # assigned to the retained state's mu or terminal price.
            score=float(z@model['coefficient'])
            cost=2*(spec['fee_bps']+spec['impact_bps'])/10000/unit[i,k]
            scores[i,k]=score;actions[i,k]=int(action(np.asarray(score),cost));available[i,k]=True;stamps[i,k]=model['fit_time']
    return actions,scores,available,stamps,records


def map_gate(times,source_times,actions,available,baseline):
    state={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in baseline.items()}
    source=state['signal_times'];ix=np.searchsorted(source_times,source);bounded=np.clip(ix,0,len(source_times)-1)
    valid=(ix<len(source_times))&(source_times[bounded]==source)&(source<=times[:,None])
    kk=np.arange(source.shape[1])[None,:]
    valid&=available[bounded,kk]
    state['economic_action']=np.where(valid,actions[bounded,kk],0)
    state['economic_available']=valid
    return state
