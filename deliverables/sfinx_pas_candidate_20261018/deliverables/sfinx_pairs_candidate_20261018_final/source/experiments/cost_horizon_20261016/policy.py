"""Inventory-aware convex decisions, deterministic state, hard risk first."""
import numpy as np
from replay_lab.common_comparison.engine import project,risk_breaches,rounded
from replay_lab.ema_rebuild.engine import Controller


def utility(w,old,mu,q,cost,reserve):
    return float(mu@w-.5*w@q@w-cost@abs(w-old)-reserve@abs(w))


def one_coordinate(a,b,old,cost,reserve,bound):
    """Exact minimum of .5*a*x²-b*x+cost*|x-old|+reserve*|x|."""
    knots=sorted(set([-bound,bound,float(np.clip(old,-bound,bound)),0.]))
    candidates=knots.copy()
    for lo,hi in zip(knots,knots[1:]):
        mid=(lo+hi)/2
        stationary=(b-cost*np.sign(mid-old)-reserve*np.sign(mid))/a
        candidates.append(float(np.clip(stationary,lo,hi)))
    return min(candidates,key=lambda x:(.5*a*x*x-b*x+cost*abs(x-old)+reserve*abs(x),abs(x-old)))


def optimize(mu,cov,old,gamma,horizon,cost,ambiguity=None,bound=.135,max_iter=400):
    mu=np.asarray(mu,float);old=np.asarray(old,float);cov=np.asarray(cov,float)
    cost=np.broadcast_to(cost,mu.shape).copy();ambiguity=np.zeros_like(mu) if ambiguity is None else np.asarray(ambiguity,float)
    if not all(np.isfinite(x).all() for x in (mu,cov,old,cost,ambiguity)) or gamma<=0 or np.any(cost<0) or np.any(ambiguity<0):
        raise ValueError('Invalid optimizer input')
    q=gamma*horizon*(cov+cov.T)/2
    if np.linalg.eigvalsh(q).min()<=0:raise ValueError('Covariance must be positive definite')
    reserve=cost+ambiguity
    w=np.clip(old,-bound,bound).copy()
    for iteration in range(max_iter):
        previous=w.copy()
        for j in range(len(w)):
            b=mu[j]-(q[j]@w-q[j,j]*w[j])
            w[j]=one_coordinate(q[j,j],b,old[j],cost[j],reserve[j],bound)
        if np.max(abs(w-previous))<1e-10:break
    else:raise RuntimeError('Coordinate optimizer did not converge')
    if utility(w,old,mu,q,cost,reserve)<utility(np.clip(old,-bound,bound),old,mu,q,cost,reserve)-1e-10:
        raise AssertionError('Optimizer reduced utility')
    return w,{'iterations':iteration+1,'utility':utility(w,old,mu,q,cost,reserve),
              'hold_utility':utility(old,old,mu,q,cost,reserve)}


class CostController:
    def __init__(self,times,mu,ambiguity,gamma,horizon,*,saved=None,cost_bps=19.,partial=.5):
        self.times=np.asarray(times);self.mu=np.asarray(mu);self.ambiguity=np.asarray(ambiguity)
        self.gamma=float(gamma);self.horizon=int(horizon);self.cost=cost_bps/10000;self.partial=partial
        self.last_decision=int((saved or {}).get('last_decision',-1))

    def snapshot(self):return {'last_decision':self.last_decision}

    def propose(self,time,prices,qty,equity,raw,sigma,volume,trades,protocol,buffer=.2,halted=False,valid_input=True,**kwargs):
        if (not valid_input or not np.isfinite(equity) or equity<=0 or
            not all(np.isfinite(v).all() for v in (prices,qty,sigma,volume,trades)) or np.any(np.asarray(prices)<=0)):
            return None
        forced=halted or bool(risk_breaches(qty,equity,prices,sigma,protocol['risk']))
        if time<=self.last_decision or (time%21600 and not forced):return None
        if forced:
            c=Controller(self.last_decision)
            action=c.propose(time,prices,qty,equity,np.zeros_like(qty),sigma,volume,trades,protocol,halted=halted)
            if action is not None:self.last_decision=time
            return action
        i=int(np.searchsorted(self.times,time))
        if i==len(self.times) or self.times[i]!=time:return None
        mu=self.mu[i,:3];amb=self.ambiguity[i,:3]
        if not np.isfinite(mu).all() or not np.isfinite(amb).all():return None
        current=np.asarray(qty)*prices/equity
        target=current.copy()
        proposed,info=optimize(mu,sigma[:3,:3],current[:3],self.gamma,self.horizon,self.cost,amb,
                              bound=min(protocol['risk']['asset_fraction'],protocol['risk']['asset_notional']/equity)*protocol['risk']['sizing_headroom'])
        target[:3]=current[:3]+self.partial*(proposed-current[:3])
        fresh=(volume>0)&(trades>0)
        increases=(target*current<0)|(abs(target)>abs(current))
        target=np.where(~fresh&increases,current,target)
        target=project(target,equity,sigma,protocol['risk'])
        change=abs(target-current).sum()
        if change>protocol['risk']['normal_turnover_fraction']:
            target=current+(target-current)*protocol['risk']['normal_turnover_fraction']/change
        costs=np.full(3,self.cost);q=self.gamma*self.horizon*sigma[:3,:3]
        if utility(target[:3],current[:3],mu,q,costs,costs+amb)<=utility(current[:3],current[:3],mu,q,costs,costs+amb)+1e-12:
            target=current.copy();info['reason']='no_trade_zone'
        else:info['reason']='positive_net_utility'
        qtarget=np.array([rounded(v,d) for v,d in zip(target*equity/prices,protocol['execution']['decimals'])])
        # Do not manufacture tiny rebalances by flooring an unchanged position.
        qtarget=np.where(abs(target-current)<1e-12,qty,qtarget)
        self.last_decision=int(time)
        return {'target_qty':qtarget.tolist(),'risk_forced':False,'defensive':False,
                'optimizer':info,'forecast':mu.tolist(),'ambiguity':amb.tolist(),'horizon':self.horizon}
