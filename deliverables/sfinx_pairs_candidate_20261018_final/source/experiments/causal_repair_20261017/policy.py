"""Sequential risk calibration and a baseline-anchored, cost-aware alpha sleeve."""
import numpy as np

from experiments.cost_horizon_20261016.policy import optimize,utility
from experiments.cost_horizon_20261016.research import SecondController
from replay_lab.common_comparison.engine import rounded,risk_breaches


def path_calibrate(evaluate,target,grid,tolerance=.02):
    """Select by sequential position-path risk, never by PNL or a flat proxy.

    A grid is used because fills and no-trade regions make risk nonmonotonic.
    evaluate must replay only mature pre-deployment observations.
    """
    if target<0 or not np.isfinite(target) or tolerance<0:raise ValueError('Invalid path risk budget')
    observations=[];selected=None
    for gamma in grid:
        if gamma<=0:raise ValueError('Positive gamma required')
        row=evaluate(float(gamma))
        risk=float(row['mean_expected_daily_volatility'])
        valid=bool(row['engineering_pass']) and np.isfinite(risk) and risk<=target*(1+tolerance)+1e-12
        observations.append({'gamma':float(gamma),**row,'admissible':bool(valid)})
        if valid:
            selected=float(gamma);break
    return {'gamma':selected,'target_path_risk':float(target),'tolerance':tolerance,'observations':observations,
            'criterion':'first admissible sequential risk; no return selection','fallback':selected is None}


class AnchoredController:
    def __init__(self,times,retention,mu,eligible,gammas,*,model_share=.25,partial=.5,cost_bps=19,horizon=72,saved=None):
        self.times=np.asarray(times);self.mu=np.asarray(mu);self.eligible=np.asarray(eligible,bool)
        self.gammas=np.asarray(gammas);self.model_share=model_share;self.partial=partial;self.cost=cost_bps/10000;self.horizon=horizon
        if not 0<=model_share<=1 or not 0<partial<=1:raise ValueError('Invalid adjustment')
        if self.mu.shape!=(len(times),3) or self.eligible.shape!=self.mu.shape or self.gammas.shape!=(len(times),):raise ValueError('Signal shape')
        self.base=SecondController(self.times,retention)
        if saved:self.base.controller.last_decision=int(saved['last_decision'])

    def snapshot(self):return {'last_decision':self.base.last_decision}

    def propose(self,time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs):
        action=self.base.propose(time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs)
        if action is None or action['risk_forced']:return action
        i=int(np.searchsorted(self.times,time));gamma=self.gammas[i]
        valid=self.eligible[i]&np.isfinite(self.mu[i])
        if not valid.any() or not np.isfinite(gamma) or gamma<=0:
            return {**action,'alpha_reason':'baseline_fallback','alpha_eligible':valid.tolist()}
        prices=np.asarray(prices);qty=np.asarray(qty);sigma=np.asarray(sigma)
        current=qty*prices/equity;baseqty=np.asarray(action['target_qty']);basew=baseqty*prices/equity
        active=np.flatnonzero(valid);inactive=np.setdiff1d(np.arange(len(qty)),active)
        h=self.horizon
        # Fixed baseline holdings in other markets enter cross-covariance risk.
        effective_mu=self.mu[i,active]-gamma*h*sigma[np.ix_(active,inactive)]@basew[inactive]
        model,_=optimize(effective_mu,sigma[np.ix_(active,active)],current[active],gamma,h,self.cost,
                         bound=min(protocol['risk']['asset_fraction'],protocol['risk']['asset_notional']/equity)*protocol['risk']['sizing_headroom'])
        proposed=basew.copy()
        desired=(1-self.model_share)*basew[active]+self.model_share*model
        proposed[active]=current[active]+self.partial*(desired-current[active])
        fresh=(np.asarray(volume)>0)&(np.asarray(trades)>0)
        increases=(proposed*current<0)|(abs(proposed)>abs(current))
        proposed=np.where(~fresh&increases,current,proposed)
        base_variance=float(basew@sigma@basew)

        def feasible(w):
            return (float(w@sigma@w)<=base_variance+1e-14
                    and not risk_breaches(w*equity/prices,equity,prices,sigma,protocol['risk'])
                    and abs(w-current).sum()<=max(abs(basew-current).sum(),protocol['risk']['normal_turnover_fraction'])+1e-12)
        if not feasible(proposed):
            lo,hi=0.,1.
            for _ in range(45):
                mid=(lo+hi)/2
                if feasible(basew+mid*(proposed-basew)):lo=mid
                else:hi=mid
            proposed=basew+lo*(proposed-basew)
        target=baseqty.copy()
        for j in active:
            target[j]=rounded(proposed[j]*equity/prices[j],protocol['execution']['decimals'][j])
        # Preserve lot-grid holdings exactly and reject economics AFTER rounding.
        target=np.where(abs(proposed-current)<1e-12,qty,target)
        w=target*prices/equity
        mu=np.zeros(len(qty));mu[active]=self.mu[i,active]
        c=np.full(len(qty),self.cost);q=gamma*h*sigma
        improvement=utility(w,current,mu,q,c,c)-utility(basew,current,mu,q,c,c)
        if not feasible(w) or improvement<=1e-12:
            return {**action,'alpha_reason':'no_incremental_utility_or_risk_room','alpha_eligible':valid.tolist()}
        if np.any(abs(target[inactive]-baseqty[inactive])>1e-10):raise AssertionError('Unrelated baseline market changed')
        # A reduced risk ceiling remains binding; alpha never restores its cut.
        capactive=np.array(action['risk_cap_active'],bool);ceiling=np.array(action['risk_cap_qty'])
        if np.any(abs(target[capactive])>ceiling[capactive]+1e-10):
            return {**action,'alpha_reason':'baseline_risk_ceiling','alpha_eligible':valid.tolist()}
        return {**action,'target_qty':target.tolist(),'alpha_reason':'positive_incremental_net_utility',
                'alpha_eligible':valid.tolist(),'alpha_utility_improvement':float(improvement),
                'baseline_expected_variance':base_variance,'candidate_expected_variance':float(w@sigma@w),
                'gamma':float(gamma)}
