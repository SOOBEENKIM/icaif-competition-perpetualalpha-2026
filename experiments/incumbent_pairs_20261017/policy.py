"""One account executor: incumbent targets plus an explicit paired lifecycle."""
import numpy as np
from itertools import product
from experiments.cost_horizon_20261016.research import SecondController
from experiments.pairs_forecast_20261017.policy import balanced_inventory
from replay_lab.common_comparison.engine import risk_breaches,project,rounded


def feasible_scale(other, units, ids, qty, prices, equity, sigma, risk):
    """Largest feasible pair size, including when closing everything is infeasible.

    All constraints are convex in one scalar. Turnover is the intersection of
    four signed linear constraints, so we never assume zero is feasible.
    """
    o=other*prices/equity;u=np.zeros_like(o);u[ids]=units*prices[ids]/equity
    current=qty*prices/equity;lo,hi=0.,1.;head=risk['sizing_headroom']
    def cut(a,b):
        nonlocal lo,hi
        if abs(a)<1e-15:
            if b < -1e-12:lo,hi=1.,0.
        elif a>0:hi=min(hi,b/a)
        else:lo=max(lo,b/a)
    cap=min(risk['asset_fraction'],risk['asset_notional']/equity)*head
    for a,b in zip(u,o):cut(a,cap-b);cut(-a,cap+b)
    cut(abs(u).sum(),risk['gross_fraction']*head-abs(o).sum())
    cut(u.sum(),risk['net_fraction']*head-o.sum())
    cut(-u.sum(),risk['net_fraction']*head+o.sum())
    other_turnover=abs(o-current).sum()-abs(current[ids]).sum()
    for signs in product((-1.,1.),repeat=len(ids)):
        v=np.array(signs)
        cut(v@u[ids],risk['normal_turnover_fraction']-other_turnover+v@current[ids])
    a=float(24*u@sigma@u);b=float(48*o@sigma@u)
    c=float(24*o@sigma@o-(risk['daily_volatility']*head)**2)
    if a<1e-18:cut(b,-c)
    else:
        disc=b*b-4*a*c
        if disc<0:return None
        lo=max(lo,(-b-np.sqrt(disc))/(2*a));hi=min(hi,(-b+np.sqrt(disc))/(2*a))
    return None if hi<lo-1e-12 or hi<0 or lo>1 else float(np.clip(hi,0,1))


def cap_pair(actual, ceiling, active):
    """A directional ceiling may reduce both legs; it cannot add exposure."""
    ratios=np.divide(ceiling,abs(actual),out=np.ones_like(actual),where=abs(actual)>1e-15)
    factor=min(1.,float(np.min(np.where(active,ratios,1.))))
    return actual*factor


class IncumbentPairController:
    def __init__(self,m,weights,retention,d,predictions,stamps,cfg,variant,scenario,last):
        self.m,self.weights,self.retention=m,weights,retention
        self.d,self.predictions,self.stamps=d,predictions,stamps
        self.cfg,self.variant,self.scenario,self.last=cfg,variant,scenario,last
        self.baseline=SecondController(m.times,retention)
        self.pairs=d['pairs'];self.states={};self.cooldown={};self.events=[]

    def propose(self,time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs):
        i=self.m.exact(time);s=self.cfg['strategy'];overlay=self.variant['mode']=='overlay'
        forced=bool(kwargs.get('halted') or risk_breaches(qty,equity,prices,sigma,protocol['risk']))
        ending=i>=self.last-2
        if overlay:
            action=self.baseline.propose(time,prices,qty,equity,self.weights[i],sigma,volume,trades,protocol,**kwargs)
            target=np.array(action['target_qty']) if action else qty.copy()
        else:target=qty.copy()
        if forced:
            target=np.zeros_like(qty) if kwargs.get('halted') else project(qty*prices/equity,equity,sigma,protocol['risk'])*equity/prices
            target=np.sign(qty)*np.minimum(abs(qty),abs(target))
            for state in self.states.values():state['phase']='closing'
            # Close both legs when risk has invalidated a paired position.
            for k in self.states:target[self.pairs[k]]=0.
        else:
            for k,state in list(self.states.items()):
                ids=self.pairs[k];actual=qty[ids]
                if not np.any(actual):
                    self.events.append(dict(time=time,pair=k,event='flat_confirmed'))
                    self.cooldown[k]=i+s['cooldown_hours'];del self.states[k]
                    target[ids]=0.;continue
                target[ids]=actual
                if state['phase'] in ('entering','balancing','holding'):
                    balanced,reduced=balanced_inventory(actual,state['planned'],s['balanced_fill_tolerance'])
                    if balanced and state['phase']!='holding':
                        state.update(phase='holding',entry_prices=prices[ids].copy())
                        self.events.append(dict(time=time,pair=k,event='balanced_open',qty=actual.tolist()))
                    elif not balanced and i-state['entry']<=s['maximum_balance_hours'] and np.all(reduced):
                        state['phase']='balancing';target[ids]=reduced
                        self.events.append(dict(time=time,pair=k,event='reduce_imbalance',qty=actual.tolist(),target=reduced.tolist()))
                        continue
                    elif not balanced:
                        state['phase']='closing';self.events.append(dict(time=time,pair=k,event='unhedged_abort'))
                reason=None
                if ending:reason='evaluation_end'
                elif state['phase']=='closing':reason='closing_retry'
                elif i+1-state['entry']>=self.variant['horizon']:reason='planned_horizon'
                elif state['phase']=='holding':
                    value=float(actual@(prices[ids]-state['entry_prices']))/max(float(abs(actual)@state['entry_prices']),1e-12)
                    if value < -s['stop_sigma']*state['vol']*np.sqrt(self.variant['horizon']):reason='spread_stop'
                if reason:
                    if state['phase']!='closing':self.events.append(dict(time=time,pair=k,event='exit_intent',reason=reason))
                    state['phase']='closing';target[ids]=0.
                elif state['phase']=='holding':
                    ceiling=abs(self.weights[i,ids])*equity/prices[ids]
                    capped=cap_pair(actual,ceiling,self.retention[i,ids]<1-1e-12)
                    if np.any(abs(capped)<abs(actual)-1e-12):
                        target[ids]=capped
                        self.events.append(dict(time=time,pair=k,event='risk_cap_pair_reduction',target=capped.tolist()))
        if not forced and not ending and time%21600==0 and not self.states:
            choices=[]
            cost=2*(protocol['execution']['fee_bps']+self.scenario['impact_bps'])/10000
            for k,ids in enumerate(self.pairs):
                if self.cooldown.get(k,0)>i:continue
                wa,wb=self.weights[i,ids]
                if wa*wb>=0 or np.any(volume[ids]<=0) or np.any(trades[ids]<=0):continue
                correlation=self.d['x'][i,k,self.d['names'].index('correlation_168')]
                if correlation<s['minimum_correlation']:continue
                direction=int(np.sign(wa));mu=self.predictions[i,k]
                if self.variant['model']=='calibrated':
                    if not np.isfinite(mu) or direction*mu<=cost+s['entry_buffer_bps']/10000:continue
                    score=direction*mu-cost
                else:
                    score=min(abs(wa)*np.sqrt(sigma[ids[0],ids[0]]),abs(wb)*np.sqrt(sigma[ids[1],ids[1]]))
                choices.append((score,k,direction))
            for score,k,direction in sorted(choices,reverse=True):
                ids=self.pairs[k];beta=self.d['beta'][i,k]
                gross=min(float(abs(self.weights[i,ids]).sum()),s['pair_gross_cap'])
                pair_weights=direction*np.array([1.,-beta])/(1+beta)*gross
                units=pair_weights*equity/prices[ids]
                # Reference-only capacity, unchanged across comparator models.
                factor=1.
                for j,u in zip(ids,units):
                    factor=min(factor,.9*self.scenario['participation']*volume[j]/max(abs(u),1e-12))
                units*=factor
                units=cap_pair(units,abs(self.weights[i,ids])*equity/prices[ids],self.retention[i,ids]<1-1e-12)
                # Preserve aggregate normal turnover and risk limits. The other
                # incumbent positions are already in target and cannot be ignored.
                other=target.copy();other[ids]=0.
                factor=feasible_scale(other,units,ids,qty,prices,equity,sigma,protocol['risk'])
                if factor is None:continue
                units*=factor
                units=np.array([rounded(u,protocol['execution']['decimals'][j]) for j,u in zip(ids,units)])
                if np.any(abs(units)*prices[ids]<protocol['execution']['minimum_order_value']):continue
                trial=other.copy();trial[ids]=units
                if risk_breaches(trial,equity,prices,sigma,protocol['risk']):continue
                if abs((trial-qty)*prices/equity).sum()>protocol['risk']['normal_turnover_fraction']+1e-10:continue
                target[ids]=units
                self.states[k]=dict(phase='entering',planned=units.copy(),entry=i+1,vol=self.d['vol'][i,k])
                self.events.append(dict(time=time,pair=k,event='entry_intent',direction=direction,beta=float(beta),
                    target=units.tolist(),gross_budget=gross,ranking_score=float(score),
                    forecast=float(self.predictions[i,k]) if np.isfinite(self.predictions[i,k]) else None,
                    fit_time=int(self.stamps[i,k]) if self.variant['model']=='calibrated' else None))
                break
        if ending:
            # Both baselines are reported with residual-exit costs separately;
            # paired positions must actually clear in the candidate ledger.
            for k in self.states:target[self.pairs[k]]=0.
        if risk_breaches(target,equity,prices,sigma,protocol['risk']):
            # Uniform reduction preserves every pair ratio. Per-asset clipping
            # would silently break a hedge already accepted as balanced.
            lo,hi=0.,1.
            for _ in range(32):
                mid=(lo+hi)/2
                if risk_breaches(target*mid,equity,prices,sigma,protocol['risk']):hi=mid
                else:lo=mid
            target*=lo*protocol['risk']['sizing_headroom']
        target=np.array([rounded(q,d) for q,d in zip(target,protocol['execution']['decimals'])])
        if np.allclose(target,qty,rtol=0,atol=1e-12):return None
        return dict(target_qty=target.tolist(),risk_forced=forced,defensive=forced)
