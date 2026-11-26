"""Hourly paired lifecycle manager; actual inventory, not signal-only positions."""
import numpy as np
from replay_lab.common_comparison.engine import risk_breaches,rounded
from .models import conditional,segmented_area,stopping_tables


class PairsController:
    def __init__(self,m,formations,cfg,family,scenario,last):
        self.m=m;self.formations=formations;self.cfg=cfg;self.family=family
        self.scenario=scenario;self.last=last;self.states={};self.cooldown={};self.events=[];self.tables={}

    def signal(self,f,i,direction=0,held=0):
        s=self.cfg['strategy'];prices=self.m.close[i];a,b=f['assets']
        cost=(9+self.scenario['impact_bps'])/10000
        if f['kind']=='reference':
            residual=np.array([prices[f['reference']]-leg['alpha']-leg['beta']*prices[j] for leg,j in zip(f['legs'],(a,b))])
            h=conditional(f['copula'],residual)
            d=1 if h[0]<s['copula_entry'] and h[1]>1-s['copula_entry'] else -1 if h[1]<s['copula_entry'] and h[0]>1-s['copula_entry'] else 0
            units=np.array([-f['legs'][0]['beta'],f['legs'][1]['beta']]);gross=float(np.sum(abs(units)*prices[[a,b]]))
            rhos=np.array([leg['rho'] for leg in f['legs']]);means=np.array([leg['mean'] for leg in f['legs']])
            future=(means-residual)*(1-rhos**s['max_hold_hours'])
            edge=float(d*(future[0]-future[1])/gross)
            w=units*prices[[a,b]]/gross
            rate=float(w@np.mean(self.m.funding[max(0,i-168):i,[a,b]],axis=0))
            reserve=max(0.,d*rate)*s['max_hold_hours']
            allowed=d if edge>2*cost+reserve+s['entry_buffer_bps']/10000 else 0
            hit=(direction==1 and (h[0]>=.5 or h[1]<=.5)) or (direction==-1 and (h[0]<=.5 or h[1]>=.5))
            stop=any(abs((r-leg['mean'])/leg['sd'])>=s['stop_z'] for r,leg in zip(residual,f['legs']))
            return dict(direction=allowed,exit=bool(hit or stop or held>=s['max_hold_hours']),
                reason='reference_convergence' if hit else 'stop_or_timeout',edge=edge,reserve=reserve,
                weights=w,h=h.tolist(),stop=bool(stop),cost=2*cost)
        residual=np.log(prices[a])-f['alpha']-f['beta']*np.log(prices[b])-f['mean']
        z=residual/f['sd'];d=-int(np.sign(z));w=np.array([1.,-f['beta']])/(1+f['beta'])
        reserve=max(0.,d*f['relative_funding'])*s['max_hold_hours']
        edge=abs(residual)*(1-f['rho']**s['max_hold_hours'])/(1+f['beta'])
        allowed=d if s['entry_z']<=abs(z)<s['stop_z'] and edge>2*cost+reserve+s['entry_buffer_bps']/10000 else 0
        reason='convergence';close=bool(direction and direction*z>=-s['exit_z'])
        if self.family=='signature':
            path=np.log(self.m.close[i-s['signature_hours']+1:i+1,[a,b]])
            area=segmented_area(path);same_move=np.prod(path[-1]-path[0])>0
            if area>=f['signature_threshold'] or not same_move:allowed=0
        if self.family=='stopping':
            key=(f['key'],f['formation_index'])
            if key not in self.tables:
                carry=[max(0.,v*f['relative_funding']) for v in (-1,1)]
                self.tables[key]=stopping_tables(f,cost,s['max_hold_hours'],carry)
            table=self.tables[key];horizon=max(1,s['max_hold_hours']-int(held));grid=table['grid']
            # Signal is acted on at the next hour, not at this observed close.
            expected_z=f['rho']*z
            gains=[float(np.interp(expected_z,grid,table['entry'][s['max_hold_hours'],k])) for k in (0,1)]
            candidate=(-1,1)[int(np.argmax(gains))]
            waiting=float(np.interp(expected_z,grid,table['transition']@table['flat'][s['max_hold_hours']-1]))
            allowed=candidate if (candidate==d and 1<=abs(z)<s['stop_z'] and max(gains)>waiting+1e-8
                                  and max(gains)>s['entry_buffer_bps']/10000) else 0
            edge=max(gains)+2*cost
            if direction:
                k=0 if direction==-1 else 1
                continuation=float(np.interp(expected_z,grid,table['continuation'][horizon,k]))
                close=continuation<=-cost+1e-10;reason='optimal_liquidation'
        stop=bool(abs(z)>=s['stop_z'] or held>=s['max_hold_hours'])
        return dict(direction=allowed,exit=bool(close or stop),reason='stop_or_timeout' if stop else reason,
                    edge=float(edge),reserve=float(reserve),weights=w,z=float(z),stop=stop,cost=2*cost)

    def propose(self,time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs):
        i=self.m.exact(time);s=self.cfg['strategy'];target=qty.copy()
        forced=bool(kwargs.get('halted') or risk_breaches(qty,equity,prices,sigma,protocol['risk']))
        ending=i>=self.last-2
        used=set()
        for key,state in list(self.states.items()):
            ids=state['model']['assets'];actual=qty[ids]
            if state['pending']:
                state['pending']=False
                if not np.allclose(actual,state['target'],rtol=0,atol=1e-9):
                    state['closing']=True
                    self.events.append(dict(time=time,key=key,event='partial_entry_abort',filled=actual.tolist(),requested=state['target'].tolist()))
                else:self.events.append(dict(time=time,key=key,event='both_legs_filled'))
            if not np.any(actual):
                self.events.append(dict(time=time,key=key,event='flat_confirmed'))
                self.cooldown[key]=i+s['cooldown_hours'];del self.states[key];continue
            used.update(ids)
            sig=self.signal(state['model'],i,state['direction'],i-state['entered'])
            if forced or ending or state['closing'] or sig['exit']:
                if not state['closing']:
                    self.events.append(dict(time=time,key=key,event='exit_intent',reason='account_risk' if forced else 'period_end' if ending else sig['reason']))
                state['closing']=True;target[ids]=0.
            # Holding preserves quantities; a new fit cannot manufacture gains
            # by moving the spread anchor or silently rehedging an open position.
        if not forced and not ending:
            available=[f for f in self.formations if f['index']<=i]
            latest=available[-1] if available else None
            models=latest['copulas'] if latest and self.family=='reference_copula' else latest['models'] if latest else []
            candidates=[]
            for f in models:
                if f['key'] in self.states or self.cooldown.get(f['key'],0)>i:continue
                sig=self.signal(f,i)
                if sig['direction']:candidates.append((sig['edge']-sig['cost']-sig['reserve'],f,sig))
            for _,f,sig in sorted(candidates,key=lambda z:(-z[0],z[1]['key'])):
                ids=f['assets']
                if len(self.states)>=s['maximum_pairs'] or used.intersection(ids):continue
                if np.any(volume[ids]<=0) or np.any(trades[ids]<=0):continue
                w=sig['direction']*sig['weights']*s['budget_per_pair']
                # Scale both legs together to observed depth, risk and turnover.
                fac=1.;r=protocol['risk'];head=r['sizing_headroom']
                cap=min(r['asset_fraction'],r['asset_notional']/equity)*head
                fac=min(fac,cap/max(abs(w)))
                for j,weight in zip(ids,w):
                    fac=min(fac,.9*self.scenario['participation']*volume[j]*prices[j]/max(abs(weight)*equity,1e-12))
                room=max(0.,r['normal_turnover_fraction']-float(np.sum(abs((target-qty)*prices/equity))))
                fac=min(fac,room/max(abs(w).sum(),1e-12))
                trial=target.copy();units=w*fac*equity/prices[ids]
                # Global caps can shrink only the new pair, preserving its hedge.
                lo,hi=0.,1.
                for _ in range(30):
                    mid=(lo+hi)/2;trial[ids]=units*mid
                    if risk_breaches(trial,equity,prices,sigma,r):hi=mid
                    else:lo=mid
                units*=lo
                units=np.array([rounded(v,protocol['execution']['decimals'][j]) for v,j in zip(units,ids)])
                if np.any(abs(units)*prices[ids]<protocol['execution']['minimum_order_value']):continue
                target[ids]=units;used.update(ids)
                self.states[f['key']]=dict(model=f,direction=sig['direction'],entered=i+1,pending=True,closing=False,target=units.copy())
                self.events.append(dict(time=time,key=f['key'],event='entry_intent',direction=sig['direction'],
                    target=units.tolist(),trained_through=f['trained_through'],edge=sig['edge'],cost=sig['cost'],reserve=sig['reserve'],z=sig.get('z'),h=sig.get('h')))
        if np.array_equal(qty,target):return None
        return dict(target_qty=target.tolist(),risk_forced=forced,defensive=forced)
