"""Causal pair state machine over the existing conserved shared-account ledger.

Signals and quantities are sealed on a completed close. A one-hour-later IOC
simulation may reduce their sizes; it can never use future data to enlarge them.
"""
from dataclasses import dataclass
import copy
import numpy as np
import pandas as pd
from ..common_comparison.engine import Ledger, execute_intent, rounded, risk_values, risk_breaches
from ..common_comparison.models import covariance


class PairFeatures:
    def __init__(self, market, assets, protocol):
        self.assets=tuple(assets)
        a,b=self.assets; self.x=np.log(market.close[:,a]); self.y=np.log(market.close[:,b])
        spec=protocol['strategy']; n=spec['beta_hours']; w=spec['z_hours']
        x,y=pd.Series(self.x),pd.Series(self.y)
        self.beta=(x.rolling(n).cov(y,ddof=0)/y.rolling(n).var(ddof=0)).to_numpy()
        self.mx=x.rolling(w).mean().to_numpy(); self.my=y.rolling(w).mean().to_numpy()
        self.vx=x.rolling(w).var(ddof=0).to_numpy(); self.vy=y.rolling(w).var(ddof=0).to_numpy()
        self.xy=x.rolling(w).cov(y,ddof=0).to_numpy()

    def values(self,index,beta=None,sigma=None):
        beta=float(self.beta[index]) if beta is None else float(beta)
        if sigma is None:
            sigma=float(np.sqrt(max(1e-16,self.vx[index]+beta*beta*self.vy[index]-2*beta*self.xy[index])))
        z=float(((self.x[index]-self.mx[index])-beta*(self.y[index]-self.my[index]))/sigma)
        return beta,float(sigma),z


@dataclass
class PairState:
    pair: str
    assets: tuple
    direction: int=0
    beta: float=0.0
    sigma: float=0.0
    entered: int=0
    lock_sign: int=0
    closing: bool=False
    entry_pending: bool=False


class PairSimulation:
    def __init__(self,market,formations,protocol,start_index,end_index,scenario='base',record=True,features=None,sigmas=None):
        self.market=market; self.protocol=protocol; self.scenario_name=scenario
        self.scenario=protocol['scenarios'][scenario]; self.risk=protocol['risk']
        self.first=int(start_index); self.last=int(end_index); self.i=self.first; self.record=record
        self.formations=formations
        self.features=features if features is not None else self.make_features(market,formations,protocol)
        self.sigmas=sigmas if sigmas is not None else self.make_sigmas(market,protocol)
        self.ledger=Ledger(protocol['initial'],market.symbols)
        self.states={}; self.orders=[]; self.decisions=[]; self.snapshots=[]; self.days={}; self.breaches=[]
        self.events=[]; self.opportunities=0; self.partial_pairs=0
        self.peak=self.ledger.initial; self.drawdown=0.0; self.halted=False
        self.streak=self.maximum_streak=self.observed_hours=self.missing_hours=0
        self.previous_equity=self.ledger.initial
        self._snapshot()

    @staticmethod
    def make_features(market,formations,protocol):
        assets={r['pair']:r['assets'] for f in formations for r in f['rows'] if r['pair'] in f['selected']}
        return {p:PairFeatures(market,a,protocol) for p,a in assets.items()}

    @staticmethod
    def make_sigmas(market,protocol):
        result=np.zeros((len(market.times),len(market.symbols),len(market.symbols)))
        for i in range(168,len(result)):
            result[i]=covariance(market,i,protocol['risk'])[1]
        return result

    def active_pairs(self):
        available=[f for f in self.formations if f['close_index']<=self.i]
        return list(available[-1]['selected']) if available else []

    def _sync_states(self):
        active=self.active_pairs()
        for pair in active:
            if pair not in self.states:
                self.states[pair]=PairState(pair,self.features[pair].assets)
        for pair,s in self.states.items():
            held=bool(s.direction and np.any(self.ledger.qty[list(s.assets)]!=0))
            if not held and not s.entry_pending:
                s.direction=0; s.closing=False
            if pair not in active and held:
                s.closing=True
        return active

    def context(self,pair):
        s=self.states[pair]; f=self.features[pair]; spec=self.protocol['strategy']
        beta,sigma,z=f.values(self.i,s.beta if s.direction else None,s.sigma if s.direction else None)
        _,_,previous=f.values(self.i-1)
        held_hours=max(0,(int(self.market.times[self.i])-s.entered)/3600) if s.direction else 0.0
        if s.lock_sign and z*s.lock_sign<=0:
            s.lock_sign=0
        boundary=spec['stop_z']*min(1.,max(0.,(spec['max_hold_hours']-held_hours)/
            (spec['max_hold_hours']-spec['stop_decay_start_hours'])))
        stop=bool(s.direction and (-s.direction*z>=boundary or held_hours>=spec['max_hold_hours']))
        mean_exit=bool(s.direction and s.direction*z>=spec['exit_z'])
        cost=(2*(self.protocol['execution']['fee_bps']+self.scenario['impact_bps'])+
              spec['entry_cost_buffer_bps'])/10000
        edge=abs(z)*sigma/(1+beta) if beta>0 else 0.0
        fresh=bool(np.all(self.market.volume[self.i,list(s.assets)]>0) and
                   np.all(self.market.trades[self.i,list(s.assets)]>0))
        signal=1 if previous>-spec['entry_z']>=z else (-1 if previous<spec['entry_z']<=z else 0)
        valid=(self.protocol['screen']['beta_min']<=beta<=self.protocol['screen']['beta_max'] and
               abs(z)<spec['stop_z'] and edge>cost and fresh and not s.lock_sign and not self.halted)
        allowed=int(signal if valid and not s.direction and not s.closing else 0)
        obs=np.array([np.clip(z/6,-5,5),s.direction,min(held_hours/168,2),allowed,
                      np.clip(edge/max(cost,1e-9),0,5)/5],dtype=np.float32)
        return dict(beta=beta,sigma=sigma,z=z,previous_z=previous,stop=stop,mean_exit=mean_exit,
            allowed=allowed,held_hours=held_hours,edge=edge,cost=cost,observation=obs)

    def observations(self):
        active=self._sync_states()
        return {pair:self.context(pair)['observation'] for pair in sorted(set(active)|
            {p for p,s in self.states.items() if s.direction})}

    def rule_actions(self):
        self._sync_states(); result={}
        for p,s in self.states.items():
            c=self.context(p)
            result[p]=0 if s.closing or c['stop'] or c['mean_exit'] else (s.direction or c['allowed'])
        return result

    def _target(self,actions):
        active=self._sync_states(); qty=self.ledger.qty.copy(); price=self.market.close[self.i]
        equity=self.ledger.equity(price); sigma=self.sigmas[self.i]
        opening=[]; spec=self.protocol['strategy']
        forced=self.halted or bool(risk_breaches(qty,equity,price,sigma,self.risk))
        for pair,s in self.states.items():
            if pair not in active and not s.direction:
                continue
            c=self.context(pair); a,b=s.assets; held=bool(s.direction and np.any(qty[[a,b]]!=0))
            action=int(actions.get(pair,0))
            if action not in (-1,0,1):
                raise ValueError('Invalid action')
            if c['allowed']:
                self.opportunities+=1
            if held:
                if forced or s.closing or c['stop'] or c['mean_exit'] or action!=s.direction:
                    qty[[a,b]]=0; s.closing=True
                    if c['stop']:
                        s.lock_sign=int(np.sign(c['z']))
                    if self.record:
                        self.events.append(dict(time=int(self.market.times[self.i]),pair=pair,
                            event='exit_intent',reason='risk' if forced else 'stop' if c['stop'] else
                            'mean' if c['mean_exit'] else 'formation_or_policy',z=c['z']))
                continue
            if forced or pair not in active or not c['allowed'] or action!=c['allowed']:
                continue
            # Existing residual legs from an expiring pair must be flat first.
            if any(other.direction and np.any(self.ledger.qty[list(other.assets)]!=0) and set(other.assets)&set(s.assets)
                   for name,other in self.states.items() if name!=pair):
                continue
            gross=spec['budget_per_pair']*self.risk['sizing_headroom']
            weights=np.zeros(len(qty)); weights[a]=action*gross/(1+c['beta']); weights[b]=-action*gross*c['beta']/(1+c['beta'])
            # Joint scaling preserves the hedge. Never clip the legs independently.
            factor=1.
            per_asset=min(self.risk['asset_fraction'],self.risk['asset_notional']/equity)*self.risk['sizing_headroom']
            factor=min(factor,per_asset/max(abs(weights)))
            turn_room=max(0.,self.risk['normal_turnover_fraction']-abs((qty-self.ledger.qty)*price/equity).sum())
            factor=min(factor,turn_room/max(abs(weights).sum(),1e-12))
            # Do not knowingly request more than either observed leg can fill.
            # Future execution volume is used only by the broker, never here.
            for j in (a,b):
                raw_units=abs(weights[j])*equity/price[j]
                known_capacity=.9*self.scenario['participation']*self.market.volume[self.i,j]
                factor=min(factor,known_capacity/max(raw_units,1e-12))
            target=weights*factor*equity/price
            for j in (a,b):
                target[j]=rounded(target[j],self.protocol['execution']['decimals'][j])
            if any(abs(target[j])*price[j]<self.protocol['execution']['minimum_order_value'] for j in (a,b)):
                continue
            candidate=qty.copy(); candidate[[a,b]]=target[[a,b]]
            if risk_breaches(candidate,equity,price,sigma,self.risk):
                continue
            qty=candidate; s.direction=action; s.beta=c['beta']; s.sigma=c['sigma']
            s.entered=int(self.market.times[self.i])+3600; s.entry_pending=True
            opening.append(pair)
            if self.record:
                self.events.append(dict(time=int(self.market.times[self.i]),pair=pair,event='entry_intent',
                    direction=action,beta=c['beta'],sigma=c['sigma'],z=c['z'],edge=c['edge'],cost=c['cost']))
        return qty,opening,forced

    def _snapshot(self):
        t=int(self.market.times[self.i]); prices=self.market.close[self.i]
        equity=self.ledger.reconcile(prices)
        self.peak=max(self.peak,equity); self.drawdown=min(self.drawdown,equity/self.peak-1)
        self.halted|=equity/self.peak-1<=-self.risk['drawdown_halt']
        violations=risk_breaches(self.ledger.qty,equity,prices,self.sigmas[self.i],self.risk)
        self.streak=self.streak+1 if violations else 0; self.maximum_streak=max(self.maximum_streak,self.streak)
        if violations and self.record:
            self.breaches.append(dict(time=t,constraints=violations))
        if self.i==self.first and t%86400==0:
            self.days[t]=equity
        if self.record:
            self.snapshots.append(dict(time=t,equity=equity,cash=self.ledger.cash,qty=self.ledger.qty.tolist(),
                realized=self.ledger.realized,unrealized=equity-self.ledger.cash,fees=self.ledger.fees,
                funding=self.ledger.funding,**risk_values(self.ledger.qty,equity,prices,self.sigmas[self.i])))
        return equity

    def advance(self,actions):
        if self.i>=self.last:
            raise RuntimeError('Episode already finished')
        old_equity=self.ledger.equity(self.market.close[self.i])
        target,opening,forced=self._target(actions)
        t=int(self.market.times[self.i]); source=self.i
        intent=dict(decision_time=t,execution_time=t+3600,source_close=t,source_available=t,
            target_qty=target.tolist(),source_volume=self.market.volume[self.i].copy(),
            source_trades=self.market.trades[self.i].copy(),sigma=self.sigmas[self.i],risk_forced=forced)
        changed=not np.array_equal(target,self.ledger.qty)
        if changed and self.record:
            self.decisions.append(dict(decision_time=t,execution_time=t+3600,source_available=t,
                target_qty=target.tolist(),position_before=self.ledger.qty.tolist()))
        self.i+=1; time=int(self.market.times[self.i]); prices=self.market.close[self.i]
        if time!=t+3600:
            raise ValueError('Missing hour')
        if time%86400==0:
            self.days[time]=self.ledger.equity(prices)
        held=self.ledger.qty!=0
        self.observed_hours+=int(np.sum(held & self.market.observed[self.i]))
        self.missing_hours+=int(np.sum(held & ~self.market.observed[self.i]))
        self.ledger.charge_funding(time,prices,self.market.funding[self.i],self.market.observed[self.i],
                                   self.scenario['adverse_funding_bps_hour'])
        if changed and self.i<self.last:
            intent['sigma']=self.sigmas[self.i]
            orders=execute_intent(self.ledger,intent,self.market,self.i,self.protocol,self.scenario)
            if self.record:
                self.orders.extend(orders)
            for pair in opening:
                s=self.states[pair]; ids=list(s.assets); s.entry_pending=False
                complete=np.allclose(self.ledger.qty[ids],target[ids],rtol=0,atol=1e-9)
                if not complete and np.any(self.ledger.qty[ids]):
                    self.partial_pairs+=1; s.closing=True
                    if self.record:
                        self.events.append(dict(time=time,pair=pair,event='incomplete_pair_flatten_next_hour',
                            target=target[ids].tolist(),filled=self.ledger.qty[ids].tolist()))
        else:
            for pair in opening:
                self.states[pair].entry_pending=False
        new_equity=self._snapshot()
        reward=(new_equity/old_equity-1)*10000
        if reward<0:
            reward*=1.2
        if not self.record:
            self.ledger.fills.clear(); self.ledger.funding_events.clear()
        return float(reward),self.i>=self.last

    def result(self,name):
        ledger=self.ledger; equity=ledger.equity(self.market.close[self.i])
        daily=[dict(time=t,return_=self.days[t]/self.days[t-86400]-1) for t in sorted(self.days) if t-86400 in self.days]
        metrics=dict(roi=equity/ledger.initial-1,equity=equity,daily_vol=float(np.std([x['return_'] for x in daily],ddof=0)) if daily else None,
            complete_days=len(daily),max_drawdown=self.drawdown,turnover=ledger.turnover,fees=ledger.fees,
            funding=ledger.funding,realized=ledger.realized,unrealized=equity-ledger.cash,
            fills=len(ledger.fills),opportunities=self.opportunities,partial_pair_entries=self.partial_pairs,
            partial_or_unfilled_orders=sum(x['status']!='FILLED' for x in self.orders),
            observed_position_hours=self.observed_hours,missing_position_hours=self.missing_hours,
            maximum_ledger_error=ledger.max_error,maximum_breach_streak=self.maximum_streak,
            halted=self.halted,final_qty=ledger.qty.tolist(),
            mean_gross=float(np.mean([x['gross'] for x in self.snapshots])),
            max_gross=max(x['gross'] for x in self.snapshots))
        gates=dict(accounting=ledger.max_error<1e-6,
            information_time=all(f['source_available']<=f['decision_time']<f['time'] for f in ledger.fills),
            risk=self.maximum_streak<2,not_halted=not self.halted)
        return dict(policy=name,kind='pairs_ppo',parameter=0,scenario=self.scenario_name,
            start=int(self.market.times[self.first]),end=int(self.market.times[self.i]),metrics=metrics,gates=gates,
            snapshots=self.snapshots,decisions=self.decisions,orders=self.orders,fills=ledger.fills,
            funding_events=ledger.funding_events,daily=daily,events=self.events,risk_breaches=self.breaches,
            terminal_valuation='Mainnet close proxy; no forced liquidation',live_orders_sent=0)


def simulate_rules(market,formations,protocol,start,end,scenario='base',features=None,sigmas=None):
    sim=PairSimulation(market,formations,protocol,market.exact(start),market.exact(end),scenario,
                       features=features,sigmas=sigmas)
    while sim.i<sim.last:
        sim.advance(sim.rule_actions())
    return sim.result('rules')
