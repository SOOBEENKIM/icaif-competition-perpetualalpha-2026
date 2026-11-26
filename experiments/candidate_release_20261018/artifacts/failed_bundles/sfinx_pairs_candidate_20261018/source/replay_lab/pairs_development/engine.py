"""Pair opportunity allocation; execution remains the audited shared ledger.

Every eligible pair can request entry, but an asset may belong to only one open
pair. No synthetic netting, borrowed collateral, free partial-fill rollback, or
same-close execution is introduced.
"""
import numpy as np
import pandas as pd
from ..pairs_rl.engine import PairSimulation, PairFeatures


class DevelopmentFeatures(PairFeatures):
    def __init__(self,market,assets,protocol):
        super().__init__(market,assets,protocol)
        mode=protocol.get('development',{}).get('beta_mode','levels')
        if mode=='unit':
            self.beta=np.ones(len(market.times))
        elif mode=='returns_shrink':
            x=pd.Series(self.x).diff();y=pd.Series(self.y).diff();n=protocol['strategy']['beta_hours']
            estimate=x.rolling(n).cov(y,ddof=0)/y.rolling(n).var(ddof=0)
            self.beta=(.5+.5*estimate).to_numpy()
        elif mode!='levels':
            raise ValueError(mode)


class CandidateSimulation(PairSimulation):
    def __init__(self,*args,**kwargs):
        self.anchors={};self.last_flat={};self.close_reasons={}
        super().__init__(*args,**kwargs)
        self.development=self.protocol['development']
        # Only settled funding at or before the decision can affect this hurdle.
        rates=np.where(self.market.observed,self.market.funding,np.nan)
        self.trailing_funding=pd.DataFrame(rates).rolling(24,min_periods=24).mean().to_numpy()

    @staticmethod
    def make_features(market,formations,protocol):
        assets={r['pair']:r['assets'] for f in formations for r in f['rows'] if r['pair'] in f['selected']}
        return {p:DevelopmentFeatures(market,a,protocol) for p,a in assets.items()}

    def context(self,pair):
        c=super().context(pair);s=self.states[pair];spec=self.protocol['strategy']
        if self.development['fixed_mean'] and s.direction and pair in self.anchors:
            entered,mean=self.anchors[pair]
            if entered==s.entered:
                f=self.features[pair]
                c['z']=float((f.x[self.i]-s.beta*f.y[self.i]-mean)/s.sigma)
                boundary=spec['stop_z']*min(1.,max(0.,(spec['max_hold_hours']-c['held_hours'])/
                    (spec['max_hold_hours']-spec['stop_decay_start_hours'])))
                c['stop']=bool(-s.direction*c['z']>=boundary or c['held_hours']>=spec['max_hold_hours'])
                c['mean_exit']=bool(s.direction*c['z']>=spec['exit_z'])
        z=c['z'];beta=c['beta'];sigma=c['sigma']
        direction=1 if z<=-spec['entry_z'] else -1 if z>=spec['entry_z'] else 0
        if self.development['entry_mode']=='cross':
            direction=1 if c['previous_z']>-spec['entry_z']>=z else -1 if c['previous_z']<spec['entry_z']<=z else 0
        elif self.development['entry_mode'] in ('confirmed','inward'):
            f=self.features[pair]
            spread_change=(f.x[self.i]-f.x[self.i-1])-beta*(f.y[self.i]-f.y[self.i-1])
            # Attribute confirmation to traded prices, never to re-fitting the
            # hedge or to a moving reference mean.
            previous_same_reference=z-spread_change/max(sigma,1e-12)
            if self.development['entry_mode']=='inward':
                direction=1 if previous_same_reference<-spec['entry_z']<=z<=-.5 else \
                         -1 if previous_same_reference>spec['entry_z']>=z>=.5 else 0
            if direction*spread_change<=0:
                direction=0
        carry=0.
        if self.development['funding_hurdle'] and direction:
            a,b=s.assets;fa,fb=self.trailing_funding[self.i,[a,b]]
            if not np.isfinite([fa,fb]).all():
                carry=float('inf')
            elif beta>0:
                # Ignore forecast carry credits; reserve the full holding cap.
                carry=(max(0.,direction*(fa-beta*fb)/(1+beta))+
                       self.scenario['adverse_funding_bps_hour']/10000)*spec['max_hold_hours']
        c['carry_reserve']=carry;c['cost']+=carry
        c['edge']=abs(z)*sigma/(1+beta) if beta>0 else 0.
        ids=list(s.assets)
        fresh=bool(np.all(self.market.volume[self.i,ids]>0) and np.all(self.market.trades[self.i,ids]>0))
        cooldown=self.development['cooldown_hours']
        ready=self.i-self.last_flat.get(pair,-1000000)>=cooldown
        valid=(self.protocol['screen']['beta_min']<=beta<=self.protocol['screen']['beta_max'] and
               abs(z)<spec['stop_z'] and c['edge']>c['cost'] and fresh and ready and
               not s.lock_sign and not self.halted and not s.direction and not s.closing)
        c['allowed']=int(direction if valid else 0)
        c['observation']=np.array([np.clip(z/6,-5,5),s.direction,min(c['held_hours']/168,2),c['allowed'],
                                 np.clip(c['edge']/max(c['cost'],1e-9),0,5)/5],dtype=np.float32)
        return c

    def _target(self,actions):
        active=self._sync_states();approved={};used=set();candidates=[]
        for pair,s in self.states.items():
            held=s.direction and np.any(self.ledger.qty[list(s.assets)]!=0)
            if held:
                used.update(s.assets);approved[pair]=int(actions.get(pair,0))
            elif pair in active:
                c=self.context(pair);action=int(actions.get(pair,0))
                if c['allowed'] and action==c['allowed']:
                    candidates.append((-(c['edge']-c['cost']),pair,action))
        # Reserve the shared asset immediately, before any delayed fills. This
        # also prevents two newly opened pairs from overwriting each other's leg.
        for _,pair,action in sorted(candidates):
            s=self.states[pair]
            if not used.intersection(s.assets):
                approved[pair]=action;used.update(s.assets)
        # State iteration now matches allocation priority rather than CSV order.
        priority=[p for _,p,_ in sorted(candidates)]
        self.states=dict(sorted(self.states.items(),key=lambda kv:(
            0 if kv[1].direction else 1,priority.index(kv[0]) if kv[0] in priority else len(priority),kv[0])))
        target,opening,forced=super()._target(approved)
        if self.development['fixed_mean']:
            for pair in opening:
                s=self.states[pair];f=self.features[pair]
                z=f.values(self.i,s.beta,s.sigma)[2]
                mean=f.x[self.i]-s.beta*f.y[self.i]-z*s.sigma
                self.anchors[pair]=(s.entered,float(mean))
        return target,opening,forced

    def advance(self,actions):
        held={p:bool(s.direction and np.any(self.ledger.qty[list(s.assets)])) for p,s in self.states.items()}
        n=len(self.events)
        value=super().advance(actions)
        for pair,s in self.states.items():
            if s.entered==int(self.market.times[self.i]) and s.closing and np.any(self.ledger.qty[list(s.assets)]):
                s.lock_sign=-s.direction
                self.close_reasons[pair]='partial_pair_abort'
        for e in self.events[n:]:
            pair=e['pair']
            if e['event']=='entry_intent':
                self.close_reasons.pop(pair,None)
            elif e['event']=='incomplete_pair_flatten_next_hour':
                self.states[pair].lock_sign=-self.states[pair].direction
                self.close_reasons[pair]='partial_pair_abort'
            elif e['event']=='exit_intent':
                self.close_reasons.setdefault(pair,e['reason'])
                e['initial_exit_reason']=self.close_reasons[pair]
        for pair,was_held in held.items():
            if was_held and not np.any(self.ledger.qty[list(self.states[pair].assets)]):
                self.last_flat[pair]=self.i
        return value
