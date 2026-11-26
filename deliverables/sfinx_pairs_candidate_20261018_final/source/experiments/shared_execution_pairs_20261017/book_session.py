"""Offline L2 pair lifecycle through the same shared execution account."""
import copy,math
import numpy as np
from experiments.basis_carry_20261017.probe import asof,walk
from replay_lab.common_comparison.engine import rounded,risk_breaches
from .book_plan import plan_entry


class BookPairSession:
    def __init__(self,account,intent,cfg,p,scenario,*,on_time=None):
        self.account=account;self.intent=copy.deepcopy(intent);self.cfg=cfg;self.p=p
        self.sc={**scenario,'impact_bps':0.};self.on_time=on_time
        self.state='new';self.events=[];self.ready_at=0;self.completions=0;self.sequence=0
        self.owner=account.owners['pair'];self.ids=intent['ids'];self.first_fill=None
        self.fees_start=self.owner.fees;self.funding_start=self.owner.funding
        self.fill_start=len(self.owner.fills)

    def books(self,now,streams):
        return {s:asof(streams[s],now*1000,self.cfg) if s in streams else None for s in self.intent['pair']}

    def marks(self,quotes,previous):
        result=np.asarray(previous,float).copy()
        for s,j in zip(self.intent['pair'],self.ids):
            if quotes[s] is not None:result[j]=quotes[s]['mid']
        return result

    def submit(self,now,target,limits,streams,marks,sigma,kind):
        self.sequence+=1
        base=self.account.owners['base'].qty.copy();pair=self.owner.qty.copy();pair[self.ids]=target
        raw=dict(packet_id=f'{self.intent["forecast_source"]}:{self.sequence}',decision_time=now,source_close=now,
            source_available=now,source_prices=np.asarray(marks).tolist(),owner_targets={'base':base.tolist(),'pair':pair.tolist()},
            owner_limits={'pair':{str(j):v for j,v in limits.items()}},risk_forced=kind=='close',sigma=sigma)
        packet=self.account.seal(raw,self.p,self.sc)
        result=self.account.execute_books(packet,streams,marks,sigma,self.p,self.sc,self.cfg,on_time=self.on_time)
        self.events.append(dict(time=now,kind=kind,result=result))
        if result['unknown_symbols']:
            self.state='reconcile_required';return result
        known_fills=self.owner.fills[self.fill_start:]
        end=max([o.get('arrival_ms',now*1000) for o in result['orders']]+[f['time']*1000 for f in known_fills]+[now*1000])
        self.ready_at=math.ceil((end+self.cfg['completion_ack_delay_ms'])/1000)
        q=self.owner.qty[self.ids]
        if not np.any(q):self.state='closed' if kind in ('close','trim') else 'unfilled';return result
        if self.first_fill is None:self.first_fill=min(f['time'] for f in known_fills)
        ratios=q/self.target
        balanced=min(ratios)>0 and abs(ratios[0]-ratios[1])/max(ratios)<=self.cfg['pair_balance_tolerance']
        if kind=='close':self.state='closing'
        elif balanced:self.state='open'
        elif min(ratios)>0:self.state='trim'
        elif kind=='entry' and self.completions==0:self.state='completion'
        else:self.state='closing'
        return result

    def enter(self,now,streams,marks,sigma):
        if self.state!='new':raise RuntimeError('Session entry already evaluated')
        quotes=self.books(now,streams);marks=self.marks(quotes,marks)
        source={**self.intent,'account_marks':marks.tolist()}
        plan=plan_entry(source,quotes,now,self.cfg,self.p,self.sc,self.account.physical.qty,
            self.account.physical.equity(marks),sigma,owned_qty=self.owner.qty)
        self.plan=plan
        if not plan['accepted']:self.state='rejected';return plan
        self.target=np.asarray(plan['target'])
        return self.submit(now,self.target,{o['asset']:o['limit'] for o in plan['orders']},streams,marks,sigma,'entry')

    def completion(self,now,quotes,marks,sigma):
        q=self.owner.qty[self.ids];present=np.flatnonzero(q)
        if len(present)!=1 or any(quotes[s] is None for s in self.intent['pair']):return None
        have=int(present[0]);missing=1-have;ids=self.ids
        target=self.target*(q[have]/self.target[have]);target[have]=q[have]
        target[missing]=rounded(target[missing],self.p['execution']['decimals'][ids[missing]])
        if not target[missing] or abs(target[missing])*marks[ids[missing]]<self.p['execution']['minimum_order_value']:return None
        ratio=target/self.target
        if abs(ratio[0]-ratio[1])/max(ratio)>self.cfg['pair_balance_tolerance']:return None
        entries=self.owner.entry[ids].copy();reverse=[];oracle=[];fund=[]
        for k,(symbol,amount) in enumerate(zip(self.intent['pair'],target)):
            quote=quotes[symbol];levels=quote['book']['levels']
            exit_=walk(levels[0 if amount>0 else 1],abs(amount),self.sc['depth_fraction'])
            if exit_ is None:return None
            reverse.append(exit_);oracle.append(quote['oracle']);fund.append(float(quote['ctx']['funding']))
            if k==missing:
                preview=walk(levels[1 if amount>0 else 0],abs(amount),self.sc['depth_fraction'])
                if preview is None:return None
                entries[k]=preview
        gross=float(abs(target)@marks[ids]);fee=self.sc['fee_bps']/10000;sign=np.sign(target[missing])
        age=(now-self.intent['forecast_source'])/3600
        carry=-target*np.asarray(oracle)*np.asarray(fund)*max(0,self.cfg['funding_forecast_hours']-age)
        carry=float(np.where(carry>0,.5*carry,carry).sum())+self.owner.funding-self.funding_start
        reserve=float(target@(marks[ids]-np.asarray(reverse)))+gross*(fee+self.sc['exit_impact_bps']/10000)
        paid=self.owner.fees-self.fees_start;terminal=np.asarray(self.intent['terminal_prices']);buffer=gross*self.cfg['entry_buffer_bps']/10000
        def value(price):
            e=entries.copy();e[missing]=price
            return float(target@(terminal-e)+carry-paid-abs(target[missing])*price*fee-reserve)
        net=value(entries[missing]);slack=net-buffer
        if not np.isfinite(slack) or slack<=0:return None
        distance=slack/(abs(target[missing])*(1+sign*fee));limit=entries[missing]+sign*distance
        if limit<=0:return None
        all_qty=self.account.physical.qty.copy();all_qty[ids]+=target-q
        eq=self.account.physical.equity(marks)-target[missing]*(limit-marks[ids[missing]])-abs(target[missing])*limit*fee
        if eq<=0 or risk_breaches(all_qty,eq,marks,sigma,self.p['risk']):return None
        worst=value(limit);assert worst>=buffer-1e-8
        return dict(target=target,limits={ids[missing]:float(limit)},expected_net_cash=net,minimum_expected_cash_at_limits=worst,
            buffer_cash=buffer,sunk_fees=paid,settled_pair_funding=self.owner.funding-self.funding_start)

    def advance(self,now,streams,marks,sigma):
        if self.state in ('new','rejected','unfilled','closed','reconcile_required') or now<self.ready_at:return None
        quotes=self.books(now,streams);marks=self.marks(quotes,marks)
        if self.state=='completion' and now>self.first_fill+self.cfg['completion_window_seconds']:self.state='closing'
        if self.first_fill is not None and now-self.first_fill>=self.cfg['maximum_holding_hours']*3600:self.state='closing'
        if self.state in ('completion','open'):
            if all(quotes[s] is not None for s in self.intent['pair']):
                z=(np.log(marks[self.ids[0]])-self.intent['beta']*np.log(marks[self.ids[1]])-self.intent['entry_mean'])/self.intent['entry_sd']
                age=(now-self.first_fill)/3600;limit=self.cfg['entry_z']*self.cfg['stop_multiplier']
                if age>self.cfg['time_decay_start_hours']:
                    limit*=max(0,(self.cfg['maximum_holding_hours']-age)/(self.cfg['maximum_holding_hours']-self.cfg['time_decay_start_hours']))
                if not -limit<self.intent['direction']*z<0:self.state='closing'
            elif self.state=='open':return None
        if self.state=='open':return None
        if self.state=='completion':
            plan=self.completion(now,quotes,marks,sigma);self.completions+=1
            if plan is not None:
                self.events.append(dict(time=now,kind='completion_budget',**{k:v for k,v in plan.items() if k not in ('target','limits')}))
                return self.submit(now,plan['target'],plan['limits'],streams,marks,sigma,'completion')
            self.state='closing'
        if self.state=='trim':
            ratio=min(self.owner.qty[self.ids]/self.target)
            target=np.array([rounded(v*ratio,self.p['execution']['decimals'][j]) for v,j in zip(self.target,self.ids)])
            if np.any(target==0):self.state='closing'
            else:return self.submit(now,target,{},streams,marks,sigma,'trim')
        if self.state=='closing':return self.submit(now,np.zeros(2),{},streams,marks,sigma,'close')
        return None
