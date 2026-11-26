"""One physical ledger, exact strategy ownership, one execution owner."""
from decimal import Decimal
import copy
import numpy as np
from replay_lab.common_comparison.engine import Ledger,execute_intent,rounded,risk_breaches
from replay_lab.pullback.broker import execution_price
from experiments.basis_carry_20261017.probe import delayed_snapshot,sweep


def difference(a,b):return float(Decimal(str(float(a)))-Decimal(str(float(b))))


def sum_targets(targets):
    return np.array([float(sum((Decimal(str(float(v))) for v in column),Decimal(0))) for column in zip(*targets.values())])


class SharedAccount:
    def __init__(self,initial,symbols):
        self.physical=Ledger(initial,symbols)
        self.owners={'base':Ledger(initial,symbols),'pair':Ledger(0.,symbols)}
        self.transfers=[];self.maximum_error=0.;self.seen=set();self.unknown=set()

    def reconcile(self,marks):
        physical=self.physical.reconcile(marks)
        q=sum((o.qty for o in self.owners.values()),np.zeros_like(self.physical.qty))
        error=max(float(np.max(abs(q-self.physical.qty))),abs(sum(o.reconcile(marks) for o in self.owners.values())-physical),
            abs(sum(o.fees for o in self.owners.values())-self.physical.fees),
            abs(sum(o.funding for o in self.owners.values())-self.physical.funding))
        self.maximum_error=max(self.maximum_error,error)
        assert error<1e-6,('ownership conservation',error)
        return physical

    def charge_funding(self,time,prices,rates,observed):
        # Apply the physical account's effective stress rates to both owners;
        # independently haircutting opposite virtual receipts breaks conservation.
        for ledger in (self.physical,*self.owners.values()):ledger.charge_funding(time,prices,rates,observed,0.)

    def owner_fill(self,name,j,amount,price,fee,time,source):
        ledger=self.owners[name];old=ledger.qty[j]
        if old*amount<0 and abs(amount)>abs(old):
            ledger.fill(j,-old,price,fee,time,allocation_type=source)
            amount=float(Decimal(str(float(amount)))+Decimal(str(float(old))))
        if amount:ledger.fill(j,amount,price,fee,time,allocation_type=source)

    def seal(self,intent,p,sc):
        result=copy.deepcopy(intent);limits={}
        # Preserve each strategy's reduction-before-reversal policy. Netting
        # independent owners may still legitimately reverse the physical account.
        for name,target in result['owner_targets'].items():
            for j,want in enumerate(target):
                if self.owners[name].qty[j]*want<0:target[j]=0.
        result['target_qty']=sum_targets(result['owner_targets']).tolist()
        for name,target in result['owner_targets'].items():
            explicit=result.get('owner_limits',{}).get(name,{})
            for j,new in enumerate(target):
                old=self.owners[name].qty[j];delta=difference(new,old)
                if not delta:continue
                if str(j) in explicit:limit=explicit[str(j)]
                elif old and (new*old<=0 or abs(new)<abs(old)):limit=None
                else:
                    side=np.sign(delta);preview=execution_price(result['source_prices'][j],side,sc['impact_bps'],p['execution']['decimals'][j])
                    limit=preview*(1+side*.001)
                limits[(name,j)]=limit
        result['sealed_owner_limits']=limits
        return result

    def _take(self,intent):
        key=intent.get('packet_id',intent['decision_time'])
        if self.unknown:raise RuntimeError('An unobserved prior order must be reconciled before another submission')
        if key in self.seen:raise RuntimeError('Duplicate execution packet')
        self.seen.add(key)

    def needs(self,intent):
        return {name:np.array([difference(v,o) for v,o in zip(target,self.owners[name].qty)])
            for name,target in intent['owner_targets'].items()}

    def cross(self,intent,j,price,time):
        needs=self.needs(intent);a,b=needs['base'][j],needs['pair'][j]
        if a*b>=0:return
        for name,amount in [('base',a),('pair',b)]:
            limit=intent['sealed_owner_limits'].get((name,j))
            if limit is not None and np.sign(amount)*(price-limit)>1e-10:return
        q=min(abs(a),abs(b));self.owner_fill('base',j,np.sign(a)*q,price,0.,time,'internal_transfer')
        self.owner_fill('pair',j,np.sign(b)*q,price,0.,time,'internal_transfer')
        self.transfers.append(dict(time=time,asset=j,signed_to_base=float(np.sign(a)*q),price=price,exchange_fee=0.))

    def net_limit(self,intent,j,side):
        values=[]
        for name,need in self.needs(intent).items():
            if side*need[j]>0:
                limit=intent['sealed_owner_limits'].get((name,j))
                if limit is not None:values.append(limit)
        return (min(values) if side>0 else max(values)) if values else None

    def allocate(self,intent,fill,fee_bps):
        j=fill['asset'];amount=fill['signed_qty'];side=np.sign(amount);remaining=abs(amount);needs=self.needs(intent)
        eligible=[name for name,v in needs.items() if side*v[j]>0]
        eligible.sort(key=lambda name:(not (self.owners[name].qty[j]*needs[name][j]<0),name!='pair'))
        for name in eligible:
            take=min(remaining,abs(needs[name][j]))
            self.owner_fill(name,j,side*take,fill['price'],fee_bps,fill['time'],'external_fill')
            remaining=difference(remaining,take)
        assert abs(remaining)<1e-8,('unallocated fill',fill,remaining)

    def physical_fill(self,j,amount,price,fee,time,**metadata):
        old=self.physical.qty[j]
        if old*amount<0 and abs(amount)>abs(old):
            self.physical.fill(j,-old,price,fee,time,**metadata)
            amount=float(Decimal(str(float(amount)))+Decimal(str(float(old))))
        if amount:self.physical.fill(j,amount,price,fee,time,**metadata)

    def feasible_size(self,j,side,amount,price,marks,sigma,p,fee):
        old=self.physical.qty[j]
        reduction=min(amount,abs(old)) if old*side<0 else 0.
        if amount<=reduction:return amount
        def safe(size):
            q=self.physical.qty.copy();q[j]+=side*size
            eq=self.physical.equity(marks)-side*size*(price-marks[j])-size*price*fee/10000
            return eq>0 and not risk_breaches(q,eq,marks,sigma,p['risk'])
        if safe(amount):return amount
        if not safe(reduction):return reduction
        lo,hi=reduction,amount
        for _ in range(40):
            mid=(lo+hi)/2
            if safe(mid):lo=mid
            else:hi=mid
        return abs(rounded(lo,p['execution']['decimals'][j]))

    def execute_candle(self,intent,market,i,p,sc):
        self._take(intent);time=int(market.times[i]);prices=market.close[i]
        for j,price in enumerate(prices):self.cross(intent,j,float(price),time)
        available=sc['participation']*np.minimum(intent['source_volume'],market.volume[i])
        available*=(intent['source_trades']>0)&(market.trades[i]>0)
        targets=np.asarray(intent['target_qty']);orders=[]
        ids=sorted(range(len(prices)),key=lambda j:self.physical.qty[j]*difference(targets[j],self.physical.qty[j])>=0)
        for j in ids:
            old=self.physical.qty[j];delta=difference(targets[j],old)
            if not delta:continue
            side=np.sign(delta);decimals=p['execution']['decimals'][j]
            request=abs(rounded(abs(delta),decimals));price=execution_price(prices[j],side,sc['impact_bps'],decimals)
            limit=self.net_limit(intent,j,side);cap=available[j]
            blocked=limit is not None and side*(price-limit)>1e-10
            if blocked:cap=0.
            got=abs(rounded(min(request,cap),decimals))
            full_close=old*side<0 and request==abs(old)
            if request*price<p['execution']['minimum_order_value'] and not full_close:got=0.
            if got:got=self.feasible_size(j,side,got,price,prices,intent['sigma'],p,p['execution']['fee_bps'])
            order=dict(time=time,decision_time=int(intent['decision_time']),source_close=int(intent['source_close']),
                source_available=int(intent['source_available']),symbol=self.physical.symbols[j],requested_qty=request,
                filled_qty=got,sign=float(side),reduce_only=bool(full_close or (old*side<0 and request<abs(old))),
                capacity=float(cap),risk_forced=bool(intent['risk_forced']),entry_limit_blocked=bool(blocked),
                status='FILLED' if got==request and got>0 else 'IOC_REMAINDER_CANCELLED')
            if got:
                nf=len(self.physical.fills)
                self.physical_fill(j,side*got,price,p['execution']['fee_bps'],time,
                    decision_time=order['decision_time'],source_close=order['source_close'],source_available=order['source_available'],risk_forced=order['risk_forced'])
                for f in self.physical.fills[nf:]:self.allocate(intent,f,p['execution']['fee_bps'])
            orders.append(order)
        self.reconcile(prices)
        return orders

    def execute_books(self,intent,streams,marks,sigma,p,sc,cfg,*,on_time=None):
        """Offline received-book adapter; missing arrival != a zero fill."""
        self._take(intent);events=[];orders=[];marks=np.asarray(marks,float).copy()
        requested=sum_targets(intent['owner_targets'])
        for j in range(len(marks)):
            if all(abs(v[j])<1e-12 for v in self.needs(intent).values()):continue
            symbol=self.physical.symbols[j]
            arrival=delayed_snapshot(streams[symbol],intent['decision_time']*1000+sc['latency_ms'],cfg) if symbol in streams else None
            if arrival is None:
                external=difference(requested[j],self.physical.qty[j])!=0
                if external:self.unknown.add(symbol)
                orders.append(dict(symbol=symbol,status='UNKNOWN_ARRIVAL' if external else 'INTERNAL_MARK_UNOBSERVED',filled_qty=None));continue
            events.append((arrival[0],j,arrival[1]))
        if on_time is None and any(at//3600000!=intent['decision_time']//3600 for at,_,_ in events):
            raise ValueError('Book adapter requires a funding settlement callback when crossing an hour')
        for at,j,snapshot in sorted(events,key=lambda e:(e[0],e[1])):
            if on_time is not None:on_time(at,self)
            levels=snapshot['book']['levels'];marks[j]=(float(levels[0][0]['px'])+float(levels[1][0]['px']))/2
            self.cross(intent,j,float(marks[j]),at/1000)
            old=self.physical.qty[j];delta=difference(requested[j],old)
            if not delta:continue
            reducing=old!=0 and (requested[j]*old<=0 or abs(requested[j])<abs(old))
            request=abs(delta);side=np.sign(delta)
            decimals=p['execution']['decimals'][j];request=abs(rounded(request,decimals))
            full_close=reducing and request==abs(old)
            if request*marks[j]<p['execution']['minimum_order_value'] and not full_close:
                orders.append(dict(symbol=self.physical.symbols[j],status='BELOW_MINIMUM',filled_qty=0.));continue
            limit=self.net_limit(intent,j,side)
            got,price=sweep(levels[1 if side>0 else 0],request,sc['depth_fraction'],side,decimals,limit)
            if got:
                got=self.feasible_size(j,side,got,price,marks,sigma,p,sc['fee_bps'])
                if got:
                    price=sweep(levels[1 if side>0 else 0],got,sc['depth_fraction'],side,decimals,limit)[1]
                    nf=len(self.physical.fills)
                    self.physical_fill(j,side*got,price,sc['fee_bps'],at/1000,decision_time=intent['decision_time'],
                        source_available=intent['source_available'],source_close=intent['source_close'],risk_forced=intent['risk_forced'])
                    for f in self.physical.fills[nf:]:self.allocate(intent,f,sc['fee_bps'])
            orders.append(dict(symbol=self.physical.symbols[j],status='FULL' if got>=request-1e-10 else 'PARTIAL_OR_UNFILLED',
                filled_qty=got,requested_qty=request,arrival_ms=at))
            self.reconcile(marks)
        return dict(orders=orders,unknown_symbols=sorted(self.unknown),orders_sent=0)
