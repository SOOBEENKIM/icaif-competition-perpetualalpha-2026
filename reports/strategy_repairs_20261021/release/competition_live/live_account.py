"""Selected lifecycle -> PAS orders -> confirmed exchange fills, never mock fills.

The launcher must hold the shared wallet lock and durably checkpoint the entire
coordinator before/after a packet. Unknown outcomes require reconciliation; the
adapter stops rather than reconstructing a fill from an order response or book.
"""
from decimal import Decimal
import time
import hashlib
import json
import numpy as np
from experiments.published_pair_lifecycle_20261017.account import BookAccount
from experiments.shared_execution_pairs_20261017.account_v2 import difference, sum_targets
from experiments.basis_carry_20261017.probe import asof
from replay_lab.common_comparison.engine import rounded
from replay_lab.common_comparison.engine import Ledger
from bot.execution import OrderRejected, OutcomeUnknown
from .execution import prepare_pas_order


class LiveAccount(BookAccount):
    actual_fills_only=True

    def __init__(self, profile, executor, checkpoint, collect_fills, *, refresh_market=None):
        super().__init__(profile['initial_capital'],profile['trading']['symbols'])
        self.profile=profile;self.executor=executor
        self.checkpoint=checkpoint;self.collect_fills=collect_fills
        self.refresh_market=refresh_market
        self.applied_fills=set();self.pending_orders={}
        self.applied_funding=set()
        # Positions/fills belong to the active sleeve. The teammate's cash is
        # represented explicitly and never used by the selected policy's sizing.
        self.reserved=Ledger(profile['reserved_capital'],profile['trading']['symbols'])

    def team_equity(self,marks):
        return self.reconcile(marks)+self.reserved.reconcile(marks)

    def apply_actual_funding(self,rows):
        """Use confirmed account payments, not current estimated hourly rates."""
        for row in sorted(rows,key=lambda r:r['time']):
            delta=row['delta'];coin=delta.get('coin','')
            if not coin.startswith('pas:'):continue
            key=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
            if key in self.applied_funding:continue
            symbol=coin.split(':',1)[1]
            if symbol not in self.physical.symbols:raise ValueError('Unknown PAS funding market')
            j=self.physical.symbols.index(symbol);at=row['time']/1000
            quantities={name:sum(f['signed_qty'] for f in ledger.fills if f['asset']==j and f['time']<at)
                for name,ledger in self.owners.items()}
            net=sum(quantities.values());payment=float(delta['usdc'])
            if not np.isfinite([net,payment]).all():raise ValueError('Invalid funding payment')
            if 'szi' in delta and abs(float(delta['szi'])-net)>1e-8:
                raise OutcomeUnknown('Funding position does not match the settled fill history')
            if abs(net)<1e-12 and abs(payment)>1e-8:
                raise OutcomeUnknown('Cannot allocate funding without settled inventory')
            def credit(ledger,cash):
                ledger.cash+=cash;ledger.shadow_cash+=cash;ledger.funding+=cash
                ledger.signed_funding+=cash
                ledger.funding_events.append(dict(time=at,signed_amount=cash,adverse_amount=0.,
                    coin=coin,source='actual_userFunding',payment_identity=key))
            credit(self.physical,payment)
            for name,qty in quantities.items():
                cash=payment*qty/net if net else 0.
                credit(self.owners[name],cash)
            self.applied_funding.add(key)
            self.checkpoint('actual_funding_applied',{'identity':key,'coin':coin,'cash':payment})

    def execute_books(self,intent,streams,marks,sigma,p,sc,cfg,*,on_time=None):
        self._take(intent)
        orders=[];marks=np.asarray(marks,float).copy()
        target=sum_targets(intent['owner_targets'])
        self.checkpoint('packet_claimed',intent)
        for j,symbol in enumerate(self.physical.symbols):
            if not any(abs(need[j])>1e-12 for need in self.needs(intent).values()):
                continue
            now=time.time()
            if on_time is not None:on_time(int(now*1000),self)
            refresh=getattr(self,'refresh_market',None)
            current=refresh(symbol) if refresh is not None else streams.get(symbol)
            # Funding and earlier confirmed fills may have taken seconds.
            now=time.time()
            quote=asof(current,int(now*1000),cfg) if current is not None else None
            if quote is None:
                orders.append(dict(symbol=symbol,status='NO_FRESH_BOOK',filled_qty=0.));continue
            marks[j]=float(quote['ctx']['markPx'])
            self.cross(intent,j,float(quote['mid']),now)
            old=self.physical.qty[j];delta=difference(target[j],old)
            if not delta:continue
            side=1 if delta>0 else -1
            levels=quote['book']['levels'][1 if side>0 else 0]
            best=float(levels[0]['px'])
            economic=self.net_limit(intent,j,side)
            bound=best*(1+side*.001)
            limit=min(bound,economic) if side>0 and economic is not None else max(bound,economic) if economic is not None else bound
            if side*(best-limit)>0:
                orders.append(dict(symbol=symbol,status='ECONOMIC_LIMIT',filled_qty=0.));continue
            visible=sum(float(x['sz'])*sc['depth_fraction'] for x in levels if side*(float(x['px'])-limit)<=0)
            request=min(abs(delta),visible,self.profile['execution']['maximum_single_order_notional']/limit)
            # Preserve reduction-before-reversal at the physical exchange.
            if old*delta<0:request=min(request,abs(old))
            request=abs(rounded(request,p['execution']['decimals'][j]))
            request=self.feasible_size(j,side,request,limit,marks,sigma,p,sc['fee_bps']) if request else 0.
            if not request:continue
            reduce_only=bool(old*delta<0 and request<=abs(old))
            if request*limit<10 and not (reduce_only and request==abs(old)):continue
            market,context=self.executor.market('pas:'+symbol)
            order=prepare_pas_order(self.profile,'pas:'+symbol,side>0,request,limit,market,context,
                current_qty=old,reduce_only=reduce_only,slot=f"{intent.get('packet_id',intent['decision_time'])}:{j}")
            self.pending_orders[order.cloid]=dict(intent=intent,order=order.__dict__,asset=j,started_ms=int(now*1000))
            self.checkpoint('before_order',self.pending_orders[order.cloid])
            try:
                self.executor.set_risk_state(self.physical.qty,sigma)
                response=self.executor.submit(order)
            except OrderRejected:
                self.pending_orders.pop(order.cloid)
                self.checkpoint('order_rejected',{'cloid':order.cloid})
                orders.append(dict(symbol=symbol,status='REJECTED',filled_qty=0.));continue
            except Exception:
                self.unknown.add(symbol);self.checkpoint('order_unresolved',{'cloid':order.cloid})
                raise
            filled=response.get('filled')
            if filled is None:
                self.unknown.add(symbol);self.checkpoint('order_unresolved',{'cloid':order.cloid})
                raise OutcomeUnknown('PAS IOC did not return a confirmed terminal fill')
            oid=filled['oid']
            # collect_fills must query userFillsByTime and return final records
            # for this oid; any propagation delay leaves the order unresolved.
            reported=Decimal(str(filled['totalSz']))
            rows=self.collect_fills(oid,int(now*1000),reported)
            got=sum((Decimal(f['sz']) for f in rows),Decimal(0))
            if got!=reported or got>Decimal(order.size):
                self.unknown.add(symbol);self.checkpoint('fills_pending',{'cloid':order.cloid})
                raise OutcomeUnknown('Actual PAS fill records do not yet match the order receipt')
            for f in rows:
                identity=(f['coin'],f['tid'])
                if (identity in self.applied_fills or f['coin']!=order.coin or f['oid']!=oid
                        or f['feeToken']!='USDP' or f['side']!=('B' if side>0 else 'A')):
                    self.unknown.add(symbol);self.checkpoint('fill_mismatch',{'cloid':order.cloid})
                    raise OutcomeUnknown('Unexpected or duplicate PAS fill')
                qty=float(f['sz']);price=float(f['px']);fee=float(f['fee'])
                if not np.isfinite([qty,price,fee]).all() or min(qty,price)<=0 or side*(price-float(order.price))>1e-8:
                    self.unknown.add(symbol);raise OutcomeUnknown('Invalid actual PAS fill')
                rate=fee/(qty*price)*10000
                nf=len(self.physical.fills)
                self.physical_fill(j,side*qty,price,rate,f['time']/1000,
                    decision_time=intent['decision_time'],source_close=intent['source_close'],
                    source_available=intent['source_available'],risk_forced=intent['risk_forced'])
                for fill in self.physical.fills[nf:]:self.allocate(intent,fill,rate)
                self.applied_fills.add(identity)
            self.pending_orders.pop(order.cloid)
            self.last_packet_arrival_ms=int(time.time()*1000)
            self.reconcile(marks)
            self.checkpoint('fills_applied',{'cloid':order.cloid,'oid':oid})
            orders.append(dict(symbol=symbol,status='ACTUAL_FULL' if got==Decimal(order.size) else 'ACTUAL_PARTIAL',
                filled_qty=float(got),requested_qty=float(order.size),arrival_ms=self.last_packet_arrival_ms,limit=float(order.price)))
        return dict(orders=orders,unknown_symbols=sorted(self.unknown),orders_sent=len([o for o in orders if o['status'].startswith('ACTUAL')]))
