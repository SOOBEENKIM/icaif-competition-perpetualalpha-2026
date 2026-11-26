"""Candidate executor: refresh each needed market after prior fill waits.

Not imported by the active deployment. Requires an injected public quote provider;
retains original economics, confirmed fills, checkpointing and reconciliation.
"""
from decimal import Decimal
import time
import numpy as np
from competition_live.live_account import LiveAccount
from competition_live.execution import prepare_pas_order
from experiments.shared_execution_pairs_20261017.account_v2 import difference, sum_targets
from experiments.basis_carry_20261017.probe import asof
from replay_lab.common_comparison.engine import rounded
from bot.execution import OrderRejected, OutcomeUnknown
from competition_live.venue import PublicPAS


class PublicQuoteProvider:
    """Fresh unsigned PAS context/book for exactly the requested market."""
    def __init__(self, profile):
        if profile['network']!='testnet' or profile['dex']!='pas':
            raise ValueError('Only the selected PAS testnet venue is supported')
        self.profile=profile

    def __call__(self, symbol):
        if symbol not in self.profile['trading']['symbols']:
            raise ValueError('Market outside the authorized universe')
        api=PublicPAS(self.profile)
        try:
            meta,contexts=api.market();context_at=int(time.time()*1000)
            if meta['collateralToken']!=self.profile['collateral_token']:
                raise ValueError('Unexpected PAS collateral')
            by_name={m['name']:c for m,c in zip(meta['universe'],contexts)}
            coin='pas:'+symbol;context=by_name[coin]
            book=api.post(dict(type='l2Book',coin=coin));book_at=int(time.time()*1000)
            if book['coin']!=coin:raise ValueError('Quote market mismatch')
            return dict(book_times=[book_at],book=[(book_at,0,book)],
                ctx_times=[context_at],ctx=[(context_at,0,context)],resets=[])
        finally:
            api.session.close()

class FreshLiveAccount(LiveAccount):
    def __init__(self,*args,refresh_market,**kwargs):
        super().__init__(*args,**kwargs)
        self.refresh_market=refresh_market

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
            refreshed=self.refresh_market(symbol)
            now=time.time()
            quote=asof(refreshed,int(now*1000),cfg) if refreshed is not None else None
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
