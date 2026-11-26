"""Confirmed-event accounting: exchange net inventory and strategy ownership.

Physical realized PnL/entry prices need not equal virtual realized PnL/entries.
Quantity, equity, total fees and funding must agree. Zero-net funding needs an
observed cash-per-unit settlement; it cannot be inferred from a zero payment.
"""
import copy
import hashlib
import json
from decimal import Decimal
import numpy as np
from replay_lab.common_comparison.engine import Ledger, rounded


class ReconciliationError(ValueError):
    pass


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def fill_ledger(ledger,j,quantity,price,fee,at,**metadata):
    """Split a physical or virtual reversal; preserve one actual fee exactly."""
    rate=fee/(abs(quantity)*price)*10000
    old=float(ledger.qty[j])
    if old*quantity<0 and abs(quantity)>abs(old):
        ledger.fill(j,-old,price,rate,at,**metadata)
        quantity=float(Decimal(str(quantity))+Decimal(str(old)))
    if quantity:ledger.fill(j,quantity,price,rate,at,**metadata)


def credit(ledger,cash,at,coin,identity):
    ledger.cash+=cash;ledger.shadow_cash+=cash
    ledger.funding+=cash;ledger.signed_funding+=cash
    ledger.funding_events.append(dict(time=at,coin=coin,signed_amount=cash,
        source='confirmed_joint_settlement',payment_identity=identity))


class JointBook:
    def __init__(self,symbols,capital):
        self.symbols=tuple(symbols)
        if len(set(self.symbols))!=len(self.symbols) or not capital or any(v<0 for v in capital.values()):
            raise ValueError('Invalid ownership definition')
        self.capital=dict(capital)
        self.physical=Ledger(sum(capital.values()),self.symbols)
        self.owners={name:Ledger(value,self.symbols) for name,value in capital.items()}
        self.events=[];self.identities={};self.version=0;self.last_ms=-1
        self.fill_times=set()

    def _identity(self,key,event):
        value=fingerprint(event)
        if key in self.identities:
            if self.identities[key]!=value:raise ReconciliationError('Conflicting duplicate event')
            return False
        if event['time']<self.last_ms:raise ReconciliationError('Late event requires chronological rebuild')
        return True

    def _commit(self,key,event):
        self.identities[key]=fingerprint(event);self.events.append(copy.deepcopy(event))
        self.last_ms=event['time'];self.version+=1

    def apply_fill(self,owner,row):
        if owner not in self.owners:raise ReconciliationError('Unknown fill owner')
        coin=row['coin'];symbol=coin.removeprefix('pas:')
        if coin!='pas:'+symbol or symbol not in self.symbols or row['feeToken']!='USDP':
            raise ReconciliationError('Wrong venue, collateral or market')
        key='fill',coin,str(row['tid']);event=dict(kind='fill',owner=owner,time=row['time'],row=row)
        if not self._identity(key,event):return False
        if row['side'] not in ('B','A'):raise ReconciliationError('Unknown fill side')
        q=float(row['sz'])*(1 if row['side']=='B' else -1)
        px,fee,start=float(row['px']),float(row['fee']),float(row['startPosition'])
        j=self.symbols.index(symbol)
        if not np.isfinite([q,px,fee,start]).all() or not q or px<=0:
            raise ReconciliationError('Invalid fill numbers')
        if abs(self.physical.qty[j]-start)>1e-8:
            raise ReconciliationError(f'Missing fill/order: {coin} tid={row["tid"]} expected={self.physical.qty[j]} observed start={start}')
        if any(e['kind']=='funding' and e['time']==row['time'] and e['coin']==coin for e in self.events):
            raise ReconciliationError('Fill/funding sequence at identical timestamps is ambiguous')
        for ledger in (self.physical,self.owners[owner]):
            fill_ledger(ledger,j,q,px,fee,row['time']/1000,exchange_tid=row['tid'],owner=owner)
        self.fill_times.add((coin,row['time']));self._commit(key,event)
        return True

    def apply_funding(self,coin,time_ms,cash,net_qty,identity,*,cash_per_unit=None):
        """cash_per_unit = -oracle_price * settled hourly rate, if observed.

        For nonzero net positions derive it from the actual wallet payment.
        For exactly hedged positions require it explicitly; never allocate an
        invented zero to two opposing virtual owners.
        """
        if coin not in ['pas:'+s for s in self.symbols]:raise ReconciliationError('Wrong funding market')
        j=self.symbols.index(coin[4:]);cash=float(cash);net_qty=float(net_qty)
        event=dict(kind='funding',coin=coin,time=time_ms,cash=cash,net_qty=net_qty,
                   identity=identity,cash_per_unit=cash_per_unit)
        key='funding',coin,str(identity)
        if not self._identity(key,event):return False
        if not np.isfinite([cash,net_qty]).all() or abs(net_qty-self.physical.qty[j])>1e-8:
            raise ReconciliationError('Funding position mismatch')
        if (coin,time_ms) in self.fill_times:raise ReconciliationError('Ambiguous fill/funding sequence')
        if abs(net_qty)>1e-12:
            unit=cash/net_qty
            if cash_per_unit is not None and abs(unit-float(cash_per_unit))*abs(net_qty)>1e-6:
                raise ReconciliationError('Funding oracle/rate conflicts with wallet payment')
        else:
            if abs(cash)>1e-8:raise ReconciliationError('Nonzero payment for a net flat account')
            opposing=any(abs(x.qty[j])>1e-12 for x in self.owners.values())
            if opposing and cash_per_unit is None:
                raise ReconciliationError('Observed oracle and settled rate required for offsetting owners')
            unit=float(cash_per_unit or 0.)
        if not np.isfinite(unit):raise ReconciliationError('Invalid funding per unit')
        credit(self.physical,cash,time_ms/1000,coin,identity)
        for ledger in self.owners.values():credit(ledger,float(ledger.qty[j])*unit,time_ms/1000,coin,identity)
        self._commit(key,event)
        return True

    def reconcile(self,marks):
        marks=np.asarray(marks,float)
        if marks.shape!=(len(self.symbols),) or not np.isfinite(marks).all() or np.any(marks<=0):
            raise ReconciliationError('All markets require valid observed marks')
        owner_qty=np.sum([x.qty for x in self.owners.values()],axis=0)
        errors=dict(quantity=float(abs(owner_qty-self.physical.qty).max()),
            equity=abs(sum(x.reconcile(marks) for x in self.owners.values())-self.physical.reconcile(marks)),
            fees=abs(sum(x.fees for x in self.owners.values())-self.physical.fees),
            funding=abs(sum(x.funding for x in self.owners.values())-self.physical.funding))
        if errors['quantity']>1e-8 or max(errors.values())>1e-6:
            raise ReconciliationError('Virtual/physical conservation failure: '+str(errors))
        return errors

    def plan(self,owner,symbol,target,price,decimals,*,max_notional,slot):
        """Owner delta is the external order, NOT the owner's absolute target.

        Virtual reversals close first. Exchange reduce-only uses the PHYSICAL
        position. A virtual close may physically increase opposite inventory.
        """
        if owner not in self.owners or symbol not in self.symbols:raise ValueError('Unknown owner/market')
        if not np.isfinite([target,price,max_notional]).all() or min(price,max_notional)<=0:
            raise ValueError('Invalid plan')
        j=self.symbols.index(symbol);old=float(self.owners[owner].qty[j]);physical=float(self.physical.qty[j])
        delta=float(Decimal(str(target))-Decimal(str(old)))
        if old*delta<0:delta=np.sign(delta)*min(abs(delta),abs(old))
        delta=np.sign(delta)*min(abs(delta),max_notional/price)
        delta=rounded(delta,decimals)
        # Split reductions at physical zero too, then evaluate another packet.
        if physical*delta<0 and abs(delta)>abs(physical):delta=-physical
        if not delta:return None
        reduce_only=bool(physical*delta<0 and abs(delta)<=abs(physical))
        if abs(delta)*price<10 and not (reduce_only and abs(delta)==abs(physical)):
            return None  # no virtual-only dust exception at the exchange
        return dict(owner=owner,coin='pas:'+symbol,delta=float(delta),price=float(price),
            reduce_only=reduce_only,expected_physical=physical,expected_owned=old,
            physical_target=physical+delta,owner_target=old+delta,version=self.version,
            cloid='0x'+fingerprint(dict(owner=owner,slot=slot,coin=symbol))[:32])

    def validate_plan(self,plan):
        j=self.symbols.index(plan['coin'][4:])
        if plan['version']!=self.version or abs(plan['expected_physical']-self.physical.qty[j])>1e-9 or abs(plan['expected_owned']-self.owners[plan['owner']].qty[j])>1e-9:
            raise ReconciliationError('State changed after decision; reconcile and replan')

    def serialize(self):
        return dict(schema='joint_wallet_v1',symbols=list(self.symbols),capital=self.capital,events=self.events)

    @classmethod
    def restore(cls,data):
        if data['schema']!='joint_wallet_v1':raise ValueError('Unknown book schema')
        book=cls(data['symbols'],data['capital'])
        for e in data['events']:
            if e['kind']=='fill':book.apply_fill(e['owner'],e['row'])
            else:book.apply_funding(e['coin'],e['time'],e['cash'],e['net_qty'],e['identity'],cash_per_unit=e['cash_per_unit'])
        return book
