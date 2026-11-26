"""Serialized, journaled owner-delta dispatcher with injected venue I/O.

No signer discovery or service management. A live caller must provide fresh
reconciliation, PAS venue preflight, IOC submission and confirmed-fill lookup.
Independent teammate programs must use virtual quantities, not absolute net
exchange targets, to preserve their own intended strategy allocations.
"""
import contextlib
import fcntl
import json
import os
from decimal import Decimal
from pathlib import Path
import numpy as np
from replay_lab.common_comparison.engine import risk_breaches
from .book import ReconciliationError,fingerprint
from bot.execution import OrderRejected


class SubmissionNotAttempted(ReconciliationError):
    """A failed validation before the durable send boundary; no order was sent."""


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp')
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_TRUNC|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump(value,f,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


class Dispatcher:
    def __init__(self,directory,profile):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'dispatch.json';self.profile=profile
        if profile['dex']!='pas' or profile['rest_url']!='https://api.hyperliquid-testnet.xyz':
            raise ValueError('PAS testnet only')

    @contextlib.contextmanager
    def locked(self):
        with (self.directory/'dispatch.lock').open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            yield

    def state(self):
        return json.loads(self.path.read_text()) if self.path.exists() else dict(pending=None,completed={})

    def validate_request(self,plan,book,marks,sigma):
        """Shared validation for unsigned preview and actual submission."""
        book.validate_plan(plan);book.reconcile(marks)
        j=book.symbols.index(plan['coin'][4:]);ledger=book.owners[plan['owner']]
        if ledger.initial!=self.profile['initial_capital']:
            raise ReconciliationError('Owner capital does not match the agreed 50000 USDP sleeve')
        proposed=ledger.qty.copy();proposed[j]+=plan['delta']
        fee=abs(plan['delta'])*plan['price']*self.profile['trading']['execution']['fee_bps']/10000
        equity=ledger.equity(marks)-plan['delta']*(plan['price']-marks[j])-fee
        before=risk_breaches(ledger.qty,ledger.equity(marks),marks,sigma,self.profile['trading']['risk'])
        after=risk_breaches(proposed,equity,marks,sigma,self.profile['trading']['risk'])
        owner_reduction=ledger.qty[j]*plan['delta']<0 and abs(plan['delta'])<=abs(ledger.qty[j])
        if after and not (before and owner_reduction):raise ReconciliationError('Owner risk limit')
        if abs(plan['delta'])*plan['price']>self.profile['execution']['maximum_single_order_notional']+1e-8:
            raise ReconciliationError('Per-order limit')

    def execute(self,plan,marks,sigma,*,refresh,preflight,send,confirmed_fills):
        """Each callback is explicit. This module cannot simulate a live fill.

        refresh -> authoritative JointBook; preflight checks real PAS schedule,
        precision, OI, margin and physical positions. send must not auto-retry.
        confirmed_fills queries exchange history; an order receipt is not enough.
        """
        with self.locked():
            state=self.state();key=plan['cloid'];digest=fingerprint(plan)
            if state['pending']:raise ReconciliationError('Unresolved prior submission; reconcile before further orders')
            if key in state['completed']:
                prior=state['completed'][key]
                if prior['plan_hash']!=digest:raise ReconciliationError('Cloid reused for another target')
                return dict(prior,already_completed=True)
            if key in state.get('rejected',{}):raise OrderRejected('Previously rejected IOC; use a new decision')
            try:
                book=refresh();self.validate_request(plan,book,marks,sigma)
                preflight(plan,book)
            except ReconciliationError as e:
                raise SubmissionNotAttempted(str(e)) from e
            state['pending']=dict(plan=plan,plan_hash=digest,stage='before_send')
            atomic_json(self.path,state)  # durable ownership BEFORE submission
            try:receipt=send(plan)
            except OrderRejected as e:
                state.setdefault('rejected',{})[key]=dict(plan_hash=digest,reason=str(e),owner=plan['owner'])
                state['pending']=None;atomic_json(self.path,state)
                raise
            state['pending'].update(stage='receipt',receipt=receipt);atomic_json(self.path,state)
            got=Decimal(str(receipt['filled_size']));rows=confirmed_fills(receipt['oid'],key)
            if got<0 or got>abs(Decimal(str(plan['delta']))):raise ReconciliationError('Invalid receipt quantity')
            actual=sum((Decimal(r['sz']) for r in rows),Decimal(0))
            if actual!=got:raise ReconciliationError('Receipt lacks confirmed exchange fills')
            seen=set()
            for r in rows:
                identity=r['coin'],r['tid']
                if (identity in seen or r['oid']!=receipt['oid'] or r['coin']!=plan['coin'] or r['feeToken']!='USDP'
                        or r['side']!=('B' if plan['delta']>0 else 'A')
                        or r.get('cloid') not in (None,key)):
                    raise ReconciliationError('Actual fill attribution mismatch')
                seen.add(identity)
                px=float(r['px']);qty=float(r['sz']);fee=float(r['fee'])
                if not np.isfinite([px,qty,fee]).all() or min(px,qty)<=0 or np.sign(plan['delta'])*(px-plan['price'])>1e-8:
                    raise ReconciliationError('Invalid fill price, quantity or fee')
            # Rebuild rather than apply out-of-order receipts if teammate filled
            # concurrently. The refresh callback uses this durable cloid/oid.
            final=refresh();final.reconcile(marks)
            own_events={(e['row']['coin'],e['row']['tid']) for e in final.events
                        if e['kind']=='fill' and e['owner']==plan['owner']}
            if not seen<=own_events:raise ReconciliationError('Confirmed fill missing from rebuilt owner book')
            result=dict(plan_hash=digest,cloid=key,oid=receipt['oid'],filled_size=str(got),status='confirmed',
                        owner=plan['owner'],book_version=final.version,already_completed=False)
            state['completed'][key]=result;state['pending']=None;atomic_json(self.path,state)
            return result
