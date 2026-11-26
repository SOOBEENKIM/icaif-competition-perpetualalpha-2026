"""Recover bookkeeping from terminal receipts plus public fills, never send orders.

Operate on a detached copy. A receipt without the complete public fills, an
unknown outcome, or an unrelated unknown symbol remains blocked.
"""
import copy
from decimal import Decimal
import numpy as np
from joint_wallet.book import ReconciliationError


def recover_confirmed(portfolio,dispatch,rows,marks,*,now):
    if portfolio.fatal or portfolio.halted:
        raise ReconciliationError('Fatal/risk-halted portfolio requires separate resolution')
    result=copy.deepcopy(portfolio);account=result.account
    if account.executor is not None or account.checkpoint is not None:
        raise ReconciliationError('Recovery requires a detached private checkpoint')
    if dispatch.get('pending'):
        raise ReconciliationError('Dispatcher has no final receipt; resolve venue order status first')
    unique={}
    for row in rows:
        key=(row['coin'],row['tid'])
        if key in unique and unique[key]!=row:
            raise ReconciliationError('Conflicting public fill identity')
        unique[key]=row
    repaired=[];count=0
    for cloid,pending in sorted(account.pending_orders.items(),key=lambda kv:kv[1]['started_ms']):
        order=pending['order'];j=pending['asset'];intent=pending['intent']
        receipt=dispatch.get('completed',{}).get(cloid)
        if (not receipt or receipt.get('status')!='confirmed' or receipt.get('owner')!='ours'
                or receipt.get('cloid')!=cloid or order.get('cloid')!=cloid
                or order.get('tif')!='Ioc' or order['coin']!='pas:'+account.physical.symbols[j]):
            raise ReconciliationError('No matching terminal owner receipt')
        fills=[r for r in unique.values() if r['oid']==receipt['oid'] and r['coin']==order['coin']]
        reported=Decimal(str(receipt['filled_size']));requested=Decimal(order['size'])
        if (not reported.is_finite() or reported<0 or reported>requested
                or sum((Decimal(r['sz']) for r in fills),Decimal(0))!=reported):
            raise ReconciliationError('Receipt quantity does not match complete public fills')
        side=1 if order['is_buy'] else -1
        # Validate every row before applying any of them, including rows already applied.
        for row in fills:
            px,qty,fee=map(float,(row['px'],row['sz'],row['fee']))
            if (row['feeToken']!='USDP' or row['side']!=('B' if side>0 else 'A')
                    or row.get('cloid') not in (None,cloid)
                    or row['time']<pending['started_ms']-2000
                    or not np.isfinite([px,qty,fee]).all() or min(px,qty)<=0
                    or side*(px-float(order['price']))>1e-8):
                raise ReconciliationError('Public fill violates pending order')
        applied=[]
        for row in sorted(fills,key=lambda r:r['time']):
            identity=(row['coin'],row['tid'])
            if identity in account.applied_fills:continue
            qty,px,fee=map(float,(row['sz'],row['px'],row['fee']));rate=fee/(qty*px)*10000
            nf=len(account.physical.fills)
            account.physical_fill(j,side*qty,px,rate,row['time']/1000,
                decision_time=intent['decision_time'],source_close=intent['source_close'],
                source_available=intent['source_available'],risk_forced=intent['risk_forced'])
            for fill in account.physical.fills[nf:]:account.allocate(intent,fill,rate)
            account.applied_fills.add(identity);applied.append(identity);count+=1
            account.last_packet_arrival_ms=max(account.last_packet_arrival_ms,row['time'])
        del account.pending_orders[cloid]
        account.unknown.discard(order['coin'][4:])
        repaired.append(dict(cloid=cloid,oid=receipt['oid'],coin=order['coin'],filled_size=str(reported),applied=applied))
    if account.unknown:raise ReconciliationError('Unresolved symbols outside confirmed pending orders')
    account.reconcile(np.asarray(marks,float))
    expired=None
    if result.base_task is not None and result.base_task['expires']<=now:
        expired=dict(expires=result.base_task['expires'],target=result.base_task['target'].tolist())
        result.base_task=None
    if repaired or expired:
        result.events.append(dict(kind='confirmed_receipt_recovery',time=now,orders=repaired,expired_base_task=expired))
    return result,dict(applied_fill_count=count,repaired_orders=repaired,expired_base_task=expired,
        orders_sent=0,strategy_changed=False)
