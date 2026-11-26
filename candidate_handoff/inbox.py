"""Teammate strategy publishes a virtual target; only the shared worker sends.

No exchange client, key or transaction exists in publish(). Targets expire, bind
the observed owner inventory, and are not retried after uncertain submissions.
"""
import hashlib,json,os,time,tempfile
from pathlib import Path
from decimal import Decimal
import numpy as np
from joint_wallet.dispatch import atomic_json
from joint_wallet.book import fingerprint,ReconciliationError
from bot.execution import rounded_price


def publish(directory,*,request_id,symbol,target,expected_owned,source_time,expires_at):
    now=time.time()
    if not source_time<=now<expires_at<=source_time+900:raise ValueError('Invalid intent availability/expiry')
    if not request_id or len(request_id)>120 or not np.isfinite([target,expected_owned]).all():raise ValueError('Invalid teammate target')
    body=dict(schema='owner_target_v1',owner='teammate',request_id=request_id,symbol=symbol,
        target=float(target),expected_owned=float(expected_owned),source_time=float(source_time),expires_at=float(expires_at))
    folder=Path(directory)/'teammate_inbox';folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    path=folder/(hashlib.sha256(request_id.encode()).hexdigest()+'.json')
    packed=json.dumps(body,sort_keys=True,allow_nan=False).encode()
    fd,temp=tempfile.mkstemp(prefix='.publishing-',dir=folder)
    try:
        with os.fdopen(fd,'wb') as f:f.write(packed);f.flush();os.fsync(f.fileno())
        try:os.link(temp,path)
        except FileExistsError:
            if json.loads(path.read_text())!=body:raise ValueError('Request ID already used for different target')
    finally:os.unlink(temp)
    return path


def process(directory,executor,profile,sigma):
    """Called by the same leased writer as our candidate, never another worker."""
    executor.lease.require();folder=Path(directory)/'teammate_inbox';done=Path(directory)/'teammate_results'
    done.mkdir(parents=True,exist_ok=True,mode=0o700)
    for file in sorted(folder.glob('*.json')):
        result_path=done/file.name
        if result_path.exists():continue
        row=json.loads(file.read_text());now=time.time()
        if row['schema']!='owner_target_v1' or row['owner']!='teammate':raise ReconciliationError('Unknown target owner/schema')
        if row['symbol'] not in profile['trading']['symbols']:raise ReconciliationError('Unknown teammate market')
        if not row['source_time']<=now<row['expires_at']:
            atomic_json(result_path,dict(status='expired',orders_sent=0));continue
        book=executor.bridge.refresh();checked=executor.bridge.reconcile_exchange(book)
        j=book.symbols.index(row['symbol']);current=float(book.owners['teammate'].qty[j])
        if abs(current-row['expected_owned'])>1e-9:
            atomic_json(result_path,dict(status='owner_inventory_changed',actual_owned=current,orders_sent=0));continue
        side=1 if row['target']>current else -1
        meta,ctx=executor.bridge.public.market();coin='pas:'+row['symbol']
        quote=executor.bridge.public.post({'type':'l2Book','coin':coin});at=time.time()
        if quote['coin']!=coin or not -1000<=at*1000-quote['time']<=2000:raise ReconciliationError('Stale teammate order book')
        levels=quote['levels'][1 if side>0 else 0]
        if not levels:atomic_json(result_path,dict(status='no_liquidity',orders_sent=0));continue
        price=float(rounded_price(Decimal(levels[0]['px'])*(Decimal('1.001') if side>0 else Decimal('.999')),meta['universe'][j]['szDecimals'],side>0))
        capacity=sum(float(v['sz']) for v in levels if side*(float(v['px'])-price)<=0)
        target=current+side*min(abs(row['target']-current),capacity)
        plan=book.plan('teammate',row['symbol'],target,price,meta['universe'][j]['szDecimals'],
            max_notional=profile['execution']['maximum_single_order_notional'],slot='teammate:'+row['request_id'])
        if plan is None:atomic_json(result_path,dict(status='no_executable_delta',orders_sent=0));continue
        def preflight(p,b):
            executor.lease.require()
            if time.time()>=row['expires_at']:raise ReconciliationError('Teammate target expired before send')
            return executor.gateway.preflight(p,b)
        def send(p):executor.lease.require();return executor.gateway.send(p)
        def fills(oid,cloid):
            from competition_live.runner import history
            for _ in range(6):
                got=history(executor.bridge.public,'userFillsByTime',profile,profile['start_unix']*1000,int(time.time()*1000))
                matched={(r['coin'],r['tid']):r for r in got if r['oid']==oid and r['coin']==coin}
                if matched:return list(matched.values())
                time.sleep(1)
            return []
        result=executor.dispatch.execute(plan,checked['marks'],sigma,refresh=executor.bridge.refresh,
            preflight=preflight,send=send,confirmed_fills=fills)
        atomic_json(result_path,dict(status='confirmed_partial_or_full',request=row,result=result,
            continuation='Read actual virtual owner qty before any new request; never assume full fill.'))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--request-id',required=True)
    p.add_argument('--symbol',required=True);p.add_argument('--target',type=float,required=True);p.add_argument('--expected-owned',type=float,required=True)
    p.add_argument('--source-time',type=float,required=True);p.add_argument('--expires-at',type=float,required=True);a=p.parse_args()
    print(publish(a.directory,request_id=a.request_id,symbol=a.symbol,target=a.target,expected_owned=a.expected_owned,source_time=a.source_time,expires_at=a.expires_at))
