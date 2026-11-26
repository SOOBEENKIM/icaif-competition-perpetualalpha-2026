"""Unsigned migration rehearsal from public fills and existing own order IDs."""
import json
import time
from pathlib import Path
import numpy as np
from .book import JointBook,ReconciliationError,fingerprint


def reconstruct(symbols,capital,fills,funding,own_cloids,own_oids=(),*,settled_units=None):
    b=JointBook(symbols,capital);events=[];seen=set();settled_units=settled_units or {}
    for sequence,row in enumerate(fills):
        if not row['coin'].startswith('pas:'):continue
        key=row['coin'],row['tid']
        if key in seen:continue
        seen.add(key)
        owner='ours' if row.get('cloid') in own_cloids or row['oid'] in own_oids else 'teammate'
        # Trade IDs are identities, NOT chronological sequence numbers.
        events.append((row['time'],1,sequence,owner,row))
    seen=set()
    for row in funding:
        if not row['delta'].get('coin','').startswith('pas:'):continue
        key=fingerprint(row)
        if key in seen:continue
        seen.add(key);events.append((row['time'],0,0,key,row))
    # At a zero-net settlement the exchange can omit the wallet funding row.
    # Still create the opposing virtual allocations using observed per-unit
    # settlement data; absence is an explicit unresolved accounting condition.
    # Settlement publication occurs milliseconds after the hour; matching
    # against an exact wall-clock boundary fabricates missing payments.
    covered={(r['delta']['coin'],r['time']//3600000) for r in funding}
    if events:
        lo=min(e[0] for e in events);hi=max(e[0] for e in events)
        for at in range((lo//3600000+1)*3600000,hi+1,3600000):
            for s in symbols:
                coin='pas:'+s
                if (coin,at//3600000) not in covered:events.append((at,0,-1,'absent',dict(coin=coin)))
    for at,kind,_,owner,row in sorted(events,key=lambda e:e[:3]):
        if kind:b.apply_fill(owner,row)
        elif owner=='absent':
            j=symbols.index(row['coin'][4:])
            if not any(abs(x.qty[j])>1e-12 for x in b.owners.values()):continue
            if abs(b.physical.qty[j])>1e-12:raise ReconciliationError('Missing hourly wallet funding event')
            b.apply_funding(row['coin'],at,0.,0.,'observed_zero_net:'+str(at),cash_per_unit=settled_units.get((row['coin'],at)))
        else:
            d=row['delta'];b.apply_funding(d['coin'],at,float(d['usdc']),float(d['szi']),owner,
                cash_per_unit=settled_units.get((d['coin'],at)))
    return b


def main():
    from competition_live.profile import ROOT,load_profile,verify_deployment
    from competition_live.venue import PublicPAS,account_equity
    from competition_live.runner import history
    p=load_profile();verify_deployment();public=PublicPAS(p)
    out=ROOT/'reports/joint_wallet_20261022';out.mkdir(exist_ok=True)
    journal=ROOT/'.runtime/pas_competition_live/orders.jsonl'
    rows=[json.loads(line) for line in journal.read_text().splitlines()]
    cloids={e['order']['cloid'] for e in rows if e.get('event')=='submit_intent'}
    oids=set()
    # Legacy orders all carry cloids, but retain explicit receipt identity as a
    # fallback for public fill API versions that omit the optional cloid field.
    for e in rows:
        if e.get('event')=='submit_response':
            for s in e.get('response',{}).get('response',{}).get('data',{}).get('statuses',[]):
                if 'filled' in s:oids.add(s['filled']['oid'])
    end=int(time.time()*1000);start=p['start_unix']*1000
    fills=history(public,'userFillsByTime',p,start,end)
    funding=history(public,'userFunding',p,start,end)
    for name,value in [('public_fills.json',fills),('public_funding.json',funding)]:
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False))
    b=reconstruct(p['trading']['symbols'],{'ours':p['initial_capital'],'teammate':p['reserved_capital']},fills,funding,cloids,oids)
    meta,ctx=public.market();snap=public.account();marks=np.array([float(c['markPx']) for c in ctx])
    actual=np.zeros(len(b.symbols))
    for r in snap['positions']['assetPositions']:
        v=r['position'];j=b.symbols.index(v['coin'][4:]);actual[j]=float(v['szi'])
        if actual[j]:marks[j]=float(v['entryPx'])+float(v['unrealizedPnl'])/actual[j]
    if not np.allclose(actual,b.physical.qty,rtol=0,atol=1e-8):raise ReconciliationError('Live snapshot raced with fills; rerun audit')
    errors=b.reconcile(marks);observed=account_equity(snap,p)['equity']
    if abs(observed-b.physical.equity(marks))>.05:raise ReconciliationError('USDP equity differs; check transfers or funding race')
    result=dict(at=time.time(),mode='unsigned_public_history_reconstruction',orders_sent=0,signer_loaded=False,
        symbols=b.symbols,equity=observed,ledger_equity=b.physical.equity(marks),physical_qty=b.physical.qty.tolist(),
        owners={name:dict(initial=l.initial,qty=l.qty.tolist(),entry=l.entry.tolist(),equity=l.equity(marks),
            pnl=l.equity(marks)-l.initial,fees=l.fees,funding=l.funding,realized=l.realized) for name,l in b.owners.items()},
        conservation_errors=errors,classification='Known own journal cloids/order IDs are ours; all other confirmed PAS fills are the external teammate sleeve. No strategy identity is inferred from position direction.',
        fill_count=len(fills),funding_count=len(funding),open_orders=snap['orders'],live_changed=False)
    for name,value in [('audit.json',result),('book.json',b.serialize()),('public_fills.json',fills),('public_funding.json',funding)]:
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False))
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
