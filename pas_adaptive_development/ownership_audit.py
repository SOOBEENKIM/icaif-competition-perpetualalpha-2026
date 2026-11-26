"""Read-only attribution of confirmed public fills; never repairs live state."""
import json
from pathlib import Path
from .study import ROOT, save

OUT=ROOT/'reports/incumbent_constraints_review_20261021'


def reconcile(fills, funding, own_cloids, marks, cutoff_ms):
    owners={name:dict(cash=0.,fees=0.,funding=0.,qty={},fill_count=0,orders=set())
            for name in ['our_program','unattributed']}
    seen=set();events=[]
    for n,f in enumerate(fills):
        if f['time']>cutoff_ms or not f['coin'].startswith('pas:'):continue
        key=f['coin'],f['tid']
        if key in seen:raise ValueError('Duplicate fill in audit input')
        seen.add(key);events.append((f['time'],1,n,f))
    seen=set()
    for n,f in enumerate(funding):
        if f['time']>cutoff_ms or not f['delta']['coin'].startswith('pas:'):continue
        key=json.dumps(f,sort_keys=True)
        if key in seen:raise ValueError('Duplicate funding in audit input')
        seen.add(key);events.append((f['time'],0,n,f))
    for time,kind,_,event in sorted(events):
        if kind:
            coin=event['coin'];owner=owners['our_program' if event.get('cloid') in own_cloids else 'unattributed']
            before=sum(o['qty'].get(coin,0.) for o in owners.values())
            if abs(before-float(event['startPosition']))>1e-7:
                raise ValueError('Incomplete fill history: startPosition differs')
            q=float(event['sz'])*(1 if event['side']=='B' else -1);px=float(event['px']);fee=float(event['fee'])
            if event['feeToken']!='USDP':raise ValueError('Unexpected fee collateral')
            owner['qty'][coin]=owner['qty'].get(coin,0.)+q
            owner['cash']-=q*px+fee;owner['fees']+=fee
            owner['fill_count']+=1;owner['orders'].add(event['oid'])
        else:
            d=event['delta'];coin=d['coin'];net=sum(o['qty'].get(coin,0.) for o in owners.values())
            if any(f['time']==time and f['coin']==coin for f in fills):
                raise ValueError('Funding/fill ordering ambiguous at identical timestamp')
            if abs(net-float(d['szi']))>1e-7:raise ValueError('Funding inventory does not reconcile')
            payment=float(d['usdc'])
            if abs(net)<1e-12 and abs(payment)>1e-10:raise ValueError('Payment without inventory')
            for owner in owners.values():
                share=payment*owner['qty'].get(coin,0.)/net if net else 0.
                owner['cash']+=share;owner['funding']+=share
    for owner in owners.values():
        missing=[s for s,q in owner['qty'].items() if abs(q)>1e-9 and s not in marks]
        if missing:raise ValueError('Unpriced inventory: '+str(missing))
        owner['net_pnl']=owner['cash']+sum(q*marks.get(s,0.) for s,q in owner['qty'].items())
        owner['gross']=sum(abs(q)*marks.get(s,0.) for s,q in owner['qty'].items())
        owner['orders']=sorted(owner['orders'])
    return owners


def main():
    def raw(n):return json.loads((OUT/'raw'/f'{n}.json').read_text())
    positions=raw('latest_own_positions')['response'];cutoff=positions['time']
    marks={p['position']['coin']:float(p['position']['positionValue'])/abs(float(p['position']['szi']))
           for p in positions['assetPositions']}
    events=[json.loads(s) for s in (ROOT/'.runtime/pas_competition_live/orders.jsonl').read_text().splitlines()]
    cloids={e['order']['cloid'] for e in events if e['event']=='submit_intent'}
    owners=reconcile(raw('latest_own_fills')['response'],raw('latest_own_userFunding')['response'],cloids,marks,cutoff)
    for p in positions['assetPositions']:
        pos=p['position'];q=sum(o['qty'].get(pos['coin'],0.) for o in owners.values())
        assert abs(q-float(pos['szi']))<1e-8
    spot=next(float(b['total']) for b in raw('latest_own_spot')['response']['balances'] if b['coin']=='USDP')
    transfers=raw('latest_own_userNonFundingLedgerUpdates')['response']
    assert len(transfers)==1 and transfers[0]['delta']['type']=='spotTransfer'
    d=transfers[0]['delta']
    assert d['token']=='USDP' and d['destination'].lower()==raw('latest_own_fills')['request']['user'].lower()
    capital=float(d['amount']);assert capital==100000.
    net=sum(o['net_pnl'] for o in owners.values())
    result=dict(cutoff_ms=cutoff,owners=owners,net_deposits=capital,reconstructed_net_pnl=net,
        spot_net_pnl=spot-capital,read_timing_difference=(spot-capital)-net,
        live_state_changed=False,orders_sent=0,unattributed_is_not_confirmed_teammate=True,
        scope='Offline cash-flow attribution only. Funding is apportioned by signed inventory at its actual settlement. No daily marks or standalone official scores inferred.')
    save(OUT/'ownership_attribution.json',result);print(json.dumps(result))


if __name__=='__main__':main()
