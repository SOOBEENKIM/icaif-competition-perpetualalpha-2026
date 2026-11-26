"""Reconcile directional marked PnL, fees and funding to the account ledger."""
import numpy as np


def attribute(result,data,scenario):
    times=data['times'];snaps=result['snapshots']
    by_time={s['time']:s for s in snaps}
    report={}
    for owner,key in [('physical','qty'),('base','base_qty'),('pair','pair_qty')]:
        buckets={side:dict(price_pnl=0.,fees=0.,execution_cost=0.,funding=0.)
                 for side in ['long','short']}
        for before,after in zip(snaps,snaps[1:]):
            a=int(np.searchsorted(times,before['time']))
            b=int(np.searchsorted(times,after['time']))
            qty=np.array(before[key]);physical=np.array(before['qty'])
            move=qty*(data['par_close'][b]-data['par_close'][a])
            rates=data['funding'][b].copy()
            physical_cash=-physical*data['reference_close'][b]*rates
            rates=np.where(physical_cash>0,rates*scenario['funding_receipt_multiplier'],rates)
            funding=-qty*data['reference_close'][b]*rates
            for side,mask in [('long',qty>0),('short',qty<0)]:
                buckets[side]['price_pnl']+=float(move[mask].sum())
                buckets[side]['funding']+=float(funding[mask].sum())
        fills=result['fills'] if owner=='physical' else result['owner_fills'][owner]
        for fill in fills:
            quantity=fill['signed_qty'];before=fill['position_before']
            closing=min(abs(before),abs(quantity)) if before*quantity<0 else 0.
            opening=abs(quantity)-closing
            index=int(np.searchsorted(times,fill['time']))
            impact=quantity*(fill['price']-data['par_close'][index,fill['asset']])
            for amount,direction in [(closing,np.sign(before)),(opening,np.sign(quantity))]:
                if amount<=0:continue
                side='long' if direction>0 else 'short'
                fraction=amount/abs(quantity)
                buckets[side]['fees']+=fill['fee']*fraction
                buckets[side]['execution_cost']+=impact*fraction
        for row in buckets.values():
            row['net_pnl']=row['price_pnl']+row['funding']-row['fees']-row['execution_cost']
        expected=result['metrics']['net_pnl' if owner=='physical' else owner+'_owned_pnl']
        error=sum(r['net_pnl'] for r in buckets.values())-expected
        assert abs(error)<1e-6,(owner,error)
        report[owner]=dict(sides=buckets,reconciliation_error=error)
    return report
