"""Conservative OI preview also covering a net position reversal.

Closing our old long need not reduce total OI: a counterparty can take it over.
The subsequently opened short can therefore add its FULL size to global OI,
even when our final absolute position is unchanged. No live hook on import.
"""
from decimal import Decimal
from competition_live.venue import check_oi as legacy_check_oi


def opening_quantity(old,delta):
    return max(Decimal(0),abs(delta)-abs(old)) if old*delta<0 else abs(delta)


def check_oi_strict(orders,positions,meta,contexts,limits):
    legacy_check_oi(orders,positions,meta,contexts,limits)
    ctx={m['name']:c for m,c in zip(meta['universe'],contexts)}
    valuation={c:max(Decimal(v['oraclePx']),Decimal(v.get('markPx',v['oraclePx']))) for c,v in ctx.items()}
    current={c:Decimal(v['openInterest'])*valuation[c] for c,v in ctx.items()}
    caps={c:Decimal(v) for c,v in limits['coinToOiCap']}
    qty={p['position']['coin']:Decimal(p['position']['szi']) for p in positions}
    added={c:Decimal(0) for c in ctx}
    for o in orders:
        old=qty.get(o.coin,Decimal(0));delta=Decimal(o.size)*(1 if o.is_buy else -1)
        added[o.coin]+=opening_quantity(old,delta)*valuation[o.coin]
        qty[o.coin]=old+delta
    for coin,increment in added.items():
        if increment>0 and current[coin]+increment>caps[coin]:raise ValueError('Market OI cap including reversal: '+coin)
    if sum(added.values())>0 and sum(current.values())+sum(added.values())>Decimal(limits['totalOiCap']):
        raise ValueError('DEX OI cap including reversal')
    return dict(current_oi=float(sum(current.values())),increment_upper_bound=float(sum(added.values())))
