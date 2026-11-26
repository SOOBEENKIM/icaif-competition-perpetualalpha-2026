"""Independent mark-to-market cash recurrence; daily samples after settlement."""
from collections import defaultdict
import numpy as np


def audit(run,market,protocol):
    fills=defaultdict(list);funding={r['time']:r for r in run['funding_events']}
    if len(funding)!=len(run['funding_events']):raise AssertionError('Duplicate settlement')
    for f in run['fills']:fills[f['time']].append(f)
    q=np.zeros(len(market.symbols));equity=float(protocol['initial']);maximum=0.;days={}
    first=market.exact(run['start']);last=market.exact(run['end']);obligations=0
    snapshots={s['time']:s for s in run['snapshots']}
    fees=paid_funding=move_pnl=execution_cost=0.
    for i in range(first,last+1):
        t=int(market.times[i]);prices=market.close[i]
        if i>first:
            move=float(q@(prices-market.close[i-1]));equity+=move;move_pnl+=move
            if np.any(q):
                obligations+=1
                f=funding[t];np.testing.assert_allclose(q,f['qty'],rtol=0,atol=1e-8)
                amount=-float(q@(prices*market.funding[i]))-float(abs(q)@prices)*protocol['scenarios'][run['scenario']]['adverse_funding_bps_hour']/10000
                np.testing.assert_allclose(amount,f['signed_amount']+f['adverse_amount'],rtol=0,atol=1e-7)
                equity+=amount;paid_funding+=amount
            elif t in funding:raise AssertionError('Funding on empty book')
        for f in fills[t]:
            if not f['source_available']<=f['decision_time']<t:raise AssertionError('Future data')
            fee=abs(f['signed_qty'])*f['price']*protocol['execution']['fee_bps']/10000
            assert abs(fee-f['fee'])<1e-7
            cost=f['signed_qty']*(f['price']-prices[f['asset']])
            equity-=cost+fee;execution_cost+=cost;fees+=fee;q[f['asset']]+=f['signed_qty']
            q[abs(q)<1e-10]=0
        if t%86400==0:days[t]=equity
        maximum=max(maximum,abs(equity-snapshots[t]['equity']))
        np.testing.assert_allclose(q,snapshots[t]['qty'],atol=1e-8,rtol=0)
    assert maximum<1e-5 and obligations==len(funding)
    daily={t:days[t]/days[t-86400]-1 for t in days if t-86400 in days}
    assert len(daily)==len(run['daily'])
    assert all(abs(daily[d['time']]-d['return_'])<1e-10 for d in run['daily'])
    if len(daily)>1:
        assert abs(np.std(list(daily.values()),ddof=1)-run['metrics']['daily_std_sample'])<1e-10
    assert abs(equity-protocol['initial']-run['metrics']['net_pnl'])<1e-5
    return {'passed':True,'maximum_equity_difference':maximum,'snapshots':len(snapshots),
            'price_move_pnl':move_pnl,'execution_cost':execution_cost,'fees':fees,'funding':paid_funding}
