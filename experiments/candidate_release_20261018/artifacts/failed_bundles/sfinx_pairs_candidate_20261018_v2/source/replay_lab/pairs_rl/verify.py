"""Independent hourly marked-cash recurrence and full funding obligation audit."""
from collections import defaultdict
import gzip
import json
import numpy as np
from .public_data import DEST,load_market
from ..common_comparison.experiment import ROOT,write,digest


def independent_accounting(run,market):
    count=len(run['metrics']['final_qty'])
    qty=np.zeros(count);equity=1e6;fees=0.;funding_total=0.;maximum=0.
    price_move_pnl=execution_adversity=0.
    fills=defaultdict(list);funding={}
    for f in run['fills']:
        fills[f['time']].append(f)
    for f in run['funding_events']:
        if f['time'] in funding:
            raise AssertionError('Duplicate hourly funding settlement')
        funding[f['time']]=f
    first=market.exact(run['start']);last=market.exact(run['end']);days={}
    snapshots={s['time']:s for s in run['snapshots']}
    if len(snapshots)!=last-first+1:
        raise AssertionError('Missing hourly account snapshot')
    obligations=0;observed_hours=missing_hours=0
    p=json.loads((ROOT/'replay_lab/pairs_rl/protocol.json').read_text())
    surcharge=p['scenarios'][run['scenario']]['adverse_funding_bps_hour']/10000
    for i in range(first,last+1):
        t=int(market.times[i]);prices=market.close[i,:count]
        if i>first:
            move=float(qty@(prices-market.close[i-1,:count]))
            equity+=move;price_move_pnl+=move
        if t%86400==0:
            days[t]=equity
        if i>first and np.any(qty):
            obligations+=1
            if t not in funding:
                raise AssertionError('Held inventory has no funding settlement')
            f=funding[t]
            np.testing.assert_allclose(f['qty'],qty,atol=1e-8,rtol=0)
            np.testing.assert_allclose(f['prices'],prices,atol=1e-10,rtol=0)
            np.testing.assert_allclose(f['rates'],market.funding[i,:count],atol=1e-15,rtol=0)
            actual=-float((qty*prices*market.funding[i,:count]).sum())
            adverse=-float((abs(qty)*prices).sum())*surcharge
            if abs(actual-f['signed_amount'])>1e-6 or abs(adverse-f['adverse_amount'])>1e-6:
                raise AssertionError('Funding amount mismatch')
            equity+=actual+adverse;funding_total+=actual+adverse
            observed_hours+=int(np.sum((qty!=0)&market.observed[i,:count]))
            missing_hours+=int(np.sum((qty!=0)&~market.observed[i,:count]))
        elif t in funding:
            raise AssertionError('Funding paid without beginning-of-hour inventory')
        for f in fills[t]:
            j=f['asset'];dq=f['signed_qty']
            if not f['source_available']<=f['decision_time']<t:
                raise AssertionError('Noncausal execution')
            expected_fee=abs(dq)*f['price']*.0009
            if abs(expected_fee-f['fee'])>1e-7:
                raise AssertionError('Fee differs from nine basis points')
            equity-=dq*(f['price']-prices[j])+expected_fee
            execution_adversity+=dq*(f['price']-prices[j])
            qty[j]+=dq
            # Exchange lot arithmetic may otherwise leave binary cancellation dust.
            if abs(qty[j])<1e-10:
                qty[j]=0.
            fees+=expected_fee
        maximum=max(maximum,abs(equity-snapshots[t]['equity']))
        np.testing.assert_allclose(qty,snapshots[t]['qty'],rtol=0,atol=1e-8)
    if maximum>1e-5:
        raise AssertionError(f'Independent marked-cash recurrence failed: {maximum}')
    daily={t:days[t]/days[t-86400]-1 for t in days if t-86400 in days}
    if len(daily)!=len(run['daily']):
        raise AssertionError('Daily boundary count mismatch')
    for d in run['daily']:
        if abs(daily[d['time']]-d['return_'])>1e-10:
            raise AssertionError('Daily return mismatch')
    std=float(np.std(list(daily.values()),ddof=0))
    if abs(std-run['metrics']['daily_vol'])>1e-10:
        raise AssertionError('Daily population standard deviation mismatch')
    if abs(equity/1e6-1-run['metrics']['roi'])>1e-10:
        raise AssertionError('Fixed initial capital ROI mismatch')
    assert obligations==len(funding)
    assert observed_hours==run['metrics']['observed_position_hours']
    assert missing_hours==run['metrics']['missing_position_hours']
    return dict(passed=True,maximum_equity_difference=maximum,checked_snapshots=len(snapshots),
        funding_obligations=obligations,checked_fills=sum(map(len,fills.values())),daily_boundaries=len(days),
        missing_position_funding_hours=missing_hours,
        fixed_trajectory_decomposition=dict(price_move_pnl=price_move_pnl,execution_adversity=execution_adversity,
            fees=fees,funding=funding_total,net_pnl=equity-1e6,
            interpretation='Same realized order path decomposed; not a zero-cost strategy rerun.'))


def main():
    m,_,_=load_market();results=json.loads((DEST/'results.json').read_text());checks={}
    for row in results['routes']:
        with gzip.open(DEST/row['file'],'rt') as f:
            run=json.load(f)
        checks[row['file']]=independent_accounting(run,m)
    manifest=json.loads((DEST/'training_manifest.json').read_text())
    changed=[name for name,sha in manifest['sources'].items() if digest(ROOT/name)!=sha]
    write(DEST/'independent_accounting.json',dict(routes=checks,training_source_changes=changed,
        passed=all(c['passed'] for c in checks.values()) and not changed))
    assert not changed
    print('Independent accounting passed:',len(checks),'routes,',sum(v['checked_snapshots'] for v in checks.values()),'hourly snapshots')


if __name__=='__main__':
    main()
