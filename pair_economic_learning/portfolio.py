"""Current-bar coordination of independent funded pair sleeves.

Trading cost and participation allowances are split, not multiplied. Opposite
virtual holdings pay their own fees and stressed funding receipts; this is a
conservative separate-sleeve simulation, not a netting windfall.
"""
import copy
import numpy as np
from .cohort_engine import replay_steps
from replay_lab.common_comparison.engine import risk_breaches,risk_values


def replay(d,s,cfg,protocol,scenario,start,end,*,count=3,disjoint_assets=True):
    if count<1:raise ValueError('Positive sleeve count required')
    p=copy.deepcopy(protocol);p['initial']/=count;p['risk']['asset_notional']/=count
    c=copy.deepcopy(cfg);c['pair_max_gross']/=count;c['scenarios'][scenario]['participation']/=count
    gens=[replay_steps(d,s,None,c,p,'pairs_dense_hourlyhedge_individual_72',scenario,start,end) for _ in range(count)]
    for g in gens:assert next(g) is None
    active=[None]*count;audits=[];children=[]
    times=d['times'];first=int(np.searchsorted(times,start));last=int(np.searchsorted(times,end))
    for i in range(first,last+1):
        now=int(times[i]);bar=[]
        for k,g in enumerate(gens):
            other={a for j,a in enumerate(active) if j!=k and a is not None}
            assets={int(x) for a in other for x in s['pairs'][a]}
            blocked={j for j,pair in enumerate(s['pairs']) if j in other or (disjoint_assets and assets.intersection(pair))}
            update=g.send(blocked);assert update['snapshot']['time']==now
            active[k]=update['active_pair'];bar.append(update['snapshot'])
            try:assert next(g) is None
            except StopIteration as done:
                assert i==last;children.append(done.value)
        if disjoint_assets:
            flattened=[int(x) for a in active if a is not None for x in s['pairs'][a]]
            assert len(flattened)==len(set(flattened)),'Coordinator allowed overlapping active assets'
        audits.append(dict(time=now,active=list(active),qty=np.sum([x['qty'] for x in bar],axis=0).tolist(),equity=sum(x['equity'] for x in bar)))
    assert len(children)==count
    if count==1:return children[0]
    snapshots=[];breaches=[];fees=0.;funding=0.;totals={}
    for i,audit in enumerate(audits):
        row=[x['snapshots'][i] for x in children];idx=first+i;qty=np.array(audit['qty']);equity=audit['equity'];prices=d['par_close'][idx]
        rr=risk_breaches(qty,equity,prices,d['sigma'][idx],protocol['risk'])
        if rr:breaches.append(dict(time=audit['time'],limits=rr))
        snapshots.append(dict(time=audit['time'],equity=equity,cash=sum(x['cash'] for x in row),qty=qty.tolist(),
             fees=sum(x['fees'] for x in row),funding=sum(x['funding'] for x in row),base_qty=[0.]*10,pair_qty=qty.tolist(),
             **risk_values(qty,equity,prices,d['sigma'][idx])))
    def flatten(key):
        return sorted([dict(cohort=k,**x) for k,r in enumerate(children) for x in r[key]],key=lambda x:(x.get('time',x.get('decision_time',0)),x['cohort']))
    fills=flatten('fills');curve=np.array([x['equity'] for x in snapshots]);net=float(curve[-1]-protocol['initial'])
    for key in ['fees','funding','execution_cost','price_move_pnl','completed_pairs','aborted_pairs','hedge_repairs','unbalanced_minutes','liquidation_cost_estimate','renewals']:
        totals[key]=sum(r['metrics'].get(key,0) for r in children)
    decomposition=abs(net-totals['price_move_pnl']+totals['execution_cost']+totals['fees']-totals['funding'])
    assert decomposition<1e-6
    # Split participation quotas bound total executed volume by the original
    # per-bar allowance, including independent opposing child orders.
    volume=np.zeros_like(d['par_volume'])
    symbols=protocol['symbols']
    for fill in fills:volume[np.searchsorted(times,fill['time']),symbols.index(fill['symbol'])]+=abs(fill['signed_qty'])
    assert np.all(volume<=cfg['scenarios'][scenario]['participation']*d['par_volume']+1e-8)
    assert all(r['metrics']['risk_breach_bars']==0 for r in children)
    metrics=dict(**totals,net_pnl=net,roi=net/protocol['initial'],base_owned_pnl=0.,pair_owned_pnl=net,
        pnl_after_liquidation_reserve=net-totals['liquidation_cost_estimate'],max_drawdown=float(-np.min(curve/np.maximum.accumulate(curve)-1)),
        mean_gross=float(np.mean([s['gross'] for s in snapshots])),maximum_gross=max(s['gross'] for s in snapshots),
        maximum_accounting_error=max(decomposition,*[r['metrics']['maximum_accounting_error'] for r in children]),
        risk_breach_bars=len(breaches),fills=len(fills),orders=sum(r['metrics']['orders'] for r in children),residual_gross=sum(r['metrics']['residual_gross'] for r in children))
    daily=[x for x in snapshots if x['time']%86400==43200]
    return dict(arm='coordinated_pair_sleeves',scenario=scenario,start=start,end=end,metrics=metrics,snapshots=snapshots,
        fills=fills,orders=flatten('orders'),decisions=flatten('decisions'),events=flatten('events'),funding_events=flatten('funding_events'),
        daily=[dict(time=b['time'],return_=b['equity']/a['equity']-1) for a,b in zip(daily,daily[1:])],
        owner_fills={'base':[],'pair':fills},internal_transfers=[],risk_breaches=breaches,coordinator=audits,children=children,
        split_participation_verified=True,disjoint_assets=disjoint_assets,orders_sent=0)
