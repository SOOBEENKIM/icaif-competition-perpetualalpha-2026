"""Post-evaluation attribution; never changes candidate selection or orders."""
import bisect
import gzip
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.par_direct_forecast_20261017.data import save

HERE=Path(__file__).resolve().parent


def episode_attribution(run,d,scenario):
    entries=[e for e in run['events'] if e['event']=='pair_intent']
    starts=[e['time'] for e in entries]
    episodes=[dict(time=e['time'],pair=e['pair'],target=e['target'],price_move=0.,execution_cost=0.,fees=0.,funding=0.,net_pnl=0.,fills=0) for e in entries]
    fills=defaultdict(list)
    for f in run['owner_fills']['pair']:fills[f['time']].append(f)
    times=np.asarray(d['times']);total=0.
    for previous,current in zip(run['snapshots'],run['snapshots'][1:]):
        now=current['time'];i=int(np.searchsorted(times,now));j=int(np.searchsorted(times,previous['time']))
        assert times[i]==now and times[j]==previous['time']
        qty=np.asarray(previous['pair_qty']);physical=np.asarray(previous['qty'])
        price=d['par_close'][i];move=float(qty@(price-d['par_close'][j]))
        funding=0.
        if now%3600==0:
            rate=d['funding'][i];ref=d['reference_close'][i]
            rate=np.where(-physical*ref*rate>0,rate*scenario['funding_receipt_multiplier'],rate)
            funding=-float(qty@(ref*rate))
        cost=sum(f['signed_qty']*(f['price']-price[f['asset']]) for f in fills[now])
        fee=sum(f['fee'] for f in fills[now]);pnl=move-cost-fee+funding;total+=pnl
        # A new intention at this timestamp has not filled yet. Attribute the
        # just-finished interval (and any closing fill) to its preceding episode.
        k=bisect.bisect_left(starts,now)-1
        if k<0:
            assert abs(pnl)<1e-10 and not fills[now]
        else:
            e=episodes[k]
            for name,value in [('price_move',move),('execution_cost',cost),('fees',fee),('funding',funding),('net_pnl',pnl),('fills',len(fills[now]))]:e[name]+=value
    assert abs(total-run['metrics']['pair_owned_pnl'])<1e-6,(total,run['metrics']['pair_owned_pnl'])
    assert abs(total-sum(e['net_pnl'] for e in episodes))<1e-6
    for e in episodes:
        exit_events=[v for v in run['events'] if v['time']>e['time'] and v['event']=='closed']
        next_start=next((t for t in starts if t>e['time']),run['end']+1)
        exit_events=[v for v in exit_events if v['time']<=next_start]
        e['closed']=bool(exit_events);e['end']=exit_events[0]['time'] if exit_events else min(run['end'],next_start)
        e['status']='unfilled' if e['fills']==0 else ('closed' if e['closed'] else 'marked_open')
    return dict(episodes=episodes,pair_owned_pnl=total,accounting_error=abs(total-run['metrics']['pair_owned_pnl']))


def main():
    cfg=json.loads((HERE/'reference_protocol.json').read_text());d,_,_=inputs(cfg);rows=[]
    for phase in ('validation','comparison'):
        for arm in ('first_plus_pairs','second_plus_pairs','first_risk_neutral','second_risk_neutral'):
            for scenario,sc in cfg['scenarios'].items():
                with gzip.open(HERE/f'artifacts/runs/reference-{phase}-{arm}-{scenario}.json.gz','rt') as f:run=json.load(f)
                audit=episode_attribution(run,d,sc)
                rows.append(dict(phase=phase,arm=arm,scenario=scenario,**audit))
    save(HERE/'artifacts/attribution.json',dict(rows=rows,post_evaluation_diagnostic=True,not_candidate_selection=True,orders_sent=0))
    for r in rows:
        if r['phase']=='comparison' and r['arm'].startswith('first'):
            print(r['arm'],r['scenario'],[(e['pair'],round(e['net_pnl'],2),e['closed']) for e in r['episodes']])


if __name__=='__main__':main()
