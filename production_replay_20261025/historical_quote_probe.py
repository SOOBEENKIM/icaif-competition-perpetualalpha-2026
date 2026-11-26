"""Execution diagnostics on old PAS L2 captures; NOT a portfolio backtest.

Both sides of every asset are probed independently at a fixed $1000 notional.
Overlapping hypothetical probes are not added into a profit/equity curve.
"""
import bisect,gzip,json,math
from collections import Counter
from datetime import datetime,timezone
import numpy as np
from competition_live.profile import load_profile
from experiments.basis_carry_20261017.probe import asof,sweep
from .protocol import ROOT,OUT,save,sha

PATHS=['reports/pas_venue_research_20261021/synchronized_public.jsonl.gz',
       'reports/pas_adaptive_development_20261021/book_capture/synchronized_public.jsonl.gz',
       'reports/pas_adaptive_development_20261021/book_capture_reconnect/synchronized_public.jsonl.gz']


def load(path):
    streams={};resets=[];epoch=0;lo=10**20;hi=0
    with gzip.open(path,'rt') as fp:
        for line in fp:
            r=json.loads(line)
            if r['network']!='pas':continue
            at=int(r['received']*1000);lo=min(lo,at);hi=max(hi,at)
            if r['channel'] in ['capture_error','connection_open']:
                epoch+=1;resets.append(at);continue
            if r['channel'] not in ['l2Book','activeAssetCtx']:continue
            d=r['data'];s=d['coin'][4:];f='book' if r['channel']=='l2Book' else 'ctx'
            streams.setdefault(s,dict(book=[],ctx=[]))[f].append((at,epoch,d if f=='book' else d['ctx']))
    for v in streams.values():
        v['resets']=resets
        for f in ['book','ctx']:
            v[f].sort(key=lambda r:r[0]);v[f+'_times']=[r[0] for r in v[f]]
    return streams,lo,hi


def run():
    dest=OUT/'historical_quote_probe'
    assert not (dest/'protocol.json').exists()
    save(dest/'protocol.json',dict(at=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        paths={p:sha(ROOT/p) for p in PATHS},notional=1000.,sampling_seconds=60,
        definition='All10assets, bothsigns, allvalidminuteobservations, perfileindependent probes. Compare mark+/-10bp with current executable side+/-10bp. Same newly received arrivalbook andlots/visibledepth. Noalpha decision, capitalportfolio, makerqueue or PnLperformanceclaim. Captures were previously inspected; development diagnostics.',orders_sent=0))
    profile=load_profile();cfg=json.loads((ROOT/'competition_live/strategy_plan.json').read_text())['book_protocol']
    rows=[];counts=Counter()
    for path in PATHS:
        streams,lo,hi=load(ROOT/path)
        for scenario,latency,fraction in [('base',1000,1.),('stress',3000,.5)]:
            for t in range(math.ceil(lo/60000)*60000,hi+1,60000):
                for j,symbol in enumerate(profile['trading']['symbols']):
                    s=streams.get(symbol)
                    if s is None:continue
                    decision=asof(s,t,cfg)
                    if decision is None:counts['missing_decision']+=1;continue
                    i=bisect.bisect_left(s['book_times'],t+latency);arrival=None
                    while i<len(s['book_times']) and s['book_times'][i]<=t+latency+5000:
                        at=s['book_times'][i];arrival=asof(s,at,cfg)
                        if arrival is not None:break
                        i+=1
                    if arrival is None:counts['missing_arrival']+=1;continue
                    mark=float(decision['ctx']['markPx']);decimals=profile['trading']['execution']['decimals'][j]
                    qty=math.floor(1000/mark*10**decimals)/10**decimals
                    for side in [-1,1]:
                        k=1 if side>0 else 0;best=float(decision['book']['levels'][k][0]['px'])
                        for name,anchor in [('mark_guard',mark),('side_quote_guard',best)]:
                            limit=anchor*(1+side*.001)
                            filled,price=sweep(arrival['book']['levels'][k],qty,fraction,side,decimals,limit)
                            rows.append(dict(tape=path,scenario=scenario,time=t,arrival=at,symbol=symbol,side=side,
                                policy=name,requested=qty,filled=filled,fill_fraction=filled/qty if qty else 0.,price=price,
                                mark=mark,limit=limit,decision_side_deviation_bps=side*(best/mark-1)*10000,
                                paid_mark_deviation_bps=side*(price/mark-1)*10000 if filled else None))
    summary=[]
    for scenario in ['base','stress']:
        for policy in ['mark_guard','side_quote_guard']:
            r=[x for x in rows if x['scenario']==scenario and x['policy']==policy]
            fills=[x for x in r if x['filled']]
            summary.append(dict(scenario=scenario,policy=policy,probes=len(r),filled_probes=len(fills),
                mean_fill_fraction=float(np.mean([x['fill_fraction'] for x in r])),
                mean_paid_mark_deviation_bps=float(np.mean([x['paid_mark_deviation_bps'] for x in fills])) if fills else None,
                symbols=sorted({x['symbol'] for x in r}),aggregate_portfolio_pnl=None,orders_sent=0))
    save(dest/'rows.json',rows);save(dest/'results.json',dict(summary=summary,counts=dict(counts)))
    print(summary,flush=True)


if __name__=='__main__':run()
