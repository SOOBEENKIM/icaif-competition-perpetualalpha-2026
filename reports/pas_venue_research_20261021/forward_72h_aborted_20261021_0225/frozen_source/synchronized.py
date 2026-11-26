import collections
import gzip
import json
import numpy as np
from .capture import OUT,SYMBOLS,REFERENCES,save


def summary(values):
    a=np.asarray(values,float)
    return dict(n=len(a),median=float(np.median(a)),p05=float(np.quantile(a,.05)),
        p95=float(np.quantile(a,.95)),minimum=float(a.min()),maximum=float(a.max())) if len(a) else None


def main():
    manifest=json.loads((OUT/'capture_manifest.json').read_text())
    with gzip.open(OUT/'synchronized_public.jsonl.gz','rt') as f:events=[json.loads(line) for line in f]
    events.sort(key=lambda x:x['received'])
    books={};contexts={};rows=collections.defaultdict(list);trades={};next_sample=manifest['start']
    for event in events:
        now=event['received'];network=event['network'];kind=event['channel'];data=event.get('data')
        if kind=='l2Book':books[network,data['coin']]=(now,data)
        elif kind=='activeAssetCtx':contexts[network,data['coin']]=(now,data['ctx'])
        elif kind=='trades':
            for trade in data:
                if manifest['start']<=trade['time']/1000<=manifest['end']:
                    trades[trade['coin'],trade['tid']]=trade
        if now<next_sample:continue
        next_sample=int(now)+1
        for s in SYMBOLS:
            key='pas','pas:'+s;rkey='main',REFERENCES[s]
            if any(k not in books for k in (key,rkey)) or key not in contexts:continue
            a,b=books[key];ra,rb=books[rkey];ca,c=contexts[key]
            if min(now-a,now-ra,now-ca)<0 or max(now-a,now-ra)>15 or now-ca>3:continue
            if not all(b['levels']) or not all(rb['levels']):continue
            if abs(b['time']-rb['time'])>2000:continue
            if max(now-b['time']/1000,now-rb['time']/1000)>15:continue
            bid,ask=[float(level[0]['px']) for level in b['levels']]
            refbid,refask=[float(level[0]['px']) for level in rb['levels']]
            mid=(bid+ask)/2;ref=(refbid+refask)/2;oracle=float(c['oraclePx'])
            rows[s].append(dict(time=now,mid=mid,reference=ref,oracle=oracle,
                basis_main_bps=(mid/ref-1)*10000,basis_oracle_bps=(mid/oracle-1)*10000,
                spread_bps=(ask-bid)/mid*10000,mark_basis_bps=(float(c['markPx'])/oracle-1)*10000,
                maker_two_fill_gross_less_fees_bps=(ask-bid)/mid*10000-6,
                top_notional=min(bid*float(b['levels'][0][0]['sz']),ask*float(b['levels'][1][0]['sz'])),
                server_gap_ms=abs(b['time']-rb['time'])))
    result=[]
    for s in SYMBOLS:
        data=rows[s];coin='pas:'+s
        ts=sorted(t['time']/1000 for (c,_),t in trades.items() if c==coin)
        result.append(dict(symbol=s,aligned_seconds=len(data),
            main_basis_bps=summary([r['basis_main_bps'] for r in data]),
            oracle_basis_bps=summary([r['basis_oracle_bps'] for r in data]),
            spread_bps=summary([r['spread_bps'] for r in data]),
            maker_roundtrip_margin_before_adverse_selection_bps=summary([r['maker_two_fill_gross_less_fees_bps'] for r in data]),
            top_notional=summary([r['top_notional'] for r in data]),
            actual_new_trades=len(ts),mean_intertrade_seconds=float(np.mean(np.diff(ts))) if len(ts)>1 else None,
            aligned_server_gap_ms=summary([r['server_gap_ms'] for r in data])))
    save(OUT/'synchronized_analysis.json',dict(start=manifest['start'],end=manifest['end'],
        markets=result,unique_new_trades=len(trades),capture_errors=manifest['errors'],
        comparison='Time-aligned received books with server timestamp separation <=2 seconds',
        caveat='Repeated seconds are not independent statistical samples; maker margins assume both fills and exclude inventory risk.'))
    save(OUT/'aligned_seconds.json',dict(rows=rows,trades=list(trades.values())))
    print(json.dumps(dict(unique_new_trades=len(trades),markets=result)))


if __name__=='__main__':
    main()
