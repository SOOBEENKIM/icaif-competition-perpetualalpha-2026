"""Causal peer relationships: every unordered pair is scored, without names.

Correlation is a formation criterion, not proof of stationarity or alpha.
The separate return model and executable-net-cash planner still choose trades.
"""
import numpy as np
from auto_pair_selection.selection import IDS, PAIRS, validate_snapshot


def evaluate(data, source, neighbors=2):
    times=np.asarray(data['market_times']);end=np.searchsorted(times,source,side='right')
    prefix={k:np.asarray(data[k])[:end] for k in ['market_times','close','volume','trades']}
    t=validate_snapshot(prefix)
    if t[-1]!=source or source%86400:raise ValueError('Missing completed daily source')
    close,volume,trades=[prefix[k] for k in ['close','volume','trades']]
    returns=np.diff(np.log(close),axis=0);active=(volume>0)&(trades>0)
    fresh=active[1:]&active[:-1];scores=np.full(len(PAIRS),-np.inf);rows=[]
    for k,(a,b) in enumerate(IDS):
        blocks=[];reasons=[]
        for block in range(3):
            lo=source-(3-block)*30*86400;hi=lo+30*86400
            use=(t[1:]>lo)&(t[1:]<=hi)&fresh[:,a]&fresh[:,b];n=int(use.sum())
            corr=None;liquid=None
            if n>=2:
                x,y=returns[use,a],returns[use,b];x=x-x.mean();y=y-y.mean()
                den=np.sqrt((x@x)*(y@y))
                corr=float(x@y/den) if den>1e-20 else None
                liquid=float(np.median(volume[1:][use][:,[a,b]]*close[1:][use][:,[a,b]],axis=0).min())
            if n<.8*720:reasons.append('insufficient_synchronous_fresh_history')
            if corr is None or corr<.2:reasons.append('weak_or_negative_relationship')
            if liquid is None or liquid<4000:reasons.append('low_reference_liquidity')
            blocks.append(dict(rows=n,correlation=corr,median_leg_notional=liquid))
        if not reasons:scores[k]=min(x['correlation'] for x in blocks)
        rows.append(dict(pair_index=k,pair=list(PAIRS[k]),blocks=blocks,reasons=sorted(set(reasons))))
    # A sparse graph of mutual peers, NOT a fixed number of allowed pairs.
    adjacency=np.full((10,10),-np.inf)
    for k,(a,b) in enumerate(IDS):adjacency[a,b]=adjacency[b,a]=scores[k]
    chosen=np.zeros_like(adjacency,dtype=bool)
    for a in range(10):
        values=adjacency[a,np.isfinite(adjacency[a])]
        if len(values):
            cutoff=np.sort(values)[max(0,len(values)-neighbors)]
            chosen[a]=adjacency[a]>=cutoff # Include exact ties; no identity tie-break.
    stable=np.isfinite(scores);mutual=np.array([chosen[a,b]&chosen[b,a] for a,b in IDS])&stable
    for k,row in enumerate(rows):
        row.update(stable=bool(stable[k]),mutual_peer=bool(mutual[k]),score=float(scores[k]) if stable[k] else None)
    return dict(source=int(source),known_through=int(t[-1]),universe_size=45,stable=stable.tolist(),mutual=mutual.tolist(),pairs=rows)
