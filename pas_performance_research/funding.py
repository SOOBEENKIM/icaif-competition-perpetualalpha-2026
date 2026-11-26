"""Venue-specific, past-only funding forecasts. No account or exchange client."""
import numpy as np

def forecast(times, observations, symbols, horizon=168., receipt=0.5, decay=False):
    """Observations use known seconds; settlements become usable after60s.

    Min6 observed hours; 6h/24h exponential estimates must agree. Confidence
    ramps to24 actual observations. Forecast horizon matches retained weekly
    model; optional48h decay is a deliberately adverse persistence test.
    """
    result=np.zeros((len(times),len(symbols)));counts=np.zeros_like(result,dtype=int)
    effective=horizon if not decay else (1-np.exp(-horizon*np.log(2)/48))/(np.log(2)/48)
    for j,s in enumerate(symbols):
        rows=sorted([r for r in observations if r['symbol']==s],key=lambda r:r['known'])
        cursor=0;fast=slow=0.;n=0;last=-np.inf
        for i,t in enumerate(times):
            while cursor<len(rows) and rows[cursor]['known']<=t:
                row=rows[cursor];rate=float(row['rate']);now=float(row['known'])
                if now<=last:raise ValueError('Repeated/nonincreasing funding timestamp')
                if n==0 or now-last>6*3600:fast=slow=rate;n=1
                else:
                    h=(now-last)/3600
                    fast=(1-2/7)**h*fast+(1-(1-2/7)**h)*rate
                    slow=(1-2/25)**h*slow+(1-(1-2/25)**h)*rate
                    n+=1
                last=now;cursor+=1
            counts[i,j]=n
            if n>=6 and t-last<=2*3600 and fast*slow>0:
                result[i,j]=np.sign(slow)*min(abs(fast),abs(slow))*effective*min(n/24,1.)
    return result,counts

def filtered_anchor(anchor, protected, price, carry, mode):
    sign=np.sign(anchor)
    edge=sign*price-np.where(sign*carry>0,sign*carry,.5*sign*carry)
    g=np.nan_to_num(np.clip((edge-.0043)/.0043,0,1),nan=0.)
    filtered=anchor*g
    # Veto applies to the core thesis, and cannot be immediately refilled.
    guard=protected|((anchor!=0)&(g<1))
    return filtered,guard
