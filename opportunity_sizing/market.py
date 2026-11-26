"""Information available at the source time, never future fill bars."""
import numpy as np
from .model import H,basis_adjustment


def market_inputs(at, prices, d, reference, funding):
    t=d['times']/1000;lo=np.searchsorted(t,at-3600,side='right');hi=np.searchsorted(t,at,side='right')
    n=len(prices);entry=np.zeros(n);exit=np.zeros(n);capacity=np.zeros(n);fresh=np.zeros(n,bool)
    drift=np.zeros(n);basis_se=np.zeros(n);basis_count=np.zeros(n,int);rho=np.ones(n)
    rate=np.zeros(n);funding_ready=np.zeros(n,bool)
    for j in range(n):
        ids=np.arange(lo,hi);active=ids[d['active'][ids,j]]
        if len(active):
            last=active[-1];fresh[j]=(at-t[last]<=3600)
            ranges=(d['high'][active,j]-d['low'][active,j])/d['close'][active,j]
            half_range=max(0,float(np.median(ranges))/2)
            # Same base-scenario fee/impact assumptions, plus lagged range proxy.
            # Historical L2 is unavailable: never use today's book retrospectively.
            entry[j]=.0009+.001+half_range;exit[j]=entry[j]+.0005
            past_value=float(np.sum(d['volume'][active,j]*d['close'][active,j]))
            capacity[j]=past_value*.25*.01 # Expected 15min flow at 1% participation.
        else:entry[j]=.0019;exit[j]=.0024
        known=[r for r in funding[j] if r[0]+60<=at]
        if len(known)>=6 and at-known[-1][0]<=7200:
            fast=slow=known[0][1];count=1;prev=known[0][0]
            for stamp,v in known[1:]:
                gap=(stamp-prev)/3600
                if gap>6:fast=slow=v;count=1
                else:
                    fast=(5/7)**gap*fast+(1-(5/7)**gap)*v
                    slow=(23/25)**gap*slow+(1-(23/25)**gap)*v;count+=1
                prev=stamp
            funding_ready[j]=count>=6
            if count>=6 and fast*slow>0:
                rate[j]=np.sign(slow)*min(abs(fast),abs(slow))*H*min(count/24,1.)
        hours=np.arange(at-72*3600,at+1,3600,dtype=np.int64)
        pidx=np.searchsorted(t,hours,side='right')-1
        ridx=np.searchsorted(reference['market_times'],hours)
        valid=(pidx>=0)&(ridx<len(reference['market_times']))
        b=np.full(len(hours),np.nan)
        for k in np.flatnonzero(valid):
            i,ri=pidx[k],ridx[k]
            if reference['market_times'][ri]!=hours[k]:continue
            price=d['close'][i,j];ref=reference['close'][ri,j]
            if (d['age_ms'][i,j]<=3600000 and np.isfinite(price) and price>0
                    and np.isfinite(ref) and ref>0 and reference['volume'][ri,j]>0):
                b[k]=np.log(price/ref)
        drift[j],basis_se[j],basis_count[j],rho[j]=basis_adjustment(b)
    return dict(entry_cost=entry,exit_cost=exit,capacity_notional=capacity,fresh=fresh,
                basis_log_change=drift,basis_uncertainty=basis_se,basis_count=basis_count,basis_rho=rho,
                funding_24h=rate,funding_ready=funding_ready,source_time=int(at))
