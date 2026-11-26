"""Fixed forecast mixture; no fitted weights or PAS target labels."""
import numpy as np
from .scaled_coherence import project


def build(data,source):
    arms=['coherent_blend_72','coherent_trend_72']
    states={a:{k:np.array(v,copy=True) for k,v in source.items()} for a in arms}
    t=data['market_times'];prices=data['close'];rows=np.searchsorted(t,source['times'])
    assert np.array_equal(t[rows],source['times'])
    for i,row in enumerate(rows):
        if row<168:
            for s in states.values():s['economic_available'][i]=False;s['economic_action'][i]=0
            continue
        asset=np.expm1(np.log(prices[row]/prices[row-168])*72/168)
        pairs=source['pairs'];beta=source['beta'][i];valid=source['economic_available'][i]
        unit=source['vol'][i]*np.sqrt(72)
        trend=np.array([(asset[a]-b*asset[c])/(1+b) for (a,c),b in zip(pairs,beta)])
        trend=np.clip(trend,-4*unit,4*unit)
        for arm,s in states.items():
            raw=trend if arm=='coherent_trend_72' else .5*trend+.5*source['mu'][i]
            mu,_=project(raw,beta,pairs,valid,unit)
            s['mu'][i]=mu;s['economic_action'][i]=np.where(valid&(abs(mu)>.0038),np.sign(np.nan_to_num(mu)),0).astype(int)
    return states
