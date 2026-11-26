"""Pair-identity based mapping; original frozen fit coefficients preserved."""
import numpy as np


def map_state(state,times,reference,raw,fits,mode):
    out={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in state.items()};audit=[]
    source_times=reference['market_times'];prices=reference['close'];cache={}
    pair_lookup={tuple(ids):j for j,ids in enumerate(raw['pairs'])}
    for i,k in zip(*np.where(state['directions']!=0)):
        source=int(state['signal_times'][i,k]);ix=int(np.searchsorted(source_times,source));assert source_times[ix]==source<=times[i]
        key=(source,int(k));source_k=pair_lookup.get(tuple(state['pairs'][k]))
        if source_k is None:raise ValueError('Source pair identity unavailable')
        confirmed=bool(raw['confirmed'][ix,source_k]);ratio=1.
        fit=next((f for f in reversed(fits) if f['time']<=source),None)
        if mode=='confirmed':keep=confirmed
        elif mode=='calibrated':
            ratio=None if fit is None else fit['groups'][str(confirmed)]['ratio']
            keep=ratio is not None and ratio>0
        else:raise ValueError(mode)
        if not keep:out['directions'][i,k]=0
        elif mode=='calibrated':
            anchor=prices[ix,raw['pairs'][source_k]]
            out['terminal_prices'][i,k]=anchor+ratio*(state['terminal_prices'][i,k]-anchor)
        if key not in cache:
            cache[key]=True;audit.append(dict(time=source,pair_index=int(k),confirmed=confirmed,kept=keep,ratio=ratio,
                fit_time=None if fit is None else fit['time'],last_maturity=None if fit is None else fit['last_maturity']))
    return out,audit
