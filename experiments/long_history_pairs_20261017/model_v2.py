"""Source pair identity mapping; fitting primitives unchanged from v1."""
import numpy as np
from .model import feature_at,training_rows,fit,predict


def map_state(state,times,reference,raw,models,kind,supported_pairs):
    out={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in state.items()};audit=[];cache={}
    pair_lookup={tuple(ids):j for j,ids in enumerate(raw['pairs'])}
    for i,k in zip(*np.where(state['directions']!=0)):
        source=int(state['signal_times'][i,k]);ix=int(np.searchsorted(reference['market_times'],source));assert reference['market_times'][ix]==source<=times[i]
        ids=state['pairs'][k];source_k=pair_lookup.get(tuple(ids))
        if source_k is None:raise ValueError('Source pair identity unavailable')
        pair=tuple(supported_pairs[k]);key=(source,int(k))
        if key not in cache:
            model=models.get(pair);anchor=reference['close'][ix,ids];terminal=state['terminal_prices'][i,k]
            sign=state['directions'][i,k];beta=state['beta'][i,k];qty=sign*np.array([1.,-beta])/(1+beta)/anchor
            gap=float(qty@(terminal-anchor));assert gap>0
            if kind=='universe':prediction=dict(ratio=float(pair in models),method='full_target_universe_control',raw_price_return=gap)
            else:
                x,unit=feature_at(reference,raw,ix,source_k);prediction=predict(model,x,unit,gap,kind)
            if model is not None:assert model['fit_time']<=source and (model['last_maturity'] is None or model['last_maturity']<model['fit_time'])
            cache[key]=prediction
            audit.append(dict(time=source,pair=list(pair),pair_index=int(k),full_gap=gap,**prediction,
                fit_time=None if model is None else model['fit_time'],last_maturity=None if model is None else model['last_maturity']))
        ratio=cache[key]['ratio']
        if ratio<=0:out['directions'][i,k]=0
        elif kind!='universe':
            anchor=reference['close'][ix,ids];out['terminal_prices'][i,k]=anchor+ratio*(state['terminal_prices'][i,k]-anchor)
    return out,audit
