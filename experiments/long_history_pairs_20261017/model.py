"""Per-pair event value trained on a separate, earlier reference venue."""
import numpy as np


def feature_at(reference,raw,i,k):
    if i<24:raise ValueError('Insufficient causal feature history')
    ids=raw['pairs'][k];beta=float(raw['beta'][i,k]);sd=float(raw['sd'][i,k]);direction=int(raw['delayed'][i,k])
    if direction==0 or sd<=0:raise ValueError('Feature requested outside source event')
    lp=np.log(reference['close'][[i-24,i-1,i]][:,ids]);current=lp[-1]
    recovery=[direction*((current[0]-v[0])-beta*(current[1]-v[1]))/sd for v in lp[:2]]
    x=np.array([-direction*raw['z'][i,k],recovery[1],recovery[0],np.log(beta),float(raw['confirmed'][i,k])])
    unit=sd/(1+beta)
    if not np.isfinite(x).all() or not np.isfinite(unit) or unit<=0:raise ValueError('Invalid feature or scale')
    return x,unit


def training_rows(reference,raw,events,cutoff):
    x=[];y=[];gaps=[];price_targets=[];keys=[];maturities=[]
    for event in events:
        if event['maturity'] is None or event['maturity']>=cutoff:continue
        i=int(np.searchsorted(reference['market_times'],event['time']));assert reference['market_times'][i]==event['time']
        xx,unit=feature_at(reference,raw,i,event['pair_index'])
        x.append(xx);y.append(event['realized_price_return']/unit);gaps.append(event['expected_full_convergence'])
        price_targets.append(event['realized_price_return']);keys.append(event['time']);maturities.append(event['maturity'])
    return dict(x=np.asarray(x,float).reshape(-1,5),y=np.asarray(y),gaps=np.asarray(gaps),price_targets=np.asarray(price_targets),
        source_times=np.asarray(keys,dtype=np.int64),maturities=np.asarray(maturities,dtype=np.int64))


def fit(rows,cutoff,spec):
    n=len(rows['y']);last=int(max(rows['maturities'])) if n else None
    if n and last>=cutoff:raise ValueError('Unmatured training label')
    model=dict(fit_time=cutoff,events=n,last_maturity=last,scalar=None,ridge=None)
    if n<spec['minimum_scalar_events']:return model
    gaps=rows['gaps'];slope=float(gaps@rows['price_targets']/(gaps@gaps))
    model['scalar']=dict(unshrunk=slope,ratio=float(np.clip(slope/(1+spec['prior_events']/n),0,1)))
    if n>=spec['minimum_ridge_events']:
        x=rows['x'];center=np.median(x,axis=0);iqr=np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0)
        scale=np.where(iqr>1e-6,iqr/1.349,1.)
        z=np.c_[np.ones(n),np.clip((x-center)/scale,-spec['feature_clip'],spec['feature_clip'])]
        y=np.clip(rows['y'],-spec['target_clip'],spec['target_clip'])
        penalty=np.eye(z.shape[1])*n*spec['ridge_penalty'];penalty[0,0]=spec['prior_events']
        coefficient=np.linalg.solve(z.T@z+penalty,z.T@y)
        model['ridge']=dict(center=center.tolist(),scale=scale.tolist(),coefficient=coefficient.tolist(),feature_clip=spec['feature_clip'])
    return model


def predict(model,x,unit,full_gap,kind):
    if model is None or model['scalar'] is None:return dict(ratio=0.,method='unavailable',raw_price_return=None)
    if kind=='scalar' or model['ridge'] is None:
        ratio=model['scalar']['ratio'];return dict(ratio=ratio,method='scalar' if kind=='scalar' else 'scalar_fallback',raw_price_return=ratio*full_gap)
    r=model['ridge'];z=np.r_[1.,np.clip((x-r['center'])/r['scale'],-r['feature_clip'],r['feature_clip'])]
    price_return=float(z@r['coefficient']*unit)
    ratio=float(np.clip(price_return/full_gap,0,1))
    return dict(ratio=ratio,method='ridge',raw_price_return=price_return)


def map_state(state,times,reference,raw,models,kind,supported_pairs):
    out={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in state.items()};audit=[];cache={}
    for i,k in zip(*np.where(state['directions']!=0)):
        source=int(state['signal_times'][i,k]);ix=int(np.searchsorted(reference['market_times'],source));assert reference['market_times'][ix]==source<=times[i]
        ids=raw['pairs'][k];pair=tuple(supported_pairs[k]);key=(source,int(k))
        if key not in cache:
            model=models.get(pair);anchor=reference['close'][ix,ids];terminal=state['terminal_prices'][i,k]
            sign=state['directions'][i,k];beta=state['beta'][i,k];qty=sign*np.array([1.,-beta])/(1+beta)/anchor
            gap=float(qty@(terminal-anchor));assert gap>0
            if kind=='universe':prediction=dict(ratio=float(pair in models),method='full_target_universe_control',raw_price_return=gap)
            else:
                x,unit=feature_at(reference,raw,ix,k);prediction=predict(model,x,unit,gap,kind)
            if model is not None:assert model['fit_time']<=source and (model['last_maturity'] is None or model['last_maturity']<model['fit_time'])
            cache[key]=prediction
            audit.append(dict(time=source,pair=list(pair),pair_index=int(k),full_gap=gap,**prediction,
                fit_time=None if model is None else model['fit_time'],last_maturity=None if model is None else model['last_maturity']))
        ratio=cache[key]['ratio']
        if ratio<=0:out['directions'][i,k]=0
        elif kind!='universe':
            anchor=reference['close'][ix,ids];out['terminal_prices'][i,k]=anchor+ratio*(state['terminal_prices'][i,k]-anchor)
    return out,audit
