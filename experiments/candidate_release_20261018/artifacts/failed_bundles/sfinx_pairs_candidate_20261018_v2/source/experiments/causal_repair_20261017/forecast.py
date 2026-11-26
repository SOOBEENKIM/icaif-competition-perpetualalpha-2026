"""Per-asset rolling fits and destination-domain past-only admission evidence."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .data import HERE,OUT,save,sha

HOUR=3600


def timestamp(s):return int(np.datetime64(s,'s').astype('int64'))


def features(a):
    c=a['close'];log=pd.Series(np.log(c));ret=log.diff()
    vol=np.maximum(ret.rolling(168).std(ddof=0).to_numpy(),.0005)
    values=[];names=[]
    def add(name,x):names.append(name);values.append(np.asarray(x))
    for h in (1,6,24,72,168):add(f'return_{h}',log.diff(h)/(vol*np.sqrt(h)))
    for h in (24,168):
        add(f'deviation_{h}',(log-log.rolling(h).mean())/(vol*np.sqrt(h)))
        up=ret.clip(lower=0).rolling(h).mean().to_numpy();down=(-ret.clip(upper=0)).rolling(h).mean().to_numpy()
        add(f'rsi_{h}',np.divide(up-down,up+down,out=np.zeros_like(up),where=(up+down)>1e-12))
    o,h,l=(a[k] for k in ('open','high','low'))
    add('range',np.log(h/l)/vol);add('body',np.log(c/o)/vol)
    add('close_location',np.divide(2*c-h-l,h-l,out=np.zeros_like(c),where=h>l))
    add('vol_ratio',ret.rolling(24).std(ddof=0)/vol)
    add('relative_volume',np.log1p(a['volume'])-np.log1p(pd.Series(a['volume']).rolling(168).mean()))
    return np.clip(np.stack(values,axis=-1),-10,10),vol,names


def labels(a,vol,h=72):
    n=len(vol);y=np.full(n,np.nan);maturity=np.full(n,np.iinfo(np.int64).max,dtype=np.int64)
    end=h+1
    y[:-end]=np.log(a['close'][end:]/a['close'][1:-h])/(vol[:-end]*np.sqrt(h))
    maturity[:-end]=a['times'][end:]
    return y,maturity


def training_mask(a,x,y,maturity,cutoff,months=12,sampling=6):
    month=np.datetime64(cutoff,'s').astype('datetime64[M]')
    start=int((month-np.timedelta64(months,'M')).astype('datetime64[s]').astype('int64'))
    # A full source window is required; missing listing history means unavailable.
    if a['times'][0]>start or a['times'][-1]<cutoff:return None
    return ((a['times']>=start)&(maturity<cutoff)&(a['times']%(sampling*HOUR)==0)
            &np.isfinite(x).all(1)&np.isfinite(y))


def admission_evidence(times,pred,actual,maturity,cutoff,config):
    start=cutoff-config['lookback_days']*86400
    # Non-overlapping maturity-complete 72h blocks within each asset. No pooling.
    ix=((times>=start)&(maturity<cutoff)&(times%(72*HOUR)==0)&np.isfinite(pred)&np.isfinite(actual))
    pp=pred[ix];yy=actual[ix];tt=times[ix]
    evidence={'cutoff':int(cutoff),'start':int(start),'blocks':len(pp),'enabled':False,
              'last_maturity':int(maturity[ix].max()) if len(pp) else None,
              'descriptive_only':True}
    if len(pp)<config['minimum_nonoverlapping_blocks']:
        evidence['reason']='insufficient_mature_destination_blocks';return evidence
    corr=float(np.corrcoef(pp,yy)[0,1]) if np.std(pp)>1e-12 and np.std(yy)>1e-12 else 0.
    mse=float(np.mean((pp-yy)**2));zero=float(np.mean(yy**2))
    # A hypothetical unit trade is admitted only beyond its own round-trip cost.
    cost=config['round_trip_cost_bps']/10000
    direction=np.where(abs(pp)>cost,np.sign(pp),0.)
    net=direction*yy-abs(direction)*cost
    halves=[float(z.mean()) for z in np.array_split(net,2)]
    checks={'positive_correlation':corr>0,'mse_below_zero':mse<zero,
            'positive_net_mean':float(net.mean())>0,'both_halves_positive':min(halves)>0}
    evidence.update(correlation=corr,mse=mse,zero_mse=zero,net_directional_mean=float(net.mean()),
                    chronological_halves_net=halves,active_blocks=int(np.count_nonzero(direction)),
                    checks=checks,enabled=all(checks.values()),reason='admitted' if all(checks.values()) else 'failed_destination_evidence')
    return evidence


def main():
    dest=OUT/'forecast';dest.mkdir(exist_ok=True)
    if (dest/'manifest.json').exists():raise FileExistsError('Forecasts already frozen')
    protocol=json.loads((HERE/'protocol.json').read_text());config=protocol['model']
    root=HERE.parents[1];path=root/protocol['data']['reference']
    with np.load(path) as z:inp={k:z[k].copy() for k in z.files}
    times=inp['market_times'];h=config['horizon_hours']
    pred=np.full((len(times),3),np.nan);mature=np.zeros((len(times),3),dtype=np.int64)
    actual=np.full_like(pred,np.nan);gates=np.zeros_like(pred,dtype=bool)
    fits=[];evidence=[];states={}
    months=np.arange(np.datetime64(config['first_model_month'],'M'),np.datetime64(config['last_model_month'],'M')+1)
    for j,symbol in enumerate(protocol['data']['symbols']):
        source=OUT/'external'/f'{symbol}.npz'
        with np.load(source) as z:a={k:z[k].copy() for k in z.files}
        x,vol,names=features(a);y,mt=labels(a,vol,h)
        target={k:inp[k][:,j] for k in ('open','high','low','close','volume','trades')};target['times']=times
        tx,tv,tnames=features(target);ty,tm=labels(target,tv,h)
        assert names==tnames
        actual[:,j]=ty*tv*np.sqrt(h);mature[:,j]=tm
        for month in months:
            cutoff=int(month.astype('datetime64[s]').astype('int64'))
            stop=int((month+1).astype('datetime64[s]').astype('int64'))
            mask=training_mask(a,x,y,mt,cutoff,config['train_months'],config['fit_sampling_hours'])
            label=f'{symbol}-{month}'
            if mask is None:
                fits.append({'asset':symbol,'cutoff':cutoff,'available':False,'reason':'less_than_12_months_source_history'});continue
            params={k:v for k,v in config['xgboost'].items() if k!='num_boost_round'}
            params.update(objective='reg:squarederror',tree_method='hist',device='cpu',verbosity=0)
            booster=xgb.train(params,xgb.DMatrix(x[mask],label=np.clip(y[mask],-4,4)),num_boost_round=config['xgboost']['num_boost_round'])
            file=dest/f'{label}.ubj';booster.save_model(file)
            valid=(times>=cutoff)&(times<stop)&np.isfinite(tx).all(1)
            pred[valid,j]=np.clip(booster.predict(xgb.DMatrix(tx[valid])),-2,2)*tv[valid]*np.sqrt(h)
            fit={'asset':symbol,'cutoff':cutoff,'available':True,'rows':int(mask.sum()),
                 'first_feature_time':int(a['times'][mask].min()),'last_label_maturity':int(mt[mask].max()),
                 'source_sha256':sha(source),'model_sha256':sha(file),'features':names,
                 'destination_prediction_start':cutoff,'destination_prediction_end_exclusive':stop}
            assert fit['last_label_maturity']<cutoff
            fits.append(fit);states[label]=fit
            print('fit',label,'rows',fit['rows'],flush=True)
    # Forecast net funding in decisions from destination information available at t.
    past_rate=pd.DataFrame(inp['funding'][:,:3]).shift(1).rolling(168).mean().to_numpy()
    mu=pred-past_rate*h
    # For gate scoring use signed price returns and actual paid unit-notional funding.
    # Labels below are forward long net returns; this remains a proxy diagnostic,
    # while the economic replay applies funding to actual changing quantities.
    funding=np.full_like(pred,np.nan)
    for i in range(len(times)-h-1):funding[i]=inp['funding'][i+2:i+h+2,:3].sum(0)
    net_actual=actual-funding
    for month in months:
        cutoff=int(month.astype('datetime64[s]').astype('int64'));stop=int((month+1).astype('datetime64[s]').astype('int64'))
        for j,symbol in enumerate(protocol['data']['symbols']):
            ev=admission_evidence(times,mu[:,j],net_actual[:,j],mature[:,j],cutoff,protocol['gate'])
            ev['asset']=symbol;evidence.append(ev)
            gates[(times>=cutoff)&(times<stop),j]=ev['enabled']
    np.savez_compressed(dest/'predictions.npz',times=times,price_prediction=pred,net_prediction=mu,
                        actual_net=net_actual,maturity=mature,gates=gates)
    save(dest/'manifest.json',{'fits':fits,'gates':evidence,'protocol_sha256':sha(HERE/'protocol.json'),
                              'reference_sha256':sha(path),'source_sha256':sha(__file__),
                              'predictions_sha256':sha(dest/'predictions.npz'),'orders_sent':0})
    for e in evidence:print('gate',e['asset'],str(np.datetime64(e['cutoff'],'s')),e['blocks'],e['enabled'],e['reason'],flush=True)


if __name__=='__main__':main()
