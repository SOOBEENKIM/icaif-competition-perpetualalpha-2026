"""Causal tabular forecasts. Training and prediction do not touch the venue."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb


def features(inputs):
    c=inputs['close'][:,:3]; n=len(c); log=np.log(c)
    ret=pd.DataFrame(log).diff(); vol=ret.rolling(168,min_periods=168).std(ddof=0).to_numpy()
    vol=np.maximum(vol,0.0005)
    values=[]; names=[]
    def add(name,v):names.append(name);values.append(v)
    for h in (1,6,24,72,168):
        add('return_'+str(h),pd.DataFrame(log).diff(h).to_numpy()/(vol*np.sqrt(h)))
    for h in (24,168):
        add('deviation_'+str(h),(log-pd.DataFrame(log).rolling(h).mean().to_numpy())/(vol*np.sqrt(h)))
        up=ret.clip(lower=0).rolling(h).mean().to_numpy()
        down=(-ret.clip(upper=0)).rolling(h).mean().to_numpy()
        add('rsi_'+str(h),np.divide(up-down,up+down,out=np.zeros_like(up),where=(up+down)>1e-12))
    o,h,l=(inputs[k][:,:3] for k in ('open','high','low'))
    add('range',np.log(h/l)/vol)
    add('body',np.log(c/o)/vol)
    add('close_location',np.divide(2*c-h-l,h-l,out=np.zeros_like(c),where=h>l))
    add('vol_ratio',ret.rolling(24).std(ddof=0).to_numpy()/vol)
    add('relative_volume',np.log1p(inputs['volume'][:,:3])-np.log1p(pd.DataFrame(inputs['volume'][:,:3]).rolling(168).mean().to_numpy()))
    for h in (24,168):
        add('past_funding_'+str(h),pd.DataFrame(inputs['funding'][:,:3]).shift(1).rolling(h).mean().to_numpy()/vol)
    r24=pd.DataFrame(log).diff(24).to_numpy()
    add('relative_momentum',(r24-r24.mean(axis=1,keepdims=True))/(vol*np.sqrt(24)))
    for j in range(3):add('asset_'+str(j),np.broadcast_to(np.eye(3)[j],(n,3)))
    x=np.stack(values,axis=-1)
    x=np.clip(x,-10,10)
    return x,vol,names


def labels(inputs,vol,horizon):
    c=inputs['close'][:,:3]; f=inputs['funding'][:,:3]
    n=len(c); y=np.full_like(c,np.nan); maturity=np.full(n,np.iinfo(np.int64).max,dtype=np.int64)
    for i in range(n-horizon-1):
        entry=i+1; end=entry+horizon
        y[i]=(np.log(c[end]/c[entry])-f[entry+1:end+1].sum(axis=0))/(vol[i]*np.sqrt(horizon))
        maturity[i]=inputs['market_times'][end]
    return y,maturity


def fit(inputs,protocol,output):
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    times=inputs['market_times']; x,vol,names=features(inputs)
    end=int(np.datetime64(protocol['fit_end_exclusive'],'s').astype('int64'))
    start=int(np.datetime64(protocol['train_start'],'s').astype('int64'))
    states={}; preds={}
    for h in protocol['horizons']:
        y,maturity=labels(inputs,vol,h)
        time_mask=(times>=start)&(maturity<end)&(times%(protocol['fit_sampling_hours']*3600)==0)
        mask=time_mask[:,None]&np.isfinite(x).all(axis=-1)&np.isfinite(y)
        xx=x[mask]; yy=np.clip(y[mask],-protocol['target_clip_sigma'],protocol['target_clip_sigma'])
        mean=xx.mean(0);scale=xx.std(0);scale=np.where(scale<1e-8,1,scale)
        z=np.column_stack([np.ones(len(xx)),(xx-mean)/scale]);penalty=np.eye(z.shape[1])*protocol['ridge_penalty'];penalty[0,0]=0
        beta=np.linalg.solve(z.T@z/len(z)+penalty,z.T@yy/len(z))
        params={k:v for k,v in protocol['xgboost'].items() if k!='num_boost_round'}
        params.update(objective='reg:squarederror',tree_method='hist',device='cpu',verbosity=0)
        booster=xgb.train(params,xgb.DMatrix(xx,label=yy),num_boost_round=protocol['xgboost']['num_boost_round'])
        booster.save_model(output/f'tree_{h}.ubj')
        state={'features':names,'mean':mean.tolist(),'scale':scale.tolist(),'beta':beta.tolist(),
               'train_rows':int(mask.sum()),'train_hours':int(time_mask.sum()),'last_label_time':int(maturity[time_mask].max()),
               'fit_end_exclusive':end,'horizon':h,'prediction_clip_sigma':protocol['prediction_clip_sigma']}
        (output/f'ridge_{h}.json').write_text(json.dumps(state,indent=2))
        states[str(h)]=state
        for kind in ('ridge','tree'):
            preds[f'{kind}_{h}']=predict(state,booster if kind=='tree' else None,x,vol)
    (output/'fit.json').write_text(json.dumps(states,indent=2))
    return preds,states


def predict(state,booster,x,vol):
    valid=np.isfinite(x).all(axis=-1)&np.isfinite(vol)
    flat=x[valid]
    p=(state['beta'][0]+((flat-state['mean'])/state['scale'])@np.array(state['beta'][1:])) if booster is None else booster.predict(xgb.DMatrix(flat))
    out=np.zeros(vol.shape)
    out[valid]=np.clip(p,-state['prediction_clip_sigma'],state['prediction_clip_sigma'])*vol[valid]*np.sqrt(state['horizon'])
    return out


def load_predict(inputs,model_dir,horizon):
    p=Path(model_dir);state=json.loads((p/f'ridge_{horizon}.json').read_text())
    tree=xgb.Booster(params={'nthread':1});tree.load_model(p/f'tree_{horizon}.ubj')
    x,vol,names=features(inputs)
    if names!=state['features']:raise ValueError('Feature schema mismatch')
    return {k:predict(state,tree if k=='tree' else None,x,vol) for k in ('ridge','tree')}
