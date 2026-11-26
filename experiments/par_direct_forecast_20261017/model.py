"""Actual-venue pair labels with strict label maturity and past-only features."""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from .data import ts,save


def design(d,cfg,symbols):
    n=len(d['times']);ids=[[symbols.index(s) for s in p] for p in cfg['pairs']]
    pc=np.log(d['par_close']);rc=np.log(d['reference_close']);basis=pc-rc
    rr=pd.DataFrame(rc).diff();hour=rr.rolling(12).sum()
    fields=[];betas=[];cors=[];fresh=[];names=[]
    # Estimate return hedge from 24h of 5m data, lower bound of 12h activity.
    # Unlike a price-level regression, this does not require prior cointegration.
    for k,(a,b) in enumerate(ids):
        ra,rb=rr[a],rr[b];va=ra.rolling(288).var(ddof=0).to_numpy();vb=rb.rolling(288).var(ddof=0).to_numpy()
        cov=ra.rolling(288).cov(rb,ddof=0).to_numpy();beta=cov/np.maximum(vb,1e-12)
        co=cov/np.sqrt(np.maximum(va*vb,1e-24));be=np.clip(beta,.1,10.)
        vol=np.sqrt(np.maximum(va+be**2*vb-2*be*cov,1e-10))/(1+be)
        values=[];columns=[]
        def add(name,value):columns.append(name);values.append(value)
        add('relative_basis',(basis[:,a]-be*basis[:,b])/(1+be)*10000)
        add('basis_a',basis[:,a]*10000);add('basis_b',basis[:,b]*10000)
        for h in (1,12,72,288):
            for label,logs in [('par',pc),('reference',rc)]:
                delta=pd.DataFrame(logs).diff(h).to_numpy()
                add(f'{label}_relative_return_{h}',(delta[:,a]-be*delta[:,b])/(1+be)*10000)
        for h in (12,72,288):
            means=pd.DataFrame(basis).rolling(h).mean().to_numpy()
            add(f'basis_deviation_{h}',((basis[:,a]-means[:,a])-be*(basis[:,b]-means[:,b]))/(1+be)*10000)
        add('hedge_beta',be);add('correlation',co);add('hourly_sigma_bps',vol*np.sqrt(12)*10000)
        add('past_funding_difference',(d['known_hourly_funding'][:,a]-be*d['known_hourly_funding'][:,b])/(1+be)*10000)
        add('common_reference_hour',((hour[a]+hour[b])/2).to_numpy()*10000)
        add('volume_a',np.log1p(d['par_volume'][:,a]*d['par_close'][:,a]));add('volume_b',np.log1p(d['par_volume'][:,b]*d['par_close'][:,b]))
        for j in range(len(ids)):add(f'pair_{j}',np.full(n,float(k==j)))
        if names:assert names==columns
        names=columns;fields.append(np.stack(values,axis=-1));betas.append(beta);cors.append(co)
        fresh.append((d['par_volume'][:,a]>0)&(d['par_volume'][:,b]>0)&(d['par_trades'][:,a]>0)&(d['par_trades'][:,b]>0))
    return dict(x=np.stack(fields,axis=1),beta=np.stack(betas,axis=1),correlation=np.stack(cors,axis=1),
                fresh=np.stack(fresh,axis=1),pairs=ids,names=names)


def labels(d,design,hours):
    h=hours*12;n=len(d['times']);y=np.full(design['beta'].shape,np.nan)
    for k,(a,b) in enumerate(design['pairs']):
        r=d['par_close'][h+1:,[a,b]]/d['par_close'][1:n-h,[a,b]]-1
        be=design['beta'][:n-h-1,k]
        y[:n-h-1,k]=(r[:,0]-be*r[:,1])/(1+be)*10000
    return y,d['times']+(h+1)*300


def fit_predict(d,cfg,symbols,out):
    out.mkdir(exist_ok=True);de=design(d,cfg,symbols);times=d['times'];x=de['x'];records=[]
    predictions={f'{model}_{h}':np.full(de['beta'].shape,np.nan) for model in cfg['models'] for h in cfg['horizons_hours']}
    dates=[ts(s) for s in cfg['fit_dates']]
    for h in cfg['horizons_hours']:
        target,maturity=labels(d,de,h)
        for fi,at in enumerate(dates):
            until=dates[fi+1] if fi+1<len(dates) else int(times[-1])+1
            usable=np.isfinite(x).all(-1)&de['fresh']&np.isfinite(target)&(de['beta']>=.1)&(de['beta']<=10)
            mask=usable&((times>=at-cfg['training_window_days']*86400)&(maturity<at)&(times%cfg['fit_sampling_seconds']==0))[:,None]
            ii,jj=np.where(mask)
            if len(ii)<cfg['minimum_training_rows']:
                records.append(dict(fit_time=at,horizon=h,rows=len(ii),status='insufficient_mature_training_rows'));continue
            xx=x[ii,jj];yy=target[ii,jj];bounds=np.quantile(yy,[.01,.99]);yy=np.clip(yy,*bounds)
            mean=xx.mean(0);scale=np.maximum(xx.std(0),1e-6);z=np.c_[np.ones(len(xx)),(xx-mean)/scale]
            penalty=np.eye(z.shape[1])*cfg['ridge_penalty'];penalty[0,0]=0
            coef=np.linalg.solve(z.T@z/len(z)+penalty,z.T@yy/len(z))
            params={k:v for k,v in cfg['tree'].items() if k!='num_boost_round'}
            params.update(objective='reg:squarederror',tree_method='hist',verbosity=0)
            tree=xgb.train(params,xgb.DMatrix(xx,label=yy),num_boost_round=cfg['tree']['num_boost_round'])
            vi,vj=np.where(np.isfinite(x).all(-1)&((times>=at)&(times<until))[:,None])
            xxv=x[vi,vj];rp=np.clip(np.c_[np.ones(len(xxv)),(xxv-mean)/scale]@coef,*bounds)
            tp=np.clip(tree.predict(xgb.DMatrix(xxv)),*bounds)
            predictions[f'ridge_{h}'][vi,vj]=rp/10000;predictions[f'tree_{h}'][vi,vj]=tp/10000
            predictions[f'blend_{h}'][vi,vj]=(rp+tp)/20000
            state=dict(fit_time=at,horizon=h,rows=len(ii),distinct_feature_times=len(set(times[ii].tolist())),
                first_feature_time=int(times[ii].min()),last_label_maturity=int(maturity[ii].max()),
                mean=mean.tolist(),scale=scale.tolist(),coef=coef.tolist(),bounds=bounds.tolist(),features=de['names'])
            assert state['last_label_maturity']<at
            records.append(state);save(out/f'ridge_{h}_{at}.json',state);tree.save_model(out/f'tree_{h}_{at}.ubj')
            print('fit',h,at,'rows',len(ii),'predict',len(vi),flush=True)
    save(out/'fits.json',records)
    return de,predictions,records
