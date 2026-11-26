"""Calibrate existing model outputs, only after their own training cutoff."""
import json
from pathlib import Path
import numpy as np
from experiments.pairs_forecast_20261017.model import design,labels


def features(m, arrays, cfg):
    d=design(m,cfg)
    n,count=d['beta'].shape;features=[]
    daily_vol=np.sqrt(np.maximum(np.diagonal(arrays['sigma'],axis1=1,axis2=2),1e-12))
    for k,(a,b) in enumerate(d['pairs']):
        beta=d['beta'][:,k];vol=d['vol'][:,k]
        f=(np.expm1(arrays['forecast'][:,a])-beta*np.expm1(arrays['forecast'][:,b]))/(1+beta)
        wa,wb=arrays['candidate'][:,a],arrays['candidate'][:,b]
        opposed=(wa*wb<0)*np.sign(wa)
        columns=[f/(vol*np.sqrt(24)),wa*daily_vol[:,a]/vol,-wb*daily_vol[:,b]/vol,opposed,
                 arrays['retention'][:,a]-arrays['retention'][:,b],
                 d['x'][:,k,d['names'].index('correlation_168')]]
        columns.extend(np.full(n,float(j==k)) for j in range(count))
        features.append(np.stack(columns,axis=-1))
    d['calibration_features']=np.clip(np.stack(features,axis=1),-8,8)
    return d


def fit(m,arrays,cfg,out):
    out=Path(out);out.mkdir(exist_ok=True)
    d=features(m,arrays,cfg);x=d['calibration_features']
    y,maturity=labels(m,d,24)
    preds=np.full(y.shape,np.nan);stamps=np.full(y.shape,-1,dtype=np.int64)
    dates=[int(np.datetime64(t,'s').astype('int64')) for t in cfg['fit_dates']]
    cutoff=int(np.datetime64(cfg['signal_training_cutoff'],'s').astype('int64'))
    records=[]
    for ix,at in enumerate(dates):
        until=dates[ix+1] if ix+1<len(dates) else int(m.times[-1])+1
        eligible=(m.times>=max(cutoff,at-cfg['fit_trailing_hours']*3600))&(maturity<at)&(m.times%86400==0)
        rows,pairs=np.where(eligible[:,None]&np.isfinite(y)&np.isfinite(x).all(-1)&d['fresh'])
        if len(rows)<cfg['minimum_training_labels']:
            records.append(dict(fit_time=at,samples=len(rows),status='insufficient_mature_labels'));continue
        assert m.times[rows].min()>=cutoff and maturity[rows].max()<at
        mean=x[rows,pairs].mean(0);scale=np.maximum(x[rows,pairs].std(0),1e-6)
        z=np.c_[np.ones(len(rows)),(x[rows,pairs]-mean)/scale];target=np.clip(y[rows,pairs],-4,4)
        penalty=np.eye(z.shape[1])*cfg['ridge_penalty'];penalty[0,0]=0.
        coef=np.linalg.solve(z.T@z/len(z)+penalty,z.T@target/len(z))
        vi,vj=np.where(((m.times>=at)&(m.times<until))[:,None]&np.isfinite(x).all(-1))
        values=np.c_[np.ones(len(vi)),(x[vi,vj]-mean)/scale]@coef
        preds[vi,vj]=np.clip(values,-2,2)*d['vol'][vi,vj]*np.sqrt(24)
        stamps[vi,vj]=at
        record=dict(fit_time=at,status='fit',samples=len(rows),time_blocks=len(set(rows.tolist())),
            feature_first_time=int(m.times[rows].min()),last_label_time=int(maturity[rows].max()),
            prediction_first=int(m.times[vi].min()),prediction_last=int(m.times[vi].max()),
            mean=mean.tolist(),scale=scale.tolist(),coef=coef.tolist())
        records.append(record);(out/f'{at}.json').write_text(json.dumps(record,indent=2))
    np.savez_compressed(out/'predictions.npz',times=m.times,predictions=preds,fit_stamps=stamps,beta=d['beta'],vol=d['vol'],
        correlation=d['x'][:,:,d['names'].index('correlation_168')])
    (out/'manifest.json').write_text(json.dumps(records,indent=2))
    return d,preds,stamps,records
