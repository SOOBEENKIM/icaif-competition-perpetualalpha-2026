"""Past-only funding forecasts. Values are funding basis points, never PnL."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HOUR=3600000


def panel(rows_by_symbol,symbols):
    values={};availability={}
    for j,symbol in enumerate(symbols):
        for row in rows_by_symbol[symbol]:
            raw=int(row['time']);hour=raw//HOUR*HOUR
            if raw-hour>=1000:raise ValueError('Settlement jitter exceeds declared decision delay')
            key=(hour,j);value=float(row['fundingRate'])*10000
            if key in values and values[key]!=value:raise ValueError('Conflicting funding records')
            values[key]=value;availability[key]=max(raw,availability.get(key,0))
    first,last=min(t for t,j in values),max(t for t,j in values)
    times=np.arange(first,last+HOUR,HOUR,dtype=np.int64)
    rates=np.full((len(times),len(symbols)),np.nan);seen=np.full(rates.shape,-1,dtype=np.int64)
    for (t,j),v in values.items():
        i=(t-first)//HOUR;rates[i,j]=v;seen[i,j]=availability[t,j]
    return dict(times=times,rates=rates,seen=seen,symbols=symbols)


def load(root,cfg):
    base=Path(root)/cfg['source'];rows={}
    for symbol in cfg['symbols']:
        payload=json.loads((base/f'funding_{symbol}.json').read_text())
        if not payload['pagination_complete']:raise ValueError('Incomplete funding pagination')
        rows[symbol]=payload['rows']
    return panel(rows,cfg['symbols'])


def features(p):
    rates=p['rates'];n,a=rates.shape;frame=pd.DataFrame(rates)
    m6=frame.rolling(6,min_periods=6).mean().to_numpy()
    m24=frame.rolling(24,min_periods=24).mean().to_numpy()
    m72=frame.rolling(72,min_periods=72).mean().to_numpy()
    sd=frame.rolling(24,min_periods=24).std(ddof=0).to_numpy()
    x=np.stack([rates,m6,m24,m72,sd,rates-m24],axis=-1)
    identity=np.broadcast_to(np.eye(a)[None],(n,a,a))
    return np.concatenate([x,identity],axis=-1)


def labels(p,horizon):
    n,a=p['rates'].shape;y=np.full((n,a),np.nan);maturity=np.full((n,a),-1,dtype=np.int64)
    for i in range(n-horizon):
        window=p['rates'][i+1:i+horizon+1];valid=np.isfinite(window).all(0)
        y[i,valid]=window[:,valid].sum(0)
        maturity[i,valid]=p['seen'][i+horizon,valid]
    return y,maturity


def fit_ridge(p,x,horizon,at,cfg):
    y,maturity=labels(p,horizon)
    mask=((p['times']>=at-cfg['training_window_hours']*HOUR)&(p['times']<at))[:,None]
    mask=mask&(maturity>0)&(maturity<at)&np.isfinite(x).all(-1)&np.isfinite(y)
    i,j=np.where(mask)
    if len(i)<cfg['minimum_mature_training_rows']:return None
    xx=x[i,j];target=y[i,j]/horizon
    mean=xx.mean(0);scale=np.maximum(xx.std(0),1e-6)
    z=np.c_[np.ones(len(xx)),(xx-mean)/scale]
    penalty=np.eye(z.shape[1])*cfg['ridge_penalty'];penalty[0,0]=0.
    coef=np.linalg.solve(z.T@z/len(z)+penalty,z.T@target/len(z))
    low,high=np.quantile(target,[.01,.99])
    return dict(fit_time=int(at),horizon=horizon,samples=len(i),distinct_hour_blocks=len(set(i.tolist())),
        last_mature_label=int(maturity[i,j].max()),first_feature_time=int(p['times'][i].min()),
        mean=mean.tolist(),scale=scale.tolist(),coef=coef.tolist(),bounds=[float(low),float(high)],
        overlapping_training_labels=True)


def ridge_predict(model,row,horizon):
    if model is None or not np.isfinite(row).all():return np.nan
    z=np.r_[1.,(row-np.array(model['mean']))/np.array(model['scale'])]
    return float(np.clip(z@model['coef'],*model['bounds'])*horizon)


def ar1_forecast(p,i,j,horizon,at,cfg):
    eligible=(p['times']>=at-cfg['training_window_hours']*HOUR)&(p['seen'][:,j]>0)&(p['seen'][:,j]<at)
    ids=np.flatnonzero(eligible[:-1]&eligible[1:]&np.isfinite(p['rates'][:-1,j])&np.isfinite(p['rates'][1:,j]))
    if len(ids)<24:return np.nan
    x,y=p['rates'][ids,j],p['rates'][ids+1,j];variance=float(np.var(x))
    phi=float(np.clip(np.mean((x-x.mean())*(y-y.mean()))/variance,*cfg['ar1_phi_bounds'])) if variance>1e-20 else 0.
    alpha=float(y.mean()-phi*x.mean());value=float(p['rates'][i,j]);total=0.
    for _ in range(horizon):value=alpha+phi*value;total+=value
    return total


def forecasts(p,cfg,fit_start=None):
    x=features(p);out={};models=[]
    for horizon in cfg['horizons_hours']:
        arrays={name:np.full(p['rates'].shape,np.nan) for name in cfg['models']};last_fit=None;model=None
        for i,time in enumerate(p['times']):
            decision=int(time)+cfg['decision_after_hour_ms']
            if fit_start is not None and decision<fit_start:continue
            if time%(cfg['decision_cadence_hours']*HOUR):continue
            fit_at=int(time//(24*HOUR)*(24*HOUR))+cfg['decision_after_hour_ms']
            if fit_at!=last_fit:
                model=fit_ridge(p,x,horizon,fit_at,cfg);last_fit=fit_at
                models.append(model or dict(fit_time=fit_at,horizon=horizon,status='insufficient_mature_labels'))
            for j in range(len(p['symbols'])):
                if not np.isfinite(x[i,j]).all() or p['seen'][i,j]>=decision:continue
                arrays['zero'][i,j]=0.
                arrays['persistence'][i,j]=p['rates'][i,j]*horizon
                arrays['mean24'][i,j]=x[i,j,2]*horizon
                arrays['ar1'][i,j]=ar1_forecast(p,i,j,horizon,fit_at,cfg)
                arrays['ridge'][i,j]=ridge_predict(model,x[i,j],horizon)
        out[horizon]=arrays
    return out,models


def paired_differential(a,b,beta):return (a-beta*b)/(1+beta)


def carry_diagnostic(predicted,actual,fee_floor,buffer):
    direction=-float(np.sign(predicted));active=abs(predicted)>fee_floor+buffer
    realized=-direction*actual
    return dict(direction=direction,active=bool(active),predicted_carry_bps=abs(predicted),
                realized_carry_bps=realized,fee_only_net_bps=realized-fee_floor if active else 0.,
                excludes_price_spread_slippage_and_oracle_changes=True)
