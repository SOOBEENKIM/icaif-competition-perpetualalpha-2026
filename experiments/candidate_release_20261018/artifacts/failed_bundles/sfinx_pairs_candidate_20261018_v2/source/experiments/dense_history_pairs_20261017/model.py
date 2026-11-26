"""Dense, price-only, per-pair forecasts on separate older venue histories."""
import numpy as np
import xgboost as xgb
from replay_lab.common_comparison.models import Market
from experiments.pairs_forecast_20261017.model import design as old_design


def design(times,close,volume,trades,symbols,pairs):
    if not np.all(np.diff(times)==3600):raise ValueError('Dense model requires contiguous hourly sources')
    zeros=np.zeros_like(close);m=Market(times,close,volume,trades,zeros,np.zeros_like(close,dtype=bool),tuple(symbols))
    d=old_design(m,{'pairs':pairs})
    keep=[i for i,n in enumerate(d['names']) if not n.startswith(('funding_','pair_'))]
    return dict(x=d['x'][:,:,keep],names=[d['names'][i] for i in keep],beta=d['beta'],vol=d['vol'],fresh=d['fresh'],pairs=d['pairs'])


def labels(times,close,d,h):
    if not np.all(np.diff(times)==3600):raise ValueError('Labels require contiguous hourly sources')
    n=len(times);y=np.full(d['beta'].shape,np.nan)
    for k,(a,b) in enumerate(d['pairs']):
        ret=close[h:,[a,b]]/close[:-h,[a,b]]-1
        mu=(ret[:,0]-d['beta'][:-h,k]*ret[:,1])/(1+d['beta'][:-h,k])
        y[:-h,k]=mu/(d['vol'][:-h,k]*np.sqrt(h))
    return y,times+h*3600


def linear_predict(record,x):
    return np.clip(np.c_[np.ones(len(x)),(x-record['center'])/record['scale']]@record['coefficient'],-4,4)


def fit(times,close,d,h,spec,train_end,calibration_end):
    if d['x'].shape[1]!=1:raise ValueError('Fit pairs separately')
    y,maturity=labels(times,close,d,h);x=d['x'][:,0];target=y[:,0]
    valid=np.isfinite(x).all(1)&np.isfinite(target)&d['fresh'][:,0]
    train=valid&(maturity<train_end)&(times%(spec['fit_sampling_hours']*3600)==0)
    calibrate=valid&(times>=train_end)&(maturity<calibration_end)&((times-train_end)%(h*3600)==0)
    ii=np.flatnonzero(train);ci=np.flatnonzero(calibrate)
    if len(ii)<spec['minimum_training_rows'] or len(ci)<spec['minimum_calibration_rows']:raise ValueError('Insufficient past training/calibration rows')
    xx=x[ii];yy=np.clip(target[ii],-4,4);center=xx.mean(0);scale=np.maximum(xx.std(0),1e-6)
    z=np.c_[np.ones(len(ii)),(xx-center)/scale];penalty=np.eye(z.shape[1])*spec['ridge_penalty'];penalty[0,0]=0
    coefficient=np.linalg.solve(z.T@z/len(ii)+penalty,z.T@yy/len(ii))
    record=dict(center=center.tolist(),scale=scale.tolist(),coefficient=coefficient.tolist(),names=d['names'],horizon=h,
        training_rows=len(ii),calibration_rows=len(ci),first_training_source=int(times[ii[0]]),last_training_maturity=int(maturity[ii[-1]]),
        first_calibration_source=int(times[ci[0]]),last_calibration_maturity=int(maturity[ci[-1]]),fit_available=calibration_end)
    params={k:v for k,v in spec['tree'].items() if k!='num_boost_round'}
    params.update(objective='reg:squarederror',tree_method='hist',verbosity=0)
    tree=xgb.train(params,xgb.DMatrix(xx,label=yy),num_boost_round=spec['tree']['num_boost_round'])
    ridge=linear_predict(record,x[ci]);tp=np.clip(tree.predict(xgb.DMatrix(x[ci])),-4,4);blend=(ridge+tp)/2;cy=target[ci]
    slope=float(np.clip(np.mean((blend-blend.mean())*(cy-cy.mean()))/(np.var(blend)+spec['calibration_slope_penalty']),0,1))
    intercept=float(np.mean(cy-slope*blend)*len(ci)/(len(ci)+spec['calibration_intercept_prior_rows']))
    record.update(calibration_slope=slope,calibration_intercept=intercept,
        calibration_normalized_rmse=dict(zero=float(np.sqrt(np.mean(cy**2))),ridge=float(np.sqrt(np.mean((ridge-cy)**2))),
            blended_raw=float(np.sqrt(np.mean((blend-cy)**2))),calibrated=float(np.sqrt(np.mean((intercept+slope*blend-cy)**2)))))
    assert record['last_training_maturity']<train_end<=record['first_calibration_source']
    assert record['last_calibration_maturity']<calibration_end
    return record,tree,dict(source_times=times[ci],maturities=maturity[ci],actual=cy,ridge=ridge,blend=blend,
        calibrated=intercept+slope*blend)


def predict(record,tree,d):
    x=d['x'][:,0];valid=np.isfinite(x).all(1);ridge=np.full(len(x),np.nan);calibrated=ridge.copy()
    if d['names']!=record['names']:raise ValueError('Feature identities differ between venues')
    rv=linear_predict(record,x[valid]);tv=np.clip(tree.predict(xgb.DMatrix(x[valid])),-4,4)
    ridge[valid]=rv*d['vol'][valid,0]*np.sqrt(record['horizon'])
    calibrated[valid]=np.clip(record['calibration_intercept']+record['calibration_slope']*(rv+tv)/2,-4,4)*d['vol'][valid,0]*np.sqrt(record['horizon'])
    return dict(ridge=ridge,calibrated=calibrated,zero=np.where(valid,0.,np.nan))


def map_state(times,reference,d,forecast,pairs,source_hours=6,lifetime_hours=1):
    rt=reference['market_times'];source=times//(source_hours*3600)*(source_hours*3600);ix=np.searchsorted(rt,source)
    bounded=np.clip(ix,0,len(rt)-1);available=(ix<len(rt))&(rt[bounded]==source)
    beta=d['beta'][bounded].copy();vol=d['vol'][bounded].copy();mu=forecast[bounded].copy()
    prices=reference['close'][bounded].copy()
    beta[~available]=np.nan;vol[~available]=np.nan;mu[~available]=np.nan;prices[~available]=np.nan
    valid=np.isfinite(mu)&np.isfinite(beta)&d['fresh'][bounded]&available[:,None]
    valid&=(times-source)[:,None]<lifetime_hours*3600
    return dict(dense=True,pairs=np.array(pairs),signal_times=np.broadcast_to(source[:,None],mu.shape).copy(),directions=valid.astype(int),
        source_prices=prices,beta=beta,vol=vol,mu=mu,forecast_times=source)
