"""Price-only reference forecasts with disjoint, matured calibration labels."""
from pathlib import Path
import numpy as np
import xgboost as xgb
from experiments.pairs_forecast_20261017.model import design as reference_design
from experiments.par_direct_forecast_20261017.data import ts,save


def design(m,cfg):
    d=reference_design(m,cfg)
    d['correlation']=d['x'][:,:,d['names'].index('correlation_168')]
    return d


def labels(m,de,h):
    n=len(m.times);y=np.full(de['beta'].shape,np.nan)
    for k,(a,b) in enumerate(de['pairs']):
        rr=m.close[h:,[a,b]]/m.close[:n-h,[a,b]]-1
        y[:n-h,k]=(rr[:,0]-de['beta'][:n-h,k]*rr[:,1])/(1+de['beta'][:n-h,k])
        y[:n-h,k]/=de['vol'][:n-h,k]*np.sqrt(h)
    return y,m.times+h*3600


def fit_predict(m,cfg,out):
    out=Path(out);out.mkdir(exist_ok=True);de=design(m,cfg);x=de['x'];times=m.times
    preds={f'{name}_{h}':np.full(de['beta'].shape,np.nan) for name in cfg['models'] for h in cfg['horizons_hours']}
    uncertainty={k:np.full(de['beta'].shape,np.nan) for k in preds};records=[]
    dates=[ts(s) for s in cfg['fit_dates']]
    for h in cfg['horizons_hours']:
        y,maturity=labels(m,de,h)
        for fi,at in enumerate(dates):
            cut=at-cfg['calibration_days']*86400;until=dates[fi+1] if fi+1<len(dates) else times[-1]+1
            valid=np.isfinite(x).all(-1)&np.isfinite(y)&de['fresh']
            train=valid&((times>=at-cfg['training_window_days']*86400)&(maturity<cut)&(times%(cfg['fit_sampling_hours']*3600)==0))[:,None]
            calib=valid&((times>=cut)&(maturity<at)&((times-cut)%(h*3600)==0))[:,None]
            ii,jj=np.where(train);ci,cj=np.where(calib)
            if len(ii)<cfg['minimum_training_rows'] or len(ci)<30:raise ValueError('Insufficient past-only training/calibration samples')
            xx=x[ii,jj];yy=np.clip(y[ii,jj],-4,4);mean=xx.mean(0);scale=np.maximum(xx.std(0),1e-6)
            z=np.c_[np.ones(len(xx)),(xx-mean)/scale];penalty=np.eye(z.shape[1])*cfg['ridge_penalty'];penalty[0,0]=0
            coef=np.linalg.solve(z.T@z/len(z)+penalty,z.T@yy/len(z))
            params={k:v for k,v in cfg['tree'].items() if k!='num_boost_round'}
            params.update(objective='reg:squarederror',tree_method='hist',verbosity=0)
            tree=xgb.train(params,xgb.DMatrix(xx,label=yy),num_boost_round=cfg['tree']['num_boost_round'])
            def predict(a):
                ridge=np.clip(np.c_[np.ones(len(a)),(a-mean)/scale]@coef,-4,4)
                nonlinear=np.clip(tree.predict(xgb.DMatrix(a)),-4,4)
                return ridge,nonlinear
            rc,tc=predict(x[ci,cj]);blend=(rc+tc)/2;cy=y[ci,cj]
            slope=float(np.clip(np.mean((blend-blend.mean())*(cy-cy.mean()))/(np.var(blend)+cfg['calibration_slope_penalty']),0,1))
            intercept=float(np.mean(cy-slope*blend)*len(cy)/(len(cy)+cfg['calibration_intercept_prior_rows']))
            vi,vj=np.where(np.isfinite(x).all(-1)&((times>=at)&(times<until))[:,None]);rv,tv=predict(x[vi,vj])
            pp={'ridge':rv,'tree':tv,'calibrated':np.clip(intercept+slope*(rv+tv)/2,-4,4)}
            cc={'ridge':rc,'tree':tc,'calibrated':np.clip(intercept+slope*blend,-4,4)}
            rmses={}
            for name in cfg['models']:
                key=f'{name}_{h}';preds[key][vi,vj]=pp[name]*de['vol'][vi,vj]*np.sqrt(h)
                per=[]
                for k in range(len(cfg['pairs'])):
                    err=cy[cj==k]-cc[name][cj==k]
                    rmse=float(np.sqrt(np.mean(err*err))) if len(err) else float(np.sqrt(np.mean((cy-cc[name])**2)))
                    per.append(rmse)
                    mask=vj==k;uncertainty[key][vi[mask],vj[mask]]=rmse*de['vol'][vi[mask],vj[mask]]*np.sqrt(h)
                rmses[name]=per
            record=dict(fit_time=at,horizon=h,training_rows=len(ii),calibration_rows=len(ci),
                first_feature_time=int(times[ii].min()),last_training_maturity=int(maturity[ii].max()),
                calibration_start=cut,last_calibration_maturity=int(maturity[ci].max()),
                calibration_observation_times=len(set(times[ci].tolist())),slope=slope,intercept=intercept,
                rmse_normalized_by_pair=rmses,features=de['names'],mean=mean.tolist(),scale=scale.tolist(),coef=coef.tolist())
            assert record['last_training_maturity']<cut and record['last_calibration_maturity']<at
            records.append(record);save(out/f'fit_{h}_{at}.json',record);tree.save_model(out/f'tree_{h}_{at}.ubj')
            print('fit',h,at,'train',len(ii),'calibrate',len(ci),'slope',round(slope,3),flush=True)
    preds['zero_24']=np.where(np.isfinite(preds['ridge_24']),0.,np.nan)
    uncertainty['zero_24']=uncertainty['ridge_24'].copy()
    save(out/'fits.json',records)
    return de,preds,uncertainty,records


def convert_to_par(d,m,de,forecasts,uncertainty):
    """An exact change of source-price units, without using future prices."""
    ix=np.searchsorted(m.times,d['times'],side='right')-1
    assert np.all(ix>=0) and np.all(m.times[ix]<=d['times'])
    beta=np.zeros((len(ix),len(de['pairs'])));out={k:np.full(beta.shape,np.nan) for k in forecasts}
    errors={k:np.full(beta.shape,np.nan) for k in forecasts}
    for k,(a,b) in enumerate(de['pairs']):
        pa,pb=d['par_close'][:,a],d['par_close'][:,b];ra,rb=m.close[ix,a],m.close[ix,b]
        ratio=de['beta'][ix,k]*ra/rb
        beta[:,k]=ratio*pb/pa
        ref_gross=ra+ratio*rb;par_gross=pa+ratio*pb
        correction=(ra-pa)-ratio*(rb-pb)
        for name,mu in forecasts.items():
            out[name][:,k]=(ref_gross*mu[ix,k]+correction)/par_gross
            errors[name][:,k]=ref_gross*uncertainty[name][ix,k]/par_gross
    mapped=dict(pairs=de['pairs'],beta=beta,correlation=de['correlation'][ix],fresh=de['fresh'][ix],forecast_times=m.times[ix])
    return mapped,out,errors
