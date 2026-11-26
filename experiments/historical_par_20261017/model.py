"""Rolling hedge estimates and purged walk-forward relative-return ridge.

This is an interpretable baseline, not a reproduction of any neural/RL paper.
Model labels start at the next close and mature 72 hours later. Every fit uses
only labels strictly matured before that fit; scalers are fitted on those rows.
"""
import numpy as np
from .data import ts


def pair_features(close, pairs, symbols, window=168):
    n = len(close); x = np.full((n,len(pairs),9),np.nan)
    beta = np.full((n,len(pairs)),np.nan); corr = beta.copy()
    logs = np.log(close)
    for i in range(window,n):
        for k,(sa,sb) in enumerate(pairs):
            a,b = symbols.index(sa),symbols.index(sb)
            r = np.diff(logs[i-window:i+1,[a,b]],axis=0)
            c = np.cov(r,rowvar=False,ddof=1)
            if c[1,1] < 1e-12 or c[0,0] < 1e-12: continue
            be = float(c[0,1]/c[1,1]); co = float(c[0,1]/np.sqrt(c[0,0]*c[1,1]))
            beta[i,k],corr[i,k] = be,co
            if not .1 <= be <= 10: continue
            spread = (logs[i-window:i+1,a]-be*logs[i-window:i+1,b])/(1+be)
            ret = r[:,0]/(1+be)-r[:,1]*be/(1+be)
            x[i,k] = [*[spread[-1]-spread[-1-h] for h in (1,6,24,72,168)],
                      (spread[-1]-spread.mean())/max(spread.std(),1e-6),ret.std(),co,be]
    return x,beta,corr


def build(d,cfg,symbols):
    times,close=d['ref_times'],d['ref_close']; x,beta,corr=pair_features(close,cfg['pairs'],symbols)
    h=cfg['ridge']['horizon_hours']; n=len(times); y=np.full(beta.shape,np.nan)
    for k,(sa,sb) in enumerate(cfg['pairs']):
        a,b=symbols.index(sa),symbols.index(sb)
        rr=close[h+1:,[a,b]]/close[1:n-h,[a,b]]-1
        be=beta[:n-h-1,k]
        y[:n-h-1,k]=(rr[:,0]-be*rr[:,1])/(1+be)
    ix=np.searchsorted(times,d['times']); out=np.full((len(ix),len(cfg['pairs'])),np.nan)
    maturity=times+(h+1)*3600; begin=ts(cfg['ridge']['training_start'])
    first=ts(cfg['ridge']['initial_cutoff']); step=cfg['ridge']['refit_days']*86400
    models=[]; last=None; fitted={}
    for ii,i in enumerate(ix):
        at=first+(int(times[i])-first)//step*step
        if at!=last:
            fitted={}; last=at
            for k,pair in enumerate(cfg['pairs']):
                use=(times>=begin)&(maturity<at)&(times%21600==0)&np.isfinite(x[:,k]).all(1)&np.isfinite(y[:,k])
                if use.sum()<40: continue
                xx=x[use,k]; yy=y[use,k]; mean=xx.mean(0); scale=np.maximum(xx.std(0),1e-6)
                z=np.c_[np.ones(len(xx)),(xx-mean)/scale]
                penalty=np.eye(z.shape[1])*cfg['ridge']['penalty']; penalty[0,0]=0
                coef=np.linalg.solve(z.T@z/len(z)+penalty,z.T@yy/len(z))
                model=dict(fit_time=int(at),pair=pair,samples=int(use.sum()),last_label_maturity=int(maturity[use].max()),
                           first_feature_time=int(times[use][0]),mean=mean.tolist(),scale=scale.tolist(),coef=coef.tolist())
                assert model['last_label_maturity']<at<=int(times[i])
                fitted[k]=model; models.append(model)
        for k,model in fitted.items():
            if np.isfinite(x[i,k]).all():
                z=np.r_[1.,(x[i,k]-model['mean'])/model['scale']]
                out[ii,k]=np.clip(z@model['coef'],-.1,.1)
    # Par funding forecast: t-1 and trailing mean ending t-1, never the funding
    # announced with subsecond timestamp jitter after the decision boundary.
    forecast=np.full(d['funding'].shape,np.nan)
    for i,t in enumerate(d['times']):
        j=np.searchsorted(d['funding_times'],t)-1
        if j>=23:
            rates=d['funding_history'][j-23:j+1]
            good=np.isfinite(rates).all(0)
            forecast[i,good]=(.5*rates[-1,good]+.5*rates[:,good].mean(0))*h
    return dict(beta=beta[ix],correlation=corr[ix],ridge=out,funding_forecast=forecast),models
