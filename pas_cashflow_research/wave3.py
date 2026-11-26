import sys
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from .data import ROOT,OUT,START,DEV_END,SYMBOLS,read,save,sha
from .study import validate
from .policy_v2 import VenueController
from .runner import evaluate,persist
from pas_strategy_audit.data import load
from pas_performance_research.funding import forecast

WAVE=OUT/'wave3'
ARMS={'pas_reversion6':6,'pas_reversion12':12,'pas_adaptive_reversion6':6,'funding_persistent72':72,'funding_persistent168':168}

def predictions(directory):
    cfg,d=load(directory);f=dict(np.load(directory/'candidate_inputs.npz'));t=np.arange(START-21600,cfg['cutoff_ms']//1000+1,3600)
    mi=np.searchsorted(d['times'],t*1000,side='right')-1;good=mi>=0
    px=np.full((len(t),10),np.nan);fresh=np.zeros_like(px,bool);px[good]=d['close'][mi[good]];fresh[good]=d['age_ms'][mi[good]]<=3600000
    lp=pd.DataFrame(np.log(px));vol=lp.diff().ewm(span=24,min_periods=6,adjust=False).std().clip(lower=.0005).to_numpy()
    ix=np.searchsorted(t,f['times']);arrays={'times':f['times'],'fresh':fresh[ix]};audits=[]
    for h in [6,12]:
        past=np.full_like(px,np.nan);past[h:]=px[h:]/px[:-h]-1
        valid=fresh.copy();valid[h:]&=fresh[:-h];valid[:h]=False
        pred=np.clip(-.25*past,-4*vol*np.sqrt(h),4*vol*np.sqrt(h));pred[~valid]=np.nan
        arrays[f'pas_reversion{h}']=pred[ix]
        if h==6:
            unit=vol*np.sqrt(h);x=past/unit;y=np.full_like(px,np.nan);y[:-h]=(px[h:]/px[:-h]-1)/unit[:-h]
            terminal=np.zeros_like(fresh);terminal[:-h]=fresh[h:]
            valid&=terminal&np.isfinite(x)&np.isfinite(y)
            fit=None;out=np.full_like(px,np.nan);maturity=t+h*3600
            for i,now in enumerate(t):
                if now%21600==0:
                    mask=valid&(maturity<now)[:,None]&(t>=now-72*3600)[:,None];a,b=np.where(mask)
                    if len(a)>=20 and len(set(a))>=3:
                        w=2.**(-(now-maturity[a])/86400)/np.bincount(a,minlength=len(t))[a];w/=w.sum()
                        xx=np.clip(x[a,b],-4,4);yy=np.clip(y[a,b],-4,4)
                        fit=float(np.clip(np.sum(w*xx*yy)/(1+np.sum(w*xx*xx)),-1,1))
                        audits.append(dict(source=int(now),last_maturity=int(maturity[a].max()),rows=len(a),coefficient=fit))
                if fit is not None:
                    out[i]=np.clip(fit*past[i],-4*vol[i]*np.sqrt(h),4*vol[i]*np.sqrt(h));out[i,~fresh[i]]=np.nan
            arrays['pas_adaptive_reversion6']=out[ix]
    funding=read(directory/'funding.json');obs=[dict(known=r['time']/1000+60,symbol=r['coin'][4:],rate=r['rate']) for r in funding]
    for h in set(ARMS.values()):
        carry,_=forecast(t,obs,SYMBOLS,horizon=h,decay=h<72);arrays[f'carry{h}']=carry[ix]
    for arm in ['funding_persistent72','funding_persistent168']:arrays[arm]=np.zeros_like(arrays['carry72'])
    return arrays,audits

def prepare():
    validate()
    if (WAVE/'protocol.json').exists():raise FileExistsError('Preserve wave')
    save(WAVE/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,
        hypotheses=['Prior selection48h diagnostics show negative short-lag serial dependence; test quarter reversal vs past-only adaptive shared slope, never hardcode market winners.',
                    'Continuous same-signed PAS funding may not justify a48h exponential decay. Remove decay, preserve50%receipt discount, compare72hand168h cashflow horizons.'],
        model='6h/12h quarter reversal and single-feature pooled regression6h. Adaptive updates6h,72hlookback24hweightdecay,equal sourceweights,ridgeprior1,min20rows3sources,strictmaturity. Parameters fixed before these runs.',
        evaluation='All exposed PAS development partitions/scenarios. Uninspected4h tail remains sealed. No claim these development runs are independent.'))
    arrays,audits=predictions(OUT/'development');np.savez_compressed(WAVE/'forecasts.npz',**arrays);save(WAVE/'model_audit.json',audits)
    save(WAVE/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'pas_cashflow_research/wave3.py',ROOT/'pas_cashflow_research/runner.py',ROOT/'pas_cashflow_research/policy_v2.py',WAVE/'protocol.json',WAVE/'forecasts.npz']})

def verify():
    validate()
    for p,h in read(WAVE/'freeze.json').items():assert sha(ROOT/p)==h,p

def job(arm,scenario,phase,start,end):
    verify();z=dict(np.load(WAVE/'forecasts.npz'));h=ARMS[arm];sc=read(OUT/'protocol.json')['fees'][scenario]
    pred=z[arm];valid=z['fresh']&np.isfinite(pred)
    c=VenueController(z['times'],np.nan_to_num(pred,nan=0.),z[f'carry{h}'],valid,arm,horizon=h,receipt=.5,impact_bps=sc['impact_bps'])
    row,result=evaluate(c,arm,scenario,phase,start,end);persist(WAVE,row,result);return row

def run():
    verify()
    if (WAVE/'started.json').exists():raise FileExistsError('Already run')
    save(WAVE/'started.json',dict(at=datetime.now(timezone.utc).isoformat()))
    args=[(a,s,ph,lo,hi) for a in ARMS for s in ['base','stress','no_receipts'] for ph,lo,hi in [('selection48h',START,START+48*3600),('later44h',START+48*3600,DEV_END),('continuous92h',START,DEV_END)]]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,*a) for a in args]):
            r=f.result();rows.append(r);save(WAVE/'results.json',rows)
            print({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','risk_breach_events']},flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
