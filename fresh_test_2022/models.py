"""Same frozen model rules on the available two-asset historical universe."""
import numpy as np
from .data import ROOT,OUT,read,save,verify
from pair_model_research.adaptive_hedge import fit_pair
from pair_value_learning.scaled_coherence import project
from competition_live.profile import load_profile
import competition_live
from experiments.dense_history_pairs_20261017.model import design,labels
from experiments.rolling_feature_pairs_20261017.model import fit as direct_fit
from experiments.decision_gate_pairs_20261017.model import fit as gate_fit,action


def empty(d):
    t=d['market_times'];shape=(len(t),1)
    return dict(times=t.copy(),dense=np.array(True),pairs=np.array([[0,1]]),
        signal_times=t[:,None].copy(),source_prices=d['close'].copy(),forecast_times=t.copy(),
        beta=np.full(shape,np.nan),vol=np.full(shape,np.nan),mu=np.full(shape,np.nan),
        directions=np.zeros(shape,int),economic_available=np.zeros(shape,bool),economic_action=np.zeros(shape,int))


def candidate(d,start):
    t=d['market_times'];logs=np.log(d['close']);active=(d['volume']>0)&(d['trades']>0)
    s=empty(d);records=[];model=None
    for i,now in enumerate(t):
        if now<start:continue
        if now%86400==0:
            model=fit_pair(logs,active,i,(0,1))
            if model is not None:assert model['last_price_index']<=i
            records.append(dict(source=int(now),model=model))
        if model is None or not active[i].all():continue
        beta=model['beta'];vol=model['vol'];spread=logs[i,0]-beta*logs[i,1]
        raw=np.clip((model['equilibrium']-spread)*(1-model['rho']**72)/(1+beta),-4*vol*np.sqrt(72),4*vol*np.sqrt(72))
        corrected,_=project(np.array([raw]),np.array([beta]),[(0,1)],np.array([True]),np.array([vol*np.sqrt(72)]))
        mu=float(corrected[0]);assert abs(mu-raw)<1e-12
        s['beta'][i,0]=beta;s['vol'][i,0]=vol;s['mu'][i,0]=mu
        s['directions'][i,0]=1;s['economic_available'][i,0]=True
        s['economic_action'][i,0]=int(np.sign(mu)) if abs(mu)>.0038 else 0
    return s,records


def incumbent(d,start,progress=False):
    ds=read(ROOT/'competition_live/strategy_plan.json')['direct_spec'];gs=load_profile()['gate_spec']
    assert ds['horizon']==gs['horizon']==24
    t=d['market_times'];f=design(t,d['close'],d['volume'],d['trades'],('BTC','ETH'),[('BTC','ETH')])
    y,maturity=labels(t,d['close'],f,24);unit=f['vol']*np.sqrt(24)
    terminal=np.zeros_like(f['fresh']);terminal[:-24]=f['fresh'][24:]
    good=np.isfinite(f['x']).all(-1)&f['fresh']&terminal&np.isfinite(y)&np.isfinite(unit)&(unit>0)
    s=empty(d);s['beta']=f['beta'];s['vol']=f['vol'];records=[];direct=gate=None
    for i,now in enumerate(t):
        if now<start:continue
        if now%86400==0:
            ix=np.flatnonzero((t>=now-ds['lookback_days']*86400)&(maturity<now)&(t%(ds['sample_hours']*3600)==0)&good[:,0])
            direct=direct_fit(f['x'][ix,0],y[ix,0],np.zeros(len(ix)),maturity[ix],now,ds)
            failure=None
            try:gate=gate_fit(f['x'][ix,0],y[ix,0],unit[ix,0],maturity[ix],now,gs)
            except ArithmeticError as e:gate=None;failure=str(e)
            records.append(dict(source=int(now),direct=direct,gate=gate,failure=failure))
            if len(ix):assert int(maturity[ix].max())<now
            if progress and len(records)%30==0:print('causal comparator fits',len(records),flush=True)
        if direct is None or not f['fresh'][i,0] or not np.isfinite(f['x'][i,0]).all():continue
        u=unit[i,0]
        if not np.isfinite(u) or u<=0:continue
        z=np.r_[1.,(f['x'][i,0]-direct['center'])/direct['scale']]
        s['mu'][i,0]=np.clip(z@direct['direct'],-ds['prediction_clip'],ds['prediction_clip'])*u
        s['directions'][i,0]=1
        if gate is not None:
            z=np.r_[1.,(f['x'][i,0]-gate['center'])/gate['scale']]
            cost=2*(gs['fee_bps']+gs['impact_bps'])/10000/u
            s['economic_action'][i,0]=int(action(np.asarray(z@gate['coefficient']),cost));s['economic_available'][i,0]=True
    return s,records


def build():
    verify();d=dict(np.load(OUT/'raw_inputs.npz'));p=read(OUT/'protocol.json');start=p['test_start']//86400*86400
    for name,fn in [('candidate72',candidate),('incumbent24',incumbent)]:
        s,a=fn(d,start);np.savez_compressed(OUT/f'{name}_state.npz',**s);save(OUT/f'{name}_fit_audit.json',a)
        print('Prepared',name,'dailyfits',len(a),flush=True)


if __name__=='__main__':build()
