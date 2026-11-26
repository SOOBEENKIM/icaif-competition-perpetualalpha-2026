import argparse
import copy
import gzip
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/pas_adaptive_development_20261021'
ARMS=['incumbent','size_only','adaptive_size','add24','add72','online72','combined','exposure_control']


def ts(s):return int(datetime.fromisoformat(s.replace('Z','+00:00')).timestamp())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def native(x):
    if hasattr(x,'tolist'):return x.tolist()
    raise TypeError(type(x).__name__)
def save(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,indent=2,default=native,allow_nan=False))


def freeze():
    path=OUT/'protocol.json'
    if path.exists():return json.loads(path.read_text())
    from competition_live.profile import verify_deployment
    pointer=json.loads((ROOT/'practice_live/latest.json').read_text())
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),pointer=pointer,
        input_sha=sha(ROOT/pointer['inputs']),target_sha=sha(ROOT/pointer['run']/'targets.npz'),
        hypotheses=['Preserve existing alpha when new forecasts abstain, instead of replacing it.',
            '72h terminal returns may overcome fixed entry/exit costs better than 24h.',
            'Monthly refits on mature past labels can adapt while a frozen model cannot.',
            'Restore part of original position size only as a smooth function of past net strategy returns.'],
        arms=ARMS,nominee='combined',training_start=ts('2026-04-01T00:00:00Z'),
        first_fit=ts('2026-07-01T00:00:00Z'),calibration_days=30,lookback_days=120,
        label_sample_hours=12,tree=dict(max_depth=2,eta=.04,min_child_weight=80,
            reg_lambda=20.,reg_alpha=1.,num_boost_round=80,seed=26,nthread=2),
        model='Shallow pooled boosted regression, 24h/72h source-anchored terminal return, standardized by past volatility. No named-asset exclusions.',
        calibration='Mature final 30 days, excluded from coefficient fitting. Nonnegative forecast attenuation [0,1], never invert a failed model.',
        refits='online72: first UTC day of each month; static24/static72: July1 model forever. No evaluation labels are read before maturity.',
        overlay=dict(daily_risk=.0005,maximum_asset_weight=.05,maximum_gross=.15,
            entry_cost=.0038,edge_buffer=.0005,decision_hours=6,maximum_holding_hours=72,
            entry_rule='New side only when mean remaining return exceeds roundtrip cost and buffer; close on forecast sign failure or holding expiry; one decision cool-off.',
            allocation='Add to incumbent net target, subject to same account limits. No borrowing teammate reserve. Positive and negative overlays are symmetric.'),
        adaptive_size='Daily multiplier = 1 + (1.44117372816997-1)*tanh(30-day exponentially weighted t statistic of causal incumbent net sleeve returns). Last completed day only. Range [0.558826,1.441174]. No calendar or named-asset rule.',
        exposure_control='One scalar from July1-29 mean target gross of combined / incumbent, frozen before August; actual later exposure differences reported.',
        selection='One nominee combined is fixed before outcomes; other arms are ablations, never substituted as winner after later results.',
        evaluation='Keep exact earlier August, Sep1-Oct15 and continuous Aug1-Oct15 windows, base/stress, same paired overlay and equity. No favorable-date switch.',
        promotion='Nominee beats current and size-only PnL AND Q in continuous and later windows under base AND stress; August failures disclosed. Zero causality/accounting/risk violations. Compare exposure control. Historical improvement is development evidence, not proof of generalization or automatic live approval.',
        history_reused=True,independent_test=False,live_orders=0,live_changed=False,
        prior_study='pas_allocation_study_20261021, including all failed models/results',
        deployment=verify_deployment()['files'])
    save(path,p);return p


def features(d):
    lp=pd.DataFrame(np.log(d['close']));r=lp.diff()
    vol=r.ewm(span=168,min_periods=168,adjust=False).std().clip(lower=.0005)
    values=[(lp-lp.shift(h))/(vol*np.sqrt(h)) for h in [6,24,72,168,336]]
    values += [(lp-lp.ewm(span=h,min_periods=h,adjust=False).mean())/(vol*np.sqrt(h)) for h in [24,168]]
    values += [r.ewm(span=24,min_periods=24,adjust=False).std()/vol-1]
    funding=pd.DataFrame(np.where(d['observed'],d['funding'],np.nan)).shift(1).rolling(24,min_periods=12).mean().fillna(0)
    values += [funding*24/(vol*np.sqrt(24))]
    x=np.clip(np.stack([v.to_numpy() for v in values],-1),-5,5)
    group=np.eye(4)[np.array([0,0,0,1,1,1,2,2,3,3])]
    x=np.concatenate([x,np.tile(group,(len(x),1,1))],-1)
    return x,vol.to_numpy(),funding.to_numpy()


def fit_predict(d,p,horizon,rolling,*,stop=None):
    times=d['market_times'];x,vol,funding=features(d);n,m=vol.shape
    labels=np.full_like(vol,np.nan)
    labels[:-horizon]=np.log(d['close'][horizon:]/d['close'][:-horizon])/(vol[:-horizon]*np.sqrt(horizon))
    maturity=times+horizon*3600
    fresh=(d['volume']>0)&(d['trades']>0)
    good=np.isfinite(x).all(-1)&np.isfinite(labels)&fresh
    good[:-horizon]&=fresh[horizon:]
    output=np.full_like(vol,np.nan);records=[]
    fits=[p['first_fit']]
    if rolling:
        for month in [8,9,10]:fits.append(ts(f'2026-{month:02d}-01T00:00:00Z'))
    if stop is not None:fits=[t for t in fits if t<=stop]
    for k,now in enumerate(fits):
        cutoff=now-p['calibration_days']*86400
        mask=(times>=max(p['training_start'],now-p['lookback_days']*86400))&(maturity<cutoff)&(times%(12*3600)==0)
        ii,jj=np.where(good&mask[:,None])
        if len(ii)<200:raise ValueError('Insufficient purged model rows')
        y=np.clip(labels[ii,jj],-4,4)
        params={k:v for k,v in p['tree'].items() if k!='num_boost_round'}
        params.update(objective='reg:squarederror',tree_method='hist',base_score=float(y.mean()),
            subsample=1.,colsample_bytree=1.,verbosity=0)
        weights=np.exp2(-(cutoff-times[ii])/(60*86400));weights*=len(weights)/weights.sum()
        booster=xgb.train(params,xgb.DMatrix(x[ii,jj],label=y,weight=weights,nthread=2),
            num_boost_round=p['tree']['num_boost_round'])
        cal=(times>=cutoff)&(maturity<now)&(times%86400==0)
        ci,cj=np.where(good&cal[:,None])
        raw=booster.predict(xgb.DMatrix(x[ci,cj],nthread=2))
        scale=float(np.clip(raw@labels[ci,cj]/(raw@raw+1.),0,1))
        next_fit=fits[k+1] if k+1<len(fits) else int(times[-1])+1
        mask=(times>=now)&(times<next_fit)
        indices=np.flatnonzero(mask)
        pred=booster.predict(xgb.DMatrix(x[indices].reshape(-1,x.shape[-1]),nthread=2)).reshape(len(indices),m)
        output[indices]=np.clip(pred*scale,-1.5,1.5)*vol[indices]*np.sqrt(horizon)
        blob=booster.save_raw(raw_format='ubj')
        dest=OUT/'models'/f'{horizon}_{"rolling" if rolling else "fixed"}_{now}.ubj'
        dest.parent.mkdir(exist_ok=True);dest.write_bytes(blob)
        records.append(dict(fit=now,training_last_label=int(maturity[ii].max()),calibration_last_label=int(maturity[ci].max()),
            train_cutoff=cutoff,train_rows=len(ii),calibration_rows=len(ci),calibration_scale=scale,
            asset_rows=np.bincount(jj,minlength=10).tolist(),model_sha=hashlib.sha256(blob).hexdigest(),
            calibration_mae=float(np.mean(abs(raw-labels[ci,cj]))),calibration_zero_mae=float(np.mean(abs(labels[ci,cj])))))
    return output,records


def adaptive_size(d,base):
    t=d['market_times'];r=np.zeros_like(base);r[1:]=d['close'][1:]/d['close'][:-1]-1
    lag=np.vstack([np.zeros(10),base[:-1]])
    # Costed counterfactual strategy-return feature, never an exchange-PnL claim.
    gross=(lag*r).sum(1)-(lag*d['funding']).sum(1)-abs(base-lag).sum(1)*.0019
    daily=np.flatnonzero(t%86400==0);out=np.ones(len(t));history=[];factor=1.
    prior=0
    for i in daily:
        if i-prior==24:history.append(float(gross[prior+1:i+1].sum()))
        prior=i
        if len(history)>=30:
            values=np.array(history[-90:]);weights=np.exp2(-np.arange(len(values)-1,-1,-1)/30);weights/=weights.sum()
            mean=float(weights@values);sd=float(np.sqrt(weights@((values-mean)**2)))
            factor=1+.44117372816997*np.tanh(np.sqrt(30)*mean/max(sd,1e-8))
        out[i:min(i+24,len(t))]=factor
    return out


def overlay(d,mu,h,p,sigma,trading):
    from replay_lab.common_comparison.engine import project
    _,vol,funding=features(d);times=d['market_times'];o=p['overlay']
    out=np.zeros_like(mu);held=np.zeros(10);entered=np.full(10,-10**12)
    for i,t in enumerate(times):
        if t<p['first_fit'] or t%21600:out[i]=held;continue
        net=mu[i]-funding[i]*h
        fresh=(d['volume'][i]>0)&(d['trades'][i]>0)&np.isfinite(net)
        exiting=(held!=0)&((held*net<=0)|(t-entered>=h*3600)|~np.isfinite(net))
        held[exiting]=0
        enter=(held==0)&~exiting&fresh&(abs(net)>o['entry_cost']+o['edge_buffer'])
        conviction=np.tanh(np.maximum(abs(net)-o['entry_cost'],0)/np.maximum(vol[i]*np.sqrt(h),.003))
        desired=np.sign(net)*conviction*o['daily_risk']/(np.sqrt(10)*np.maximum(vol[i]*np.sqrt(24),.003))
        desired=np.clip(np.nan_to_num(desired,nan=0),-o['maximum_asset_weight'],o['maximum_asset_weight'])
        held[enter]=desired[enter];entered[enter]=t
        # Entry units held between lifecycle changes; no six-hour chase resizing.
        risk=float(np.sqrt(max(0,24*held@sigma[i]@held)))
        div=max(1.,abs(held).sum()/o['maximum_gross'],risk/o['daily_risk'])
        held/=div
        out[i]=held
    return out


def prepare():
    p=freeze()
    if (OUT/'targets.npz').exists():raise FileExistsError('Frozen experiment already exists')
    from competition_live.profile import load_profile
    from replay_lab.common_comparison.engine import project
    from replay_lab.common_comparison.models import Market,covariance
    assert sha(ROOT/p['pointer']['inputs'])==p['input_sha']
    d=dict(np.load(ROOT/p['pointer']['inputs'],allow_pickle=False));targets=dict(np.load(ROOT/p['pointer']['run']/'targets.npz',allow_pickle=False))
    trading=load_profile()['trading'];base=targets['default'];t=d['market_times']
    m=Market(t,d['close'],d['volume'],d['trades'],d['funding'],d['observed'],tuple(trading['symbols']))
    sigma=np.tile(np.eye(10)*.0005**2,(len(t),1,1))
    for i in range(168,len(t)):sigma[i]=covariance(m,i,trading['risk'])[1]
    predictions={};records={};extras={}
    for kind,h,rolling in [('add24',24,False),('add72',72,False),('online72',72,True)]:
        mu,rec=fit_predict(d,p,h,rolling);predictions[kind]=mu;records[kind]=rec
        extras[kind]=overlay(d,mu,h,p,sigma,trading)
    factor=adaptive_size(d,base)
    weights=dict(incumbent=base,size_only=base*1.44117372816997,adaptive_size=base*factor[:,None],
        **{k:base+a for k,a in extras.items()})
    weights['combined']=base*factor[:,None]+extras['online72']
    cal=(t>=p['first_fit'])&(t<ts('2026-07-29T00:00:00Z'))
    scalar=float(np.mean(abs(weights['combined'][cal]).sum(1))/max(np.mean(abs(base[cal]).sum(1)),1e-12))
    weights['exposure_control']=base*scalar
    for name,w in weights.items():
        if name=='incumbent':continue
        for i in range(168,len(t)):w[i]=project(w[i],trading['initial'],sigma[i],trading['risk'])
    np.savez_compressed(OUT/'targets.npz',times=t,sigma=sigma,adaptive_factor=factor,**weights,
        **{k+'_forecast':v for k,v in predictions.items()},**{k+'_overlay':v for k,v in extras.items()})
    save(OUT/'models.json',records);save(OUT/'calibration.json',dict(exposure_control_scale=scalar,
        interval_end=ts('2026-07-29T00:00:00Z'),model_hyperparameters_selected_on_PAS=False))
    save(OUT/'source_freeze.json',{str(x.relative_to(ROOT)):sha(x) for x in
        [Path(__file__),OUT/'protocol.json',OUT/'models.json',OUT/'targets.npz',OUT/'calibration.json']})
    print(json.dumps(dict(model_records=records,exposure_scalar=scalar,
        overlay_asset_nonzero={k:(abs(v)>0).sum(0).tolist() for k,v in extras.items()})),flush=True)


def verify():
    from competition_live.profile import verify_deployment
    for rel,digest in json.loads((OUT/'source_freeze.json').read_text()).items():
        assert sha(ROOT/rel)==digest,rel
    assert verify_deployment()['files']==freeze()['deployment']


def run():
    verify();p=freeze()
    from competition_live.profile import BUNDLE,load_profile
    from experiments.decision_gate_pairs_20261017.portfolio import replay
    from pas_allocation_study.study import summarize
    d0=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False));state=dict(np.load(BUNDLE/'models/reference_gate_state.npz',allow_pickle=False))
    z=dict(np.load(OUT/'targets.npz',allow_pickle=False));ix=np.searchsorted(z['times'],d0['times'])
    np.testing.assert_allclose(z['incumbent'][ix],d0['first'],atol=1e-12)
    profile=load_profile();windows=[r for r in json.loads((BUNDLE/'expected.json').read_text())['runs'] if r['venue']=='reference' and r['arm']==profile['candidate']]
    rows=[]
    for win in windows:
        for arm in ARMS:
            dest=OUT/'reference'/f"{arm}_{win['phase']}_{win['scenario']}.json.gz"
            if dest.exists():
                with gzip.open(dest,'rt') as f:r=json.load(f)
            else:
                d=dict(d0);d['first']=z[arm][ix];cfg=copy.deepcopy(profile['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
                r=replay(d,state,None,cfg,profile['trading'],profile['candidate'],win['scenario'],win['start'],win['end'])
                dest.parent.mkdir(exist_ok=True)
                with gzip.open(dest,'wt') as f:json.dump(r,f,default=native)
            row=summarize(r,arm,win['phase'],win['scenario'])
            rows.append(row);save(OUT/'reference_results.json',rows);print(json.dumps(row),flush=True)
    verify()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','prepare','run'])
    globals()[ap.parse_args().stage]()
