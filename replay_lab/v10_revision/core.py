"""V10-centred study. No orders or network endpoints; reused history disclosed."""
import copy, gzip, json
from pathlib import Path
import numpy as np
import pandas as pd
from ..competition_rebuild.common import ROOT, OUT as V9, sha, stamp, write, covariance_path
from ..common_comparison.models import Market
from ..macro_development.engine import simulate
from ..pairs_rl.verify import independent_accounting
from ..common_comparison.experiment import json_value
from ..competition_rebuild.regimes import regime_features, fit_jump, paths
from ..competition_rebuild.persistence import episode_overlay
from ..competition_rebuild.causal_risk import normalize_risk
from ..portfolio_rebuild_v10.lifecycle_allocation import targets

OUT=ROOT/'replay_lab/output/v10_revision/20261013-v1'
V10=ROOT/'replay_lab/output/portfolio_rebuild/20261013-v10'

def panel(z,plan):
    return Market(z['market_times'],z['close'],z['volume'],z['trades'],z['funding'],z['observed'],tuple(plan['symbols']))

def sizing(raw,sigma,times):
    return .9*np.mean([targets(raw,sigma,times,days=d) for d in (3,7)],axis=0)

def fit_regimes(m,cutoff):
    ids,x,r=regime_features(m);end=np.searchsorted(m.times[ids],cutoff)
    states=[]
    for j in range(10):
        s=fit_jump(x[max(28,end-360):end,j],r[max(28,end-360):end,j])
        states.append({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in s.items() if k!='states'})
    return dict(cutoff=cutoff,last_training_time=int(m.times[ids[end-1]]),states=states)

def frozen_prob(m,fit,training_probability):
    ids,x,r=regime_features(m);start=np.searchsorted(m.times[ids],fit['cutoff'])
    prob=np.full_like(m.close,.5);n=min(len(prob),len(training_probability));prob[:n]=training_probability[:n]
    alpha=1-np.exp(np.log(.5)/4)
    for j,ss in enumerate(fit['states']):
        s={k:np.array(v) if isinstance(v,list) else v for k,v in ss.items()}
        cost=s['cost'].copy();smooth=prob[ids[start]-1,j] if start<len(ids) else .5
        for k in range(start,len(ids)):
            xx=(x[k,j]-s['mean'])/s['std'];loss=.5*((s['centres']-xx)**2).sum(1)
            cost=loss+(cost[:,None]+10*(1-np.eye(2))).min(0);cost-=cost.min()
            smooth=(1-alpha)*smooth+alpha*float(cost.argmin()==s['bull'])
            prob[ids[k]:min(ids[k]+6,len(prob)),j]=smooth
    return prob

def slow_trend(m,sigma):
    """Independent adaptation of Sepp/Lucic (2026), equations 4.1--4.3.

    Completed UTC daily prices, 20/60-day variance-preserving EWMA of
    lag-vol-normalized returns. Bounded conviction and downside-only risk cap
    are competition adaptations, not the paper's exact implementation.
    """
    ids=np.flatnonzero(m.times%86400==0);p=pd.DataFrame(m.close[ids])
    r=p.pct_change().fillna(0.)
    vol=np.sqrt(r.pow(2).ewm(span=20,adjust=False).mean()).clip(lower=.003)
    z=r/vol.shift(1).fillna(.003)
    s=sum(np.sqrt(span)*z.ewm(span=span,adjust=False).mean() for span in (20,60))/2
    raw=np.tanh(s/2)*(.001/np.sqrt(10))/vol
    raw=raw.to_numpy();out=np.zeros_like(m.close)
    for k,i in enumerate(ids):
        if i<1440:continue
        w=np.clip(raw[k],-.135,.135)
        v=np.sqrt(max(0,24*w@sigma[i]@w))
        w/=max(1.,v/.001,abs(w).sum()/.54,abs(w.sum())/.36)
        out[i:ids[k+1] if k+1<len(ids) else len(out)]=w
    return out

def components(m,sigma,base,forecast,prob):
    _,jm=paths(m,prob,base);cg=episode_overlay(m,base,forecast,'sticky')
    gated=(jm+cg)/2
    ratio=np.divide(gated,base,out=np.ones_like(base),where=abs(base)>1e-15)
    normal=normalize_risk(base,gated,sigma,m.times)
    return dict(v10_frozen=sizing(normal,sigma,m.times),
                gate_preserved=sizing(base,sigma,m.times)*ratio,
                slow_trend=slow_trend(m,sigma)),ratio

def cap(w):
    w=np.clip(w,-.135,.135)
    div=np.maximum(1.,np.maximum(abs(w).sum(1)/.54,abs(w.sum(1))/.36))
    return w/div[:,None]

def policies(parts,fit):
    p={k:cap(v*fit.get('scales',{}).get(k,1.)) for k,v in parts.items()}
    p['mix25']=.75*p['gate_preserved']+.25*p['slow_trend']
    p['mix50']=.5*p['gate_preserved']+.5*p['slow_trend']
    for key,w in fit.get('ensemble_weights',{}).items():
        p[key]=sum(a*p[k] for a,k in zip(w,fit['experts']))
    return p

def replay(m,p,sigma,w,name,phase,start,end,scenario='base'):
    result=simulate(m,{'target_weight':w},name,p,scenario,int(start),int(end),buffer=.2,sigmas=sigma)
    audit=independent_accounting(result,m)
    assert audit['passed'] and result['gates']['information_time'] and result['gates']['accounting']
    path=OUT/'runs'/phase/f'{name}-{scenario}.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    with gzip.open(path,'wt') as f:json.dump(result,f,allow_nan=False,default=json_value)
    row={k:result[k] for k in ('policy','scenario','start','end','metrics','gates')}
    row.update(phase=phase,audit=audit,path=str(path.relative_to(OUT)))
    print(phase,name,scenario,'ROI',round(result['metrics']['roi']*100,5),'STD',round(result['metrics']['daily_vol']*100,5),flush=True)
    return row,result

def select(rows):
    if any(r['phase']!='validation' for r in rows):raise ValueError('Only validation may select')
    idx={(r['policy'],r['scenario']):r for r in rows}
    ref=idx['v10_original','base']['metrics'];stress=idx['v10_original','stress']['metrics']
    eligible=[];ranked=[]
    for name in sorted({r['policy'] for r in rows}-{'v10_original'}):
        b,s=idx[name,'base'],idx[name,'stress'];m=b['metrics']
        if not all(b['gates'].values()) or not all(s['gates'].values()):continue
        item=(m['roi']/max(m['daily_vol'],1e-6),name)
        ranked.append(item)
        if m['roi']>max(0,ref['roi']) and m['daily_vol']<=ref['daily_vol'] and s['metrics']['roi']>=stress['roi']:
            eligible.append(item)
    if not ranked:raise ValueError('No engineering-valid candidate')
    winner=sorted(eligible or ranked,key=lambda x:(-x[0],x[1]))[0][1]
    return dict(policy=winner,validation_improvement=bool(eligible),eligible=eligible,ranked=ranked,
                interpretation='If none dominates baseline, highest validation utility is a diagnostic nominee, not an adopted improvement.')

def source_hashes():
    # All local Python imports, not only the study entry point.
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'replay_lab').rglob('*.py')) if 'output' not in p.parts and '__pycache__' not in p.parts}

def check_freeze():
    selected=json.loads((OUT/'selected.json').read_text())
    for path,digest in selected['source_hashes'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Changed frozen source: '+path)
    for path,digest in selected['artifact_hashes'].items():
        if sha(OUT/path)!=digest:raise ValueError('Changed frozen artifact: '+path)
    return selected
