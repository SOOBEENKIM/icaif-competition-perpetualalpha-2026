"""Causal single-venue funding signals; these are NOT hedged arbitrage."""
import copy,gzip,json
import numpy as np
import pandas as pd
from .data import ROOT,OUT,load
from .experiment import protocol
from .engine import simulate
from .model import inputs,targets
from ..ema_rebuild.models import build
from ..common_comparison.models import covariance,timestamp
from ..common_comparison.experiment import json_value
from ..pairs_development.experiment import write,digest,stamp
from ..pairs_rl.verify import independent_accounting

DEST=OUT/'carry'

def signals(m):
    # Settlement may be published just AFTER the timestamp, so use t-1 only.
    f=pd.DataFrame(m.funding).shift(1)
    fast=f.ewm(span=24,adjust=False,min_periods=24).mean().to_numpy()
    slow=f.ewm(span=168,adjust=False,min_periods=168).mean().to_numpy()
    same=fast*slow>0
    conservative=np.where(same,np.sign(slow)*np.minimum(abs(fast),abs(slow)),0.)
    conservative=np.nan_to_num(conservative,nan=0.)
    x,sleeve=inputs(m)
    direction=-np.sign(conservative)
    # 7-day *forecast* carry must cover entry AND exit 9+10bp on each side.
    # This is a price-risk-bearing position; price PnL may overwhelm carry.
    carry=abs(conservative)*168
    trend=x[:,:,[1,2]].mean(-1)
    result={}
    for name,align in [('carry_only',False),('carry_trend',True)]:
        held=np.zeros(len(m.symbols));w=np.zeros_like(m.close)
        for i,t in enumerate(m.times):
            if t%21600==0:
                for j in range(len(m.symbols)):
                    agreement=(not align) or direction[i,j]*trend[i,j]>0
                    if direction[i,j]!=np.sign(held[j]) or carry[i,j]<.0019 or not agreement:
                        held[j]=0
                    if carry[i,j]>.0038 and agreement:
                        held[j]=direction[i,j]*sleeve[i,j]
                w[i]=held
            else:w[i]=held
        result[name]=w
    return result,dict(forecast=conservative,weekly_carry=carry,trend=trend)

def main():
    DEST.mkdir(parents=True,exist_ok=True);m,a,_=load();p=protocol(m)
    write(DEST/'plan.json',dict(time=stamp(),hypothesis='Funding was included in costs but not used as an input. Test persistent carry after costs, with and without price-trend agreement, uniformly across all ten assets.',
        parameters=dict(funding_ewm_hours=[24,168],funding_lag_hours=1,forecast_horizon_hours=168,entry_cost=.0038,exit_hysteresis=.0019),
        selection='Highest validation ROI above EMA and no higher validation daily STD; positive May17-July29 development ROI, all engineering gates. No asset chosen by full-period performance.',
        deviations='Unhedged single-venue directional carry, not CEX-DEX or spot-perpetual arbitrage. Reference market funding is NOT pa funding; oracle prices missing, close-price funding notional proxy retained.',
        references=['https://link.springer.com/article/10.1007/s42521-026-00213-3','https://arxiv.org/html/2605.11263v1','https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding'],
        prior_inspection='Per-asset full-period returns and funding sums inspected before hypothesis; all periods are adaptively reused development.',independent_holdout=False))
    selected=json.loads((ROOT/'replay_lab/ema_development/selected_policy.json').read_text());manifest=json.loads((ROOT/'replay_lab/output/retained_strategies/manifest.json').read_text())
    ema=build(a,manifest,selected)['ema_base']['target_weight'];raw,features=signals(m)
    raw.update(ema_carry=.8*ema+raw['carry_only'],ema_carry_trend=.8*ema+raw['carry_trend'])
    np.savez_compressed(DEST/'raw_targets.npz',**raw);np.savez_compressed(DEST/'features.npz',**features)
    sigmas=np.full((len(m.times),10,10),np.nan)
    for i in range(168,len(m.times)):sigmas[i]=covariance(m,i,p['risk'])[1]
    va=m.exact(timestamp('2026-06-01T00:00:00Z'));vb=m.exact(timestamp('2026-07-29T00:00:00Z'));rows=[]
    def run(name,w,phase,aa,bb,scenario='base'):
        r=simulate(m,dict(target_weight=w),name,p,scenario,int(m.times[aa]),int(m.times[bb]),buffer=.025 if name=='ema_original' else .2,sigmas=sigmas)
        check=independent_accounting(r,m);file=DEST/f'{phase}-{name}-{scenario}.json.gz'
        with gzip.open(file,'wt') as h:json.dump(r,h,default=json_value,allow_nan=False)
        row={k:r[k] for k in ('policy','scenario','metrics','gates','start','end')};row.update(phase=phase,file=file.name,audit=check,sha256=digest(file));rows.append(row)
        write(DEST/(file.name+'.summary.json'),row);write(DEST/'running_results.json',rows)
        print(phase,name,scenario,r['metrics']['roi'],r['metrics']['daily_vol'],flush=True)
        return r
    ref=run('ema_original',ema,'validation',va,vb);models={'ema_original':ema};scales={}
    for name,w in raw.items():
        r=run(name,w,'calibration_unscaled',va,vb)
        scale=min(3.,.95*ref['metrics']['daily_vol']/max(r['metrics']['daily_vol'],1e-9))
        scales[name]=scale;models[name]=w*scale
        run(name,models[name],'validation',va,vb)
    write(DEST/'scales.json',scales);np.savez_compressed(DEST/'targets.npz',**models)
    periods={'development':(1439,vb),'select':(vb,m.exact(timestamp('2026-08-26T00:00:00Z'))),
        'audit_reused':(m.exact(timestamp('2026-08-26T00:00:00Z')),m.exact(timestamp('2026-10-05T16:00:00Z'))),
        'extension_reused':(m.exact(timestamp('2026-10-05T16:00:00Z')),len(m.times)-1)}
    selection=None
    for phase,(aa,bb) in periods.items():
        for key,w in models.items():
            for scenario in ('base','stress'):run(key,w,phase,aa,bb,scenario)
        if phase=='development':
            eligible=[]
            for name in raw:
                v=next(z for z in rows if z['policy']==name and z['phase']=='validation');d=next(z for z in rows if z['policy']==name and z['phase']=='development' and z['scenario']=='base')
                if v['metrics']['roi']>ref['metrics']['roi'] and v['metrics']['daily_vol']<=ref['metrics']['daily_vol'] and d['metrics']['roi']>0 and all(all(z['gates'].values()) for z in rows if z['policy']==name):eligible.append(v)
            eligible.sort(key=lambda z:-z['metrics']['roi']);selection=dict(time=stamp(),choice=eligible[0]['policy'] if eligible else None,eligible=[z['policy'] for z in eligible],future_metrics_used_for_choice=False)
            write(DEST/'selection.json',selection)
    write(DEST/'results.json',dict(selection=selection,routes=rows))

if __name__=='__main__':main()
