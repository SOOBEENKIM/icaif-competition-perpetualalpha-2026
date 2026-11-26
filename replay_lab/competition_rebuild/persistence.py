"""Causal cost diagnosis: lock forecast risk decisions within signal episodes."""
import json
import numpy as np
from .common import OUT,load,write,stamp,sha,covariance_path,run
from .models import rolling,HORIZON


def episode_overlay(m,base,pred,mode,start=1440):
    fund=np.vstack([np.zeros((1,10)),rolling(m.funding,24,'mean')[:-1]])*HORIZON
    out=base.copy();mult=np.ones(10);last_sign=np.zeros(10)
    for i in range(start,len(base)):
        if m.times[i]%21600==0:
            sign=np.sign(base[i]);new=(sign!=last_sign)
            conflict=np.isfinite(pred[i])&np.isfinite(fund[i])&(sign*(pred[i]-fund[i])<-.0038)
            mult[new]=np.where(conflict[new],.5,1.)
            if mode=='sticky':mult=np.where(conflict,np.minimum(mult,.5),mult)
            mult[sign==0]=1.;last_sign=sign.copy()
        out[i]=base[i]*mult
    return out


def main():
    stage=OUT/'persistence';stage.mkdir(exist_ok=True)
    plan=dict(created_at=stamp(),previous_stage='10 new policies, none nominated; all failures retained',
        evidence='Conflict overlay slightly improved gross price PnL in development but produced97fills versus38, losing the gain to costs.',
        hypothesis='Lock half-exposure decisions at entry, or allow one reduction per continuous same-sign baseline episode without restoration before exit. Avoid recurrent half/full trades.',
        candidates=['v7_control','constant_half']+[f'{model}_{mode}' for model in ('chronos_groups','chronos_all','ridge_utility') for mode in ('entry','sticky')],
        entry_threshold=.0038,conflict_fraction=.5,
        calibration='June1-July29:95% control STD, scale capped3, one downward correction',
        selection='Positive net return above v7, no higher daily STD, base/stress gates, rank worst three21day windows. Later results not used by selection function.',
        all_history_reused=True,prior_later_history_seen=True,live_orders_sent=0)
    if (stage/'plan.json').exists():raise RuntimeError('Never overwrite a completed plan; create a separate revision')
    write(stage/'plan.json',plan)
    m,z,b,p=load();sigma=covariance_path(m,p);base=np.load(OUT/'expected_targets.npy')
    models={'v7_control':base,'constant_half':base*.5}
    for model in ('chronos_groups','chronos_all','ridge_utility'):
        pred=np.load(OUT/(model+'.npz'))['point'] if model.startswith('chronos') else np.load(OUT/'ridge_forecasts.npz')[model]
        for mode in ('entry','sticky'):models[f'{model}_{mode}']=episode_overlay(m,base,pred,mode)
    np.savez_compressed(stage/'raw_targets.npz',**models)
    idx=lambda date:m.exact(int(np.datetime64(date,'s').astype('int64')))
    a=idx('2026-06-01T00:00:00');cut=idx('2026-07-29T00:00:00');rows=[]
    def evaluate(name,w,phase,start,end,scenario='base'):
        r=run(m,p,w,name,start,end,sigma,scenario,stage/'runs'/phase)
        row={k:r[k] for k in ('policy','scenario','start','end','metrics','gates')};row['phase']=phase;rows.append(row)
        write(stage/'running_results.json',rows)
        print(phase,name,scenario,round(r['metrics']['roi']*100,6),round(r['metrics']['daily_vol']*100,6),flush=True)
        return r
    ref=evaluate('v7_control',base,'calibration',a,cut);scaled={'v7_control':base};scales={'v7_control':1.}
    for name,w in models.items():
        if name=='v7_control':continue
        r=evaluate(name,w,'unscaled_calibration',a,cut)
        s=min(3.,.95*ref['metrics']['daily_vol']/max(r['metrics']['daily_vol'],1e-9))
        r=evaluate(name,w*s,'calibration',a,cut)
        if r['metrics']['daily_vol']>ref['metrics']['daily_vol']:
            s*=.95*ref['metrics']['daily_vol']/r['metrics']['daily_vol'];evaluate(name,w*s,'calibration_corrected',a,cut)
        scaled[name]=w*s;scales[name]=s
    write(stage/'scales.json',scales);np.savez_compressed(stage/'targets.npz',**scaled)
    phases={'development':(1439,cut)}
    for k in range(3):
        aa=idx('2026-05-18T00:00:00')+k*504;phases[f'window{k+1}']=(aa,aa+504)
    for phase,(aa,bb) in phases.items():
        for name,w in scaled.items():
            for scenario in (('base','stress') if phase=='development' else ('base',)):evaluate(name,w,phase,aa,bb,scenario)
    index={(r['phase'],r['policy'],r['scenario']):r for r in rows};ref=index['development','v7_control','base'];eligible=[]
    for name in scaled:
        if name in ('v7_control','constant_half'):continue
        r=index['development',name,'base'];s=index['development',name,'stress']
        if r['metrics']['roi']>max(0,ref['metrics']['roi']) and r['metrics']['daily_vol']<=ref['metrics']['daily_vol'] and all(r['gates'].values()) and all(s['gates'].values()):
            differences=[index[f'window{k}',name,'base']['metrics']['roi']-index[f'window{k}','v7_control','base']['metrics']['roi'] for k in (1,2,3)]
            eligible.append(dict(policy=name,window_excess=differences,worst_window_excess=min(differences)))
    eligible.sort(key=lambda r:-r['worst_window_excess'])
    write(stage/'selection.json',dict(time=stamp(),nominee=eligible[0]['policy'] if eligible else None,eligible=eligible,
        all_history_reused=True,prior_later_history_seen=True))
    for phase,(aa,bb) in {'later_reused':(cut,len(m.times)-1),'whole_reused':(1439,len(m.times)-1),
                         'last40_reused':(idx('2026-08-26T00:00:00'),idx('2026-10-05T16:00:00'))}.items():
        for name,w in scaled.items():
            for scenario in ('base','stress'):evaluate(name,w,phase,aa,bb,scenario)
    write(stage/'results.json',dict(completed_at=stamp(),routes=rows,live_orders_sent=0))

if __name__=='__main__':main()
