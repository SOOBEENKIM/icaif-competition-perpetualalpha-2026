"""Match incumbent predicted portfolio risk with past-only covariance."""
import json
import numpy as np
from .common import OUT,load,write,stamp,covariance_path,run


def normalize_risk(base,raw,sigmas,times):
    result=raw.copy();scale=1.
    for i in range(1440,len(base)):
        if times[i]%21600==0:
            reference=np.sqrt(max(0,float(base[i]@sigmas[i]@base[i])))
            candidate=np.sqrt(max(0,float(raw[i]@sigmas[i]@raw[i])))
            scale=min(2.,.95*reference/max(candidate,1e-12))
        result[i]=raw[i]*scale
    return result


def main():
    stage=OUT/'causal_risk';stage.mkdir(exist_ok=True)
    if (stage/'plan.json').exists():raise RuntimeError('No overwrite')
    write(stage/'plan.json',dict(created_at=stamp(),reason='Earlier risk matching was fitted onJune1-July29 and only executable as a fixed rule afterJuly29. Development and whole metrics were diagnostic, not a fully causal full-period backtest.',
      hypothesis='Compare overlays at95% incumbent predicted portfolio risk using only covariance observed at each decision, without any future evaluation-period risk calibration.',
      formula='raw_weight * min(2, .95*sqrt(base_weight.T Sigma base_weight)/sqrt(raw_weight.T Sigma raw_weight))',
      covariance='Same168hour causal covariance and shrinkage as common risk controller',
      candidates=['v7_control','constant95','chronos_all_sticky','chronos_groups_sticky','ridge_utility_sticky','fixed_ridge_chronos','jm_gate'],
      selection='Positive developmentreturn abovev7,STD<=v7,gates; best worst3x21dayexcess. Also retain later and whole outcomes with all failures.',
      calibration='NONE for this additional risk overlay; inherited v7 scale and its historical development remain disclosed.',
      all_history_reused=True,prior_later_history_seen=True,live_orders_sent=0))
    m,z,b,p=load();sigma=covariance_path(m,p);base=np.load(OUT/'expected_targets.npy')
    raw=dict(np.load(OUT/'persistence/raw_targets.npz'));models={'v7_control':base,'constant95':base*.95}
    for name in ('chronos_all_sticky','chronos_groups_sticky','ridge_utility_sticky'):models[name]=normalize_risk(base,raw[name],sigma,m.times)
    models['fixed_ridge_chronos']=normalize_risk(base,(raw['chronos_all_sticky']+raw['ridge_utility_sticky'])/2,sigma,m.times)
    models['jm_gate']=normalize_risk(base,np.load(OUT/'regimes/raw_targets.npz')['v7_jm_gate'],sigma,m.times)
    np.savez_compressed(stage/'targets.npz',**models)
    idx=lambda d:m.exact(int(np.datetime64(d,'s').astype('int64')));cut=idx('2026-07-29T00:00:00');rows=[]
    def evaluate(name,w,phase,a,b,scenario):
        r=run(m,p,w,name,a,b,sigma,scenario,stage/'runs'/phase)
        row={k:r[k] for k in ('policy','scenario','start','end','metrics','gates')};row['phase']=phase;rows.append(row)
        write(stage/'running_results.json',rows);print(phase,name,scenario,round(r['metrics']['roi']*100,6),round(r['metrics']['daily_vol']*100,6),flush=True)
    phases={'development':(1439,cut)}
    for k in range(3):
        a=idx('2026-05-18T00:00:00')+k*504;phases[f'window{k+1}']=(a,a+504)
    for phase,(a,b) in phases.items():
        for name,w in models.items():
            for scenario in (('base','stress') if phase=='development' else ('base',)):evaluate(name,w,phase,a,b,scenario)
    indexed={(r['phase'],r['policy'],r['scenario']):r for r in rows};ref=indexed['development','v7_control','base'];eligible=[]
    for name in models:
        if name in ('v7_control','constant95'):continue
        r=indexed['development',name,'base'];stress=indexed['development',name,'stress']
        if r['metrics']['roi']>max(0,ref['metrics']['roi']) and r['metrics']['daily_vol']<=ref['metrics']['daily_vol'] and all(r['gates'].values()) and all(stress['gates'].values()):
            diffs=[indexed[f'window{k}',name,'base']['metrics']['roi']-indexed[f'window{k}','v7_control','base']['metrics']['roi'] for k in (1,2,3)]
            eligible.append(dict(policy=name,window_excess=diffs,worst_window_excess=min(diffs)))
    eligible.sort(key=lambda r:-r['worst_window_excess'])
    write(stage/'selection.json',dict(time=stamp(),eligible=eligible,nominee=eligible[0]['policy'] if eligible else None,all_history_reused=True,prior_later_history_seen=True))
    for phase,(a,b) in {'later_reused':(cut,len(m.times)-1),'whole_reused':(1439,len(m.times)-1),
      'last40_reused':(idx('2026-08-26T00:00:00'),idx('2026-10-05T16:00:00'))}.items():
        for name,w in models.items():
            for scenario in ('base','stress'):evaluate(name,w,phase,a,b,scenario)
    write(stage/'results.json',dict(completed_at=stamp(),routes=rows,live_orders_sent=0))

if __name__=='__main__':main()
