"""Freeze hypotheses, calibrate on past dates and retain every comparison."""
import copy,gzip,json
import numpy as np
from .models import make
from ..macro_development.data import ROOT,OUT as PREVIOUS,load
from ..macro_development.experiment import protocol
from ..macro_development.engine import simulate
from ..common_comparison.models import covariance,timestamp
from ..common_comparison.experiment import json_value
from ..pairs_development.experiment import write,digest,stamp
from ..pairs_rl.verify import independent_accounting
OUT=ROOT/'replay_lab/output/portfolio_development/20261013-v7'

def main(plan_overrides=None,risk_fraction=.95):
    OUT.mkdir(parents=True,exist_ok=True);m,a,_=load();p=protocol(m)
    protected=json.loads((PREVIOUS/'preservation.json').read_text())['files']
    protected.update({str(f.relative_to(ROOT)):digest(f) for f in PREVIOUS.rglob('*') if f.is_file()})
    protected.update({str(f.relative_to(ROOT)):digest(f) for f in (ROOT/'replay_lab/macro_development').glob('*.py')})
    write(OUT/'preservation.json',dict(time=stamp(),files=protected))
    plan=dict(time=stamp(),control='Frozen v6 consensus_carry; no alteration of v6 code/data/outputs.',
        hypotheses=['Market beta concentration: conditional half BTC hedge reduces joint-downside losses; estimated vs unit beta ablation.',
          'Weak convergence premise: residual momentum trades continued relative strength rather than assuming spread convergence.',
          'Binary entry ignores signal magnitude: fixed nonlinear momentum shape reduces weak/extreme exposure vs linear gate.',
          'Market drawdown state: halve crypto sleeve at BTC 63-day price drawdown of 15%; own adaptation, not portfolio drawdown or exact paper reproduction.'],
        references=['https://www.sciencedirect.com/science/article/pii/S0275531926000255',
          'https://wp.lancs.ac.uk/fofi2026/files/2026/03/FoFI-2026-056-Bjorn-Uhl.pdf',
          'https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7115459'],
        reference_access='Residual-factor and crash-state papers: publisher/author abstracts only. Nonlinear paper methods pp6-13 read; cover dated December10 2025, hosted in 2026 conference. No exact reproduction claims.',
        risk_calibration=f'June1-July29 only; {100*risk_fraction:g}% control daily STD, multiplier capped at3, one downward correction if needed.',
        selection='Positive May17-July29 ROI above control; STD no higher; all base/stress engineering gates. Rank eligible by minimum ROI excess over three fixed 21-day development windows.',
        history_status='All history through Oct13 03UTC already reused. Later confirmation is separated in execution but NOT a fresh holdout.',
        live_orders_sent=0)
    plan.update(plan_overrides or {})
    write(OUT/'plan.json',plan)
    baseline=np.load(PREVIOUS/'consensus/targets.npz')['consensus_carry'];raw,features=make(m,baseline)
    np.savez_compressed(OUT/'raw_targets.npz',**raw);np.savez_compressed(OUT/'features.npz',**features)
    sigmas=np.full((len(m.times),10,10),np.nan)
    for i in range(168,len(m.times)):sigmas[i]=covariance(m,i,p['risk'])[1]
    rows=[];va=m.exact(timestamp('2026-06-01T00:00:00Z'));vb=m.exact(timestamp('2026-07-29T00:00:00Z'))
    def run(name,w,phase,aa,bb,scenario='base'):
        r=simulate(m,dict(target_weight=w),name,p,scenario,int(m.times[aa]),int(m.times[bb]),buffer=.2,sigmas=sigmas)
        check=independent_accounting(r,m);file=OUT/f'{phase}-{name}-{scenario}.json.gz'
        with gzip.open(file,'wt') as h:json.dump(r,h,default=json_value,allow_nan=False)
        row={k:r[k] for k in ('policy','scenario','metrics','gates','start','end')};row.update(phase=phase,file=file.name,audit=check,sha256=digest(file));rows.append(row)
        write(OUT/(file.name+'.summary.json'),row);write(OUT/'running_results.json',rows)
        print(phase,name,scenario,r['metrics']['roi'],r['metrics']['daily_vol'],flush=True);return r
    ref=run('v6_control',baseline,'calibration',va,vb);models={'v6_control':baseline};scales={}
    for name,w in raw.items():
        r=run(name,w,'calibration_unscaled',va,vb);scale=min(3.,risk_fraction*ref['metrics']['daily_vol']/max(r['metrics']['daily_vol'],1e-9))
        calibrated=run(name,w*scale,'calibration',va,vb)
        if calibrated['metrics']['daily_vol']>ref['metrics']['daily_vol']:
            scale*=min(risk_fraction,.999)*ref['metrics']['daily_vol']/calibrated['metrics']['daily_vol'];run(name,w*scale,'calibration_corrected',va,vb)
        scales[name]=scale;models[name]=w*scale
    write(OUT/'scales.json',scales);np.savez_compressed(OUT/'targets.npz',**models)
    early={'development':(1439,vb)}
    for k in range(3):
        start=timestamp('2026-05-18T00:00:00Z')+k*21*86400;early[f'window{k+1}']=(m.exact(start),m.exact(start+21*86400))
    for phase,(aa,bb) in early.items():
        for name,w in models.items():
            for scenario in (('base','stress') if phase=='development' else ('base',)):run(name,w,phase,aa,bb,scenario)
    ref=next(z for z in rows if z['policy']=='v6_control' and z['phase']=='development' and z['scenario']=='base');eligible=[]
    for name in raw:
        r=next(z for z in rows if z['policy']==name and z['phase']=='development' and z['scenario']=='base')
        if r['metrics']['roi']>max(0,ref['metrics']['roi']) and r['metrics']['daily_vol']<=ref['metrics']['daily_vol'] and all(all(z['gates'].values()) for z in rows if z['policy']==name):
            diff=[]
            for phase in ('window1','window2','window3'):
                own=next(z for z in rows if z['policy']==name and z['phase']==phase);control=next(z for z in rows if z['policy']=='v6_control' and z['phase']==phase);diff.append(own['metrics']['roi']-control['metrics']['roi'])
            eligible.append(dict(policy=name,worst_window_excess=min(diff),window_excess=diff))
    eligible.sort(key=lambda x:-x['worst_window_excess']);selection=dict(time=stamp(),choice=eligible[0]['policy'] if eligible else None,eligible=eligible,new_later_results_used=False)
    write(OUT/'selection.json',selection)
    later={'confirm_reused':(vb,len(m.times)-1),'whole_reused':(1439,len(m.times)-1),'last40_reused':(m.exact(timestamp('2026-08-26T00:00:00Z')),4826),'last_week_reused':(4826,len(m.times)-1)}
    for phase,(aa,bb) in later.items():
        for name,w in models.items():
            for scenario in ('base','stress'):run(name,w,phase,aa,bb,scenario)
    changed=[f for f,h in protected.items() if digest(ROOT/f)!=h];assert not changed
    write(OUT/'results.json',dict(time=stamp(),selection=selection,routes=rows,protected_files=len(protected),changed_protected=changed))

if __name__=='__main__':main()
