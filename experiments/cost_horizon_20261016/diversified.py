"""Second, separately frozen hypothesis: persistent multi-asset diversification."""
import gzip,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from replay_lab.common_comparison.engine import project
from experiments.risk_preserving_20261016.policy import RiskCapController
from experiments.performance_attribution_20261016.engine import simulate
from .research import HERE,ROOT,OUT,data,ts,sha,save,paired_interval,SecondController
from replay_lab.ema_rebuild.engine import Controller


def paths(inputs,lookbacks,config):
    prices=inputs['close']; times=inputs['market_times']; log=pd.DataFrame(np.log(prices))
    vol=np.maximum(log.diff().rolling(168).std(ddof=0).to_numpy(),.0005)
    components=[np.clip(log.diff(h).to_numpy()/(vol*np.sqrt(h)),-2,2)/2 for h in lookbacks]
    trend=np.mean(components,axis=0)
    funding=pd.DataFrame(inputs['funding']).shift(1).rolling(168).mean().to_numpy()
    carry=np.clip(-funding*24/(vol*np.sqrt(24)),-1,1)
    signal=(1-config['carry_weight'])*trend+config['carry_weight']*carry
    direction=np.zeros(prices.shape[1]);entered=np.full(prices.shape[1],-1,dtype=np.int64)
    weights=np.zeros_like(prices);directions=np.zeros_like(prices)
    h=config['hysteresis']
    for i,t in enumerate(times):
        if np.isfinite(signal[i]).all() and t%21600==0:
            for j in range(len(direction)):
                if direction[j] and t-entered[j]>=h['minimum_signal_hold_hours']*3600 and direction[j]*signal[i,j]<-h['opposite_exit']:
                    direction[j]=0
                if direction[j]==0 and abs(signal[i,j])>=h['entry']:
                    direction[j]=np.sign(signal[i,j]);entered[j]=t
        directions[i]=direction
        if not np.isfinite(vol[i]).all():continue
        inverse=1/vol[i]
        shares=np.r_[.5*inverse[:3]/inverse[:3].sum(),.5*inverse[3:]/inverse[3:].sum()]
        weights[i]=direction*shares
    return weights,{'signal':signal,'directions':directions,'volatility':vol}


class SleeveController(RiskCapController):
    """Only incumbent-sleeve risk ceilings override the soft no-trade buffer."""
    def __init__(self,times,active):super().__init__();self.times=times;self.active=active
    def propose(self,time,*args,**kwargs):
        i=int(np.searchsorted(self.times,time))
        return super().propose(time,*args,retention=np.where(self.active[i],.75,1.),**kwargs)


def calibrate(m,w,sigma,baseline,p):
    ix=np.flatnonzero((m.times>=ts('2026-07-01'))&(m.times<ts('2026-08-01'))&(m.times%21600==0))
    risk=np.mean([np.sqrt(24*baseline[i]@sigma[i]@baseline[i]) for i in ix])
    gross=np.mean(abs(baseline[ix]).sum(axis=1))
    rawrisk=np.mean([np.sqrt(24*w[i]@sigma[i]@w[i]) for i in ix])
    rawgross=np.mean(abs(w[ix]).sum(axis=1))
    scale=min(risk/max(rawrisk,1e-12),1.25*gross/max(rawgross,1e-12))
    return scale,{'target_expected_vol':float(risk),'gross_ceiling':float(1.25*gross),
                  'raw_expected_vol':float(rawrisk),'raw_mean_gross':float(rawgross),'scale':float(scale)}


def main():
    out=OUT/'diversified';out.mkdir(exist_ok=True)
    if (out/'frozen.json').exists():raise FileExistsError('Phase two already frozen')
    config=json.loads((HERE/'diversified_protocol.json').read_text())
    save(out/'frozen.json',{'at_unix':time.time(),'source_sha256':sha(__file__),'protocol_sha256':sha(HERE/'diversified_protocol.json'),
        'previous_failed_experiment_sha256':sha(OUT/'results.json'),'sequential_research':True,'historical_independence':False})
    inp,a,p,m=data();fixed=json.loads((ROOT/'experiments/performance_attribution_20261016/artifacts/budget_frozen.json').read_text())['multiplier']
    second=a['candidate']*fixed;w={};cal={};components={}
    for name,lookbacks in config['lookbacks_hours'].items():w[name],components[name]=paths(inp,lookbacks,config)
    w['mixed_horizon']=(w['medium']+w['slow'])/2
    for name in list(w):
        scale,cal[name]=calibrate(m,w[name],a['sigma'],a['incumbent'],p);w[name]*=scale
    for name in list(w):w['blend_'+name]=(second+w[name])/2
    save(out/'calibration.json',cal);np.savez_compressed(out/'targets.npz',times=m.times,**w)
    # Prefix replay of the stateful signal must exactly match the full run.
    n=m.exact(ts('2026-09-01'))+1
    for name,lookbacks in config['lookbacks_hours'].items():
        prefix,_=paths({k:v[:n] for k,v in inp.items()},lookbacks,config)
        np.testing.assert_allclose(prefix*cal[name]['scale'],w[name][:n],atol=0,rtol=0)
    (out/'runs').mkdir(exist_ok=True);runs={};rows=[]
    def run(phase,name,start,end,scenario):
        if name=='first_strategy':weights=a['incumbent'];control=Controller()
        elif name=='second_strategy':weights=second;control=SecondController(m.times,a['retention'])
        else:
            variant=name if name in w else selected;weights=w[variant]
            # A cap on the incumbent half need not cap an offsetting hedge. An
            # override applies only when its risk-reduced sleeve contributes in
            # the same direction as the aggregate target.
            active=(a['retention']<1)&(weights*second>0) if variant.startswith('blend_') else np.zeros_like(weights,dtype=bool)
            control=SleeveController(m.times,active)
        r=simulate(m,weights,a['sigma'],p,m.exact(ts(start)),m.exact(ts(end)),scenario,control)
        r.update(phase=phase,policy=name);key=f'{phase}-{name}-{scenario}';runs[key]=r
        with gzip.open(out/'runs'/f'{key}.json.gz','wt') as f:json.dump(r,f,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist(),allow_nan=False)
        rows.append({k:r[k] for k in ('phase','policy','scenario','metrics','gates')})
        z=r['metrics'];print(key,'ROI',round(z['roi']*100,6),'STD',round(z['daily_std_sample']*100,6),'MDD',round(z['max_drawdown_magnitude']*100,6),'fills',z['fills'],flush=True)
        return r
    for name in ['first_strategy','second_strategy',*config['variants']]:
        for scenario in ('base','stress'):run('validation',name,'2026-08-01','2026-09-01',scenario)
    scores=[]
    for name in config['variants']:
        b,s=[runs[f'validation-{name}-{sc}'] for sc in ('base','stress')];z=b['metrics']
        good=all(b['gates'].values()) and all(s['gates'].values()) and z['roi']>0 and s['metrics']['roi']>0 and z['fills']>0
        scores.append({'variant':name,'score':z['roi']-z['daily_std_sample']-z['max_drawdown_magnitude'],'eligible':good})
    scores.sort(key=lambda z:(-z['eligible'],-z['score'],z['variant']));selected=scores[0]['variant']
    save(out/'selection.json',{'selected':selected,'scores':scores,'frozen_before_comparison':True,'at_unix':time.time()})
    for phase,start,end in [('comparison','2026-09-01','2026-10-13T12:00:00'),('first_half','2026-09-01','2026-09-22'),('second_half','2026-09-22','2026-10-13T12:00:00')]:
        for name in ('first_strategy','second_strategy','selected_third'):
            for scenario in ('base','stress'):run(phase,name,start,end,scenario)
    checks=[]
    for scenario in ('base','stress'):
        c=runs[f'comparison-selected_third-{scenario}'];z=c['metrics']
        for name in ('first_strategy','second_strategy'):
            b=runs[f'comparison-{name}-{scenario}'];bb=b['metrics']
            gates={'higher_return':z['roi']>bb['roi'],'daily_std_no_worse':z['daily_std_sample']<=bb['daily_std_sample'],
                   'mdd_no_worse':z['max_drawdown_magnitude']<=bb['max_drawdown_magnitude'],'engineering':all(c['gates'].values()),
                   'positive_base_return':z['roi']>0 if scenario=='base' else True}
            checks.append({'scenario':scenario,'comparator':name,'gates':gates,'passed':all(gates.values()),'paired_difference':paired_interval(c,b)})
    save(out/'results.json',{'selected':selected,'scores':scores,'rows':rows,'checks':checks,'calibration':cal,
         'causal_prefix_exact':True,'historical_gates_pass':all(z['passed'] for z in checks),'historical_independence':False,
         'independent_forward_completed':False,'live_replacement_approved':False,'orders_sent':0})
    print('DONE',selected,'historical_gates_pass',all(z['passed'] for z in checks),flush=True)


if __name__=='__main__':main()
