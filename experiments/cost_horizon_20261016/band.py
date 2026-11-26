"""A fixed 5% inward risk-ceiling band: remove repeated micro reductions.

This third, disclosed follow-up is an execution hypothesis after both return
forecasting and macro diversification failed return-improvement gates. All
earlier results remain available. No parameter search is performed here.
"""
import gzip,json,time
import numpy as np
from replay_lab.common_comparison.engine import rounded,project,risk_breaches
from experiments.performance_attribution_20261016.engine import simulate
from .research import HERE,ROOT,OUT,data,ts,sha,save,paired_interval,SecondController
from replay_lab.ema_rebuild.engine import Controller


class InwardBandController(SecondController):
    def propose(self,time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs):
        action=super().propose(time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs)
        if action is None or action['risk_forced']:return action
        target=np.array(action['target_qty']);before=target.copy()
        binding=np.array(action['cap_overrode_soft_constraint'],bool)
        same=target*np.asarray(qty)>0
        # Trade inward only when the risk ceiling overrode the soft buffer.
        # This grants 5% headroom without ever raising the risk ceiling.
        target=np.where(binding&same,.95*target,target)
        target=project(target*prices/equity,equity,sigma,protocol['risk'])*equity/prices
        target=np.array([rounded(q,d) for q,d in zip(target,protocol['execution']['decimals'])])
        # Preserve unchanged lot quantities exactly.
        target=np.where((~binding)&(abs(target-before)<1e-10),before,target)
        if risk_breaches(target,equity,prices,sigma,protocol['risk']):raise AssertionError('Band must respect hard limits')
        active=np.array(action['risk_cap_active'],bool);ceiling=np.array(action['risk_cap_qty'])
        if np.any(abs(target[active])>ceiling[active]+1e-10):raise AssertionError('Risk ceiling restored')
        action.update(target_qty=target.tolist(),inward_band_fraction=.05,band_binding=binding.tolist())
        return action


def main():
    out=OUT/'inward_band';out.mkdir(exist_ok=True)
    if (out/'frozen.json').exists():raise FileExistsError('Fixed band trial already exists')
    save(out/'frozen.json',{'at_unix':time.time(),'source_sha256':sha(__file__),'inward_band_fraction':.05,
        'hypothesis':'Repeated cap-overrides defeat the soft 20% hold band and create micro reductions. Trade 5% inside a binding soft ceiling once, while never exceeding the original ceiling or hard risk limits.',
        'parameter_choice':'One fixed 5% band, one quarter of the inherited 20% soft buffer. No candidate search.',
        'after_previous_results':{'forecasting':sha(OUT/'results.json'),'diversification':sha(OUT/'diversified/results.json')},
        'historical_independence':False,'live_orders_permitted':False})
    inp,a,p,m=data();fixed=json.loads((ROOT/'experiments/performance_attribution_20261016/artifacts/budget_frozen.json').read_text())['multiplier']
    second=a['candidate']*fixed;rows=[];runs={};(out/'runs').mkdir(exist_ok=True)
    for phase,start,end in [('validation','2026-08-01','2026-09-01'),('comparison','2026-09-01','2026-10-13T12:00:00'),('first_half','2026-09-01','2026-09-22'),('second_half','2026-09-22','2026-10-13T12:00:00')]:
        for name in ('first_strategy','second_strategy','inward_band'):
            for scenario in ('base','stress'):
                control=Controller() if name=='first_strategy' else (InwardBandController if name=='inward_band' else SecondController)(m.times,a['retention'])
                r=simulate(m,a['incumbent'] if name=='first_strategy' else second,a['sigma'],p,m.exact(ts(start)),m.exact(ts(end)),scenario,control)
                r.update(phase=phase,policy=name);key=f'{phase}-{name}-{scenario}';runs[key]=r
                with gzip.open(out/'runs'/f'{key}.json.gz','wt') as f:json.dump(r,f,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist(),allow_nan=False)
                rows.append({k:r[k] for k in ('phase','policy','scenario','metrics','gates')});z=r['metrics']
                print(key,'ROI',round(z['roi']*100,6),'STD',round(z['daily_std_sample']*100,6),'MDD',round(z['max_drawdown_magnitude']*100,6),'fills',z['fills'],flush=True)
    checks=[]
    for scenario in ('base','stress'):
        c=runs[f'comparison-inward_band-{scenario}'];z=c['metrics']
        for name in ('first_strategy','second_strategy'):
            b=runs[f'comparison-{name}-{scenario}'];bb=b['metrics']
            gates={'higher_return':z['roi']>bb['roi'],'daily_std_no_worse':z['daily_std_sample']<=bb['daily_std_sample'],
                   'mdd_no_worse':z['max_drawdown_magnitude']<=bb['max_drawdown_magnitude'],'engineering':all(c['gates'].values()),
                   'positive_base_return':z['roi']>0 if scenario=='base' else True}
            checks.append({'scenario':scenario,'comparator':name,'gates':gates,'passed':all(gates.values()),'paired_difference':paired_interval(c,b)})
    save(out/'results.json',{'candidate':'inward_band','rows':rows,'checks':checks,'historical_gates_pass':all(x['passed'] for x in checks),
        'historical_independence':False,'independent_forward_completed':False,'orders_sent':0,'live_replacement_approved':False})
    print('DONE historical_gates_pass',all(x['passed'] for x in checks),flush=True)


if __name__=='__main__':main()
