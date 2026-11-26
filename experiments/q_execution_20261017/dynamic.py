"""A disclosed, single causal budget follow-up. Previous results remain intact."""
import gzip,json,time
import numpy as np
from .research import HERE,ROOT,OUT,data,ts,sha,save,simulate,protected
from .policy import HistoricalController
from .metrics import official_metrics,paired_uncertainty
from experiments.cost_horizon_20261016.research import SecondController


def budget_path(times,equities,valid,lookback=168):
    scale=np.ones(len(times));current=1.;events=[]
    for i,t in enumerate(times):
        if t%86400==43200:
            recent=i-1;old=recent-lookback
            usable=old>=0 and valid[old:recent+1].all()
            ret=equities[recent]/equities[old]-1 if usable else None
            current=4. if ret is not None and ret>0 else 1.
            events.append(dict(decision_time=int(t),latest_input_time=int(times[recent]) if usable else None,
                trailing_net_return=float(ret) if ret is not None else None,scale=current))
        scale[i]=current
    return scale,events


def main():
    dest=OUT/'dynamic';dest.mkdir(exist_ok=True)
    if (dest/'frozen.json').exists():raise FileExistsError('Follow-up already frozen')
    previous=json.loads((OUT/'results.json').read_text());cfg=json.loads((HERE/'protocol.json').read_text())
    rules=json.loads((HERE/'dynamic_protocol.json').read_text());inp,a,p,m=data();protected()
    save(dest/'frozen.json',dict(at_unix=time.time(),source_sha256=sha(__file__),policy_sha256=sha(HERE/'policy.py'),
        rules_sha256=sha(HERE/'dynamic_protocol.json'),previous_results_sha256=sha(OUT/'results.json'),
        historical_independence=False,after_prior_results=True,orders_sent=0))
    p=json.loads((OUT/'simulation_protocol.json').read_text())
    fixed=json.loads((ROOT/'experiments/performance_attribution_20261016/artifacts/budget_frozen.json').read_text())['multiplier']
    second=a['candidate']*fixed
    ref=simulate(m,second,a['sigma'],p,m.exact(ts(rules['reference_initialization_utc'])),len(m.times)-1,'base',SecondController(m.times,a['retention']))
    eq=np.full(len(m.times),np.nan);valid=np.zeros(len(m.times),bool)
    for s in ref['snapshots']:
        i=m.exact(s['time']);eq[i]=s['equity'];valid[i]=True
    scale,events=budget_path(m.times,eq,valid)
    # Prefix equality and explicit strict-prior input check prevent future PNL
    # from changing an earlier risk allocation.
    cut=m.exact(ts('2026-09-01T12:00:00'))+1
    pre,_=budget_path(m.times[:cut],eq[:cut],valid[:cut])
    np.testing.assert_array_equal(pre,scale[:cut])
    assert all(e['latest_input_time'] is None or e['latest_input_time']<e['decision_time'] for e in events)
    save(dest/'budget_events.json',events)
    np.savez_compressed(dest/'budget_path.npz',times=m.times,scale=scale,reference_equity=eq)
    (dest/'runs').mkdir(exist_ok=True)
    def sim(scale,start,end,scenario):
        w=second*(scale[:,None] if isinstance(scale,np.ndarray) else scale)
        return simulate(m,w,a['sigma'],p,m.exact(ts(start)),m.exact(ts(end)),scenario,HistoricalController('partial_6h',m.times,a['retention']))
    start,end=cfg['selection_period'];cal=sim(scale,start,end,'base');target=cal['metrics']['mean_gross']
    lo,hi=1.,4.;probes=[]
    for _ in range(12):
        mid=(lo+hi)/2;r=sim(mid,start,end,'base');gross=r['metrics']['mean_gross'];probes.append(dict(scale=mid,mean_gross=gross))
        if gross>target:hi=mid
        else:lo=mid
    matched=(lo+hi)/2
    save(dest/'exposure_calibration.json',dict(scale=matched,target_mean_gross=target,probes=probes,
        selected_without_pnl=True,before_later_runs=True))
    rows=[];comparisons=[]
    for phase,(start,end) in dict(selection=cfg['selection_period'],**cfg['comparison_periods']).items():
        for scenario in cfg['scenarios']:
            local={}
            for name,sc in [('causal_7d_budget',scale),('fixed_matched_exposure',matched)]:
                r=sim(sc,start,end,scenario);r.update(phase=phase,policy=name,official_formula=official_metrics(r));r['daily']=r['official_formula']['daily']
                r['metrics']['daily_std_sample']=r['official_formula']['daily_sample_std']
                r['metrics']['mean_expected_daily_volatility']=float(np.mean([s['daily_volatility'] for s in r['snapshots']]))
                cost=p['execution']['fee_bps']+p['scenarios'][scenario]['impact_bps']
                terminal=float(abs(np.array(r['metrics']['final_qty']))@m.close[m.exact(ts(end))]*cost/10000)
                r['metrics'].update(estimated_terminal_liquidation_cost=terminal,roi_after_estimated_terminal_liquidation=r['metrics']['roi']-terminal/p['initial'])
                local[name]=r;rows.append({k:r[k] for k in ('phase','policy','scenario','metrics','gates','official_formula')})
                with gzip.open(dest/'runs'/f'{phase}-{name}-{scenario}.json.gz','wt') as f:json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
                print(phase,name,scenario,'R%',round(r['metrics']['roi']*100,6),'Q',round(r['official_formula']['financial_score'],9),flush=True)
            c=local['causal_7d_budget']
            for name in ('point_x1','partial_6h_x4','fixed_matched_exposure'):
                b=local[name] if name in local else next(r for r in previous['rows'] if r['phase']==phase and r['scenario']==scenario and r['policy']==name)
                comparisons.append(dict(phase=phase,scenario=scenario,comparator=name,
                    Q_difference=c['official_formula']['financial_score']-b['official_formula']['financial_score'],
                    R_difference=c['metrics']['roi']-b['metrics']['roi'],
                    MDD_difference=c['metrics']['max_drawdown_magnitude']-b['metrics']['max_drawdown_magnitude'],
                    uncertainty=paired_uncertainty(c['official_formula'],b['official_formula'])))
    protected();frozen=json.loads((dest/'frozen.json').read_text())
    assert sha(__file__)==frozen['source_sha256'] and sha(HERE/'policy.py')==frozen['policy_sha256']
    save(dest/'results.json',dict(rows=rows,comparisons=comparisons,fixed_exposure_scale=matched,
        causal_prefix_exact=True,reference_ledger_gates=ref['gates'],historical_independence=False,
        fixed_after_previous_results=True,orders_sent=0,automatic_promotion=False))
    print('DONE dynamic follow-up',flush=True)


if __name__=='__main__':main()
