"""Freeze, select on August, then disclose reused-history later-period results."""
import gzip
import json
import time
from pathlib import Path
import numpy as np

from experiments.cost_horizon_20261016.research import data,ts,sha,save,SecondController
from experiments.performance_attribution_20261016.engine import simulate
from experiments.complementary_alpha_20261017.research import protected
from replay_lab.ema_rebuild.engine import Controller
from .policy import HistoricalController
from .metrics import official_metrics,paired_uncertainty

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'artifacts'


def main():
    started=time.monotonic();OUT.mkdir(exist_ok=True)
    if (OUT/'frozen.json').exists():raise FileExistsError('This comparison is frozen')
    cfg=json.loads((HERE/'protocol.json').read_text());inp,a,p,m=data()
    checked=protected()
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources={str(q.relative_to(ROOT)):sha(q) for q in [*HERE.glob('*.py'),HERE/'protocol.json']},
        inputs={str(q.relative_to(ROOT)):sha(q) for q in [ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',ROOT/'experiments/risk_preserving_20261016/artifacts/targets.npz']},
        protected_counts=checked,orders_sent=0,sealed_forward_outcomes_read=False,historical_independence=False))
    p['scenarios']['execution_stress']={**p['scenarios']['stress'],'adverse_funding_bps_hour':0.}
    p['scenarios']['funding_stress']=p['scenarios']['stress'].copy()
    save(OUT/'simulation_protocol.json',p)
    fixed=json.loads((ROOT/'experiments/performance_attribution_20261016/artifacts/budget_frozen.json').read_text())['multiplier']
    second=a['candidate']*fixed
    variants={f'{name}_x{int(scale)}':dict(policy=name,scale=scale) for name in cfg['execution_policies'] for scale in cfg['budget_scales']}
    (OUT/'runs').mkdir(exist_ok=True);rows=[];runs={}

    def run(phase,name,start,end,scenario):
        if name=='first_strategy':w=a['incumbent'];control=Controller()
        else:
            v=variants[name];w=second*v['scale'];control=HistoricalController(v['policy'],m.times,a['retention'])
        r=simulate(m,w,a['sigma'],p,m.exact(ts(start)),m.exact(ts(end)),scenario,control)
        r['official_formula']=official_metrics(r)
        r['daily']=r['official_formula']['daily']
        metrics=r['metrics'];fee=p['execution']['fee_bps']+p['scenarios'][scenario]['impact_bps']
        terminal=float(abs(np.array(metrics['final_qty']))@m.close[m.exact(ts(end))]*fee/10000)
        metrics['estimated_terminal_liquidation_cost']=terminal
        metrics['roi_after_estimated_terminal_liquidation']=metrics['roi']-terminal/p['initial']
        metrics['mean_expected_daily_volatility']=float(np.mean([s['daily_volatility'] for s in r['snapshots']]))
        metrics['daily_std_sample']=r['official_formula']['daily_sample_std'];metrics['complete_days']=len(r['daily'])
        r.update(phase=phase,policy=name);key=f'{phase}-{name}-{scenario}';runs[key]=r
        with gzip.open(OUT/'runs'/f'{key}.json.gz','wt') as f:json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        row={k:r[k] for k in ('phase','policy','scenario','metrics','gates','official_formula')}
        rows.append(row)
        print(key,'R%',round(metrics['roi']*100,6),'Q',round(r['official_formula']['financial_score'],9),
            'sigma%',round(r['official_formula']['volatility_horizon']*100,6),'fees',round(metrics['fees'],2),flush=True)
        return r

    start,end=cfg['selection_period']
    for name in variants:
        for scenario in cfg['scenarios']:run('selection',name,start,end,scenario)
    # The wrapper must reproduce the currently used controller exactly.
    old=simulate(m,second,a['sigma'],p,m.exact(ts(start)),m.exact(ts(end)),'base',SecondController(m.times,a['retention']))
    current=runs['selection-point_x1-base']
    assert old['fills']==current['fills']
    np.testing.assert_array_equal([s['equity'] for s in old['snapshots']],[s['equity'] for s in current['snapshots']])
    rankings=[]
    for name,v in variants.items():
        if v['policy']=='point':continue
        r=[runs[f'selection-{name}-{s}'] for s in cfg['scenarios']]
        eligible=all(all(x['gates'].values()) for x in r) and r[0]['metrics']['roi']>0 and r[0]['metrics']['fills']>0
        deltas=[runs[f'selection-{name}-{s}']['official_formula']['financial_score']-runs[f'selection-point_x1-{s}']['official_formula']['financial_score'] for s in ('base','execution_stress')]
        rankings.append(dict(name=name,eligible=eligible,min_Q_improvement=min(deltas),scale=v['scale']))
    rankings.sort(key=lambda r:(not r['eligible'],-r['min_Q_improvement'],r['scale'],r['name']))
    selected=rankings[0]['name'];v=variants[selected];same_scale=f"point_x{int(v['scale'])}"
    selection=dict(selected=selected,configuration=v,same_budget_comparator=same_scale,rankings=rankings,
        selected_before_later_runs=True,at_unix=time.time(),source_sha256=sha(__file__),
        improves_selection=rankings[0]['eligible'] and rankings[0]['min_Q_improvement']>0,
        no_automatic_deployment=True)
    save(OUT/'selection.json',selection)
    print('FROZEN SELECTION',selected,flush=True)
    selected_hash=sha(OUT/'selection.json')
    comparisons=[]
    for phase,(start,end) in cfg['comparison_periods'].items():
        for name in dict.fromkeys(['first_strategy','point_x1',same_scale,selected]):
            for scenario in cfg['scenarios']:run(phase,name,start,end,scenario)
        for scenario in cfg['scenarios']:
            c=runs[f'{phase}-{selected}-{scenario}']
            for name in dict.fromkeys(['first_strategy','point_x1',same_scale]):
                b=runs[f'{phase}-{name}-{scenario}'];cc=c['official_formula'];bb=b['official_formula']
                comparisons.append(dict(phase=phase,scenario=scenario,comparator=name,
                    Q_difference=cc['financial_score']-bb['financial_score'],R_difference=c['metrics']['roi']-b['metrics']['roi'],
                    sigma_difference=cc['volatility_horizon']-bb['volatility_horizon'],
                    MDD_difference=c['metrics']['max_drawdown_magnitude']-b['metrics']['max_drawdown_magnitude'],
                    fees_difference=c['metrics']['fees']-b['metrics']['fees'],
                    same_budget=name==same_scale,uncertainty=paired_uncertainty(cc,bb)))
    frozen=json.loads((OUT/'frozen.json').read_text())
    for path,h in frozen['sources'].items():assert sha(ROOT/path)==h,path
    assert sha(OUT/'selection.json')==selected_hash
    protected()
    save(OUT/'results.json',dict(selection=selection,rows=rows,comparisons=comparisons,
        baseline_reproduced_exactly=True,protected_sources_unchanged=True,orders_sent=0,
        historical_independence=False,independent_par_validation_completed=False,
        elapsed_seconds=time.monotonic()-started))
    print('DONE',len(rows),'runs',round(time.monotonic()-started,2),'seconds',flush=True)


if __name__=='__main__':main()
