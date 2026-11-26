import gzip
import json
import time
from pathlib import Path
import numpy as np
from strategy.agent import TAIL
from experiments.complementary_alpha_20261017.research import protected
from experiments.q_execution_20261017.metrics import paired_uncertainty
from experiments.par_direct_forecast_20261017.data import ts,save,sha,ROOT,OUT as INPUT
from .model import fit_predict, labels
from .engine import replay

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'


def main():
    OUT.mkdir(exist_ok=False)
    cfg=json.loads((HERE/'protocol.json').read_text());p=json.loads((TAIL/'baseline.json').read_text())['protocol']
    with np.load(INPUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
    dependencies=['replay_lab/common_comparison/engine.py','replay_lab/pullback/broker.py',
        'replay_lab/ema_rebuild/engine.py','experiments/cost_horizon_20261016/research.py',
        'experiments/basis_carry_20261017/score.py','experiments/basis_carry_20261017/probe.py']
    paths=[*HERE.glob('*.py'),HERE/'protocol.json',INPUT/'inputs.npz',*[ROOT/x for x in dependencies]]
    sources={str(q.relative_to(ROOT)):sha(q) for q in paths}
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),historical_independence=False,orders_sent=0))
    de,pred,fits=fit_predict(d,cfg,p['symbols'],OUT/'models')
    np.savez_compressed(OUT/'predictions.npz',**pred,beta=de['beta'],correlation=de['correlation'])
    rows=[];runs={};(OUT/'runs').mkdir()
    arms=['first_strategy','second_strategy',*pred]
    def record(phase,arm,scenario,start,end,overlay=False):
        r=replay(d,de,pred.get(arm),cfg,p,arm,scenario,ts(start),ts(end),overlay=overlay)
        label=('second_plus_'+arm) if overlay else arm;r.update(label=label,phase=phase)
        key=f'{phase}-{label}-{scenario}';runs[key]=r
        with gzip.open(OUT/'runs'/f'{key}.json.gz','wt') as f:
            json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        rows.append(dict(phase=phase,arm=label,scenario=scenario,metrics=r['metrics']))
        m=r['metrics'];print(key,'PnL',round(m['net_pnl'],4),'Q',round(m['financial_score'],8),'pairs',m['completed_pairs'],'aborts',m['aborted_pairs'],flush=True)
    for arm in arms:
        for scenario in cfg['scenarios']:record('validation',arm,scenario,cfg['validation_start'],cfg['comparison_start'])
    ranking=[]
    for arm in pred:
        checks=[];advantages=[]
        for scenario in cfg['scenarios']:
            a=runs[f'validation-{arm}-{scenario}']['metrics']
            checks.extend([a['net_pnl']>0,a['completed_pairs']>=3,a['risk_breach_bars']==0])
            for bn in arms[:2]:
                b=runs[f'validation-{bn}-{scenario}']['metrics']
                checks.extend([a['net_pnl']>b['net_pnl'],a['financial_score']>b['financial_score']])
                advantages.append(a['net_pnl']-b['net_pnl'])
        a=runs[f'validation-{arm}-base']['metrics']
        ranking.append(dict(arm=arm,eligible=all(checks),Q=a['financial_score'],PnL=a['net_pnl'],minimum_pnl_advantage=min(advantages)))
    eligible=[r for r in ranking if r['eligible']]
    ranking.sort(key=lambda r:(-r['eligible'],-(r['minimum_pnl_advantage'] if eligible else r['Q']),r['arm']))
    selected=ranking[0]['arm'];selection=dict(selected=selected,eligible=ranking[0]['eligible'],ranking=ranking,
        frozen_before_comparison_replay=True,historical_independence=False)
    save(OUT/'selection.json',selection);selection_hash=sha(OUT/'selection.json')
    for arm in arms:
        for scenario in cfg['scenarios']:record('comparison',arm,scenario,cfg['comparison_start'],cfg['end'])
    for phase,start,end in [('validation',cfg['validation_start'],cfg['comparison_start']),('comparison',cfg['comparison_start'],cfg['end'])]:
        for scenario in cfg['scenarios']:record(phase,selected,scenario,start,end,overlay=True)
    comparisons=[]
    for scenario in cfg['scenarios']:
        for arm in (selected,'second_plus_'+selected):
            c=runs[f'comparison-{arm}-{scenario}']
            for bn in arms[:2]:
                b=runs[f'comparison-{bn}-{scenario}'];cm,bm=c['metrics'],b['metrics']
                comparisons.append(dict(scenario=scenario,candidate=arm,baseline=bn,pnl_difference=cm['net_pnl']-bm['net_pnl'],
                    Q_difference=cm['financial_score']-bm['financial_score'],lower_volatility=cm['volatility_horizon']<=bm['volatility_horizon'],
                    candidate_completed_pairs=cm['completed_pairs'],candidate_positive=cm['net_pnl']>0,
                    mean_gross_ratio=cm['mean_gross']/bm['mean_gross'] if bm['mean_gross'] else None,
                    uncertainty=paired_uncertainty(c,b)))
    diagnostics=[]
    for name,pr in pred.items():
        h=int(name.rsplit('_',1)[1]);y,mat=labels(d,de,h)
        mask=(d['times']>=ts(cfg['comparison_start']))&(mat<=ts(cfg['end']))&((d['times']-ts(cfg['comparison_start']))%(h*3600)==0)
        for k,pair in enumerate(cfg['pairs']):
            use=mask&np.isfinite(pr[:,k])&np.isfinite(y[:,k]);a=y[use,k]/10000;f=pr[use,k]
            if len(a):diagnostics.append(dict(model=name,pair=pair,blocks=len(a),mse=float(np.mean((a-f)**2)),zero_mse=float(np.mean(a*a)),direction_accuracy=float(np.mean(np.sign(a)==np.sign(f)))))
    assert sha(OUT/'selection.json')==selection_hash
    for path,h in sources.items():assert sha(ROOT/path)==h,path
    protected()
    save(OUT/'results.json',dict(rows=rows,selection=selection,comparisons=comparisons,forecast_diagnostics=diagnostics,
        model_fits=len(fits),historical_replays=len(rows),historical_independence=False,protected_sources_unchanged=True,orders_sent=0,
        automatic_replacement=False,independent_prospective_evidence=False))
    print('DONE',len(rows),'replays; selected',selected,'eligible',selection['eligible'],flush=True)


if __name__=='__main__':main()
