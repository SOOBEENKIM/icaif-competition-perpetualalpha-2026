import copy,gzip,json,time
import numpy as np
from strategy.agent import TAIL
from experiments.complementary_alpha_20261017.research import protected
from experiments.q_execution_20261017.metrics import paired_uncertainty
from .data import HERE,OUT,ROOT,ts,save,sha
from .model import fit_predict,labels
from .engine import replay

def main():
    if (OUT/'frozen.json').exists():raise FileExistsError('Family already evaluated')
    cfg=json.loads((HERE/'protocol.json').read_text());p=json.loads((TAIL/'baseline.json').read_text())['protocol']
    with np.load(OUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
    sources={str(q.relative_to(ROOT)):sha(q) for q in [*HERE.glob('*.py'),HERE/'protocol.json',OUT/'inputs.npz']}
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),historical_independence=False,orders_sent=0))
    de,pred,fits=fit_predict(d,cfg,p['symbols'],OUT/'models')
    np.savez_compressed(OUT/'predictions.npz',**pred,beta=de['beta'],correlation=de['correlation'],fresh=de['fresh'])
    rows=[];runs={};(OUT/'runs').mkdir(exist_ok=True)
    arms=['first_strategy','second_strategy',*pred]
    def record(phase,arm,scenario,start,end,overlay=False):
        r=replay(d,de,pred.get(arm),cfg,p,arm,scenario,ts(start),ts(end),overlay=overlay)
        label=('blend_second_'+arm) if overlay else arm;r['label']=label;r['phase']=phase
        key=f'{phase}-{label}-{scenario}';runs[key]=r
        with gzip.open(OUT/'runs'/f'{key}.json.gz','wt') as f:json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        rows.append(dict(phase=phase,arm=label,scenario=scenario,metrics=r['metrics']))
        m=r['metrics'];print(key,'PnL',round(m['net_pnl'],4),'Q',round(m['financial_score'],8),'pairs',m['completed_pairs'],'aborts',m['aborted_pairs'],flush=True)
    for arm in arms:
        for scenario in cfg['scenarios']:record('validation',arm,scenario,cfg['validation_start'],cfg['comparison_start'])
    ranking=[]
    for arm in pred:
        a=runs[f'validation-{arm}-base']['metrics'];b=runs[f'validation-{arm}-stress']['metrics']
        base=[runs[f'validation-{n}-base']['metrics'] for n in arms[:2]]
        eligible=bool(a['net_pnl']>0 and b['net_pnl']>0 and a['completed_pairs']>=3 and not a['risk_breach_bars'] and not b['risk_breach_bars']
            and all(a['net_pnl']>q['net_pnl'] and a['financial_score']>q['financial_score'] for q in base))
        ranking.append(dict(arm=arm,eligible=eligible,Q=a['financial_score'],PnL=a['net_pnl']))
    ranking.sort(key=lambda r:(-r['eligible'],-r['Q'],r['arm']))
    selected=ranking[0]['arm'];selection=dict(selected=selected,eligible=ranking[0]['eligible'],ranking=ranking,
        frozen_before_comparison_replay=True,historical_independence=False)
    save(OUT/'selection.json',selection);selected_hash=sha(OUT/'selection.json')
    for arm in arms:
        for scenario in cfg['scenarios']:record('comparison',arm,scenario,cfg['comparison_start'],cfg['end'])
    for phase,start,end in [('validation',cfg['validation_start'],cfg['comparison_start']),('comparison',cfg['comparison_start'],cfg['end'])]:
        for scenario in cfg['scenarios']:record(phase,selected,scenario,start,end,overlay=True)
    comparisons=[]
    for scenario in cfg['scenarios']:
        for arm in (selected,'blend_second_'+selected):
            c=runs[f'comparison-{arm}-{scenario}']
            for bn in arms[:2]:
                b=runs[f'comparison-{bn}-{scenario}'];cm,bm=c['metrics'],b['metrics']
                comparisons.append(dict(scenario=scenario,candidate=arm,baseline=bn,
                    pnl_difference=cm['net_pnl']-bm['net_pnl'],Q_difference=cm['financial_score']-bm['financial_score'],
                    lower_volatility=cm['volatility_horizon']<=bm['volatility_horizon'],
                    candidate_completed_pairs=cm['completed_pairs'],candidate_positive=cm['net_pnl']>0,
                    mean_gross_ratio=cm['mean_gross']/bm['mean_gross'] if bm['mean_gross'] else None,
                    uncertainty=paired_uncertainty(c,b)))
    forecasts=[]
    for name,pr in pred.items():
        h=int(name.split('_')[-1]);y,mat=labels(d,de,h)
        # Nonoverlapping decision-time blocks and per-pair reporting.
        mask=((d['times']>=ts(cfg['comparison_start']))&(mat<=ts(cfg['end']))&((d['times']-ts(cfg['comparison_start']))%(h*3600)==0))[:,None]
        for k,pair in enumerate(cfg['pairs']):
            use=mask[:,0]&np.isfinite(pr[:,k])&np.isfinite(y[:,k]);a=y[use,k]/10000;f=pr[use,k]
            if len(a):forecasts.append(dict(model=name,pair=pair,blocks=len(a),mse=float(np.mean((a-f)**2)),zero_mse=float(np.mean(a*a)),direction_accuracy=float(np.mean(np.sign(a)==np.sign(f)))))
    assert sha(OUT/'selection.json')==selected_hash
    for path,h in sources.items():assert sha(ROOT/path)==h,path
    protected()
    save(OUT/'results.json',dict(rows=rows,selection=selection,comparisons=comparisons,forecast_diagnostics=forecasts,
        model_fits=len(fits),historical_replays=len(rows),historical_independence=False,protected_sources_unchanged=True,orders_sent=0,
        automatic_replacement=False,independent_prospective_evidence=False))
    print('DONE selected',selected,'eligible',selection['eligible'],flush=True)

if __name__=='__main__':main()
