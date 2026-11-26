import difflib,gzip,json,time
import numpy as np
from strategy.agent import TAIL
from experiments.complementary_alpha_20261017.research import protected
from .data import HERE,OUT,ROOT,ts,save,sha
from .model import design
from .basis_model import predict
from .basis_engine import replay

def main():
    dest=OUT/'basis';dest.mkdir(exist_ok=True)
    if (dest/'frozen.json').exists():raise FileExistsError('Basis family already evaluated')
    cfg=json.loads((HERE/'protocol.json').read_text());p=json.loads((TAIL/'baseline.json').read_text())['protocol']
    save(dest/'protocol.json',dict(inherited=cfg,models=['basis_ar_6','basis_ar_24','basis_blend_6','basis_blend_24'],
        changes='AR1 on per-asset par/reference log basis, last 7 days of consecutive active observations only, phi clipped 0..0.999. Forecast remaining convergence after a 5-minute arrival. Equal blend with the already fitted tree, never refit from later results. Entry IOC price cannot exceed source-price adverse 10bp; same gate for both incumbents.',
        selection='Validation-positive base and stress, at least 3 completed pairs, higher base PnL and Q than both incumbents. Freeze before comparison. If none, label diagnostic only.',
        prior_results_seen=True,historical_independence=False,orders_sent=0))
    (dest/'engine.diff').write_text(''.join(difflib.unified_diff((HERE/'engine.py').read_text().splitlines(True),(HERE/'basis_engine.py').read_text().splitlines(True))))
    sources={str(q.relative_to(ROOT)):sha(q) for q in [HERE/'basis_model.py',HERE/'basis_engine.py',HERE/'basis_research.py',dest/'protocol.json',OUT/'inputs.npz',OUT/'predictions.npz']}
    save(dest/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected()))
    with np.load(OUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
    with np.load(OUT/'predictions.npz') as z:old={k:z[k].copy() for k in z.files}
    de=design(d,cfg,p['symbols']);pred,records=predict(d,de,cfg)
    for h in cfg['horizons_hours']:pred[f'basis_blend_{h}']=.5*(pred[f'basis_ar_{h}']+old[f'tree_{h}'])
    save(dest/'models.json',records);np.savez_compressed(dest/'predictions.npz',**pred)
    rows=[];runs={};(dest/'runs').mkdir(exist_ok=True)
    def one(phase,arm,scenario,start,end,overlay=False):
        r=replay(d,de,pred.get(arm),cfg,p,arm,scenario,ts(start),ts(end),overlay=overlay);label=('second_plus_'+arm) if overlay else arm
        key=f'{phase}-{label}-{scenario}';runs[key]=r
        with gzip.open(dest/'runs'/f'{key}.json.gz','wt') as f:json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        rows.append(dict(phase=phase,arm=label,scenario=scenario,metrics=r['metrics']))
        m=r['metrics'];print(key,'PnL',round(m['net_pnl'],4),'Q',round(m['financial_score'],8),'pairs',m['completed_pairs'],'aborts',m['aborted_pairs'],flush=True)
    arms=['first_strategy','second_strategy',*pred]
    for arm in arms:
        for sc in cfg['scenarios']:one('validation',arm,sc,cfg['validation_start'],cfg['comparison_start'])
    ranking=[]
    for arm in pred:
        a=runs[f'validation-{arm}-base']['metrics'];b=runs[f'validation-{arm}-stress']['metrics']
        bases=[runs[f'validation-{n}-base']['metrics'] for n in arms[:2]]
        eligible=bool(a['net_pnl']>0 and b['net_pnl']>0 and a['completed_pairs']>=3 and not a['risk_breach_bars'] and not b['risk_breach_bars'] and all(a['net_pnl']>q['net_pnl'] and a['financial_score']>q['financial_score'] for q in bases))
        ranking.append(dict(arm=arm,eligible=eligible,Q=a['financial_score']))
    ranking.sort(key=lambda r:(-r['eligible'],-r['Q'],r['arm']));selected=ranking[0]['arm']
    save(dest/'selection.json',dict(ranking=ranking,selected=selected,eligible=ranking[0]['eligible'],frozen_before_comparison=True));frozen_selection=sha(dest/'selection.json')
    for arm in arms:
        for sc in cfg['scenarios']:one('comparison',arm,sc,cfg['comparison_start'],cfg['end'])
    for phase,start,end in [('validation',cfg['validation_start'],cfg['comparison_start']),('comparison',cfg['comparison_start'],cfg['end'])]:
        for sc in cfg['scenarios']:one(phase,selected,sc,start,end,True)
    assert sha(dest/'selection.json')==frozen_selection
    for path,h in sources.items():assert sha(ROOT/path)==h,path
    protected()
    save(dest/'results.json',dict(rows=rows,selection=json.loads((dest/'selection.json').read_text()),orders_sent=0,
        historical_independence=False,protected_sources_unchanged=True,automatic_replacement=False))

if __name__=='__main__':main()
