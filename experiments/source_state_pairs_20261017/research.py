import gzip,json,time
import numpy as np
from strategy.agent import TAIL
from experiments.hedge_completion_pairs_20261017.engine import replay
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,sha,save,ts
from experiments.complementary_alpha_20261017.research import protected
from .audit import HERE,inputs
from .signals import reference_confirmation


def main():
    out=HERE/'artifacts';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    cfg,d,original,reference=inputs();revised=reference_confirmation(d,original,reference,cfg['entry_z'])
    p=json.loads((TAIL/'baseline.json').read_text())['protocol']
    paths=[*HERE.glob('*.py'),HERE/'protocol.json',INPUT/'inputs.npz',ROOT/cfg['reference_run']/'inputs.npz',
        ROOT/'experiments/hedge_completion_pairs_20261017/engine.py',ROOT/'experiments/hedge_completion_pairs_20261017/completion.py',
        ROOT/'experiments/adaptive_mean_pairs_20261017/artifacts/state.npz']
    sources={str(f.relative_to(ROOT)):sha(f) for f in paths}
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),orders_sent=0))
    np.savez_compressed(out/'reference_state.npz',**revised)
    arms=['first_strategy','second_strategy',*cfg['arms']];rows=[];runs={}
    def record(phase,arm,scenario,overlay=False):
        start=cfg['validation_start'] if phase=='validation' else cfg['comparison_start']
        end=cfg['comparison_start'] if phase=='validation' else cfg['end']
        de=revised if 'reference' in arm else original
        r=replay(d,de,None,cfg,p,arm,scenario,ts(start),ts(end),overlay=overlay)
        label='second_plus_'+arm if overlay else arm;key=f'{phase}-{label}-{scenario}'
        r.update(label=label,phase=phase);runs[key]=r
        if not overlay and 'reference' not in arm:
            old_arm='fixed_delayed_complete_168' if arm in cfg['arms'] else arm
            path=ROOT/f'experiments/hedge_completion_pairs_20261017/par_artifacts/runs/{phase}-{old_arm}-{scenario}.json.gz'
            with gzip.open(path,'rt') as f:old=json.load(f)
            assert old['fills']==r['fills'] and old['snapshots']==r['snapshots']
        with gzip.open(out/'runs'/f'{key}.json.gz','wt') as f:
            json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        rows.append(dict(phase=phase,arm=label,scenario=scenario,metrics=r['metrics']))
        m=r['metrics'];print(key,'PnL',round(m['net_pnl'],4),'Q',round(m['financial_score'],7),'pairs',m['completed_pairs'],flush=True)
    for arm in arms:
        for scenario in cfg['scenarios']:record('validation',arm,scenario)
    ranking=[]
    for arm in cfg['arms']:
        checks=[];diff=[]
        for sc in cfg['scenarios']:
            a=runs[f'validation-{arm}-{sc}']['metrics'];checks.extend([a['net_pnl']>0,a['completed_pairs']>=3,a['risk_breach_bars']==0])
            for bn in arms[:2]:
                b=runs[f'validation-{bn}-{sc}']['metrics'];diff.append(a['net_pnl']-b['net_pnl'])
                checks.extend([a['net_pnl']>b['net_pnl'],a['financial_score']>b['financial_score']])
        ranking.append(dict(arm=arm,eligible=all(checks),minimum_pnl_advantage=min(diff)))
    ranking.sort(key=lambda r:(-r['eligible'],-r['minimum_pnl_advantage'],r['arm']))
    selected=ranking[0]['arm'];selection=dict(selected=selected,eligible=ranking[0]['eligible'],ranking=ranking,frozen_before_comparison_replay=True)
    save(out/'selection.json',selection);sh=sha(out/'selection.json')
    for arm in arms:
        for scenario in cfg['scenarios']:record('comparison',arm,scenario)
    for phase in ('validation','comparison'):
        for scenario in cfg['scenarios']:record(phase,selected,scenario,True)
    assert sha(out/'selection.json')==sh
    for path,h in sources.items():assert sha(ROOT/path)==h,path
    protected()
    save(out/'results.json',dict(rows=rows,selection=selection,historical_replays=len(rows),historical_independence=False,
        original_source_and_baselines_reproduced_exactly=True,orders_sent=0,automatic_live_replacement=False))
    print('DONE',len(rows),'replays; eligible',selection['eligible'],flush=True)


if __name__=='__main__':main()
