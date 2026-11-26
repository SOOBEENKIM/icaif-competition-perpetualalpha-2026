import gzip,json,time
from pathlib import Path
import numpy as np
from strategy.agent import TAIL
from experiments.cost_horizon_20261016.research import data as reference_data
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs as reference_inputs
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,ts,save,sha
from experiments.complementary_alpha_20261017.research import protected
from .portfolio import replay

HERE=Path(__file__).resolve().parent;PREVIOUS=HERE.parent/'persistent_intent_pairs_20261017'


def main():
    out=HERE/'artifacts';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    cfgs={v:json.loads((HERE/f'{v}_protocol.json').read_text()) for v in ('reference','par')}
    paths=[*HERE.glob('*.py'),*HERE.glob('*protocol.json'),INPUT/'inputs.npz',
        ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',
        ROOT/'experiments/risk_preserving_20261016/artifacts/targets.npz',
        ROOT/cfgs['par']['reference_run']/'inputs.npz',PREVIOUS/'engine.py',PREVIOUS/'artifacts/reference_immediate_state.npz',PREVIOUS/'artifacts/par_immediate_state.npz',
        ROOT/'experiments/quote_anchored_pairs_20261017/pricing.py']
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),historical_independence=False,orders_sent=0))
    rd,_,p=reference_inputs(cfgs['reference'])
    with np.load(INPUT/'inputs.npz') as z:pd={k:z[k].copy() for k in z.files}
    data={'reference':rd,'par':pd};states={}
    for venue in data:
        with np.load(ROOT/f'experiments/persistent_intent_pairs_20261017/artifacts/{venue}_immediate_state.npz') as z:
            states[venue]={k:z[k].copy() for k in z.files}
    runs={};rows=[];reused=0
    arms=[*cfgs['reference']['controls'],*cfgs['reference']['candidates']]
    def record(venue,phase,arm,scenario):
        cfg=cfgs[venue];start=cfg['validation_start'] if phase=='validation' else cfg['comparison_start']
        end=cfg['comparison_start'] if phase=='validation' else cfg['end']
        r=replay(data[venue],states[venue],None,cfg,p,arm,scenario,ts(start),ts(end))
        if arm in cfg['controls']:
            old_arm='fixed_delayed_immediate_complete_168' if arm=='pairs_alone' else arm
            with gzip.open(PREVIOUS/f'artifacts/runs/{venue}-{phase}-{old_arm}-{scenario}.json.gz','rt') as f:old=json.load(f)
            assert old['fills']==r['fills'],(venue,phase,arm,scenario,'fills')
            clean=[{k:v for k,v in s.items() if k not in ('base_qty','pair_qty')} for s in r['snapshots']]
            assert old['snapshots']==clean,(venue,phase,arm,scenario,'snapshots')
        label=f'{venue}-{phase}-{arm}-{scenario}';runs[label]=r
        with gzip.open(out/'runs'/f'{label}.json.gz','wt') as f:
            json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        m=r['metrics'];rows.append(dict(venue=venue,phase=phase,arm=arm,scenario=scenario,metrics=m))
        print(label,'PnL',round(m['net_pnl'],4),'Q',round(m['financial_score'],7),'pairs',m['completed_pairs'],'fills',m['fills'],flush=True)
    for arm in arms:
        for sc in cfgs['reference']['scenarios']:record('reference','validation',arm,sc)
    ranking=[]
    for arm in cfgs['reference']['candidates']:
        checks=[];diff=[]
        for sc in cfgs['reference']['scenarios']:
            a=runs[f'reference-validation-{arm}-{sc}']['metrics']
            checks.extend([a['net_pnl']>0,a['completed_pairs']>=3,a['risk_breach_bars']==0])
            for base in arms[:2]:
                b=runs[f'reference-validation-{base}-{sc}']['metrics'];diff.append(a['net_pnl']-b['net_pnl'])
                checks.extend([a['net_pnl']>b['net_pnl'],a['financial_score']>b['financial_score']])
        ranking.append(dict(arm=arm,eligible=all(checks),minimum_pnl_advantage=min(diff)))
    ranking.sort(key=lambda r:(-r['eligible'],-r['minimum_pnl_advantage'],r['arm']))
    selection=dict(selected=ranking[0]['arm'],eligible=ranking[0]['eligible'],ranking=ranking,
        frozen_before_later_reference_and_par=True,historical_independence=False)
    save(out/'selection.json',selection);selection_hash=sha(out/'selection.json')
    for arm in arms:
        for sc in cfgs['reference']['scenarios']:record('reference','comparison',arm,sc)
    for phase in ('validation','comparison'):
        for arm in arms:
            for sc in cfgs['par']['scenarios']:record('par',phase,arm,sc)
    assert sha(out/'selection.json')==selection_hash
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path
    save(out/'results.json',dict(rows=rows,selection=selection,actual_replays=len(runs),
        native_baselines_reproduced_exactly=True,protected=protected(),orders_sent=0,automatic_live_replacement=False,historical_independence=False))
    print('DONE',len(runs),'account replays; eligible',selection['eligible'],flush=True)


if __name__=='__main__':main()
