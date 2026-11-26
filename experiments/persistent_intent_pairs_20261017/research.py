import gzip,json,time
from pathlib import Path
import numpy as np
from strategy.agent import TAIL
from experiments.cost_horizon_20261016.research import data as reference_data
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs as reference_inputs
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,ts,save,sha
from experiments.complementary_alpha_20261017.research import protected
from .signals import intent_state
from .engine import replay

HERE=Path(__file__).resolve().parent;PREVIOUS=HERE.parent/'hedge_completion_pairs_20261017'


def main():
    out=HERE/'artifacts';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    cfgs={v:json.loads((HERE/f'{v}_protocol.json').read_text()) for v in ('reference','par')}
    paths=[*HERE.glob('*.py'),*HERE.glob('*protocol.json'),INPUT/'inputs.npz',
        ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',
        ROOT/'experiments/risk_preserving_20261016/artifacts/targets.npz',
        ROOT/cfgs['par']['reference_run']/'inputs.npz',PREVIOUS/'engine.py',PREVIOUS/'completion.py',
        ROOT/'experiments/quote_anchored_pairs_20261017/pricing.py']
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),historical_independence=False,orders_sent=0))
    rd,_,p=reference_inputs(cfgs['reference']);ref,_,_,_=reference_data()
    with np.load(INPUT/'inputs.npz') as z:pd={k:z[k].copy() for k in z.files}
    with np.load(ROOT/cfgs['par']['reference_run']/'inputs.npz') as z:par_ref={k:z[k].copy() for k in z.files}
    data={'reference':rd,'par':pd};references={'reference':ref,'par':par_ref};states={}
    for venue in data:
        for kind,hours in cfgs[venue]['signal_lifetime_hours'].items():
            state=intent_state(data[venue],references[venue],cfgs[venue],p['symbols'],hours)
            states[(venue,kind)]=state
            np.savez_compressed(out/f'{venue}_{kind}_state.npz',**state)
    runs={};rows=[];reused=0
    arms=['first_strategy','second_strategy',*cfgs['reference']['arms']]
    def record(venue,phase,arm,scenario):
        cfg=cfgs[venue];start=cfg['validation_start'] if phase=='validation' else cfg['comparison_start']
        end=cfg['comparison_start'] if phase=='validation' else cfg['end']
        kind='persistent' if 'persistent' in arm else 'immediate'
        r=replay(data[venue],states[(venue,kind)],None,cfg,p,arm,scenario,ts(start),ts(end))
        if arm in arms[:2]:
            with gzip.open(PREVIOUS/f'{venue}_artifacts/runs/{phase}-{arm}-{scenario}.json.gz','rt') as f:old=json.load(f)
            assert old['fills']==r['fills'] and old['snapshots']==r['snapshots']
        label=f'{venue}-{phase}-{arm}-{scenario}';runs[label]=r
        with gzip.open(out/'runs'/f'{label}.json.gz','wt') as f:
            json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        m=r['metrics'];rows.append(dict(venue=venue,phase=phase,arm=arm,scenario=scenario,metrics=m))
        print(label,'PnL',round(m['net_pnl'],4),'Q',round(m['financial_score'],7),'pairs',m['completed_pairs'],'fills',m['fills'],flush=True)
    for arm in arms:
        for sc in cfgs['reference']['scenarios']:record('reference','validation',arm,sc)
    ranking=[]
    for arm in cfgs['reference']['arms']:
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
    for venue in cfgs:
        for phase in ('validation','comparison'):
            for sc in cfgs[venue]['scenarios']:
                arm='fixed_delayed_complete_168'
                with gzip.open(PREVIOUS/f'{venue}_artifacts/runs/{phase}-{arm}-{sc}.json.gz','rt') as f:r=json.load(f)
                rows.append(dict(venue=venue,phase=phase,arm='previous_fixed_delayed_complete_168',scenario=sc,metrics=r['metrics']));reused+=1
    assert sha(out/'selection.json')==selection_hash
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path
    save(out/'results.json',dict(rows=rows,selection=selection,actual_replays=len(runs),previous_pair_source_reuses=reused,
        native_baselines_reproduced_exactly=True,protected=protected(),orders_sent=0,automatic_live_replacement=False,historical_independence=False))
    print('DONE',len(runs),'account replays;',reused,'previous pair controls; eligible',selection['eligible'],flush=True)


if __name__=='__main__':main()
