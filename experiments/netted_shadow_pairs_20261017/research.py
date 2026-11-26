"""Fixed sleeve blending; source inventories are signals, never account cash."""
import gzip,json,time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs as reference_inputs
from experiments.hedge_completion_pairs_20261017.engine import replay as native_replay
from experiments.complementary_alpha_20261017.research import protected
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,ts,save,sha
from .engine import replay,map_shadow

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'hedge_completion_pairs_20261017'


def load(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def main():
    cfg=json.loads((HERE/'protocol.json').read_text());out=HERE/'artifacts'
    out.mkdir(exist_ok=False);(out/'runs').mkdir()
    configs={venue:json.loads((PREVIOUS/f'{venue}_protocol.json').read_text()) for venue in ('reference','par')}
    source=ROOT/cfg['shadow_source'];shadows={}
    for phase in ('validation','comparison'):
        for arm in ('first_strategy','second_strategy',cfg['pair_policy']):
            for sc in configs['reference']['scenarios']:
                key=f'{phase}-{arm}-{sc}';shadows[key]=load(source/f'{key}.json.gz')
    paths=[*HERE.glob('*.py'),HERE/'protocol.json',PREVIOUS/'engine.py',
        PREVIOUS/'reference_protocol.json',PREVIOUS/'par_protocol.json',INPUT/'inputs.npz',
        ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',
        ROOT/'experiments/risk_preserving_20261016/artifacts/targets.npz',
        ROOT/'experiments/adaptive_mean_pairs_20261017/artifacts/state.npz',
        ROOT/'experiments/quote_anchored_pairs_20261017/pricing.py',
        *[source/f'{key}.json.gz' for key in shadows]]
    sources={str(path.relative_to(ROOT)):sha(path) for path in paths}
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),orders_sent=0))
    rd,_,p=reference_inputs(configs['reference'])
    with np.load(INPUT/'inputs.npz') as z:pd={k:z[k].copy() for k in z.files}
    with np.load(ROOT/'experiments/adaptive_mean_pairs_20261017/artifacts/state.npz') as z:
        de={k:z[k].copy() for k in z.files}
    data={'reference':rd,'par':pd};rows=[];runs={};actual_replays=0;source_reuses=0
    arms=['first_strategy','second_strategy',*cfg['controls'],*cfg['candidates']]

    def record(venue,phase,arm,sc):
        nonlocal actual_replays,source_reuses
        start,end=map(ts,cfg[f'{venue}_periods'][phase]);d=data[venue]
        if arm in ('first_strategy','second_strategy'):
            if venue=='reference':
                r=shadows[f'{phase}-{arm}-{sc}'];source_reuses+=1
                assert r['start']==start and r['end']==end
            else:
                r=native_replay(d,de,None,configs['par'],p,arm,sc,start,end);actual_replays+=1
        else:
            # Only map observations inside the account's declared period.
            mask=(d['times']>=start)&(d['times']<=end);sliced={k:v[mask] for k,v in d.items()}
            shadow_phase=phase if venue=='reference' else 'comparison'
            base_arm='second_strategy' if arm.startswith('second') else 'first_strategy'
            b,bt=map_shadow(sliced['times'],shadows[f'{shadow_phase}-{base_arm}-{sc}'])
            q,qt=map_shadow(sliced['times'],shadows[f'{shadow_phase}-{cfg["pair_policy"]}-{sc}'])
            share=1. if arm=='pair_shadow' else cfg['pair_share'] if arm in cfg['candidates'] else 0.
            c={**cfg,'scenarios':configs[venue]['scenarios']}
            r=replay(sliced,p,c,sc,start,end,b,q,share,np.maximum(bt,qt));actual_replays+=1
        label=f'{venue}-{phase}-{arm}-{sc}';runs[label]=r
        with gzip.open(out/'runs'/f'{label}.json.gz','wt') as f:
            json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        m=r['metrics'];rows.append(dict(venue=venue,phase=phase,arm=arm,scenario=sc,metrics=m))
        print(label,'PnL',round(m['net_pnl'],4),'Q',round(m['financial_score'],7),'fills',m['fills'],flush=True)

    for arm in arms:
        for sc in configs['reference']['scenarios']:record('reference','validation',arm,sc)
    ranking=[]
    for arm in cfg['candidates']:
        checks=[];diff=[]
        for sc in configs['reference']['scenarios']:
            m=runs[f'reference-validation-{arm}-{sc}']['metrics']
            checks.extend([m['net_pnl']>0,m['filled_decision_times']>=3,m['risk_breach_bars']==0])
            for baseline in arms[:2]:
                b=runs[f'reference-validation-{baseline}-{sc}']['metrics']
                diff.append(m['net_pnl']-b['net_pnl'])
                checks.extend([m['net_pnl']>b['net_pnl'],m['financial_score']>b['financial_score']])
        ranking.append(dict(arm=arm,eligible=all(checks),minimum_pnl_advantage=min(diff)))
    ranking.sort(key=lambda x:(-x['eligible'],-x['minimum_pnl_advantage'],x['arm']))
    selection=dict(selected=ranking[0]['arm'],eligible=ranking[0]['eligible'],ranking=ranking,
        frozen_before_later_reference_and_par_evaluation=True,historical_independence=False)
    save(out/'selection.json',selection);selection_hash=sha(out/'selection.json')
    for arm in arms:
        for sc in configs['reference']['scenarios']:record('reference','comparison',arm,sc)
    for phase in ('validation','comparison'):
        for arm in arms:
            for sc in configs['par']['scenarios']:record('par',phase,arm,sc)
    assert sha(out/'selection.json')==selection_hash
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path
    save(out/'results.json',dict(rows=rows,selection=selection,actual_replays=actual_replays,
        native_reference_source_reuses=source_reuses,protected=protected(),orders_sent=0,
        automatic_live_replacement=False,historical_independence=False,
        source_shadow_fills_are_not_actual_account_fills=True))
    print('DONE',actual_replays,'actual account replays and',source_reuses,'unchanged reference controls',flush=True)


if __name__=='__main__':main()
