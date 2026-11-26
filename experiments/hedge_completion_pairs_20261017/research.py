import argparse,gzip,json,time
from pathlib import Path
import numpy as np
from strategy.agent import TAIL
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs as reference_inputs
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,ts,save,sha
from experiments.complementary_alpha_20261017.research import protected
from .engine import replay

HERE=Path(__file__).resolve().parent;PREVIOUS=HERE.parent/'adaptive_mean_pairs_20261017'


def main(venue):
    out=HERE/f'{venue}_artifacts';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    cfg=json.loads((HERE/f'{venue}_protocol.json').read_text())
    if venue=='reference':d,de,p=reference_inputs(cfg)
    else:
        with np.load(INPUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
        with np.load(PREVIOUS/'artifacts/state.npz') as z:de={k:z[k].copy() for k in z.files}
        p=json.loads((TAIL/'baseline.json').read_text())['protocol']
    paths=[*HERE.glob('*.py'),HERE/f'{venue}_protocol.json',PREVIOUS/'signals.py',
        ROOT/'experiments/quote_anchored_pairs_20261017/pricing.py',INPUT/'inputs.npz',
        ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',
        ROOT/'experiments/risk_preserving_20261016/artifacts/targets.npz',PREVIOUS/'artifacts/state.npz']
    sources={str(f.relative_to(ROOT)):sha(f) for f in paths}
    save(out/'frozen.json',dict(venue=venue,at_unix=time.time(),sources=sources,protected=protected(),orders_sent=0,historical_independence=False))
    arms=['first_strategy','second_strategy',*cfg['arms']];rows=[];runs={}
    def record(phase,arm,scenario,overlay=False):
        start=cfg['validation_start'] if phase=='validation' else cfg['comparison_start']
        end=cfg['comparison_start'] if phase=='validation' else cfg['end']
        r=replay(d,de,None,cfg,p,arm,scenario,ts(start),ts(end),overlay=overlay)
        label='second_plus_'+arm if overlay else arm;key=f'{phase}-{label}-{scenario}'
        r.update(label=label,phase=phase,venue=venue);runs[key]=r
        if not overlay and 'complete' not in arm:
            prior_arm='fixed_delayed_168' if arm=='fixed_delayed_abort_168' else arm
            with gzip.open(PREVIOUS/f'economic_{venue}_artifacts/runs/{phase}-{prior_arm}-{scenario}.json.gz','rt') as f:old=json.load(f)
            assert old['fills']==r['fills'] and old['snapshots']==r['snapshots'],key
        m=r['metrics'];m.update(completion_attempts=sum(e['event']=='completion_intent' for e in r['events']),
            completion_balanced=sum(e['event']=='completion_balanced' for e in r['events']),
            completion_trim=sum(e['event']=='completion_trim_requested' for e in r['events']))
        with gzip.open(out/'runs'/f'{key}.json.gz','wt') as f:
            json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        rows.append(dict(phase=phase,arm=label,scenario=scenario,metrics=m))
        print(venue,key,'PnL',round(m['net_pnl'],3),'Q',round(m['financial_score'],6),
            'completed/attempted',m['completion_balanced'],m['completion_attempts'],'pairs',m['completed_pairs'],flush=True)
    for arm in arms:
        for scenario in cfg['scenarios']:record('validation',arm,scenario)
    ranks=[]
    for arm in cfg['arms']:
        checks=[];diff=[]
        for sc in cfg['scenarios']:
            a=runs[f'validation-{arm}-{sc}']['metrics']
            checks.extend([a['net_pnl']>0,a['completed_pairs']>=3,a['risk_breach_bars']==0])
            for bn in arms[:2]:
                b=runs[f'validation-{bn}-{sc}']['metrics'];diff.append(a['net_pnl']-b['net_pnl'])
                checks.extend([a['net_pnl']>b['net_pnl'],a['financial_score']>b['financial_score']])
        ranks.append(dict(arm=arm,eligible=all(checks),minimum_pnl_advantage=min(diff)))
    ranks.sort(key=lambda r:(-r['eligible'],-r['minimum_pnl_advantage'],r['arm']))
    selected=ranks[0]['arm'];selection=dict(selected=selected,eligible=ranks[0]['eligible'],ranking=ranks,
        frozen_before_comparison_replay=True,historical_independence=False)
    save(out/'selection.json',selection);selection_hash=sha(out/'selection.json')
    for arm in arms:
        for scenario in cfg['scenarios']:record('comparison',arm,scenario)
    for phase in ('validation','comparison'):
        for scenario in cfg['scenarios']:record(phase,selected,scenario,True)
    assert sha(out/'selection.json')==selection_hash
    for path,h in sources.items():assert sha(ROOT/path)==h,path
    guards=protected()
    save(out/'results.json',dict(rows=rows,selection=selection,venue=venue,reference_is_not_par=venue=='reference',
        historical_replays=len(rows),baseline_and_abort_reproduced_exactly=True,protected=guards,
        orders_sent=0,automatic_live_replacement=False,historical_independence=False))
    print('DONE',venue,len(rows),'replays; eligible',selection['eligible'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--venue',choices=['reference','par'],required=True)
    main(parser.parse_args().venue)
