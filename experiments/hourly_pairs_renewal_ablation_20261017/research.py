"""Isolate renewal with unchanged hourly forecasts and paid fixed-maturity exits."""
import gzip
import json
import time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.dense_history_pairs_20261017.portfolio import replay
from experiments.par_direct_forecast_20261017.data import ROOT, OUT as INPUT, save, sha, ts
from experiments.complementary_alpha_20261017.research import protected
from experiments.hourly_pairs_robustness_20261017.analysis import load
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'hourly_refresh_pairs_20261017'
OLD=HERE.parent/'adaptive_residual_pairs_20261017/artifacts_v2'


def read_npz(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}


def main():
    out=HERE/'artifacts';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    spec=json.loads((HERE/'protocol.json').read_text())
    cfgs={v:json.loads((PRIOR/f'{v}_protocol.json').read_text()) for v in spec['venues']}
    prior=json.loads((PRIOR/'artifacts/frozen.json').read_text())['sources']
    for rel,digest in prior.items():assert sha(ROOT/rel)==digest,rel
    paths=[HERE/'research.py',HERE/'protocol.json',PRIOR/'artifacts/frozen.json',PRIOR/'artifacts/results.json',
           PRIOR/'artifacts/selection.json',HERE.parent/'hourly_pairs_robustness_20261017/analysis.py',
           HERE.parent/'risk_neutral_pairs_20261017/attribution.py']
    for v,phase,sc in spec['controls']:
        paths.extend([OLD/f'{v}_bias_24_state.npz',OLD/'runs'/f'{v}-{phase}-first_dense_bias_24-{sc}.json.gz'])
    sources={**prior,**{str(p.relative_to(ROOT)):sha(p) for p in paths}}
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),
         historical_independence=False,orders_sent=0,previous_goal_turn='progress_hourly_bias24_robustness_verified'))
    reference,_,p=inputs(cfgs['reference']);data={'reference':reference,'par':read_npz(INPUT/'inputs.npz')}
    verified=[]
    for v,phase,sc in spec['controls']:
        cfg=cfgs[v];old=load(OLD/'runs'/f'{v}-{phase}-first_dense_bias_24-{sc}.json.gz')
        check=replay(data[v],read_npz(OLD/f'{v}_bias_24_state.npz'),None,cfg,p,'first_dense_bias_24',sc,old['start'],old['end'])
        for k in ['fills','snapshots','events','metrics']:assert check[k]==old[k],(v,phase,sc,k)
        verified.append(dict(venue=v,phase=phase,scenario=sc,exact=True))
    current=json.loads((PRIOR/'artifacts/results.json').read_text())
    lookup={(x['venue'],x['phase'],x['arm'],x['scenario']):x['metrics'] for x in current['rows']}
    rows=[];audits=[]
    for v in spec['venues']:
        cfg=cfgs[v];state=read_npz(PRIOR/'prepared'/f'{v}_bias_24_state.npz')
        for phase in spec['phases']:
            start=cfg['validation_start'] if phase=='validation' else cfg['comparison_start']
            end=cfg['comparison_start'] if phase=='validation' else cfg['end']
            for sc in spec['scenarios']:
                run=replay(data[v],state,None,cfg,p,'first_dense_bias_24',sc,ts(start),ts(end))
                run['arm']=spec['ablation'];run['ablation']='no renewal; fresh hourly predictions retained'
                assert not any(e['event']=='renewed' for e in run['events'])
                for e in run['events']:
                    if e['event']=='exit_requested' and e['reason']=='forecast_maturity':
                        assert e['maturity']<=e['time']+int(data[v]['times'][1]-data[v]['times'][0])
                key=f"{v}-{phase}-{spec['ablation']}-{sc}"
                with gzip.open(out/'runs'/f'{key}.json.gz','wt') as f:
                    json.dump(run,f,allow_nan=False,default=lambda z:z.item() if isinstance(z,np.generic) else z.tolist())
                m=run['metrics'];old=lookup[v,phase,spec['source_candidate'],sc]
                delta={k:m[k]-old[k] for k in ['net_pnl','financial_score','volatility_horizon','fees','execution_cost','funding','completed_pairs','aborted_pairs']}
                native={n:dict(pnl=m['net_pnl']-lookup[v,phase,n,sc]['net_pnl'],q=m['financial_score']-lookup[v,phase,n,sc]['financial_score']) for n in ['first_strategy','second_strategy']}
                rows.append(dict(venue=v,phase=phase,arm=spec['ablation'],scenario=sc,metrics=m,
                                 delta_from_renewal=delta,native_advantage=native))
                audit=episode_attribution(run,data[v],cfg['scenarios'][sc])
                audits.append(dict(venue=v,phase=phase,scenario=sc,**audit))
                print(key,'PnL',round(m['net_pnl'],3),'Q',round(m['financial_score'],7),
                      'delta renewal',round(delta['net_pnl'],3),'closed',m['completed_pairs'],flush=True)
    for rel,digest in sources.items():assert sha(ROOT/rel)==digest,rel
    save(out/'results.json',dict(rows=rows,control_verification=verified,new_account_replays=len(rows),
         control_replays=len(verified),protected=protected(),orders_sent=0,historical_independence=False,
         selection_unchanged=True))
    save(out/'attribution.json',dict(rows=audits,post_evaluation_only=True,orders_sent=0))
    print('DONE',len(rows),'new accounts;',len(verified),'exact control replays',flush=True)


if __name__=='__main__':main()
