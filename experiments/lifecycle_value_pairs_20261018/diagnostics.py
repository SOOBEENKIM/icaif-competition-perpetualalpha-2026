"""Post-evaluation cash attribution and as-issued episodic prediction checks."""
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from .research_v2 import ROOT,HERE,OUT,inputs,readrun,save,sha,verify,protected
from .model import predict
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution


def main():
    result=json.loads((OUT/'results.json').read_text())
    labels=json.loads((OUT/'labels.json').read_text())
    models={int(k):v for k,v in json.loads((OUT/'models.json').read_text()).items()}
    cfg=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/reference_protocol.json').read_text())
    d,_,_=inputs(cfg)
    files=[Path(__file__),OUT/'results.json',OUT/'labels.json',OUT/'models.json',
           ROOT/'experiments/risk_neutral_pairs_20261017/attribution.py',*sorted((OUT/'runs').glob('reference-*.json.gz'))]
    sources={str(path.relative_to(ROOT)):sha(path) for path in files}
    verify(json.loads((OUT/'evaluation_frozen_v2.json').read_text())['sources']);verify(sources)
    save(OUT/'diagnostics_frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected()))
    predictions=[]
    for row in labels['rows']:
        if row['status'] not in ('closed','unfilled'):continue
        model=models.get(row['source']//86400*86400)
        if model is None:continue
        assert model['last_outcome']<model['fit_time']<=row['source']<row['outcome_time']
        value=predict(model,np.asarray(row['features']))*row['unit']
        actual=row['net_pnl']/row['intended_gross']
        predictions.append(dict(sample_id=row['sample_id'],source=row['source'],outcome_time=row['outcome_time'],
                                pair_index=row['pair_index'],prediction=value,actual=actual,fit_time=model['fit_time']))
    rows=[];forecasts=[]
    for phase in ['validation','comparison']:
        for sc in ['base','stress']:
            for kind in ['filter','rank']:
                arm=f'first_dense_hourlyhedge_lifecycle_{kind}_24'
                run=readrun(OUT/'runs'/f'reference-{phase}-{arm}-{sc}.json.gz')
                attribution=episode_attribution(run,d,cfg['scenarios'][sc]);totals=defaultdict(lambda:defaultdict(float))
                for e in attribution['episodes']:
                    key='/'.join(e['pair'])
                    for metric in ['price_move','execution_cost','fees','funding','net_pnl','fills']:totals[key][metric]+=e[metric]
                    totals[key]['closed']+=int(e['closed']);totals[key]['episodes']+=1
                assert abs(sum(v['net_pnl'] for v in totals.values())-run['metrics']['pair_owned_pnl'])<1e-6
                rows.append(dict(phase=phase,scenario=sc,kind=kind,metrics=run['metrics'],
                                 by_pair={k:dict(v) for k,v in totals.items()},episodes=attribution['episodes'],
                                 risk_breaches=run['risk_breaches'],
                                 entries=[e for e in run['events'] if e['event']=='pair_intent']))
                print('ATTRIBUTION',phase,sc,kind,{k:round(v['net_pnl'],2) for k,v in totals.items()},flush=True)
        start=run['start'];end=run['end']
        issued=[r for r in predictions if start<=r['source']<end and r['outcome_time']<=end]
        pred=np.asarray([r['prediction'] for r in issued]);actual=np.asarray([r['actual'] for r in issued])
        forecasts.append(dict(phase=phase,rows=len(issued),rmse_bps=float(np.sqrt(np.mean((pred-actual)**2))*1e4),
                              zero_rmse_bps=float(np.sqrt(np.mean(actual**2))*1e4),positive_predictions=int(np.sum(pred>0)),
                              positive_actual=int(np.sum(actual>0)),
                              positive_prediction_mean_realized_bps=float(actual[pred>0].mean()*1e4) if np.any(pred>0) else None))
    verify(sources)
    save(OUT/'diagnostics.json',dict(rows=rows,forecasts=forecasts,issued_episode_predictions=predictions,
         scope='Variable-duration overlapping virtual episode diagnostics. Not independent trades or full-portfolio PnL; inference values are evaluated as issued before their outcome. No post-hoc re-selection.',
         protected=protected(),orders_sent=0,historical_independence=False))
    print('DONE lifecycle diagnostics',forecasts,flush=True)


if __name__=='__main__':main()
