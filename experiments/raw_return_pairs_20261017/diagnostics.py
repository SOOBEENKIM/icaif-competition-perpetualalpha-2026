"""Forecast calibration and account attribution after fixed candidate evaluation."""
import gzip,json,time
import numpy as np
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha
from experiments.dense_history_pairs_20261017.research import load
from .research import HERE,OUT,PARENT,MATCHED,ARM,verify,path_for


def main():
    cfg=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/reference_protocol.json').read_text());d,_,_=inputs(cfg)
    a=load(OUT/'reference_issued_forecasts.npz');actual=a['target']*a['unit'];times=a['times']
    valid=a['fresh']&a['terminal_fresh']&np.isfinite(actual)&np.isfinite(a['prediction_raw'])&np.isfinite(a['prediction_direct'])
    prediction_rows=[];accounts=[];entry_rows=[]
    for phase in ['validation','comparison']:
        with gzip.open(path_for('reference',phase,'base'),'rt') as f:parent=json.load(f)
        mask=valid&((times>=parent['start'])&(a['maturity']<=parent['end'])&(times%86400==43200))[:,None]
        for kind,pred in [('zero',np.zeros_like(actual)),('direct',a['prediction_direct']),('raw',a['prediction_raw'])]:
            y=actual[mask];v=pred[mask]
            prediction_rows.append(dict(phase=phase,model=kind,pair_days=len(y),rmse_bps=float(np.sqrt(np.mean((v-y)**2))*10000),
                mean_actual_bps=float(y.mean()*10000),mean_prediction_bps=float(v.mean()*10000),directional_accuracy=float(np.mean(y*v>0)) if kind!='zero' else None))
        for kind,arm in [('direct',MATCHED),('raw',ARM)]:
            for sc in ['base','stress']:
                path=(PARENT if kind=='direct' else OUT)/'runs'/f'reference-{phase}-{arm}-{sc}.json.gz'
                with gzip.open(path,'rt') as f:run=json.load(f)
                audit=episode_attribution(run,d,cfg['scenarios'][sc])
                intents=[e for e in run['events'] if e['event']=='pair_intent']
                for e in audit['episodes']:
                    intent=next(v for v in intents if v['time']==e['time']);e['expected_price_bps']=intent['expected_price_bps'];e['direction']=intent['direction']
                    source=intent['forecast_source'];j=int(np.searchsorted(times,source));k=intent['pair_index']
                    if j<len(times) and times[j]==source and valid[j,k] and a['maturity'][j]<=parent['end']:
                        entry_rows.append(dict(phase=phase,model=kind,scenario=sc,time=e['time'],pair=e['pair'],
                            source_prediction_bps=float((a['prediction_raw'] if kind=='raw' else a['prediction_direct'])[j,k]*intent['direction']*10000),
                            source_realized_24h_bps=float(actual[j,k]*intent['direction']*10000),
                            includes_fill=e['fills']>0,note='Reference source-to-fixed-24h price label; not actual delayed entry/renewed exit return.'))
                totals={key:sum(e[key] for e in audit['episodes']) for key in ['price_move','execution_cost','fees','funding','net_pnl']}
                accounts.append(dict(phase=phase,model=kind,scenario=sc,base_owned_pnl=run['metrics']['base_owned_pnl'],totals=totals,**audit))
    books=json.loads((OUT/'books/results.json').read_text())
    diagnostics=dict(at_unix=time.time(),prediction_checks=prediction_rows,account_attribution=accounts,entry_source_labels=entry_rows,
        book_summary=books['summary'],interpretation='Target normalization ablation only. Nonoverlapping source days still share assets across pairs. Entry-conditioned outcomes are post-evaluation diagnostics, not selection or independent evidence. Account attribution uses actual recorded fills and cash; fixed 24h labels are not treated as those fills.',
        orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False)
    save(OUT/'diagnostics.json',diagnostics)
    for r in prediction_rows:print('FORECAST',r,flush=True)
    for r in accounts:
        if r['phase']=='comparison':print('ACCOUNT',r['model'],r['scenario'],r['totals'],flush=True)
    for model in ['direct','raw']:
        rows=[r for r in entry_rows if r['phase']=='comparison' and r['model']==model and r['scenario']=='base']
        print('ENTRY',model,'count',len(rows),'predicted',[round(r['source_prediction_bps'],2) for r in rows],
              'actual',[round(r['source_realized_24h_bps'],2) for r in rows],flush=True)
    verify(json.loads((OUT/'frozen.json').read_text())['sources']);verify(json.loads((OUT/'books/frozen.json').read_text())['sources'])
    verify(json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'])
    paths=[HERE/'diagnostics.py',OUT/'diagnostics.json',OUT/'results.json',OUT/'books/results.json']
    save(OUT/'diagnostics_frozen.json',dict(at_unix=time.time(),sources={str(p.relative_to(ROOT)):sha(p) for p in paths},post_evaluation_only=True))
    print('DONE diagnostics and protected source verification',flush=True)


if __name__=='__main__':main()
