"""Separate gate proxy diagnostics from actual recorded account attribution."""
import gzip,json,time
from pathlib import Path
import numpy as np
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha
from experiments.dense_history_pairs_20261017.research import load
from .research import HERE,OUT,PARENT,MATCHED,ARM,path_for,verify
from .model import action


def main():
    cfg=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/reference_protocol.json').read_text());d,_,_=inputs(cfg)
    a=load(OUT/'reference_issued_forecasts.npz');actual=a['target']*a['unit'];times=a['times'];cost=.0038
    valid=a['fresh']&a['terminal_fresh']&np.isfinite(actual)&np.isfinite(a['prediction_direct'])&a['economic_available']
    proxy_rows=[];accounts=[];entry_rows=[]
    for phase in ['validation','comparison']:
        with gzip.open(path_for('reference',phase,'base'),'rt') as f:parent=json.load(f)
        mask=valid&((times>=parent['start'])&(a['maturity']<=parent['end'])&(times%86400==43200))[:,None]
        direct=action(a['prediction_direct'],cost)
        gated=np.where(direct==a['economic_action'],direct,0)
        for kind,decisions in [('direct',direct),('gate',gated)]:
            y=actual[mask];selected=decisions[mask];payoff=y*selected-cost*abs(selected)
            proxy_rows.append(dict(phase=phase,model=kind,source_pair_days=len(y),nonflat_sources=int(np.count_nonzero(selected)),
                mean_proxy_payoff_bps=float(payoff.mean()*10000),sum_proxy_payoff_bps=float(payoff.sum()*10000),
                note='Fixed-source 24h independent opportunity proxy with 38bp roundtrip cost. Excludes funding, actual fills, capacity, concurrency, stops and renewal. Not account return or Q.'))
        for kind,arm in [('direct',MATCHED),('gate',ARM)]:
            for sc in ['base','stress']:
                path=(PARENT if kind=='direct' else OUT)/'runs'/f'reference-{phase}-{arm}-{sc}.json.gz'
                with gzip.open(path,'rt') as f:run=json.load(f)
                audit=episode_attribution(run,d,cfg['scenarios'][sc]);intents=[e for e in run['events'] if e['event']=='pair_intent']
                for e in audit['episodes']:
                    intent=next(v for v in intents if v['time']==e['time']);e['expected_price_bps']=intent['expected_price_bps'];e['direction']=intent['direction']
                    source=intent['forecast_source'];j=int(np.searchsorted(times,source));k=intent['pair_index']
                    if j<len(times) and times[j]==source and valid[j,k] and a['maturity'][j]<=parent['end']:
                        entry_rows.append(dict(phase=phase,model=kind,scenario=sc,time=e['time'],pair=e['pair'],
                            source_prediction_bps=float(a['prediction_direct'][j,k]*intent['direction']*10000),
                            source_realized_24h_bps=float(actual[j,k]*intent['direction']*10000),
                            decision_action=int(a['economic_action'][j,k]),decision_score=float(a['decision_score'][j,k]),includes_fill=e['fills']>0,
                            note='Decision score is dimensionless classification score, not expected return. Fixed 24h source label is not actual delayed/renewed trade PnL.'))
                totals={key:sum(e[key] for e in audit['episodes']) for key in ['price_move','execution_cost','fees','funding','net_pnl']}
                accounts.append(dict(phase=phase,model=kind,scenario=sc,base_owned_pnl=run['metrics']['base_owned_pnl'],totals=totals,**audit))
    records=json.loads((OUT/'reference_fits.json').read_text())
    fit_summary=dict(models=len(records),gap_max=max(r['solver']['gap'] for r in records),
        iteration_quantiles=np.quantile([r['solver']['iterations'] for r in records],[0,.5,.9,1]).tolist())
    books=json.loads((OUT/'books/results.json').read_text())
    diagnostics=dict(at_unix=time.time(),proxy_decision_checks=proxy_rows,solver=fit_summary,account_attribution=accounts,
        entry_source_labels=entry_rows,book_summary=books['summary'],
        interpretation='Gate changes only new entry eligibility; unchanged parent mu/limits/holding logic. Source proxy excludes execution constraints and cannot substitute for the full account comparisons. No post-evaluation gate retuning.',
        orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False)
    save(OUT/'diagnostics.json',diagnostics)
    for row in proxy_rows:print('PROXY',row,flush=True)
    print('SOLVER',fit_summary,flush=True)
    for row in accounts:
        if row['phase']=='comparison':print('ACCOUNT',row['model'],row['scenario'],row['totals'],flush=True)
    verify(json.loads((OUT/'frozen.json').read_text())['sources']);verify(json.loads((OUT/'books/frozen.json').read_text())['sources'])
    verify(json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'])
    paths=[Path(__file__),OUT/'diagnostics.json',OUT/'results.json',OUT/'books/results.json']
    save(OUT/'diagnostics_frozen.json',dict(at_unix=time.time(),sources={str(p.relative_to(ROOT)):sha(p) for p in paths},post_evaluation_only=True))
    print('DONE diagnostics',flush=True)


if __name__=='__main__':main()
