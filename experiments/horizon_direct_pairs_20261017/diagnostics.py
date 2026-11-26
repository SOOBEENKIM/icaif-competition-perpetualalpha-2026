"""After-selection diagnostics; no forecast or candidate changes."""
import gzip,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha,ts
from experiments.dense_history_pairs_20261017.research import load

HERE=Path(__file__).resolve().parent;ART=HERE/'artifacts'
PARENT=ROOT/'experiments/rolling_feature_pairs_20261017'
HOURLY=ROOT/'experiments/hourly_refresh_pairs_20261017'


def read_run(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def main():
    for path,digest in json.loads((ART/'frozen.json').read_text())['sources'].items():assert sha(ROOT/path)==digest,path
    cfg=json.loads((HOURLY/'reference_protocol.json').read_text());d,_,p=inputs(cfg)
    pairs=json.loads((HERE/'protocol.json').read_text())['pairs'];forecasts=[];attributions=[];intent_rows=[]
    for h in (6,24,72):
        arrays=load(PARENT/'artifacts/reference_issued_forecasts.npz' if h==24 else ART/f'reference_{h}_forecasts.npz')
        prediction=arrays['prediction_direct'] if h==24 else arrays['predictions']
        for phase in ['validation','comparison']:
            start=ts(cfg['validation_start'] if phase=='validation' else cfg['comparison_start'])
            end=ts(cfg['comparison_start'] if phase=='validation' else cfg['end'])
            # Different horizons require different non-overlapping samples.
            eligible=(arrays['times']>=start)&(arrays['maturity']<=end)&(arrays['times']%(h*3600)==0)
            for k,pair in enumerate(pairs):
                mask=eligible&arrays['fresh'][:,k]&arrays['terminal_fresh'][:,k]
                mask &= np.isfinite(arrays['target'][:,k])&np.isfinite(prediction[:,k])
                actual=arrays['target'][mask,k];pred=prediction[mask,k]/arrays['unit'][mask,k]
                forecasts.append(dict(horizon=h,phase=phase,pair=pair,samples=int(mask.sum()),
                    normalized_rmse=float(np.sqrt(np.mean((actual-pred)**2))),
                    zero_rmse=float(np.sqrt(np.mean(actual**2))),
                    sign_accuracy=float(np.mean(np.sign(actual)==np.sign(pred))),
                    mean_abs_forecast_bp=float(np.mean(abs(prediction[mask,k]))*1e4),
                    forecast_rms_bp=float(np.sqrt(np.mean(prediction[mask,k]**2))*1e4),
                    times=arrays['times'][mask].tolist()))
            for sc in ['base','stress']:
                arm=f'first_dense_hourlyhedge_direct_{h}'
                directory=PARENT/'artifacts' if h==24 else ART
                run=read_run(directory/'runs'/f'reference-{phase}-{arm}-{sc}.json.gz')
                audit=episode_attribution(run,d,cfg['scenarios'][sc]);by_pair=defaultdict(Counter)
                for e in audit['episodes']:
                    v=by_pair['/'.join(e['pair'])]
                    for key in ['price_move','execution_cost','fees','funding','net_pnl','fills']:v[key]+=e[key]
                    v['episodes']+=1;v['closed']+=int(e['closed'])
                attributions.append(dict(horizon=h,phase=phase,scenario=sc,by_pair=dict(by_pair),
                    events=dict(Counter(e['event'] for e in run['events'])),
                    exit_reasons=dict(Counter(e['reason'] for e in run['events'] if e['event']=='exit_requested')),
                    metrics=run['metrics'],**audit))
                # Observe whether accepted entry hypotheses match future prices.
                # This is hypothetical fixed inventory, including zero fills;
                # it is NOT realized account PnL and does not change selection.
                for e in run['events']:
                    if e['event']!='pair_intent':continue
                    maturity=e['forecast_source']+h*3600
                    j=int(np.searchsorted(d['times'],maturity))
                    i=int(np.searchsorted(d['times'],e['time']))
                    if j>=len(d['times']) or d['times'][j]!=maturity or maturity>end:continue
                    ids=e['ids'];q=np.asarray(e['target']);gross=float(abs(q)@d['par_close'][i,ids])
                    realized=float(q@(d['par_close'][j,ids]-d['par_close'][i,ids]))
                    intent_rows.append(dict(horizon=h,phase=phase,scenario=sc,time=e['time'],pair=e['pair'],
                        maturity=maturity,forecast_source=e['forecast_source'],planned_gross=gross,
                        forecast_price_bp=e['expected_price_bps'],fixed_quantity_future_price_bp=realized/gross*1e4,
                        fixed_quantity_future_price_cash=realized,forecast_net_cash=e['expected_net_cash'],
                        actual_fills_not_assumed=True,not_account_pnl=True))
    save(ART/'forecast_diagnostics.json',dict(rows=forecasts,nonoverlapping_within_horizon=True,
        comparable_with_zero_at_same_horizon=True,not_same_samples_across_horizons=True,not_used_for_selection=True))
    save(ART/'attribution.json',dict(rows=attributions,not_used_for_selection=True,orders_sent=0))
    save(ART/'entry_forecast_diagnostics.json',dict(rows=intent_rows,not_actual_fills=True,
        scope='Hypothetical original planned quantities held unchanged to their forecast maturity, before funding/costs. Includes rejected-by-execution zero-fill intentions. Only price-forecast calibration, never reported as profit.',
        not_used_for_selection=True,orders_sent=0))
    for r in forecasts:print('FORECAST',{k:v for k,v in r.items() if k!='times'})
    for r in attributions:
        if r['horizon']!=24:
            print('ATTRIBUTION',r['horizon'],r['phase'],r['scenario'],r['by_pair'],'exits',r['exit_reasons'])


if __name__=='__main__':main()
