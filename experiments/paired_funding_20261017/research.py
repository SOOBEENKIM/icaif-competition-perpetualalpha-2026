"""Funding-rate comparison, not a trade backtest or account-level score."""
import json
import time
from pathlib import Path
import numpy as np
from experiments.cost_horizon_20261016.research import data,save,sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.pairs_forecast_20261017.model import design
from .model import HOUR,load,forecasts,labels,paired_differential,carry_diagnostic

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'artifacts'
def ts(s):return int(np.datetime64(s,'ms').astype('int64'))


def summarize(rows,cfg):
    summaries=[]
    for h in cfg['horizons_hours']:
        for kind in cfg['models']:
            selected=[r for r in rows if r['horizon']==h and r['model']==kind]
            if not selected:continue
            error=np.array([r['prediction_bps']-r['actual_bps'] for r in selected])
            part=dict(horizon=h,model=kind,observations=len(selected),distinct_decision_times=len({r['time'] for r in selected}),
                nonoverlapping_time_blocks=len({r['time'] for r in selected if r['nonoverlap']}),
                mae_bps=float(np.mean(abs(error))),mse_bps_squared=float(np.mean(error**2)))
            for scenario in ('base','stress'):
                active=[r[scenario] for r in selected if r[scenario]['active']]
                values=[r['fee_only_net_bps'] for r in active]
                part[scenario]=dict(active_predictions=len(active),positive_realized_carry_after_fees=sum(v>0 for v in values),
                    average_active_fee_only_net_bps=float(np.mean(values)) if values else None,
                    total_fee_only_net_bps=float(sum(values)),not_a_strategy_pnl=True)
            summaries.append(part)
    return summaries


def observations(p,cfg,pred,d,m,phase):
    start,end=map(ts,cfg[phase]);rows=[];missing={};ys={h:labels(p,h) for h in cfg['horizons_hours']}
    for i,at in enumerate(p['times']):
        if not start<=at<end or at%(cfg['decision_cadence_hours']*HOUR):continue
        ri=int(np.searchsorted(m.times,at//1000,side='right')-1)
        if ri<0 or at//1000-m.times[ri]>21600:raise ValueError('Reference hedge input unavailable')
        for k,(a,b) in enumerate(d['pairs']):
            beta=float(d['beta'][ri,k]);corr=float(d['x'][ri,k,d['names'].index('correlation_168')])
            if not np.isfinite(beta) or corr<.3:continue
            # Funding and reference universes are explicitly aligned in main.
            for h in cfg['horizons_hours']:
                y,mat=ys[h]
                if not np.isfinite(y[i,[a,b]]).all():continue
                outcome=int(max(mat[i,a],mat[i,b]))
                if phase=='selection' and outcome>=ts(cfg['selection_outcome_deadline']):continue
                missing_models=[name for name in cfg['models'] if not np.isfinite(pred[h][name][i,[a,b]]).all()]
                if missing_models:
                    for name in missing_models:missing[f'{h}:{name}']=missing.get(f'{h}:{name}',0)+1
                    continue
                actual=paired_differential(y[i,a],y[i,b],beta)
                for name in cfg['models']:
                    forecast=float(paired_differential(pred[h][name][i,a],pred[h][name][i,b],beta))
                    rows.append(dict(phase=phase,horizon=h,model=name,pair=cfg['pairs'][k],pair_id=k,time=int(at)+1000,
                        last_outcome_time=outcome,reference_source_time=int(m.times[ri])*1000,beta=beta,correlation=corr,
                        prediction_bps=forecast,actual_bps=float(actual),nonoverlap=bool((at-start)%(h*HOUR)==0),
                        base=carry_diagnostic(forecast,actual,18,5),stress=carry_diagnostic(forecast,actual,24,5)))
    return rows,missing


def main():
    OUT.mkdir(exist_ok=False);cfg=json.loads((HERE/'protocol.json').read_text());p=load(ROOT,cfg)
    source=ROOT/cfg['source']
    files=[*HERE.glob('*.py'),HERE/'protocol.json',*(source/f'funding_{s}.json' for s in cfg['symbols']),
        ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',ROOT/'experiments/pairs_forecast_20261017/model.py']
    frozen=dict(at=time.time(),sources={str(f.relative_to(ROOT)):sha(f) for f in files},protected=protected(),historical_independence=False)
    save(OUT/'frozen.json',frozen)
    _,_,_,m=data();assert tuple(cfg['symbols'])==m.symbols
    d=design(m,cfg);pred,models=forecasts(p,cfg,fit_start=ts(cfg['selection'][0]))
    save(OUT/'models.json',models)
    np.savez_compressed(OUT/'predictions.npz',times=p['times'],**{f'{name}_{h}':v for h,x in pred.items() for name,v in x.items()})
    selection_rows,missing_selection=observations(p,cfg,pred,d,m,'selection')
    selection_summaries=summarize(selection_rows,cfg)
    selected={}
    for h in cfg['horizons_hours']:
        ranked=sorted([r for r in selection_summaries if r['horizon']==h and r['model']!='zero'],key=lambda r:(r['mae_bps'],r['model']))
        if not ranked:raise ValueError('No comparable forecast observations')
        best=ranked[0];zero=next(r for r in selection_summaries if r['horizon']==h and r['model']=='zero')
        selected[h]=dict(model=best['model'],selection_mae_bps=best['mae_bps'],zero_mae_bps=zero['mae_bps'],
                         selection_beats_zero=best['mae_bps']<zero['mae_bps'],ranking=[r['model'] for r in ranked])
    save(OUT/'selection.json',dict(at=time.time(),selected=selected,comparison_outcomes_not_evaluated=True,
                                   missing_common_support=missing_selection,summaries=selection_summaries))
    comparison_rows,missing_comparison=observations(p,cfg,pred,d,m,'comparison')
    later=summarize(comparison_rows,cfg);checks=[]
    for h,choice in selected.items():
        by_name={r['model']:r for r in later if r['horizon']==h};r=by_name[choice['model']]
        checks.append(dict(horizon=h,selected=choice['model'],beats_persistence=r['mae_bps']<by_name['persistence']['mae_bps'],
            beats_zero=r['mae_bps']<by_name['zero']['mae_bps'],adequate_nonoverlapping_blocks=r['nonoverlapping_time_blocks']>=cfg['minimum_nonoverlapping_comparison_time_blocks'],
            nonoverlapping_time_blocks=r['nonoverlapping_time_blocks']))
    allowances=[]
    for h,choice in selected.items():
        for k,pair in enumerate(cfg['pairs']):
            rows=[r for r in selection_rows+comparison_rows if r['horizon']==h and r['model']==choice['model'] and r['pair_id']==k and r['nonoverlap']]
            errors=[abs(r['prediction_bps']-r['actual_bps']) for r in rows]
            allowances.append(dict(horizon=h,pair=pair,blocks=len(rows),last_label=max((r['last_outcome_time'] for r in rows),default=None),
                diagnostic_error_allowance_bps=float(np.quantile(errors,.8)) if len(rows)>=5 else None,coverage_guarantee=False))
    for name,digest in frozen['sources'].items():assert sha(ROOT/name)==digest,name
    save(OUT/'results.json',dict(selected=selected,selection=selection_summaries,comparison=later,checks=checks,
        missing_comparison_common_support=missing_comparison,observations=selection_rows+comparison_rows,error_allowances=allowances,
        history_hours=len(p['times']),latest_settlement=int(p['seen'].max()),protected_after=protected(),
        rate_forecast_only=True,actual_strategy_PnL=None,actual_strategy_Q=None,historical_independence=False,orders_sent=0))
    print(json.dumps(dict(selected=selected,comparison=later,checks=checks),indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),flush=True)


if __name__=='__main__':main()
