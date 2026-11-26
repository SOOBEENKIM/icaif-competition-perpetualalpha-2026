"""Account attribution and explicit pre-risk candle opportunity decomposition."""
import gzip,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,save,sha,ts
from experiments.dense_history_pairs_20261017.research import load
from replay_lab.common_comparison.engine import rounded
from replay_lab.pullback.broker import execution_price
from experiments.quote_anchored_pairs_20261017.pricing import value_at_entry

HERE=Path(__file__).resolve().parent;ART=HERE/'artifacts'
HOURLY=ROOT/'experiments/hourly_refresh_pairs_20261017'


def funnel(d,state,cfg,p,model):
    rows=[]
    for scenario,sc in cfg['scenarios'].items():
        counts=Counter();values=[]
        for i in np.flatnonzero(d['times']%900==0):
            now=d['times'][i]
            for k,ids in enumerate(state['pairs']):
                ids=list(ids)
                if not state['directions'][i,k]:continue
                counts['valid_pair_decisions']+=1
                marks=d['par_close'][i,ids];anchors=state['source_prices'][i,ids]
                beta=state['beta'][i,k];mu=state['mu'][i,k];ratio=beta*anchors[0]/anchors[1]
                if np.any(d['par_volume'][i,ids]<=0):counts['one_or_both_zero_volume']+=1
                best=None
                for direction in (-1,1):
                    weights=direction*np.array([1.,-ratio*marks[1]/marks[0]]);weights/=abs(weights).sum()
                    gross=min(cfg['pair_max_gross'],float(np.min(sc['participation']*d['par_volume'][i,ids]*marks/abs(weights))))
                    q=np.array([rounded(gross*w/px,p['execution']['decimals'][j]) for j,w,px in zip(ids,weights,marks)])
                    if np.any(abs(q)*marks<p['execution']['minimum_order_value']):continue
                    if abs(abs(q[1]/q[0])/ratio-1)>cfg['pair_balance_tolerance']:continue
                    preview=np.array([execution_price(px,np.sign(a),sc['impact_bps'],p['execution']['decimals'][j]) for px,a,j in zip(marks,q,ids)])
                    terminal=anchors*np.array([1+mu,1-mu]);gross=float(abs(q)@marks)
                    funding=-q*d['reference_close'][i,ids]*d['known_hourly_funding'][i,ids]*(24-(now-state['signal_times'][i,k])/3600)
                    funding=float(np.where(funding>0,.5*funding,funding).sum())
                    value=value_at_entry(q,marks,preview,float(q@(terminal-marks))/gross,funding,sc['fee_bps'],sc['impact_bps'])
                    parts=dict(model_bp=float(q@(terminal-anchors))/gross*1e4,
                        source_aging_bp=float(q@(anchors-d['reference_close'][i,ids]))/gross*1e4,
                        venue_basis_bp=float(q@(d['reference_close'][i,ids]-marks))/gross*1e4,
                        funding_bp=funding/gross*1e4,
                        cost_bp=(float(q@(preview-marks))+value['entry_fee_cash']+value['exit_reserve_cash'])/gross*1e4)
                    net=value['expected_net_cash']/gross*1e4
                    assert abs(net-(parts['model_bp']+parts['source_aging_bp']+parts['venue_basis_bp']+parts['funding_bp']-parts['cost_bp']))<1e-7
                    item=dict(time=int(now),pair=[p['symbols'][j] for j in ids],direction=direction,gross=gross,net_bp=net,**parts)
                    if best is None or item['net_bp']>best['net_bp']:best=item
                if best is not None:
                    values.append(best);counts['volume_lot_minimum_and_hedge_pass']+=1
                    if best['net_bp']>cfg['entry_buffer_bps']:counts['positive_buffered_edge_before_global_risk']+=1
        quantiles={key:np.quantile([r[key] for r in values],[0,.5,.9,1]).tolist() if values else []
                   for key in ['net_bp','model_bp','source_aging_bp','venue_basis_bp','funding_bp','cost_bp','gross']}
        rows.append(dict(model=model,scenario=scenario,counts=dict(counts),quantiles=quantiles,
                         positive=[r for r in values if r['net_bp']>cfg['entry_buffer_bps']]))
    return rows


def main():
    for rel,digest in json.loads((ART/'frozen.json').read_text())['sources'].items():assert sha(ROOT/rel)==digest,rel
    results=json.loads((ART/'results.json').read_text());cfg=json.loads((HOURLY/'reference_protocol.json').read_text())
    d,_,p=inputs(cfg);attribution=[]
    for row in results['rows']:
        if row['venue']!='reference' or '_sector_' not in row['arm']:continue
        key='-'.join(row[k] for k in ['venue','phase','arm','scenario'])
        with gzip.open(ART/'runs'/f'{key}.json.gz','rt') as f:run=json.load(f)
        audit=episode_attribution(run,d,cfg['scenarios'][row['scenario']]);by_pair=defaultdict(lambda:Counter())
        for episode in audit['episodes']:
            bucket=by_pair['/'.join(episode['pair'])]
            for key in ['price_move','execution_cost','fees','funding','net_pnl','fills']:bucket[key]+=episode[key]
            bucket['episodes']+=1;bucket['closed']+=int(episode['closed'])
        attribution.append(dict(phase=row['phase'],arm=row['arm'],scenario=row['scenario'],by_pair=dict(by_pair),**audit))
    save(ART/'attribution.json',dict(rows=attribution,not_used_for_selection=True,orders_sent=0))
    par=load(INPUT/'inputs.npz');pcfg=json.loads((HOURLY/'par_protocol.json').read_text());flow=[]
    for model,path in [('two_pair_direct',ROOT/'experiments/rolling_feature_pairs_20261017/artifacts/par_direct_24_state.npz'),
                       ('two_pair_bias',HOURLY/'prepared/par_bias_24_state.npz'),('six_pair_direct',ART/'par_state.npz')]:
        flow.extend(funnel(par,load(path),pcfg,p,model))
    save(ART/'entry_funnel.json',dict(rows=flow,start=int(par['times'][0]),end=int(par['times'][-1]),
        scope='Flat opportunity diagnostic before global risk, account occupancy, delayed independent fills, holding or exits. Candle volume is not actual book depth. Both directions inspected but only best per pair/time counted.',
        quantities_in_bps=True,not_account_returns=True,not_used_for_selection=True,orders_sent=0))
    arrays=load(ART/'reference_forecasts.npz');pairs=json.loads((HERE/'protocol.json').read_text())['pairs'];forecasts=[]
    for phase in ['validation','comparison']:
        start=ts(cfg['validation_start'] if phase=='validation' else cfg['comparison_start']);end=ts(cfg['comparison_start'] if phase=='validation' else cfg['end'])
        time_mask=(arrays['times']>=start)&(arrays['maturity']<=end)&(arrays['times']%86400==0)
        for k,pair in enumerate(pairs):
            mask=time_mask&arrays['fresh'][:,k]&arrays['terminal_fresh'][:,k]&np.isfinite(arrays['target'][:,k])&np.isfinite(arrays['predictions'][:,k])
            actual=arrays['target'][mask,k];pred=arrays['predictions'][mask,k]/arrays['unit'][mask,k]
            forecasts.append(dict(phase=phase,pair=pair,samples=len(actual),normalized_rmse=float(np.sqrt(np.mean((actual-pred)**2))),
                zero_rmse=float(np.sqrt(np.mean(actual**2))),sign_accuracy=float(np.mean(np.sign(actual)==np.sign(pred))),
                mean_abs_return_bps=float(np.mean(abs(arrays['predictions'][mask,k]))*1e4)))
    save(ART/'forecast_diagnostics.json',dict(rows=forecasts,nonoverlapping_sources=True,not_used_for_selection=True))
    print('FUNNEL',[(r['model'],r['scenario'],r['counts']) for r in flow])
    for r in attribution:
        if r['phase']=='comparison' and r['arm'].startswith('first'):
            print('ATTRIBUTION',r['scenario'],{k:round(v['net_pnl'],3) for k,v in r['by_pair'].items()})


if __name__=='__main__':main()
