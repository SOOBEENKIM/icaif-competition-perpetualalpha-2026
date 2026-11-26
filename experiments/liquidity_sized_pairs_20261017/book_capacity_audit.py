"""Compare past-volume sizing with contemporaneous old visible book depth.

No reserved tape, current market request, live order, or PnL inference is used.
This is a diagnostic of the historical execution proxy, not a new strategy.
"""
import json,math,time
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.basis_carry_20261017.probe import load_tape,asof
from experiments.forecast_validity_pairs_20261017.book_plan import plan_entry
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,save,sha
from experiments.dense_history_pairs_20261017.research import load
from experiments.complementary_alpha_20261017.research import protected
from replay_lab.common_comparison.engine import rounded
from .policy import capacity_volume

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts/book_capacity_audit'

def paired_capacity(cap,weights,prices,ids,p,tolerance):
    if not np.isfinite(cap).all():return dict(gross=0.,minimum_and_hedge_pass=False,quantity=[0.,0.],observed=False)
    gross=min(120000.,float(np.min(cap*prices/abs(weights))))
    q=np.array([rounded(gross*w/price,p['execution']['decimals'][j]) for j,w,price in zip(ids,weights,prices)])
    minimum=bool(np.all(abs(q)*prices>=p['execution']['minimum_order_value']))
    balance=minimum and abs(abs(q[1]*prices[1]/(q[0]*prices[0]))/abs(weights[1]/weights[0])-1)<=tolerance
    return dict(gross=float(abs(q)@prices),minimum_and_hedge_pass=bool(balance),quantity=q.tolist(),observed=True)

def main():
    OUT.mkdir(exist_ok=False)
    allowed=json.loads((ROOT/'experiments/basis_carry_20261017/protocol.json').read_text())['tapes']
    cfg=json.loads((ROOT/'experiments/shared_execution_pairs_20261017/book_lifecycle_settings.json').read_text())
    cfg.update(book_intent_lifetime_hours=1,funding_forecast_hours=24)
    candle_cfg=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/par_protocol.json').read_text())
    p=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['trading_protocol']
    statepath=ROOT/'experiments/rolling_feature_pairs_20261017/artifacts/par_direct_24_state.npz'
    data=load(INPUT/'inputs.npz');state=load(statepath)
    sources={str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__),HERE/'policy.py',INPUT/'inputs.npz',statepath,*(ROOT/path for path in allowed)]}
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),settings=dict(
        cadence_seconds=900,history_window_bars=12,visible_depth_band_bps=cfg['visible_depth_band_bps'],
        method='Same received clock for books; completed 5m history as of the decision. Compare rounded minimum paired lot feasibility before global risk and financial edge. Book planner additionally checks account risk and all expected entry/exit/funding costs.',
        not_an_execution_or_return_test=True,reserved_raw_read=False)))
    cap=capacity_volume(data['par_volume'],data['par_trades'],12);rows=[];counts=Counter();tapes=[]
    for tape in allowed:
        streams,meta=load_tape(ROOT/tape);tapes.append(meta)
        first=math.ceil(meta['start_ms']/900000)*900
        for now in range(first,meta['end_ms']//1000+1,900):
            i=int(np.searchsorted(data['times'],now,side='right')-1)
            if i<0 or now-data['times'][i]>=300:continue
            for k,ids0 in enumerate(state['pairs']):
                if not state['directions'][i,k]:continue
                ids=list(map(int,ids0));pair=[p['symbols'][j] for j in ids]
                quotes={s:asof(streams[s],now*1000,cfg) if s in streams else None for s in pair}
                if any(q is None for q in quotes.values()):counts['pair_time_missing_fresh_book_or_context']+=1;continue
                prices=np.array([quotes[s]['mid'] for s in pair]);anchors=state['source_prices'][i,ids]
                beta=float(state['beta'][i,k]);venue_beta=beta*anchors[0]/anchors[1]*prices[1]/prices[0]
                mu=state['mu'][i,k];source=int(state['signal_times'][i,k])
                for scenario,sc in cfg['scenarios'].items():
                    part=candle_cfg['scenarios'][scenario]['participation']
                    for side in [-1,1]:
                        weights=side*np.array([1.,-venue_beta])/(1+venue_beta);depth=[]
                        for symbol,w in zip(pair,weights):
                            levels=quotes[symbol]['book']['levels'][1 if w>0 else 0];best=float(levels[0]['px'])
                            limit=best*(1+np.sign(w)*cfg['visible_depth_band_bps']/10000)
                            depth.append(sum(float(x['sz'])*sc['depth_fraction'] for x in levels if np.sign(w)*(float(x['px'])-limit)<=1e-10))
                        original=paired_capacity(part*data['par_volume'][i,ids],weights,prices,ids,p,cfg['pair_balance_tolerance'])
                        median=paired_capacity(part*cap[i,ids],weights,prices,ids,p,cfg['pair_balance_tolerance'])
                        book=paired_capacity(np.array(depth),weights,prices,ids,p,cfg['pair_balance_tolerance'])
                        intent=dict(pair=pair,ids=ids,direction=side,source_beta=beta,source_prices=anchors.tolist(),
                            terminal_prices=(anchors*np.array([1+mu,1-mu])).tolist(),forecast_source=source,horizon=24,
                            account_marks=data['par_close'][i].tolist())
                        plan=plan_entry(intent,quotes,now,cfg,p,sc,np.zeros(10),1e6,data['sigma'][i])
                        rows.append(dict(tape=tape,time=now,pair=pair,scenario=scenario,direction=side,
                            original_volume=original,median_volume=median,visible_depth=book,
                            original_volume_units=data['par_volume'][i,ids].tolist(),median_volume_units=cap[i,ids].tolist(),
                            book_capped_units=depth,book_mid=prices.tolist(),candle_close=data['par_close'][i,ids].tolist(),
                            candle_vs_book_mid_bps=((data['par_close'][i,ids]/prices-1)*10000).tolist(),
                            planner_reason=plan['reason'],planner_accepted=plan['accepted']))
    aggregate={}
    for scenario in cfg['scenarios']:
        selected=[r for r in rows if r['scenario']==scenario]
        c=Counter();errors=[]
        for row in selected:
            c['pair_direction_observations']+=1
            for name in ['original_volume','median_volume','visible_depth']:
                c[name+'_passes_minimum_and_hedge']+=int(row[name]['minimum_and_hedge_pass'])
            if row['visible_depth']['minimum_and_hedge_pass']:
                c['book_available_original_volume_rejects']+=int(not row['original_volume']['minimum_and_hedge_pass'])
                c['book_available_median_volume_rejects']+=int(not row['median_volume']['minimum_and_hedge_pass'])
            c['planner_'+row['planner_reason']]+=1;errors.extend(abs(v) for v in row['candle_vs_book_mid_bps'])
        aggregate[scenario]=dict(counts=dict(c),abs_candle_book_mid_difference_bps_quantiles=np.quantile(errors,[0,.5,.9,1]).tolist() if errors else [])
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path
    save(OUT/'results.json',dict(rows=rows,aggregate=aggregate,tape_metadata=tapes,missing=dict(counts),
        warning='Overlapping observations and both directions are diagnostic counts, not independent trades. Displayed depth can change before arrival and before exit. Price gaps use last-trade candle close versus contemporaneous order-book mid, not executable arbitrage profit.',
        complete_lifecycles=0,account_return=None,financial_score=None,reserved_raw_read=False,orders_sent=0,protected=protected()))
    print(json.dumps(aggregate,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
