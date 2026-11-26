"""Compare candle trade-flow gating with fresh historical visible book depth.

This never substitutes books from other dates into a strategy backtest.
"""
import json,math,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from strategy.agent import TAIL
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,save,sha
from experiments.basis_carry_20261017.probe import load_tape,asof,delayed_snapshot,sweep
from replay_lab.common_comparison.engine import rounded
from .signals import proposals

HERE=Path(__file__).resolve().parent


def requested_size(notional,price,decimals,minimum):
    step=10**-decimals
    q=abs(rounded(notional/price,decimals))
    if q*price<minimum:q=math.ceil(minimum/price/step)*step
    return q


def main():
    out=HERE/'liquidity_artifacts';out.mkdir(exist_ok=False)
    cfg=json.loads((HERE/'par_protocol.json').read_text());p=json.loads((TAIL/'baseline.json').read_text())['protocol']
    old=json.loads((ROOT/'experiments/basis_carry_20261017/protocol.json').read_text())
    spec=dict(tapes=old['tapes'],symbols=['BTC','ETH','HYPE'],notionals=[10,100],decision_step_ms=60000,
        book_max_age_ms=15000,context_max_age_ms=3000,future_exit_observation_grace_ms=5000,
        scenarios={'base':dict(depth_fraction=1.,latency_ms=1000,participation=.1),
                   'stress':dict(depth_fraction=.5,latency_ms=3000,participation=.05)},
        limit_from_best_bps=10,purpose='Execution-model coverage audit; no alpha fitting or PnL, no tape-to-other-date substitution.')
    save(out/'protocol.json',spec)
    paths=[HERE/'liquidity_audit.py',HERE/'signals.py',HERE/'par_protocol.json',out/'protocol.json',
        INPUT/'inputs.npz',HERE/'artifacts/par_persistent_state.npz',*[ROOT/x for x in spec['tapes']]]
    hashes={str(f.relative_to(ROOT)):sha(f) for f in paths}
    save(out/'frozen.json',dict(at_unix=time.time(),sources=hashes,sealed_streams_read=False,orders_sent=0))
    with np.load(INPUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
    with np.load(HERE/'artifacts/par_persistent_state.npz') as z:de={k:z[k].copy() for k in z.files}
    funnels=[]
    for scenario,sc in cfg['scenarios'].items():
        c=Counter();signal_ids=set()
        for i,t in enumerate(d['times']):
            if t%cfg['signal_cadence_seconds']:continue
            for k,ids in enumerate(de['pairs']):
                direction=int(de['directions'][i,k])
                if not direction:continue
                c['source_intent_views']+=1;signal_ids.add((k,int(de['signal_times'][i,k])))
                if not -cfg['entry_z']*cfg['stop_multiplier']<direction*de['z'][i,k]<0:
                    c['venue_spread_invalid']+=1;continue
                be=de['beta'][i,k];price=d['par_close'][i,ids];w=direction*np.array([1,-be])/(1+be)
                gross=min(cfg['pair_max_gross'],float(np.min(sc['participation']*d['par_volume'][i,ids]*price/abs(w))))
                q=np.array([rounded(gross*z/price[j],p['execution']['decimals'][a]) for j,(z,a) in enumerate(zip(w,ids))])
                if np.any(abs(q)*price<p['execution']['minimum_order_value']):c['quantity_below_minimum']+=1;continue
                hedge=abs(q[1])*price[1]/(abs(q[0])*price[0])
                if abs(hedge/be-1)>cfg['pair_balance_tolerance']:c['lot_imbalance']+=1;continue
                dr=np.zeros(len(de['pairs']),int);dr[k]=direction
                chosen=proposals(i,d,de,None,168,cfg,p,sc,np.zeros(10),1e6,np.zeros(10),dr,'fixed_delayed_persistent_complete_168')
                c['candidate_pass' if chosen else 'budget_or_risk_rejected']+=1
        funnels.append(dict(scenario=scenario,unique_signals=len(signal_ids),counts=dict(c)))
    rows=[];coverage=[];counts=defaultdict(Counter);intent_views=Counter()
    for tape in spec['tapes']:
        streams,meta=load_tape(ROOT/tape);coverage.append(meta)
        step=spec['decision_step_ms']
        for t in range((meta['start_ms']+step-1)//step*step,meta['end_ms']+1,step):
            i=int(np.searchsorted(d['times'],t//1000,side='right')-1)
            if i<0 or d['times'][i]*1000>t:continue
            for k in range(len(de['pairs'])):
                intent_views['pair_views']+=1;intent_views['active_source_intent_views']+=int(de['directions'][i,k]!=0)
            for symbol in spec['symbols']:
                if symbol not in streams:continue
                s=streams[symbol];now=asof(s,t,spec)
                if now is None:continue
                j=p['symbols'].index(symbol);dec=p['execution']['decimals'][j]
                for scenario,sc in spec['scenarios'].items():
                    later=delayed_snapshot(s,t+sc['latency_ms'],spec)
                    cap=abs(rounded(sc['participation']*d['par_volume'][i,j],dec))
                    candle_reject=cap*d['par_close'][i,j]<p['execution']['minimum_order_value']
                    for notional in spec['notionals']:
                        for side in (1,-1):
                            key=(symbol,scenario,notional);c=counts[key];c['valid_decision_books']+=1
                            level=1 if side==1 else 0;best=float(now['book']['levels'][level][0]['px'])
                            qty=requested_size(notional,best,dec,p['execution']['minimum_order_value'])
                            limit=best*(1+side*spec['limit_from_best_bps']/10000)
                            preview,_=sweep(now['book']['levels'][level],qty,sc['depth_fraction'],side,dec,limit)
                            c['candle_source_below_minimum']+=int(candle_reject)
                            c['decision_depth_full']+=int(preview>=qty-1e-10)
                            arrival_qty=None;arrival_time=None
                            if later is None:c['arrival_unobserved']+=1
                            else:
                                arrival_time,book=later
                                arrival_qty,_=sweep(book['book']['levels'][level],qty,sc['depth_fraction'],side,dec,limit)
                                full=preview>=qty-1e-10 and arrival_qty>=qty-1e-10
                                c['decision_and_arrival_depth_full']+=int(full)
                                c['candle_rejected_but_both_books_full']+=int(candle_reject and full)
                            rows.append(dict(tape=tape,time=t,symbol=symbol,scenario=scenario,notional=notional,side=side,
                                requested_qty=qty,limit=limit,decision_depth_qty=preview,arrival_depth_qty=arrival_qty,
                                arrival_time=arrival_time,candle_source_rejected=bool(candle_reject),candle_source_capacity=cap,
                                last_closed_candle=int(d['times'][i]),last_candle_volume=float(d['par_volume'][i,j])))
    summaries=[dict(symbol=k[0],scenario=k[1],notional=k[2],counts=dict(v)) for k,v in sorted(counts.items())]
    for rel,h in hashes.items():assert sha(ROOT/rel)==h,rel
    save(out/'results.json',dict(funnels=funnels,summary=summaries,coverage=coverage,intent_coverage=dict(intent_views),rows=rows,
        independent_observations=False,observed_books_are_not_actual_own_fills=True,orders_sent=0,
        conclusion_limit='Candle volume is historical trade flow, not standing liquidity. Book-supported hypothetical fills still have latency/cancellation uncertainty. No book exists at most historical intent times, so these observations cannot certify a whole-period pair PnL.'))
    print(json.dumps(dict(funnels=funnels,summary=summaries,intent_coverage=dict(intent_views)),indent=2),flush=True)


if __name__=='__main__':main()
