"""Optimistic size-independent economic bound on the same old quote samples.

Best touch for unlimited entry/exit volume is only an upper bound: never an
execution assumption. If it is below the buffer, depth sizing cannot rescue
that fixed forecast and hedge. Include the entire allowed ratio tolerance.
"""
import json,time
from pathlib import Path
import numpy as np
from experiments.basis_carry_20261017.probe import load_tape,asof
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,save,sha
from experiments.dense_history_pairs_20261017.research import load

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts/top_of_book_bound'

def unit_bound(weights,marks,entry,exit_,terminal,oracles,funding,remaining,fee_bps,exit_impact_bps):
    q=weights/marks
    price=float(q@(terminal-entry))
    carry=-q*oracles*funding*remaining;carry=float(np.where(carry>0,carry*.5,carry).sum())
    entry_fee=float(abs(q)@entry)*fee_bps/10000
    exit_cost=float(q@(marks-exit_))+(fee_bps+exit_impact_bps)/10000
    return (price+carry-entry_fee-exit_cost)*10000

def main():
    OUT.mkdir(exist_ok=False)
    audit_path=HERE/'artifacts/book_capacity_audit/results.json';audit=json.loads(audit_path.read_text())
    allowed=json.loads((ROOT/'experiments/basis_carry_20261017/protocol.json').read_text())['tapes']
    assert all(row['tape'] in allowed for row in audit['rows'])
    cfg=json.loads((ROOT/'experiments/shared_execution_pairs_20261017/book_lifecycle_settings.json').read_text())
    state_path=ROOT/'experiments/rolling_feature_pairs_20261017/artifacts/par_direct_24_state.npz'
    state=load(state_path);data=load(INPUT/'inputs.npz');rows=[]
    sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),audit_path,state_path,INPUT/'inputs.npz',*(ROOT/t for t in allowed)]}
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,buffer_bps=cfg['entry_buffer_bps'],
        bound='Unlimited best-touch size on both entry and reverse exit, same fee/funding/exit reserve as the book planner. Linear cash per gross; largest value over the exact unit hedge and +/-5% allowed ratio endpoints. This is an optimistic economic ceiling, not a fill or profit claim.',reserved_raw_read=False))
    for tape in allowed:
        streams,_=load_tape(ROOT/tape)
        for row in audit['rows']:
            if row['tape']!=tape:continue
            now=row['time'];i=int(np.searchsorted(data['times'],now,side='right')-1)
            symbols=row['pair'];pair_ids=[['BTC','ETH','HYPE','SP500','EWY','EWJ','BRENT','GOLD','USDEUR','USDJPY'].index(s) for s in symbols]
            k=next(k for k,ids in enumerate(state['pairs']) if list(ids)==pair_ids)
            quotes=[asof(streams[s],now*1000,cfg) for s in symbols];assert all(q is not None for q in quotes)
            marks=np.array([q['mid'] for q in quotes]);np.testing.assert_array_equal(marks,row['book_mid'])
            anchors=state['source_prices'][i,pair_ids];mu=state['mu'][i,k]
            terminal=anchors*np.array([1+mu,1-mu]);beta=state['beta'][i,k]*anchors[0]/anchors[1]*marks[1]/marks[0]
            source=int(state['signal_times'][i,k]);remaining=24-(now-source)/3600
            oracles=np.array([q['oracle'] for q in quotes]);funding=np.array([float(q['ctx']['funding']) for q in quotes])
            signs=row['direction']*np.array([1.,-1.]);entry=[];exit_=[]
            for side,quote in zip(signs,quotes):
                entry.append(float(quote['book']['levels'][1 if side>0 else 0][0]['px']))
                exit_.append(float(quote['book']['levels'][0 if side>0 else 1][0]['px']))
            sc=cfg['scenarios'][row['scenario']];values=[]
            for relative in [1-cfg['pair_balance_tolerance'],1.,1+cfg['pair_balance_tolerance']]:
                weights=signs*np.array([1.,beta*relative]);weights/=abs(weights).sum()
                values.append(unit_bound(weights,marks,np.array(entry),np.array(exit_),terminal,oracles,funding,remaining,sc['fee_bps'],sc['exit_impact_bps']))
            rows.append(dict(tape=tape,time=now,pair=symbols,scenario=row['scenario'],direction=row['direction'],
                planner_reason=row['planner_reason'],exact_hedge_optimistic_net_bps=values[1],
                tolerance_best_optimistic_net_bps=max(values),ratio_multipliers=[.95,1.,1.05],bounds_bps=values))
    summary={}
    for sc in cfg['scenarios']:
        rr=[r for r in rows if r['scenario']==sc];a=np.array([r['exact_hedge_optimistic_net_bps'] for r in rr]);b=np.array([r['tolerance_best_optimistic_net_bps'] for r in rr])
        summary[sc]=dict(observations=len(rr),exact_hedge_above_buffer=int((a>cfg['entry_buffer_bps']).sum()),
            any_allowed_ratio_above_buffer=int((b>cfg['entry_buffer_bps']).sum()),exact_quantiles_bps=np.quantile(a,[0,.5,.9,1]).tolist(),
            tolerance_best_quantiles_bps=np.quantile(b,[0,.5,.9,1]).tolist(),positive=[r for r in rr if r['tolerance_best_optimistic_net_bps']>cfg['entry_buffer_bps']])
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path
    save(OUT/'results.json',dict(rows=rows,summary=summary,not_used_to_select_existing_candidates=True,actual_fills=0,account_return=None,financial_score=None,orders_sent=0,reserved_raw_read=False))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
