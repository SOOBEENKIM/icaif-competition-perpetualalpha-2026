"""Same historical quote observations, exact direct control, no fill claims."""
import json,time
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.basis_carry_20261017.probe import load_tape,asof
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,save,sha
from experiments.dense_history_pairs_20261017.research import load
from experiments.forecast_validity_pairs_20261017.book_plan import plan_entry
from experiments.liquidity_sized_pairs_20261017.top_of_book_bound import unit_bound
def terminal_prices(state,i,k,ids):
    mu=float(state["mu"][i,k]);return state["source_prices"][i,ids]*np.array([1+mu,1-mu])

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts/books'


def main():
    OUT.mkdir(exist_ok=False)
    previous=ROOT/'experiments/liquidity_sized_pairs_20261017/artifacts/top_of_book_bound/results.json'
    observations=json.loads(previous.read_text())['rows']
    allowed=json.loads((ROOT/'experiments/basis_carry_20261017/protocol.json').read_text())['tapes']
    assert all(r['tape'] in allowed for r in observations)
    cfg=json.loads((ROOT/'experiments/shared_execution_pairs_20261017/book_lifecycle_settings.json').read_text())
    cfg.update(book_intent_lifetime_hours=1,funding_forecast_hours=24)
    p=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['trading_protocol']
    paths={'direct':ROOT/'experiments/rolling_feature_pairs_20261017/artifacts/par_direct_24_state.npz',
           'raw':HERE/'artifacts/par_raw_state.npz'}
    states={k:load(path) for k,path in paths.items()};data=load(INPUT/'inputs.npz')
    sources={str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__),HERE/'model.py',previous,*paths.values(),INPUT/'inputs.npz',*(ROOT/t for t in allowed)]}
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,comparison='Same 68 directions per cost, 17 quote times, retained direct forecasts versus joint and common. Flat risk check. Visible order plan and unlimited-touch optimistic bound are diagnostics, not actual fills or independent trades.',reserved_raw_read=False))
    rows=[];exact=0
    for tape in allowed:
        streams,_=load_tape(ROOT/tape)
        for r in observations:
            if r['tape']!=tape:continue
            now=r['time'];i=int(np.searchsorted(data['times'],now,side='right')-1)
            ids=[p['symbols'].index(s) for s in r['pair']]
            quotes={s:asof(streams[s],now*1000,cfg) for s in r['pair']};assert all(q is not None for q in quotes.values())
            qq=[quotes[s] for s in r['pair']];marks=np.array([q['mid'] for q in qq]);sc=cfg['scenarios'][r['scenario']]
            for kind,state in states.items():
                k=next(k for k,pair in enumerate(state['pairs']) if list(pair)==ids)
                if not state['directions'][i,k]:
                    rows.append(dict(model=kind,tape=tape,time=now,scenario=r['scenario'],reason='unavailable_forecast'));continue
                source=int(state['signal_times'][i,k]);anchors=state['source_prices'][i,ids]
                terminal=terminal_prices(state,i,k,ids);beta=float(state['beta'][i,k]);remaining=24-(now-source)/3600
                intent=dict(pair=r['pair'],ids=ids,direction=r['direction'],source_beta=beta,source_prices=anchors.tolist(),terminal_prices=terminal.tolist(),forecast_source=source,horizon=24,account_marks=data['par_close'][i].tolist())
                plan=plan_entry(intent,quotes,now,cfg,p,sc,np.zeros(10),1e6,data['sigma'][i])
                ratio=beta*anchors[0]/anchors[1]*marks[1]/marks[0]
                signs=r['direction']*np.array([1.,-1.]);entry=[];exit_=[]
                for side,q in zip(signs,qq):
                    entry.append(float(q['book']['levels'][1 if side>0 else 0][0]['px']))
                    exit_.append(float(q['book']['levels'][0 if side>0 else 1][0]['px']))
                values=[]
                for relative in [.95,1.,1.05]:
                    w=signs*np.array([1.,ratio*relative]);w/=abs(w).sum()
                    values.append(unit_bound(w,marks,np.array(entry),np.array(exit_),terminal,
                                  np.array([q['oracle'] for q in qq]),np.array([float(q['ctx']['funding']) for q in qq]),
                                  remaining,sc['fee_bps'],sc['exit_impact_bps']))
                if kind=='direct':
                    np.testing.assert_allclose(values,r['bounds_bps'],rtol=0,atol=1e-10);exact+=1
                rows.append(dict(model=kind,tape=tape,time=now,pair=r['pair'],direction=r['direction'],scenario=r['scenario'],
                                 reason=plan['reason'],accepted=plan['accepted'],plan=plan,bounds_bps=values,
                                 exact_hedge_bound_bps=values[1],tolerance_best_bound_bps=max(values),
                                 terminal_returns=(terminal/anchors-1).tolist()))
    summaries=[]
    for kind in states:
        for scenario in cfg['scenarios']:
            rr=[r for r in rows if r['model']==kind and r['scenario']==scenario]
            a=np.array([r['tolerance_best_bound_bps'] for r in rr if 'tolerance_best_bound_bps' in r])
            summaries.append(dict(model=kind,scenario=scenario,count=len(rr),reasons=dict(Counter(r['reason'] for r in rr)),
                             optimistic_above_buffer=int((a>cfg['entry_buffer_bps']).sum()),
                             optimistic_quantiles_bps=np.quantile(a,[0,.5,.9,1]).tolist() if len(a) else None))
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path
    save(OUT/'results.json',dict(rows=rows,summary=summaries,exact_control_bound_observations=exact,actual_fills=0,complete_lifecycles=0,account_return=None,financial_score=None,orders_sent=0,reserved_raw_read=False))
    print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
