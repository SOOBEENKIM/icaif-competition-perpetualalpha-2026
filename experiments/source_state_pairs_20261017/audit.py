import json
from pathlib import Path
import numpy as np
from strategy.agent import TAIL
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,save,sha
from experiments.basis_carry_20261017.probe import load_tape,asof
from .signals import reference_confirmation

HERE=Path(__file__).resolve().parent


def inputs():
    cfg=json.loads((HERE/'protocol.json').read_text())
    with np.load(INPUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
    with np.load(ROOT/'experiments/adaptive_mean_pairs_20261017/artifacts/state.npz') as z:de={k:z[k].copy() for k in z.files}
    with np.load(ROOT/cfg['reference_run']/'inputs.npz') as z:ref={k:z[k].copy() for k in z.files}
    return cfg,d,de,ref


def main():
    out=HERE/'audit_artifacts';out.mkdir(exist_ok=False);cfg,d,de,ref=inputs()
    revised=reference_confirmation(d,de,ref,cfg['entry_z']);hour=d['times']%3600==0
    paths=[HERE/'audit.py',HERE/'signals.py',HERE/'protocol.json',INPUT/'inputs.npz',ROOT/cfg['reference_run']/'inputs.npz',ROOT/'experiments/adaptive_mean_pairs_20261017/artifacts/state.npz']
    save(out/'frozen.json',dict(sources={str(p.relative_to(ROOT)):sha(p) for p in paths},orders_sent=0))
    rows=[]
    for k,pair in enumerate(cfg['pairs']):
        for kind in ('standard','delayed'):
            a=de[kind][hour,k];b=revised[kind][hour,k]
            rows.append(dict(pair=pair,kind=kind,par_hour_signals=int(np.count_nonzero(a)),reference_hour_signals=int(np.count_nonzero(b)),
                agreeing_nonzero=int(np.count_nonzero((a==b)&(a!=0))),different_hours=int(np.count_nonzero(a!=b))))
    p=json.loads((TAIL/'baseline.json').read_text())['protocol'];symbols=p['symbols'];activity=[]
    for symbol in ('BTC','ETH','HYPE'):
        j=symbols.index(symbol);active=d['par_trades'][:,j]>0
        last=None;ages=[]
        for t,ok in zip(d['times'],active):
            if ok:last=t
            if last is not None:ages.append((t-last)/3600)
        activity.append(dict(symbol=symbol,bars=len(active),active_bars=int(active.sum()),maximum_age_after_first_observed_trade=max(ages)))
    passive=ROOT/'experiments/pairs_maker_20261017/artifacts/results.json'
    views=[{k:q[k] for k in ('tape','symbol','decision','side','requested_qty','requested_price')} for q in json.loads(passive.read_text())['quotes'] if q['queue_model']=='all_displayed_ahead_no_cancellation_credit' and q['ttl_seconds']==120]
    tapes=json.loads((ROOT/'experiments/basis_carry_20261017/protocol.json').read_text())['tapes']
    life=json.loads((ROOT/'experiments/pair_execution_calibration_20261017/lifecycle_protocol.json').read_text())
    quote_rows=[]
    for tape in tapes:
        streams,_=load_tape(ROOT/tape)
        for q in views:
            if q['tape']!=tape:continue
            i=int(np.searchsorted(d['times'],q['decision']//1000,side='right')-1)
            for k,pair in enumerate(cfg['pairs']):
                if q['symbol'] not in pair:continue
                snaps=[asof(streams[s],q['decision'],life) if s in streams else None for s in pair]
                z=None
                if all(s is not None for s in snaps):
                    mids=[.5*(float(s['book']['levels'][0][0]['px'])+float(s['book']['levels'][1][0]['px'])) for s in snaps]
                    z=float((np.log(mids[0])-de['beta'][i,k]*np.log(mids[1])-de['mean'][i,k])/de['sd'][i,k])
                for name,state in [('par',de),('reference',revised)]:
                    direction=int(state['delayed'][i,k]);side=q['side']*(1 if q['symbol']==pair[0] else -1)
                    quote_rows.append(dict(source=name,tape=tape,quote_time=q['decision'],pair=pair,
                        fresh_book=z is not None,source_signal=direction,direction_matches=direction!=0 and direction==side,
                        current_book_spread_valid=bool(direction and z is not None and -cfg['entry_z']*cfg['stop_multiplier']<direction*z<0)))
    summary=[]
    for source in ('par','reference'):
        rr=[r for r in quote_rows if r['source']==source]
        summary.append(dict(source=source,quote_pair_views=len(rr),source_signal_views=sum(r['source_signal']!=0 for r in rr),
            direction_matches=sum(r['direction_matches'] for r in rr),valid_current_book=sum(r['direction_matches'] and r['current_book_spread_valid'] for r in rr)))
    save(out/'results.json',dict(signal_comparison=rows,activity=activity,quote_summary=summary,quote_rows=quote_rows,
        purpose='Causal feature/availability audit only, no fill outcomes or PnL selection.',orders_sent=0))
    print(json.dumps(dict(signals=rows,activity=activity,quotes=summary),indent=2),flush=True)


if __name__=='__main__':main()
