"""Re-execute prior accepted probes with the corrected common book adapter.

Only the five previously inspected old tapes are permitted. This tests the
effect of an execution correction; it does not turn short probes into a full
holding-period return or an independent performance sample.
"""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.basis_carry_20261017.probe import load_tape
from experiments.forecast_validity_pairs_20261017.book_session import FixedForecastEntrySession
from .account import BookAccount
from .forward import ROOT, HERE, save, sha


def main():
    parent=ROOT/'experiments/hourly_refresh_pairs_20261017'
    source=parent/'artifacts/books/results.json';prior=json.loads(source.read_text())
    cfg=json.loads((parent/'artifacts/books/protocol.json').read_text())
    allowed=json.loads((ROOT/'experiments/basis_carry_20261017/protocol.json').read_text())['tapes']
    p=json.loads((ROOT/'experiments/paired_venue_20261017/artifacts/forward/plan.json').read_text())['trading_protocol']
    with np.load(ROOT/'experiments/par_direct_forecast_20261017/artifacts/inputs.npz') as z:
        d={k:z[k].copy() for k in ['times','par_close','sigma']}
    states={};results=[];digests={str(source.relative_to(ROOT)):sha(source)}
    for tape in allowed:
        trials=[(i,t) for i,t in enumerate(prior['trials']) if t['tape']==tape]
        if not trials:continue
        streams,meta=load_tape(ROOT/tape);digests[tape]=sha(ROOT/tape)
        for index,trial in trials:
            model,h=trial['model'],trial['horizon'];key=model,h
            if key not in states:
                path=parent/'prepared'/f'par_{model}_{h}_state.npz'
                with np.load(path) as z:states[key]={k:z[k].copy() for k in z.files}
                digests[str(path.relative_to(ROOT))]=sha(path)
            state=states[key];now=trial['decision_time'];i=int(np.searchsorted(d['times'],now,side='right')-1)
            ids=[p['symbols'].index(s) for s in trial['pair']]
            k=next(k for k,v in enumerate(state['pairs']) if list(v)==ids)
            anchors=state['source_prices'][i,ids];mu=state['mu'][i,k]
            intent=dict(pair=trial['pair'],ids=ids,direction=trial['direction'],source_beta=float(state['beta'][i,k]),
                source_prices=anchors.tolist(),terminal_prices=(anchors*np.array([1+mu,1-mu])).tolist(),
                forecast_source=int(state['signal_times'][i,k]),horizon=h,account_marks=d['par_close'][i].tolist())
            options={**cfg,'book_intent_lifetime_hours':trial['validity_hours'],'funding_forecast_hours':h}
            account=BookAccount(1e6,p['symbols'])
            def no_settlement(at,account):
                if int(at)//3600000!=now//3600:raise ValueError('Old entry probe cannot skip funding settlement')
            session=FixedForecastEntrySession(account,intent,options,p,cfg['scenarios'][trial['scenario']],on_time=no_settlement)
            session.enter(now,streams,d['par_close'][i],d['sigma'][i])
            if session.plan['target']!=trial['plan']['target']:raise ValueError('Decision plan unexpectedly changed')
            while session.state in ('completion','trim','closing'):
                at=session.ready_at
                if at>now+cfg['maximum_entry_reconciliation_seconds'] or at*1000>meta['end_ms']:break
                session.advance(at,streams,d['par_close'][i],d['sigma'][i])
            old_fills=[(f['asset'],f['signed_qty'],f['price'],f['time']) for f in trial['known_fills']]
            new_fills=[(f['asset'],f['signed_qty'],f['price'],f['time']) for f in account.physical.fills]
            results.append(dict(prior_trial=index,model=model,horizon=h,scenario=trial['scenario'],tape=tape,
                original_state=trial['end_state'],corrected_state=session.state,
                fill_path_unchanged=old_fills==new_fills,old_fills=old_fills,new_fills=new_fills,
                old_fees=trial['known_fees'],new_fees=account.physical.fees,orders_sent=0))
    if len(results)!=len(prior['trials']):raise ValueError('Prior probe outside old tape allowlist')
    for rel,digest in digests.items():
        if sha(ROOT/rel)!=digest:raise ValueError('Prior evidence changed during audit')
    result=dict(trials=results,prior_entries=len(results),unchanged_fill_paths=sum(r['fill_path_unchanged'] for r in results),
        corrected_states=dict(Counter(r['corrected_state'] for r in results)),
        scope='Only previously accepted entry probes; no selection of new signals, no full future exits, no account PnL/Q.',
        sources=digests,reserved_raw_read=False,orders_sent=0)
    out=HERE/'artifacts';out.mkdir(exist_ok=True);save(out/'entry_adapter_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('trials','sources')}))


if __name__=='__main__':main()
