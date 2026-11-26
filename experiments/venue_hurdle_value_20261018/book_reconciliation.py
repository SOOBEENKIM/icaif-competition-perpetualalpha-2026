"""Reproduce fixed old-book entries, then observe bounded paid repair/unwind."""
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from .dataset import ROOT,HERE,OUT,setup,save,sha,verify,protected
from experiments.venue_execution_value_20261018.book_completion import OriginalLimitSession
from experiments.basis_carry_20261017.probe import load_tape,asof
from experiments.published_pair_lifecycle_20261017.account import BookAccount


def main():
    dest=OUT/'book_reconciliation';dest.mkdir(exist_ok=False)
    old=json.loads((OUT/'books/results.json').read_text())
    spec,cfg,p,d,de=setup()
    settings=ROOT/'experiments/shared_execution_pairs_20261017/book_lifecycle_settings.json'
    books=json.loads(settings.read_text());books['completion_window_seconds']=300
    sources=json.loads((OUT/'books/frozen.json').read_text())['sources'].copy()
    files=[Path(__file__),ROOT/'experiments/venue_execution_value_20261018/book_completion.py',OUT/'books/results.json']
    sources.update({str(f.relative_to(ROOT)):sha(f) for f in files});verify(sources)
    save(dest/'frozen.json',dict(sources=sources,at_unix=time.time(),maximum_observation_seconds=60,
        interpretation='Event-driven entry repair only; not a recalibration of the 5-minute model or a horizon PnL backtest.'))
    loaded={};rows=[]
    for trial in old['trials']:
        tape=trial['tape']
        if tape not in loaded:loaded[tape]=load_tape(ROOT/tape)
        streams,metadata=loaded[tape];now=trial['time'];intent=trial['plan'];i=int(np.searchsorted(d['times'],now))
        marks=d['par_close'][i].copy()
        for symbol,j in zip(intent['pair'],intent['ids']):
            quote=asof(streams[symbol],now*1000,books)
            marks[j]=float(quote['ctx'].get('markPx',quote['mid']))
        account=BookAccount(p['initial'],p['symbols'])
        def no_unobserved_funding(at,ledger):assert int(at)//3600000==now//3600
        session=OriginalLimitSession(account,intent,books,p,books['scenarios'][trial['scenario']],on_time=no_unobserved_funding)
        session.enter(now,streams,marks,d['sigma'][i])
        assert account.physical.fills==trial['fills'],'Entry fill path changed'
        assert account.physical.fees==trial['fees']
        np.testing.assert_array_equal(account.owners['pair'].qty[intent['ids']],trial['qty'])
        while session.state in ('completion','trim','closing'):
            at=session.ready_at
            if at>now+60 or at*1000>metadata['end_ms']:break
            session.advance(at,streams,marks,d['sigma'][i])
        status={'open':'balanced_entry_maturity_exit_unobserved','closed':'aborted_entry_paid_flat',
            'unfilled':'observed_unfilled','reconcile_required':'unknown_order_outcome'}.get(session.state,'partial_still_unresolved')
        flat=not np.any(account.physical.qty)
        rows.append(dict(tape=tape,time=now,kind=trial['kind'],horizon=trial['horizon'],scenario=trial['scenario'],
            status=status,state=session.state,entry_exact=True,events=session.events,fills=account.physical.fills,
            qty=account.physical.qty.tolist(),fees=account.physical.fees,
            realized_cash_pnl=account.physical.cash-p['initial'] if flat and not account.unknown else None,
            maturity_exit_observed=False,financial_score=None,orders_sent=0))
    verify(sources)
    save(dest/'results.json',dict(rows=rows,statuses=dict(Counter(r['status'] for r in rows)),
        exact_prior_entries=len(rows),complete_horizon_exits=0,orders_sent=0,live_strategy_changed=False,
        protected=protected(),historical_independence=False,reserved_raw_read=False))
    print('DONE book reconciliation',len(rows),dict(Counter(r['status'] for r in rows)),flush=True)


if __name__=='__main__':main()
