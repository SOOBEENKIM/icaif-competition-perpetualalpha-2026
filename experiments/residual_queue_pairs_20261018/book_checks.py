"""Received-quote parity and explicitly forced exit diagnostics, not strategy ROI."""
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
from experiments.basis_carry_20261017.probe import load_tape
from experiments.forecast_validity_pairs_20261017.book_session import FixedForecastEntrySession
from experiments.published_pair_lifecycle_20261017.account import BookAccount
from experiments.dense_history_pairs_20261017.research import load
from .account import ResidualBookAccount
from .research import ROOT,HERE,OUT,GRAPH,PAR_INPUT,save,sha,verify,protected


class QueueSessionAccount(ResidualBookAccount):
    def seal(self,intent,p,sc):
        packet=copy.deepcopy(intent)
        packet['owner_targets']['residual']=np.zeros_like(self.physical.qty).tolist()
        return super().seal(packet,p,sc)


def main():
    out=OUT/'books';out.mkdir(exist_ok=False)
    original=json.loads((GRAPH/'books/results.json').read_text())
    trials=[t for t in original['trials'] if t['model']=='formation_graph']
    allowed=json.loads((ROOT/'experiments/basis_carry_20261017/protocol.json').read_text())['tapes']
    assert all(t['tape'] in allowed for t in trials)
    cfgpath=ROOT/'experiments/shared_execution_pairs_20261017/book_lifecycle_settings.json'
    cfg=json.loads(cfgpath.read_text());cfg.update(book_intent_lifetime_hours=1,funding_forecast_hours=24,maximum_entry_reconciliation_seconds=30)
    p=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['trading_protocol']
    data=load(PAR_INPUT/'inputs.npz')
    files=[Path(__file__),HERE/'account.py',HERE/'test_results.txt',cfgpath,GRAPH/'books/results.json',
           PAR_INPUT/'inputs.npz',*(ROOT/t for t in allowed)]
    sources={str(p.relative_to(ROOT)):sha(p) for p in files};verify(sources)
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),
       purpose='Forced 60-second liquidation tests reuse allowed quotes. They do not obey the strategy holding rule and cannot establish its profitability.'))
    tapes={t:load_tape(ROOT/t) for t in allowed};rows=[]
    for case,trial in enumerate(trials):
        now=trial['time'];streams,metadata=tapes[trial['tape']];sc=cfg['scenarios'][trial['scenario']]
        i=int(np.searchsorted(data['times'],now,side='right')-1);marks=data['par_close'][i];sigma=data['sigma'][i]
        sessions=[]
        def clock(at,account):
            if int(at)//3600000!=now//3600:raise ValueError('Diagnostic cannot invent an hourly funding settlement')
        for account in [BookAccount(1e6,p['symbols']),QueueSessionAccount(1e6,p['symbols'])]:
            if isinstance(account,QueueSessionAccount):account.start_episode(trial['intent'],now)
            session=FixedForecastEntrySession(account,trial['intent'],cfg,p,sc,on_time=clock)
            session.enter(now,streams,marks,sigma)
            while session.state in ('completion','trim','closing'):
                at=session.ready_at
                if at>now+30 or at*1000>metadata['end_ms']:break
                session.advance(at,streams,marks,sigma)
            assert session.events==trial['events']
            assert account.physical.fills==trial['known_fills']
            assert session.state==trial['state']
            sessions.append(session)
        baseline,queue=sessions;new=queue.account;old=baseline.account
        exit_at=max(s.ready_at for s in sessions)+60
        assert exit_at*1000+sc['latency_ms']+cfg['future_exit_observation_grace_ms']<metadata['end_ms']
        assert exit_at//3600==now//3600
        i=int(np.searchsorted(data['times'],exit_at,side='right')-1);marks=data['par_close'][i];sigma=data['sigma'][i]
        for session in sessions:session.submit(exit_at,np.zeros(2),{},streams,marks,sigma,'close')
        assert old.physical.fills==new.physical.fills
        np.testing.assert_array_equal(old.physical.qty,new.physical.qty)
        handoff=None;additional=[]
        if np.any(new.owners['pair'].qty) and not new.unknown:
            ready=max(s.ready_at for s in sessions)
            handoff=new.handoff(dict(pair=trial['intent']['pair'],ids=trial['intent']['ids'],closing=True,
                                      observed_close_attempt=True,balanced=True),marks,ready)
            # Both accounts submit the same further closing targets. Only
            # strategy ownership changes; received depth and fees must not.
            for attempt in range(3):
                at=max(ready,math.ceil(new.last_packet_arrival_ms/1000)+1)
                if at>exit_at+30 or at//3600!=now//3600 or at*1000+10000>metadata['end_ms']:break
                baseline.submit(at,np.zeros(2),{},streams,marks,sigma,'close')
                raw=dict(packet_id=f'residual-check-{case}-{attempt}',decision_time=at,source_close=at,source_available=at,
                    source_prices=marks.tolist(),owner_targets={k:np.zeros(10).tolist() for k in new.owners},risk_forced=True)
                packet=new.seal(raw,p,sc)
                result=new.execute_books(packet,streams,marks,sigma,p,sc,cfg,on_time=clock)
                assert new.physical.fills==old.physical.fills
                np.testing.assert_array_equal(new.physical.qty,old.physical.qty)
                additional.append(result)
                if new.unknown or not np.any(new.physical.qty):break
                ready=max(baseline.ready_at,math.ceil(new.last_packet_arrival_ms/1000)+1)
        new.reconcile(marks);old.reconcile(marks)
        assert abs(new.physical.equity(marks)-old.physical.equity(marks))<1e-8
        rows.append(dict(tape=trial['tape'],scenario=trial['scenario'],source_time=now,forced_exit_at=exit_at,
          original_entry_events_and_fills_exact=True,physical_exit_fills_exact=True,
          residual_handoff=handoff,additional_exits=additional,unknown=sorted(new.unknown),
          final_qty=new.physical.qty.tolist(),fees=new.physical.fees,
          diagnostic_closed_net_cash=new.physical.cash-1e6 if not np.any(new.physical.qty) and not new.unknown else None,
          full_strategy_return=None,full_strategy_q=None,episodes=new.episode_results(marks)))
    verify(sources)
    save(out/'results.json',dict(rows=rows,exact_entry_probes=len(rows),
       forced_exit_diagnostics=len(rows),post_handoff_diagnostics=sum(r['residual_handoff'] is not None for r in rows),
       scope='Forced short exits, not the strategy maturity. No new live orders, synthetic historical quotes or full-horizon profit claim.',
       protected=protected(),orders_sent=0,reserved_raw_read=False,historical_independence=False))
    print('BOOK',len(rows),'exact entry/exit probes;',sum(r['residual_handoff'] is not None for r in rows),'partial exits transferred',flush=True)


if __name__=='__main__':main()
