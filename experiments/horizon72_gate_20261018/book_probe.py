"""Matched 24h/72h entry economics on the five allowed historical quote tapes."""
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from experiments.basis_carry_20261017.probe import load_tape, asof
from experiments.forecast_validity_pairs_20261017.book_plan import plan_entry
from experiments.forecast_validity_pairs_20261017.book_session import FixedForecastEntrySession
from experiments.published_pair_lifecycle_20261017.account import BookAccount
from .research import ROOT, HERE, OUT, GATE, BUNDLE, load, plain, save, sha, verify, protected


def main():
    out=OUT/'books';out.mkdir(exist_ok=False)
    prior_path=GATE/'books/results.json';prior=json.loads(prior_path.read_text())
    observations=[r for r in prior['rows'] if r['model']=='direct']
    old_lookup={(r['model'],r['tape'],r['time'],tuple(r.get('pair',[])),r.get('direction'),r['scenario']):r for r in prior['rows']}
    allowed=json.loads((ROOT/'experiments/basis_carry_20261017/protocol.json').read_text())['tapes']
    cfg=json.loads((ROOT/'experiments/shared_execution_pairs_20261017/book_lifecycle_settings.json').read_text())
    cfg.update(book_intent_lifetime_hours=1,funding_forecast_hours=72,maximum_entry_reconciliation_seconds=30)
    trading=json.loads((BUNDLE/'config.json').read_text())['trading']
    data=load(BUNDLE/'data/par.npz')
    paths={'direct24':ROOT/'experiments/rolling_feature_pairs_20261017/artifacts/par_direct_24_state.npz',
           'gate24':GATE/'par_gate_state.npz','direct72':OUT/'par_direct_state.npz','gate72':OUT/'par_gate_state.npz'}
    states={kind:load(path) for kind,path in paths.items()}
    sources=json.loads((OUT/'frozen.json').read_text())['sources'].copy()
    files=[Path(__file__),OUT/'results.json',prior_path,*paths.values(),*(ROOT/t for t in allowed)]
    sources.update({str(path.relative_to(ROOT)):sha(path) for path in files});verify(sources)
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected()))
    rows=[];trials=[];exact=0
    for tape in allowed:
        streams,metadata=load_tape(ROOT/tape);choices=defaultdict(list)
        for observation in observations:
            if observation['tape']!=tape:continue
            now=observation['time'];i=int(np.searchsorted(data['times'],now,side='right')-1)
            ids=[trading['symbols'].index(s) for s in observation['pair']]
            quotes={symbol:asof(streams[symbol],now*1000,cfg) for symbol in observation['pair']}
            assert all(quote is not None for quote in quotes.values())
            scenario=observation['scenario'];sc=cfg['scenarios'][scenario]
            for kind,state in states.items():
                horizon=24 if kind.endswith('24') else 72
                k=next(k for k,pair in enumerate(state['pairs']) if list(pair)==ids)
                if not state['directions'][i,k]:
                    rows.append(dict(model=kind,tape=tape,time=now,pair=observation['pair'],
                        scenario=scenario,direction=observation['direction'],reason='unavailable_forecast',accepted=False))
                    continue
                source=int(state['signal_times'][i,k]);anchors=state['source_prices'][i,ids];mu=float(state['mu'][i,k])
                intent=dict(pair=observation['pair'],ids=ids,direction=observation['direction'],
                    source_beta=float(state['beta'][i,k]),source_prices=anchors.tolist(),
                    terminal_prices=(anchors*np.array([1+mu,1-mu])).tolist(),forecast_source=source,
                    horizon=horizon,account_marks=data['par_close'][i].tolist())
                original=plan_entry(intent,quotes,now,cfg,trading,sc,np.zeros(10),1e6,data['sigma'][i])
                plan=original
                if kind.startswith('gate') and (not state['economic_available'][i,k] or state['economic_action'][i,k]!=observation['direction']):
                    plan={**original,'accepted':False,'original_reason':original['reason'],'reason':'decision_gate_rejected'}
                if horizon==24:
                    key=('direct' if kind=='direct24' else 'gate',tape,now,tuple(observation['pair']),observation['direction'],scenario)
                    old=old_lookup[key]
                    assert plain(plan)==old['plan'],key
                    exact+=1
                row=dict(model=kind,tape=tape,time=now,pair=observation['pair'],scenario=scenario,
                    direction=observation['direction'],reason=plan['reason'],accepted=plan['accepted'],
                    plan=plain(plan),ungated_reason=original['reason'])
                rows.append(row)
                if horizon==72 and plan['accepted']:
                    choices[kind,scenario,now].append((plan,intent,i))
        # One first admissible entry probe per model/scenario/tape. Later
        # hypothetical flat-account proposals do not become extra actual trades.
        attempted=set()
        for (kind,scenario,now),options in sorted(choices.items(),key=lambda item:(item[0][0],item[0][1],item[0][2])):
            if (kind,scenario) in attempted:continue
            attempted.add((kind,scenario))
            plan,intent,i=max(options,key=lambda row:row[0]['expected_net_cash'])
            account=BookAccount(1e6,trading['symbols'])
            def settlement(at,account):
                if int(at)//3600000!=now//3600:
                    raise ValueError('Entry-only probe crossed an unobserved settled funding boundary')
            session=FixedForecastEntrySession(account,intent,cfg,trading,cfg['scenarios'][scenario],on_time=settlement)
            session.enter(now,streams,data['par_close'][i],data['sigma'][i])
            assert session.plan['target']==plan['target']
            while session.state in ('completion','trim','closing'):
                at=session.ready_at
                if at>now+cfg['maximum_entry_reconciliation_seconds'] or at*1000>metadata['end_ms']:break
                session.advance(at,streams,data['par_close'][i],data['sigma'][i])
            status={'open':'open_72h_exit_unobserved','closed':'paid_abort_flat',
                    'unfilled':'zero_fill','reconcile_required':'unknown_outcome'}.get(session.state,'partial_inventory')
            flat=not np.any(account.physical.qty)
            trials.append(dict(model=kind,tape=tape,scenario=scenario,time=now,intent=intent,plan=plain(plan),
                state=session.state,status=status,events=session.events,fills=account.physical.fills,
                qty=account.physical.qty.tolist(),fees=account.physical.fees,
                cash_pnl_if_flat=account.physical.cash-1e6 if flat else None,
                full_strategy_pnl=None,financial_score=None,complete_future_exit_observed=False))
    summaries=[]
    for kind in states:
        for scenario in cfg['scenarios']:
            selected=[row for row in rows if row['model']==kind and row['scenario']==scenario]
            summaries.append(dict(model=kind,scenario=scenario,observations=len(selected),
                accepted_plans=sum(row['accepted'] for row in selected),reasons=dict(Counter(row['reason'] for row in selected))))
    assert exact==272
    verify(sources)
    save(out/'results.json',dict(rows=rows,summaries=summaries,trials=trials,exact_24h_plans=exact,
        statuses=dict(Counter(t['model']+':'+t['status'] for t in trials)),complete_future_exit_observed=False,
        historical_independence=False,reserved_raw_read=False,orders_sent=0,protected=protected()))
    print('BOOKS',exact,'exact 24h plans;',len(trials),'first 72h entry probes',flush=True)


if __name__=='__main__':main()
