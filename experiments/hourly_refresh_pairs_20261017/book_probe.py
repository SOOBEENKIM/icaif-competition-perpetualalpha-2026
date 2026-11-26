"""Actual old par books: fixed-source entry feasibility, not portfolio returns."""
import json
import math
import time
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.basis_carry_20261017.probe import load_tape, asof
from experiments.shared_execution_pairs_20261017.account_v2 import SharedAccount
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.par_direct_forecast_20261017.data import ROOT, OUT as INPUT, save, sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.dense_history_pairs_20261017.research import load
from experiments.forecast_validity_pairs_20261017.book_plan import plan_entry
from experiments.forecast_validity_pairs_20261017.book_session import FixedForecastEntrySession


HERE=Path(__file__).resolve().parent


def main():
    out=HERE/'artifacts/books';out.mkdir(exist_ok=False)
    main_results=json.loads((HERE/'artifacts/results.json').read_text())
    cfg=json.loads((HERE/'par_protocol.json').read_text())
    tape_cfg=json.loads((ROOT/'experiments/basis_carry_20261017/protocol.json').read_text())
    settings=ROOT/'experiments/shared_execution_pairs_20261017/book_lifecycle_settings.json'
    life=json.loads(settings.read_text());life['pairs']=cfg['dense_pairs']
    life['purpose']='Entry feasibility on disconnected old tapes; no synthetic 24/72h exits or account Q'
    life['maximum_entry_reconciliation_seconds']=30
    save(out/'protocol.json',life)
    d=load(INPUT/'inputs.npz');_,_,p=inputs(json.loads((HERE/'reference_protocol.json').read_text()))
    states={};paths=[HERE/'book_probe.py',ROOT/'experiments/forecast_validity_pairs_20261017/book_plan.py',ROOT/'experiments/forecast_validity_pairs_20261017/book_session.py',ROOT/'experiments/forecast_validity_pairs_20261017/test_books.py',
                     HERE/'policy.py',ROOT/'experiments/forecast_validity_pairs_20261017/book_test_results.txt',out/'protocol.json',settings,
                     HERE/'artifacts/frozen.json',HERE/'artifacts/selection.json',INPUT/'inputs.npz',
                     ROOT/'experiments/shared_execution_pairs_20261017/account_v2.py',
                     ROOT/'experiments/shared_execution_pairs_20261017/book_session.py',
                     ROOT/'experiments/basis_carry_20261017/probe.py',
                     ROOT/'experiments/basis_carry_20261017/protocol.json']
    for kind in ['ridge','bias','affine','recent']:
        parent=HERE/'prepared'
        for h in [24,72]:
            path=parent/f'par_{kind}_{h}_state.npz';paths.append(path)
            states[kind,h,1]=load(path)
    paths.extend(ROOT/t for t in tape_cfg['tapes'])
    digests={str(path.relative_to(ROOT)):sha(path) for path in paths}
    save(out/'frozen.json',dict(at_unix=time.time(),sources=digests,protected=protected(),
         previous_account_selection=main_results['selection'],historical_independence=False,orders_sent=0))
    trials=[];summaries=[];coverage=[]
    for tape in tape_cfg['tapes']:
        streams,meta=load_tape(ROOT/tape);coverage.append(meta)
        cadence=cfg['signal_cadence_seconds'];lo=math.ceil(meta['start_ms']/1000/cadence)*cadence
        for (kind,h,ttl),de in states.items():
            for scenario,sc in life['scenarios'].items():
                options={**life,'book_intent_lifetime_hours':ttl,'funding_forecast_hours':h}
                count=Counter();attempts=Counter();trial=None
                for now in range(lo,meta['end_ms']//1000+1,cadence):
                    i=int(np.searchsorted(d['times'],now,side='right')-1)
                    if i<0 or now-d['times'][i]>=300:
                        count['missing_asof_risk_state']+=1;continue
                    marks=d['par_close'][i];sigma=d['sigma'][i];choices=[]
                    count['scheduled_decisions']+=1
                    for k,pair in enumerate(cfg['dense_pairs']):
                        if not de['directions'][i,k]:
                            count['source_not_valid']+=1;continue
                        ids=list(map(int,de['pairs'][k]));source=int(de['signal_times'][i,k])
                        assert [p['symbols'][j] for j in ids]==pair
                        if attempts[k,source]>=cfg['maximum_zero_fill_attempts']:
                            count['zero_fill_attempt_limit']+=1;continue
                        quotes={s:asof(streams[s],now*1000,options) if s in streams else None for s in pair}
                        anchors=de['source_prices'][i,ids];mu=float(de['mu'][i,k])
                        for direction in [-1,1]:
                            intent=dict(pair=pair,ids=ids,direction=direction,source_beta=float(de['beta'][i,k]),
                                        source_prices=anchors.tolist(),terminal_prices=(anchors*np.array([1+mu,1-mu])).tolist(),
                                        forecast_source=source,horizon=h,account_marks=marks.tolist())
                            plan=plan_entry(intent,quotes,now,options,p,sc,np.zeros(10),1e6,sigma)
                            count[plan['reason']]+=1
                            if plan['accepted']:choices.append((plan,intent,k))
                    if not choices:continue
                    plan,intent,k=max(choices,key=lambda c:c[0]['expected_net_cash'])
                    # Selection above uses current books only. Arrival books are
                    # examined exclusively by the delayed execution adapter.
                    account=SharedAccount(1e6,p['symbols'])
                    def no_unobserved_settlement(at,account):
                        if int(at)//3600000!=now//3600:
                            raise ValueError('Entry-only probe cannot skip an intervening funding settlement')
                    session=FixedForecastEntrySession(account,intent,options,p,sc,on_time=no_unobserved_settlement)
                    session.enter(now,streams,marks,sigma)
                    assert session.plan['target']==plan['target']
                    if session.state=='rejected':raise AssertionError('Sealed entry plan changed before execution')
                    while session.state in ['completion','trim','closing']:
                        at=session.ready_at
                        if at>now+life['maximum_entry_reconciliation_seconds'] or at*1000>meta['end_ms']:break
                        session.advance(at,streams,marks,sigma)
                    status={'open':'open_maturity_exit_unobserved','reconcile_required':'unknown_order_outcome',
                            'closed':'entry_aborted_and_paid_flat','unfilled':'entry_zero_fill'}.get(session.state,'partial_inventory_unresolved')
                    last=max([f['time'] for f in account.physical.fills]+[now])
                    snapshots=session.books(last,streams);final_marks=session.marks(snapshots,marks)
                    equity=account.reconcile(final_marks)
                    # There is no scheduled renewal/maturity in these short
                    # entry observation tapes; do not invent an exit at tape end.
                    maturity=intent['forecast_source']+h*3600
                    assert meta['end_ms']/1000<maturity-6*3600
                    trial=dict(tape=tape,model=kind,horizon=h,validity_hours=ttl,source_cadence_hours=1,scenario=scenario,
                               decision_time=now,source_time=intent['forecast_source'],source_age_hours=(now-intent['forecast_source'])/3600,
                               maturity=maturity,pair=intent['pair'],direction=intent['direction'],plan=plan,status=status,
                               end_state=session.state,events=session.events,known_fills=account.physical.fills,
                               known_fees=account.physical.fees,known_funding=account.physical.funding,
                               qty=account.physical.qty.tolist(),residual_gross=float(abs(account.physical.qty)@final_marks),
                               closed_abort_net_pnl=equity-1e6 if session.state=='closed' else None,
                               account_return=None,financial_score=None,future_exit_observed=False,orders_sent=0)
                    trials.append(trial);count[status]+=1
                    if session.state=='unfilled':
                        attempts[k,intent['forecast_source']]+=1;continue
                    # One account would own this pair or await reconciliation.
                    # Do not count later overlapping entries as new independent opportunities.
                    break
                summaries.append(dict(tape=tape,model=kind,horizon=h,validity_hours=ttl,source_cadence_hours=1,scenario=scenario,counts=dict(count)))
        print('book tape',len(coverage),'trials',len(trials),flush=True)
    for rel,digest in digests.items():assert sha(ROOT/rel)==digest,rel
    save(out/'results.json',dict(trials=trials,summaries=summaries,coverage=coverage,
         aggregate_statuses=dict(Counter(t['status'] for t in trials)),flat_initial_probe_capital=1e6,
         not_native_strategy_portfolio=True,not_full_horizon_performance=True,account_return=None,financial_score=None,
         historical_independence=False,protected=protected(),orders_sent=0))
    print('DONE',len(trials),'entry probes',Counter(t['status'] for t in trials),flush=True)


if __name__=='__main__':main()
