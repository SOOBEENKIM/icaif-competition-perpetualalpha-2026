"""Old-book entry audit of the separately fitted lifecycle value."""
import json,math,time
from collections import Counter
from pathlib import Path
from .model import feature,predict
import numpy as np
from experiments.basis_carry_20261017.probe import load_tape,asof
from experiments.forecast_validity_pairs_20261017.book_plan import plan_entry
from experiments.forecast_validity_pairs_20261017.book_session import FixedForecastEntrySession
from experiments.published_pair_lifecycle_20261017.account import BookAccount
from experiments.dense_history_pairs_20261017.research import load
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,save,sha
from experiments.complementary_alpha_20261017.research import protected

HERE=Path(__file__).resolve().parent


def main():
    out=HERE/'artifacts/books';out.mkdir(exist_ok=False)
    allowed=json.loads((ROOT/'experiments/basis_carry_20261017/protocol.json').read_text())['tapes']
    cfg=json.loads((ROOT/'experiments/shared_execution_pairs_20261017/book_lifecycle_settings.json').read_text())
    cfg.update(book_intent_lifetime_hours=1,funding_forecast_hours=24,maximum_entry_reconciliation_seconds=30)
    p=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['trading_protocol']
    data=load(INPUT/'inputs.npz')
    paths={'two_pair_gate':ROOT/'experiments/decision_gate_pairs_20261017/artifacts/par_gate_state.npz',
           'six_pair_gate':ROOT/'experiments/decision_sector_pairs_20261018/artifacts/par_sector_state.npz',
           'lifecycle_filter':HERE/'artifacts/par_state.npz',
           'lifecycle_rank':HERE/'artifacts/par_state.npz'}
    states={k:load(path) for k,path in paths.items()}
    values={int(k):v for k,v in json.loads((HERE/'artifacts/models.json').read_text()).items()}
    sources={str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__),HERE/'protocol.json',HERE/'model.py',HERE/'artifacts/models.json',HERE/'artifacts/evaluation_frozen_v2.json',HERE/'artifacts/results.json',INPUT/'inputs.npz',ROOT/'experiments/decision_sector_pairs_20261018/artifacts/books/results.json',*paths.values(),*(ROOT/t for t in allowed)]}
    save(out/'frozen.json',dict(sources=sources,at_unix=time.time(),orders_sent=0,protected=protected()))
    trials=[];summaries=[]
    for tape in allowed:
        streams,metadata=load_tape(ROOT/tape);cadence=cfg['signal_cadence_seconds']
        first=math.ceil(metadata['start_ms']/1000/cadence)*cadence
        for model,state in states.items():
            for scenario,sc in cfg['scenarios'].items():
                count=Counter();attempts=Counter()
                for now in range(first,metadata['end_ms']//1000+1,cadence):
                    i=int(np.searchsorted(data['times'],now,side='right')-1)
                    if i<0 or now-data['times'][i]>=300:count['missing_risk_state']+=1;continue
                    marks=data['par_close'][i];sigma=data['sigma'][i];choices=[];count['scheduled_decisions']+=1
                    for k,raw_ids in enumerate(state['pairs']):
                        if model=='lifecycle_filter' and k>=2:count['outside_two_pair_scope']+=1;continue
                        if not state['directions'][i,k]:count['invalid_source']+=1;continue
                        ids=list(map(int,raw_ids));pair=[p['symbols'][j] for j in ids];source=int(state['signal_times'][i,k])
                        if attempts[k,source]>=cfg['maximum_zero_fill_attempts']:count['zero_fill_attempt_limit']+=1;continue
                        anchors=state['source_prices'][i,ids];mu=state['mu'][i,k]
                        quotes={s:asof(streams[s],now*1000,cfg) if s in streams else None for s in pair}
                        for direction in (-1,1):
                            intent=dict(pair=pair,ids=ids,direction=direction,source_beta=float(state['beta'][i,k]),
                                source_prices=anchors.tolist(),terminal_prices=(anchors*np.array([1+mu,1-mu])).tolist(),
                                forecast_source=source,horizon=24,account_marks=marks.tolist())
                            plan=plan_entry(intent,quotes,now,cfg,p,sc,np.zeros(10),1e6,sigma)
                            if 'economic_action' in state:
                                count['ungated_reason:'+plan['reason']]+=1
                                if not state['economic_available'][i,k] or state['economic_action'][i,k]!=direction:
                                    plan={**plan,'accepted':False,'original_reason':plan['reason'],'reason':'decision_gate_rejected'}
                            count[plan['reason']]+=1
                            if plan['accepted'] and model.startswith('lifecycle_'):
                                value_model=values.get(now//86400*86400)
                                if value_model is None:
                                    count['value_unavailable_fallback']+=1
                                else:
                                    gross=float(abs(np.asarray(plan['target']))@marks[ids])
                                    proposal=dict(pair_index=k,ids=ids,target=plan['target'],horizon=24,direction=direction,
                                        expected_net_bps=plan['expected_net_cash']/gross*10000,
                                        expected_funding_bps=plan['expected_funding_cash']/gross*10000,
                                        source_beta=float(state['beta'][i,k]))
                                    x,unit,gross=feature(proposal,data,state,i,1e6)
                                    prediction=predict(value_model,x)
                                    plan=dict(plan,lifecycle_value=prediction*unit*gross,lifecycle_fit_time=value_model['fit_time'])
                                    if prediction<=0:
                                        count['lifecycle_value_rejected']+=1
                                        continue
                            if plan['accepted']:choices.append((plan,intent,k))
                    if not choices:continue
                    plan,intent,k=max(choices,key=lambda v:v[0].get('lifecycle_value',v[0]['expected_net_cash']) if model=='lifecycle_rank' else v[0]['expected_net_cash'])
                    account=BookAccount(1e6,p['symbols'])
                    def no_settlement(at,account):
                        if int(at)//3600000!=now//3600:raise ValueError('No unobserved funding in entry-only probe')
                    session=FixedForecastEntrySession(account,intent,cfg,p,sc,on_time=no_settlement)
                    session.enter(now,streams,marks,sigma)
                    assert session.plan['target']==plan['target']
                    while session.state in ('completion','trim','closing'):
                        at=session.ready_at
                        if at>now+cfg['maximum_entry_reconciliation_seconds'] or at*1000>metadata['end_ms']:break
                        session.advance(at,streams,marks,sigma)
                    status={'open':'open_maturity_exit_unobserved','closed':'entry_aborted_and_paid_flat',
                            'unfilled':'entry_zero_fill','reconcile_required':'unknown_order_outcome'}.get(session.state,'unresolved_partial_inventory')
                    trials.append(dict(tape=tape,model=model,scenario=scenario,time=now,intent=intent,plan=plan,
                        status=status,state=session.state,events=session.events,known_fills=account.physical.fills,
                        known_fees=account.physical.fees,qty=account.physical.qty.tolist(),
                        complete_future_exit_observed=False,account_return=None,financial_score=None,orders_sent=0))
                    count[status]+=1
                    if session.state=='unfilled':attempts[k,intent['forecast_source']]+=1;continue
                    break
                summaries.append(dict(tape=tape,model=model,scenario=scenario,counts=dict(count)))
    previous=json.loads((ROOT/'experiments/decision_sector_pairs_20261018/artifacts/books/results.json').read_text())
    original_trials=[t for t in trials if t['model'] in ('two_pair_gate','six_pair_gate')]
    assert original_trials==[r for r in previous['trials'] if r['model'] in ('two_pair_gate','six_pair_gate')],'Original old-book trials changed'
    old_summaries=[s for s in summaries if s['model'] in ('two_pair_gate','six_pair_gate')]
    assert old_summaries==[r for r in previous['summaries'] if r['model'] in ('two_pair_gate','six_pair_gate')],'Original quote opportunity counts changed'
    for rel,digest in sources.items():assert sha(ROOT/rel)==digest,rel
    save(out/'results.json',dict(trials=trials,summaries=summaries,aggregate_statuses=dict(Counter(t['status'] for t in trials)),
        exact_prior_trials=len(original_trials),exact_prior_summaries=len(old_summaries),not_full_horizon_performance=True,historical_independence=False,reserved_raw_read=False,
        full_portfolio_return=None,financial_score=None,orders_sent=0,protected=protected()))
    print('BOOKS',len(trials),dict(Counter((t['model'],t['status']) for t in trials)))


if __name__=='__main__':main()
