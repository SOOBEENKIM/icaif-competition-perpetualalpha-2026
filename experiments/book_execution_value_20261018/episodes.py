"""Observed entry-and-repair outcomes; open exposure is never realized profit."""
import numpy as np
from experiments.venue_execution_value_20261018.book_completion import OriginalLimitSession
from experiments.published_pair_lifecycle_20261017.account import BookAccount
from experiments.basis_carry_20261017.probe import walk


def episode(plan,streams,quotes,now,end_ms,marks,sigma,p,sc,books,spec):
    account=BookAccount(p['initial'],p['symbols']);cfg=dict(books,completion_window_seconds=300)
    def no_settlement(at,ledger):
        if int(at)//3600000!=now//3600:raise ValueError('Unobserved funding boundary')
    session=OriginalLimitSession(account,plan,cfg,p,sc,on_time=no_settlement)
    cutoff=min(now+spec['maximum_reconciliation_seconds'],(now//3600+1)*3600-.001,end_ms/1000)
    worst_delay=(sc['latency_ms']+books['future_exit_observation_grace_ms'])/1000
    if now+worst_delay>cutoff:return None
    session.enter(now,streams,marks,sigma);initial_state=session.state
    while session.state in ('completion','trim','closing'):
        at=session.ready_at
        if at+worst_delay>cutoff:break
        session.advance(at,streams,marks,sigma)
    final=session.state
    outcome=2 if final=='open' else 1 if final=='closed' else 0 if final=='unfilled' else 3
    last=max([now]+[o.get('arrival_ms',e['time']*1000)/1000 for e in session.events if 'result' in e for o in e['result']['orders']])
    # An unresolved-at-cutoff label becomes available at the cutoff, never earlier.
    availability=cutoff if outcome==3 else last
    gross=plan['intended_gross'];owned=account.owners['pair'].qty[plan['ids']]
    fraction=float(abs(owned)@marks[plan['ids']]/gross) if outcome==2 else 0.
    cash=None;cost=None;ideal=None
    if outcome in (0,1):
        assert not np.any(account.physical.qty) and not account.unknown
        cash=account.physical.cash-p['initial'];cost=max(0.,-cash)/gross
    elif outcome==2:
        ideal=0.
        for symbol,j,q in zip(plan['pair'],plan['ids'],owned):
            source_price=walk(quotes[symbol]['book']['levels'][1 if q>0 else 0],abs(q),sc['depth_fraction'])
            if source_price is None:raise AssertionError('Reduced held size exceeds original depth')
            ideal+=q*(source_price-marks[j])+abs(q)*source_price*sc['fee_bps']/10000
        actual_entry_cost=p['initial']-account.physical.equity(marks)
        cost=max(0.,actual_entry_cost-ideal)/gross
    assert availability>now
    return dict(initial_state=initial_state,state=final,outcome=outcome,available_at=availability,
        held_fraction=fraction,entry_cost_return=cost,realized_cash_pnl=cash,ideal_entry_cost=ideal,
        fees=account.physical.fees,qty=account.physical.qty.tolist(),events=session.events,fills=account.physical.fills,
        complete_horizon_exit_observed=False,financial_score=None,orders_sent=0)
