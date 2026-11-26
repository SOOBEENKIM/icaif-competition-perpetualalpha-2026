"""One bounded hedge retry valued against liquidating the actual inventory."""
import numpy as np
from experiments.dense_history_pairs_20261017.completion import completion_plan as original,after_completion
from experiments.dense_history_pairs_20261017.policy import held_state
from experiments.paid_rehedge_pairs_20261017.policy import exit_value
from experiments.retained_sizing_pairs_20261017.policy import variance_charge
from replay_lab.common_comparison.engine import rounded,risk_breaches
from replay_lab.pullback.broker import execution_price

def completion_plan(pair,ledger,i,d,de,cfg,p,sc,*,risk_ledger=None):
    if 'conditional_completion' not in cfg:
        return original(pair,ledger,i,d,de,cfg,p,sc,risk_ledger=risk_ledger)
    risk_ledger=ledger if risk_ledger is None else risk_ledger
    now=int(d['times'][i]);ids=list(pair['ids']);owned=ledger.qty[ids].copy();want=np.asarray(pair['target'])
    if now>=pair['completion_deadline'] or pair['forecast_source']>now or np.count_nonzero(owned)!=1:return None
    if held_state(pair,i,d,de,cfg)['reason'] is not None:return None
    have=int(np.flatnonzero(owned)[0]);missing=1-have;ratio=owned[have]/want[have]
    if not 0<ratio<=1+1e-8:return None
    target=want*ratio;target[have]=owned[have]
    target[missing]=rounded(target[missing],p['execution']['decimals'][ids[missing]])
    if target[missing]==0:return None
    ratios=target/want
    if min(ratios)<=0 or abs(ratios[0]-ratios[1])/max(ratios)>cfg['pair_balance_tolerance']:return None
    marks=d['par_close'][i];prices=marks[ids];j=ids[missing];delta=target-owned;qty=delta[missing];side=np.sign(qty)
    preview=execution_price(marks[j],side,sc['impact_bps'],p['execution']['decimals'][j])
    if abs(qty)*preview<p['execution']['minimum_order_value']:return None
    equity=risk_ledger.equity(marks)
    if not np.isfinite(equity) or equity<=0:return None
    maturity=pair['forecast_source']+pair['horizon']*3600
    terminal=np.asarray(pair['terminal_prices']);gross=float(abs(target)@prices)
    if maturity<=now or not np.isfinite(terminal).all() or np.any(terminal<=0):return None
    continuation=exit_value(target,prices,terminal,ids,i,d,maturity,p,sc)
    immediate=exit_value(owned,prices,terminal,ids,i,d,maturity,p,sc)['immediate_close_cash']
    gamma=cfg['conditional_completion']['risk_aversion']
    risk=variance_charge(target,owned,risk_ledger.qty,ids,marks,d['sigma'][i],equity,(maturity-now)/3600,gamma)
    fee=sc['fee_bps']/10000;buffer=gross*cfg['entry_buffer_bps']/10000
    def cost(price):return float(qty*(price-marks[j])+abs(qty)*price*fee)
    cash_advantage=continuation['expected_hold_cash']-cost(preview)-immediate
    utility_advantage=cash_advantage-risk
    slack=min(cash_advantage,utility_advantage)-buffer
    if not np.isfinite(slack) or slack<=0:return None
    distance=min(slack/(abs(qty)*(1+side*fee)),preview*.99)
    limit=preview+side*distance
    worst_cash=continuation['expected_hold_cash']-cost(limit)-immediate
    worst_utility=worst_cash-risk
    assert min(worst_cash,worst_utility)>=buffer-1e-8
    full=risk_ledger.qty.copy();full[ids]+=delta
    if risk_breaches(full,equity-cost(limit),marks,d['sigma'][i],p['risk']):return None
    return dict(target=target.tolist(),missing_asset=j,entry_limits={str(j):float(limit)},preview_price=float(preview),
        conditional_completion=True,risk_aversion=gamma,held_qty=owned.tolist(),
        immediate_close_cash=immediate,completion_cash=continuation['expected_hold_cash']-cost(preview),
        cash_advantage=cash_advantage,utility_advantage=utility_advantage,risk_charge=risk,
        minimum_cash_advantage_at_limit=worst_cash,minimum_utility_advantage_at_limit=worst_utility,
        buffer_cash=buffer,missing_leg_cost_preview=cost(preview),sunk_entry_costs_charged_again=0,
        expected_future_funding=continuation['expected_funding_cash'],exit_reserve=continuation['estimated_future_exit_cost'],
        frozen_terminal_prices=terminal.tolist(),deadline=pair['completion_deadline'],maturity=maturity)
