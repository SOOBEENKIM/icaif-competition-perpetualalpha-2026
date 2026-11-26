"""Intersect two model-derived cost budgets without widening original limits."""
import numpy as np
from replay_lab.pullback.broker import execution_price
from experiments.paid_rehedge_pairs_20261017.policy import renewal_plan as base_renewal, exit_value
from experiments.remaining_value_pairs_20261017.policy import (
    held_state, anchor_stop, reconcile_adjustment, remaining_check, renewal_plan as veto_renewal,
)
from experiments.remaining_value_pairs_20261017.model import terminal_for_maturity


def intersect_limits(owned, target, marks, ids, original_limits, maximum_cost, p, sc):
    owned=np.asarray(owned,dtype=float);target=np.asarray(target,dtype=float);marks=np.asarray(marks,dtype=float)
    delta=target-owned;active=np.flatnonzero(abs(delta)>1e-10);fee=sc['fee_bps']/10000
    if not len(active):raise ValueError('Adjustment has no traded quantity')
    preview=marks.copy()
    for n in active:preview[n]=execution_price(marks[n],np.sign(delta[n]),sc['impact_bps'],p['execution']['decimals'][ids[n]])
    cost=float(delta@(preview-marks)+abs(delta)@preview*fee)
    if not np.isfinite(maximum_cost) or cost>=maximum_cost:
        return None
    slack=maximum_cost-cost;limits=preview.copy()
    for n in active:
        side=np.sign(delta[n]);distance=slack/len(active)/(abs(delta[n])*(1+side*fee))
        proposed=preview[n]+side*min(distance,preview[n]*.99)
        old=float(original_limits[str(ids[n])])
        limits[n]=min(old,proposed) if side>0 else max(old,proposed)
        if side*(limits[n]-preview[n]) < -1e-8:return None
        assert side*(limits[n]-old)<=1e-8
    worst=float(delta@(limits-marks)+abs(delta)@limits*fee)
    assert worst<=maximum_cost+1e-8
    return dict(entry_limits={str(ids[n]):float(limits[n]) for n in active},
                preview_cost=cost,worst_cost=worst,maximum_cost=maximum_cost,
                original_limits=original_limits.copy())


def renewal_plan(pair,owned_qty,i,d,de,cfg,p,sc,account_qty,equity,*,risk_blocked=False,ledger=None):
    if cfg.get('remaining_value',{}).get('mode')!='budget':
        return veto_renewal(pair,owned_qty,i,d,de,cfg,p,sc,account_qty,equity,risk_blocked=risk_blocked,ledger=ledger)
    original=base_renewal(pair,owned_qty,i,d,de,cfg,p,sc,account_qty,equity,risk_blocked=risk_blocked,ledger=ledger)
    if original is None or not original['accepted']:return original
    now=int(d['times'][i]);ids=pair['ids']
    forecast=terminal_for_maturity(de,i,pair['pair_index'],now,original['new_maturity'])
    if forecast is None:return dict(original,remaining_model_available=False)
    owned=np.asarray(owned_qty)[ids];marks=d['par_close'][i,ids]
    target=np.asarray(original['target']) if original.get('requires_adjustment') else owned
    value=exit_value(target,marks,forecast['terminal_prices'],ids,i,d,original['new_maturity'],p,sc)
    immediate=exit_value(owned,marks,forecast['terminal_prices'],ids,i,d,original['new_maturity'],p,sc)['immediate_close_cash']
    maximum_cost=value['expected_hold_cash']-immediate-original['buffer_cash']
    info=dict(remaining_model_available=True,joint_terminal_prices=forecast['terminal_prices'].tolist(),
              joint_horizon_hours=forecast['horizon_hours'],joint_fit_time=forecast['fit_time'],joint_source=forecast['source'],
              joint_hold_cash=value['expected_hold_cash'],joint_owned_qty=owned.tolist(),joint_target=target.tolist(),
              joint_maximum_adjustment_cost=maximum_cost,joint_immediate_close_cash=immediate)
    budget=None
    if original.get('requires_adjustment'):
        budget=intersect_limits(owned,target,marks,ids,original['entry_limits'],maximum_cost,p,sc)
        reject=budget is None
    else:
        reject=not np.isfinite(maximum_cost) or maximum_cost<=0
    if reject:
        return dict(accepted=False,reason='remaining_budget_cannot_fund_preview',old_maturity=original['old_maturity'],
                    proposed_maturity=original['new_maturity'],buffer_cash=original['buffer_cash'],**info)
    if budget is None:return dict(original,**info,joint_advantage_at_sealed_limits=maximum_cost+original['buffer_cash'])
    original_advantage=original['expected_hold_cash']-budget['worst_cost']-original['immediate_close_cash']
    assert original_advantage>=original['buffer_cash']-1e-8
    info.update(joint_advantage_at_sealed_limits=value['expected_hold_cash']-budget['worst_cost']-immediate,
                joint_worst_adjustment_cost=budget['worst_cost'],joint_preview_cost=budget['preview_cost'],
                original_entry_limits=budget['original_limits'],original_advantage_at_joint_limits=original_advantage)
    return dict(original,entry_limits=budget['entry_limits'],**info)
