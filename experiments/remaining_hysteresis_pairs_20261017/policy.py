"""A paid holding has a separate exit threshold from a new adjustment."""
import numpy as np
from experiments.remaining_budget_pairs_20261017.policy import (
    held_state, anchor_stop, remaining_check, renewal_plan,
)
from experiments.remaining_value_pairs_20261017.policy import reconcile_adjustment as budget_reconcile
from experiments.paid_rehedge_pairs_20261017.policy import reconcile_adjustment as original_reconcile,exit_value


def reconcile_adjustment(pair,ledger,i,d,de,cfg,p,sc,account_qty,equity,*,all_outcomes_observed,risk_blocked=False):
    if not cfg.get('remaining_value',{}).get('arrival_hysteresis',False):
        return budget_reconcile(pair,ledger,i,d,de,cfg,p,sc,account_qty,equity,
                                all_outcomes_observed=all_outcomes_observed,risk_blocked=risk_blocked)
    plan=pair.get('rehedge_plan',{}).copy()
    result=original_reconcile(pair,ledger,i,d,de,cfg,p,sc,account_qty,equity,
                              all_outcomes_observed=all_outcomes_observed,risk_blocked=risk_blocked)
    if result['outcome']!='renewed' or 'joint_terminal_prices' not in plan:return result
    ids=pair['ids'];q=ledger.qty[ids];marks=d['par_close'][i,ids]
    value=exit_value(q,marks,np.array(plan['joint_terminal_prices']),ids,i,d,plan['new_maturity'],p,sc)
    advantage=value['expected_hold_cash']-value['immediate_close_cash']
    gross=float(abs(q)@marks)
    exit_buffer=gross*cfg['remaining_value']['exit_buffer_bps']/10000
    acceptance_buffer=gross*cfg['renewal_policy']['incremental_buffer_bps']/10000
    info=dict(joint_arrival_advantage=advantage,joint_arrival_exit_threshold=-exit_buffer,
              joint_pretrade_acceptance_buffer=acceptance_buffer,paid_costs_recharged=0)
    if not np.isfinite(advantage) or advantage < -exit_buffer:
        pair.update(closing=True,rehedge_failed=True)
        return dict(result,event='rehedge_unwind',outcome='unwind',reason='arrival_joint_remaining_cash_insufficient',**info)
    return dict(result,**info)
