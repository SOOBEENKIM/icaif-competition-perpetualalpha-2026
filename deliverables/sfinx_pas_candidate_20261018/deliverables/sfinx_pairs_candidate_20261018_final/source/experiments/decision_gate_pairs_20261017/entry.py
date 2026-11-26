"""Filter only new proposals; retain original forecasts, prices and sizes."""
from experiments.dense_history_pairs_20261017.policy import proposals as original_proposals


def filter_proposals(choices, actions, available):
    return [p for p in choices if available[p['pair_index']] and actions[p['pair_index']]==p['direction']]


def proposals(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,*,risk_qty=None):
    choices=original_proposals(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,risk_qty=risk_qty)
    if 'economic_action' not in de:return choices
    return filter_proposals(choices,de['economic_action'][i],de['economic_available'][i])
