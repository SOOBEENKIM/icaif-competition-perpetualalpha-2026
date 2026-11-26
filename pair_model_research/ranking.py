"""Rank only already executable positive-net-cash choices by spread risk.

This changes selection, not price forecasts, fees, fills, or position caps.
The risk estimate is an ex-ante scale, not a guaranteed forecast interval.
"""
import numpy as np
from experiments.decision_gate_pairs_20261017.entry import proposals as original_proposals


def rank(choices,prices):
    result=[]
    for choice in choices:
        c=dict(choice);q=np.asarray(c['target']);gross=float(abs(q)@prices[c['ids']])
        risk=gross*float(c['entry_vol'])*np.sqrt(c['horizon'])
        if not np.isfinite(risk) or risk<=0:continue
        c['ranking_cash_risk']=risk;c['ranking_net_to_risk']=c['expected_net_cash']/risk
        result.append(c)
    return sorted(result,key=lambda c:(-c['ranking_net_to_risk'],-c['expected_net_cash'],c['pair']))


def proposals(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,*,risk_qty=None):
    choices=original_proposals(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,risk_qty=risk_qty)
    return rank(choices,d['par_close'][i])
