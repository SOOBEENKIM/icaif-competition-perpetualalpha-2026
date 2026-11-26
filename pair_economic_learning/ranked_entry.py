"""Rank only already admissible, cost-positive, physically feasible proposals."""
import numpy as np
from experiments.decision_gate_pairs_20261017.entry import proposals as original


def proposals(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,*,risk_qty=None):
    choices=original(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,risk_qty=risk_qty)
    if 'ranking_score' not in de:return choices
    allowed=[]
    for x in choices:
        score=float(de['ranking_score'][i,x['pair_index']])
        if np.isfinite(score):allowed.append(dict(x,ranking_score=score))
    return sorted(allowed,key=lambda x:(-x['ranking_score'],-x['expected_cash'],x['pair']))
