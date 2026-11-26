"""Reassess held-pair value from fresh signals without increasing exposure."""
import numpy as np
from experiments.paid_rehedge_pairs_20261017.policy import exit_value


def reassess(state,pair,ledger,i,d,de,cfg,p,sc):
    if not cfg.get('early_value_exit') or state['reason'] is not None:return state
    now=int(d['times'][i]);k=pair['pair_index'];source=int(de['signal_times'][i,k]);ids=list(pair['ids'])
    if now-pair.get('first_fill',now)<3*3600:return state
    if any(pair.get(x,False) for x in ['closing','completing','repairing','rehedging']):return state
    if not de['directions'][i,k] or not source<=now<source+3600:return state
    if source<=pair.get('value_review_source',-1):return state
    previous=pair.get('value_review_source',-1);pair['value_review_source']=source
    anchors=de['source_prices'][i,ids];mu=float(de['mu'][i,k]);terminal=anchors*np.array([1+mu,1-mu])
    if not np.isfinite(terminal).all() or np.any(terminal<=0):
        pair['negative_value_reviews']=0;return state
    q=ledger.qty[ids];marks=d['par_close'][i,ids]
    value=exit_value(q,marks,terminal,ids,i,d,source+pair['horizon']*3600,p,sc)
    advantage=value['expected_hold_cash']-value['immediate_close_cash'];buffer=float(abs(q)@marks)*5/10000
    negative=advantage < -buffer
    count=(pair.get('negative_value_reviews',0)+1 if source-previous==3600 else 1) if negative else 0
    pair['negative_value_reviews']=count
    if count>=2:
        return dict(state,reason='fresh_forecast_negative_holding_value',review_source=source,negative_reviews=count,holding_advantage=float(advantage),exit_hysteresis_cash=buffer)
    return state
