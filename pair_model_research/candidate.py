"""Reviewable all45 adaptive-relative-value forecast adapter; no order sender.

This is a candidate interface, NOT imported by the active live process. Live
promotion must explicitly wire its gate,72h horizon and expanded universe into
the coordinator; the old24h staging validator must not be silently reused.
"""
import numpy as np
from auto_pair_selection.selection import PAIRS,IDS,validate_snapshot
from .adaptive_hedge import fit_pair


def snapshot(data,source,received_at):
    t=validate_snapshot(data)
    if t[-1]!=source or not source<=received_at or received_at+5>=source+3600:
        raise ValueError('Only a current completed prefix with its actual receipt is allowed')
    day=source//86400*86400;index=int(np.searchsorted(t,day))
    if index>=len(t) or t[index]!=day:raise ValueError('Daily model source missing')
    logs=np.log(data['close']);active=(data['volume']>0)&(data['trades']>0);forecasts=[];audit=[]
    for k,(names,ids) in enumerate(zip(PAIRS,IDS)):
        model=fit_pair(logs,active,index,ids)
        if model is None:
            audit.append(dict(pair=list(names),pair_index=k,forecast_available=False,reason='past_relationship_or_history_not_admissible'))
            continue
        a,b=ids
        if not active[-1,a] or not active[-1,b]:
            audit.append(dict(pair=list(names),pair_index=k,forecast_available=False,reason='current_source_leg_not_fresh'))
            continue
        spread=logs[-1,a]-model['beta']*logs[-1,b]
        raw=(model['equilibrium']-spread)*(1-model['rho']**72)/(1+model['beta'])
        mu=float(np.clip(raw,-4*model['vol']*np.sqrt(72),4*model['vol']*np.sqrt(72)))
        action=int(np.sign(mu)) if abs(mu)>.0038 else 0
        anchors=data['close'][-1,list(ids)]
        forecasts.append(dict(model='adaptive_relative_ar72',pair=list(names),ids=list(ids),stable_pair_index=k,horizon=72,
          forecast_source=int(source),coefficient_fit_source=int(day),available_time=float(received_at+5),
          source_prices=anchors.tolist(),source_beta=model['beta'],source_vol=model['vol'],mu=mu,
          terminal_prices=(anchors*np.array([1+mu,1-mu])).tolist(),economic_available=True,economic_action=action,
          formula_family='symmetric_major_axis_hedge_AR1',estimated_half_life=model['half_life']))
        audit.append(dict(pair=list(names),pair_index=k,forecast_available=True,reason='forecast_only_execution_checks_pending' if action else 'proxy_cost_gate',economic_action=action))
    return dict(source=int(source),model_source=int(day),universe_size=45,forecasts=forecasts,pair_audit=audit,
                orders_sent=0,live_connected=False,mode='offline_candidate_snapshot')
