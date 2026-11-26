"""Cost current ownership versus another pair; never reuse its future quantity."""
import numpy as np
from experiments.decision_gate_pairs_20261017.entry import proposals
from experiments.dense_history_pairs_20261017.policy import usable_signals
from experiments.paid_rehedge_pairs_20261017.policy import exit_value
from replay_lab.common_comparison.engine import risk_breaches
from replay_lab.pullback.broker import execution_price
from experiments.quote_anchored_pairs_20261017.pricing import value_at_entry


def tighten_entry(proposal,marks,p,sc,cap_bps):
    result=dict(proposal);ids=proposal['ids'];q=np.asarray(proposal['target'])
    prices=np.asarray(marks)[ids]
    preview=np.array([execution_price(px,np.sign(z),sc['impact_bps'],p['execution']['decimals'][j])
                      for px,z,j in zip(prices,q,ids)])
    old=np.array([proposal['entry_limits'][str(j)] for j in ids])
    cap=preview*(1+np.sign(q)*cap_bps/10000)
    limits=np.where(q>0,np.minimum(old,cap),np.maximum(old,cap))
    assert np.all(np.sign(q)*(limits-old)<=1e-8)
    gross=float(abs(q)@prices)
    value=value_at_entry(q,prices,limits,proposal['expected_price_bps']/10000,
                        proposal['expected_funding_bps']*gross/10000,sc['fee_bps'],sc['impact_bps'])
    assert value['expected_net_cash']>=proposal['minimum_expected_cash_at_limits']-1e-8
    result.update(entry_limits={str(j):float(v) for j,v in zip(ids,limits)},
                  minimum_expected_cash_at_limits=value['expected_net_cash'],switch_limit_cap_bps=cap_bps)
    return result


def comparison(hold_cash, close_cash, new_net_at_limits, old_gross, new_gross,
               extra_close_bps, buffer_bps):
    reserve = old_gross * extra_close_bps / 10000
    switch_cash = close_cash - reserve + new_net_at_limits
    buffer = max(old_gross,new_gross) * buffer_bps / 10000
    advantage = switch_cash - hold_cash
    finite = np.isfinite([hold_cash,close_cash,new_net_at_limits,old_gross,new_gross,
                          reserve,switch_cash,buffer,advantage]).all()
    return dict(accepted=bool(finite and new_net_at_limits>0 and advantage>buffer),
                expected_hold_cash=float(hold_cash),immediate_close_cash=float(close_cash),
                extra_close_reserve=float(reserve),new_net_at_limits=float(new_net_at_limits),
                expected_switch_cash=float(switch_cash),advantage=float(advantage),
                buffer_cash=float(buffer),sunk_entry_costs_charged_again=0)


def switch_plan(pair,owned_qty,i,d,de,cfg,p,sc,account_qty,equity,base_target,used,locks):
    mode=cfg.get('opportunity_switch')
    if not mode or not pair.get('balanced') or any(pair.get(k) for k in ['closing','completing','repairing','rehedging']):
        return None
    now=int(d['times'][i]);ids=list(pair['ids']);k=int(pair['pair_index'])
    if now%cfg['signal_cadence_seconds'] or now-pair.get('first_fill',now)<mode['minimum_holding_seconds']:
        return None
    def reject(reason,**kw):
        return dict(accepted=False,reason=reason,time=now,held_pair=pair['pair'],**kw)
    if list(de['pairs'][k])!=ids or cfg['dense_pairs'][k]!=pair['pair']:
        raise ValueError('Held inventory identity differs from forecast pair')
    source=int(de['signal_times'][i,k]);h=pair['horizon'];maturity=source+h*3600
    if not de['directions'][i,k] or not source<=now<source+cfg['dense_policy']['source_lifetime_hours']*3600:
        return reject('held_forecast_unavailable_or_stale')
    hard=pair['first_fill']+cfg['renewal_policy']['maximum_holding_hours']*3600
    if maturity>hard:return reject('common_maturity_exceeds_holding_limit')
    marks=d['par_close'][i];q=np.asarray(owned_qty)[ids]
    ratios=q/np.asarray(pair['target'])
    if min(ratios)<=0 or abs(ratios[0]-ratios[1])/max(ratios)>cfg['pair_balance_tolerance']:
        return reject('held_inventory_unbalanced')
    anchor=de['source_prices'][i,ids];mu=float(de['mu'][i,k]);terminal=anchor*np.array([1+mu,1-mu])
    if not np.isfinite(terminal).all() or np.any(terminal<=0):return reject('held_terminal_unavailable')
    capacity=sc['participation']*d['par_volume'][i,ids]
    if np.any(d['par_trades'][i,ids]<=0) or not np.isfinite(capacity).all() or np.any(capacity+1e-10<abs(q)):
        return reject('insufficient_source_close_capacity')
    value=exit_value(q,marks[ids],terminal,ids,i,d,maturity,p,sc)
    if not np.isfinite(list(value.values())).all():return reject('held_value_unavailable')
    cap=mode['switch_slippage_cap_bps']
    old_gross=float(abs(q)@marks[ids]);close_reserve=old_gross*cap/10000
    post_equity=equity+value['immediate_close_cash']-close_reserve
    if not np.isfinite(post_equity) or post_equity<=0:return reject('invalid_post_close_equity')
    for mask in ([1,0],[0,1],[1,1]):
        remaining=np.asarray(account_qty).copy();remaining[ids]-=q*np.asarray(mask)
        if risk_breaches(remaining,post_equity,marks,d['sigma'][i],p['risk']):
            return reject('partial_close_global_risk')
    post_qty=np.asarray(account_qty)-np.asarray(owned_qty)
    directions=usable_signals(i,d,de,cfg,'fixed_delayed_immediate_complete_168',locks,used)
    directions[k]=0
    candidates=proposals(i,d,de,None,h,cfg,p,sc,np.zeros_like(account_qty),post_equity,
                        np.zeros_like(account_qty),directions,'first_dense_hourlyhedge_switch_24',risk_qty=post_qty)
    best=None;count=0
    for candidate in candidates:
        if candidate['forecast_source']!=source or candidate['horizon']!=h:continue
        candidate=tighten_entry(candidate,marks,p,sc,cap)
        new_ids=candidate['ids'];target=np.asarray(candidate['target'])
        # The base controller may also request a different target this round.
        risk_ok=True
        for mask in ([1,0],[0,1],[1,1]):
            full=np.asarray(base_target).copy();full[new_ids]+=target*np.asarray(mask)
            new_gross=float(abs(target)@marks[new_ids])
            eq=post_equity-new_gross*2*(sc['fee_bps']+sc['impact_bps'])/10000
            if risk_breaches(full,eq,marks,d['sigma'][i],p['risk']):risk_ok=False;break
        if not risk_ok:continue
        count+=1
        result=comparison(value['expected_hold_cash'],value['immediate_close_cash'],
            candidate['minimum_expected_cash_at_limits'],old_gross,new_gross,
            cap,cfg['entry_buffer_bps'])
        if best is None or result['advantage']>best['advantage']:
            best=dict(**result,time=now,held_pair=pair['pair'],held_ids=ids,held_qty=q.tolist(),
                held_source=source,common_maturity=maturity,held_value=value,old_gross=old_gross,
                new_pair=candidate['pair'],new_pair_index=candidate['pair_index'],new_gross=new_gross,
                proposed_entry=candidate,post_flat_equity_estimate=post_equity,
                reason='positive_costed_switch_advantage' if result['accepted'] else 'switch_advantage_below_buffer')
    if best is None:return reject('no_cost_qualified_alternative')
    best['alternatives_compared']=count
    return best
