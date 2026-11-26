"""Fixed-maturity cash forecasts, independent of rolling z-score exits."""
import numpy as np
from experiments.adaptive_mean_pairs_20261017.signals import held_state as original_held
from experiments.persistent_intent_pairs_20261017.signals import usable_signals as original_signals
from experiments.shared_execution_pairs_20261017.proposals import proposals as original_proposals
from experiments.quote_anchored_pairs_20261017.pricing import reservation_limits
from replay_lab.common_comparison.engine import rounded,risk_breaches
from replay_lab.pullback.broker import execution_price


def held_state(pair,i,d,de,cfg):
    if not pair.get('dense'):return original_held(pair,i,d,de,cfg)
    now=int(d['times'][i]);step=int(d['times'][1]-d['times'][0]);maturity=pair['forecast_source']+pair['horizon']*3600
    reason='forecast_maturity' if now>=maturity-step else None
    value=None;bound=cfg['dense_policy']['stop_sigma']*pair['entry_vol']*np.sqrt(pair['horizon'])
    if 'risk_entry_prices' in pair:
        q=np.array(pair['risk_quantities']);entry=np.array(pair['risk_entry_prices'])
        value=float(q@(d['par_close'][i,pair['ids']]-entry)/(abs(q)@entry))
        if reason is None and value<=-bound:reason='pair_return_stop'
    return dict(reason=reason,age_hours=(now-pair.get('first_fill',pair['forecast_source']))/3600,
        maturity=maturity,pair_return=value,return_stop_bound=bound)


def usable_signals(i,d,de,cfg,arm,locks,used):
    if not de.get('dense',False):return original_signals(i,d,de,cfg,arm,locks,used)
    directions=de['directions'][i].copy()
    for k in range(len(directions)):
        if int(de['signal_times'][i,k])<=used.get(k,-1):directions[k]=0
    return directions


def proposals(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,*,risk_qty=None):
    if not de.get('dense',False):return original_proposals(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,risk_qty=risk_qty)
    risk_qty=qty if risk_qty is None else risk_qty
    now=int(d['times'][i]);price=d['par_close'][i];sigma=d['sigma'][i];results=[];cost=2*(sc['fee_bps']+sc['impact_bps'])/10000
    for k,ids in enumerate(de['pairs']):
        if not directions[k]:continue
        ids=list(ids);source=int(de['signal_times'][i,k]);remaining=h-(now-source)/3600
        assert source<=now
        if remaining<=0 or np.any(qty[ids]) or np.any(abs(reserved[ids])>1e-12):continue
        beta=float(de['beta'][i,k]);mu=float(de['mu'][i,k]);anchor=de['source_prices'][i,ids]
        if not np.isfinite([beta,mu]).all() or not .1<=beta<=10:continue
        terminal=anchor*np.array([1+mu,1-mu])
        if not np.isfinite(terminal).all() or np.any(terminal<=0):continue
        venue_beta=beta*anchor[0]/anchor[1]*price[ids[1]]/price[ids[0]]
        for direction in (-1,1):
            w=direction*np.array([1.,-venue_beta])/(1+venue_beta)
            gross=min(cfg['pair_max_gross'],float(np.min(sc['participation']*d['par_volume'][i,ids]*price[ids]/abs(w))))
            def allowed(g):
                for mask in ([1,0],[0,1],[1,1]):
                    q=risk_qty.copy();q[ids]+=g*w/price[ids]*mask
                    if risk_breaches(q,equity-g*cost,price,sigma,p['risk']):return False
                return True
            if not allowed(gross):
                lo,hi=0.,gross
                for _ in range(35):
                    mid=(lo+hi)/2
                    if allowed(mid):lo=mid
                    else:hi=mid
                gross=lo
            q=np.array([rounded(gross*z/price[j],p['execution']['decimals'][j]) for j,z in zip(ids,w)])
            if np.any(abs(q)*price[ids]<p['execution']['minimum_order_value']):continue
            actual_beta=abs(q[1])*price[ids[1]]/(abs(q[0])*price[ids[0]])
            if abs(actual_beta/venue_beta-1)>cfg['pair_balance_tolerance']:continue
            preview=np.array([execution_price(price[j],np.sign(z),sc['impact_bps'],p['execution']['decimals'][j]) for j,z in zip(ids,q)])
            carry=-q*d['reference_close'][i,ids]*d['known_hourly_funding'][i,ids]*remaining
            carry=float(np.where(carry>0,carry*.5,carry).sum())
            gross=float(abs(q)@price[ids]);edge=float(q@(terminal-price[ids]))/gross
            budget=reservation_limits(q,price[ids],preview,edge,carry,sc['fee_bps'],sc['impact_bps'],cfg['entry_buffer_bps'],cfg['entry_slippage_cap_bps'])
            if budget is None:continue
            source_gross=float(abs(q)@anchor)
            results.append(dict(dense=True,pair=cfg['dense_pairs'][k],pair_index=k,ids=ids,target=q.tolist(),horizon=h,
                beta=float(venue_beta),source_beta=beta,direction=direction,entry_vol=float(de['vol'][i,k])*source_gross/gross,
                expected_net_bps=budget['expected_net_cash']/gross*10000,expected_price_bps=edge*10000,
                expected_funding_bps=carry/gross*10000,expected_cash=budget['expected_net_cash'],expected_net_cash=budget['expected_net_cash'],
                terminal_prices=terminal.tolist(),source_prices=anchor.tolist(),source_price_alpha=mu,
                entry_limits={str(j):v for j,v in zip(ids,budget['limits'])},minimum_expected_cash_at_limits=budget['minimum_expected_cash_at_limits'],
                buffer_cash=budget['buffer_cash'],intent_time=now,forecast_source=source,
                funding_known_through=now//3600*3600-3600))
    return sorted(results,key=lambda r:(-r['expected_cash'],r['pair']))
