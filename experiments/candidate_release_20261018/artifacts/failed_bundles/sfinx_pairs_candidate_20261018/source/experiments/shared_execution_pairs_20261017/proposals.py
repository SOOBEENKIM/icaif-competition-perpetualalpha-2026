import numpy as np
from replay_lab.common_comparison.engine import rounded,risk_breaches
from replay_lab.pullback.broker import execution_price
from experiments.quote_anchored_pairs_20261017.pricing import reservation_limits

def proposals(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,*,risk_qty=None):
    risk_qty=qty if risk_qty is None else risk_qty
    price=d['par_close'][i];sigma=d['sigma'][i];results=[];cost=2*(sc['fee_bps']+sc['impact_bps'])/10000
    for k,ids in enumerate(de['pairs']):
        if not directions[k]:continue
        source=int(de['signal_times'][i,k]);assert source<=d['times'][i]
        be=de['beta'][i,k];terminal=de['terminal_prices'][i,k]
        if not .1<=be<=10 or not np.isfinite(terminal).all():continue
        if np.any(qty[ids]) or np.any(abs(reserved[ids])>1e-12):continue
        direction=int(directions[k]);w=direction*np.array([1.,-be])/(1+be)
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
        if abs(actual_beta/be-1)>cfg['pair_balance_tolerance']:continue
        preview=np.array([execution_price(price[j],np.sign(z),sc['impact_bps'],p['execution']['decimals'][j]) for j,z in zip(ids,q)])
        remaining=max(0.,cfg['funding_forecast_hours']-(int(d['times'][i])-source)/3600)
        carry=-q*d['reference_close'][i,ids]*d['known_hourly_funding'][i,ids]*remaining
        carry=float(np.where(carry>0,carry*.5,carry).sum())
        gross=float(abs(q)@price[ids]);edge=float(q@(terminal-price[ids]))/gross
        budget=reservation_limits(q,price[ids],preview,edge,carry,sc['fee_bps'],sc['impact_bps'],
            cfg['entry_buffer_bps'],cfg['entry_slippage_cap_bps'])
        if budget is None:continue
        results.append(dict(pair=cfg['pairs'][k],pair_index=k,ids=ids,target=q.tolist(),horizon=h,
            beta=float(be),direction=direction,entry_mean=float(de['mean'][i,k]),entry_sd=float(de['sd'][i,k]),
            entry_z=float(de['z'][i,k]),adaptive_mean=False,expected_net_bps=budget['expected_net_cash']/gross*10000,
            expected_price_bps=edge*10000,expected_funding_bps=carry/gross*10000,expected_cash=budget['expected_net_cash'],
            terminal_prices=terminal.tolist(),expected_net_cash=budget['expected_net_cash'],
            entry_limits={str(j):v for j,v in zip(ids,budget['limits'])},minimum_expected_cash_at_limits=budget['minimum_expected_cash_at_limits'],
            buffer_cash=budget['buffer_cash'],intent_time=int(d['times'][i]),forecast_source=source,
            funding_known_through=int(d['times'][i]//3600*3600-3600)))
    return sorted(results,key=lambda r:(-r['expected_cash'],r['pair']))
