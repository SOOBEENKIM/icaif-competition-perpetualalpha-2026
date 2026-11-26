"""Original causal intent construction; preserve the source price hypothesis."""
import numpy as np
from experiments.adaptive_mean_pairs_20261017.signals import design,held_state
from experiments.par_direct_forecast_20261017.data import ts
from experiments.quote_anchored_pairs_20261017.pricing import reservation_limits
from replay_lab.common_comparison.engine import rounded,risk_breaches
from replay_lab.pullback.broker import execution_price


def intent_state(d,reference,cfg,symbols,lifetime_hours):
    rt=reference['market_times'];rp=reference['close']
    rd=dict(times=rt,par_close=rp)
    raw=design(rd,reference,cfg,symbols);pairs=raw['pairs'];n=len(rt);k=len(pairs)
    source=np.full((n,k),-1,dtype=np.int64);direction=np.zeros((n,k),dtype=int)
    beta=np.full((n,k),np.nan);mean=beta.copy();sd=beta.copy();terminal=np.full((n,k,2),np.nan)
    for j,(a,b) in enumerate(pairs):
        last=None
        for i,t in enumerate(rt):
            if 'formation_schedule' in cfg:
                forms=[f for f in cfg['formation_schedule'] if ts(f['date'])<=t]
                allowed=bool(forms and cfg['pairs'][j] in forms[-1]['pairs'])
            else:allowed=t>=ts(cfg['formation_cutoff'])
            # A month boundary can forbid a new trade; it does not liquidate an existing one.
            if not allowed:last=None;continue
            sign=int(raw['delayed'][i,j])
            if sign:
                be=raw['beta'][i,j];mu=raw['mean'][i,j];scale=raw['sd'][i,j]
                displacement=mu-(np.log(rp[i,a])-be*np.log(rp[i,b]))
                terminal_prices=rp[i,[a,b]]*np.exp(np.array([1.,-1.])*displacement/(1+be))
                last=(int(t),sign,be,mu,scale,terminal_prices)
            if last is None or t-last[0]>=lifetime_hours*3600:continue
            current_z=(np.log(rp[i,a])-last[2]*np.log(rp[i,b])-last[3])/last[4]
            if not -cfg['entry_z']*cfg['stop_multiplier']<last[1]*current_z<0:
                last=None;continue
            source[i,j],direction[i,j],beta[i,j],mean[i,j],sd[i,j]=last[:5]
            terminal[i,j]=last[5]
    ix=np.searchsorted(rt,d['times'],side='right')-1
    if np.any(ix<0):raise ValueError('Reference history unavailable at decision')
    assert np.all(rt[ix]<=d['times'])
    mapped={name:arr[ix] for name,arr in dict(signal_times=source,directions=direction,beta=beta,mean=mean,sd=sd,terminal_prices=terminal).items()}
    mapped['directions']=mapped['directions'].copy()
    expired=d['times'][:,None]-mapped['signal_times']>=lifetime_hours*3600
    mapped['directions'][expired]=0
    z=np.full((len(ix),k),np.nan)
    for j,(a,b) in enumerate(pairs):
        z[:,j]=(np.log(d['par_close'][:,a])-mapped['beta'][:,j]*np.log(d['par_close'][:,b])-mapped['mean'][:,j])/mapped['sd'][:,j]
    return dict(**mapped,pairs=pairs,z=z,forecast_times=rt[ix],correlation=np.ones_like(z),
        mean_log_prices=raw['mean_log_prices'][ix],fresh=np.isfinite(z))


def usable_signals(i,d,de,cfg,arm,locks,used):
    out=de['directions'][i].copy()
    for k,direction in enumerate(out):
        lock=locks.get(k)
        if lock:
            state=held_state(lock,i,d,de,cfg)
            if lock['direction']*state['z']>=0:del locks[k]
            else:out[k]=0
        if int(de['signal_times'][i,k])<=used.get(k,-1):out[k]=0
        if not np.isfinite(de['z'][i,k]) or not -cfg['entry_z']*cfg['stop_multiplier']<direction*de['z'][i,k]<0:
            out[k]=0
            if direction and np.isfinite(de['z'][i,k]):used[k]=int(de['signal_times'][i,k])
    return out


def proposals(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm):
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
                q=qty.copy();q[ids]+=g*w/price[ids]*mask
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
