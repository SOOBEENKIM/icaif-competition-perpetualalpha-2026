"""Past-matured event value, with the trading engine left unchanged."""
import numpy as np
from experiments.adaptive_mean_pairs_20261017.signals import design,held_state
from experiments.par_direct_forecast_20261017.data import ts


def source_design(reference,cfg,symbols):
    t=reference['market_times'];price=reference['close'];lp=np.log(price)
    raw=design(dict(times=t,par_close=price),reference,cfg,symbols)
    confirmed=np.zeros_like(raw['delayed'],bool);price_recovery=confirmed.copy()
    previous_fixed_z=np.full_like(raw['z'],np.nan)
    for k,(a,b) in enumerate(raw['pairs']):
        beta=raw['beta'][1:,k];mu=raw['mean'][1:,k];sd=raw['sd'][1:,k];direction=raw['delayed'][1:,k]
        zprev=(lp[:-1,a]-beta*lp[:-1,b]-mu)/sd
        previous_fixed_z[1:,k]=zprev
        confirmed[1:,k]=(direction!=0)&(direction*zprev<=-cfg['entry_z'])&(direction*raw['z'][1:,k]>-cfg['entry_z'])&(direction*raw['z'][1:,k]<0)
        price_recovery[1:,k]=direction*((lp[1:,a]-lp[:-1,a])-beta*(lp[1:,b]-lp[:-1,b]))>0
    return dict(**raw,confirmed=confirmed,price_recovery=price_recovery,previous_fixed_z=previous_fixed_z)


def completed_events(reference,raw,cfg):
    times=reference['market_times'];prices=reference['close'];d=dict(times=times,par_close=prices);events=[]
    for k,ids in enumerate(raw['pairs']):
        blocked_until=-1
        for i in np.flatnonzero(raw['delayed'][:,k]):
            if i<=blocked_until or i+1>=len(times):continue
            sign=int(raw['delayed'][i,k]);beta=float(raw['beta'][i,k]);sd=float(raw['sd'][i,k]);mu=float(raw['mean'][i,k])
            if not np.isfinite([beta,sd,mu]).all() or sd<=0:continue
            if not -cfg['entry_z']*cfg['stop_multiplier']<sign*raw['z'][i,k]<0:continue
            anchor=prices[i,ids];gap=mu-(np.log(anchor[0])-beta*np.log(anchor[1]))
            terminal=anchor*np.exp(np.array([1.,-1.])*gap/(1+beta));qty=sign*np.array([1.,-beta])/(1+beta)/anchor
            expected=float(qty@(terminal-anchor))
            assert expected>0
            pair=dict(ids=ids,beta=beta,entry_mean=mu,entry_sd=sd,adaptive_mean=False,direction=sign,
                first_fill=int(times[i+1]),forecast_source=int(times[i]))
            maturity=None;reason=None
            for j in range(i+1,min(len(times)-1,i+cfg['maximum_holding_hours']+2)):
                state=held_state(pair,j,d,raw,cfg)
                if state['reason']:
                    maturity=j+1;reason=state['reason'];break
            blocked_until=len(times) if maturity is None else maturity
            event=dict(time=int(times[i]),pair_index=k,pair=cfg['pairs'][k],confirmed=bool(raw['confirmed'][i,k]),
                price_recovery=bool(raw['price_recovery'][i,k]),expected_full_convergence=expected,
                beta=beta,direction=sign,entry_z=float(raw['z'][i,k]),maturity=None if maturity is None else int(times[maturity]),
                exit_reason=reason,realized_price_return=None if maturity is None else float(qty@(prices[maturity,ids]-anchor)))
            events.append(event)
    return sorted(events,key=lambda e:(e['time'],e['pair_index']))


def fit_at(events,at,spec):
    history=[e for e in events if e['maturity'] is not None and e['maturity']<at and e['time']>=at-spec['training_window_days']*86400]
    fit=dict(time=at,events=len(history),last_maturity=max((e['maturity'] for e in history),default=None),groups={})
    for group in (False,True):
        same=[e for e in history if e['confirmed']==group]
        sample=same if len(same)>=spec['minimum_group_events'] else history
        if len(history)<spec['minimum_events']:
            ratio=None;unshrunk=None
        else:
            x=np.array([e['expected_full_convergence'] for e in sample]);y=np.array([e['realized_price_return'] for e in sample])
            unshrunk=float(x@y/(x@x));ratio=float(np.clip(unshrunk/(1+spec['prior_events']/len(sample)),0,1))
        fit['groups'][str(group)]=dict(events=len(sample),same_group=len(same),pooled=len(same)<spec['minimum_group_events'],
            ratio=ratio,unshrunk_ratio=unshrunk,event_keys=[[e['time'],e['pair_index']] for e in sample])
    return fit


def map_state(state,times,reference,raw,fits,mode):
    out={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in state.items()};audit=[]
    source_times=reference['market_times'];prices=reference['close'];cache={}
    for i,k in zip(*np.where(state['directions']!=0)):
        source=int(state['signal_times'][i,k]);ix=int(np.searchsorted(source_times,source));assert source_times[ix]==source<=times[i]
        key=(source,int(k));confirmed=bool(raw['confirmed'][ix,k]);ratio=1.
        fit=next((f for f in reversed(fits) if f['time']<=source),None)
        if mode=='confirmed':keep=confirmed
        elif mode=='calibrated':
            ratio=None if fit is None else fit['groups'][str(confirmed)]['ratio']
            keep=ratio is not None and ratio>0
        else:raise ValueError(mode)
        if not keep:out['directions'][i,k]=0
        elif mode=='calibrated':
            anchor=prices[ix,raw['pairs'][k]]
            out['terminal_prices'][i,k]=anchor+ratio*(state['terminal_prices'][i,k]-anchor)
        if key not in cache:
            cache[key]=True;audit.append(dict(time=source,pair_index=int(k),confirmed=confirmed,kept=keep,ratio=ratio,
                fit_time=None if fit is None else fit['time'],last_maturity=None if fit is None else fit['last_maturity']))
    return out,audit
