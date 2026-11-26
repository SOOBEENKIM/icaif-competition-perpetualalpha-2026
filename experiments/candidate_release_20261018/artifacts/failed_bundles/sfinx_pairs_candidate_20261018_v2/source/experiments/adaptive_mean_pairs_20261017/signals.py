"""Original statistical state machine for fixed hedge / adaptive mean pairs."""
import numpy as np


def crossing(previous,current,threshold,delayed=False):
    if not np.isfinite(previous+current):return 0
    if delayed:
        if previous<=-threshold<current<0:return 1
        if previous>=threshold>current>0:return -1
    else:
        if previous>-threshold>=current:return 1
        if previous<threshold<=current:return -1
    return 0


def projected_return(displacement,beta):
    """Equal/opposite log-price changes reaching the proposed spread mean."""
    move=displacement/(1+beta)
    return float((np.expm1(move)-beta*np.expm1(-move))/(1+beta))


def design(d,reference,cfg,symbols):
    times=reference['market_times'];lp=np.log(reference['close'])
    pair_ids=[[symbols.index(x) for x in pair] for pair in cfg['pairs']]
    n=len(times);k=len(pair_ids);wb=cfg['beta_window_hours'];wm=cfg['mean_window_hours']
    beta=np.full((n,k),np.nan);mean=np.full_like(beta,np.nan);sd=np.full_like(beta,np.nan)
    for i in range(wb-1,n):
        for j,(a,b) in enumerate(pair_ids):
            yy=lp[i-wb+1:i+1,a];xx=lp[i-wb+1:i+1,b]
            xc=xx-xx.mean();den=xc@xc
            if den<=1e-15:continue
            be=xc@(yy-yy.mean())/den
            if not .1<=be<=10:continue
            residual=lp[i-wm+1:i+1,a]-be*lp[i-wm+1:i+1,b]
            beta[i,j]=be;mean[i,j]=residual.mean();sd[i,j]=residual.std(ddof=1)
    ix=np.searchsorted(times,d['times'],side='right')-1
    assert np.all(ix>=0) and np.all(times[ix]<=d['times'])
    pa=np.log(d['par_close']);z=np.full((len(ix),k),np.nan)
    for j,(a,b) in enumerate(pair_ids):z[:,j]=(pa[:,a]-beta[ix,j]*pa[:,b]-mean[ix,j])/sd[ix,j]
    # Each source hour's signal remains known during that hour. No future-hour lookup.
    source=times[ix];src_index=np.searchsorted(d['times'],source,side='left')
    prev_index=np.searchsorted(d['times'],source-3600,side='left')
    valid=(source>=d['times'][0])&(source-3600>=d['times'][0])
    standard=np.zeros_like(z,dtype=int);delayed=np.zeros_like(z,dtype=int)
    for i in range(len(ix)):
        if not valid[i]:continue
        assert d['times'][src_index[i]]==source[i] and d['times'][prev_index[i]]==source[i]-3600
        for j in range(k):
            for delay,dest in ((False,standard),(True,delayed)):
                dest[i,j]=crossing(z[prev_index[i],j],z[src_index[i],j],cfg['entry_z'],delay)
    means=np.full((n,len(symbols)),np.nan)
    for i in range(wm-1,n):means[i]=lp[i-wm+1:i+1].mean(0)
    return dict(pairs=pair_ids,beta=beta[ix],mean=mean[ix],sd=sd[ix],z=z,
        mean_log_prices=means[ix],forecast_times=source,standard=standard,delayed=delayed,
        correlation=np.ones_like(z),fresh=np.isfinite(z))


def held_state(pair,i,d,de,cfg):
    a,b=pair['ids'];beta=pair['beta']
    mean=pair['entry_mean']
    if pair['adaptive_mean']:mean=float(de['mean_log_prices'][i,a]-beta*de['mean_log_prices'][i,b])
    spread=float(np.log(d['par_close'][i,a])-beta*np.log(d['par_close'][i,b]))
    z=(spread-mean)/pair['entry_sd']
    age=(int(d['times'][i])-pair.get('first_fill',pair['forecast_source']))/3600
    limit=cfg['entry_z']*cfg['stop_multiplier']
    if age>cfg['time_decay_start_hours']:
        limit*=max(0.,(cfg['maximum_holding_hours']-age)/(cfg['maximum_holding_hours']-cfg['time_decay_start_hours']))
    directional_z=pair['direction']*z
    reason=None
    if age>=cfg['maximum_holding_hours']:reason='time_stop'
    elif directional_z>=0:reason='mean_cross'
    elif directional_z<=-limit:reason='spread_stop'
    return dict(z=float(z),mean=float(mean),stop_bound=float(limit),age_hours=float(age),reason=reason)


def usable_signals(i,d,de,cfg,arm,locks,used):
    directions=(de['delayed'] if 'delayed' in arm else de['standard'])[i].copy()
    for k,direction in enumerate(directions):
        lock=locks.get(k)
        if lock:
            state=held_state(lock,i,d,de,cfg)
            if lock['direction']*state['z']>=0:del locks[k]
            else:directions[k]=0
        if int(de['forecast_times'][i])<=used.get(k,-1):directions[k]=0
        if not np.isfinite(de['z'][i,k]) or not -cfg['entry_z']*cfg['stop_multiplier']<direction*de['z'][i,k]<0:
            directions[k]=0
    return directions
