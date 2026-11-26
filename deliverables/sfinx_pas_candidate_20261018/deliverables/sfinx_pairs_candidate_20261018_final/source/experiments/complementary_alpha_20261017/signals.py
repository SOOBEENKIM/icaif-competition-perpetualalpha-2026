"""Four causal signal paths. No account access or exchange requests.

Parameters are fixed before costed comparisons. Statistical models use only
past formation data; none of these simplified mechanisms reproduces a paper.
"""
import numpy as np
import pandas as pd


def formation(log_prices, index, config):
    window=config['formation_hours']; step=config['formation_sampling_hours']
    if index < window:return None
    xy=log_prices[index-window:index:step, :2]
    x,y=xy[:,0],xy[:,1]; anchor=float(x.mean())
    design=np.column_stack([np.ones(len(x)),x-anchor])
    intercept,beta=np.linalg.lstsq(design,y,rcond=None)[0]
    if not config['beta_range'][0] <= beta <= config['beta_range'][1]:return None
    residual=y-design@np.array([intercept,beta]); sd=float(residual.std(ddof=1))
    if sd<1e-8:return None
    ar=np.linalg.lstsq(np.column_stack([np.ones(len(residual)-1),residual[:-1]]),residual[1:],rcond=None)[0]
    rho=float(ar[1]); half_life=-step*np.log(2)/np.log(rho) if 0<rho<1 else float('inf')
    betas=[np.linalg.lstsq(design[ix],y[ix],rcond=None)[0][1] for ix in np.array_split(np.arange(len(y)),2)]
    stable=max(abs(np.array(betas)-beta))/abs(beta)<=config['beta_relative_instability_limit']
    allowed=stable and config['half_life_hours'][0]<=half_life<=config['half_life_hours'][1]
    return dict(anchor=anchor,intercept=float(intercept),beta=float(beta),sd=sd,
                mean=float(ar[0]/(1-rho)) if 0<rho<1 else 0.,rho=rho,
                half_life_hours=float(half_life) if np.isfinite(half_life) else None,
                eligible=bool(allowed),last_source_index=index-1)


def pair_vector(beta,direction,n):
    out=np.zeros(n);out[:2]=direction*np.array([-beta,1.])/(1+beta)
    return out


def pairs(inp,config,kind):
    pc=config['pair'];kc=config['kalman'];times=inp['market_times'];price=inp['close'];log=np.log(price)
    result=np.zeros_like(price); z_path=np.full(len(times),np.nan);beta_path=np.full(len(times),np.nan)
    direction=0;entry_time=0;cooldown=-1;fit=None;last_fit=-10**18;position_beta=0.
    kstate=None;kvar=None;noise=None;events=[];formations=[];last_target=np.zeros(price.shape[1])
    past_funding=pd.DataFrame(inp['funding']).shift(1).rolling(168).mean().to_numpy()
    for i,t in enumerate(times):
        if i<pc['formation_hours']:continue
        if kind=='pair_ou':
            if not direction and (fit is None or t-last_fit>=pc['refit_hours']*3600):
                fit=formation(log,i,pc);last_fit=int(t)
                if fit is not None:formations.append(dict(time=int(t),**fit))
            if fit is None:continue
            beta=fit['beta']; residual=log[i,1]-fit['intercept']-beta*(log[i,0]-fit['anchor'])-fit['mean']
            sd=fit['sd'];z=residual/sd;allowed=fit['eligible'];max_hold=pc['maximum_hold_hours']
        else:
            if kstate is None:
                initial=formation(log,i,pc)
                if initial is None:continue
                fit=initial;formations.append(dict(time=int(t),**fit))
                kstate=np.array([fit['intercept'],fit['beta']]);kvar=np.diag([fit['sd']**2,.01])
                rr=log[i-pc['formation_hours']:i,1]-fit['beta']*log[i-pc['formation_hours']:i,0]
                noise=max(float(np.var(np.diff(rr)))*kc['noise_residual_difference_horizon_hours'],kc['measurement_noise_floor'])
            h=np.array([1.,log[i,0]-fit['anchor']]);prior=kvar+np.diag([noise,noise])*kc['state_noise_fraction']
            residual=float(log[i,1]-h@kstate);sd=float(np.sqrt(h@prior@h+noise));z=residual/sd
            beta=float(kstate[1]);allowed=pc['beta_range'][0]<=beta<=pc['beta_range'][1];max_hold=kc['maximum_hold_hours']
            # A pre-update innovation drives the signal; today's observation is
            # then incorporated for the NEXT forecast, never smoothed backwards.
            gain=prior@h/(h@prior@h+noise);kstate=kstate+gain*residual
            kh=np.eye(2)-np.outer(gain,h);kvar=kh@prior@kh.T+np.outer(gain,gain)*noise
        z_path[i]=z;beta_path[i]=beta
        if t%21600:
            result[i]=last_target;continue
        if direction and (direction*z>=-pc['exit_z'] or abs(z)>=pc['stop_z'] or t-entry_time>=max_hold*3600 or not allowed):
            events.append(dict(time=int(t),action='exit',z=float(z),held_hours=(t-entry_time)/3600))
            direction=0;last_target=np.zeros(price.shape[1]);cooldown=int(t)+pc['cooldown_hours']*3600
        if not direction and t>=cooldown and allowed and pc['entry_z']<=abs(z)<pc['stop_z']:
            # P&L is per gross notional: both legs' entry AND exit are charged.
            edge=max(0.,abs(residual)-pc['exit_z']*sd)/(1+abs(beta))
            vec=pair_vector(beta,-np.sign(z),price.shape[1])
            funding=max(0.,float(vec@np.nan_to_num(past_funding[i])))*max_hold
            threshold=pc['entry_edge_cost_multiple']*pc['round_trip_bps_per_gross']/10000+funding
            if edge>threshold:
                direction=-np.sign(z);position_beta=beta;entry_time=int(t);last_target=vec
                events.append(dict(time=int(t),action='enter',z=float(z),beta=position_beta,
                                   predicted_convergence_per_gross=edge,required_edge=threshold))
        result[i]=last_target
    return result,dict(events=events,formations=formations,z=z_path,beta=beta_path)


def relative_momentum(inp,config):
    c=config['relative_momentum'];p=inp['close'];times=inp['market_times'];result=np.zeros_like(p)
    last=np.zeros(p.shape[1]);selected=None;changed=-10**18;events=[]
    for i,t in enumerate(times):
        if i>=c['lookback_hours'] and t%(c['signal_update_hours']*3600)==0:
            score=np.log(p[i-c['skip_recent_hours'],:3]/p[i-c['lookback_hours'],:3])
            hi=int(np.argmax(score));lo=int(np.argmin(score))
            pair=(hi,lo) if score[hi]-score[lo]>=c['minimum_return_dispersion'] else None
            if pair!=selected and t-changed>=c['minimum_hold_hours']*3600:
                last=np.zeros(p.shape[1]);selected=pair;changed=int(t)
                if pair is not None:last[hi]=.5;last[lo]=-.5
                events.append(dict(time=int(t),pair=pair,score=score.tolist()))
        result[i]=last
    return result,dict(events=events)


def channel_breakout(inp,config):
    c=config['breakout'];p=inp['close'];times=inp['market_times'];frame=pd.DataFrame(p)
    # Shift removes the current price from each breakout/exit boundary.
    upper=frame.shift(1).rolling(c['entry_hours']).max().to_numpy()
    lower=frame.shift(1).rolling(c['entry_hours']).min().to_numpy()
    exit_high=frame.shift(1).rolling(c['exit_hours']).max().to_numpy()
    exit_low=frame.shift(1).rolling(c['exit_hours']).min().to_numpy()
    vol=np.maximum(np.log(frame).diff().rolling(c['volatility_hours']).std(ddof=0).to_numpy(),.0005)
    direction=np.zeros(p.shape[1]);last=np.zeros(p.shape[1]);result=np.zeros_like(p);events=[]
    for i,t in enumerate(times):
        if i>=c['entry_hours'] and t%21600==0:
            old=direction.copy()
            direction=np.where((direction>0)&(p[i]<exit_low[i]),0,direction)
            direction=np.where((direction<0)&(p[i]>exit_high[i]),0,direction)
            direction=np.where((direction==0)&(p[i]>upper[i]),1,direction)
            direction=np.where((direction==0)&(p[i]<lower[i]),-1,direction)
            inverse=1/vol[i];last=direction*inverse/inverse.sum()
            if np.any(old!=direction):events.append(dict(time=int(t),direction=direction.tolist()))
        result[i]=last
    return result,dict(events=events)


def build(inp,config):
    weights={};diagnostics={}
    for kind in ('pair_ou','pair_kalman'):weights[kind],diagnostics[kind]=pairs(inp,config,kind)
    weights['relative_momentum'],diagnostics['relative_momentum']=relative_momentum(inp,config)
    weights['channel_breakout'],diagnostics['channel_breakout']=channel_breakout(inp,config)
    for value in weights.values():assert np.isfinite(value).all()
    return weights,diagnostics
