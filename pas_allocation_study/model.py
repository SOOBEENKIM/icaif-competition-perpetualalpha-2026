"""One partially pooled model. Fit, calibration, and inference are separate."""
import numpy as np
import pandas as pd

# Economic asset groups, not groups selected from realized returns.
GROUPS = np.array([0,0,0,1,1,1,2,2,3,3])


def features(d):
    log = pd.DataFrame(np.log(d['close']))
    ret = log.diff()
    vol = ret.ewm(span=168, min_periods=168, adjust=False).std().clip(lower=.0005)
    components = [(log-log.shift(h))/(vol*np.sqrt(h)) for h in (6,24,72,168)]
    components += [(log-log.ewm(span=h, adjust=False).mean())/(vol*np.sqrt(h)) for h in (24,168)]
    components += [ret.ewm(span=24,min_periods=24,adjust=False).std()/vol-1]
    x = np.clip(np.stack([v.to_numpy() for v in components],-1),-4,4)
    # The group interactions share the pooled slope and shrink group deviations.
    unit = vol.to_numpy()*np.sqrt(24)
    fresh = (np.asarray(d['volume'])>0)&(np.asarray(d['trades'])>0)
    return {'x':x,'unit':unit,'fresh':fresh}


def design(x, asset, center, scale):
    z = (x-center)/scale
    base = np.c_[np.ones(len(x)),z]
    groups = np.eye(4)[GROUPS[asset]]
    return np.c_[base, (groups[:,:,None]*base[:,None,:]).reshape(len(x),-1)]


def label_rows(d,f,start,end):
    t=d['market_times']; H=24;delay=0
    i,j=np.where((t[:,None]>=start)&((t[:,None]+24*3600)<end)&
        ((t[:,None] % 86400)==0)&np.isfinite(f['x']).all(-1))
    ok=(i+H+delay<len(t));i,j=i[ok],j[ok]
    ok=f['fresh'][i+delay,j]&f['fresh'][i+delay+H,j]
    i,j=i[ok],j[ok]
    y=np.log(d['close'][i+delay+H,j]/d['close'][i+delay,j])/f['unit'][i,j]
    return i,j,y


def train(d,p):
    # No label construction or feature scaling ever sees the next interval.
    prefix={k:v[d['market_times']<p['train_end']] for k,v in d.items()}
    f=features(prefix)
    i,j,y=label_rows(prefix,f,p['train_start'],p['train_end'])
    if len(y)<200: raise ValueError('Insufficient mature training labels')
    x=f['x'][i,j];center=x.mean(0);scale=np.maximum(x.std(0),.1)
    z=design(x,j,center,scale);v=z.shape[1]//5
    penalty=np.r_[np.ones(v)*p['common_ridge'],np.ones(4*v)*p['group_ridge']]
    beta=np.linalg.solve(z.T@z/len(y)+np.diag(penalty),z.T@np.clip(y,-4,4)/len(y))
    return dict(center=center,scale=scale,beta=beta,cutoff=p['train_end'],
        rows=len(y),asset_counts=np.bincount(j,minlength=10),
        last_feature=int(prefix['market_times'][i].max()),
        last_maturity=int((prefix['market_times'][i]+24*3600).max()))


def predict(d,model):
    f=features(d);n,m=f['unit'].shape
    z=design(f['x'].reshape(n*m,-1),np.tile(np.arange(m),n),
             np.asarray(model['center']),np.asarray(model['scale']))
    norm=np.clip((z@np.asarray(model['beta'])).reshape(n,m),-1,1)
    forecast=norm*f['unit']
    forecast[d['market_times']<model['cutoff']]=np.nan
    return f,forecast


def calibrate(d,model,p):
    prefix={k:v[d['market_times']<p['calibration_end']] for k,v in d.items()}
    f,forecast=predict(prefix,model)
    i,j,y=label_rows(prefix,f,p['calibration_start'],p['calibration_end'])
    x=forecast[i,j]/f['unit'][i,j]
    scale=float(np.clip(x@y/(x@x+1.),0,1))
    return dict(scale=scale,rows=len(y),last_maturity=int((prefix['market_times'][i]+24*3600).max()),
        purpose='Attenuate forecast only, never amplify based on validation PnL')


def funding_forecast(d):
    r=pd.DataFrame(np.where(d['observed'],d['funding'],np.nan))
    # Settlement at t is only used starting the following completed hour.
    return r.shift(1).rolling(24,min_periods=12).mean().fillna(0).to_numpy()*12


def allocate(net,vol,old,p):
    """Symmetric entry and thesis exit; no compulsory trade or short quota."""
    finite=np.isfinite(net)&np.isfinite(vol)&(vol>0)
    enter=finite&(abs(net)>p['entry_cost']+p['entry_buffer'])
    same=finite&(old*net>0)
    # Incumbent positions may remain until expected remaining return is <=0.
    edge=np.where(same,abs(net),np.maximum(abs(net)-p['entry_cost']-p['entry_buffer'],0))
    conviction=np.minimum(edge/np.maximum(vol,.003),1.)
    w=np.where(enter|same,np.sign(net)*conviction*p['daily_target_risk']/(np.sqrt(10)*np.maximum(vol,.003)),0.)
    return np.where(finite,w,0.)


def weights(d,forecast,sigma,p,trading):
    from replay_lab.common_comparison.engine import project
    f=features(d);funding=funding_forecast(d);out=np.zeros_like(forecast);held=np.zeros(10)
    for i,t in enumerate(d['market_times']):
        if t>=p['calibration_end'] and t%(p['decision_hours']*3600)==0:
            held=allocate(forecast[i]-funding[i],f['unit'][i],held,p)
            risk=float(np.sqrt(max(24*held@sigma[i]@held,0)))
            if risk>p['daily_target_risk']: held*=p['daily_target_risk']/risk
            held=project(held,trading['initial'],sigma[i],trading['risk'])
        out[i]=held
    return out
