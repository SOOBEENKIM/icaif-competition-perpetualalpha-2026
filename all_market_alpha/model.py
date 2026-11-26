"""Causal normalized features, mature-label ridge and symmetric target policy."""
import numpy as np
import pandas as pd
from replay_lab.common_comparison.engine import project


def features(d):
    price = np.asarray(d["close"], float)
    log = pd.DataFrame(np.log(price))
    ret = log.diff()
    vol = ret.ewm(span=168, min_periods=168, adjust=False).std().clip(lower=.0005)
    short_vol = ret.ewm(span=24, min_periods=24, adjust=False).std().clip(lower=.0005)
    momenta = [(log-log.shift(h))/(vol*np.sqrt(h)) for h in (24,72,168,336)]
    deviation = (log-log.rolling(168).mean())/(
        log.rolling(168).std().clip(lower=.001))
    relative = momenta[1].sub(momenta[1].mean(axis=1), axis=0)
    funding = pd.DataFrame(d["funding"]).shift(1).rolling(24).mean().fillna(0)
    x = np.stack([a.to_numpy() for a in momenta + [
        deviation, short_vol/vol-1, relative, funding*24/(vol*np.sqrt(24))]], axis=-1)
    x = np.clip(x, -5, 5)
    fresh = (pd.DataFrame((d["volume"]>0)&(d["trades"]>0))
             .rolling(6,min_periods=1).max().to_numpy().astype(bool))
    return dict(x=x, unit=vol.to_numpy()*np.sqrt(24), fresh=fresh,
                funding=funding.to_numpy()*24)


def labels(d, f, horizon=24, delay=1):
    n, width = d["close"].shape
    y = np.full((n,width), np.nan)
    maturity = np.asarray(d["market_times"])+3600*(horizon+delay)
    valid = np.zeros((n,width), bool)
    stop = n-horizon-delay
    if stop > 0:
        y[:stop] = np.log(d["close"][horizon+delay:]/d["close"][delay:n-horizon])/f["unit"][:stop]
        valid[:stop] = ((d["volume"][delay:n-horizon]>0)&
                       (d["trades"][delay:n-horizon]>0)&
                       (d["volume"][horizon+delay:]>0)&
                       (d["trades"][horizon+delay:]>0))
    return y, maturity, valid


def fit(x, y, sample_times, maturity, cutoff, spec):
    if len(y)<spec["minimum_rows"]:
        return None
    if np.any(maturity>=cutoff):
        raise ValueError("Unmatured training label")
    weight=np.exp2(-(cutoff-sample_times)/(86400*spec["half_life_days"]))
    weight/=weight.sum()
    center=weight@x
    scale=np.sqrt(weight@((x-center)**2))
    scale=np.maximum(scale,.1)
    z=np.c_[np.ones(len(y)),(x-center)/scale]
    penalty=np.eye(z.shape[1])*spec["ridge_penalty"]
    penalty[0,0]=.1
    beta=np.linalg.solve(z.T@(weight[:,None]*z)+penalty,
                         z.T@(weight*np.clip(y,-spec["target_clip"],spec["target_clip"])))
    return dict(center=center,scale=scale,beta=beta,rows=len(y),
                last_maturity=int(np.max(maturity)),fit_time=int(cutoff))


def rolling_forecast(d, f, spec):
    t=np.asarray(d["market_times"]);x=f["x"]
    y,maturity,terminal=labels(d,f,spec["horizon"],spec["execution_delay"])
    valid=(np.isfinite(x).all(-1)&np.isfinite(y)&f["fresh"]&terminal)
    out=np.full(y.shape,np.nan);states=[];model=None
    for i,now in enumerate(t):
        if now%(spec["refit_hours"]*3600)==0:
            mask=((t>=now-spec["lookback_days"]*86400)&(maturity<now)&
                  (t%(spec["sample_hours"]*3600)==0))
            ii,jj=np.where(valid&mask[:,None])
            model=fit(x[ii,jj],y[ii,jj],t[ii],maturity[ii],now,spec)
            if model is not None:
                states.append(dict(model,asset_counts=np.bincount(jj,minlength=x.shape[1])))
        if model is not None:
            out[i]=np.clip(model["beta"][0]+((x[i]-model["center"])/model["scale"])@model["beta"][1:],
                           -spec["forecast_clip"],spec["forecast_clip"])*f["unit"][i]
    return out,states


def symmetric_state(old, signal, eligible, entry=.25, exit=.1):
    """No same-decision reversal. Asset names/order cannot affect this state."""
    old=np.asarray(old)
    result=old.copy()
    close=(old!=0)&((old*signal<=exit)|~np.isfinite(signal))
    result[close]=0
    enter=(old==0)&eligible&np.isfinite(signal)&(abs(signal)>=entry)
    result[enter]=np.sign(signal[enter])
    return result


def targets(d, f, forecast, sigma, protocol, policy, kind):
    n,width=d["close"].shape
    out=np.zeros((n,width));states=np.zeros_like(out)
    held=np.zeros(width);state=np.zeros(width)
    # A smooth, symmetric bounded trend signal; no fixed volatility screen.
    trend=np.nan_to_num(np.tanh(f["x"][:,:,:4].mean(-1)/2),nan=0)
    if kind=="multiscale":
        signal=trend
    elif kind=="pooled_ridge":
        net=forecast-f["funding"]
        signal=np.sign(net)*np.maximum(abs(net)-policy["roundtrip_cost"],0)/f["unit"]
        signal=np.clip(signal,-1,1)
    else:
        raise ValueError(kind)
    for i,now in enumerate(d["market_times"]):
        if i>=policy["minimum_warmup_hours"] and now%(policy["decision_hours"]*3600)==0:
            eligible=f["fresh"][i]&np.isfinite(f["unit"][i])
            state=symmetric_state(state,signal[i],eligible,
                                  policy["entry_signal"],policy["exit_signal"])
            conviction=np.where(state*signal[i]>0,abs(signal[i]),0)
            raw=state*conviction*policy["daily_risk_budget"]/(
                np.sqrt(width)*np.maximum(f["unit"][i],.003))
            held=project(raw,protocol["initial"],sigma[i],protocol["risk"])
        out[i]=held;states[i]=state
    if not np.isfinite(out).all():
        raise ValueError("Invalid target")
    return out,states
