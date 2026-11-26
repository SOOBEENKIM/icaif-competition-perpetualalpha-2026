"""Universe-only extension of the selected trend/reversion price policy.

The frozen crypto implementation remains untouched. ``scope='legacy'`` is an
audit control, never the all-market deployment. Thresholds, capital calibration,
Chronos/regime forecasts, funding signal, and execution rules are unchanged.
"""
import json
from pathlib import Path

import numpy as np

from replay_lab.common_comparison.models import Market, covariance
from replay_lab.paper_reproductions.ema_wf.model import ema
from replay_lab.macro_development.carry import signals as funding_signals
from replay_lab.portfolio_development.funding_model import weights as funding_weights
from replay_lab.v10_revision.core import components, policies
from replay_lab.portfolio_rebuild_v10.lifecycle_allocation import targets
from replay_lab.reversion_blend.model import features as reversion_features, signal as reversion_signal
from replay_lab.alpha_rebuild.common import limits

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'replay_lab/output/tail_calibration/20261014-v1'
SYMBOLS = ('BTC','ETH','HYPE','SP500','EWY','EWJ','BRENT','GOLD','USDEUR','USDJPY')


def channel_trend(a, allowed, symmetric=False):
    """Same daily channels and lifecycle mixture, with no named-asset filter."""
    n, width = a['close'].shape
    positions = np.zeros((4, width)); stops = np.zeros_like(positions)
    enabled = np.zeros_like(positions, dtype=bool)
    daily = []; sigma = np.full(width, np.nan)
    out = np.zeros((n, width)); vol = np.full_like(out, np.nan)
    held = np.zeros(width); blend = np.full(4, .25); trend_start = None
    for i, known in enumerate(a['known']):
        if known % 86400 == 0:
            daily.append(a['close'][i].copy()); d = np.asarray(daily)
            if len(d) >= 31:
                sigma = np.maximum(np.diff(np.log(d[-21:]), axis=0).std(0, ddof=1)*np.sqrt(365), .01)
                current = d[-1]
                for h, length in enumerate((5,10,20,30)):
                    upper, lower = d[-length:].max(0), d[-length:].min(0)
                    middle = (upper+lower)/2
                    old = positions[h].copy()
                    leave = (old == 1) & (current <= stops[h])
                    stay = (old == 1) & ~leave
                    enter = (old == 0) & (upper > lower) & (current >= upper)
                    positions[h, leave] = 0
                    stops[h, stay] = np.maximum(stops[h, stay], middle[stay])
                    positions[h, enter] = 1; stops[h, enter] = middle[enter]
                    if symmetric:
                        leave_short = (old == -1) & (current >= stops[h])
                        stay_short = (old == -1) & ~leave_short
                        enter_short = (old == 0) & (upper > lower) & (current <= lower)
                        positions[h, leave_short] = 0
                        stops[h, stay_short] = np.minimum(stops[h, stay_short], middle[stay_short])
                        positions[h, enter_short] = -1
                        stops[h, enter_short] = middle[enter_short]
                active = (np.median(abs(d[-30:]/d[-31:-1]-1), axis=0) >= .005) & allowed
                raw = positions.mean(0)*active
                changes = np.diff(np.log(d[-8:]), axis=0)
                confidence = (np.log(d[-1])-np.log(d[-8])) / np.maximum(abs(changes).sum(0), 1e-12)
                enabled &= positions != 0
                enabled |= (positions != 0) & (positions*confidence[None, :] >= .4)
                state = (enabled*positions).mean(0)*active
                if np.any(state != 0):
                    if trend_start is None: trend_start = int(known)
                else: trend_start = None
                age = 0 if trend_start is None else (known-trend_start)/86400
                fraction = min(1., age/14)
                blend = (1-fraction)*np.array([.4,.3,.2,.1])+fraction*np.array([.1,.2,.3,.4])
                held = (blend[:,None]*enabled*positions).sum(0)*active
        vol[i] = sigma
        if np.isfinite(sigma).all(): out[i] = held*np.minimum(.25/np.maximum(sigma,.01),2)*.01
    return out, vol


def ema_trend(a, volatility, policy, allowed, symmetric=False):
    """Same selected completed-daily 10/60 EMA and 3-ATR trailing exit."""
    n, width = a['close'].shape
    frame = policy['frame']; ids = np.flatnonzero(a['known'] % (frame*3600) == 0)
    positions = np.searchsorted(ids, np.arange(n), side='right')-1
    banks = {}
    for period in (policy['fast'], policy['slow']):
        values = np.column_stack([ema(a['close'][ids,j], period) for j in range(width)])
        full = np.full((n,width), np.nan); ready = positions >= 0
        full[ready] = values[positions[ready]]; banks[period] = full
    fast, slow = banks[policy['fast']], banks[policy['slow']]
    state = np.zeros(width); stops = np.full(width, -np.inf)
    prior_exit = np.zeros(width, dtype=np.int64)
    daily = []; ranges = []; previous = -1
    atr = np.full(width, np.nan); efficiency = np.zeros(width)
    out = np.zeros((n,width)); states = np.zeros_like(out)
    if policy['breakout'] or policy['reentry']:
        raise ValueError('This extension preserves the selected zero-breakout/zero-reentry policy only')
    for i, known in enumerate(a['known']):
        price = a['close'][i]
        if known % 86400 == 0:
            daily.append(price.copy()); old = daily[-2] if len(daily)>1 else price
            high = a['high'][previous+1:i+1].max(0); low = a['low'][previous+1:i+1].min(0)
            ranges.append(np.maximum(high-low, np.maximum(abs(high-old), abs(low-old)))); previous = i
            if len(daily)>=20: atr = np.mean(ranges[-20:], axis=0)
            if len(daily)>=8:
                changes = np.diff(np.log(np.asarray(daily[-8:])), axis=0)
                efficiency = changes.sum(0)/np.maximum(abs(changes).sum(0),1e-12)
        if known % (policy['decision_hours']*3600) == 0:
            ready = np.isfinite(atr) & np.isfinite(slow[i])
            up = (fast[i]>slow[i]) & (price>fast[i]*1.0038) & (efficiency>=policy['confidence']) & ready & allowed
            down = (fast[i]<slow[i]) & (price<fast[i]*(1-.0038)) & (efficiency<=-policy['confidence']) & ready & allowed
            existing = state>0
            stops = np.where(existing, np.maximum(stops,price-policy['atr_multiple']*atr), stops)
            leave = existing & ((price<=stops) | ((fast[i]<slow[i]) & policy['exit_cross']))
            if symmetric:
                short = state<0
                stops = np.where(short, np.minimum(stops,price+policy['atr_multiple']*atr), stops)
                leave |= short & ((price>=stops) | ((fast[i]>slow[i]) & policy['exit_cross']))
            state[leave]=0; prior_exit[leave]=known
            enter = up & (state==0) & ~leave & (known-prior_exit>=86400)
            state[enter]=1; stops[enter]=price[enter]-policy['atr_multiple']*atr[enter]
            if symmetric:
                enter_short = down & (state==0) & ~leave & (known-prior_exit>=86400)
                state[enter_short]=-1
                stops[enter_short]=price[enter_short]+policy['atr_multiple']*atr[enter_short]
        states[i]=state
        budget=np.where(np.isfinite(volatility[i]),np.minimum(.25/np.maximum(volatility[i],.01),2)*.01,0)
        out[i]=state*budget*policy['scale']
    return out, states


def build(data, forecast, probability, *, scope='all', symmetric=False):
    if scope not in ('all','legacy'): raise ValueError('Unknown universe')
    shape=(len(data['market_times']),len(SYMBOLS))
    if any(np.asarray(data[k]).shape!=shape for k in ('close','open','high','low','volume','trades','funding','observed')):
        raise ValueError('All ten markets require complete aligned history')
    if not np.array_equal(data['known'],data['market_times']): raise ValueError('Unsettled source timestamps')
    if forecast.shape!=shape or probability.shape!=shape: raise ValueError('Forecast shape mismatch')
    if not np.isfinite(forecast).all() or not np.isfinite(probability).all(): raise ValueError('Invalid predictions')
    m=Market(data['market_times'],data['close'],data['volume'],data['trades'],data['funding'],data['observed'],SYMBOLS)
    bundle=json.loads((FROZEN/'baseline.json').read_text())
    v7=bundle['signal_bundle']['v7']; core=v7['control_bundle']
    if v7['forecast_mode']!='fixed_half_forecast': raise ValueError('Changed retained funding method')
    allowed=np.ones(len(SYMBOLS),bool) if scope=='all' else np.arange(len(SYMBOLS))<3
    channel,vol=channel_trend(data,allowed,symmetric=symmetric)
    policy=dict(core['selected_ema']['policy'])
    if symmetric: policy['exit_cross']=True
    trend,trend_state=ema_trend(data,vol,policy,allowed,symmetric=symmetric)
    carry=funding_weights(m,.5*funding_signals(m)[1]['forecast'])
    base=((trend+channel)/2+carry)*core['scale']*v7['risk_scale']
    sigma=np.full((len(m.times),10,10),np.nan)
    for i in range(168,len(m.times)): sigma[i]=covariance(m,i,bundle['protocol']['risk'])[1]
    parts,_=components(m,sigma,base,forecast,probability)
    allocation=json.loads((FROZEN/'allocation_fit.json').read_text())
    frozen_base=policies(parts,allocation)['v10_frozen']
    f=reversion_features(m,7)
    reversal=reversion_signal(m,f,7,'all' if scope=='all' else 'crypto','aligned')
    raw=np.nan_to_num(reversal/np.maximum(f['vol'],.003),nan=0)
    reversal_weight=.9*targets(raw,sigma,m.times,days=3,risk=.001)
    scale=json.loads((ROOT/'strategy/profiles.json').read_text())['default']['total_scale']
    result=limits(frozen_base+.5*reversal_weight)*scale
    if not np.isfinite(result).all(): raise ValueError('Nonfinite all-market targets')
    return result,dict(ema=trend_state,channel=channel,carry=carry,reversion=reversal,
                      final=result,sigma=sigma,allowed=allowed)
