"""Frozen v10 sizing ensemble. Computes order intents; no exchange connection."""
import copy
from dataclasses import asdict
import numpy as np
from ..competition_rebuild.candidate_agent import CausalHybridAgent
from ..common_comparison.models import covariance
from ..ema_rebuild.engine import Controller
from .lifecycle_allocation import targets

class SizingEnsembleAgent:
    def __init__(self,bundle,checkpoint,saved=None):
        if bundle['kind']!='jm_chronos_sizing_ensemble_v10':raise ValueError('Unsupported candidate')
        if bundle['relaxation_days']!=[3,7] or bundle['risk_headroom']!=.9:raise ValueError('Unfrozen candidate parameters')
        self.bundle=copy.deepcopy(bundle);self.signal=CausalHybridAgent(bundle['signal_bundle'],checkpoint)
        state=saved or {};self.controller=Controller(**state.get('controller',{}))
        self.peak=float(state.get('peak',bundle['protocol']['initial']));self.halted=bool(state.get('halted',False))
        self.last_observation=max(int(state.get('last_observation',-1)),self.controller.last_decision)
        self.market=None

    def prepare_history(self,market,original,*,funding_source):
        # A failed refresh must not leave old signals eligible for new orders.
        self.market=None
        self.validate_history(market,original)
        self.signal.prepare_history(market,original,funding_source=funding_source)
        sigma=np.full((len(market.times),10,10),np.nan)
        for i in range(168,len(market.times)):sigma[i]=covariance(market,i,self.bundle['protocol']['risk'])[1]
        paths=[targets(self.signal.weights,sigma,market.times,days=d,risk=self.bundle['daily_risk']) for d in self.bundle['relaxation_days']]
        self.weights=self.bundle['risk_headroom']*np.mean(paths,axis=0);self.market=copy.deepcopy(market);self.sigma=sigma

    @staticmethod
    def validate_history(market,original):
        n=len(market.times);shape=(n,10)
        if (np.asarray(market.times).shape!=(n,) or not np.isfinite(market.times).all()
                or np.any(market.times%3600) or np.any(np.diff(market.times)!=3600)):
            raise ValueError('History must contain contiguous completed hourly timestamps')
        if not np.array_equal(original['times']+3600,market.times):
            raise ValueError('Candle open/close timestamps differ by more than one hour')
        for name in ('open','high','low','close','volume','trades'):
            value=np.asarray(original[name])
            if value.shape!=shape or not np.isfinite(value).all():
                raise ValueError('Invalid original '+name)
            if np.any(value<0) or (name in ('open','high','low','close') and np.any(value==0)):
                raise ValueError('Invalid original '+name)
        if (np.any(original['high']<np.maximum(original['open'],original['close']))
                or np.any(original['low']>np.minimum(original['open'],original['close']))):
            raise ValueError('Inconsistent OHLC range')
        for name in ('close','volume','trades'):
            if not np.array_equal(getattr(market,name),original[name]):
                raise ValueError('Market/source '+name+' mismatch')
        for name in ('funding','observed'):
            if np.asarray(getattr(market,name)).shape!=shape:
                raise ValueError('Invalid market '+name+' shape')

    def decide(self,time,qty,equity,valid_input=True):
        if (self.market is None or not valid_input or not np.isfinite(equity) or equity<=0
                or not np.isfinite(time) or time!=int(time)
                or time<=max(self.last_observation,self.controller.last_decision)):return None
        qty=np.asarray(qty,dtype=float)
        if qty.shape!=(10,) or not np.isfinite(qty).all():return None
        try:i=self.market.exact(time)
        except ValueError:return None
        if i<1439:return None
        if not np.isfinite(self.weights[i]).all() or not np.isfinite(self.sigma[i]).all():return None
        p=self.bundle['protocol'];self.peak=max(self.peak,equity)
        self.halted|=equity/self.peak-1<=-p['risk']['drawdown_halt']
        self.last_observation=int(time)
        return self.controller.propose(time,self.market.close[i],qty,equity,self.weights[i],self.sigma[i],
             self.market.volume[i],self.market.trades[i],p,buffer=.2,halted=self.halted,valid_input=valid_input)

    def snapshot(self):return dict(controller=asdict(self.controller),peak=self.peak,halted=bool(self.halted),last_observation=self.last_observation)
