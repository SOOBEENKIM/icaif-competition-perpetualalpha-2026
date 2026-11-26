"""Frozen trend/carry ensemble inference. No exchange or signing capability."""
import json
from dataclasses import asdict
import numpy as np
from .carry import signals
from ..ema_rebuild.models import build
from ..ema_rebuild.engine import Controller
from ..common_comparison.models import covariance

class TrendCarryAgent:
    def __init__(self,bundle,saved=None):
        if bundle['kind']!='fixed_trend_consensus_carry_v1':raise ValueError('Unsupported candidate')
        self.bundle=bundle;saved=saved or {}
        self.controller=Controller(**saved.get('controller',{}))
        self.peak=float(saved.get('peak',bundle['protocol']['initial']))
        self.halted=bool(saved.get('halted',False))

    def prepare_history(self,market,original,*,funding_source):
        if funding_source!='mainnet_reference_settled':raise ValueError('Candidate requires explicitly identified external mainnet funding signals')
        if list(market.symbols)!=self.bundle['symbols']:raise ValueError('Wrong market order')
        if len(market.times)<1440 or np.any(np.diff(market.times)!=3600):raise ValueError('Need at least sixty days of contiguous completed hourly history')
        if not np.array_equal(market.times,original['known']) or not np.array_equal(market.close,original['close']):raise ValueError('Price/source timestamp mismatch')
        if not np.isfinite(market.close).all() or (market.close<=0).any():raise ValueError('Invalid close')
        if not market.observed.all() or not np.isfinite(market.funding).all():raise ValueError('Missing settled funding history; do not substitute zero')
        b=build(original,self.bundle['retained_manifest'],self.bundle['selected_ema'])
        price=(b['ema_base']['target_weight']+b['adaptive-horizons-lifecycle-d14-s1']['target_weight'])/2
        self.weights=(price+signals(market)[0]['carry_trend'])*self.bundle['scale']
        self.market=market

    def decide(self,time,qty,equity,valid_input=True):
        if not valid_input or not np.isfinite(equity) or equity<=0:return None
        i=self.market.exact(time)
        if i<1439:return None
        p=self.bundle['protocol'];self.peak=max(self.peak,equity)
        self.halted|=equity/self.peak-1<=-p['risk']['drawdown_halt']
        sigma=covariance(self.market,i,p['risk'])[1]
        return self.controller.propose(time,self.market.close[i],np.asarray(qty),equity,self.weights[i],sigma,
            self.market.volume[i],self.market.trades[i],p,buffer=.2,halted=self.halted,valid_input=valid_input)

    def snapshot(self):return dict(controller=asdict(self.controller),peak=self.peak,halted=bool(self.halted))

def bundle():
    from .data import ROOT,OUT,load
    from .experiment import protocol
    from ..pairs_development.experiment import digest,stamp
    m,_,_=load()
    return dict(kind='fixed_trend_consensus_carry_v1',created_at=stamp(),symbols=list(m.symbols),
        selected_ema=json.loads((ROOT/'replay_lab/ema_development/selected_policy.json').read_text()),
        retained_manifest=json.loads((ROOT/'replay_lab/output/retained_strategies/manifest.json').read_text()),
        scale=json.loads((OUT/'consensus/scales.json').read_text())['consensus_carry'],protocol=protocol(m),
        price_expert_weights=[.5,.5],rebalance_buffer=.2,
        external_funding_signal='mainnet_reference_settled, lagged one hour; 24h/168h conservative forecast; same rule for ten assets',
        carry_entry_threshold=.0038,carry_exit_threshold=.0019,forecast_horizon_hours=168,
        scale_calibration='June1-July29 reused development; 95% of EMA daily STD, not future risk calibration',
        nomination='Retrospective multiobjective development candidate; not an independently validated winner or the carry-only study primary selection.',
        live_trading_enabled=False,live_orders_sent=0,
        execution_limitations=['Reference close and volume proxy','Funding notionals use reference closes, not missing historical oracle','Daily UTC boundary and population STD are conventions; organizer daily marking answer unresolved','Actual pa liquidity and funding not verified; public testnet probe found only par'],
        sources={str(f.relative_to(ROOT)):digest(f) for f in (ROOT/'replay_lab/macro_development').glob('*.py')})
