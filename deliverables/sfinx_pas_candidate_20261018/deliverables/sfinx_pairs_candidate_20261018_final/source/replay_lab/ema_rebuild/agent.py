"""Offline decision adapter. Shared features/controller; no signed-order capability."""
from dataclasses import asdict
import numpy as np
from .engine import Controller
from .guard import inputs, score, defensive_flags
from .models import build
from ..common_comparison.models import covariance


class OfflineEMAAgent:
    def __init__(self, key, protocol, selected_ema, vae_state, saved=None):
        self.key, self.protocol = key, protocol
        self.selected_ema, self.vae_state = selected_ema, vae_state
        saved = saved or {}
        self.controller = Controller(**saved.get('controller', {}))
        self.peak = saved.get('peak', protocol['initial'])
        self.halted = saved.get('halted', False)

    def prepare_history(self, market, original, manifest):
        self.market = market
        self.model = build(original, manifest, self.selected_ema)[self.key]
        self.flags = (defensive_flags(score(inputs(market), self.vae_state), market.times, self.vae_state)
                      if self.key.endswith('_vae') else np.zeros(len(market.times), dtype=bool))

    def decide(self, time, qty, equity, valid_input=True):
        if not valid_input:
            return None
        i = self.market.exact(time)
        if i < 168:
            return None
        _, sigma = covariance(self.market, i, self.protocol['risk'])
        if not np.isfinite(equity) or equity <= 0:
            return None
        self.peak = max(self.peak, equity)
        self.halted |= equity/self.peak-1 <= -self.protocol['risk']['drawdown_halt']
        return self.controller.propose(time, self.market.close[i], np.asarray(qty), equity,
            self.model['target_weight'][i, :3], sigma, self.market.volume[i], self.market.trades[i],
            self.protocol, defensive=bool(self.flags[i]), halted=self.halted, valid_input=valid_input)

    def snapshot(self):
        return dict(controller=asdict(self.controller), peak=self.peak, halted=bool(self.halted))
