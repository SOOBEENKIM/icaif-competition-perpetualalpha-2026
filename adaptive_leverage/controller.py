"""Opt-in replay controller. The running strategy never imports this module."""
import numpy as np
from pas_performance_research.soft_timing import SoftTimingController
from .policy import select_exposure, Settings


class AdaptiveController(SoftTimingController):
    def __init__(self, *args, leverage_settings=Settings(), **kwargs):
        super().__init__(*args, **kwargs)
        self.leverage_settings = leverage_settings
        self.leverage_trace = []

    def propose(self, time, prices, qty, equity, raw, sigma, volume, trades, protocol, **kwargs):
        action = super().propose(time, prices, qty, equity, raw, sigma, volume, trades, protocol, **kwargs)
        if action is None:
            return None
        i = int(np.searchsorted(self.times, time))
        result = select_exposure(base_qty=action['target_qty'], current_qty=qty, prices=prices,
            equity=equity, forecast=self.forecast[i], funding=self.carry[i],
            hourly_covariance=sigma, protected=self.protected[i],
            entry_allowed=self.trace[-1]['entry_filter_pass'], liquid=(volume > 0) & (trades > 0),
            decimals=protocol['execution']['decimals'], risk=protocol['risk'], settings=self.leverage_settings)
        action['target_qty'] = result['target_qty']
        self.leverage_trace.append(dict(time=int(time), **result))
        return action
