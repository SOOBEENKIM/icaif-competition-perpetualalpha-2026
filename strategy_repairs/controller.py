"""Carry a risk-reduction contract through rounding, buffering and turnover."""
import numpy as np
from decimal import Decimal
from replay_lab.ema_rebuild.engine import Controller
from replay_lab.common_comparison.engine import rounded


class RiskPreservingController(Controller):
    def __init__(self, times, weights, protected, last_decision=-1):
        super().__init__(last_decision=last_decision)
        self.times=np.asarray(times)
        self.weights=np.asarray(weights)
        self.protected=np.asarray(protected)

    def propose(self,time,prices,qty,equity,raw,sigma,volume,trades,protocol,
                buffer=.025,defensive=False,halted=False,valid_input=True):
        action=super().propose(time,prices,qty,equity,raw,sigma,volume,trades,protocol,
                              buffer,defensive,halted,valid_input)
        i=int(np.searchsorted(self.times,time,side='right')-1)
        if not valid_input or i<0 or time-self.times[i]>=21600:
            return action
        if not all(np.isfinite(x).all() for x in (prices,qty,sigma,raw)) or not np.isfinite(equity) or equity<=0 or np.any(prices<=0):
            return action
        mask=self.protected[i]
        if not np.any(mask):
            return action
        cap=np.array([float(Decimal(str(abs(float(w))))*Decimal(str(float(equity)))/
                            Decimal(str(float(price)))) for w,price in zip(self.weights[i],prices)])
        proposed=np.asarray(action['target_qty'],float) if action is not None else qty.copy()
        trimmed=np.where(mask,np.sign(proposed)*np.minimum(abs(proposed),cap),proposed)
        trimmed=np.array([rounded(x,d) for x,d in zip(trimmed,protocol['execution']['decimals'])])
        changed=np.any(abs(trimmed-proposed)>1e-10)
        if changed:
            action=dict(action or {},target_qty=trimmed.tolist(),risk_forced=True,
                        defensive=defensive,risk_cap_applied=np.flatnonzero(mask & (abs(trimmed)<abs(proposed))).tolist())
        return action
