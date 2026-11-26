"""Final directional risk ceilings. No network, credentials or exchange actions."""
from dataclasses import asdict
import numpy as np

from replay_lab.competition_rebuild.regimes import paths
from replay_lab.competition_rebuild.persistence import episode_overlay
from replay_lab.common_comparison.engine import project, rounded, risk_breaches
from replay_lab.ema_rebuild.engine import Controller


def target_path(market, incumbent, base, probability, forecast):
    arrays=[incumbent,base,probability,forecast]
    if any(np.shape(a)!=market.close.shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Invalid or misaligned signal arrays')
    _,regime=paths(market,probability,base)
    chronos=episode_overlay(market,base,forecast,'sticky')
    regime_ratio=np.divide(regime,base,out=np.ones_like(base),where=abs(base)>1e-15)
    chronos_ratio=np.divide(chronos,base,out=np.ones_like(base),where=abs(base)>1e-15)
    retention=(regime_ratio+chronos_ratio)/2
    if np.any(retention<0) or np.any(retention>1+1e-12):
        raise ValueError('Risk filters must only attenuate')
    # The inherited filter is directional: do not apply a bearish-on-long gate
    # to an opposite, short reversion position, or invent a gate for a flat base.
    applied=np.where(incumbent*base>0,np.clip(retention,0,1),1.)
    candidate=incumbent*applied
    return {'candidate':candidate,'retention':applied,'regime_retention':regime_ratio,
            'chronos_retention':chronos_ratio,'underlying_trend':base,
            'incumbent':incumbent,'frozen_probability':probability,'forecast':forecast}


class RiskCapController:
    def __init__(self,saved=None):
        self.controller=Controller(**(saved or {}).get('controller',{}))

    @property
    def last_decision(self):return self.controller.last_decision

    def snapshot(self):return {'controller':asdict(self.controller)}

    def propose(self,time,prices,qty,equity,raw,sigma,volume,trades,protocol,
                buffer=.2,defensive=False,halted=False,valid_input=True,retention=None):
        raw=np.asarray(raw,dtype=float)
        retention=np.ones_like(raw) if retention is None else np.asarray(retention,dtype=float)
        if retention.shape!=raw.shape or not np.isfinite(retention).all() or np.any((retention<0)|(retention>1)):
            return None
        action=self.controller.propose(time,prices,qty,equity,raw,sigma,volume,trades,protocol,
             buffer=buffer,defensive=defensive,halted=halted,valid_input=valid_input)
        if action is None:return None
        prior=np.asarray(action['target_qty'],dtype=float)
        ceiling=abs(raw)*equity/prices
        active=retention<1-1e-12
        capped=np.where(active,np.sign(prior)*np.minimum(abs(prior),ceiling),prior)
        # Reducing a hedge can raise net/portfolio risk; recheck after per-leg caps.
        if np.any(capped!=prior):
            capped=project(capped*prices/equity,equity,sigma,protocol['risk'])*equity/prices
        target=np.array([rounded(q,d) for q,d in zip(capped,protocol['execution']['decimals'])])
        if action['risk_forced']:
            target=np.sign(qty)*np.minimum(abs(qty),abs(target))
        if np.any(abs(target[active])>ceiling[active]+1e-10):
            raise AssertionError('Final target restored a risk reduction')
        if risk_breaches(target,equity,prices,sigma,protocol['risk']):
            # Tiny per-asset rounding can remove a hedge: shrink once more.
            target*=.999999
            target=np.array([rounded(q,d) for q,d in zip(target,protocol['execution']['decimals'])])
            if risk_breaches(target,equity,prices,sigma,protocol['risk']):
                target=np.zeros_like(qty)
        action.update(target_qty=target.tolist(),risk_cap_active=active.tolist(),
            risk_cap_qty=ceiling.tolist(),pre_cap_target_qty=prior.tolist(),
            cap_overrode_soft_constraint=(abs(prior)>abs(target)+1e-10).tolist(),
            retention=retention.tolist(),decision_reason='risk_cap' if np.any(abs(prior-target)>1e-10) else 'scheduled_or_hard_risk')
        return action
