"""Reviewable repaired coordinator. Not connected to the running service."""
import copy
import hashlib
import json
from decimal import Decimal
import numpy as np
from competition_live.coordinator import Portfolio
from competition_live.candidate import build_candidate
from replay_lab.common_comparison.engine import rounded, risk_breaches


def publication_with_repairs(publication,weights,protected):
    result=copy.deepcopy(publication)
    weights=np.asarray(weights,float);protected=np.asarray(protected,bool)
    if weights.shape!=(10,) or protected.shape!=(10,) or not np.isfinite(weights).all():
        raise ValueError('Invalid repaired publication')
    frame=result['data']['frame']
    frame.update(weights=weights.tolist(),risk_protected=protected.tolist(),
                 risk_weight_caps=abs(weights).tolist(),repair_contract='base_owner_only')
    result['data']['incumbent']=weights.tolist()
    result['data']['candidate']=weights.tolist()
    packed=json.dumps(frame,sort_keys=True,allow_nan=False).encode()
    result['data']['publication_sha256']=hashlib.sha256(packed).hexdigest()
    return result


class RepairPortfolio(Portfolio):
    def protected_target(self,target,marks,equity):
        frame=getattr(self,'repair_frame',None)
        if frame is None:return target.copy(),False
        mask=np.asarray(frame['risk_protected'],bool)
        if not mask.any():return target.copy(),False
        key=(frame['source_time'],tuple(frame['risk_weight_caps']),tuple(frame['risk_protected']))
        # A model risk cut creates one quantity ceiling per publication, not
        # a new trim on every mark-price tick. Global account risk checks remain
        # continuous and use current marks in the inherited executor.
        if getattr(self,'risk_cap_source',None)!=key:
            self.risk_cap_units=np.array([float(Decimal(str(float(w)))*Decimal(str(float(equity)))/
                Decimal(str(float(px)))) for w,px in zip(frame['risk_weight_caps'],marks)])
            self.risk_cap_source=key
        caps=self.risk_cap_units
        trimmed=np.where(mask,np.sign(target)*np.minimum(abs(target),caps),target)
        trimmed=np.array([rounded(q,d) for q,d in zip(trimmed,self.p['execution']['decimals'])])
        return trimmed,bool(np.any(abs(trimmed-target)>1e-10))

    def submit_baseline(self,now,target,streams,marks,sigma,forced):
        target,cut=self.protected_target(target,marks,self.account.reconcile(marks))
        return super().submit_baseline(now,target,streams,marks,sigma,forced or cut)

    def step(self,now,streams,publication,forecasts):
        self.repair_frame=None
        frame=publication['data']['frame'] if publication else None
        eligible=(frame is not None and frame.get('repair_contract')=='base_owner_only'
                  and publication['received_ms']/1000<=now
                  and 0<=now-frame['source_time']<21600)
        if eligible:
            caps=np.asarray(frame['risk_weight_caps'],float)
            mask=np.asarray(frame['risk_protected'])
            if caps.shape!=(10,) or mask.shape!=(10,) or not np.isfinite(caps).all() or np.any(caps<0):
                raise ValueError('Invalid risk cap contract')
            self.repair_frame=frame
        active=self.plan['paper_start_unix']<=now<self.plan['capture_end_unix']-self.plan['end_liquidation_seconds']
        if eligible and active and now>=self.ready_at and not self.account.unknown and not self.halted:
            self.settlement(now*1000,self.account)
            marks,_=self.prices(now,streams)
            if marks is not None:
                equity=self.account.reconcile(marks);sigma=np.asarray(frame['sigma'])
                globally_forced=bool(risk_breaches(self.account.physical.qty,equity,marks,sigma,self.p['risk']))
                globally_forced |= equity/max(self.peak,equity)-1<=-self.p['risk']['drawdown_halt']
                if not globally_forced:
                    held=self.account.owners['base'].qty.copy()
                    target,cut=self.protected_target(held,marks,equity)
                    if cut:
                        self.base_task=None
                        self.events.append(dict(time=now,kind='risk_cap_request',source=frame['source_time'],
                                                prior_qty=held.tolist(),target_qty=target.tolist()))
                        self.submit_baseline(now,target,streams,marks,sigma,True)
                        return
        return super().step(now,streams,publication,forecasts)


def build_repaired_candidate(profile,settlement,*,account=None):
    candidate=build_candidate(profile,settlement,account=account)
    original=candidate['portfolio']
    portfolio=RepairPortfolio(original.name,'base',candidate['plan'],settlement)
    portfolio.account=original.account
    candidate['portfolio']=portfolio
    candidate['repair_status']='isolated_candidate_not_deployed'
    return candidate
