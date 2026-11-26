"""Verified wire contract for the isolated final-risk candidate."""
import copy
import hashlib
import json
from decimal import Decimal
import numpy as np
from experiments.published_pair_lifecycle_20261017.signals import FRAME_KEYS, validate_envelope
from replay_lab.common_comparison.engine import rounded
from strategy_repairs.candidate import RepairPortfolio
from competition_live.candidate import build_candidate


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def publication(original,weights,protected):
    result=copy.deepcopy(original)
    weights=np.asarray(weights,float);protected=np.asarray(protected)
    if weights.shape!=(10,) or protected.shape!=(10,) or protected.dtype!=bool or not np.isfinite(weights).all():
        raise ValueError('Invalid final-risk targets')
    data=result['data'];frame=data['frame']
    frame['weights']=weights.tolist()
    data['incumbent']=weights.tolist();data['candidate']=weights.tolist()
    data['publication_sha256']=digest({key:frame[key] for key in FRAME_KEYS})
    contract=dict(schema='final_mix_risk_v1',source_time=frame['source_time'],
        publication_sha256=data['publication_sha256'],protected=protected.tolist(),
        weight_caps=abs(weights).tolist())
    data['risk_contract']=contract;data['risk_contract_sha256']=digest(contract)
    validate(result)
    return result


def validate(event):
    frame=validate_envelope(event);data=event['data'];contract=data['risk_contract']
    if (contract['schema']!='final_mix_risk_v1' or contract['source_time']!=frame['source_time']
            or contract['publication_sha256']!=data['publication_sha256']
            or digest(contract)!=data['risk_contract_sha256']):
        raise ValueError('Risk contract identity/digest mismatch')
    mask=np.asarray(contract['protected']);caps=np.asarray(contract['weight_caps'],float)
    if mask.shape!=(10,) or mask.dtype!=bool or caps.shape!=(10,):
        raise ValueError('Invalid risk mask/caps')
    if not np.array_equal(caps,abs(np.asarray(frame['weights'],float))):
        raise ValueError('Risk caps differ from published weights')
    if not all(np.array_equal(data[k],frame['weights']) for k in ('incumbent','candidate')):
        raise ValueError('Execution target differs from verified frame')
    return frame,contract


class FinalRiskPortfolio(RepairPortfolio):
    def protected_target(self,target,marks,equity):
        frame=getattr(self,'repair_frame',None)
        if frame is None:return target.copy(),False
        mask=np.asarray(frame['risk_protected'],bool)
        key=(frame['source_time'],tuple(frame['risk_weight_caps']),tuple(mask))
        if getattr(self,'risk_cap_source',None)!=key:
            # Missing flat markets use a zero-notional sentinel; never divide
            # by it or permit new exposure there.
            self.risk_cap_units=np.array([
                float(Decimal(str(float(w)))*Decimal(str(float(equity)))/Decimal(str(float(px))))
                if px>0 else 0. for w,px in zip(frame['risk_weight_caps'],marks)])
            self.risk_cap_source=key
        trimmed=np.where(mask,np.sign(target)*np.minimum(abs(target),self.risk_cap_units),target)
        trimmed=np.array([rounded(q,d) for q,d in zip(trimmed,self.p['execution']['decimals'])])
        return trimmed,bool(np.any(abs(trimmed-target)>1e-10))

    def step(self,now,streams,event,forecasts):
        if event is not None:
            _,contract=validate(event)
            event=copy.deepcopy(event)
            # The inherited coordinator receives a validated in-memory view.
            # Extra fields are NOT incorrectly included in FRAME_KEYS hashing.
            event['data']['frame'].update(repair_contract='base_owner_only',
                risk_protected=contract['protected'],risk_weight_caps=contract['weight_caps'])
        return super().step(now,streams,event,forecasts)


def build(profile,settlement,account):
    result=build_candidate(profile,settlement,account=account)
    original=result['portfolio']
    result['portfolio']=FinalRiskPortfolio(original.name,'base',result['plan'],settlement)
    result['portfolio'].account=account
    return result
