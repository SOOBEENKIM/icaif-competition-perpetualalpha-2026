import copy,numpy as np
from ..v10_revision.agent import RevisionAgent
from .model import build

class ReversionBlendAgent(RevisionAgent):
    def __init__(self,bundle,checkpoint,regime_fit,allocation_fit,selection,saved=None):
        if regime_fit['last_training_time']>=regime_fit['cutoff']:raise ValueError('Future regime observations')
        self.selection=copy.deepcopy(selection);self.deployment_start=int(selection['test_start'])
        if regime_fit['cutoff']>self.deployment_start:raise ValueError('Premature deployment')
        if not 0<selection['risk_scale']<=1:raise ValueError('Invalid risk calibration')
        super().__init__(bundle,checkpoint,regime_fit,allocation_fit,'v10_frozen',saved)

    def prepare_history(self,market,original,*,funding_source):
        super().prepare_history(market,original,funding_source=funding_source)
        self.frozen_baseline_weights=self.weights.copy()
        candidates,_=build(market,self.sigma,self.frozen_baseline_weights)
        self.weights=candidates[self.selection['policy']]*self.selection['risk_scale']

    def decide(self,time,qty,equity,valid_input=True):
        if not np.isfinite(time) or time<self.deployment_start:return None
        return super().decide(time,qty,equity,valid_input)
