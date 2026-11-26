"""Offline order-intent adapter for the frozen validation nominee."""
import numpy as np
from .core import components,policies,frozen_prob
from ..portfolio_rebuild_v10.candidate_agent import SizingEnsembleAgent

class RevisionAgent(SizingEnsembleAgent):
    def __init__(self,bundle,checkpoint,regime_fit,allocation_fit,policy,saved=None):
        super().__init__(bundle,checkpoint,saved)
        self.regime_fit=regime_fit;self.allocation_fit=allocation_fit;self.policy=policy

    def prepare_history(self,market,original,*,funding_source):
        super().prepare_history(market,original,funding_source=funding_source)
        self.original_weights=self.weights.copy()
        self.signal.base.prepare_history(market,original,funding_source=funding_source)
        self.base_weights=self.signal.base.weights.copy()
        prob=frozen_prob(market,self.regime_fit,self.signal.probability)
        parts,_=components(market,self.sigma,self.base_weights,self.signal.forecast,prob)
        self.weights=policies(parts,self.allocation_fit)[self.policy]
        self.frozen_probability=prob

