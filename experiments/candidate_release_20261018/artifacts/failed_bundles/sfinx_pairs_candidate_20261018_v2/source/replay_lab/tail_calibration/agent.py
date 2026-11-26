"""Fixed tail-calibrated pullback ensemble; no dynamic risk timing."""
from ..reversion_blend.agent import ReversionBlendAgent

class TailCalibratedAgent(ReversionBlendAgent):
    def __init__(self,bundle,checkpoint,regime_fit,allocation_fit,selection,saved=None):
        self.extra_scale=selection['incremental_scale']
        if not 0<self.extra_scale<=1:raise ValueError('Invalid fixed risk cushion')
        super().__init__(bundle,checkpoint,regime_fit,allocation_fit,selection['alpha_selection'],saved)

    def prepare_history(self,market,original,*,funding_source):
        super().prepare_history(market,original,funding_source=funding_source)
        self.weights=self.weights*self.extra_scale
        self.same_exposure_weights=self.frozen_baseline_weights*(self.selection['risk_scale']*self.extra_scale)
