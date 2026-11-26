"""Risk-filtered allocation with a fixed, prior-exposure budget calibration."""
import json
from pathlib import Path
import numpy as np
from experiments.risk_preserving_20261016.agent import RiskPreservingAgent

def apply_budget(incumbent,retention,multiplier):
    if not 1<=multiplier<4/3:raise ValueError('Budget calibration may not erase a 25% risk reduction')
    incumbent=np.asarray(incumbent,float);retention=np.asarray(retention,float)
    if incumbent.shape!=retention.shape or not np.isfinite(incumbent).all() or not np.isfinite(retention).all() or np.any((retention<0)|(retention>1)):
        raise ValueError('Invalid allocation inputs')
    if np.any((retention<1)&(retention>.75+1e-12)):raise ValueError('Unexpected risk-retention level')
    return incumbent*retention*multiplier

class BudgetMatchedAgent(RiskPreservingAgent):
    PROFILE='risk_cap_budget_matched_v1'
    def __init__(self,saved=None):
        self.multiplier=json.loads((Path(__file__).parent/'artifacts/budget_frozen.json').read_text())['multiplier']
        if saved is not None and saved.get('budget_multiplier')!=self.multiplier:raise ValueError('Budget differs from saved state')
        super().__init__(saved)
    def prepare_history(self,market,original,*,funding_source):
        super().prepare_history(market,original,funding_source=funding_source)
        incumbent=self.incumbent.weights
        self.weights=apply_budget(incumbent,self.retention,self.multiplier)
        self.components['candidate']=self.weights
        self.components['fixed_budget_adjustment']=incumbent*(self.multiplier-1)
        self.components['final_risk_adjustment']=self.weights-incumbent*self.multiplier
        np.testing.assert_allclose(self.components['scaled_trend_before_blend']+
           self.components['reversion_and_blend_cap_delta']+self.components['fixed_budget_adjustment']+
           self.components['final_risk_adjustment'],self.weights,atol=1e-15,rtol=0)
    def decide(self,time,qty,equity,valid_input=True):
        action=super().decide(time,qty,equity,valid_input)
        if action is not None:
            i=self.market.exact(time)
            action['signal_attribution']['fixed_budget_adjustment']=self.components['fixed_budget_adjustment'][i].tolist()
            action['budget_multiplier']=self.multiplier
        return action
    def snapshot(self):return {**super().snapshot(),'budget_multiplier':self.multiplier}

