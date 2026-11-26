"""Frozen v7 inference: retained price experts plus learned funding persistence."""
import json
import numpy as np
from ..macro_development.candidate_agent import TrendCarryAgent
from ..macro_development.carry import signals
from .funding_model import features,weights

class FundingTrendAgent(TrendCarryAgent):
    def __init__(self,bundle,saved=None):
        if bundle['kind']!='pooled_funding_trend_v2':raise ValueError('Unsupported candidate')
        self.candidate=bundle
        super().__init__(bundle['control_bundle'],saved)

    def prepare_history(self,market,original,*,funding_source):
        super().prepare_history(market,original,funding_source=funding_source)
        state=self.candidate['funding_model']
        if state['last_training_label_end']>=state['cutoff']:raise ValueError('Unmatured training labels')
        mode=self.candidate.get('forecast_mode','nnls')
        if mode=='nnls':forecast=features(market)@np.asarray(state['nnls'])
        elif mode=='fixed_half_forecast':forecast=.5*signals(market)[1]['forecast']
        else:raise ValueError('Unknown forecast mode')
        scale=self.bundle['scale']
        price=self.weights-signals(market)[0]['carry_trend']*scale
        self.weights=(price+weights(market,forecast)*scale)*self.candidate['risk_scale']

    def decide(self,time,qty,equity,valid_input=True):
        if time<self.candidate['funding_model']['cutoff']:return None
        return super().decide(time,qty,equity,valid_input)

def bundle(mode='nnls'):
    from .experiment import OUT,PREVIOUS
    from ..pairs_development.experiment import stamp
    family='equal_risk' if mode=='nnls' else 'ablation'
    return dict(kind='pooled_funding_trend_v2',created_at=stamp(),forecast_mode=mode,
        control_bundle=json.loads((PREVIOUS/'candidate.json').read_text()),
        funding_model=json.loads((OUT/'funding_forecast/fit.json').read_text()),
        risk_scale=json.loads((OUT/f'{family}/scales.json').read_text())[mode],
        selection=('NNLS selected by prespecified early-window tie-break. Same model then rerun at 100% of control calibration risk.' if mode=='nnls' else
                   'Fixed50% forecast attenuation, post-freeze ablation promoted as a separate retrospective candidate. No learned forecast coefficients used in trading.'),
        history_status='All evaluation history already reused; retrospective nomination, not independently validated selection.',
        funding_signal='External mainnet settled funding, lagged one hour. Does not estimate actual pa funding.',
        live_trading_enabled=False,live_orders_sent=0)
