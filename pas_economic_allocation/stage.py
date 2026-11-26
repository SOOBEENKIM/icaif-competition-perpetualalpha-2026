"""Build a reviewable current candidate frame without training or ordering.

Reference forecasts remain reference forecasts. A PAS venue bridge must preserve
price/carry source semantics; this export is not proof of PAS profitability.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
# Import research implementations before profile adds its frozen dependency path.
from strategy_repairs.model import build
from strategy_final_risk.model import final_mix
from replay_lab.common_comparison.models import Market
from .model import features,predict
from .anchored import RATIO
from .study import OUT


def main():
    from competition_live.profile import ROOT,load_profile,verify_deployment
    from competition_live.producer import read_reference
    from pas_adaptive_development.study import save,sha
    profile=load_profile();verify_deployment()
    data,original,digest=read_reference(ROOT,profile)
    pointer=json.loads((ROOT/'practice_live/latest.json').read_text())
    saved=dict(np.load(ROOT/pointer['run']/'targets.npz',allow_pickle=False))
    arm=dict(scope='legacy',short=False,cross_exit=False,risk=True,original_size=False)
    _,parts=build(data,saved['chronos_forecast'],saved['frozen_regime_probability'],arm)
    market=Market(data['market_times'],data['close'],data['volume'],data['trades'],data['funding'],data['observed'],tuple(profile['trading']['symbols']))
    anchors,trace=final_mix(market,parts['ungated_sized_trend'],.5*parts['reversion_weight'],
        float(parts['final_scale']),saved['frozen_regime_probability'],saved['chronos_forecast'])
    # Rebuild old overlap exactly before extending; no silent source/parameter drift.
    frozen=dict(np.load(ROOT/'reports/strategy_repairs_20261021/final_risk_targets.npz'))
    common=np.intersect1d(data['market_times'],frozen['times'])
    a=np.searchsorted(data['market_times'],common);b=np.searchsorted(frozen['times'],common)
    np.testing.assert_allclose(anchors[a],frozen['final_mix_risk'][b],atol=1e-10,rtol=0)
    fit=json.loads((OUT/'fit.json').read_text());x,u,carry=features(data)
    forecasts=predict(x,u,fit)*fit['attenuation']
    source=int(data['market_times'][-1]);now=time.time()
    frame=dict(candidate='budget_lifecycle',created_at=now,source_time=source,source_hash=digest,
        symbols=profile['trading']['symbols'],anchor_weights=(anchors[-1]*RATIO*2).tolist(),
        protected=trace['protected'][-1].tolist(),price_forecast_168h=forecasts[-1].tolist(),
        reference_funding_forecast_168h=carry[-1].tolist(),sigma=parts['sigma'][-1].tolist(),
        volume=data['volume'][-1].tolist(),trades=data['trades'][-1].tolist(),
        risk_aversion=json.loads((OUT/'selected.json').read_text())['selected']['risk_aversion'],
        initial_strategy_budget=50000.,teammate_initial_budget=50000.,
        ownership='same-market virtual ownership',reference_overlap_rows=len(common),
        overlap_max_error=float(abs(anchors[a]-frozen['final_mix_risk'][b]).max()),
        next_normal_decision=(int(now)//21600+1)*21600,orders_sent=0,model_refit=False,
        live_activation=False,pas_bridge_validated=False,
        limitation='Price/carry predictions are from the reference market. They are not PAS expected executable returns. No current PAS orders are generated from this export.')
    if not np.isfinite(forecasts[-1]).all():raise ValueError('Missing current forecast')
    folder=ROOT/'reports/strategy_replacement_review_20261022';folder.mkdir(exist_ok=True)
    save(folder/'candidate_frame.json',frame)
    save(folder/'candidate_frame_manifest.json',{str(ROOT/pointer['inputs']):digest,
        str(OUT/'fit.json'):sha(OUT/'fit.json'),str(folder/'candidate_frame.json'):sha(folder/'candidate_frame.json')})
    print(json.dumps({k:v for k,v in frame.items() if k!='sigma'},allow_nan=False))


if __name__=='__main__':main()
