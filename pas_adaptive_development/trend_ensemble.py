"""Symmetric multi-asset trend sleeve, frozen rules rather than fitted labels.

This is a declared development experiment, not a claim of a newly independent
test or a reproduction of a paper's empirical results.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from .study import ROOT, OUT as PARENT, sha, save
from . import covariance_v2 as runner

OUT=ROOT/'reports/pas_trend_ensemble_20261021'
ARMS=['incumbent','size_only','trend','blend','blend_restored','exposure_control']


def signal(d):
    lp=pd.DataFrame(np.log(d['close']));hourly=lp.diff()
    vol=hourly.ewm(span=168,min_periods=168,adjust=False).std().clip(lower=.0005).to_numpy()
    # Equal weighting of weekly and monthly trends, no asset or test-date rule.
    scores=[];drifts=[]
    for h in [168,672]:
        delta=(lp-lp.shift(h)).to_numpy()
        scores.append(np.tanh(delta/np.maximum(vol*np.sqrt(h),1e-12)))
        drifts.append(delta/h)
    conviction=np.mean(scores,axis=0)
    expected=np.mean(drifts,axis=0)*96
    carry=pd.DataFrame(np.where(d['observed'],d['funding'],np.nan)).shift(1).rolling(24,min_periods=12).mean().fillna(0).to_numpy()*96
    return conviction,expected-carry,vol


def targets(d,sigma,trading):
    from replay_lab.common_comparison.engine import project
    conviction,net,vol=signal(d);t=d['market_times'];out=np.zeros_like(net)
    held=np.zeros(10);entered=np.full(10,-10**12)
    for i,now in enumerate(t):
        if now%21600 or i<672:
            out[i]=held;continue
        # Flat/flip decisions use six-hour signals; size updates occur daily.
        valid=np.isfinite(net[i])&(d['volume'][i]>0)&(d['trades'][i]>0)
        side=np.sign(conviction[i]);exiting=(held!=0)&((held*net[i]<=0)|(held*side<=0))
        held[exiting]=0
        enter=(held==0)&~exiting&valid&(abs(net[i])>.0043)&(side*net[i]>0)
        desired=conviction[i]*.001/(np.sqrt(10)*vol[i]*np.sqrt(24))
        desired=np.clip(np.nan_to_num(desired,nan=0),-.15,.15)
        held[enter]=desired[enter];entered[enter]=now
        if now%86400==0:
            resize=(held!=0)&valid&~enter
            held[resize]=desired[resize]
        # A weak consensus is NOT rescaled up to spend the full risk budget.
        risk=np.sqrt(max(0,24*held@sigma[i]@held))
        if risk>.001:held*=.001/risk
        held=project(held,trading['initial'],sigma[i],trading['risk'])
        out[i]=held
    return out


def prepare():
    from competition_live.profile import load_profile,verify_deployment
    from replay_lab.common_comparison.engine import project
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    parent=json.loads((PARENT/'protocol.json').read_text());profile=load_profile()
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,nominee='blend_restored',
        hypothesis='Existing base is crypto-centric and long/flat. A low-turnover symmetric weekly/monthly trend sleeve may diversify directions and markets without a high-variance fitted forecaster.',
        horizons_hours=[168,672],holding_horizon_for_cost_gate_hours=96,
        formula='Equal-weight tanh of trailing log returns / past hourly volatility / sqrt(lookback). Expected 96h move is equal-weight trailing drift *96 less prior 24h mean funding *96.',
        entry_cost_buffer=.0043,entry='Signal direction and net forecast agree, absolute expected net move exceeds cost. Exit at sign/thesis failure; no forced minimum number of shorts.',
        allocation='Inverse volatility, raw .001 daily risk /sqrt(10); scale only DOWN to .001 daily covariance risk. Daily size refresh, six-hour exit/entry. Same existing account limits.',
        blending='Equal target-weight average of existing base and new sleeve. Restored version uses existing 1.44117372816997 ratio, not a fitted leverage optimum.',
        exposure_control='Mean blended restored gross / incumbent gross during July1-29 only; scalar frozen before evaluations.',
        selection='Nominee blend_restored fixed; components retained as ablations. No tuning after viewing results. Same six reference windows and costs.',
        promotion='Higher PnL and Q than incumbent and size-only in continuous and later windows, both base/stress. Then separate PAS validation.',
        model_fit='No supervised coefficients or calibration; fixed economic rules, same historical market data as other arms.',
        prior_results='v2 hedging preserves original quantity changes but late stress deteriorates; all prior failures retained.',
        independence=False,orders_sent=0,live_changed=False,pointer=parent['pointer'],
        deployment=verify_deployment()['files'],parent_targets_sha=sha(PARENT/'targets.npz'))
    save(OUT/'protocol.json',p)
    z=dict(np.load(PARENT/'targets.npz',allow_pickle=False));d=dict(np.load(ROOT/p['pointer']['inputs'],allow_pickle=False))
    w={k:z[k] for k in ['incumbent','size_only']}
    w['trend']=targets(d,z['sigma'],profile['trading']);w['blend']=(w['incumbent']+w['trend'])/2
    w['blend_restored']=w['blend']*1.44117372816997
    cal=(z['times']>=1782864000)&(z['times']<1785283200)
    scalar=abs(w['blend_restored'][cal]).sum(1).mean()/abs(w['incumbent'][cal]).sum(1).mean()
    w['exposure_control']=w['incumbent']*scalar
    for name in ARMS:
        if name in ['incumbent','size_only']:continue
        for i in range(168,len(z['times'])):
            w[name][i]=project(w[name][i],profile['trading']['initial'],z['sigma'][i],profile['trading']['risk'])
    np.savez_compressed(OUT/'targets.npz',times=z['times'],sigma=z['sigma'],**w)
    save(OUT/'calibration.json',dict(exposure_scalar=float(scalar)))
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in
        [Path(__file__),Path(runner.__file__),OUT/'protocol.json',OUT/'targets.npz',OUT/'calibration.json']})
    print('Prepared declared symmetric trend family',flush=True)


def run():
    runner.OUT=OUT;runner.ARMS=ARMS;runner.run()


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
