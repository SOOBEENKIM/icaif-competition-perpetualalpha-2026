"""Longer-horizon all-market trend hypothesis; no fitted test-date choices."""
import copy,gzip,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .policy import EconomicController
from .anchored import AnchoredController,RATIO
from .study import OUT as ECONOMIC
from pas_adaptive_development.study import ROOT,OUT as PRIOR,save,sha,native

OUT=ROOT/'reports/pas_economic_slow_20261022'


def forecast(prices):
    log=np.log(prices);out=np.full_like(log,np.nan)
    # Fixed 60/120-day mean log return extrapolated one week. This is an
    # uncalibrated trend hypothesis, not an estimated probability of profit.
    h1,h2=60*24,120*24
    out[h2:]=.5*168*((log[h2:]-log[h2-h1:-h1])/h1+(log[h2:]-log[:-h2])/h2)
    return out


def prepare():
    from competition_live.profile import verify_deployment
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    parent=json.loads((PRIOR/'protocol.json').read_text())
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),pointer=parent['pointer'],
      nominee='slow_augmented',arms=['incumbent','size_only','slow_augmented','slow_unified'],
      hypothesis='Shorter7/28day extensions and weak fitted weekly forecasts have not improved the complete comparison. Add slow60/120day drift signals to all markets with actual-inventory cost hysteresis; preserve the corrected incumbent where active.',
      forecast='Equal average of60day and120day mean log returns, projected168h. Fixed two horizons, no fit, no test-date or asset selection. Not a calibrated probability/guaranteed return. All ten assets use identical rules.',
      allocator='Reuse sealed19bp one-way cost,5bp buffer, current inventory, final gates, and shared-capacity trims. July-selected risk aversion5 fixed. Original pre-gate scale1.44117372816997; no risk-budget increase beyond original profile.',
      selection='slow_augmented is primary before results; standalone is an ablation. Existing pair expert retained in every comparison.',
      promotion='Beat incumbent PnL and Q for full and later intervals in both cost scenarios with zero engineering violations. Size-only, same-budget corrected-anchor and all prior failures retained. PAS separate.',
      interpretation='Any gains may be market exposure/diversification rather than a learned forecast improvement; disclose realized exposure, signed positions, costs, volatility and drawdown.',
      sources=['https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum'],
      independent=False,history_reused=True,changed_after_previous_results=True,
      live_changed=False,orders_sent=0,deployment=verify_deployment()['files'])
    save(OUT/'protocol.json',p)
    d=dict(np.load(ROOT/p['pointer']['inputs']));pred=forecast(d['close'])
    prior=dict(np.load(ECONOMIC/'forecasts.npz'))
    np.savez_compressed(OUT/'forecasts.npz',times=d['market_times'],forecast=pred,carry=prior['carry'])
    files=[Path(__file__),Path(__file__).with_name('policy.py'),Path(__file__).with_name('anchored.py'),Path(__file__).with_name('replay.py'),
           OUT/'protocol.json',OUT/'forecasts.npz']
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in files})


def run():
    from competition_live.profile import load_profile,BUNDLE,verify_deployment
    from .replay import replay
    from pas_allocation_study.study import summarize
    p=json.loads((OUT/'protocol.json').read_text())
    for rel,h in json.loads((OUT/'source_freeze.json').read_text()).items():assert sha(ROOT/rel)==h
    profile=load_profile();d=dict(np.load(BUNDLE/'data/reference.npz'));s=dict(np.load(BUNDLE/'models/reference_gate_state.npz'))
    z=dict(np.load(OUT/'forecasts.npz'));ix=np.searchsorted(z['times'],d['times'])
    a=dict(np.load(ROOT/'reports/strategy_repairs_20261021/final_risk_targets.npz'));ia=np.searchsorted(a['times'],d['times'])
    np.testing.assert_array_equal(z['times'][ix],d['times']);np.testing.assert_array_equal(a['times'][ia],d['times'])
    cfg=copy.deepcopy(profile['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
    lam=json.loads((ECONOMIC/'selected.json').read_text())['selected']['risk_aversion']
    windows=[w for w in json.loads((BUNDLE/'expected.json').read_text())['runs'] if w['venue']=='reference' and w['arm']==profile['candidate']]
    controls=json.loads((ROOT/'reports/pas_covariance_v2_20261021/reference_results.json').read_text());rows=[]
    for w in windows:
        rows.extend([r for r in controls if r['arm'] in ['incumbent','size_only'] and r['phase']==w['phase'] and r['scenario']==w['scenario']])
        for arm in p['arms'][2:]:
            file=OUT/'reference'/f"{arm}_{w['phase']}_{w['scenario']}.json.gz"
            if file.exists():
                with gzip.open(file,'rt') as f:result=json.load(f)
            else:
                if arm=='slow_augmented':c=AnchoredController(d['times'],z['forecast'][ix],z['carry'][ix],a['final_mix_risk'][ia]*RATIO,a['protected'][ia],lam)
                else:c=EconomicController(d['times'],z['forecast'][ix],z['carry'][ix],lam)
                result=replay(d,s,None,cfg,profile['trading'],profile['candidate'],w['scenario'],w['start'],w['end'],base_policy=c)
                result['allocation_trace']=c.trace
                file.parent.mkdir(exist_ok=True)
                with gzip.open(file,'wt') as f:json.dump(result,f,default=native,allow_nan=False)
            row=summarize(result,arm,w['phase'],w['scenario']);rows.append(row)
            assert row['timing_pass'] and row['accounting_error']<1e-6
            save(OUT/'reference_results.json',rows);print(json.dumps(row),flush=True)
    assert verify_deployment()['files']==p['deployment']


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
