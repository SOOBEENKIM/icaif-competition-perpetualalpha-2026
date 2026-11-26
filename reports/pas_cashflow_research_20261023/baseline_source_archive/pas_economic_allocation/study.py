"""One sealed fit/calibration/selection pipeline; no live trading imports."""
import copy
import gzip
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from pas_adaptive_development.study import ROOT,OUT as PRIOR,save,sha,ts,native
from . import model

OUT=ROOT/'reports/pas_economic_allocation_20261022'


def prepare():
    from competition_live.profile import verify_deployment
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    old=json.loads((PRIOR/'protocol.json').read_text())
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),pointer=old['pointer'],
      hypothesis='Replace stacked directional sizing with all-market weekly price/carry utility, transaction-cost no-trade regions, actual inventory feedback, and one final risk allocation. Preserve the incumbent pair expert and paid shared-capacity repairs.',
      train_start=ts('2026-03-18T00:00:00Z'),train_end=ts('2026-06-01T00:00:00Z'),
      calibration_start=ts('2026-06-01T00:00:00Z'),calibration_end=ts('2026-07-01T00:00:00Z'),
      selection_start=ts('2026-07-01T12:00:00Z'),selection_end=ts('2026-07-29T12:00:00Z'),
      risk_aversion_grid=[5.,20.,80.],horizon=168,
      model='Pooled ridge on 24h/7d/28d normalized momentum, 7d deviation and volatility ratio. Weekly nonoverlapping labels per asset; overlapping cross-asset rows are not independent days. No asset IDs. Train-only standardization. Ridge1, intercept penalty0.25, normalized labels clipped4. June attenuation in[0,1].',
      funding='Lagged completed rates; signed minimum magnitude of 24h/168h EWM when signs agree. Weekly forecast. Price-risk-bearing carry, not arbitrage. Full expected payment, half receipt; actual settlement charged separately exactly once.',
      allocation='Coordinate convex utility solve with 19bp entry, 19bp paid future exit+5bp buffer, full covariance over168h. Final shared caps only reduce targets. Recompute from ACTUAL inventory every6h. No compulsory short or minimum utilization. Same50000 strategy budget plus50000 comparison reserve.',
      selection='Maximize worst base/stress Q over July1-29, then lower risk aversion on exact tie. One selected lambda. The same lambda is reused by carry-only ablation; no August/September result may alter it.',
      evaluation='Exactly Aug1-Sep1, Sep1-Oct15, and continuous Aug1-Oct15; same original delayed/partial fill, fees, capacity, funding, and terminal close reserve. July selection is earlier than every evaluation interval.',
      promotion='Selected primary exceeds incumbent PnL and Q in both full and later periods under base/stress, zero hard-risk/accounting/timing defects. Compare size-only, carry-only, terminal costs, drawdown, and realized exposure. Improvements must be labelled reused-history development; PAS deployment needs venue evidence separately.',
      data_independence=False,history_reused=True,prior_families=13,nominee='unified',
      copy_trading='Not included: leaders chosen using current PnL cannot be retroactively treated as preselected predictors. No timestamped pre-period leader-selection record exists.',
      live_changed=False,orders_sent=0,deployment=verify_deployment()['files'],
      sources=['https://web.stanford.edu/~boyd/papers/cvx_portfolio.html','https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding'])
    save(OUT/'protocol.json',p)
    d=dict(np.load(ROOT/p['pointer']['inputs'],allow_pickle=False))
    fit=model.train(d,p);x,u,carry=model.features(d)
    forecast=model.predict(x,u,fit)*fit['attenuation']
    forecast[d['market_times']<p['calibration_end']]=np.nan
    save(OUT/'fit.json',fit)
    np.savez_compressed(OUT/'forecasts.npz',times=d['market_times'],forecast=forecast,carry=carry)
    files=[*Path(__file__).parent.glob('*.py'),OUT/'protocol.json',OUT/'fit.json',OUT/'forecasts.npz',ROOT/p['pointer']['inputs']]
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in files})
    print(json.dumps(fit),flush=True)


def run():
    from competition_live.profile import load_profile,BUNDLE,verify_deployment
    from .policy import EconomicController
    from .replay import replay
    from pas_allocation_study.study import summarize
    p=json.loads((OUT/'protocol.json').read_text())
    for name,digest in json.loads((OUT/'source_freeze.json').read_text()).items():
        assert sha(ROOT/name)==digest,name
    profile=load_profile();d=dict(np.load(BUNDLE/'data/reference.npz'))
    states=dict(np.load(BUNDLE/'models/reference_gate_state.npz'))
    z=dict(np.load(OUT/'forecasts.npz'));ids=np.searchsorted(z['times'],d['times'])
    np.testing.assert_array_equal(z['times'][ids],d['times'])
    cfg=copy.deepcopy(profile['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
    rows=[]
    def evaluate(arm,lam,phase,scenario,start,end):
        file=OUT/'reference'/f'{arm}_{phase}_{scenario}.json.gz'
        if file.exists():
            with gzip.open(file,'rt') as f:r=json.load(f)
        else:
            controller=EconomicController(d['times'],z['forecast'][ids],z['carry'][ids],lam,
                                          mode='carry_only' if arm=='carry_only' else 'unified')
            r=replay(d,states,None,cfg,profile['trading'],profile['candidate'],scenario,start,end,base_policy=controller)
            r['allocation_trace']=controller.trace
            file.parent.mkdir(exist_ok=True)
            with gzip.open(file,'wt') as f:json.dump(r,f,default=native,allow_nan=False)
        row=summarize(r,arm,phase,scenario);row['risk_aversion']=lam
        assert row['timing_pass'] and row['accounting_error']<1e-6
        rows.append(row);save(OUT/'running_results.json',rows)
        print(json.dumps(row),flush=True)
        return row
    for lam in p['risk_aversion_grid']:
        for scenario in ['base','stress']:
            evaluate(f'lambda_{lam:g}',lam,'selection',scenario,p['selection_start'],p['selection_end'])
    candidates=[]
    for lam in p['risk_aversion_grid']:
        rr=[r for r in rows if r['risk_aversion']==lam]
        candidates.append(dict(risk_aversion=lam,min_Q=min(r['Q'] for r in rr),
                               valid=all(r['risk_breach_bars']==0 for r in rr)))
    good=[c for c in candidates if c['valid']]
    if not good:raise ValueError('No valid July allocator; no later period selection allowed')
    chosen=max(good,key=lambda c:(c['min_Q'],-c['risk_aversion']))
    selection=dict(selected=chosen,candidates=candidates,chosen_before_later_results=True)
    file=OUT/'selected.json'
    if file.exists():assert json.loads(file.read_text())==selection
    else:save(file,selection)
    lam=chosen['risk_aversion']
    windows=[r for r in json.loads((BUNDLE/'expected.json').read_text())['runs'] if r['venue']=='reference' and r['arm']==profile['candidate']]
    for w in windows:
        for arm in ['unified','carry_only']:
            evaluate(arm,lam,w['phase'],w['scenario'],w['start'],w['end'])
    save(OUT/'reference_results.json',rows)
    assert verify_deployment()['files']==p['deployment']


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
