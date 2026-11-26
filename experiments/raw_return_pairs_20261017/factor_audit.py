"""Causal BTC-factor attribution of recorded pair price cash, not new trades."""
import gzip,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha
from .research import HERE,OUT,PARENT,MATCHED,ARM,verify


def coefficients(prices,window=168):
    returns=pd.DataFrame(prices).pct_change(fill_method=None)
    variance=returns[0].rolling(window,min_periods=window).var(ddof=0)
    beta=np.column_stack([(returns[j].rolling(window,min_periods=window).cov(returns[0],ddof=0)/variance).to_numpy() for j in returns])
    # At cash interval i use only returns observed through i-1.
    prior=np.full_like(beta,np.nan);prior[1:]=beta[:-1]
    return returns.to_numpy(),prior


def main():
    dest=OUT/'factor_audit';dest.mkdir(exist_ok=False)
    cfgpath=ROOT/'experiments/hourly_refresh_pairs_20261017/reference_protocol.json';cfg=json.loads(cfgpath.read_text())
    d,_,_=inputs(cfg);prices=d['par_close'];returns,beta=coefficients(prices)
    cut=len(prices)-48;_,prefix=coefficients(prices[:cut]);np.testing.assert_array_equal(beta[:cut],prefix)
    # Future observations cannot alter a previously available factor estimate.
    changed=prices.copy();changed[cut:]*=np.arange(1,prices.shape[1]+1)[None,:]
    _,after=coefficients(changed);np.testing.assert_array_equal(beta[:cut+1],after[:cut+1])
    paths=[Path(__file__),cfgpath,ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz']
    for phase in ['validation','comparison']:
        for arm,folder in [(MATCHED,PARENT),(ARM,OUT)]:
            for sc in ['base','stress']:paths.append(folder/'runs'/f'reference-{phase}-{arm}-{sc}.json.gz')
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    save(dest/'frozen.json',dict(at_unix=time.time(),sources=sources,
        definition='For each recorded interval: pair_notional(previous)*past_168h_beta_to_BTC*realized_BTC_return is factor cash; actual pair price cash minus factor cash is residual. Coefficients are estimated strictly before the interval from simple returns. Attribution is not causal proof, a tradable hedge, or an independent PnL component.',no_selection=True,orders_sent=0))
    rows=[]
    for phase in ['validation','comparison']:
        for kind,arm,folder in [('direct',MATCHED,PARENT),('raw',ARM,OUT)]:
            for sc in ['base','stress']:
                with gzip.open(folder/'runs'/f'reference-{phase}-{arm}-{sc}.json.gz','rt') as f:run=json.load(f)
                factor=residual=total=0.;exposures=[];gross_ratios=[];intervals=[]
                for previous,current in zip(run['snapshots'],run['snapshots'][1:]):
                    now=current['time'];i=int(np.searchsorted(d['times'],now));j=int(np.searchsorted(d['times'],previous['time']))
                    assert i==j+1
                    q=np.asarray(previous['pair_qty']);active=np.flatnonzero(q)
                    if not len(active):continue
                    assert np.isfinite(beta[i,active]).all()
                    notional=q[active]*prices[j,active];exposure=float(notional@beta[i,active]);gross=float(abs(notional).sum())
                    cash=float(q@(prices[i]-prices[j]));common=exposure*returns[i,0]
                    factor+=common;residual+=cash-common;total+=cash
                    exposures.append(exposure);gross_ratios.append(exposure/gross)
                    intervals.append(dict(time=now,factor_cash=common,residual_cash=cash-common,actual_price_cash=cash,
                                          beta_dollar_exposure=exposure,gross=gross))
                # Independently sum actual price PnL from the saved attribution.
                diagnostic=json.loads((OUT/'diagnostics.json').read_text())
                expected=next(r['totals']['price_move'] for r in diagnostic['account_attribution'] if r['phase']==phase and r['model']==kind and r['scenario']==sc)
                assert abs(total-expected)<1e-6 and abs(factor+residual-total)<1e-6
                rows.append(dict(phase=phase,model=kind,scenario=sc,active_intervals=len(intervals),
                    actual_price_cash=total,factor_cash=factor,residual_cash=residual,
                    mean_absolute_beta_dollars=float(np.mean(abs(np.asarray(exposures)))) if exposures else 0.,
                    exposure_per_gross_quantiles=np.quantile(gross_ratios,[0,.1,.5,.9,1]).tolist() if gross_ratios else [],
                    intervals=intervals))
                print(phase,kind,sc,'price',round(total,4),'factor',round(factor,4),'residual',round(residual,4),
                      'median factor exposure/gross',rows[-1]['exposure_per_gross_quantiles'][2],flush=True)
    verify(sources)
    save(dest/'results.json',dict(rows=rows,meaning='Variance-minimizing two-asset hedge is not necessarily BTC-factor neutral. These realized decompositions depend on the stated past-only factor model and exclude fees/funding, which remain in full account reports. Do not infer that simply forcing beta-neutrality improves PnL.',
        future_perturbation_check_passed=True,prefix_check_passed=True,orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False))
    print('DONE factor audit',flush=True)


if __name__=='__main__':main()
