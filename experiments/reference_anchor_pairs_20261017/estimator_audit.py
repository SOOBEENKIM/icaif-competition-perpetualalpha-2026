"""Compare formation estimators; mathematical implementation, no vendor code.

For a self-similar path Var[X(t+k)-X(t)] is proportional to k**(2H).
Estimate H as half the least-squares slope of log variance on log lag.
This differs from classical R/S, particularly at the short 2..19 hour scale.
"""
import json
import numpy as np
from strategy.agent import TAIL
from experiments.par_direct_forecast_20261017.data import save, sha, ts
from experiments.pairs_research_20261017.models import coint, multipletests
from .research import HERE, OUT, ROOT


def variance_scaling(path, lags=tuple(range(2, 20))):
    path = np.asarray(path, dtype=float)
    if path.ndim != 1 or len(path) < 2 * max(lags):
        return None
    variances = np.asarray([np.var(np.subtract(path[k:], path[:-k])) for k in lags])
    if np.any(~np.isfinite(variances)) or np.any(variances <= 0):
        return None
    x = np.log(np.asarray(lags, dtype=float)); y = np.log(variances)
    x -= x.mean(); y -= y.mean()
    return float(np.dot(x, y) / (2 * np.dot(x, x)))


def main():
    dest = OUT / 'estimator_audit'; dest.mkdir(exist_ok=False)
    old = json.loads((OUT/'universe_audit_v2/results.json').read_text())
    cfg = json.loads((HERE/'protocol.json').read_text())
    source = ROOT/cfg['reference_run']/'inputs.npz'
    with np.load(source) as z:
        times = z['market_times'].copy(); prices = z['close'].copy()
    symbols = json.loads((TAIL/'baseline.json').read_text())['protocol']['symbols']
    protocol = dict(purpose='Formation-estimator sensitivity, no performance selection.',
        hurst='Variance scaling H = 0.5*slope(log variance of lagged differences, log lag), lags 2..19.',
        eg='statsmodels coint defaults: intercept, AIC-selected lag with automatic maximum.',
        relationship='Original mathematical implementation. Paper algorithm reviewed for methodology only; no external trading code imported or copied.',
        reference='https://raw.githubusercontent.com/damianlebiedz/pair-trading-with-rl/main/modules/core/indicators.py',
        limits='10 competition underlyings, four reused formation windows, not full paper reproduction. EG ordering is canonical, not both directions. Point Hurst estimates and BH q values do not certify profitability.',
        trading_results_accessed=False, orders_sent=0)
    paths=[HERE/'estimator_audit.py', OUT/'universe_audit_v2/results.json', source]
    save(dest/'frozen.json', dict(protocol=protocol,sources={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    formations=[]
    for old_f in old['formations']:
        date=old_f['date']; month=np.datetime64(date,'M')
        lo=int((month-2).astype('datetime64[s]').astype('int64'))
        x=np.log(prices[(times>=lo)&(times<ts(date))]); rows=[]
        for previous in old_f['rows']:
            a,b=[symbols.index(s) for s in previous['pair']]
            beta=previous['beta']; residual=x[:,a]-beta*x[:,b]
            H=variance_scaling(residual)
            p=float(coint(x[:,a],x[:,b])[1])
            row={**previous, 'hurst_variance_scaling':H,'eg_default_p':p,
                'variance_structural_pass':bool(H is not None and H<.5 and beta>0),
                'variance_score':.5*(1-p)+.5*previous['r_squared']}
            rows.append(row)
        qs=multipletests([r['eg_default_p'] for r in rows],alpha=.1,method='fdr_bh')[1]
        for r,q in zip(rows,qs):
            r['eg_default_q']=float(q)
            r['variance_fdr_pass']=bool(q<=.1 and r['variance_structural_pass'])
        rows.sort(key=lambda r:(-r['variance_structural_pass'],-r['variance_score'],r['pair']))
        record=dict(date=date,rows=rows,structural_count=sum(r['variance_structural_pass'] for r in rows),
            fdr_count=sum(r['variance_fdr_pass'] for r in rows),
            top_two=[r['pair'] for r in rows if r['variance_structural_pass']][:2],
            fdr_top_two=[r['pair'] for r in rows if r['variance_fdr_pass']][:2])
        formations.append(record)
        print(date,record['structural_count'],record['fdr_count'],record['top_two'],flush=True)
    save(dest/'results.json',dict(protocol=protocol,formations=formations,orders_sent=0))


if __name__=='__main__':main()
