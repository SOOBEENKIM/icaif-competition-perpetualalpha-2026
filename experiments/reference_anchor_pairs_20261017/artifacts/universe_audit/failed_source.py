"""Formation-only audit; original implementation, no external trading code."""
import itertools
import json
from pathlib import Path
import numpy as np
from strategy.agent import TAIL
from experiments.pairs_research_20261017.models import coint,multipletests
from experiments.par_direct_forecast_20261017.data import ts,save,sha
from .research import HERE,ROOT,OUT


def hurst_rs_walk(path):
    """R/S of increments of the residual path; explicitly not R/S of levels.

    Classical rescaled-range estimator, implemented locally. Not asserted to
    be numerically identical to the paper repository's implementation.
    """
    increments=np.diff(np.asarray(path,float));sizes=[];values=[]
    for width in (16,32,64,128,256):
        if len(increments)<2*width:continue
        ratios=[]
        for i in range(0,len(increments)-width+1,width):
            block=increments[i:i+width];sd=np.std(block,ddof=1)
            if sd<=1e-12:continue
            walk=np.r_[0,np.cumsum(block-block.mean())];span=walk.max()-walk.min()
            if span>0:ratios.append(span/sd)
        if ratios:sizes.append(width);values.append(float(np.mean(ratios)))
    if len(sizes)<3:return None
    return float(np.polyfit(np.log(sizes),np.log(values),1)[0])


def screen(times,prices,symbols,cut,lo):
    mask=(times>=lo)&(times<cut);x=np.log(prices[mask]);rows=[]
    if len(x)<1000:raise ValueError('Incomplete two-month formation history')
    for a,b in itertools.combinations(range(len(symbols)),2):
        y,z=x[:,a],x[:,b];alpha,beta=np.linalg.lstsq(np.c_[np.ones(len(z)),z],y,rcond=None)[0]
        residual=y-alpha-beta*z
        p=float(coint(y,z,trend='c',maxlag=24,autolag='aic')[1])
        r2=float(np.corrcoef(y,z)[0,1]**2);H=hurst_rs_walk(residual)
        structural=H is not None and H<.5 and beta>0
        rows.append(dict(pair=[symbols[a],symbols[b]],alpha=float(alpha),beta=float(beta),p_eg=p,
            r_squared=r2,hurst_rs=H,structural_pass=structural,raw_score=.5*(1-p)+.5*r2,
            formation_start=int(times[mask][0]),formation_last=int(times[mask][-1]),samples=len(x)))
    adjusted=multipletests([r['p_eg'] for r in rows],alpha=.1,method='fdr_bh')[1]
    for r,q in zip(rows,adjusted):
        r['q_bh']=float(q);r['fdr_structural_pass']=bool(r['structural_pass'] and q<=.1)
    return sorted(rows,key=lambda r:(-r['structural_pass'],-r['raw_score'],r['pair']))


def main():
    dest=OUT/'universe_audit';dest.mkdir(exist_ok=False)
    cfg=json.loads((HERE/'protocol.json').read_text());run=ROOT/cfg['reference_run']
    with np.load(run/'inputs.npz') as z:times=z['market_times'].copy();prices=z['close'].copy()
    symbols=json.loads((TAIL/'baseline.json').read_text())['protocol']['symbols']
    protocol=dict(formation_dates=['2026-07-01','2026-08-01','2026-09-01','2026-10-01'],
        lookback='Previous two calendar months; timestamps are completed bars and strictly before formation cutoff.',
        pairs='All 45 canonical-order combinations of the 10 competition reference instruments.',
        tests='Engle-Granger/MacKinnon p-values, maxlag24 AIC. Log-level OLS hedge; local increment R/S Hurst.',
        outputs='Paper-inspired structural H<0.5 and beta>0 ranking, plus separately reported BH q<=0.1 screen. BH under dependence and Hurst point estimates do not certify tradability.',
        selected_top_n=2,trade_performance_tested=False,independent=False,orders_sent=0,
        paper='https://arxiv.org/html/2606.04574v1',
        code_relationship='Original local estimator and existing project statistical dependencies. No external trading source copied; not a full reproduction of paper or PPO.')
    paths=[HERE/'universe_audit.py',run/'inputs.npz',ROOT/'experiments/pairs_research_20261017/models.py']
    save(dest/'frozen.json',dict(protocol=protocol,sources={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    results=[]
    for date in protocol['formation_dates']:
        month=np.datetime64(date,'M');lo=int((month-2).astype('datetime64[s]').astype('int64'));cut=ts(date)
        rows=screen(times,prices,symbols,cut,lo)
        paper=[r for r in rows if r['structural_pass']][:2]
        strict=[r for r in rows if r['fdr_structural_pass']][:2]
        record=dict(date=date,rows=rows,paper_structural_top=paper,fdr_structural_top=strict,
            structural_count=sum(r['structural_pass'] for r in rows),fdr_count=sum(r['fdr_structural_pass'] for r in rows))
        results.append(record)
        print(date,'structural',record['structural_count'],'FDR',record['fdr_count'],
              'top',[r['pair'] for r in paper],'FDR top',[r['pair'] for r in strict],flush=True)
    save(dest/'results.json',dict(protocol=protocol,formations=results,trade_performance_tested=False,orders_sent=0))


if __name__=='__main__':main()
