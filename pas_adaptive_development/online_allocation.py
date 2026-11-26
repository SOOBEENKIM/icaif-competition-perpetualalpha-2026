"""Causal allocation among fixed sleeves from completed, costed daily returns."""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .study import ROOT,OUT as PARENT,sha,save
from . import covariance_v2 as runner
from . import trend_ensemble

OUT=ROOT/'reports/pas_online_allocation_20261021'
ARMS=['incumbent','size_only','positive_kelly','ensemble','equal_experts','exposure_control']


def allocation(d,experts):
    t=d['market_times'];n=len(t);k=len(experts)
    weights=np.stack(experts,axis=1)
    price_return=np.zeros_like(d['close']);price_return[1:]=d['close'][1:]/d['close'][:-1]-1
    lag=np.concatenate([np.zeros((1,k,10)),weights[:-1]])
    # Predictive feature only: this is NOT presented as execution-realized PnL.
    returns=(lag*price_return[:,None,:]).sum(2)-(lag*d['funding'][:,None,:]).sum(2)-.0019*abs(weights-lag).sum(2)
    out=np.zeros((n,10));scales=np.zeros((n,k));history=[];previous=0;fractions=np.zeros(k)
    fractions[0]=1.
    for i,now in enumerate(t):
        if now%86400==0:
            if i-previous==24:history.append(returns[previous+1:i+1].sum(0))
            previous=i
            if len(history)>=30:
                h=np.array(history[-90:]);ew=np.exp2(-np.arange(len(h)-1,-1,-1)/30);ew/=ew.sum()
                mu=ew@h;sd=np.sqrt(ew@((h-mu)**2));evidence=np.sqrt(30)*mu/np.maximum(sd,1e-8)
                # Fractional, bounded positive Kelly-like confidence. It is an
                # uncertainty-scaled rule, not the exact log-optimal portfolio.
                confidence=np.tanh(np.maximum(evidence,0))
                if k==1:fractions=1.44117372816997*confidence
                else:
                    fractions=confidence**2
                    if fractions.sum()>0:
                        fractions=1.44117372816997*confidence.max()*fractions/fractions.sum()
            
        scales[i]=fractions
        out[i]=fractions@weights[i]
    return out,scales


def prepare():
    from competition_live.profile import load_profile,verify_deployment
    from replay_lab.common_comparison.engine import project
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    parent=json.loads((PARENT/'protocol.json').read_text())
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,nominee='ensemble',
        hypothesis='A permanently funded new sleeve dilutes working signals. Allocate among fixed original, covariance-hedged and symmetric 10-market trend sleeves using only their completed past costed returns.',
        experts=['original base','corrected partial covariance hedge','symmetric 7/28-day trend'],
        history='Last 90 completed UTC days, exponential half-life 30 days, minimum 30 days; known-at-close target-return proxy with 19bp one-way costs and observed funding.',
        formula='confidence=tanh(max(sqrt(30)*EW mean/EW standard deviation,0)); expert shares proportional confidence^2; aggregate factor <= existing 1.44117372816997 times max confidence.',
        initial='Original only at factor1 until history matures.',
        constraints='No exponent/half-life/threshold optimization. Daily updates, same order caps and execution engine. No future reward access.',
        exposure_control='One July1-29 mean gross multiplier, before evaluation; realized gross/risk still disclosed.',
        selection='One ensemble nominee fixed before its results. Full six windows, both cost conditions, all arms preserved.',
        promotion='Beat both incumbent and size-only PnL and Q in continuous and later windows under base and stress, then separate PAS evaluation.',
        prior_failures='Always-on symmetric trend ensemble did not improve; corrected covariance hedges failed late stress. This design uses all three experts, with no asset/date exclusions.',
        independent=False,history_reused=True,orders_sent=0,live_changed=False,pointer=parent['pointer'],
        deployment=verify_deployment()['files'],expert_hashes={str(f.relative_to(ROOT)):sha(f) for f in
            [PARENT/'targets.npz',runner.OUT/'targets.npz',trend_ensemble.OUT/'targets.npz']})
    save(OUT/'protocol.json',p)
    z=dict(np.load(PARENT/'targets.npz',allow_pickle=False));d=dict(np.load(ROOT/p['pointer']['inputs'],allow_pickle=False))
    hedge=np.load(runner.OUT/'targets.npz')['hedged'];trend=np.load(trend_ensemble.OUT/'targets.npz')['trend']
    w={k:z[k] for k in ['incumbent','size_only']}
    w['positive_kelly'],single=allocation(d,[z['incumbent']])
    w['ensemble'],coefficients=allocation(d,[z['incumbent'],hedge,trend])
    w['equal_experts']=1.44117372816997*(z['incumbent']+hedge+trend)/3
    cal=(z['times']>=1782864000)&(z['times']<1785283200)
    scalar=abs(w['ensemble'][cal]).sum(1).mean()/abs(w['incumbent'][cal]).sum(1).mean()
    w['exposure_control']=z['incumbent']*scalar
    trading=load_profile()['trading']
    for name in ARMS:
        if name in ['incumbent','size_only']:continue
        for i in range(168,len(z['times'])):w[name][i]=project(w[name][i],trading['initial'],z['sigma'][i],trading['risk'])
    np.savez_compressed(OUT/'targets.npz',times=z['times'],sigma=z['sigma'],**w,
        coefficients=coefficients,single_coefficients=single)
    save(OUT/'calibration.json',dict(exposure_scalar=float(scalar)))
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in
        [Path(__file__),Path(runner.__file__),OUT/'protocol.json',OUT/'targets.npz',OUT/'calibration.json']})
    print('Prepared causal expert allocation family',flush=True)


def run():
    runner.OUT=OUT;runner.ARMS=ARMS;runner.run()


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
