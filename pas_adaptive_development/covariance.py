"""Second declared family: covariance-aware construction of existing signals."""
import copy,gzip,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .study import ROOT,OUT as PARENT,sha,save,native,ts

OUT=ROOT/'reports/pas_covariance_development_20261021'
ARMS=['incumbent','size_only','decorrelated','hedged','decorrelated_restored','hedged_restored']


def targets(base,sigma,times,kind):
    out=np.zeros_like(base);held=np.zeros(base.shape[1]);previous=np.zeros(base.shape[1])
    for i,t in enumerate(times):
        if i<168 or t%21600:out[i]=held;continue
        b=base[i];s=sigma[i]
        # Recompute at each daily close or original strategy lifecycle change.
        if t%86400==0 or np.any(np.sign(b)!=np.sign(previous)):
            if kind=='decorrelated':
                # Diagonal-only risk reproduces b exactly; covariance adds hedges.
                held=np.linalg.solve(s,np.diag(s)*b)
            elif kind=='hedged':
                owned=np.flatnonzero(abs(b)>1e-12);empty=np.flatnonzero(abs(b)<=1e-12)
                held=b.copy()
                if len(owned) and len(empty):
                    h=-np.linalg.solve(s[np.ix_(empty,empty)],s[np.ix_(empty,owned)]@b[owned])
                    # Partial hedge prevents treating uncertain correlations as exact.
                    h*=.5
                    bound=.5*abs(b).sum()
                    if abs(h).sum()>bound:h*=bound/abs(h).sum()
                    held[empty]=h
            else:raise ValueError(kind)
            # Preserve base risk budget, never lever up after a risk reduction.
            before=float(b@s@b);after=float(held@s@held)
            if after>before and after>0:held*=np.sqrt(before/after)
            previous=b.copy()
        # Existing signal exits are not deferred until the next midnight.
        if not np.any(b):held=np.zeros_like(b)
        out[i]=held
    return out


def prepare():
    from competition_live.profile import load_profile,verify_deployment
    from replay_lab.common_comparison.engine import project
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,nominee='hedged_restored',
        purpose='Retain original predictive signal; use covariance across all ten markets for hedging/portfolio construction.',
        formulas={'decorrelated':'solve(existing shrunk hourly covariance, diag(covariance)*incumbent target)',
                  'hedged':'Keep occupied original targets. Add half the minimum-variance hedge on currently unoccupied assets, hedge gross <= half original base gross.',
                  'restored':'Multiply the complete base+hedge target by the already existing 1.44117372816997 original-size ratio; original account caps still apply.'},
        cadence='UTC daily or any incumbent sign/flat transition, checked on existing six-hour decision cadence.',
        covariance='Same causal 168-hour covariance and 75% diagonal shrinkage used in current risk checks; no test-period fit of shrinkage.',
        risk='Raw hedge construction may not exceed original predicted variance. Restored variants explicitly raise size and must beat size-only control.',
        selection='Nominee fixed before results. Full six original windows, all arms retained. No date change or PAS calibration.',
        promotion='Higher PnL AND Q than both incumbent and size-only in continuous and later windows under base and stress, zero implementation violations. Then evaluate PAS applicability separately.',
        known_prior_failures='Additive trees mostly abstained; adaptive sizing improved full period but not late stress.',
        historical_independence=False,orders_sent=0,live_changed=False,
        deployment=verify_deployment()['files'],parent_targets_sha=sha(PARENT/'targets.npz'))
    save(OUT/'protocol.json',p)
    z=dict(np.load(PARENT/'targets.npz',allow_pickle=False));trading=load_profile()['trading']
    w={k:z[k] for k in ['incumbent','size_only']}
    for kind in ['decorrelated','hedged']:
        w[kind]=targets(z['incumbent'],z['sigma'],z['times'],kind)
        w[kind+'_restored']=w[kind]*1.44117372816997
    for k,a in w.items():
        if k=='incumbent':continue
        for i in range(168,len(a)):a[i]=project(a[i],trading['initial'],z['sigma'][i],trading['risk'])
    np.savez_compressed(OUT/'targets.npz',times=z['times'],sigma=z['sigma'],**w)
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),OUT/'protocol.json',OUT/'targets.npz']})
    print('Prepared covariance family',flush=True)


def run():
    from competition_live.profile import load_profile,BUNDLE,verify_deployment
    from experiments.decision_gate_pairs_20261017.portfolio import replay
    from pas_allocation_study.study import summarize
    p=json.loads((OUT/'protocol.json').read_text());profile=load_profile()
    for rel,h in json.loads((OUT/'source_freeze.json').read_text()).items():assert sha(ROOT/rel)==h
    d0=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False));state=dict(np.load(BUNDLE/'models/reference_gate_state.npz',allow_pickle=False))
    z=dict(np.load(OUT/'targets.npz',allow_pickle=False));ix=np.searchsorted(z['times'],d0['times'])
    windows=[r for r in json.loads((BUNDLE/'expected.json').read_text())['runs'] if r['venue']=='reference' and r['arm']==profile['candidate']]
    rows=[]
    for win in windows:
        for arm in ARMS:
            path=OUT/'reference'/f"{arm}_{win['phase']}_{win['scenario']}.json.gz"
            if path.exists():
                with gzip.open(path,'rt') as f:r=json.load(f)
            else:
                d=dict(d0);d['first']=z[arm][ix];cfg=copy.deepcopy(profile['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
                r=replay(d,state,None,cfg,profile['trading'],profile['candidate'],win['scenario'],win['start'],win['end'])
                path.parent.mkdir(exist_ok=True)
                with gzip.open(path,'wt') as f:json.dump(r,f,default=native)
            row=summarize(r,arm,win['phase'],win['scenario']);rows.append(row);save(OUT/'reference_results.json',rows)
            print(json.dumps(row),flush=True)
    assert verify_deployment()['files']==p['deployment']


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
