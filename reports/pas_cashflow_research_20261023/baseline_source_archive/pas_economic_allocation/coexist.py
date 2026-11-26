"""Matched ETH-excluded comparison required by the actual shared-wallet setup."""
import copy,gzip,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .lifecycle import LifecycleController
from .anchored import RATIO,AnchoredController
from .study import OUT as ECONOMIC
from pas_adaptive_development.study import ROOT,save,sha,native

OUT=ROOT/'reports/pas_economic_coexist_20261022'


def prepare():
    from competition_live.profile import verify_deployment
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),nominee='candidate',
      selection='budget_lifecycle chosen RETROSPECTIVELY from preserved diagnostics because PnL/Q beat current in all four required full/later cases. Not the original slow-family primary and not an independently selected winner.',
      arms=['incumbent','size_only','repaired_budget_only','candidate'],
      experiment='Apply identical teammate ETH exclusion to every arm. Own50k+fixed50k reserve. Zero ownedETH from inception; reject whole BTC-ETH pair before any leg. Keep other pair and exact paid risk controls.',
      baseline='Permitted-market counterfactual of current alpha; not a reproduction of the old operational halt when an ETH proposal raises an exception.',
      candidate='Final risk gates retained, original pre-gate daily risk0.1%->0.2% subject to0.5% hard cap, all-market168h pooled forecast/carry, post-projection economic check, actual extra units held while same signed net thesis persists. No model refit.',
      success='Candidate PnL and Q exceed incumbent and size_only in both continuous and later windows base/stress, zero hard-risk/timing/accounting/ETH-order errors. Report against same-budget repaired-only control to identify allocation versus alpha.',
      independent=False,history_reused=True,changed_after_inspection=True,
      live_changed=False,orders_sent=0,deployment=verify_deployment()['files'])
    save(OUT/'protocol.json',p)
    files=[Path(__file__),Path(__file__).with_name('coexist_replay.py'),Path(__file__).with_name('lifecycle.py'),
           Path(__file__).with_name('projected.py'),Path(__file__).with_name('anchored.py'),Path(__file__).with_name('policy.py'),
           OUT/'protocol.json',ECONOMIC/'forecasts.npz',ROOT/'reports/strategy_repairs_20261021/final_risk_targets.npz']
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in files})


def run():
    from competition_live.profile import load_profile,BUNDLE,verify_deployment
    from .coexist_replay import replay
    from pas_allocation_study.study import summarize
    p=json.loads((OUT/'protocol.json').read_text())
    for rel,h in json.loads((OUT/'source_freeze.json').read_text()).items():assert sha(ROOT/rel)==h
    profile=load_profile();d0=dict(np.load(BUNDLE/'data/reference.npz'));states=dict(np.load(BUNDLE/'models/reference_gate_state.npz'))
    z=dict(np.load(ECONOMIC/'forecasts.npz'));ix=np.searchsorted(z['times'],d0['times'])
    r=dict(np.load(ROOT/'reports/strategy_repairs_20261021/final_risk_targets.npz'));ir=np.searchsorted(r['times'],d0['times'])
    cfg=copy.deepcopy(profile['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
    lam=json.loads((ECONOMIC/'selected.json').read_text())['selected']['risk_aversion']
    windows=[w for w in json.loads((BUNDLE/'expected.json').read_text())['runs'] if w['venue']=='reference' and w['arm']==profile['candidate']]
    rows=[]
    for w in windows:
        for arm in p['arms']:
            file=OUT/'reference'/f"{arm}_{w['phase']}_{w['scenario']}.json.gz"
            if file.exists():
                with gzip.open(file,'rt') as f:result=json.load(f)
            else:
                d={k:v.copy() for k,v in d0.items()};d['first'][:,1]=0.
                c=None
                if arm=='size_only':d['first']*=RATIO
                if arm in ['candidate','repaired_budget_only']:
                    anchor=r['final_mix_risk'][ir]*2*RATIO;anchor[:,1]=0.
                    protected=r['protected'][ir].copy();protected[:,1]=True
                    cls=LifecycleController if arm=='candidate' else AnchoredController
                    c=cls(d['times'],z['forecast'][ix],z['carry'][ix],anchor,protected,lam,augment=arm=='candidate')
                result=replay(d,states,None,cfg,profile['trading'],profile['candidate'],w['scenario'],w['start'],w['end'],base_policy=c)
                if c is not None:result['allocation_trace']=c.trace
                assert not any(f['asset']==1 for f in result['fills'])
                file.parent.mkdir(exist_ok=True)
                with gzip.open(file,'wt') as f:json.dump(result,f,default=native,allow_nan=False)
            row=summarize(result,arm,w['phase'],w['scenario']);rows.append(row)
            assert row['timing_pass'] and row['accounting_error']<1e-6
            save(OUT/'reference_results.json',rows);print(json.dumps(row),flush=True)
    assert verify_deployment()['files']==p['deployment']


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
