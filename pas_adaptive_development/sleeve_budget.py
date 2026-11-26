"""Separate base sizing from pair sizing; no claim of forecast improvement."""
import copy,gzip,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .study import ROOT,OUT as PARENT,sha,save,native
from . import covariance_v2,online_allocation

OUT=ROOT/'reports/pas_sleeve_budget_20261021'
RATIO=1.44117372816997
ARMS=['incumbent','base_restore','pair_restore','both_restore','hedged_both_restore','adaptive_pair_restore']


def prepare():
    from competition_live.profile import verify_deployment
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,nominee='both_restore',
        hypothesis='Earlier size-only control restored base targets but left pair capacity fixed. Restore the SAME pre-existing size ratio in each sleeve separately to distinguish capital allocation from predictive changes.',
        ratio=RATIO,pair_gross_before=6000,pair_gross_after=6000*RATIO,
        budget='Same 50k strategy capital, 50k untouched reserve, original gross/net/per-asset/daily-risk caps, no additional leverage allowance.',
        arms_description={'base_restore':'Previous size_only exactly',
            'pair_restore':'Original base; restore pair capacity only',
            'both_restore':'Restore both base target sizes and pair capacity',
            'hedged_both_restore':'Corrected hedged restored base and restored pair capacity',
            'adaptive_pair_restore':'Earlier smooth adaptive base and restored pair capacity'},
        selection='both_restore fixed nominee; all ablations and exact six windows preserved; ratio inherited, no search for profit-maximizing leverage.',
        promotion='Higher PnL and Q than original and base-only restoration in continuous and later periods under base and stress. Original risk caps and accounting invariants. PAS fillability and signal activity separate.',
        interpretability='Any success of both_restore is sizing/capital-allocation improvement, NOT evidence of stronger forecasts or a new multi-asset model.',
        independence=False,history_reused=True,orders_sent=0,live_changed=False,
        deployment=verify_deployment()['files'])
    save(OUT/'protocol.json',p)
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in
        [Path(__file__),OUT/'protocol.json',PARENT/'targets.npz',covariance_v2.OUT/'targets.npz']})


def run():
    from competition_live.profile import load_profile,BUNDLE,verify_deployment
    from experiments.decision_gate_pairs_20261017.portfolio import replay
    from pas_allocation_study.study import summarize
    p=json.loads((OUT/'protocol.json').read_text())
    for rel,h in json.loads((OUT/'source_freeze.json').read_text()).items():assert sha(ROOT/rel)==h
    profile=load_profile();d0=dict(np.load(BUNDLE/'data/reference.npz'));state=dict(np.load(BUNDLE/'models/reference_gate_state.npz'))
    z=dict(np.load(PARENT/'targets.npz'));h=dict(np.load(covariance_v2.OUT/'targets.npz'));ix=np.searchsorted(z['times'],d0['times'])
    weights={'incumbent':z['incumbent'],'base_restore':z['size_only'],'pair_restore':z['incumbent'],
        'both_restore':z['size_only'],'hedged_both_restore':h['hedged_restored'],
        'adaptive_pair_restore':z['adaptive_size']}
    rows=[]
    windows=[r for r in json.loads((BUNDLE/'expected.json').read_text())['runs'] if r['venue']=='reference' and r['arm']==profile['candidate']]
    for win in windows:
        for arm in ARMS:
            dest=OUT/'reference'/f"{arm}_{win['phase']}_{win['scenario']}.json.gz"
            if dest.exists():
                with gzip.open(dest,'rt') as f:r=json.load(f)
            else:
                d=dict(d0);d['first']=weights[arm][ix];cfg=copy.deepcopy(profile['configs']['reference'])
                cfg['conditional_completion']={'risk_aversion':0.}
                if arm not in ['incumbent','base_restore']:cfg['pair_max_gross']*=RATIO
                r=replay(d,state,None,cfg,profile['trading'],profile['candidate'],win['scenario'],win['start'],win['end'])
                dest.parent.mkdir(exist_ok=True)
                with gzip.open(dest,'wt') as f:json.dump(r,f,default=native)
            row=summarize(r,arm,win['phase'],win['scenario'])
            assert row['timing_pass'] and row['accounting_error']<1e-6
            rows.append(row);save(OUT/'reference_results.json',rows);print(json.dumps(row),flush=True)
    assert verify_deployment()['files']==p['deployment']


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
