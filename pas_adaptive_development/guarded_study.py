"""Paid proactive reductions and shared-capacity reservation experiment."""
import copy,gzip,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .study import ROOT,OUT as PARENT,sha,save,native
from . import sleeve_budget

OUT=ROOT/'reports/pas_guarded_allocation_20261021'
ARMS=['incumbent','base_restore','guarded_current','guarded_base','guarded_both']


def prepare():
    from competition_live.profile import verify_deployment
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,nominee='guarded_both',
        hypothesis='Entry headroom alone is consumed by later price gains and base additions. Preserve shared capacity and reduce proportionally BEFORE hard-limit violations, instead of flattening and reopening the whole account.',
        risk_rules='Soft planning limits 90% of existing hard caps. When crossed, submit paid proportional owner reductions toward 81%; actual hard caps and drawdown halt unchanged. At least 10 USD delta except full dust closes. Partial pair trim that breaks hedge balance unwinds.',
        addition_rule='Reserve current pair inventory when allocating new base additions. Base reductions continue. Pair completion/renewal check soft limits. Futures, current prices, cost and partial fill assumptions remain identical.',
        accounting='Actual fills only. Pair stop uses original episode price-PnL anchor after reductions, preserving realized losses and original maturity; no reset. Ledger fees/funding counted once. No atomic fills.',
        ratio=sleeve_budget.RATIO,
        selection='guarded_both fixed nominee; guarded_current and guarded_base ablate safety vs sizing. All original windows/cost scenarios retained.',
        promotion='Beat original and base_restore PnL/Q in continuous and later windows under base/stress; no hard-cap breaches or accounting/timing violations. PAS applicability separate.',
        independent=False,history_reused=True,orders_sent=0,live_changed=False,
        deployment=verify_deployment()['files'])
    save(OUT/'protocol.json',p)
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in
        [Path(__file__),Path(__file__).with_name('guarded_portfolio.py'),Path(__file__).with_name('risk_trim.py'),
         Path(__file__).with_name('engine_diff.patch'),OUT/'protocol.json',PARENT/'targets.npz']})


def run():
    from competition_live.profile import load_profile,BUNDLE,verify_deployment
    from .guarded_portfolio import replay
    from pas_allocation_study.study import summarize
    p=json.loads((OUT/'protocol.json').read_text())
    for rel,h in json.loads((OUT/'source_freeze.json').read_text()).items():assert sha(ROOT/rel)==h
    profile=load_profile();d0=dict(np.load(BUNDLE/'data/reference.npz'));state=dict(np.load(BUNDLE/'models/reference_gate_state.npz'))
    z=dict(np.load(PARENT/'targets.npz'));ix=np.searchsorted(z['times'],d0['times']);rows=[]
    windows=[r for r in json.loads((BUNDLE/'expected.json').read_text())['runs'] if r['venue']=='reference' and r['arm']==profile['candidate']]
    for win in windows:
        for arm in ARMS:
            dest=OUT/'reference'/f"{arm}_{win['phase']}_{win['scenario']}.json.gz"
            if dest.exists():
                with gzip.open(dest,'rt') as f:r=json.load(f)
            elif arm in ['incumbent','base_restore']:
                with gzip.open(sleeve_budget.OUT/'reference'/f"{arm}_{win['phase']}_{win['scenario']}.json.gz",'rt') as f:r=json.load(f)
                dest.parent.mkdir(exist_ok=True)
                with gzip.open(dest,'wt') as f:json.dump(r,f,default=native)
            else:
                d=dict(d0);d['first']=z['incumbent' if arm=='guarded_current' else 'size_only'][ix]
                cfg=copy.deepcopy(profile['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
                if arm=='guarded_both':cfg['pair_max_gross']*=sleeve_budget.RATIO
                r=replay(d,state,None,cfg,profile['trading'],profile['candidate'],win['scenario'],win['start'],win['end'])
                dest.parent.mkdir(exist_ok=True)
                with gzip.open(dest,'wt') as f:json.dump(r,f,default=native)
            row=summarize(r,arm,win['phase'],win['scenario']);assert row['timing_pass'] and row['accounting_error']<1e-6
            row['risk_trim_requests']=sum(e['event']=='risk_trim_requested' for e in r['events'])
            row['risk_trim_unwinds']=sum(e['event']=='risk_trim_partial_unwind' for e in r['events'])
            rows.append(row);save(OUT/'reference_results.json',rows);print(json.dumps(row),flush=True)
    assert verify_deployment()['files']==p['deployment']


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
