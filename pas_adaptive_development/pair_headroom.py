"""Research-only risk headroom adapter; deployment source is never modified."""
import copy,gzip,json
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .study import ROOT,OUT as PARENT,sha,save,native
from . import sleeve_budget

OUT=ROOT/'reports/pas_pair_headroom_20261021'
ARMS=['incumbent','base_restore','unbuffered_both','buffered_current','buffered_both']


def soft_protocol(p):
    result=copy.deepcopy(p);fraction=p['risk']['sizing_headroom']
    for key in ['gross_fraction','net_fraction','asset_fraction','asset_notional','daily_volatility']:
        result['risk'][key]*=fraction
    return result


@contextmanager
def buffered_policy(enabled=True):
    from experiments.decision_gate_pairs_20261017 import portfolio
    if not enabled:
        yield
        return
    original_entry=portfolio.proposals;original_completion=portfolio.completion_plan;original_renewal=portfolio.renewal_plan
    def entry(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,**kwargs):
        return original_entry(i,d,de,pred,h,cfg,soft_protocol(p),sc,qty,equity,reserved,directions,arm,**kwargs)
    def completion(pair,ledger,i,d,de,cfg,p,sc,**kwargs):
        return original_completion(pair,ledger,i,d,de,cfg,soft_protocol(p),sc,**kwargs)
    def renewal(pair,owned_qty,i,d,de,cfg,p,sc,account_qty,equity,**kwargs):
        return original_renewal(pair,owned_qty,i,d,de,cfg,soft_protocol(p),sc,account_qty,equity,**kwargs)
    with patch.object(portfolio,'proposals',entry),patch.object(portfolio,'completion_plan',completion),patch.object(portfolio,'renewal_plan',renewal):
        yield


def prepare():
    from competition_live.profile import verify_deployment
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,nominee='buffered_both',
        hypothesis='Pair sizing uses hard risk caps while base sizing already leaves 10% headroom. Enlarged pair targets touch hard caps; price drift triggers whole-account forced exit/re-entry. Apply EXISTING headroom consistently to entry, completion and paid rehedging.',
        changes='Soft proposal limits = hard limits * existing sizing_headroom 0.9. Actual execution and monitoring retain original hard caps. No smaller fees, retroactive cancellation, atomic fills, or look-ahead.',
        ratio=sleeve_budget.RATIO,pair_gross_before=6000,pair_gross_after=6000*sleeve_budget.RATIO,
        evidence='Unbuffered both_restore later base run had 11 hard-limit breach bars, 11 completed pairs and 7 aborted pairs versus incumbent 0 breaches and 3 completed pairs. Failures retained.',
        selection='Nominee buffered_both fixed before outcomes. All original windows/scenarios and unbuffered ablation retained.',
        promotion='Higher PnL and Q than incumbent AND base_restore in continuous and later windows under base and stress, no hard-risk breaches or implementation violations. PAS conditions separately evaluated.',
        interpretation='Execution/risk consistency plus inherited size restoration; forecast model unchanged. Not evidence of new prediction accuracy.',
        independent=False,history_reused=True,orders_sent=0,live_changed=False,
        deployment=verify_deployment()['files'])
    save(OUT/'protocol.json',p)
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in
        [Path(__file__),OUT/'protocol.json',PARENT/'targets.npz']})


def run():
    from competition_live.profile import load_profile,BUNDLE,verify_deployment
    from experiments.decision_gate_pairs_20261017.portfolio import replay
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
            elif arm in ['incumbent','base_restore','unbuffered_both']:
                source='both_restore' if arm=='unbuffered_both' else arm
                with gzip.open(sleeve_budget.OUT/'reference'/f"{source}_{win['phase']}_{win['scenario']}.json.gz",'rt') as f:r=json.load(f)
                dest.parent.mkdir(exist_ok=True)
                with gzip.open(dest,'wt') as f:json.dump(r,f,default=native)
            else:
                d=dict(d0);d['first']=z['size_only' if arm=='buffered_both' else 'incumbent'][ix]
                cfg=copy.deepcopy(profile['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
                if arm=='buffered_both':cfg['pair_max_gross']*=sleeve_budget.RATIO
                with buffered_policy():
                    r=replay(d,state,None,cfg,profile['trading'],profile['candidate'],win['scenario'],win['start'],win['end'])
                dest.parent.mkdir(exist_ok=True)
                with gzip.open(dest,'wt') as f:json.dump(r,f,default=native)
            row=summarize(r,arm,win['phase'],win['scenario']);assert row['timing_pass'] and row['accounting_error']<1e-6
            rows.append(row);save(OUT/'reference_results.json',rows);print(json.dumps(row),flush=True)
    assert verify_deployment()['files']==p['deployment']


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
