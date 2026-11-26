"""Scale the whole retained portfolio before identical hard risk limits.

The earlier2x experiment scaled only the directional core, not the pair
sleeve. This experiment explicitly separates whole-portfolio budgeting from
that asymmetric change. No alpha or independent-performance claim.
"""
import copy,gzip,json
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from competition_live.profile import load_profile,BUNDLE
from pas_economic_allocation.replay import replay
from pas_allocation_study.study import summarize
from pas_performance_research.selected import controller
from .protocol import ROOT,OUT,save,sha

DEST=OUT/'whole_budget'


def job(arm,w):
    p=load_profile();d=dict(np.load(BUNDLE/'data/reference.npz'));state=dict(np.load(BUNDLE/'models/reference_gate_state.npz'))
    z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz'));ix=np.searchsorted(z['times'],d['times'])
    f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz'));fi=np.searchsorted(f['times'],d['times'])
    source=json.loads((ROOT/'reports/universe_repair_20261022/protocol.json').read_text())['input']['inputs'];raw=dict(np.load(ROOT/source))
    np.testing.assert_array_equal(z['times'][ix],d['times']);np.testing.assert_array_equal(f['times'][fi],d['times'])
    multiplier=1. if arm=='incumbent' else 1.5
    c=controller(d['times'],f['forecast'][fi],f['carry'][fi],z['all_market_signed_core_v1'][ix]*multiplier,
        z['all_market_signed_core_v1_protected'][ix],price_times=raw['market_times'],prices=raw['close'])
    cfg=copy.deepcopy(p['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
    if arm=='whole_1p5':cfg['pair_max_gross']*=multiplier
    r=replay(d,state,None,cfg,p['trading'],p['candidate'],w['scenario'],w['start'],w['end'],base_policy=c)
    row=summarize(r,arm,w['phase'],w['scenario']);row.update(venue='external_reference',independent_test=False,
        core_budget_multiplier=multiplier,pair_budget_multiplier=multiplier if arm=='whole_1p5' else 1.,
        hard_limits=p['trading']['risk'],capital=p['trading']['initial'])
    with gzip.open(DEST/f"{arm}_{w['phase']}_{w['scenario']}.json.gz",'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
    assert row['timing_pass'] and row['accounting_error']<1e-6
    return row


def main():
    assert not (DEST/'protocol.json').exists()
    p=load_profile();windows=[w for w in json.loads((BUNDLE/'expected.json').read_text())['runs'] if w['venue']=='reference' and w['arm']==p['candidate']]
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),primary='whole_1p5',
        controls=['incumbent','core_only_1p5'],reason='Prior2x scale changed the losing reference core while leaving the profitable retained pair sleeve unchanged. Test a predeclared modest1.5 whole-signal budget with the identical50kcapital,gross/net/asset/daily-risk caps,fees,volume and decision/exit rules. This is an allocation change, not predictive improvement.',
        selection='One primary fixed before outcomes, no multiplier grid. Diagnostic core_only cannot become primary retrospectively.',
        limitation='Reference history was repeatedly inspected. Pair profit comes from few episodes. No independent-test claim or proof that larger positions will fill on PAS; no replacement certification from this screen.',
        success='Higher netPnL,Q,and terminal-cost-adjustedPnL on continuous and comparison in both cost scenarios with zero risk/accounting/timing failure; August is reported, not erased.',
        periods=[{k:w[k] for k in ['phase','start','end','scenario']} for w in windows],
        file_hash=sha(__file__),orders_sent=0,live_changed=False))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(job,a,w) for w in windows for a in ['incumbent','whole_1p5','core_only_1p5']]):
            r=future.result();rows.append(r);save(DEST/'results.json',rows)
            print({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','base_pnl','pair_pnl','completed_pairs','risk_breach_bars']},flush=True)

if __name__=='__main__':main()
