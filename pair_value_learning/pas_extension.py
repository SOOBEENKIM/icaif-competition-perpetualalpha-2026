"""84-hour PAS diagnostic; fixed alternatives, not a fresh holdout."""
import copy,gzip,json,sys
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .study import ROOT,read,save,sha
from . import pas_study_v2 as parent
from .scaled_study import verify as verify_scaled
from .pas_intraday_replay import replay
from competition_live.profile import load_profile
from auto_pair_selection.selection import PAIRS

OUT=ROOT/'reports/pair_value_learning_20261023/pas84'
START=1792411200
END=1792713600  # 2026-10-23 00:00 UTC; includes 3 completed days and 12 hours.
SOURCES={
 'adaptive_reference_72':'reports/pair_model_research_20261023/wave9/adaptive_hedge_72.npz',
 'adaptive_pas_72':'reports/pair_model_research_20261023/wave9/adaptive_hedge_72.npz',
 'adaptive_scaled_reference_72':'reports/pair_value_learning_20261023/scaled_coherence/coherent_scaled_72.npz',
 'adaptive_scaled_pas_72':'reports/pair_value_learning_20261023/scaled_coherence/coherent_scaled_72.npz',
}


def freeze():
    parent.verify();verify_scaled()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),start=START,end=END,
      primary='adaptive_scaled_pas_72',sources=SOURCES,
      hypothesis='Keep source72h relative forecasts and all45/source hedges. Compare original reference-price convergence hypothesis against current PAS-price anchoring, with original renewable paid lifecycle in every arm.',
      purpose='Extend previously inspected72h to84h to observe holding lifecycle. The endpoint is not selected by PnL. This remains development data, not an independent test.',
      constraints='Pair module only, same50k sleeve+50k reserve,6000 paircap. Conditional next-hour fills and actual funding events, same three cost conditions; no historical L2 or actual oracle series.',
      score='Return includes latest partial day. Volatility uses only 3 completed noon daily returns. Q is conditional simulated account metric, not official leaderboard.',
      independent_test=False,orders_sent=0,live_changed=False))
    paths=[Path(__file__),Path(__file__).with_name('pas_intraday_replay.py'),Path(parent.__file__),OUT/'protocol.json',*[ROOT/p for p in SOURCES.values()],*[parent.RAW/f'pas_{s}.json' for s in parent.SYMBOLS]]
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})


def verify():
    parent.verify();verify_scaled()
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p


def prepare():
    verify()
    old=(parent.OUT,parent.END,parent.SOURCES)
    try:
        parent.OUT=OUT;parent.END=END;parent.SOURCES=SOURCES
        parent.prepare()
    finally:
        parent.OUT,parent.END,parent.SOURCES=old
    d=dict(np.load(OUT/'dataset.npz'))
    for arm in SOURCES:
        if '_pas_' in arm:
            s=dict(np.load(OUT/f'{arm}.npz'));s['source_prices']=d['par_close'].copy()
            np.savez_compressed(OUT/f'{arm}.npz',**s)
    verify()


def job(arm,scenario):
    from pas_allocation_study.study import summarize
    d=dict(np.load(OUT/'dataset.npz'));s=dict(np.load(OUT/f'{arm}.npz'));s.pop('times');p=load_profile()
    cfg=copy.deepcopy(p['configs']['reference']);cfg['dense_pairs']=[list(x) for x in PAIRS];cfg['conditional_completion']={'risk_aversion':0.}
    cfg['scenarios']['no_receipts']={**cfg['scenarios']['base'],'funding_receipt_multiplier':0.}
    r=replay(d,s,None,cfg,p['trading'],'pairs_dense_hourlyhedge_individual_72',scenario,START,END)
    row=summarize(r,arm,'PAS84h_pair_module',scenario)
    row['completed_day_return_fraction']=row['return_fraction']
    R=(r['snapshots'][-1]['equity']+50000)/(r['snapshots'][0]['equity']+50000)-1
    Q=(max(R,0)+.005)/(abs(R)+1.5*row['volatility']+.05)
    row.update(return_fraction=R,intraday_return_fraction=R,Q=Q,score=.8*Q if row['eligible'] else None,
               official_Q=False,conditional_execution=True,base_included=False)
    assert not r['owner_fills']['base'] and row['accounting_error']<1e-6 and row['risk_breach_bars']==0
    assert all(f['decision_time']<f['time'] for f in r['fills'])
    path=OUT/'runs'/f'{arm}_{scenario}.json.gz';path.parent.mkdir(exist_ok=True)
    with gzip.open(path,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
    save(OUT/'rows'/f'{arm}_{scenario}.json',row)
    return row


def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,s) for a in SOURCES for s in ['base','stress','no_receipts']]):
            r=f.result();print(json.dumps({k:r[k] for k in ['arm','scenario','net_pnl','Q','completed_pairs','funding','fills']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
