import json,sys
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from pair_value_learning.study import ROOT,INPUT,read,save,sha
from .study import OUT as PARENT,verify as verify_parent
from .reference import run as replay
from .fast_regime import build as build_models
OUT=ROOT/'reports/pair_economic_learning_20261023/wave7'
ARMS=['regime24_72','regime6_72']


def freeze():
    verify_parent()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),input=str(INPUT.relative_to(ROOT)),input_sha256=sha(INPUT),arms=ARMS,primary='regime24_72',
      hypothesis='72h realizedlabels update regime evidence too slowly. Use separately generated24h(primary)/6h(control) prequentialexpert errors to choose the current72h AR versus trend model. Labels must fully maturebeforeselection; selection refreshed6-hourly.',
      selection='Past28d nonoverlaporigins atlabelhorizon,7ddecay,source-risk normalized squared errors;localmeanlosssharedprior4 andminimum8origins. Choose lowererror expert, then source-risk coherenceproject. This short-horizon selector isnot proof of72h forecasting skill.',
      controls='Same45 sourcehedges/forecast clipping;50k+50kreserve,6000cap,originalpaidrenewal,costs. Paironlycore dueindependentcoreaudit. No PASlabels, no sourcefuture coefficients.',
      evaluation='All6reference/all6PASconditions;reuseddevelopmentandselectionbiasdisclosed;noautomaticlivechanges.',orders_sent=0,live_changed=False,independent_test=False,history_reused=True))
    paths=[Path(__file__),Path(__file__).with_name('fast_regime.py'),Path(__file__).with_name('test_fast_regime.py'),Path(__file__).with_name('reference.py'),OUT/'protocol.json',PARENT/'features.npz',PARENT/'formation.json']
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})


def verify():
    verify_parent()
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p


def build():
    verify();states,audit=build_models(dict(np.load(INPUT)),dict(np.load(PARENT/'features.npz')),read(PARENT/'formation.json'),1785542400)
    for arm,s in states.items():np.savez_compressed(OUT/f'{arm}.npz',**s)
    save(OUT/'fit_audit.json',audit);verify();print('BUILT fast regime selection',flush=True)


def job(arm,phase,scenario):return replay(OUT,OUT/f'{arm}.npz',arm,phase,scenario,include_base=False)


def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,p,s) for a in ARMS for p in ['validation','comparison','continuous'] for s in ['base','stress']]):
            r=f.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs','risk_breach_bars']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
