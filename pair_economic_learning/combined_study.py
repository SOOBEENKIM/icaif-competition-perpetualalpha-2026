"""Combine separately tested core isolation and source turning confirmation."""
import sys,json,shutil
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
from pair_value_learning.study import ROOT,INPUT,read,save,sha
from .reliability_study import verify as verify_reliability,OUT as REL
from .execution_study import verify as verify_execution
from .reference import run as replay
OUT=ROOT/'reports/pair_economic_learning_20261023/wave5'
SOURCES={'confirmed24_pairs_72':'confirmed24_alpha_72','confirmed6_pairs_72':'confirmed6_alpha_72'}
ARMS=list(SOURCES)


def freeze():
    verify_reliability();verify_execution()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),input=str(INPUT.relative_to(ROOT)),input_sha256=sha(INPUT),arms=ARMS,primary='confirmed24_pairs_72',sources=SOURCES,
       hypothesis='The original directionalcore had negativePnL and worsened pairentryaccess. Purecoherentpairs improved all6referencePnL/Q overincumbent.24hturnconfirmation improvedfullstress withcore. Combine these independently motivated changes while retaining6hcontrol; no scaleincrease.',
       controls='Same45 sourcehedges/ARcoherence/6000paircap/50k+50kreserve/fees/funding/limits/paidrenewal. No directionalcore. Entrydirectionmustagreewithpast24h(or6h)physicalrelativeprice change.',
       selection_bias='Combination chosen after earlierdevelopmentresults;not independenttest. All6matchedreferenceconditionsreported, includingfailures.',
       PAS='Exactpairmodule forecasts and entry/lifecycle equal correspondingwave4 PAS arms because PAS alreadyomitsdirectionalcore. Reuse those results explicitly after byte-identicalstatecheck; do not double-countasnewtests.',
       orders_sent=0,live_changed=False,independent_test=False,history_reused=True))
    paths=[Path(__file__),Path(__file__).with_name('reference.py'),OUT/'protocol.json',*[REL/f'{a}.npz' for a in SOURCES.values()]]
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})


def verify():
    verify_reliability();verify_execution()
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p


def build():
    verify()
    for arm,parent in SOURCES.items():
        shutil.copy2(REL/f'{parent}.npz',OUT/f'{arm}.npz')
        assert sha(OUT/f'{arm}.npz')==sha(REL/f'{parent}.npz')


def job(arm,phase,scenario):
    return replay(OUT,OUT/f'{arm}.npz',arm,phase,scenario,include_base=False)


def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,p,s) for a in ARMS for p in ['validation','comparison','continuous'] for s in ['base','stress']]):
            r=f.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs','risk_breach_bars']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
