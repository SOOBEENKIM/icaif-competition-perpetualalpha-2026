import sys,json,shutil
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
from pair_value_learning.study import ROOT,INPUT,read,save,sha
from pair_value_learning.scaled_study import verify as verify_parent
from .reference import run as replay
OUT=ROOT/'reports/pair_economic_learning_20261023/wave3'
SOURCE=ROOT/'reports/pair_value_learning_20261023/scaled_coherence/coherent_scaled_72.npz'
ARMS=['bounded_entry_72','bounded_pairs_72','unbounded_pairs_72']


def freeze():
    verify_parent()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),input=str(INPUT.relative_to(ROOT)),input_sha256=sha(INPUT),arms=ARMS,primary='bounded_entry_72',
      hypothesis='Existing10000bpslippagecap allows spending most estimatedalpha in entrylimits. A10bpcap tests protectionagainstforecastoverstatement. Separate no-base ablations test whether the losing directionalcore interferes with pairinventory; do notattribute this to forecasting.',
      controls='Samecoherent_scaled72predictions,all45,6000paircap,50k+50kreserve,fees/funding/liquidity/risk/renewal. No borrowedcapitalorassumedmakerfee.10bpchosenbeforeoutcomes,no grid.',
      arms_design=dict(bounded_entry_72=dict(cap=10,base=True),bounded_pairs_72=dict(cap=10,base=False),unbounded_pairs_72=dict(cap=10000,base=False)),
      evaluation='All6originalreferenceconditionsandmatchedPASchecks. PASpair-onlymeansbounded_entryandbounded_pairsidenticalthere. Reuseddevelopment,noindependentclaim.',orders_sent=0,live_changed=False,independent_test=False,history_reused=True))
    paths=[Path(__file__),Path(__file__).with_name('reference.py'),SOURCE,OUT/'protocol.json']
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})


def verify():
    verify_parent()
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p


def build():
    verify()
    for arm in ARMS:shutil.copy2(SOURCE,OUT/f'{arm}.npz')


def job(arm,phase,scenario):
    spec=read(OUT/'protocol.json')['arms_design'][arm]
    return replay(OUT,OUT/f'{arm}.npz',arm,phase,scenario,cap_bps=spec['cap'],include_base=spec['base'])


def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,p,s) for a in ARMS for p in ['validation','comparison','continuous'] for s in ['base','stress']]):
            r=f.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs','risk_breach_bars']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
