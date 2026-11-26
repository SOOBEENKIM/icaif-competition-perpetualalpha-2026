"""Same PAS replay with only the coherence forecast transform changed."""
import json,sys,shutil
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .study import ROOT,read,save,sha
from .scaled_study import verify as verify_scaled
from .pas_study_v2 import OUT as PARENT,verify as verify_pas
OUT=ROOT/'reports/pair_value_learning_20261023/pas_coherence'
SOURCES={
 'adaptive_coherent_72':'reports/pair_value_learning_20261023/coherence/coherent_adaptive_72.npz',
 'adaptive_scaled_72':'reports/pair_value_learning_20261023/scaled_coherence/coherent_scaled_72.npz',
 'adaptive_stein_72':'reports/pair_value_learning_20261023/scaled_coherence/scaled_stein_72.npz',
 'adaptive_consensus_72':'reports/pair_value_learning_20261023/scaled_coherence/scaled_consensus_72.npz',
}


def freeze():
    verify_scaled();verify_pas()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),primary='adaptive_scaled_72',sources=SOURCES,
        purpose='Compare coherent relativeforecasts withoriginaladaptive72 overexact same actualPAS72h pairmodule dataset. Retainreferenceanchors andoriginalrenewablelifecyclefor everyarm; onlyforecasts/gatechange.',
        controls='Same50k+50kreserve,6000pairmax,conditionalvolume/priceexecution,actualfundingsettlementrecognition,all45,fees,risks. No baseowner. Neitherwholeaccountnorexactfillablebacktest.',
        scientific_limit='AlreadyinspectedPASperiod. Modeltransformsuseonlysourceforecasts/covariances andnoPASlabels. Thisdoesnotcreate anindependentholdout. HistoricalL2/oraclemissinglimitationsretain.',orders_sent=0,live_changed=False))
    paths=[Path(__file__),Path(__file__).with_name('pas_study_v2.py'),Path(__file__).with_name('pas_replay.py'),PARENT/'dataset.npz',OUT/'protocol.json',*[ROOT/p for p in SOURCES.values()]]
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths});print('FROZEN PAScoherence comparison')


def verify():
    verify_scaled();verify_pas()
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p


def prepare():
    verify();shutil.copy2(PARENT/'dataset.npz',OUT/'dataset.npz');d=dict(np.load(OUT/'dataset.npz'));t=d['times']
    for arm,path in SOURCES.items():
        s=dict(np.load(ROOT/path));st=s.pop('times');ix=np.searchsorted(st,t);assert np.array_equal(st[ix],t)
        s={k:(v[ix] if v.ndim and len(v)==len(st) else v) for k,v in s.items()}
        visible=np.array([(d['par_volume'][:,a]>0)&(d['par_volume'][:,b]>0) for a,b in s['pairs']]).T
        s['directions']=s['directions']*visible
        np.savez_compressed(OUT/f'{arm}.npz',times=t,**s)


def job(arm,scenario):
    from . import pas_study_v2 as engine
    engine.OUT=OUT;return engine.job(arm,scenario)


def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,s) for a in SOURCES for s in ['base','stress','no_receipts']]):
            r=f.result();print(json.dumps({k:r[k] for k in ['arm','scenario','net_pnl','Q','completed_pairs','funding']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
