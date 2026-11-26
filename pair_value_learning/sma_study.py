import json,sys
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .study import ROOT,INPUT,read,save,sha,verify as verify_calibration
from .sma_calibration import build as build_models
OUT=ROOT/'reports/pair_value_learning_20261023/sma_calibration'
ARMS=['pooled_sma_72','local_sma_72']


def freeze():
    verify_calibration()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),input=str(INPUT.relative_to(ROOT)),input_sha256=sha(INPUT),primary='local_sma_72',arms=ARMS,
       hypothesis='Replace neither successfulARprior nor its sourcehedge. Calibrate publishedrawmeanreversionforecast magnitude with fixed-source actual72h returns. Separate calibrationscope from newcomplexforecasters which failed.',
       calibration='90d maturedhistory,72h disjointorigins,30drecency. Sharedslope shrunktoward0 withpenalty6; localslope shrunktowardsharedwithpenalty6,min8maturedorigins. Bound0..1; no negative-slope reversal. All45, no winningpair whitelist. Zero slope means no demonstratedcalibratedpriceedge.',
       relation_limit='UseslegacySMArelative-value prior. Empiricalcalibration isnot proof of stablecointegrationor minimumvariancehedge. Recordweakrelationshipdiagnostics ratherthan claim structuralv2checksarepresent. This is a researchablation; cannot automatically replace thelive strategy.',
       evaluation='Same6 referenceconditionswith originalrenewablelifecycle; botharms reported. Alreadyreusedhistory, notindependenttest. Any favorable result requires actualPASseparateassessment.',history_reused=True,independent_test=False,live_changed=False,orders_sent=0))
    paths=[Path(__file__),Path(__file__).with_name('sma_calibration.py'),Path(__file__).with_name('test_sma_calibration.py'),Path(__file__).with_name('study.py'),ROOT/'pair_model_research/adaptive_hedge.py',OUT/'protocol.json']
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths});print('FROZEN sourceARcalibration')


def verify():
    verify_calibration()
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p


def build():
    verify();d=dict(np.load(INPUT));states,records,formation=build_models(d,1785542400)
    for arm,s in states.items():np.savez_compressed(OUT/f'{arm}.npz',**s)
    save(OUT/'calibration.json',records);save(OUT/'formation.json',formation);print({a:int(np.count_nonzero(s['economic_action'])) for a,s in states.items()},flush=True)


def job(arm,phase,scenario):
    from . import study as engine
    engine.OUT=OUT
    return engine.job(arm,phase,scenario)


def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,p,s) for p in ['validation','comparison','continuous'] for a in ARMS for s in ['base','stress']]):
            r=f.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
