import json,sys
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .study import ROOT,INPUT,read,save,sha,verify as verify_calibration
from .scaled_coherence import build as build_models
OUT=ROOT/'reports/pair_value_learning_20261023/scaled_coherence'
SOURCE=ROOT/'reports/pair_model_research_20261023/wave9/adaptive_hedge_72.npz'
ARMS=['coherent_scaled_72','scaled_stein_72','scaled_consensus_72']


def freeze():
    verify_calibration()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),input=str(INPUT.relative_to(ROOT)),input_sha256=sha(INPUT),source=str(SOURCE.relative_to(ROOT)),arms=ARMS,primary='coherent_scaled_72',
       prior='Unweightedcoherence improvedfullbasePnL646.66->798.81,Q.1591->.1716,but drawdown/volatilityworsened andlaterstressfailed. Preservefailure,donotclaimdominance.',
       hypothesis='Scale eachrelativeforecastbyits ownpastvolatility*sqrt72 beforecoherenceprojection, so highvolatilitypairsdo notdominate leastsquares. The complete-cohort weightedSSE identity is exact; profitandselected-subseterrorsarenotguaranteed.',
       ablations='PositivepartStein shrinkage estimates noise fromorthogonalresidual energy andrank; isotropicindependenceassumptionsarenotestablishedforfinancialpairs,so noSteindominanceclaim. Separateconsensusarm requiresraw/projected same sign andcapsprojectedmagnitude atoriginalmagnitude. Theseviewscorrelated,notindependentexpertvotes.',
       controls='SameoriginaladaptiveSMAmean,sourcehedge,all45,renewablelifecycle,capital,risk,fees,liquidity,delayedfills. No hindsightassets,parametersorchanges tolivecode.',
       evaluation='All3arms all6referenceconditions plus frozenPASchecks. Primaryfixedbeforeoutcomes. Reuseddevelopment; neither newindependenttestnorpeerperformanceproof.',history_reused=True,independent_test=False,orders_sent=0,live_changed=False))
    paths=[Path(__file__),Path(__file__).with_name('scaled_coherence.py'),Path(__file__).with_name('test_scaled_coherence.py'),Path(__file__).with_name('study.py'),SOURCE,OUT/'protocol.json']
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths});print('FROZEN risk-scaledcoherence')


def verify():
    verify_calibration()
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p


def build():
    verify();states,audit=build_models(dict(np.load(SOURCE)))
    for arm,s in states.items():np.savez_compressed(OUT/f'{arm}.npz',**s)
    save(OUT/'projection_audit.json',audit);print({a:int(np.count_nonzero(s['economic_action'])) for a,s in states.items()},flush=True)


def job(arm,phase,scenario):
    from . import study as engine
    engine.OUT=OUT;return engine.job(arm,phase,scenario)


def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,p,s) for p in ['validation','comparison','continuous'] for a in ARMS for s in ['base','stress']]):
            r=f.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
