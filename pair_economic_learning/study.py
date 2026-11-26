import sys,json
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from pair_value_learning.study import ROOT,INPUT,read,save,sha
from pair_value_learning.scaled_study import verify as verify_parent
from .model import Spec,prepare as prepare_features,build as build_models

OUT=ROOT/'reports/pair_economic_learning_20261023/wave1'
ARMS=['online_mixture_72','tree_direct_72','tree_coherent_72','tree_residual_72']


def freeze():
    verify_parent()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    import xgboost as xgb
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),input=str(INPUT.relative_to(ROOT)),input_sha256=sha(INPUT),arms=ARMS,primary='tree_coherent_72',spec=asdict(Spec()),xgboost=xgb.__version__,
       hypothesis='Past calibrated nonlinear dependence on trend, relation strength and forecast size can reduce false mean reversion. A causal prequential AR/trend blend isolates regime mixing. Reconciled forecasts isolate cross-pair algebra from supervised prediction.',
       features='17 current/past features: coherentAR and168htrend,physical6/24/72/168hmomentum,acceleration,volratio,return/levelcorrelation,logbeta,halflife,ARtstat,normalizedcost,absolute72hmomentum,trendinteraction,expert disagreement. No asset names or PAS target outcomes.',
       training='Daily00UTC,90d past sources,6h stride,30d decay,equal total weight per origin across dependent pairs. Target physical source-beta72h return/source risk,clip4 onlyfortraining. Strict maturity<fit. No test-based earlystopping/tuning. Shallow depth2,64rounds,eta.04 fixed before results.',
       mixture='Daily pooled scalar weight on AR versus trend minimizes matured source-risk weighted error;fixed72h calibration spacing,prior6 pseudo-origins at50/50,constrained[0,1]. Not fitted to PAS labels.',
       controls='Same original source hedge/admission(all45 checked),same50k+50kreserve/6000paircap,renewablepaidlifecycle,base strategy,costs,liquidity,risk. Threshold38bp retained for this single model comparison.',
       evaluation='All6 original reference conditions andall3PAS84h conditions. Previously reused development data, not independent. Report every arm; no live updates.',
       criteria='Functionalcausality/accounting/risk;fullreferencePnL/Q improve againstincumbentunderbaseandstress;latercomparisonPnL/Qnotworse;PASpositiveaftercosts/reserveandclosedpairinbase/stress. No claim that passing implies future dominance or zero possible issues.',
       history_reused=True,independent_test=False,orders_sent=0,live_changed=False))
    paths=[*Path(__file__).parent.glob('*.py'),OUT/'protocol.json',ROOT/'pair_model_research/adaptive_hedge.py',ROOT/'pair_value_learning/scaled_coherence.py']
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})


def verify():
    verify_parent();p=read(OUT/'protocol.json');assert sha(INPUT)==p['input_sha256']
    for f,h in read(OUT/'freeze.json').items():assert sha(ROOT/f)==h,f
    return p


def build():
    p=verify();d=dict(np.load(INPUT))
    f,formation=prepare_features(d);np.savez_compressed(OUT/'features.npz',**f);save(OUT/'formation.json',formation)
    def progress(r):
        if r['source']//86400%10==0:print('FIT',datetime.fromtimestamp(r['source'],timezone.utc).isoformat(),r['mixture'],r['tree'],flush=True)
    states,fits,models=build_models(d,f,1785542400,Spec(**p['spec']),progress)
    for arm,s in states.items():np.savez_compressed(OUT/f'{arm}.npz',**s)
    save(OUT/'fit_audit.json',fits)
    folder=OUT/'models';folder.mkdir(exist_ok=True)
    for day,mm in models:
        for k,m in mm.items():m.save_model(folder/f'{day}_{k}.json')
    verify();print('BUILT',len(fits),'daily sources',flush=True)


def job(arm,phase,scenario):
    from pair_value_learning import study as engine
    engine.OUT=OUT;return engine.job(arm,phase,scenario)


def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,p,s) for a in ARMS for p in ['validation','comparison','continuous'] for s in ['base','stress']]):
            r=f.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs','risk_breach_bars']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
