import json,sys
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .study import ROOT,INPUT,read,save,sha,verify as verify_calibration
from .net_value import state
from auto_pair_selection.study import verify as verify_live
OUT=ROOT/'reports/pair_value_learning_20261023/net_value'
SOURCES={
 'hierarchical_net_72':'reports/pair_value_learning_20261023/hierarchical_72.npz',
 'adaptive_net_72':'reports/pair_model_research_20261023/wave9/adaptive_hedge_72.npz',
 'carry_net_24':'reports/pair_regime_adaptive_20261023/regime_choice_72.npz',
 'carry_net_72':'reports/pair_regime_adaptive_20261023/regime_choice_72.npz',
}


def freeze():
    verify_calibration()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),input=str(INPUT.relative_to(ROOT)),input_sha256=sha(INPUT),sources=SOURCES,primary='hierarchical_net_72',
        hypothesis='Prior priceonly38bp prefilter rejects trades before existingplanner can account for funding and venue/referencepricegap. Remove that prefilter, preserve relationshippermission, and require the original fullnetcash planner to pay all costs. No guaranteedfundingor convergence assumedproven.',
        parent='Fixedmaturityversion for strict72h/168h labelalignment, not renewablebaseline. Compare withboth fixed andrenewableparent results. Initialfixedmaturity stopanchor omission correctedin isolatedengine with exactoldpathparity.',
        carry_ablation='Carryonly state deliberately predicts zero relativepricechange, ratherthan fabricating positivepricealpha. Carryprojection remains conservative halfpositive/fullnegative knownrateoverhorizon; itmay fail. Positivebeta riskrelationfilterretained.24h and72h predeclared.',
        controls='Same50k+50kreserve,6000pairgross,onepair,currentknownfunding,costs,risk,paidpartialfills,lotconstraints. No makerfeeassumptionchange. No livepromotion.',
        evaluation='All4arms all6referenceconditions plus actualPASconditionalchecks. No pricesignalaccuracyclaimfor carryonlystrategy. Already reusedhistories, notindependenttest.',history_reused=True,independent_test=False,live_changed=False,orders_sent=0))
    paths=[Path(__file__),Path(__file__).with_name('net_value.py'),Path(__file__).with_name('test_net_value.py'),Path(__file__).with_name('fixed_replay.py'),Path(__file__).with_name('lifecycle_study_v2.py'),OUT/'protocol.json',*[ROOT/v for v in SOURCES.values()]]
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths});print('FROZEN fullnetvaluegate')


def verify():
    verify_live();verify_calibration()
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p


def build():
    verify()
    for arm,path in SOURCES.items():np.savez_compressed(OUT/f'{arm}.npz',**state(dict(np.load(ROOT/path)),carry_only=arm.startswith('carry')))


def job(arm,phase,scenario):
    from . import lifecycle_study_v2 as engine
    engine.OUT=OUT;engine.SOURCES={a:str((OUT/f'{a}.npz').relative_to(ROOT)) for a in SOURCES}
    return engine.job(arm,phase,scenario)


def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(job,a,p,s) for p in ['validation','comparison','continuous'] for a in SOURCES for s in ['base','stress']]
        for done in as_completed(jobs):
            r=done.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
