import json,sys
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .study import ROOT,INPUT,read,save,sha,verify as verify_calibration
from .coherence import build as build_models
OUT=ROOT/'reports/pair_value_learning_20261023/coherence'
SOURCES={
 'coherent_adaptive_72':'reports/pair_model_research_20261023/wave9/adaptive_hedge_72.npz',
 'coherent_hierarchical_72':'reports/pair_value_learning_20261023/hierarchical_72.npz',
}


def freeze():
    verify_calibration()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),input=str(INPUT.relative_to(ROOT)),input_sha256=sha(INPUT),sources=SOURCES,primary='coherent_adaptive_72',
      hypothesis='Independentlypredicted45pair means neednotbe realizable by any10asset futurepricevector. Project onto columnspaceofsourcefixedphysicalhedge matrix. Remove algebraicallyimpossible cyclicforecastcomponent withoutfittinganynewtarget or thresholds.',
      mathematical_scope='For complete alignedpairtargets y=W*r, Euclideanorthogonalprojection cannotincrease total squaredforecasterror: ||mu-y||²-||Pmu-y||²=||(I-P)mu||². Doesnotguarantee correctsign, profitability, smallererroronselectedsubset,orPAS execution.',
      preservation='Samehedges,originalforecastparents,relationpermissions,renewablelifecycle,sourceinputs,budget,risk,fees,funding,latency. No asset exclusion, no leverageincrease. Originaladaptiveparent stilllacksstructuralcointegrationproof; projectiondoesnot supplythatproof.',
      evaluation='Botharms all6referenceconditions. Primaryfixednow; reportfailures. Separate complete-cohorterroridentitycheck and PAS assessment. Reuseddevelopment,notindependenttest.',history_reused=True,independent_test=False,orders_sent=0,live_changed=False))
    paths=[Path(__file__),Path(__file__).with_name('coherence.py'),Path(__file__).with_name('test_coherence.py'),Path(__file__).with_name('study.py'),OUT/'protocol.json',*[ROOT/p for p in SOURCES.values()]]
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths});print('FROZEN crosspaircoherence')


def verify():
    verify_calibration()
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p


def build():
    verify()
    for arm,path in SOURCES.items():
        state,audit=build_models(dict(np.load(ROOT/path)));np.savez_compressed(OUT/f'{arm}.npz',**state);save(OUT/f'{arm}_audit.json',audit)
        print(arm,'removed forecastenergy',sum(r['removed_energy'] for r in audit)/sum(r['original_energy'] for r in audit),flush=True)


def job(arm,phase,scenario):
    from . import study as engine
    engine.OUT=OUT;return engine.job(arm,phase,scenario)


def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,p,s) for p in ['validation','comparison','continuous'] for a in SOURCES for s in ['base','stress']]):
            r=f.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
