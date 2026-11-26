import json,sys
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .study import ROOT,INPUT,read,save,sha
from .scaled_study import verify as verify_scaled
from .trend_blend import build as build_models
OUT=ROOT/'reports/pair_value_learning_20261023/trend_blend'
SOURCE=ROOT/'reports/pair_value_learning_20261023/scaled_coherence/coherent_scaled_72.npz'
ARMS=['coherent_blend_72','coherent_trend_72']


def freeze():
    verify_scaled()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),input=str(INPUT.relative_to(ROOT)),input_sha256=sha(INPUT),source=str(SOURCE.relative_to(ROOT)),primary='coherent_blend_72',arms=ARMS,
        hypothesis='Coherence alone did not change the actual PAS chosen trade, whose mean-reversion forecast was wrong. Test a fixed 50/50 mixture of coherent AR and 168-hour asset trend projected to72h. Pure trend ablation distinguishes benefit of trend from mixture.',
        model='Current/past168h log price momentum, scaled72/168 then expm1 to arithmetic asset return. Source-frozen hedge maps asset forecasts to pairs. Same4volsqrt72 cap then risk-scaled coherence. No learned mixture weight; no PAS target training.',
        controls='Same45 evaluations, original model admission/source hedge, renewable paid lifecycle, unchanged capital/6000paircap/risks/fees/funding/nextbarfill. No after-result threshold tuning.',
        evaluation='All6 reference conditions, then identical PAS84h conditional module replay. Already used development history, not an independent test.',independent_test=False,orders_sent=0,live_changed=False))
    paths=[Path(__file__),Path(__file__).with_name('trend_blend.py'),Path(__file__).with_name('test_trend_blend.py'),SOURCE,OUT/'protocol.json']
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})


def verify():
    verify_scaled()
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p


def build():
    verify()
    states=build_models(dict(np.load(INPUT)),dict(np.load(SOURCE)))
    for arm,s in states.items():np.savez_compressed(OUT/f'{arm}.npz',**s)


def job(arm,phase,scenario):
    from . import study as engine
    engine.OUT=OUT;return engine.job(arm,phase,scenario)


def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,p,s) for a in ARMS for p in ['validation','comparison','continuous'] for s in ['base','stress']]):
            r=f.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
