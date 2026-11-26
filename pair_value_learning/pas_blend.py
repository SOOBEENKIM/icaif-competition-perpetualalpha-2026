"""Fixed reference/PAS anchor alternatives for the predeclared trend mixture."""
import json,sys,shutil
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .study import ROOT,read,save,sha
from .blend_study import verify as verify_blend
from . import pas_extension as engine
OUT=ROOT/'reports/pair_value_learning_20261023/pas_blend'
SOURCES={f'adaptive_{kind}_{anchor}_72':f'reports/pair_value_learning_20261023/trend_blend/coherent_{kind}_72.npz' for kind in ['blend','trend'] for anchor in ['reference','pas']}


def freeze():
    engine.verify();verify_blend()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),primary='adaptive_blend_pas_72',sources=SOURCES,
       design='Same actualPAS84h pair-module data, costs, originalrenewal and caps. Both reference and PAS source-price anchors, all4reported. No outcome-selectedweights orperiods. Reuseddevelopment, conditionalfills, not wholeaccount.',
       score='Engine rows include hypothetical intradayQ. Only completed-noonQ follows current leaderboard standings; report PnL andcompletedpairs without conflating.',independent_test=False,live_changed=False,orders_sent=0))
    paths=[Path(__file__),Path(engine.__file__),OUT/'protocol.json',engine.OUT/'dataset.npz',*[ROOT/p for p in SOURCES.values()]]
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})


def verify():
    engine.verify();verify_blend()
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p


def prepare():
    verify();shutil.copy2(engine.OUT/'dataset.npz',OUT/'dataset.npz');d=dict(np.load(OUT/'dataset.npz'));t=d['times']
    for arm,path in SOURCES.items():
        s=dict(np.load(ROOT/path));st=s.pop('times');ix=np.searchsorted(st,t);assert np.array_equal(st[ix],t)
        s={k:(v[ix] if v.ndim and len(v)==len(st) else v) for k,v in s.items()}
        if '_pas_' in arm:s['source_prices']=d['par_close'].copy()
        visible=np.array([(d['par_volume'][:,a]>0)&(d['par_volume'][:,b]>0) for a,b in s['pairs']]).T
        s['directions']=s['directions']*visible
        np.savez_compressed(OUT/f'{arm}.npz',times=t,**s)


def job(arm,scenario):
    engine.OUT=OUT;return engine.job(arm,scenario)


def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,s) for a in SOURCES for s in ['base','stress','no_receipts']]):
            r=f.result();print(json.dumps({k:r[k] for k in ['arm','scenario','net_pnl','completed_pairs']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
