import json,sys
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from pair_value_learning.study import ROOT,INPUT,read,save,sha
from .realized_study import verify as verify_parent,OUT as LABELS,SOURCE
from .study import OUT as FEATURES
from .rank_reference import run as replay
from .execution_value import build as build_models
OUT=ROOT/'reports/pair_economic_learning_20261023/wave10'
ARMS=['paid_value_pairs_72','robust_value_pairs_72']


def freeze():
    verify_parent()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,primary=ARMS[0],
        hypothesis='Ordinal ranking must choose least bad opportunities even if every opportunity is negative. Estimate absolute paid cash utility and reject predicted nonpositive entries.',
        label='Reuse3490 exact conditional execution outcomes of wave9;96h strictmaturity; no new labels from test/PAS. Target=costed utility/(sourceknowncapacity*sourcevolsqrt72),scale floored1USDP; base target and min(base,stress) ablation.',
        model='Huberregression,slope1,depth2,64rounds,eta.04,lambda5,minchild1,base0,fixedseed. Dailyfit90dpast,minimum30dailyorigins,30ddecay,equal totalweight perorigin. No hyperparametersearch or earlystopping.',
        policy='Original all45 source economic/riskadmission,originalphysicalquantity/6000cap; require predicted cash>0 and rank by predicted cash. No priceforecast replacement or larger sizing. Pair-only. Original renewal andpaidexit retained.',
        evaluation='Same6reference and6PAS conditions; previously reused development. Outcomes can reject this model, not justify retrospective parameterfit.',orders_sent=0,live_changed=False,independent_test=False))
    paths=[Path(__file__),Path(__file__).with_name('execution_value.py'),LABELS/'execution_labels.json',OUT/'protocol.json']
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})


def verify():
    verify_parent()
    for path,h in read(OUT/'freeze.json').items():assert sha(ROOT/path)==h,path


def build():
    verify();d=dict(np.load(INPUT));f=dict(np.load(FEATURES/'features.npz'))
    def progress(a):
        if a['source']//86400%20==0:print('FIT VALUE',a,flush=True)
    scores,audit,models=build_models(d,f,read(LABELS/'execution_labels.json'),1785542400,progress)
    s=dict(np.load(SOURCE));ii=np.searchsorted(f['times'],s['times']);assert np.array_equal(f['times'][ii],s['times'])
    for kind,arm in zip(['paid_value','robust_value'],ARMS):
        cash=scores[kind][ii];out={k:v.copy() for k,v in s.items()}
        out['ranking_score']=cash;out['economic_action']=np.where(cash>0,out['economic_action'],0)
        np.savez_compressed(OUT/f'{arm}.npz',**out)
    save(OUT/'fit_audit.json',audit);folder=OUT/'models';folder.mkdir(exist_ok=True)
    for day,kind,model in models:model.save_model(folder/f'{day}_{kind}.json')


def job(arm,phase,scenario):return replay(OUT,OUT/f'{arm}.npz',arm,phase,scenario,include_base=False)


def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,p,s) for a in ARMS for p in ['validation','comparison','continuous'] for s in ['base','stress']]):
            r=f.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs','risk_breach_bars']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
