import sys,json
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from pair_value_learning.study import ROOT,INPUT,read,save,sha
from .study import verify as verify_parent,OUT as PARENT
from .meta import build as build_models
OUT=ROOT/'reports/pair_economic_learning_20261023/wave2'
ARMS=['cost_meta_72','cost_value_72']


def freeze():
    verify_parent()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),input=str(INPUT.relative_to(ROOT)),input_sha256=sha(INPUT),arms=ARMS,primary='cost_meta_72',
       hypothesis='Select the prior best coherent AR entries by costed outcomes conditional on both original cash-budget leg limits filling; generic mean regression did not improve referenceportfolio.',
       label='Source-fixed normalized physical hedge,exact original reservation_limits9bpfee10bpimpact5bpbuffer10000bpscap;independentnextbarlimitchecks;source+72hpaidexit;actualreferencefundingafterarrivalthroughclose. No lot/queue/partialfill equivalence claimed. Full replay separately includes those constraints.',
       learning='Same17causalfeaturesplus proposedside,90dpast6hsourcegrid,strictmaturity<fit,30ddecay,equaloriginweight;depth2/64rounds/eta.04/lambda5;classifier>0.5 orconditionalvalue>0 gates,unchangedparentmu/sourcehedge. No tuned threshold, no PAS targets.',
       controls='Priorbestcoherentforecast,caps,costs,base,liquidity,renewalallunchanged. Modelgateonly. All6referenceandall6PASconditionsreport. Reuseddevelopment;not independent.',
       orders_sent=0,live_changed=False,independent_test=False,history_reused=True))
    paths=[Path(__file__),Path(__file__).with_name('meta.py'),Path(__file__).with_name('test_meta.py'),PARENT/'features.npz',OUT/'protocol.json']
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})


def verify():
    verify_parent()
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p


def build():
    verify();d=dict(np.load(INPUT));f=dict(np.load(PARENT/'features.npz'))
    def progress(a):
        if a['source']//86400%20==0:print('FIT',a,flush=True)
    states,audit,models,pred,net,both=build_models(d,f,1785542400,progress)
    for arm,s in states.items():np.savez_compressed(OUT/f'{arm}.npz',**s)
    np.savez_compressed(OUT/'labels_predictions.npz',net=net,both=both,**pred);save(OUT/'fit_audit.json',audit)
    folder=OUT/'models';folder.mkdir(exist_ok=True)
    for day,mm in models:
        for k,m in mm.items():m.save_model(folder/f'{day}_{k}.json')
    verify();print('BUILT conditional cost gates',flush=True)


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
