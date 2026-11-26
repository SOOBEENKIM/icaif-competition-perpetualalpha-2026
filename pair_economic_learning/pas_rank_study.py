"""Matched actual-PAS module checks, with daily scores kept separate."""
import json,sys,shutil,gzip,copy
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from pair_value_learning.study import ROOT,read,save,sha
from pair_value_learning import pas_extension as engine
from competition_live.profile import load_profile
from .ranked_pas_replay import replay as ranked_replay
from competition_live.scoring import evaluate
from auto_pair_selection.selection import PAIRS


def configure(wave):
    global WAVE,OUT,SOURCE,ARMS
    WAVE=wave;SOURCE=ROOT/f'reports/pair_economic_learning_20261023/{wave}';OUT=SOURCE/'pas'
    ARMS=read(SOURCE/'protocol.json')['arms']


def verify():
    engine.verify()
    for path,h in read(SOURCE/'freeze.json').items():assert sha(ROOT/path)==h,path
    for path,h in read(OUT/'freeze.json').items():assert sha(ROOT/path)==h,path


def freeze():
    engine.verify()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,anchors=['reference','pas'],scenarios=['base','stress','no_receipts'],
         source=str(SOURCE.relative_to(ROOT)),dataset=str((engine.OUT/'dataset.npz').relative_to(ROOT)),
         design='Same84h PASpairmodule,50k+50kreserve,6000cap,renewablepaidlifecycle,nextbarindependentfills,actualsettledfunding. Bothanchorhypothesesretained. Strictnodeployment/noPASlabeltraining.',
         score='Daily12UTC Q;latestpartialreturnseparate. Not officialaccountresult. PastL2/oraclemissing:conditionalexecution.',independent_test=False,orders_sent=0))
    paths=[Path(__file__),OUT/'protocol.json',engine.OUT/'dataset.npz',*[SOURCE/f'{a}.npz' for a in ARMS]]
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})


def prepare():
    verify();shutil.copy2(engine.OUT/'dataset.npz',OUT/'dataset.npz');d=dict(np.load(OUT/'dataset.npz'));t=d['times']
    for arm in ARMS:
        s=dict(np.load(SOURCE/f'{arm}.npz'));st=s.pop('times');ii=np.searchsorted(st,t);assert np.array_equal(st[ii],t)
        s={k:(v[ii] if v.ndim and len(v)==len(st) else v) for k,v in s.items()}
        visible=np.array([(d['par_volume'][:,a]>0)&(d['par_volume'][:,b]>0) for a,b in s['pairs']]).T
        s['directions']=s['directions']*visible
        for anchor in ['reference','pas']:
            out={k:v.copy() for k,v in s.items()}
            if anchor=='pas':out['source_prices']=d['par_close'].copy()
            np.savez_compressed(OUT/f'{arm}_{anchor}.npz',times=t,**out)


def job(arm,anchor,scenario):
    from pas_allocation_study.study import summarize
    d=dict(np.load(OUT/'dataset.npz'));s=dict(np.load(OUT/f'{arm}_{anchor}.npz'));s.pop('times');p=load_profile()
    cfg=copy.deepcopy(p['configs']['reference']);cfg['dense_pairs']=[list(x) for x in PAIRS];cfg['conditional_completion']={'risk_aversion':0.}
    cfg['scenarios']['no_receipts']={**cfg['scenarios']['base'],'funding_receipt_multiplier':0.}
    r=ranked_replay(d,s,None,cfg,p['trading'],'pairs_dense_hourlyhedge_individual_72',scenario,engine.START,engine.END)
    row=summarize(r,arm,'PAS84h_pair_module',scenario)
    samples=[dict(time=x['time'],equity=x['equity']+50000) for x in r['snapshots'] if x['time']%86400==43200];last=r['snapshots'][-1]
    row.update(evaluate(samples,latest=dict(time=last['time'],equity=last['equity']+50000),trading_days=len({f['time']//86400 for f in r['fills']})))
    row.update(anchor=anchor,official_Q=False,base_included=False,conditional_execution=True)
    assert row['timing_pass'] and row['accounting_error']<1e-6 and not r['owner_fills']['base']
    path=OUT/'runs'/f'{arm}_{anchor}_{scenario}.json.gz';path.parent.mkdir(exist_ok=True)
    with gzip.open(path,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
    save(OUT/'rows'/f'{arm}_{anchor}_{scenario}.json',row)
    return row


def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,b,s) for a in ARMS for b in ['reference','pas'] for s in ['base','stress','no_receipts']]):
            r=f.result();print(json.dumps({k:r[k] for k in ['arm','anchor','scenario','net_pnl','terminal_pnl','completed_pairs','risk_breach_bars']}),flush=True)
    verify()


if __name__=='__main__':
    configure(sys.argv[1]);globals()[sys.argv[2]]()
