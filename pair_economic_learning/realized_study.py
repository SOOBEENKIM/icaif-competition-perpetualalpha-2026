import json,sys,gzip
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from pair_value_learning.study import ROOT,INPUT,read,save,sha
from competition_live.profile import BUNDLE
from .ranking_study import verify as verify_parent,SOURCE
from .study import OUT as FEATURES
from .rank_reference import run as replay
from . import realized_labels as labels
from .realized_ranker import build as build_models
OUT=ROOT/'reports/pair_economic_learning_20261023/wave9'
ARMS=['realized_rank_pairs_72','robust_rank_pairs_72']


def freeze():
    verify_parent()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,primary=ARMS[1],
        hypothesis='Auxiliary markout labels omit the real losses of partial fills, repairs, rounding and paid exits. Rank eligible pairs using source-causal features and outcomes from the same conditional executor.',
        label='Independent flat50k counterfactuals,6000cap,all45eligiblepairs,daily12UTC origins,72h forecast then24h closure allowance,original paid independent-leg fills/completion/stops/funding. Disable new opportunities after initial source. Utility=marked96hPnL less residual exit reserve, not assumed fully executed liquidation. Train only after96h maturity strictly before fit.',
        training='Reference input through16Oct07UTC; final label source12Oct12UTC. No PAS outcomes. 90d past dailyorigins,minimum30origins,30ddecay; same fixed19features/depth2/64rounds/eta.04/lambda5. Robust target=min(base,stress); base-only ablation. No earlystopping/test tuning.',
        limitations='Training labels are isolated 72h episodes; portfolio replay retains original renewal to168h and sequential capital contention. Candle execution is conditional, not historical L2 evidence. Reused research periods, not an independent holdout.',
        controls='Pair-only,original6000cap,full frozen forecasts,all45admission,costs,risk,renewals. Rank ordering only. Original live untouched.',
        evaluation='All6 reference conditions then6 PAS conditions each arm. Preserve failed arms. No automatic live change.',orders_sent=0,live_changed=False,independent_test=False))
    paths=[Path(__file__),*[Path(__file__).with_name(n) for n in ['realized_labels.py','realized_ranker.py','test_realized.py']],BUNDLE/'data/reference.npz',FEATURES/'features.npz',SOURCE,OUT/'protocol.json']
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})


def verify():
    verify_parent()
    for path,h in read(OUT/'freeze.json').items():assert sha(ROOT/path)==h,path


def collect():
    verify();labels.initialize();t=labels.DATA['times']
    ii=np.flatnonzero((t%86400==43200)&(np.arange(len(t))+labels.HOURS<len(t))&labels.STATE['economic_available'].any(1))
    allrows=[]
    with ProcessPoolExecutor(max_workers=2,initializer=labels.initialize) as pool:
        for j,future in enumerate(as_completed([pool.submit(labels.origin,int(i)) for i in ii])):
            rows=future.result();allrows.extend(rows)
            if j%10==0:print('LABELS',j+1,'/',len(ii),'episodes',len(allrows),flush=True)
    allrows.sort(key=lambda x:(x['source'],x['pair'],x['scenario']))
    save(OUT/'execution_labels.json',allrows)
    repair=next((r for r in allrows if r['one_leg'] and r['fills']),None)
    if repair:
        row,trace=labels.episode(repair['index'],repair['pair'],repair['scenario'],retain=True)
        assert row==repair
        with gzip.open(OUT/'one_leg_label_fixture.json.gz','wt') as fp:json.dump(trace,fp,allow_nan=False,default=lambda x:x.tolist())
    save(OUT/'label_audit.json',dict(episodes=len(allrows),origins=len(ii),one_leg=sum(r['one_leg'] for r in allrows),repairs=sum(r['repaired'] for r in allrows),residual=sum(r['residual']>0 for r in allrows),breached=sum(r['breaches']>0 for r in allrows),fixture=repair))
    verify()


def build():
    verify();d=dict(np.load(INPUT));f=dict(np.load(FEATURES/'features.npz'))
    def progress(a):
        if a['source']//86400%20==0:print('FIT',a,flush=True)
    scores,audit,models=build_models(d,f,read(OUT/'execution_labels.json'),1785542400,progress)
    s=dict(np.load(SOURCE));ii=np.searchsorted(f['times'],s['times']);assert np.array_equal(f['times'][ii],s['times'])
    for kind,arm in zip(['realized','robust'],ARMS):np.savez_compressed(OUT/f'{arm}.npz',**s,ranking_score=scores[kind][ii])
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
