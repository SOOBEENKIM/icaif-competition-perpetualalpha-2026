import copy,gzip,json,sys
from dataclasses import asdict
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .model import Spec,build as build_models
from pair_regime_research.study import ROOT,INPUT,read,save,sha
from auto_pair_selection.selection import PAIRS
from auto_pair_selection.study import verify as verify_live
from competition_live.profile import BUNDLE,load_profile
OUT=ROOT/'reports/pair_value_learning_20261023'
ARMS=[f'{kind}_{h}' for h in [72,168] for kind in ['pooled','hierarchical']]


def freeze():
    verify_live()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),input=str(INPUT.relative_to(ROOT)),input_sha256=sha(INPUT),spec=asdict(Spec()),arms=ARMS,primary='hierarchical_72',
        hypothesis='Previous regime selector chose between uncalibrated magnitude forecasts. Learn actual source-frozen hedge returns from matured outcomes, pooling relationships then shrinking local calibration toward a common prior; compare72h and168h economic horizons.',
        relation='Retain previous positive hourly-return minimum-variance hedge,>=5% estimated riskreduction,three20daypositivebeta blocks,maxratio3,60dformation. All45 assessed, no asset exclusions.',
        train='Daily90d rolling,6h source sampling,30d decay; strict target maturity<fit source; equal total weight per source across dependent pairs; feature and target scale from sourcevolatility. No PAS target training. Local ridge shrinks to pooled model, pair name isnot a predictor or winning-pair whitelist.',
        target='Subsequent arithmetic relative price return holding sourcebeta and quantities fixed. Never difference future-refitted residuals. Clip4sigma only for training/predictions; report unclipped observed errors/PnL.',
        features='Source-frozenAR forecast,24h72h168hrelative momentum,trend interaction,acceleration,AR availability,logbeta. Predictionmagnitude is calibrated toactual tradable sourcehedge returns.',
        controls='Same50k+50kreserve,6000pairgross,onepair,allmarketdirectionalbase,fees/funding/delayedpartialfills/risk. Existing paid lifecycle retained. No scale increase, no livepromotion.',
        evaluation_start=1785542400,history_reused=True,independent_test=False,
        selection='Primary hierarchical72 fixed now. All4 arms all6conditions; no claim that ex-post winner is independent test. Separate actual PAS price diagnostics; no officialrank claim.',
        research=[dict(url='https://arxiv.org/html/2606.00060v1',use='2026 research illustrates prediction-to-trading gap and cost-awareposition changes. This is not a reproduction of itsBTC hourlyXGBoost/LSTM/Transformer or performance.')],orders_sent=0,live_changed=False))
    paths=[*Path(__file__).parent.glob('*.py'),OUT/'protocol.json',ROOT/'pair_regime_adaptive/model.py',ROOT/'pair_regime_research/replay.py',ROOT/'pair_regime_research/lifecycle.py']
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths});print('FROZEN calibration experiment',flush=True)


def verify():
    verify_live();p=read(OUT/'protocol.json');assert sha(ROOT/p['input'])==p['input_sha256']
    for path,digest in read(OUT/'freeze.json').items():assert sha(ROOT/path)==digest,path
    return p


def build():
    p=verify();d=dict(np.load(ROOT/p['input']))
    for h in [72,168]:
        states,fits,audit,f=build_models(d,p['evaluation_start'],h,Spec(**p['spec']))
        for arm,state in states.items():np.savez_compressed(OUT/f'{arm}.npz',**state)
        save(OUT/f'fits_{h}.json',fits);save(OUT/f'formation_{h}.json',audit)
        np.savez_compressed(OUT/f'features_{h}.npz',**f)
        print(h,{arm:int(np.count_nonzero(s['economic_action'])) for arm,s in states.items()},flush=True)
    verify()


def job(arm,phase,scenario):
    from pair_regime_research.replay import replay
    from pas_performance_research.selected import controller
    from pas_allocation_study.study import summarize
    p=read(OUT/'protocol.json');profile=load_profile();w=next(x for x in read(BUNDLE/'expected.json')['runs'] if x['venue']=='reference' and x['arm']==profile['candidate'] and x['phase']==phase and x['scenario']==scenario);dest=OUT/'reference'/f'{arm}_{phase}_{scenario}.json.gz'
    if dest.exists():
        with gzip.open(dest,'rt') as fp:r=json.load(fp)
    else:
        d=dict(np.load(BUNDLE/'data/reference.npz'));s=dict(np.load(OUT/f'{arm}.npz'));t=s.pop('times');ii=np.searchsorted(t,d['times']);assert np.array_equal(t[ii],d['times']);s={k:(v[ii] if v.ndim and len(v)==len(t) else v) for k,v in s.items()}
        z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz'));zi=np.searchsorted(z['times'],d['times']);f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz'));fi=np.searchsorted(f['times'],d['times']);raw=dict(np.load(ROOT/p['input']))
        cfg=copy.deepcopy(profile['configs']['reference']);cfg['dense_pairs']=[list(x) for x in PAIRS];cfg['conditional_completion']={'risk_aversion':0.};c=controller(d['times'],f['forecast'][fi],f['carry'][fi],z['all_market_signed_core_v1'][zi],z['all_market_signed_core_v1_protected'][zi],price_times=raw['market_times'],prices=raw['close'])
        h=int(arm.rsplit('_',1)[1]);r=replay(d,s,None,cfg,profile['trading'],f'first_dense_hourlyhedge_individual_{h}',scenario,w['start'],w['end'],base_policy=c);dest.parent.mkdir(exist_ok=True)
        with gzip.open(dest,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
    row=summarize(r,arm,phase,scenario);row['relationship_exits']=sum(e['event']=='exit_requested' and e.get('reason')=='relationship_invalidated' for e in r['events']);row['entered_pairs']=sorted({tuple(e['pair']) for e in r['events'] if e['event']=='pair_intent'});assert row['timing_pass'] and row['accounting_error']<1e-6
    row['eligible']=row['risk_breach_bars']==0;save(OUT/'rows'/f'{arm}_{phase}_{scenario}.json',row);return row


def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(job,a,p,s) for p in ['validation','comparison','continuous'] for a in ARMS for s in ['base','stress']]
        for done in as_completed(jobs):
            r=done.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs','risk_breach_bars']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
