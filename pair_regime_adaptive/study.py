"""Second, explicitly frozen experiment after strict cointegration excluded all."""
import copy,gzip,json,hashlib,sys
from dataclasses import asdict
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .model import Spec,ARMS,build as build_models
from pair_regime_research.study import ROOT,INPUT,read,save,sha
from auto_pair_selection.selection import PAIRS
from auto_pair_selection.study import verify as verify_live
from competition_live.profile import BUNDLE,load_profile
OUT=ROOT/'reports/pair_regime_adaptive_20261023'
def freeze():
    verify_live()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),input=str(INPUT.relative_to(ROOT)),input_sha256=sha(INPUT),spec=asdict(Spec()),arms=list(ARMS),primary='regime_choice_72',
        prior_result='Strict long-run cointegration + blockstability producedzero qualified entry signals. Preserve previous experiment, not relax its tests to manufacture significance.',
        hypothesis='Distinguish statistically stable price-level reversion from a predictive relative-return trade. Use positive return minimum-variance hedge, requiring>=5% estimated variance reduction and3-block positive beta ratio<=3. Unrelated/negative-return hedges are rejected. This doesnotclaim cointegration.',
        experts='72h localAR reversion versus72h relative continuation; hedge andAR fitted daily onpast60d. Every past forecast stored atits source. Label is subsequent72h beta-weighted arithmetic relative return.',
        regime='Daily compare only matured previouslypublished predictions: last60days,origins spaced72h apart,min10,strict maturity<selectionsource. Need MAE<zero prediction and positive mean38bp-cost proxy,>=3 active sources. Choose larger recencyweightedproxy net; ensemble diagnostic weights positive forecast skill. No PASlabelsused.',
        holding='Keep forecasts/IDs for existing inventory even if no new expert passes. Structural hedgefailure uses shared two-distinct-source paid close rule. Mere lack of model selection evidence doesnotforce a paid close.',
        fixed='Same4+6 replay controls,costs,funding,risk,6000pairmax,50k sleeve+50kreserve,original allmarket base. No livepromotion.',
        evaluation_start=1785542400,history_reused=True,independent_test=False,experiment_number=2,
        evaluation='Both candidates all6 reference conditions plus same frozen PASdiagnostics. Primary fixed before results; preserve failures. Perpairpast modelselection and subsequent top-pair execution still have selectionrisk.',orders_sent=0,live_changed=False))
    paths=[*Path(__file__).parent.glob('*.py'),OUT/'protocol.json',ROOT/'pair_regime_research/replay.py',ROOT/'pair_regime_research/lifecycle.py']
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths});print('FROZEN predictive regime selection',flush=True)
def verify():
    verify_live();p=read(OUT/'protocol.json');assert sha(ROOT/p['input'])==p['input_sha256']
    for path,digest in read(OUT/'freeze.json').items():assert sha(ROOT/path)==digest,path
    return p
def build():
    p=verify();d=dict(np.load(ROOT/p['input']));states,audits,experts=build_models(d,p['evaluation_start'],Spec(**p['spec']))
    for arm,state in states.items():np.savez_compressed(OUT/f'{arm}.npz',**state)
    np.savez_compressed(OUT/'expert_forecasts.npz',times=d['market_times'],**experts);save(OUT/'formation_regime_audit.json',audits)
    save(OUT/'counts.json',{a:dict(available=int(s['economic_available'].sum()),entry_signals=int((s['economic_action']!=0).sum()),reversion_selected=int((s['regime']==1).sum()),trend_selected=int((s['regime']==2).sum()),ensemble_selected=int((s['regime']==3).sum())) for a,s in states.items()});verify();print(read(OUT/'counts.json'),flush=True)
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
        r=replay(d,s,None,cfg,profile['trading'],'first_dense_hourlyhedge_individual_72',scenario,w['start'],w['end'],base_policy=c);dest.parent.mkdir(exist_ok=True)
        with gzip.open(dest,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
    row=summarize(r,arm,phase,scenario);row['relationship_exits']=sum(e['event']=='exit_requested' and e.get('reason') in ['relationship_invalidated','persistent_divergence'] for e in r['events']);row['entered_pairs']=sorted({tuple(e['pair']) for e in r['events'] if e['event']=='pair_intent'});assert row['timing_pass'] and row['accounting_error']<1e-6
    row['eligible']=row['risk_breach_bars']==0;save(OUT/'rows'/f'{arm}_{phase}_{scenario}.json',row);return row
def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(job,a,p,s) for p in ['validation','comparison','continuous'] for a in ARMS for s in ['base','stress']]
        for done in as_completed(jobs):
            r=done.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs','risk_breach_bars']}),flush=True)
    verify()
if __name__=='__main__':globals()[sys.argv[1]]()
