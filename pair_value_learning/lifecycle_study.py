"""Ablate renewed inventory against a source-fixed forecast maturity."""
import copy,gzip,json,sys
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .study import ROOT,INPUT,read,save,sha,verify as verify_calibration
from auto_pair_selection.study import verify as verify_live
from auto_pair_selection.selection import PAIRS
from competition_live.profile import BUNDLE,load_profile
OUT=ROOT/'reports/pair_value_learning_20261023/fixed_maturity'
SOURCES={
 'adaptive_hedge_72':'reports/pair_model_research_20261023/wave9/adaptive_hedge_72.npz',
 'regime_choice_72':'reports/pair_regime_adaptive_20261023/regime_choice_72.npz',
 'hierarchical_72':'reports/pair_value_learning_20261023/hierarchical_72.npz',
 'hierarchical_168':'reports/pair_value_learning_20261023/hierarchical_168.npz',
}


def freeze():
    verify_calibration()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=SOURCES,primary='hierarchical_72',
        hypothesis='Forecast labels use72h/168h fixedsourcehedge, but renewableinventory often holds138-146h for72h forecast. Disable renewal and paidnewbeta adjustments to isolate horizon mismatch, keep originalriskstops and paidpartialclosures.',
        interpretation='Renewal based on new signals is not by itself a codebug. This is a policycomparison and may decrease performance. Actual closure can still exceed maturity because fills are constrained; no forcedfreefills.',
        controls='Frozen pricepredictions,formationflags,positionsizes,fees,funding,all45,pastinput,directionalbase,risklimits; only arm disables renewal. No trainingor outcome-drivenforecastselection.',
        evaluation='All4sourcecandidates,all6referenceconditions. Primary fixed before results. No independent test; reused development. No livepromotion.',live_changed=False,orders_sent=0))
    files=[Path(__file__),OUT/'protocol.json',*[ROOT/p for p in SOURCES.values()]]
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in files});print('FROZEN fixed maturity comparison')


def verify():
    verify_live();verify_calibration()
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p


def job(arm,phase,scenario):
    from pair_regime_research.replay import replay
    from pas_performance_research.selected import controller
    from pas_allocation_study.study import summarize
    profile=load_profile();w=next(x for x in read(BUNDLE/'expected.json')['runs'] if x['venue']=='reference' and x['arm']==profile['candidate'] and x['phase']==phase and x['scenario']==scenario);dest=OUT/'reference'/f'{arm}_{phase}_{scenario}.json.gz'
    if dest.exists():
        with gzip.open(dest,'rt') as fp:r=json.load(fp)
    else:
        d=dict(np.load(BUNDLE/'data/reference.npz'));s=dict(np.load(ROOT/SOURCES[arm]));t=s.pop('times');ii=np.searchsorted(t,d['times']);assert np.array_equal(t[ii],d['times']);s={k:(v[ii] if v.ndim and len(v)==len(t) else v) for k,v in s.items()}
        z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz'));zi=np.searchsorted(z['times'],d['times']);f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz'));fi=np.searchsorted(f['times'],d['times']);raw=dict(np.load(INPUT))
        cfg=copy.deepcopy(profile['configs']['reference']);cfg['dense_pairs']=[list(x) for x in PAIRS];cfg['conditional_completion']={'risk_aversion':0.};c=controller(d['times'],f['forecast'][fi],f['carry'][fi],z['all_market_signed_core_v1'][zi],z['all_market_signed_core_v1_protected'][zi],price_times=raw['market_times'],prices=raw['close'])
        h=int(arm.rsplit('_',1)[1]);r=replay(d,s,None,cfg,profile['trading'],f'first_dense_individual_{h}',scenario,w['start'],w['end'],base_policy=c);dest.parent.mkdir(exist_ok=True)
        assert not any(e['event'] in ['renewed','hedge_adjustment_planned'] for e in r['events'])
        with gzip.open(dest,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
    row=summarize(r,arm,phase,scenario);assert row['timing_pass'] and row['accounting_error']<1e-6 and row['risk_breach_bars']==0
    row['lifecycle']='fixed_maturity';save(OUT/'rows'/f'{arm}_{phase}_{scenario}.json',row);return row


def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(job,a,p,s) for p in ['validation','comparison','continuous'] for a in SOURCES for s in ['base','stress']]
        for done in as_completed(jobs):
            r=done.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
