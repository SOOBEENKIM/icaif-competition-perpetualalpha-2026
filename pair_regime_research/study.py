"""Frozen causal repair ablations; no automatic live deployment or outcome search."""
import copy,gzip,json,hashlib,sys,difflib
from dataclasses import asdict
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .model import Spec,ARMS,build as build_models
from auto_pair_selection.selection import PAIRS
from auto_pair_selection.study import verify as verify_live
from competition_live.profile import BUNDLE,load_profile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/pair_regime_research_20261023'
INPUT=ROOT/'practice_live/updates/20261023T080007449138Z/inputs.npz'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else x))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze():
    verify_live()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),input=str(INPUT.relative_to(ROOT)),input_sha256=sha(INPUT),spec=asdict(Spec()),arms=list(ARMS),primary='stable_regime_72',
        hypothesis=['SMA hedge ignores relationship strength; useOLS hedge with stable positive correlation and coefficient across3 disjoint20day blocks.','Residual rho<1 alone is insufficient: Engle-Granger6lag evidence, Holm45 correction .05, block equilibrium shift<=2 residual sigmas. Gap-free history required.','6h,24h,72h spread movement away from equilibrium exceeding fitted daily correction vetoes countertrend entry. A fresh6h turn releases this veto only; formation and economic gates still required.'],
        lifecycle='Blocked entry also blocks maturity extension/rehedging. Two consecutive distinct hourly invalid sources request paid close through original partial-fill engine. No same-source double count; risk and completion protections preserved.',
        fixed='50k sleeve+50k inactive reserve,6000 pair gross,one active pair,all45 eligible for evaluation,existing directional base,fees/funding/execution/risk and six reference windows unchanged.',
        comparisons='OLS-only,relationship-only,legacySMA+trend-only,and fixed primarycombined. No winner substitution after results. Compareagainst both incumbent_fixed_two and adaptive_hedge_72.',
        evaluation_start=1785542400,history_reused=True,independent_test=False,
        evaluation='All4 arms in all6 historical development conditions. Separate matured72h PAS price-markout diagnostic; not portfolioPnL. PASlabels notused to fit parameters. Alreadyreviewed PASperiod is development data.',
        limitations='Coint assumesI(1); per-day Holm doesnotcontrol all repeated daily tests. Cointegration/heuristic trend guard doesnotguarantee future convergence. Thresholds frozen before these results, but researcher has seen reusedmarketoutcomes.',
        orders_sent=0,live_changed=False))
    paths=[*Path(__file__).parent.glob('*.py'),OUT/'protocol.json',ROOT/'pair_regime_research/requirements.txt']
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    original=(ROOT/'pas_economic_allocation/replay.py').read_text().splitlines(True);new=(ROOT/'pair_regime_research/replay.py').read_text().splitlines(True)
    (OUT/'engine_diff.patch').write_text(''.join(difflib.unified_diff(original,new,fromfile='frozen_replay',tofile='research_regime_replay')))
    print('FROZEN four ablations; primary stable_regime_72',flush=True)
def verify():
    verify_live();p=read(OUT/'protocol.json');assert sha(ROOT/p['input'])==p['input_sha256']
    for path,digest in read(OUT/'freeze.json').items():assert sha(ROOT/path)==digest,path
    return p
def build():
    p=verify();d=dict(np.load(ROOT/p['input']));states,audits,counts=build_models(d,p['evaluation_start'],Spec(**p['spec']))
    for arm,state in states.items():np.savez_compressed(OUT/f'{arm}.npz',**state)
    save(OUT/'formation_audit.json',audits);save(OUT/'gate_counts.json',counts);verify();print(json.dumps(counts),flush=True)
def job(arm,phase,scenario):
    from .replay import replay
    from pas_performance_research.selected import controller
    from pas_allocation_study.study import summarize
    p=read(OUT/'protocol.json');profile=load_profile();w=next(x for x in read(BUNDLE/'expected.json')['runs'] if x['venue']=='reference' and x['arm']==profile['candidate'] and x['phase']==phase and x['scenario']==scenario);dest=OUT/'reference'/f'{arm}_{phase}_{scenario}.json.gz'
    if dest.exists():
        with gzip.open(dest,'rt') as f:r=json.load(f)
    else:
        d=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False));s=dict(np.load(OUT/f'{arm}.npz'));t=s.pop('times');ii=np.searchsorted(t,d['times']);assert np.array_equal(t[ii],d['times']);s={k:(v[ii] if v.ndim and len(v)==len(t) else v) for k,v in s.items()}
        z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz'));zi=np.searchsorted(z['times'],d['times']);f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz'));fi=np.searchsorted(f['times'],d['times']);raw=dict(np.load(ROOT/p['input']))
        cfg=copy.deepcopy(profile['configs']['reference']);cfg['dense_pairs']=[list(x) for x in PAIRS];cfg['conditional_completion']={'risk_aversion':0.}
        c=controller(d['times'],f['forecast'][fi],f['carry'][fi],z['all_market_signed_core_v1'][zi],z['all_market_signed_core_v1_protected'][zi],price_times=raw['market_times'],prices=raw['close'])
        r=replay(d,s,None,cfg,profile['trading'],'first_dense_hourlyhedge_individual_72',scenario,w['start'],w['end'],base_policy=c);dest.parent.mkdir(exist_ok=True)
        with gzip.open(dest,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
    row=summarize(r,arm,phase,scenario);row['relationship_exits']=sum(e['event']=='exit_requested' and e.get('reason') in ['relationship_invalidated','persistent_divergence'] for e in r['events']);row['entered_pairs']=sorted({tuple(e['pair']) for e in r['events'] if e['event']=='pair_intent'});assert row['timing_pass'] and row['accounting_error']<1e-6
    row['eligible']=row['risk_breach_bars']==0;save(OUT/'rows'/f'{arm}_{phase}_{scenario}.json',row);return row

def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(job,a,p,s) for p in ['validation','comparison','continuous'] for a in ARMS for s in ['base','stress']]
        for done in as_completed(jobs):
            r=done.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs','risk_breach_bars','relationship_exits']}),flush=True)
    verify()
if __name__=='__main__':globals()[sys.argv[1]]()
