import copy,gzip,json,sys
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from pair_value_learning.study import ROOT,INPUT,read,save,sha
from pair_value_learning.scaled_study import verify as verify_parent
from competition_live.profile import load_profile,BUNDLE
from auto_pair_selection.selection import PAIRS
from .portfolio import replay
OUT=ROOT/'reports/pair_economic_learning_20261023/wave6'
SOURCE=ROOT/'reports/pair_value_learning_20261023/scaled_coherence/coherent_scaled_72.npz'
ARMS=['disjoint_asset_portfolio_72','disjoint_pair_portfolio_72']


def freeze():
    verify_parent()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),input=str(INPUT.relative_to(ROOT)),input_sha256=sha(INPUT),arms=ARMS,primary='disjoint_asset_portfolio_72',
      hypothesis='Onefalsepositivepair occupies the entire pairmodule fordays. Test3causallycoordinated virtualsleeves withthe SAMEtotal paircapitalcap6000andSAMEtotal participation. Exclude activeotherassets(primary)oronlysamepair(ablation)atcurrentbar; all45remain evaluated.',
      capital='50kown split3;50kreserveunchanged. Eachpaircap2000,assetnotionalcap2500,participation1/3;otherfractionalrisklimitsunchanged. Combinedphysicalrisksandaggregatevolumecheckedallbars.',
      accounting='Eachvirtualsleevepaysfullfees/impact/funding. No savingsassumedfrominternalnetting;positivefundinghaircutappliedbeforeaggregationisconservative. Predetermined priority0,1,2; onlycurrentknownoccupiedpairs passedtocoordinator.',
      controls='OriginalcoherentARsourcehedges/forecasts/costgate/72hrenewablepaidlifecycle, no directionalbase. One-sleeveparitymustexactlyreproducepriorfullengine. Noaddedmodel/price/feeassumption.',
      evaluation='All6referenceconditions thenPASsame84h. Reuseddevelopment; fullportfolio redesign notproof of betterforecast accuracy. No liveactivation.',orders_sent=0,live_changed=False,independent_test=False,history_reused=True))
    paths=[Path(__file__),Path(__file__).with_name('portfolio.py'),Path(__file__).with_name('cohort_engine.py'),SOURCE,OUT/'protocol.json']
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})


def verify():
    verify_parent()
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p


def inputs(phase,scenario):
    p=load_profile();w=next(x for x in read(BUNDLE/'expected.json')['runs'] if x['venue']=='reference' and x['arm']==p['candidate'] and x['phase']==phase and x['scenario']==scenario)
    d=dict(np.load(BUNDLE/'data/reference.npz'));s=dict(np.load(SOURCE));t=s.pop('times');ii=np.searchsorted(t,d['times']);assert np.array_equal(t[ii],d['times']);s={k:(v[ii] if v.ndim and len(v)==len(t) else v) for k,v in s.items()}
    cfg=copy.deepcopy(p['configs']['reference']);cfg['dense_pairs']=[list(x) for x in PAIRS];cfg['conditional_completion']={'risk_aversion':0.}
    return d,s,cfg,p,w


def parity():
    verify();d,s,c,p,w=inputs('continuous','base')
    r=replay(d,s,c,p['trading'],'base',w['start'],w['end'],count=1)
    path=ROOT/'reports/pair_economic_learning_20261023/wave3/reference/unbounded_pairs_72_continuous_base.json.gz'
    expected=json.loads(gzip.open(path,'rt').read());actual=json.loads(json.dumps(r,default=lambda x:x.tolist()))
    checks={k:actual[k]==v for k,v in expected.items()}
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    save(OUT/'parity.json',dict(exact_full_path_match=True,checks=checks));print('PASS one-sleeve entire-path parity',flush=True)


def job(arm,phase,scenario):
    from pas_allocation_study.study import summarize
    d,s,c,p,w=inputs(phase,scenario);r=replay(d,s,c,p['trading'],scenario,w['start'],w['end'],disjoint_assets=arm.startswith('disjoint_asset'))
    row=summarize(r,arm,phase,scenario);assert row['risk_breach_bars']==0
    path=OUT/'reference'/f'{arm}_{phase}_{scenario}.json.gz';path.parent.mkdir(exist_ok=True)
    with gzip.open(path,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
    save(OUT/'rows'/f'{arm}_{phase}_{scenario}.json',row);return row


def run():
    verify();assert read(OUT/'parity.json')['exact_full_path_match']
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,p,s) for a in ARMS for p in ['validation','comparison','continuous'] for s in ['base','stress']]):
            r=f.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs','risk_breach_bars']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
