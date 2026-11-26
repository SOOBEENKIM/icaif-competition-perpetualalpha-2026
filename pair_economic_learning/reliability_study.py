"""Keep the original hedge while testing signal reliability and turning evidence."""
import sys,json
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from pair_value_learning.study import ROOT,INPUT,read,save,sha
from .study import OUT as PARENT,verify as verify_parent
from .reference import run as replay
OUT=ROOT/'reports/pair_economic_learning_20261023/wave4'
SOURCE=ROOT/'reports/pair_value_learning_20261023/scaled_coherence/coherent_scaled_72.npz'
ARMS=['reliable_alpha_72','confirmed6_alpha_72','confirmed24_alpha_72']


def freeze():
    verify_parent()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),input=str(INPUT.relative_to(ROOT)),input_sha256=sha(INPUT),arms=ARMS,primary='reliable_alpha_72',
      hypothesis='Weakly comoving assets get large SMA/AR reversion forecasts. Prior repairs changed hedge and forecast together. Preserve sourcehedge and onlyattenuate alpha by positive168hreturncorrelation squared. Separate turning-point gates test whether waiting for6h/24hrelative momentum to agree withforecast prevents catching persistent divergence.',
      formulas='reliable:mu=coherent_AR*max(corr168h,0)^2; this is a sharedvariance heuristic,not probability or provedforecastcalibration. confirmed6/24 retain mu butrequire sourcephysicalrelativepastreturn*mu>0. No asset labels or handpickedpair allowlist.',
      controls='Same45assessments,sourcehedge,50k+50kreserve,6000paircap,originalbase/renewal/costs/execution. Eacharm changes onlyalphaorentrypermission. No thresholdgrid.',
      evaluation='All6reference andall6PASconditions,reuseddevelopment,reportfailures. No PAS outcome fitting or livecodechanges.',orders_sent=0,live_changed=False,independent_test=False,history_reused=True))
    paths=[Path(__file__),SOURCE,PARENT/'features.npz',OUT/'protocol.json',Path(__file__).with_name('reference.py')]
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})


def verify():
    verify_parent()
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p


def build():
    verify();f=dict(np.load(PARENT/'features.npz'));source=dict(np.load(SOURCE));ix=np.searchsorted(f['times'],source['times']);assert np.array_equal(f['times'][ix],source['times'])
    for arm in ARMS:
        s={k:v.copy() for k,v in source.items()};mu=s['mu'].copy();valid=s['economic_available'].copy()
        if arm=='reliable_alpha_72':mu*=np.maximum(f['x'][ix,:,8],0.)**2
        else:valid &= mu*f['x'][ix,:,2 if '6_' in arm else 3]>0
        s['mu']=mu;s['economic_action']=np.where(valid&(abs(mu)>.0038),np.sign(np.nan_to_num(mu)),0).astype(int)
        np.savez_compressed(OUT/f'{arm}.npz',**s)


def job(arm,phase,scenario):
    return replay(OUT,OUT/f'{arm}.npz',arm,phase,scenario)


def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,p,s) for a in ARMS for p in ['validation','comparison','continuous'] for s in ['base','stress']]):
            r=f.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs','risk_breach_bars']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
