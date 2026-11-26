import sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from auto_pair_validation.study import OUT as V2,verify as verify_v2
from auto_pair_selection.selection import IDS
from .study import ROOT,read,save,sha,verify as verify_wave1
from .adaptive_hedge import forecasts
from . import horizon_evaluation as evaluation

OUT=ROOT/'reports/pair_model_research_20261023/wave9'
ARMS=['adaptive_hedge_72','adaptive_hedge_screened_72']


def freeze():
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=verify_v2();verify_wave1()
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),wave=9,arms=ARMS,input=p['input'],input_sha256=p['input_sha256'],
      hypothesis='Fixed1:1 dollar hedges may not describe the local relation. Estimate a symmetric major-axis log-price hedge separately for all45 from past60d; fit AR to that fixed-window residual. Retained venue planner checks the new physical hedge.',
      model='Daily60d with>=80% synchronous active rows; positive covariance; beta=std(logA)/std(logB),supported .1..10 without clipping; local AR positive stationary point estimate and6..336h half life.72h expected relative return divided by1+beta,clip4 source sigmas. Past168h fixed-beta volatility. Screened arm requires diagnostic AR tstat<-3; this is NOT an EG p-value, cointegration proof or multiple-testing correction.',
      action='All45 assessed every day, no named pair limit. Current marks,costs,funding,depth,rounding,physical risk decide executable positive-net choices. Same original lifecycle,6000gross,50k own+50k reserve.',
      selection='August worst Q then PnL, zero risk violations; save choice before later development comparison.',history_reused=True,independent_test=False,orders_sent=0,live_changed=False))
    paths=[Path(__file__),Path(__file__).with_name('adaptive_hedge.py'),Path(__file__).with_name('horizon_evaluation.py'),OUT/'protocol.json']
    save(OUT/'freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in paths})


def verify():
    verify_v2();verify_wave1()
    for path,digest in read(OUT/'freeze.json').items():assert sha(ROOT/path)==digest,path
    return read(OUT/'protocol.json')


def build():
    p=verify();v=read(V2/'protocol.json');d=dict(np.load(ROOT/p['input']));t=d['market_times'];mu,beta,vol,strong,records=forecasts(d,v['evaluation_start'])
    common=dict(times=t,dense=True,pairs=np.array(IDS),signal_times=np.broadcast_to(t[:,None],mu.shape),source_prices=d['close'],beta=beta,vol=vol,forecast_times=t,mu=mu,directions=np.isfinite(mu).astype(int))
    for arm in ARMS:
        available=np.isfinite(mu);entry=available&strong if 'screened' in arm else available
        action=np.where(entry&(abs(mu)>.0038),np.sign(np.nan_to_num(mu)),0).astype(int)
        np.savez_compressed(OUT/f'{arm}.npz',**common,economic_available=available,economic_action=action)
    save(OUT/'model_audit.json',records);verify();print('built',len(records),'day/pair evaluations',flush=True)


def stage_a():
    p=verify();evaluation.stage_a(OUT,p['arms']);verify()


def stage_b():
    verify();evaluation.stage_b(OUT);verify()


if __name__=='__main__':globals()[sys.argv[1]]()
