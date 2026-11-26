import sys,copy
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from auto_pair_validation.study import OUT as V2,verify as verify_v2
from auto_pair_selection.selection import IDS
from auto_pair_selection.signals import prepare_features
from experiments.dense_history_pairs_20261017.model import labels
from .study import ROOT,read,save,sha,verify as verify_wave1
from .relative import forecasts as ar_forecasts
from .long_horizon import fit_day,predict,consensus
from . import horizon_evaluation as evaluation

OUT=ROOT/'reports/pair_model_research_20261023/wave7'
ARMS=[f'{kind}_{h}' for h in [72,168] for kind in ['individual','consensus']]


def freeze():
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=verify_v2();verify_wave1()
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),wave=7,arms=ARMS,input=p['input'],input_sha256=p['input_sha256'],spec=p['direct_spec'],
      hypothesis='AR alone shows continuous-base gains but unstable cost/path performance. Learn each pair over72/168h instead of pooling economically heterogeneous pairs, then use horizon-aligned AR/learned ensemble agreement. All45 per-pair models, no names or backtest-winner identities as features.',
      training='Daily90d rolling individual ridge,existing20 past-only features,6h sampled labels. Targets are realized equal-dollar fixed-source relative cash returns, with source equal-dollar168h volatility. Source+h strictly before fit; training samples overlap and are not independent. Retained ridge/intercept priors. This differs from wave1 pooled dynamic-beta targets.',
      consensus='Arithmetic average of individual ridge and AR at the SAME horizon and target unit; entry requires both available and same sign and mean>43bp magnitude. Mean remains for holdings if entry disagreement. AR past720h fit,6..336h estimated half life. No claim of cointegration.',
      fixed='Same6000gross,50k own+50k reserve,one active pair,base/risk/fees/partial fills; original lifecycle (wave6 early exit not used).',
      selection='August worst Q advantage then PnL, zero risk violations, freeze choice before later comparison.',history_reused=True,independent_test=False,orders_sent=0,live_changed=False))
    paths=[Path(__file__),*[Path(__file__).with_name(x) for x in ['long_horizon.py','horizon_evaluation.py','relative.py']],OUT/'protocol.json']
    save(OUT/'freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in paths})


def verify():
    verify_v2();verify_wave1()
    for path,digest in read(OUT/'freeze.json').items():assert sha(ROOT/path)==digest,path
    return read(OUT/'protocol.json')


def build():
    p=verify();v=read(V2/'protocol.json');d=dict(np.load(ROOT/p['input']));t=d['market_times'];old,_,_,_=prepare_features(d)
    for h in [72,168]:
        ar,rel,_,_=ar_forecasts(d,h,'reversion');f=dict(old,beta=np.ones_like(old['beta']),vol=rel['vol'],fresh=old['fresh']&rel['valid'])
        y,maturity=labels(t,d['close'],f,h);terminal=np.zeros_like(f['fresh']);terminal[:-h]=f['fresh'][h:]
        mu=np.full_like(y,np.nan);unit=f['vol']*np.sqrt(h);spec=copy.deepcopy(p['spec']);spec['horizon']=h;records=[]
        for day in np.unique(t[t>=v['evaluation_start']]//86400*86400):
            models=fit_day(t,f['x'],y,maturity,f['fresh'],terminal,int(day),spec);ix=np.flatnonzero((t>=day)&(t<day+86400))
            mu[ix]=predict(f['x'][ix],unit[ix],f['fresh'][ix],models,spec)
            save(OUT/'models'/f'{h}_{day}.json',models);records.append(dict(source=int(day),models=len(models),last_maturity=max((m['last_maturity'] for m in models.values()),default=None)))
        blend,agreement=consensus(mu,ar)
        for kind,value,entry in [('individual',mu,np.isfinite(mu)),('consensus',blend,agreement)]:
            available=np.isfinite(value)&f['fresh'];action=np.where(available&entry&(abs(value)>.0038),np.sign(np.nan_to_num(value)),0).astype(int)
            np.savez_compressed(OUT/f'{kind}_{h}.npz',times=t,dense=True,pairs=np.array(IDS),signal_times=np.broadcast_to(t[:,None],value.shape),source_prices=d['close'],beta=f['beta'],vol=f['vol'],forecast_times=t,mu=value,directions=available.astype(int),economic_available=available,economic_action=action)
        save(OUT/f'audit_{h}.json',records);print('built horizon',h,flush=True)
    verify()


def stage_a():
    p=verify();evaluation.stage_a(OUT,p['arms']);verify()


def stage_b():
    verify();evaluation.stage_b(OUT);verify()


if __name__=='__main__':globals()[sys.argv[1]]()
