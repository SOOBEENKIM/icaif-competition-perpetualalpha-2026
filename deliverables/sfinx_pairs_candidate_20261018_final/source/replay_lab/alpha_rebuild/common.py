import gzip,json,hashlib
from pathlib import Path
import numpy as np
from ..competition_rebuild.common import ROOT,sha,stamp,write
from ..common_comparison.models import Market
from ..common_comparison.experiment import json_value
from ..macro_development.engine import simulate
from ..pairs_rl.verify import independent_accounting

OUT=ROOT/'replay_lab/output/alpha_rebuild/20261014-v1'
PREV=ROOT/'replay_lab/output/v10_revision/20261013-v1'

def market(z,p):return Market(z['market_times'],z['close'],z['volume'],z['trades'],z['funding'],z['observed'],tuple(p['symbols']))

def replay(m,p,sigma,w,name,phase,start,end,scenario='base',save=True):
    run=simulate(m,dict(target_weight=w),name,p,scenario,int(start),int(end),buffer=.2,sigmas=sigma)
    audit=independent_accounting(run,m)
    assert audit['passed'] and run['gates']['information_time']
    row={k:run[k] for k in ('policy','scenario','metrics','gates','start','end')};row.update(phase=phase,audit=audit)
    if save:
        path=OUT/'runs'/phase/f'{name}-{scenario}.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError(path)
        with gzip.open(path,'wt') as f:json.dump(run,f,default=json_value,allow_nan=False)
        row['file']=str(path.relative_to(OUT))
    print(phase,name,scenario,round(row['metrics']['roi']*100,4),round(row['metrics']['daily_vol']*100,4),flush=True)
    return row,run

def limits(w):
    w=np.clip(w,-.135,.135)
    w/=np.maximum(1.,np.maximum(abs(w).sum(1)/.54,abs(w.sum(1))/.36))[:,None]
    return w

def source_hashes():
    # Freeze our study plus all actual project modules, excluding copied
    # dependencies and output archives to keep this manifest reviewable.
    paths=list((ROOT/'replay_lab/alpha_rebuild').glob('*.py'))
    for folder in ('v10_revision','portfolio_rebuild_v10','competition_rebuild','common_comparison','ema_rebuild','ema_development','macro_development','portfolio_development','pairs_rl','pullback'):
        paths+=list((ROOT/'replay_lab'/folder).glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}

def check_freeze():
    s=json.loads((OUT/'selected.json').read_text())
    for f,h in s['sources'].items():
        if sha(ROOT/f)!=h:raise ValueError('Frozen code changed '+f)
    for f,h in s['artifacts'].items():
        if sha(OUT/f)!=h:raise ValueError('Frozen artifact changed '+f)
    return s

