import copy,gzip,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from ..common_comparison.models import Market,covariance
from ..common_comparison.experiment import json_value
from ..macro_development.engine import simulate
from ..pairs_rl.verify import independent_accounting

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'replay_lab/output/competition_rebuild/20261013-v9'
GROUPS=((0,1,2),(3,4,5),(6,),(7,),(8,9))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2,allow_nan=False,default=json_value)+'\n');tmp.replace(p)

def load():
    plan=json.loads((OUT/'plan.json').read_text())
    for f,h in plan['inputs_sha256'].items():assert sha(OUT/f)==h
    z=dict(np.load(OUT/'frozen_inputs.npz'));b=json.loads((OUT/'candidate.json').read_text())
    m=Market(z['market_times'],z['close'],z['volume'],z['trades'],z['funding'],z['observed'],tuple(b['control_bundle']['symbols']))
    return m,z,b,copy.deepcopy(b['control_bundle']['protocol'])

def covariance_path(m,p):
    result=np.full((len(m.times),10,10),np.nan)
    for i in range(168,len(m.times)):result[i]=covariance(m,i,p['risk'])[1]
    return result

def run(m,p,weights,name,start,end,sigmas,scenario='base',folder=None):
    result=simulate(m,dict(target_weight=weights),name,p,scenario,int(m.times[start]),int(m.times[end]),buffer=.2,sigmas=sigmas)
    audit=independent_accounting(result,m)
    assert audit['passed'] and result['gates']['accounting'] and result['gates']['information_time']
    assert np.isfinite(result['metrics']['roi'])
    if result['metrics']['complete_days']<2:result['metrics']['daily_vol']=None
    if folder is not None:
        path=folder/f'{name}-{scenario}.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(path,'wt') as h:json.dump(result,h,default=json_value,allow_nan=False)
        row={k:result[k] for k in ('policy','start','end','scenario','metrics','gates')}
        row.update(audit=audit,file=str(path.relative_to(OUT)),sha256=sha(path));write(path.with_suffix('.summary.json'),row)
    return result
