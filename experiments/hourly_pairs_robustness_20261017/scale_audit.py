"""Actual native strategy scale controls fitted only to August volatility."""
import gzip
import json
import time
from pathlib import Path
import numpy as np
from experiments.par_direct_forecast_20261017.data import ROOT, save, sha, ts
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.complementary_alpha_20261017.research import protected
from experiments.hourly_refresh_pairs_20261017.policy import replay
from experiments.hourly_refresh_pairs_20261017.research import control_directory
from .analysis import load

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent/'hourly_refresh_pairs_20261017'


def main():
    out=HERE/'scale_artifacts';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    cfg=json.loads((PRIOR/'reference_protocol.json').read_text());spec=json.loads((HERE/'protocol.json').read_text())
    previous=json.loads((PRIOR/'artifacts/frozen.json').read_text())['sources']
    for rel,digest in previous.items(): assert sha(ROOT/rel)==digest,rel
    paths=[HERE/'scale_audit.py',HERE/'analysis.py',HERE/'protocol.json',HERE/'test_results.txt',
           PRIOR/'artifacts/selection.json',PRIOR/'artifacts/results.json',PRIOR/'artifacts/frozen.json']
    sources={**previous,**{str(p.relative_to(ROOT)):sha(p) for p in paths}}
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),orders_sent=0,historical_independence=False))
    d,_,p=inputs(cfg)
    with np.load(PRIOR/'prepared/reference_bias_24_state.npz') as z:de={k:z[k].copy() for k in z.files}
    target=load(PRIOR/'artifacts/runs'/f"reference-validation-{spec['candidate']}-base.json.gz")['metrics'][spec['scale_fit']['metric']]
    count=0;rows=[];fit=[]
    def simulate(base,scale,phase,scenario,label):
        nonlocal count
        dd={**d};key='first' if base=='first_strategy' else 'second';dd[key]=d[key]*scale
        start=cfg['validation_start'] if phase=='validation' else cfg['comparison_start']
        end=cfg['comparison_start'] if phase=='validation' else cfg['end']
        r=replay(dd,de,None,cfg,p,base,scenario,ts(start),ts(end));count+=1
        with gzip.open(out/'runs'/f'{label}.json.gz','wt') as f:
            json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        print('run',label,'scale',scale,'PnL',round(r['metrics']['net_pnl'],3),
              'sigma',round(r['metrics']['volatility_horizon'],7),flush=True)
        return r
    for base in spec['baselines']:
        cache={};trace=[]
        def evaluate(scale):
            if scale not in cache:
                label=f'fit-{base}-{len(cache):02d}'
                run=simulate(base,scale,'validation','base',label)
                if scale==1:
                    old=load(control_directory(base)/'runs'/f'reference-validation-{base}-base.json.gz')
                    for k in ['fills','snapshots','events','metrics']:
                        assert run[k]==old[k],(base,k)
                value=run['metrics'][spec['scale_fit']['metric']]
                cache[scale]=(value,label,run);trace.append(dict(scale=scale,value=value,path=label))
            return cache[scale][0]
        for scale in spec['scale_fit']['grid']:evaluate(scale)
        for _ in range(spec['scale_fit']['refinement_rounds']):
            best=min(cache,key=lambda x:(abs(cache[x][0]-target),x));keys=sorted(cache);i=keys.index(best)
            additions=[]
            if i>0:additions.append((keys[i-1]+best)/2)
            if i<len(keys)-1:additions.append((best+keys[i+1])/2)
            for scale in additions:evaluate(scale)
        best=min(cache,key=lambda x:(abs(cache[x][0]-target),x));value,label,run=cache[best]
        fit.append(dict(baseline=base,scale=best,target=target,achieved=value,relative_error=value/target-1,trace=trace))
        rows.append(dict(venue='reference',phase='validation',arm=base+'_august_vol_matched',scenario='base',scale=best,metrics=run['metrics']))
    save(out/'calibration.json',dict(fit=fit,frozen_before_stress_and_later_replay=True,
         not_unseen_historical_data=True,fit_end=cfg['comparison_start'],fit_criterion='volatility error only'))
    calibration_hash=sha(out/'calibration.json')
    for f in fit:
        base=f['baseline'];arm=base+'_august_vol_matched'
        for phase,sc in [('validation','stress'),('comparison','base'),('comparison','stress')]:
            run=simulate(base,f['scale'],phase,sc,f'{phase}-{arm}-{sc}')
            rows.append(dict(venue='reference',phase=phase,arm=arm,scenario=sc,scale=f['scale'],metrics=run['metrics']))
    assert sha(out/'calibration.json')==calibration_hash
    for rel,digest in sources.items():assert sha(ROOT/rel)==digest,rel
    save(out/'results.json',dict(rows=rows,fit=fit,actual_account_replays=count,
         unit_scale_controls_exact=2,protected=protected(),orders_sent=0,historical_independence=False,
         diagnostic_controls_only=True))
    print('DONE',count,'actual account replays',flush=True)


if __name__=='__main__':main()
