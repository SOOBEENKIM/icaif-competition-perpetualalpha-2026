import gzip,json,time
from pathlib import Path
import numpy as np
from experiments.shared_execution_pairs_20261017.portfolio_v2 import replay
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha,ts
from experiments.complementary_alpha_20261017.research import protected

HERE=Path(__file__).resolve().parent;PREVIOUS=HERE.parent/'shared_execution_pairs_20261017'


def load(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def metric(run,name):
    if name=='mean_gross':return run['metrics']['mean_gross']
    return float(np.mean([s['daily_volatility'] for s in run['snapshots']]))


def main():
    out=HERE/'scale_artifacts';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    cfg=json.loads((PREVIOUS/'reference_protocol.json').read_text());spec=json.loads((HERE/'scale_protocol.json').read_text())
    d,de_unused,p=inputs(cfg)
    with np.load(ROOT/'experiments/persistent_intent_pairs_20261017/artifacts/reference_immediate_state.npz') as z:de={k:z[k].copy() for k in z.files}
    paths=[HERE/'scale_audit.py',HERE/'scale_protocol.json',PREVIOUS/'reference_protocol.json',PREVIOUS/'portfolio_v2.py',PREVIOUS/'account_v2.py',
        ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',ROOT/'experiments/risk_preserving_20261016/artifacts/targets.npz',
        ROOT/'experiments/persistent_intent_pairs_20261017/artifacts/reference_immediate_state.npz']
    for hybrid in spec['baseline_to_hybrid'].values():paths.append(PREVIOUS/f'artifacts_v2/runs/reference-validation-{hybrid}-base.json.gz')
    hashes={str(q.relative_to(ROOT)):sha(q) for q in paths}
    save(out/'frozen.json',dict(at_unix=time.time(),sources=hashes,protected=protected(),orders_sent=0,historical_independence=False))
    runs={};rows=[];fit=[];count=0
    def run(base,scale,phase,scenario,label):
        nonlocal count
        dd={**d};key='first' if base=='first_strategy' else 'second';dd[key]=d[key]*scale
        start=cfg['validation_start'] if phase=='validation' else cfg['comparison_start']
        end=cfg['comparison_start'] if phase=='validation' else cfg['end']
        r=replay(dd,de,None,cfg,p,base,scenario,ts(start),ts(end));count+=1
        with gzip.open(out/'runs'/f'{label}.json.gz','wt') as f:json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        return r
    for baseline,hybrid in spec['baseline_to_hybrid'].items():
        reference=load(PREVIOUS/f'artifacts_v2/runs/reference-validation-{hybrid}-base.json.gz')
        for name in spec['metrics']:
            target=metric(reference,name);cache={};trace=[]
            def evaluate(scale):
                if scale not in cache:
                    label=f'fit-{baseline}-{name}-{len(cache):02d}'
                    r=run(baseline,scale,'validation','base',label);value=metric(r,name)
                    cache[scale]=(value,label,r);trace.append(dict(scale=scale,matching_value=value,run=label))
                return cache[scale][0]
            lo,hi=spec['scale_bounds'];low=evaluate(lo);high=evaluate(hi)
            if low<=target<=high:
                for _ in range(spec['bisection_steps']):
                    mid=(lo+hi)/2
                    if evaluate(mid)<target:lo=mid
                    else:hi=mid
            scale=min(cache,key=lambda k:(abs(cache[k][0]-target),k))
            value,label,r=cache[scale];fit.append(dict(baseline=baseline,hybrid=hybrid,metric=name,scale=scale,
                target=target,achieved=value,relative_error=value/target-1,bracketed=low<=target<=high,trace=trace))
            arm=baseline+'_'+name+'_matched'
            rows.append(dict(venue='reference',phase='validation',arm=arm,scenario='base',scale=scale,metrics=r['metrics']))
            print('CALIBRATED',baseline,name,'scale',round(scale,6),'relative error',round(value/target-1,6),flush=True)
    save(out/'calibration.json',dict(fit=fit,fit_end=cfg['comparison_start'],fit_scenario='base',frozen_before_later_replay=True));cal_hash=sha(out/'calibration.json')
    for f in fit:
        arm=f['baseline']+'_'+f['metric']+'_matched'
        for phase,scenario in [('validation','stress'),('comparison','base'),('comparison','stress')]:
            r=run(f['baseline'],f['scale'],phase,scenario,f'{phase}-{arm}-{scenario}')
            rows.append(dict(venue='reference',phase=phase,arm=arm,scenario=scenario,scale=f['scale'],metrics=r['metrics']))
            print(phase,arm,scenario,'PnL',round(r['metrics']['net_pnl'],3),'Q',round(r['metrics']['financial_score'],7),flush=True)
    assert sha(out/'calibration.json')==cal_hash
    for path,digest in hashes.items():assert sha(ROOT/path)==digest,path
    save(out/'results.json',dict(rows=rows,fit=fit,actual_replays=count,protected=protected(),orders_sent=0,historical_independence=False,
        interpretation='Scales match the August base metric only, not every later regime. This isolates simple fixed scaling as a control; it does not prove causal alpha or future superiority.'))
    print('DONE size-control audit',count,'replays',flush=True)


if __name__=='__main__':main()
