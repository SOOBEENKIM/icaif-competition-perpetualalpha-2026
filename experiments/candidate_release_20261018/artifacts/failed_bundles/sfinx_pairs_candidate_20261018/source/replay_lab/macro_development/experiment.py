"""Seal training-only choices, then compare every fit on reused periods."""
import copy,gzip,json
import numpy as np
import torch
from .data import ROOT,OUT,PREVIOUS,load
from .model import inputs,targets,train
from .engine import simulate
from ..common_comparison.models import covariance,timestamp
from ..ema_rebuild.experiment import configure
from ..ema_rebuild.models import build
from ..pairs_rl.verify import independent_accounting
from ..pairs_development.experiment import write,digest,stamp
from ..common_comparison.experiment import json_value

def protocol(m):
    _,p=configure();p['symbols']=list(m.symbols)
    p['execution']['decimals']=[4,4,2,3,3,3,2,4,1,2]
    p['funding_assumption']='Signed frozen mainnet reference funding, fully observed; stress adds 1bp/hour gross. Not historical pa funding.'
    return p

def record(r,m,phase):
    audit=independent_accounting(r,m)
    f=OUT/f"{phase}-{r['policy']}-{r['scenario']}.json.gz"
    with gzip.open(f,'wt') as h:json.dump(r,h,allow_nan=False,default=json_value)
    row={k:r[k] for k in ('policy','scenario','metrics','gates','start','end')}
    row.update(phase=phase,file=f.name,audit=audit,sha256=digest(f));write(OUT/(f.name+'.summary.json'),row)
    print(json.dumps(dict(phase=phase,policy=r['policy'],scenario=r['scenario'],roi=r['metrics']['roi'],std=r['metrics']['daily_vol'],fills=r['metrics']['fills'],gates=r['gates']),default=json_value),flush=True)
    return row

def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True,parents=True)
    protected=json.loads((PREVIOUS/'preservation.json').read_text())['files']
    protected.update({str(f.relative_to(ROOT)):digest(f) for f in PREVIOUS.rglob('*') if f.is_file()})
    write(OUT/'preservation.json',dict(files=protected,time=stamp()))
    plan=dict(time=stamp(),change='Replace pair convergence alpha and PPO exploration with multiasset trend and direct net-utility learning.',
        diagnosis={'prompt':'No runtime LLM prompt in SB3 policy.', 'agent':'Keep verified broker/risk/ledger; do not change costs.',
            'strategy':'Mixed-asset selected pairs lack stable cost-positive convergence; 10 complete training trades provide very few independent learning examples.',
            'model':'Warm-start repaired PPO learnability but mostly copied a weak teacher. Test a parsimonious direct portfolio objective.'},
        hypotheses=[
            'H1: If shared trend exists across asset classes, a fixed ten-asset trend policy improves net returns without paired leg failures.',
            'H2: If poor objective alignment matters, seven-parameter net-utility learning beats its fixed equal-coefficient baseline.',
            'H3: If regime concentration matters, SoftMin improves worst-period returns versus identical pooled-only training.',
            'H4: If noise trades matter, mean weights across all fixed seeds improve net results; no best-seed hindsight selection.'],
        references=['https://arxiv.org/html/2601.05975v1','https://arxiv.org/html/2602.11708v1'],
        deviations='7 shared parameters; hourly-derived 1/7/21-day features; four training blocks; no graph/attention, no paper performance transferred.',
        seeds=[11,29,47],steps=400,fit_cutoff='2026-07-29T00:00:00Z',new_independent_holdout=False,
        selection='Highest positive training net ROI subject to daily STD <= EMA training STD, base and stress engineering gates. Later periods never choose model.',
        controls='Original EMA at original 2.5% buffer; EMA at common 20% buffer; fixed signed/long multiscale; pooled-only loss ablation.',
        ensemble='Mean of three predeclared seeds. Fixed 50/50 mix with retained EMA; no tuning of mix weight.',
        all_prices_reused=True,live_orders_sent=0)
    write(OUT/'plan.json',plan)
    m,a,manifest=load();p=protocol(m);write(OUT/'protocol.json',p)
    sigmas=np.full((len(m.times),10,10),np.nan)
    for i in range(168,len(m.times)):sigmas[i]=covariance(m,i,p['risk'])[1]
    x,s=inputs(m);first=1439;last=m.exact(timestamp(plan['fit_cutoff']))
    retained=json.loads((ROOT/'replay_lab/output/retained_strategies/manifest.json').read_text())
    selected=json.loads((ROOT/'replay_lab/ema_development/selected_policy.json').read_text())
    ema=build(a,retained,selected)['ema_base']['target_weight']
    static=targets(m,np.tanh(x.mean(-1)),s)
    models={'ema_original':ema,'ema_buffer20':ema,'static_signed':static,
        'static_long':np.maximum(static,0)};states=[]
    if (OUT/'training_manifest.json').exists():
        prior=json.loads((OUT/'training_manifest.json').read_text())
        assert digest(OUT/'targets.npz')==prior['targets_sha256']
        for name in ('model.py','data.py','engine.py'):
            path=ROOT/'replay_lab/macro_development'/name
            assert digest(path)==prior['sources'][str(path.relative_to(ROOT))]
        models=dict(np.load(OUT/'targets.npz'))
        write(OUT/'resume.json',dict(time=stamp(),reason='Numpy bool JSON serialization; model, input and simulator code unchanged; no additional fitting.',
            targets_sha256=prior['targets_sha256'],prior_route_attempts=2))
    else:
        for seed in plan['seeds']:
            key=f'robust_seed{seed}';models[key],state=train(m,x,s,first,last,seed)
            states.append(state);write(OUT/(key+'.json'),state)
        models['pooled_seed29'],state=train(m,x,s,first,last,29,robust=False);write(OUT/'pooled_seed29.json',state)
        models['robust_mean']=np.mean([models[f'robust_seed{k}'] for k in plan['seeds']],axis=0)
        models['ema_robust_half']=(ema+models['robust_mean'])/2
        models['ema_static_half']=(ema+static)/2
        np.savez_compressed(OUT/'targets.npz',**models)
        write(OUT/'training_manifest.json',dict(time=stamp(),states=states,targets_sha256=digest(OUT/'targets.npz'),
            sources={str(f.relative_to(ROOT)):digest(f) for f in (ROOT/'replay_lab/macro_development').glob('*.py')}))
    rows=[]
    def run(phase,a,b,names):
        for name in names:
            for scenario in ('base','stress'):
                r=simulate(m,dict(target_weight=models[name]),name,p,scenario,int(m.times[a]),int(m.times[b]),
                    buffer=.025 if name=='ema_original' else .2,sigmas=sigmas)
                rows.append(record(r,m,phase))
                write(OUT/'running_results.json',rows)
    run('train',first,last,models)
    baseline=next(r for r in rows if r['policy']=='ema_original' and r['scenario']=='base')
    eligible=[]
    for r in rows:
        if r['scenario']!='base' or r['policy'].startswith('ema_original'):continue
        stress=next(z for z in rows if z['policy']==r['policy'] and z['scenario']=='stress')
        if r['metrics']['roi']>0 and r['metrics']['daily_vol']<=baseline['metrics']['daily_vol'] and all(r['gates'].values()) and all(stress['gates'].values()):eligible.append(r)
    eligible.sort(key=lambda r:(-r['metrics']['roi'],r['metrics']['daily_vol'],r['policy']))
    selection=dict(time=stamp(),choice=eligible[0]['policy'] if eligible else None,
        eligible=[r['policy'] for r in eligible],rule=plan['selection'],later_outcomes_opened=False)
    write(OUT/'selection.json',selection)
    periods={'select':(last,m.exact(timestamp('2026-08-26T00:00:00Z'))),
        'audit_reused':(m.exact(timestamp('2026-08-26T00:00:00Z')),m.exact(timestamp('2026-10-05T16:00:00Z'))),
        'extension_reused':(m.exact(timestamp('2026-10-05T16:00:00Z')),len(m.times)-1)}
    for phase,(aa,bb) in periods.items():run(phase,aa,bb,models)
    changed=[f for f,sha in protected.items() if digest(ROOT/f)!=sha];assert not changed
    write(OUT/'results.json',dict(time=stamp(),selection=selection,routes=rows,protected_files=len(protected),changed_protected=changed,
        new_independent_holdout=False,live_orders_sent=0))

if __name__=='__main__':main()
