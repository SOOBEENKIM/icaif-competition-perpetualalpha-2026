"""Freeze the stability rule, reproduce controls, compare full account paths."""
import copy
import gzip
import json
import time
from pathlib import Path
import numpy as np
from experiments.lifecycle_value_pairs_20261018.research_v2 import ROOT,inputs,load,save,sha,verify,protected,readrun
from .model import bundle
from .portfolio import replay

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'
PARENT=HERE.parent/'lifecycle_value_pairs_20261018/artifacts'
GATE=HERE.parent/'decision_gate_pairs_20261017/artifacts'


def plain(value):
    return json.loads(json.dumps(value,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))


def main():
    spec=json.loads((HERE/'protocol.json').read_text())
    learning=json.loads((PARENT.parent/'protocol.json').read_text())
    test=(HERE/'test_results.txt').read_text();assert 'Ran 8 tests' in test and test.rstrip().endswith('OK')
    OUT.mkdir(exist_ok=False);(OUT/'runs').mkdir()
    parent=json.loads((PARENT/'results.json').read_text());windows=parent['windows'];rows=parent['rows'].copy()
    configs={v:json.loads((ROOT/f'experiments/hourly_refresh_pairs_20261017/{v}_protocol.json').read_text()) for v in ['reference','par']}
    configs['par_extended']=copy.deepcopy(configs['par'])
    d,_,p=inputs(configs['reference'])
    data={'reference':d,'par':load(ROOT/'experiments/par_direct_forecast_20261017/artifacts/inputs.npz'),
          'par_extended':load(ROOT/'experiments/historical_par_extension_20261017/artifacts_v3/inputs.npz')}
    states={v:load(PARENT/f'{v}_state.npz') for v in data}
    labels=json.loads((PARENT/'labels.json').read_text())['rows']
    full_models={int(k):v for k,v in json.loads((PARENT/'models.json').read_text()).items()}
    control_paths={(v,phase,arm,sc):(GATE if arm==spec['control'] else PARENT)/'runs'/f'{v}-{phase}-{arm}-{sc}.json.gz'
                   for v,phase in windows for arm in [spec['control'],spec['parent']] for sc in ['base','stress']}
    files=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',PARENT.parent/'protocol.json',PARENT/'labels.json',
           PARENT/'models.json',PARENT/'results.json',PARENT/'conclusion_frozen.json',
           *[PARENT/f'{v}_state.npz' for v in data],*control_paths.values()]
    sources=json.loads((PARENT/'evaluation_frozen_v2.json').read_text())['sources'].copy()
    sources.update({str(path.relative_to(ROOT)):sha(path) for path in files});verify(sources)
    verify(json.loads((PARENT/'conclusion_frozen.json').read_text())['sources'])
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(prospective)
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),previous_goal_turn=spec['previous_goal_turn']))
    models={};refits=0
    for count,(stamp,model) in enumerate(full_models.items()):
        models[stamp]=bundle(labels,model,learning,spec)
        refits+=sum(b['model'] is not None for b in models[stamp]['blocks'])
        if count%5==0:print('FIT',count+1,'/',len(full_models),'deleted-block refits',refits,flush=True)
    save(OUT/'models.json',models)
    complete=sum(m['complete'] for m in models.values())
    print('BUNDLES',len(models),'complete',complete,'refits',refits,flush=True)
    def configuration(venue,arm):
        cfg=copy.deepcopy(configs[venue]);cfg.update(dense_pairs=learning['pairs'],conditional_completion={'risk_aversion':0.})
        cfg.update(lifecycle_kind='filter',lifecycle_models=full_models if arm==spec['parent'] else {})
        if arm==spec['candidate']:cfg.update(stability_models=models,stability_checks=[])
        return cfg
    controls={};verified=[]
    for (v,phase,arm,sc),path in control_paths.items():
        old=readrun(path);fresh=plain(replay(data[v],states[v],None,configuration(v,arm),p,arm,sc,old['start'],old['end']))
        for field in ['metrics','events','orders','fills','owner_fills','snapshots','decisions']:
            assert fresh[field]==old[field],(v,phase,arm,sc,field)
        controls[v,phase,arm,sc]=old;verified.append(dict(venue=v,phase=phase,arm=arm,scenario=sc))
        print('EXACT',v,phase,arm,sc,flush=True)
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in rows}
    def record(v,phase,arm,sc,start,end):
        cfg=configuration(v,arm);run=replay(data[v],states[v],None,cfg,p,arm,sc,start,end)
        if arm==spec['candidate']:run['stability_checks']=cfg['stability_checks']
        with gzip.open(OUT/'runs'/f'{v}-{phase}-{arm}-{sc}.json.gz','wt') as f:
            json.dump(run,f,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())
        m=run['metrics'];rows.append(dict(venue=v,phase=phase,arm=arm,scenario=sc,metrics=m));lookup[v,phase,arm,sc]=m
        print('RUN',v,phase,arm,sc,'PnL',round(m['net_pnl'],6),'Q',m['financial_score'],'pairs',m['completed_pairs'],flush=True)
    for sc in ['base','stress']:
        old=controls['reference','validation',spec['control'],sc]
        record('reference','validation',spec['candidate'],sc,old['start'],old['end'])
    checks=[]
    for sc in ['base','stress']:
        m=lookup['reference','validation',spec['candidate'],sc];b=lookup['reference','validation',spec['control'],sc]
        c=lookup['reference','validation',spec['parent'],sc]
        checks.append(dict(scenario=sc,pnl_difference=m['net_pnl']-b['net_pnl'],q_difference=m['financial_score']-b['financial_score'],
                           parent_pnl_difference=m['net_pnl']-c['net_pnl'],parent_q_difference=m['financial_score']-c['financial_score'],
                           above_native_pnl_q=all(m['net_pnl']>lookup['reference','validation',n,sc]['net_pnl'] and m['financial_score']>lookup['reference','validation',n,sc]['financial_score'] for n in ['first_strategy','second_strategy']),
                           no_risk_breach=m['risk_breach_bars']==0))
    promote=all(c['pnl_difference']>=-1e-8 and c['q_difference']>=-1e-12 and c['no_risk_breach'] for c in checks)
    promote &= any(c['pnl_difference']>1e-8 and c['q_difference']>1e-12 for c in checks)
    selection=dict(selected=spec['candidate'] if promote else None,checks=checks,frozen_before_later_replays=True,
                   historical_independence=False,sample_count_is_not_official_requirement=True)
    save(OUT/'selection.json',selection);selection_hash=sha(OUT/'selection.json')
    for v,phase in windows[1:]:
        for sc in ['base','stress']:
            old=controls[v,phase,spec['control'],sc]
            record(v,phase,spec['candidate'],sc,old['start'],old['end'])
    for arm in ['first_strategy','second_strategy',spec['control'],spec['parent'],spec['candidate']]:
        for sc in ['base','stress']:
            a=controls['reference','validation',spec['control'],sc]['start']
            b=controls['reference','comparison',spec['control'],sc]['end']
            record('reference','continuous',arm,sc,a,b)
    assert sha(OUT/'selection.json')==selection_hash;verify(sources);verify(prospective)
    result=dict(rows=rows,windows=windows,selection=selection,new_account_replays=22,segmented_candidates=12,
                continuous_accounts=10,exact_controls=verified,tests_passed=8,daily_bundles=len(models),complete_bundles=complete,
                successful_deleted_block_fits=refits,protected=protected(),prospective_sources=len(prospective),
                orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False)
    save(OUT/'results.json',result)
    save(HERE/'work_state.json',dict(status='progress_stability_value_accounts_completed',selection=selection,new_accounts=22,
        exact_controls=24,tests_passed=8,daily_bundles=len(models),successful_deleted_block_fits=refits,
        next_action='Attribute entry vetoes and actual cash; audit old books and compare segmented versus continuous paths. No live replacement.',
        orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,goal_complete=False,no_progress_streak=0))
    print('DONE stability comparison',flush=True)


if __name__=='__main__':main()
