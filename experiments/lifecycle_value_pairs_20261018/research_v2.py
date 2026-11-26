"""Resume after array/list comparison error; reuse already fitted values unchanged."""
import copy
import gzip
import json
import time
from pathlib import Path
import numpy as np
from .dataset import ROOT,HERE,OUT,GATE,SECTOR,inputs,load,save,sha,verify,protected,add_features
from .model import fit
from .portfolio import replay


def readrun(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def main():
    spec=json.loads((HERE/'protocol.json').read_text())
    labels=json.loads((OUT/'labels.json').read_text())
    assert (HERE/'dataset.log').read_text().rstrip().splitlines()[-1].startswith('DONE labels')
    assert not (OUT/'results.json').exists()
    assert not list((OUT/'runs').glob('*.json.gz')), 'Unexpected completed candidate runs'
    (OUT/'runs').mkdir(exist_ok=True)
    original=json.loads((GATE/'results.json').read_text());windows=original['windows']
    rows=json.loads((SECTOR/'results.json').read_text())['rows'].copy()
    configs={v:json.loads((ROOT/f'experiments/hourly_refresh_pairs_20261017/{v}_protocol.json').read_text()) for v in ['reference','par']}
    configs['par_extended']=copy.deepcopy(configs['par'])
    d,_,p=inputs(configs['reference'])
    data={'reference':d,
          'par':load(ROOT/'experiments/par_direct_forecast_20261017/artifacts/inputs.npz'),
          'par_extended':load(ROOT/'experiments/historical_par_extension_20261017/artifacts_v3/inputs.npz')}
    extended=load(ROOT/'practice_live/updates/20261017T120009814481Z/inputs.npz')
    states={'reference':load(OUT/'reference_state.npz')}
    for venue in ['par','par_extended']:
        states[venue],names=add_features(load(SECTOR/f'{venue}_sector_state.npz'),extended,p['symbols'],spec['pairs'])
        np.savez_compressed(OUT/f'{venue}_state.npz',**states[venue])
    paths=[Path(__file__),HERE/'research.py',OUT/'evaluation_frozen.json',OUT/'models.json',OUT/'labels.json',OUT/'reference_state.npz',OUT/'par_state.npz',OUT/'par_extended_state.npz',
           GATE/'results.json',SECTOR/'results.json',ROOT/'practice_live/updates/20261017T120009814481Z/inputs.npz']
    paths += [GATE/'runs'/f'{v}-{phase}-{spec["control"]}-{sc}.json.gz' for v,phase in windows for sc in ['base','stress']]
    sources=json.loads((OUT/'frozen.json').read_text())['sources'].copy()
    sources.update({str(path.relative_to(ROOT)):sha(path) for path in paths});verify(sources)
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(prospective)
    save(OUT/'evaluation_frozen_v2.json',dict(at_unix=time.time(),sources=sources,protected=protected()))
    models={int(k):v for k,v in json.loads((OUT/'models.json').read_text()).items()};records=list(models.values())
    print('FIT',len(records),'models','last rows',records[-1]['rows'] if records else 0,flush=True)
    controls={};verified=[]
    for v,phase in windows:
        for sc in ['base','stress']:
            old=readrun(GATE/'runs'/f'{v}-{phase}-{spec["control"]}-{sc}.json.gz')
            cfg=copy.deepcopy(configs[v]);cfg.update(dense_pairs=spec['pairs'],conditional_completion={'risk_aversion':0.},
                                                    lifecycle_kind='filter',lifecycle_models={})
            fresh=replay(data[v],states[v],None,cfg,p,spec['control'],sc,old['start'],old['end'])
            fresh=json.loads(json.dumps(fresh,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))
            for field in ['metrics','events','orders','fills','owner_fills','snapshots','decisions']:
                assert fresh[field]==old[field],(v,phase,sc,field)
            controls[v,phase,sc]=old;verified.append(dict(venue=v,phase=phase,scenario=sc))
            print('EXACT',v,phase,sc,flush=True)
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in rows}
    def record(v,phase,kind,sc):
        old=controls[v,phase,sc];arm=f'first_dense_hourlyhedge_lifecycle_{kind}_24'
        cfg=copy.deepcopy(configs[v]);cfg.update(dense_pairs=spec['pairs'],conditional_completion={'risk_aversion':0.},
                                                lifecycle_kind=kind,lifecycle_models=models)
        run=replay(data[v],states[v],None,cfg,p,arm,sc,old['start'],old['end'])
        with gzip.open(OUT/'runs'/f'{v}-{phase}-{arm}-{sc}.json.gz','wt') as f:
            json.dump(run,f,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())
        m=run['metrics'];rows.append(dict(venue=v,phase=phase,arm=arm,scenario=sc,metrics=m));lookup[v,phase,arm,sc]=m
        print('RUN',v,phase,kind,sc,'PnL',round(m['net_pnl'],6),'Q',m['financial_score'],'pairs',m['completed_pairs'],flush=True)
    for kind in spec['candidates']:
        for sc in ['base','stress']:record('reference','validation',kind,sc)
    ranking=[]
    for kind in spec['candidates']:
        arm=f'first_dense_hourlyhedge_lifecycle_{kind}_24';checks=[]
        for sc in ['base','stress']:
            m=lookup['reference','validation',arm,sc];b=lookup['reference','validation',spec['control'],sc]
            checks.append(dict(scenario=sc,pnl_difference=m['net_pnl']-b['net_pnl'],q_difference=m['financial_score']-b['financial_score'],
                               above_native_pnl_q=all(m['net_pnl']>lookup['reference','validation',n,sc]['net_pnl'] and m['financial_score']>lookup['reference','validation',n,sc]['financial_score'] for n in ['first_strategy','second_strategy']),
                               no_risk_breach=m['risk_breach_bars']==0))
        promote=all(c['pnl_difference']>=-1e-8 and c['q_difference']>=-1e-12 and c['no_risk_breach'] for c in checks)
        promote &= any(c['pnl_difference']>1e-8 and c['q_difference']>1e-12 for c in checks)
        ranking.append(dict(kind=kind,arm=arm,checks=checks,promote=bool(promote),minimum_pnl_advantage=min(c['pnl_difference'] for c in checks)))
    ranking.sort(key=lambda r:(r['minimum_pnl_advantage'],r['kind']),reverse=True)
    selection=dict(selected=next((r['arm'] for r in ranking if r['promote']),None),ranking=ranking,
                   frozen_before_later_replays=True,historical_independence=False,sample_count_is_not_official_requirement=True)
    save(OUT/'selection.json',selection);selection_hash=sha(OUT/'selection.json')
    for v,phase in windows[1:]:
        for kind in spec['candidates']:
            for sc in ['base','stress']:record(v,phase,kind,sc)
    assert sha(OUT/'selection.json')==selection_hash;verify(sources);verify(prospective)
    result=dict(rows=rows,windows=windows,selection=selection,new_account_replays=24,exact_controls=verified,
                tests_passed=8,dataset_counts=labels['counts'],virtual_training_accounts=labels['attempted_virtual_accounts'],
                daily_value_models=len(records),minimum_effective_rows=min(r['effective_rows'] for r in records),
                maximum_effective_rows=max(r['effective_rows'] for r in records),protected=protected(),prospective_sources=len(prospective),
                orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False)
    save(OUT/'results.json',result)
    save(HERE/'work_state.json',dict(status='progress_lifecycle_value_accounts_completed',selection=selection,
         new_accounts=24,exact_controls=12,tests_passed=8,dataset_counts=labels['counts'],daily_value_models=len(records),
         next_action='Audit per-pair cash and daily issued predictions, then old-book execution feasibility. Preserve the prior improved candidate if additional improvement is unsupported.',
         orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,goal_complete=False,no_progress_streak=0))
    print('DONE lifecycle value evaluation',flush=True)


if __name__=='__main__':main()
