"""Causal source plans and independently executed pair outcomes on old par data."""
import copy
import gzip
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.par_direct_forecast_20261017.model import design
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha,ts
from experiments.dense_history_pairs_20261017.research import load
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.complementary_alpha_20261017.research import protected
from experiments.conditional_completion_pairs_20261017.research import verify
from .portfolio import replay
from .entry import plan
from .model import features

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'


def setup():
    spec=json.loads((HERE/'protocol.json').read_text())
    cfg=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/par_protocol.json').read_text())
    cfg.update(venue_cash=spec,signal_cadence_seconds=spec['sample_seconds'])
    reference_cfg=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/reference_protocol.json').read_text())
    _,_,p=inputs(reference_cfg)
    d=load(ROOT/'experiments/historical_par_extension_20261017/artifacts_v3/inputs.npz')
    de=design(d,spec,p['symbols']);n,k=de['fresh'].shape
    valid=np.isfinite(de['x']).all(-1)&de['fresh']&(de['beta']>=.1)&(de['beta']<=10)&(de['correlation']>=spec['minimum_correlation'])
    column=de['names'].index('hourly_sigma_bps')
    de.update(dense=True,directions=valid.astype(int),signal_times=np.broadcast_to(d['times'][:,None],(n,k)).copy(),
              source_prices=d['par_close'].copy(),vol=de['x'][:,:,column]/10000,mu=np.zeros((n,k)),forecast_times=d['times'].copy())
    return spec,cfg,p,d,de


def outcome(run,proposal,d,de,i,p,sample_id):
    x,gross=features(proposal,d,de,i,p['initial'])
    source=int(d['times'][i]);entries=[e for e in run['events'] if e['event']=='pair_intent']
    assert len(entries)==1 and entries[0]['time']==source
    row=dict(sample_id=sample_id,source=source,pair_index=proposal['pair_index'],direction=proposal['direction'],
             horizon=proposal['horizon'],features=x.tolist(),intended_gross=gross,status='censored',
             outcome_time=None,net_pnl=None,requested_end=run.get('requested_end',run['end']))
    closed=[e for e in run['events'] if e['event']=='closed'];unfilled=[e for e in run['events'] if e['event']=='entry_unfilled']
    if unfilled:
        assert not run['owner_fills']['pair'] and abs(run['metrics']['pair_owned_pnl'])<1e-8
        row.update(status='unfilled',outcome_time=unfilled[0]['time'],net_pnl=0.)
    elif closed:
        assert all(v==0 for v in run['snapshots'][-1]['pair_qty'])
        row.update(status='closed',outcome_time=closed[0]['time'],net_pnl=run['metrics']['pair_owned_pnl'],
                   balanced=closed[0]['balanced'],aborted=closed[0]['aborted'])
    row.update(fills=len(run['owner_fills']['pair']),fees=run['metrics']['fees'],funding=run['metrics']['funding'],
               partial_events=sum(e['event'] in ('one_leg_completion_pending','repair_requested') for e in run['events']),
               last_observation=run['end'])
    if row['outcome_time'] is not None:assert row['outcome_time']>source
    return row


def main():
    test=(HERE/'test_results.txt').read_text();assert 'Ran 8 tests' in test and test.rstrip().endswith('OK')
    OUT.mkdir(exist_ok=False);(OUT/'episodes').mkdir()
    spec,cfg,p,d,de=setup();np.savez_compressed(OUT/'state.npz',**de)
    previous=ROOT/'experiments/lifecycle_stability_pairs_20261018/artifacts/conclusion_frozen.json'
    verify(json.loads(previous.read_text())['sources'])
    sources=json.loads((ROOT/'experiments/decision_gate_pairs_20261017/artifacts/frozen.json').read_text())['sources'].copy()
    files=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',previous,
           ROOT/'experiments/par_direct_forecast_20261017/model.py',ROOT/'experiments/lifecycle_value_pairs_20261018/model.py',
           ROOT/'experiments/historical_par_extension_20261017/artifacts_v3/inputs.npz']
    sources.update({str(f.relative_to(ROOT)):sha(f) for f in files});verify(sources)
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),previous_goal_turn=spec['previous_goal_turn']))
    rows=[];replays=0;parity=[];t=d['times'];last=int(t[t%86400==43200][-1])
    indexes=np.argwhere((de['directions']!=0)&((t%spec['sample_seconds']==0)[:,None]))
    for count,(i,k) in enumerate(indexes):
        source=int(t[i]);i=int(i);k=int(k)
        for direction in (-1,1):
            for horizon in spec['horizons_hours']:
                sample_id=f'{source}-{k}-{direction}-{horizon}'
                proposal=plan(i,k,direction,horizon,d,de,cfg,p,cfg['scenarios']['base'],np.zeros(10),p['initial'])
                if proposal is None:
                    rows.append(dict(sample_id=sample_id,source=source,pair_index=k,direction=direction,horizon=horizon,
                                     status='not_admitted',outcome_time=None,net_pnl=None));continue
                end=min(last,((source+spec['maximum_observation_hours']*3600-43200+86399)//86400)*86400+43200)
                if end<=source:
                    rows.append(dict(sample_id=sample_id,source=source,pair_index=k,direction=direction,horizon=horizon,
                                     status='unobserved_next_bar',outcome_time=None,net_pnl=None));continue
                start=(source-43200)//86400*86400+43200
                rcfg=copy.deepcopy(cfg);rcfg.update(venue_label=dict(source=source,pair_index=k,direction=direction),label_capture=True)
                arm=f'pair_dense_venue_label_{horizon}'
                run=replay(d,de,None,rcfg,p,arm,'base',start,end);replays+=1
                row=outcome(run,proposal,d,de,i,p,sample_id);rows.append(row)
                with gzip.open(OUT/'episodes'/f'{sample_id}.json.gz','wt') as f:
                    json.dump(run,f,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())
                if row['status'] in ('closed','unfilled') and row['status'] not in [r['status'] for r in parity]:
                    full_cfg=copy.deepcopy(rcfg);full_cfg.pop('label_capture')
                    full=replay(d,de,None,full_cfg,p,arm,'base',start,end)
                    assert full['fills']==run['fills'] and full['owner_fills']==run['owner_fills']
                    assert abs(full['metrics']['pair_owned_pnl']-row['net_pnl'])<1e-8
                    parity.append(dict(status=row['status'],sample_id=sample_id,identical_fills_and_cash=True))
        if count%20==0:print('SOURCE_PAIRS',count+1,'/',len(indexes),'replays',replays,dict(Counter(r['status'] for r in rows)),flush=True)
    verify(sources);protected()
    result=dict(rows=rows,counts=dict(Counter(r['status'] for r in rows)),virtual_account_replays=replays,
                source_pairs=len(indexes),early_return_full_replay_checks=parity,feature_names=de['names'],
                orders_sent=0,historical_independence=False,reserved_raw_read=False)
    save(OUT/'labels.json',result)
    save(HERE/'work_state.json',dict(status='progress_par_execution_outcomes_generated',counts=result['counts'],
         next_action='Fit fixed-date action-value models and matched price controls, select on earlier period, then evaluate later and old books.',
         orders_sent=0,live_strategy_changed=False,goal_complete=False,no_progress_streak=0))
    print('DONE venue labels',result['counts'],'actual replays',replays,'early-return checks',parity,flush=True)


if __name__=='__main__':main()
