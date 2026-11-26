"""Execution-faithful isolated historical episodes; no forward tapes or orders."""
import copy
import gzip
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.dense_history_pairs_20261017.model import design
from experiments.dense_history_pairs_20261017.research import load
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha
from experiments.conditional_completion_pairs_20261017.research import verify
from experiments.complementary_alpha_20261017.research import protected
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution
from .model import feature
from .portfolio import replay

HERE=Path(__file__).resolve().parent; OUT=HERE/'artifacts'
SECTOR=HERE.parent/'decision_sector_pairs_20261018/artifacts'
GATE=HERE.parent/'decision_gate_pairs_20261017/artifacts'


def add_features(state,reference,symbols,pairs):
    f=design(reference['market_times'],reference['close'],reference['volume'],reference['trades'],symbols,pairs)
    src=state['signal_times']; indexes=np.searchsorted(reference['market_times'],src)
    bounded=np.clip(indexes,0,len(reference['market_times'])-1)
    valid=(indexes<len(reference['market_times']))&(reference['market_times'][bounded]==src)
    state={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in state.items()}
    state['lifecycle_features']=f['x'][bounded,np.arange(len(pairs))[None,:]].copy()
    state['lifecycle_features'][~valid]=np.nan
    return state,f['names']


def label_from_run(run,d,state,p,pair_index,source,sample_id,sc):
    entries=[e for e in run['events'] if e['event']=='pair_intent']
    row=dict(sample_id=sample_id,source=source,pair_index=pair_index,status='not_admitted',
             outcome_time=None,net_pnl=None,target=None)
    assert len(entries)<=1
    if not entries:return row
    entry=entries[0];assert entry['time']==source and entry['pair_index']==pair_index
    i=int(np.searchsorted(d['times'],source));x,unit,gross=feature(entry,d,state,i,p['initial'])
    row.update(features=x.tolist(),unit=unit,intended_gross=gross,entry=entry)
    audit=episode_attribution(run,d,sc);assert len(audit['episodes'])==1
    episode=audit['episodes'][0];row['episode']=episode
    if episode['fills']==0:
        failed=[e for e in run['events'] if e['event']=='entry_unfilled']
        assert len(failed)==1 and abs(audit['pair_owned_pnl'])<1e-8
        row.update(status='unfilled',outcome_time=failed[0]['time'],net_pnl=0.,target=0.)
    elif episode['closed']:
        row.update(status='closed',outcome_time=episode['end'],net_pnl=episode['net_pnl'],
                   target=episode['net_pnl']/gross/unit)
        flat=next(s for s in run['snapshots'] if s['time']==row['outcome_time'])
        assert all(q==0 for q in flat['pair_qty'])
        assert all(q==0 for s in run['snapshots'] if s['time']>=row['outcome_time'] for q in s['pair_qty'])
    else:
        row.update(status='censored',last_observation=run['end'])
    if row['outcome_time'] is not None:assert row['outcome_time']>source
    return row


def main():
    OUT.mkdir(exist_ok=False);(OUT/'episodes').mkdir()
    spec=json.loads((HERE/'protocol.json').read_text())
    test=(HERE/'test_results.txt').read_text();assert 'Ran 8 tests' in test and test.rstrip().endswith('OK')
    cfg=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/reference_protocol.json').read_text())
    cfg['dense_pairs']=spec['pairs'];cfg['conditional_completion']={'risk_aversion':0.}
    paths=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',SECTOR/'reference_sector_state.npz',
           ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',
           ROOT/'experiments/risk_neutral_pairs_20261017/attribution.py']
    sources=json.loads((SECTOR/'frozen.json').read_text())['sources'].copy()
    sources.update({str(path.relative_to(ROOT)):sha(path) for path in paths})
    verify(sources);frozen_protected=protected()
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=frozen_protected,
                              previous_goal_turn=spec['previous_goal_turn'],historical_independence=False))
    d,_,p=inputs(cfg);reference=load(ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz')
    state,names=add_features(load(SECTOR/'reference_sector_state.npz'),reference,p['symbols'],spec['pairs'])
    np.savez_compressed(OUT/'reference_state.npz',**state)
    t=d['times'];available=state['economic_available']&(state['economic_action']!=0)&(state['directions']!=0)
    available &= np.isfinite(state['mu']) & (t%(spec['sample_hours']*3600)==0)[:,None]
    indexes=np.argwhere(available);rows=[]
    last=int(t[t%86400==43200][-1]);arm='first_dense_hourlyhedge_lifecycle_label_24'
    for count,(i,k) in enumerate(indexes):
        source=int(t[i]);k=int(k);sample_id=f'{source}-{k}'
        start=(source-43200)//86400*86400+43200
        end=min(last,((source+spec['maximum_observation_hours']*3600-43200+86399)//86400)*86400+43200)
        if end<=source:
            rows.append(dict(sample_id=sample_id,source=source,pair_index=k,status='unobserved_next_bar',
                             outcome_time=None,net_pnl=None,target=None));continue
        run_cfg=copy.deepcopy(cfg);run_cfg['lifecycle_training_opportunity']=dict(source=source,pair_index=k)
        run=replay(d,state,None,run_cfg,p,arm,'base',start,end)
        row=label_from_run(run,d,state,p,k,source,sample_id,cfg['scenarios']['base'])
        rows.append(row)
        if row['status']!='not_admitted':
            with gzip.open(OUT/'episodes'/f'{sample_id}.json.gz','wt') as f:
                json.dump(run,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        if count%10==0:
            print('EPISODES',count+1,'/',len(indexes),dict(Counter(r['status'] for r in rows)),flush=True)
    verify(sources);protected()
    result=dict(rows=rows,feature_names=names+['pair_'+str(k) for k in range(6)]+[
                'direction','expected_net_in_source_units','expected_funding_in_source_units','log_gross_fraction','source_beta'],
                counts=dict(Counter(r['status'] for r in rows)),attempted_virtual_accounts=len(indexes),
                source_cadence_hours=spec['sample_hours'],orders_sent=0,reserved_raw_read=False,historical_independence=False,
                limitations='Reference virtual episodes with reset account state and overlapping paths. Censored outcomes are not zero labels; candle volume is not real L2 depth. Pair-owner return is not total marginal account improvement.')
    save(OUT/'labels.json',result)
    save(HERE/'work_state.json',dict(status='progress_lifecycle_dataset_built',counts=result['counts'],
         next_action='Fit strictly mature pooled values and compare original two-pair filter and six-pair ranking before later account evaluation.',
         orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,goal_complete=False,no_progress_streak=0))
    print('DONE labels',result['counts'],flush=True)


if __name__=='__main__':main()
