"""Frozen one-completion candidate family with exact prior controls."""
import copy
import gzip
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.par_direct_forecast_20261017.data import ROOT, OUT as INPUT, save, sha, ts
from experiments.complementary_alpha_20261017.research import protected
from experiments.hourly_pairs_robustness_20261017.analysis import load
from experiments.retained_sizing_pairs_20261017.research import array
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution
from .portfolio import replay

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'retained_sizing_pairs_20261017'
HOURLY=HERE.parent/'hourly_refresh_pairs_20261017'


def main():
    out=HERE/'artifacts';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    spec=json.loads((HERE/'protocol.json').read_text());base_spec=json.loads((PRIOR/'protocol.json').read_text())
    cfgs={v:json.loads((HOURLY/f'{v}_protocol.json').read_text()) for v in ['reference','par']}
    original=json.loads((PRIOR/'artifacts/results.json').read_text())
    sources=json.loads((PRIOR/'artifacts/frozen.json').read_text())['sources'].copy()
    for rel,digest in sources.items():assert sha(ROOT/rel)==digest,rel
    paths=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',PRIOR/'artifacts/frozen.json',
        PRIOR/'artifacts/results.json',PRIOR/'artifacts/selection.json',PRIOR/'protocol.json']
    paths.extend(PRIOR/'artifacts/runs'/f"{r['venue']}-{r['phase']}-{r['arm']}-{r['scenario']}.json.gz" for r in original['rows'] if '_utility' in r['arm'])
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),orders_sent=0,
         historical_independence=False,previous_goal_turn='progress_retained_sizing_partial_unwind_bottleneck_identified'))
    rd,_,p=inputs(cfgs['reference']);data={'reference':rd,'par':array(INPUT/'inputs.npz')}
    states={v:array(HOURLY/'prepared'/f'{v}_bias_24_state.npz') for v in cfgs}
    def config(v):
        cfg=copy.deepcopy(cfgs[v]);cfg['retained_sizing']={k:base_spec[k] for k in ['gross_fractions','utility_buffer_bps']}
        cfg['retained_sizing']['risk_aversion']=spec['risk_aversion'];return cfg
    verification=[('reference','validation','first_dense_utility64_bias_24','base'),
        ('reference','comparison','second_dense_utility64_bias_24','base'),
        ('par','validation','first_dense_utility64_bias_24','base'),
        ('par','comparison','second_dense_utility64_bias_24','stress')]
    verified=[]
    for v,phase,arm,sc in verification:
        old=load(PRIOR/'artifacts/runs'/f'{v}-{phase}-{arm}-{sc}.json.gz')
        run=replay(data[v],states[v],None,config(v),p,arm,sc,old['start'],old['end'])
        for k in ['fills','snapshots','events','metrics']:assert run[k]==old[k],(v,phase,sc,k)
        verified.append(dict(venue=v,phase=phase,arm=arm,scenario=sc,exact_fields=['fills','snapshots','events','metrics']))
    rows=list(original['rows']);assert len(rows)==64
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in rows};audits=[];paths=[]
    arms=[f'{base}_dense_utilitycomplete64_bias_24' for base in spec['baselines']]
    def record(v,phase,arm,sc):
        cfg=config(v);start=cfg['validation_start'] if phase=='validation' else cfg['comparison_start']
        end=cfg['comparison_start'] if phase=='validation' else cfg['end']
        run=replay(data[v],states[v],None,cfg,p,arm,sc,ts(start),ts(end))
        with gzip.open(out/'runs'/f'{v}-{phase}-{arm}-{sc}.json.gz','wt') as f:
            json.dump(run,f,allow_nan=False,default=lambda z:z.item() if isinstance(z,np.generic) else z.tolist())
        m=run['metrics'];lookup[v,phase,arm,sc]=m
        pending=[e for e in run['events'] if e['event']=='retained_completion_pending']
        outcomes=[e for e in run['events'] if e.get('retained_completion_attempt')==1]
        submitted=[d for d in run['decisions'] if d.get('rehedge_order') and any(e['time']==d['decision_time'] for e in pending)]
        assert len(submitted)==len(outcomes)
        for e in pending:
            plan=e['completion_plan'];assert plan['retained_completion_attempt']==1
            assert plan['target']==e['original_target'];assert plan['new_source']<=plan['decision_time']
            assert plan['minimum_advantage_at_limits']>=plan['buffer_cash']-1e-8
            assert plan['minimum_utility_advantage_at_limits']>=plan['utility_buffer_cash']-1e-8
            old=np.array(plan['old_qty']);target=np.array(plan['target'])
            assert np.all(target*old>0) and np.all(abs(target)<=abs(old)+1e-8)
        for outcome in outcomes:
            if outcome['outcome']=='renewed':
                packet=next(s for s in submitted if s['decision_time']==outcome['adjustment_decision_time'])
                event=next(e for e in pending if e['time']==packet['decision_time'])
                assert outcome['time']<=event['completion_plan']['retained_completion_deadline']
        matched=arm.replace('_utilitycomplete64_','_utility64_');old=lookup[v,phase,matched,sc]
        hourly=arm.split('_dense_')[0]+'_dense_hourlyhedge_bias_24';h=lookup[v,phase,hourly,sc]
        rows.append(dict(venue=v,phase=phase,arm=arm,scenario=sc,metrics=m,matched=matched,
            completion_pending=len(pending),completion_submitted=len(submitted),completion_outcomes=dict(Counter(e['outcome'] for e in outcomes)),
            delta_pnl=m['net_pnl']-old['net_pnl'],delta_q=m['financial_score']-old['financial_score'],
            delta_hourly_pnl=m['net_pnl']-h['net_pnl'],delta_hourly_q=m['financial_score']-h['financial_score']))
        audits.append(dict(venue=v,phase=phase,arm=arm,scenario=sc,**episode_attribution(run,data[v],cfg['scenarios'][sc])))
        paths.append(dict(venue=v,phase=phase,arm=arm,scenario=sc,pending=pending,outcomes=outcomes,
            rejections=[e for e in run['events'] if e.get('retained_completion_rejection')]))
        print(v,phase,arm,sc,'PnL',round(m['net_pnl'],3),'Q',round(m['financial_score'],7),
            'delta_v1',round(m['net_pnl']-old['net_pnl'],3),'delta_hourly',round(m['net_pnl']-h['net_pnl'],3),
            'completion',len(submitted),dict(Counter(e['outcome'] for e in outcomes)),flush=True)
    for arm in arms:
        for sc in cfgs['reference']['scenarios']:record('reference','validation',arm,sc)
    ranking=[]
    for arm in arms:
        checks=[];adv=[]
        for sc in cfgs['reference']['scenarios']:
            m=lookup['reference','validation',arm,sc];checks.extend([m['net_pnl']>0,m['completed_pairs']>=3,m['risk_breach_bars']==0])
            for base in ['first_strategy','second_strategy']:
                b=lookup['reference','validation',base,sc];checks.extend([m['net_pnl']>b['net_pnl'],m['financial_score']>b['financial_score']]);adv.append(m['net_pnl']-b['net_pnl'])
        ranking.append(dict(arm=arm,eligible=all(checks),minimum_native_pnl_advantage=min(adv)))
    ranking.sort(key=lambda x:(-x['eligible'],-x['minimum_native_pnl_advantage'],x['arm']))
    selection=dict(selected=ranking[0]['arm'],eligible=ranking[0]['eligible'],ranking=ranking,frozen_before_later_account_replay=True,historical_independence=False)
    save(out/'selection.json',selection);selection_hash=sha(out/'selection.json')
    for v,phase in [('reference','comparison'),('par','validation'),('par','comparison')]:
        for arm in arms:
            for sc in cfgs[v]['scenarios']:record(v,phase,arm,sc)
    assert sha(out/'selection.json')==selection_hash
    for rel,digest in sources.items():assert sha(ROOT/rel)==digest,rel
    save(out/'results.json',dict(rows=rows,selection=selection,new_account_replays=16,exact_control_replays=verified,
         reused_control_accounts=64,protected=protected(),orders_sent=0,historical_independence=False,automatic_live_replacement=False))
    save(out/'attribution.json',dict(rows=audits,orders_sent=0,post_evaluation_only=True))
    save(out/'completion_paths.json',dict(rows=paths,orders_sent=0))
    print('DONE 16 new accounts; 4 exact controls; eligible',selection['eligible'],flush=True)


if __name__=='__main__':main()
