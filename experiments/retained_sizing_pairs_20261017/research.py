"""Frozen retained-size family; all trading remains offline."""
import copy
import gzip
import json
import time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.par_direct_forecast_20261017.data import ROOT, OUT as INPUT, save, sha, ts
from experiments.complementary_alpha_20261017.research import protected
from experiments.hourly_refresh_pairs_20261017.research import control_directory
from experiments.hourly_pairs_robustness_20261017.analysis import load
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution
from .portfolio import replay

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'hourly_refresh_pairs_20261017'


def array(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}


def main():
    out=HERE/'artifacts';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    spec=json.loads((HERE/'protocol.json').read_text());cfgs={v:json.loads((PRIOR/f'{v}_protocol.json').read_text()) for v in ['reference','par']}
    original=json.loads((PRIOR/'artifacts/results.json').read_text())
    sources=json.loads((PRIOR/'artifacts/frozen.json').read_text())['sources'].copy()
    for rel,digest in sources.items():assert sha(ROOT/rel)==digest,rel
    paths=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',PRIOR/'artifacts/frozen.json',PRIOR/'artifacts/results.json',
           PRIOR/'artifacts/selection.json',HERE.parent/'hourly_pairs_robustness_20261017/analysis.py',
           HERE.parent/'risk_neutral_pairs_20261017/attribution.py']
    controls=['first_strategy','second_strategy','first_dense_hourlyhedge_bias_24','second_dense_hourlyhedge_bias_24']
    control_paths={}
    for v in cfgs:
        for phase in ['validation','comparison']:
            for arm in controls:
                for sc in cfgs[v]['scenarios']:
                    parent=PRIOR/'artifacts' if '_hourlyhedge_' in arm else control_directory(arm)
                    path=parent/'runs'/f'{v}-{phase}-{arm}-{sc}.json.gz'
                    control_paths[v,phase,arm,sc]=path;paths.append(path)
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),orders_sent=0,historical_independence=False,
         previous_goal_turn='progress_renewal_ablation_8_accounts_2_exact_controls_retain_renewal'))
    rd,_,p=inputs(cfgs['reference']);data={'reference':rd,'par':array(INPUT/'inputs.npz')}
    states={v:array(PRIOR/'prepared'/f'{v}_bias_24_state.npz') for v in cfgs}
    lookup={(x['venue'],x['phase'],x['arm'],x['scenario']):x['metrics'] for x in original['rows']}
    rows=[x for x in original['rows'] if x['arm'] in controls];assert len(rows)==32
    verified=[]
    verification=[('reference','validation','first_dense_hourlyhedge_bias_24','base'),
                  ('reference','comparison','second_dense_hourlyhedge_bias_24','stress'),
                  ('par','validation','first_dense_hourlyhedge_bias_24','base'),
                  ('par','comparison','second_dense_hourlyhedge_bias_24','stress')]
    for v,phase,arm,sc in verification:
        old=load(control_paths[v,phase,arm,sc])
        run=replay(data[v],states[v],None,cfgs[v],p,arm.replace('_hourlyhedge_','_renewhedge_'),sc,old['start'],old['end'])
        for k in ['fills','snapshots','events','metrics']:assert run[k]==old[k],(v,phase,sc,k)
        verified.append(dict(venue=v,phase=phase,arm=arm,scenario=sc,exact_fields=['fills','snapshots','events','metrics']))
    arms=[f'{base}_dense_utility{int(gamma)}_bias_24' for base in spec['baselines'] for gamma in spec['risk_aversions']]
    audits=[]
    def record(v,phase,arm,sc):
        cfg=copy.deepcopy(cfgs[v]);gamma=float(arm.split('_utility')[1].split('_')[0])
        cfg['retained_sizing']={k:spec[k] for k in ['gross_fractions','utility_buffer_bps']}
        cfg['retained_sizing']['risk_aversion']=gamma
        start=cfg['validation_start'] if phase=='validation' else cfg['comparison_start'];end=cfg['comparison_start'] if phase=='validation' else cfg['end']
        run=replay(data[v],states[v],None,cfg,p,arm,sc,ts(start),ts(end))
        with gzip.open(out/'runs'/f'{v}-{phase}-{arm}-{sc}.json.gz','wt') as f:
            json.dump(run,f,allow_nan=False,default=lambda z:z.item() if isinstance(z,np.generic) else z.tolist())
        sized=[e for e in run['events'] if e['event']=='hedge_adjustment_planned' and e.get('utility_enabled')]
        closed=[e for e in run['events'] if e['event']=='utility_close_requested']
        for e in sized:
            assert e['target_gross']<=min(e['original_balanced_gross'],cfg['pair_max_gross'])+1e-6
            assert e['minimum_utility_advantage_at_limits']>=e['utility_buffer_cash']-1e-8
            assert e['minimum_advantage_at_limits']>=e['buffer_cash']-1e-8
        m=run['metrics'];lookup[v,phase,arm,sc]=m
        matched=arm.split('_dense_')[0]+'_dense_hourlyhedge_bias_24';old=lookup[v,phase,matched,sc]
        row=dict(venue=v,phase=phase,arm=arm,scenario=sc,metrics=m,matched=matched,
                 utility_resize_plans=len(sized),utility_close_requests=len(closed),
                 delta_pnl=m['net_pnl']-old['net_pnl'],delta_q=m['financial_score']-old['financial_score'])
        rows.append(row)
        audit=episode_attribution(run,data[v],cfg['scenarios'][sc]);audits.append(dict(venue=v,phase=phase,arm=arm,scenario=sc,**audit))
        print(v,phase,arm,sc,'PnL',round(m['net_pnl'],3),'Q',round(m['financial_score'],7),
              'delta',round(row['delta_pnl'],3),'resize',len(sized),'close',len(closed),'completed',m['completed_pairs'],flush=True)
    for arm in arms:
        for sc in cfgs['reference']['scenarios']:record('reference','validation',arm,sc)
    ranking=[]
    for arm in arms:
        checks=[];advantages=[]
        for sc in cfgs['reference']['scenarios']:
            m=lookup['reference','validation',arm,sc]
            checks.extend([m['net_pnl']>0,m['risk_breach_bars']==0,m['completed_pairs']>=3])
            for baseline in ['first_strategy','second_strategy']:
                b=lookup['reference','validation',baseline,sc]
                checks.extend([m['net_pnl']>b['net_pnl'],m['financial_score']>b['financial_score']])
                advantages.append(m['net_pnl']-b['net_pnl'])
        ranking.append(dict(arm=arm,eligible=all(checks),minimum_native_pnl_advantage=min(advantages)))
    ranking.sort(key=lambda x:(-x['eligible'],-x['minimum_native_pnl_advantage'],x['arm']))
    selection=dict(selected=ranking[0]['arm'],eligible=ranking[0]['eligible'],ranking=ranking,
        frozen_before_later_account_replay=True,historical_independence=False)
    save(out/'selection.json',selection);selection_hash=sha(out/'selection.json')
    for v,phase in [('reference','comparison'),('par','validation'),('par','comparison')]:
        for arm in arms:
            for sc in cfgs[v]['scenarios']:record(v,phase,arm,sc)
    assert sha(out/'selection.json')==selection_hash
    for rel,digest in sources.items():assert sha(ROOT/rel)==digest,rel
    save(out/'attribution.json',dict(rows=audits,post_evaluation_only=True,orders_sent=0))
    save(out/'results.json',dict(rows=rows,selection=selection,new_account_replays=32,exact_control_replays=verified,
        reused_control_accounts=32,protected=protected(),orders_sent=0,historical_independence=False,automatic_live_replacement=False))
    print('DONE 32 new accounts; 4 exact controls; eligible',selection['eligible'],flush=True)


if __name__=='__main__':main()
