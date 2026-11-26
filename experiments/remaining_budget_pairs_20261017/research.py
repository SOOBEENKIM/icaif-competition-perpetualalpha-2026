"""Compare a bounded adjustment budget with frozen original and veto accounts."""
import copy,csv,gzip,html,json,time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.complementary_alpha_20261017.research import protected
from experiments.conditional_completion_pairs_20261017.research import verify
from experiments.dense_history_pairs_20261017.research import load
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as PAR_INPUT,save,sha
from experiments.remaining_value_pairs_20261017.research import readrun
from .portfolio import replay

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'
V1=HERE.parent/'remaining_value_pairs_20261017/artifacts'
GATE=HERE.parent/'decision_gate_pairs_20261017/artifacts'
EXT=HERE.parent/'historical_par_extension_20261017/artifacts_v3'


def main():
    OUT.mkdir(exist_ok=False);(OUT/'runs').mkdir()
    spec=json.loads((HERE/'protocol.json').read_text())
    tests=(HERE/'test_results.txt').read_text();assert 'Ran 15 tests' in tests and tests.rstrip().endswith('OK')
    parent=json.loads((V1/'results.json').read_text());windows=parent['windows']
    arms=['first_strategy','second_strategy','first_dense_hourlyhedge_cash_direct_24',spec['control'],spec['failed_veto_control']]
    rows=[r for r in parent['rows'] if r['arm'] in arms]
    cfgs={v:json.loads((ROOT/f'experiments/hourly_refresh_pairs_20261017/{v}_protocol.json').read_text()) for v in ['reference','par']}
    cfgs['par_extended']=copy.deepcopy(cfgs['par'])
    paths={}
    for v,ph in windows:
        for sc in ['base','stress']:
            for arm in [spec['control'],spec['failed_veto_control']]:
                parent_dir=GATE if arm==spec['control'] else V1
                paths[v,ph,sc,arm]=parent_dir/'runs'/f'{v}-{ph}-{arm}-{sc}.json.gz'
    sources=json.loads((V1/'frozen.json').read_text())['sources'].copy()
    files=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',HERE/'test_integer_dtype_failure.txt',
           V1/'results.json',V1/'diagnostics.json',V1/'frozen.json',V1/'preparation.json',
           *[V1/f'{v}_remaining_state.npz' for v in cfgs],*paths.values()]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in files});verify(sources)
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(prospective)
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),previous_goal_turn=spec['previous_goal_turn']))
    rd,_,p=inputs(cfgs['reference']);data={'reference':rd,'par':load(PAR_INPUT/'inputs.npz'),'par_extended':load(EXT/'inputs.npz')}
    states={v:load(V1/f'{v}_remaining_state.npz') for v in cfgs};old={k:readrun(path) for k,path in paths.items()};verified=[]
    for (venue,phase,sc,arm),saved in old.items():
        cfg=copy.deepcopy(cfgs[venue]);cfg['conditional_completion']={'risk_aversion':0.}
        if arm==spec['failed_veto_control']:cfg['remaining_value']={'mode':'renewal','decision_cadence_hours':6,'exit_buffer_bps':2.}
        run=replay(data[venue],states[venue],None,cfg,p,arm,sc,saved['start'],saved['end'])
        for key in ['events','fills','metrics','snapshots','owner_fills']:assert run[key]==saved[key],(venue,phase,sc,arm,key)
        verified.append(dict(venue=venue,phase=phase,scenario=sc,arm=arm));print('EXACT',venue,phase,sc,arm,flush=True)
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in rows};decisions=[]
    def record(venue,phase,sc):
        cfg=copy.deepcopy(cfgs[venue]);cfg['conditional_completion']={'risk_aversion':0.}
        cfg['remaining_value']={'mode':'budget','decision_cadence_hours':6,'exit_buffer_bps':2.}
        saved=old[venue,phase,sc,spec['control']];arm=spec['candidate']
        run=replay(data[venue],states[venue],None,cfg,p,arm,sc,saved['start'],saved['end'])
        with gzip.open(OUT/'runs'/f'{venue}-{phase}-{arm}-{sc}.json.gz','wt') as f:json.dump(run,f,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())
        m=run['metrics'];lookup[venue,phase,arm,sc]=m
        rows.append(dict(venue=venue,phase=phase,arm=arm,scenario=sc,metrics=m))
        adjustments=[e for e in run['events'] if 'original_entry_limits' in e]
        tightened=sum(any(abs(e['entry_limits'][k]-e['original_entry_limits'][k])>1e-8 for k in e['entry_limits']) for e in adjustments)
        decisions.append(dict(venue=venue,phase=phase,scenario=sc,joint_budget_adjustments=len(adjustments),tightened=tightened,
            preview_rejections=sum(e.get('reason')=='remaining_budget_cannot_fund_preview' for e in run['events']),
            joint_arrival_unwinds=sum(e.get('reason')=='arrival_joint_remaining_cash_insufficient' for e in run['events'])))
        print('RUN',venue,phase,sc,'PnL',round(m['net_pnl'],6),'Q',m['financial_score'],decisions[-1],flush=True)
    for sc in ['base','stress']:record('reference','validation',sc)
    checks=[]
    for sc in ['base','stress']:
        m=lookup['reference','validation',spec['candidate'],sc];b=lookup['reference','validation',spec['control'],sc]
        checks.append(dict(scenario=sc,pnl_difference=m['net_pnl']-b['net_pnl'],q_difference=m['financial_score']-b['financial_score'],
                           no_risk_breach=m['risk_breach_bars']==0,completed_pairs=m['completed_pairs']))
    promote=all(c['pnl_difference']>=-1e-8 and c['q_difference']>=-1e-12 and c['no_risk_breach'] for c in checks)
    promote &= any(c['pnl_difference']>1e-8 and c['q_difference']>1e-12 for c in checks)
    selection=dict(candidate=spec['candidate'],promote=bool(promote),checks=checks,frozen_before_later_replays=True,historical_independence=False)
    save(OUT/'selection.json',selection);selection_hash=sha(OUT/'selection.json')
    for v,ph in windows[1:]:
        for sc in ['base','stress']:record(v,ph,sc)
    assert sha(OUT/'selection.json')==selection_hash;verify(sources);verify(prospective)
    save(OUT/'results.json',dict(rows=rows,windows=windows,selection=selection,decisions=decisions,new_account_replays=12,
         exact_controls=verified,tests_passed=15,protected=protected(),prospective_sources=len(prospective),
         historical_independence=False,orders_sent=0,live_strategy_changed=False,reserved_raw_read=False))
    save(HERE/'work_state.json',dict(status='progress_remaining_budget_accounts_completed',new_account_replays=12,
         exact_controls=24,tests_passed=15,selection=selection,orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,
         historical_independence=False,goal_complete=False,no_progress_streak=0))
    print('DONE 12 new accounts and 24 exact controls',flush=True)


if __name__=='__main__':main()
