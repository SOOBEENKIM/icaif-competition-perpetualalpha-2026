"""Fixed universe extension of the improved cost-sensitive paired strategy."""
import copy,gzip,json,time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.complementary_alpha_20261017.research import protected
from experiments.conditional_completion_pairs_20261017.research import verify
from experiments.decision_gate_pairs_20261017.portfolio import replay
from experiments.dense_history_pairs_20261017.research import load
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as PAR_INPUT,save,sha
from .model import prepare,states,assert_original_pair_fields

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'
GATE=HERE.parent/'decision_gate_pairs_20261017/artifacts'
SECTOR=HERE.parent/'sector_pairs_20261017/artifacts'
EXT=HERE.parent/'historical_par_extension_20261017/artifacts_v3'


def readrun(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def main():
    OUT.mkdir(exist_ok=False);(OUT/'runs').mkdir()
    spec=json.loads((HERE/'protocol.json').read_text())
    model_spec=json.loads((GATE.parent/'protocol.json').read_text())
    tests=(HERE/'test_results.txt').read_text();assert 'Ran 12 tests' in tests and tests.rstrip().endswith('OK')
    parent=json.loads((GATE/'results.json').read_text());windows=parent['windows'];rows=parent['rows'].copy();control=spec['control']
    cfgs={v:json.loads((ROOT/f'experiments/hourly_refresh_pairs_20261017/{v}_protocol.json').read_text()) for v in ['reference','par']}
    cfgs['par_extended']=copy.deepcopy(cfgs['par'])
    refs={'reference':ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',
          'par_extended':ROOT/'practice_live/updates/20261017T120009814481Z/inputs.npz'}
    expert=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/model_protocol.json').read_text());models={};model_paths=[]
    for pair in expert['pairs']:
        path=ROOT/expert['model_directory']/('_'.join(pair)+'_24.json');models[tuple(pair),24]=json.loads(path.read_text());model_paths.append(path)
    paths={(v,phase,sc):GATE/'runs'/f'{v}-{phase}-{control}-{sc}.json.gz' for v,phase in windows for sc in ['base','stress']}
    sources=json.loads((GATE/'frozen.json').read_text())['sources'].copy()
    files=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',GATE/'results.json',GATE/'frozen.json',GATE.parent/'protocol.json',
           SECTOR.parent/'protocol.json',SECTOR.parent/'work_state.json',SECTOR/'results.json',*model_paths,*refs.values(),*paths.values(),
           *[GATE/f'{v}_gate_state.npz' for v in cfgs],SECTOR/'reference_state.npz',SECTOR/'par_state.npz',
           ROOT/'experiments/remaining_hysteresis_pairs_20261017/artifacts/conclusion.json']
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in files});verify(sources)
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(prospective)
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),previous_goal_turn=spec['previous_goal_turn']))
    rd,_,p=inputs(cfgs['reference']);data={'reference':rd,'par':load(PAR_INPUT/'inputs.npz'),'par_extended':load(EXT/'inputs.npz')}
    controls={v:load(GATE/f'{v}_gate_state.npz') for v in cfgs};mapped={};preparation=[]
    for venue,path in refs.items():
        ref=load(path);features,arrays,mean_records,gate_records=prepare(ref,p['symbols'],spec['pairs'],model_spec,models)
        np.savez_compressed(OUT/f'{venue}_issued_forecasts.npz',**arrays)
        save(OUT/f'{venue}_mean_fits.json',mean_records);save(OUT/f'{venue}_gate_fits.json',gate_records)
        for v in (['reference'] if venue=='reference' else ['par','par_extended']):
            raw,gate,crypto=states(data[v]['times'],ref,features,arrays)
            assert_original_pair_fields(gate,controls[v])
            if v in ['reference','par']:
                old_sector=load(SECTOR/f'{v}_state.npz')
                for key in ['mu','beta','vol']:
                    np.testing.assert_allclose(raw[key],old_sector[key],rtol=1e-12,atol=1e-14,equal_nan=True,err_msg=f'{v}-{key}')
                for key in ['directions','signal_times','pairs','source_prices']:
                    np.testing.assert_array_equal(raw[key],old_sector[key],err_msg=f'{v}-{key}')
            mapped[v]={'ungated':raw,'sector':gate,'crypto':crypto}
            for kind,state in mapped[v].items():np.savez_compressed(OUT/f'{v}_{kind}_state.npz',**state)
        item=dict(venue=venue,mean_fits=len(mean_records),gate_fits=len(gate_records),
             strict_maturity=all(r['last_maturity']<r['fit_time'] for r in mean_records+gate_records),
             max_solver_gap=max(r['solver']['gap'] for r in gate_records),
             original_two_pair_states_exact=True,last_source=int(ref['market_times'][-1]),
             available_source_hours_per_pair=arrays['economic_available'].sum(0).tolist(),
             nonflat_source_hours_per_pair=np.sum(arrays['economic_action']!=0,axis=0).tolist())
        assert item['strict_maturity'];preparation.append(item);print('PREPARED',item,flush=True)
    save(OUT/'preparation.json',preparation)
    original={key:readrun(path) for key,path in paths.items()};verified=[]
    for (venue,phase,sc),old in original.items():
        cfg=copy.deepcopy(cfgs[venue]);cfg['conditional_completion']={'risk_aversion':0.}
        run=replay(data[venue],controls[venue],None,cfg,p,control,sc,old['start'],old['end'])
        for field in ['events','fills','snapshots','metrics','owner_fills']:assert run[field]==old[field],(venue,phase,sc,field)
        verified.append(dict(venue=venue,phase=phase,scenario=sc));print('EXACT',venue,phase,sc,flush=True)
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in rows}
    def record(venue,phase,kind,sc):
        cfg=copy.deepcopy(cfgs[venue]);cfg['conditional_completion']={'risk_aversion':0.};cfg['dense_pairs']=spec['pairs']
        arm=f'first_dense_hourlyhedge_decision_{kind}_24';old=original[venue,phase,sc]
        run=replay(data[venue],mapped[venue][kind],None,cfg,p,arm,sc,old['start'],old['end'])
        with gzip.open(OUT/'runs'/f'{venue}-{phase}-{arm}-{sc}.json.gz','wt') as f:json.dump(run,f,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())
        m=run['metrics'];lookup[venue,phase,arm,sc]=m;rows.append(dict(venue=venue,phase=phase,arm=arm,scenario=sc,metrics=m))
        print('RUN',venue,phase,kind,sc,'PnL',round(m['net_pnl'],6),'Q',m['financial_score'],'completed',m['completed_pairs'],flush=True)
    for kind in [*spec['candidates'],spec['diagnostic']]:
        for sc in ['base','stress']:record('reference','validation',kind,sc)
    ranking=[]
    for kind in spec['candidates']:
        arm=f'first_dense_hourlyhedge_decision_{kind}_24';checks=[]
        for sc in ['base','stress']:
            m=lookup['reference','validation',arm,sc];b=lookup['reference','validation',control,sc]
            checks.append(dict(scenario=sc,pnl_difference=m['net_pnl']-b['net_pnl'],q_difference=m['financial_score']-b['financial_score'],
                above_native_pnl_q=all(m['net_pnl']>lookup['reference','validation',native,sc]['net_pnl'] and m['financial_score']>lookup['reference','validation',native,sc]['financial_score'] for native in ['first_strategy','second_strategy']),
                no_risk_breach=m['risk_breach_bars']==0,completed_pairs=m['completed_pairs']))
        eligible=all(c['pnl_difference']>=-1e-8 and c['q_difference']>=-1e-12 and c['no_risk_breach'] for c in checks)
        eligible &= any(c['pnl_difference']>1e-8 and c['q_difference']>1e-12 for c in checks)
        ranking.append(dict(kind=kind,arm=arm,checks=checks,promote=bool(eligible),
           minimum_pnl_advantage=min(c['pnl_difference'] for c in checks),minimum_q_advantage=min(c['q_difference'] for c in checks)))
    ranking.sort(key=lambda r:(r['minimum_pnl_advantage'],r['minimum_q_advantage'],r['kind']),reverse=True)
    eligible=[r for r in ranking if r['promote']]
    selection=dict(selected=eligible[0]['arm'] if eligible else None,ranking=ranking,
                   frozen_before_later_replays=True,historical_independence=False,sample_flag_is_not_official_requirement=True)
    save(OUT/'selection.json',selection);decision_hash=sha(OUT/'selection.json')
    for v,phase in windows[1:]:
        for kind in [*spec['candidates'],spec['diagnostic']]:
            for sc in ['base','stress']:record(v,phase,kind,sc)
    assert sha(OUT/'selection.json')==decision_hash;verify(sources);verify(prospective)
    save(OUT/'results.json',dict(rows=rows,windows=windows,selection=selection,preparation=preparation,
         new_account_replays=36,exact_controls=verified,tests_passed=12,protected=protected(),prospective_sources=len(prospective),
         orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False))
    save(HERE/'work_state.json',dict(status='progress_fixed_universe_decision_gate_accounts_completed',
         selection=selection,new_accounts=36,exact_controls=12,tests_passed=12,
         next_action='Evaluate only the five allowed old-book tapes and attribute pair-specific cash contributions; retain original candidate if no improvement.',
         orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False,goal_complete=False,no_progress_streak=0))
    print('DONE fixed-universe gate comparison: 36 new accounts, 12 exact controls',flush=True)


if __name__=='__main__':main()
