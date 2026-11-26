"""One predeclared 72h cost-sensitive gate versus the frozen 24h candidate."""
import copy
import gzip
import json
import time
from pathlib import Path
import numpy as np

from experiments.adaptive_residual_pairs_20261017.model import frozen_predictions
from experiments.dense_history_pairs_20261017.model import design, labels, map_state
from experiments.rolling_feature_pairs_20261017.model import rolling as direct_rolling
from experiments.decision_gate_pairs_20261017.model import rolling, map_gate
from experiments.decision_gate_pairs_20261017.portfolio import replay
from experiments.conditional_completion_pairs_20261017.research import verify
from experiments.complementary_alpha_20261017.research import protected
from experiments.par_direct_forecast_20261017.data import ROOT, save, sha

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'
RELEASE=ROOT/'experiments/candidate_release_20261018/artifacts'
BUNDLE=ROOT/'deliverables/sfinx_pairs_candidate_20261018_final'
GATE=ROOT/'experiments/decision_gate_pairs_20261017/artifacts'


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def plain(value):
    return json.loads(json.dumps(value,allow_nan=False,
        default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()))


def readrun(path):
    with gzip.open(path,'rt') as handle:return json.load(handle)


def main():
    OUT.mkdir(exist_ok=False);(OUT/'runs').mkdir()
    spec=json.loads((HERE/'protocol.json').read_text())
    assert (HERE/'test_results.txt').read_text().rstrip().endswith('OK')
    bundle=json.loads((BUNDLE/'config.json').read_text())
    parent=json.loads((RELEASE/'reproduction_final/results.json').read_text())
    rows=copy.deepcopy(parent['rows']);windows=list(dict.fromkeys((r['venue'],r['phase']) for r in rows))
    cfgs=copy.deepcopy(bundle['configs'])
    paths={v:BUNDLE/'data'/f'{v}.npz' for v in bundle['venues']}
    oldpaths={v:BUNDLE/'models'/f'{v}_gate_state.npz' for v in paths}
    reference_paths={'reference':BUNDLE/'data/reference_training.npz','par_reference':BUNDLE/'data/par_reference_training.npz'}
    model_paths=[ROOT/'experiments/dense_history_pairs_20261017/artifacts/models'/('_'.join(pair)+'_72.json') for pair in spec['pairs']]
    controls={(v,phase,sc):RELEASE/'reproduction_final/runs'/f'{v}-{phase}-{spec["retained"]}-{sc}.json.gz'
              for v,phase in windows for sc in ['base','stress']}
    sources=json.loads((RELEASE/'conclusion_frozen.json').read_text())['sources'].copy()
    files=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',RELEASE/'conclusion_frozen.json',
           RELEASE/'reproduction_final/results.json',*paths.values(),*oldpaths.values(),*reference_paths.values(),
           *model_paths,*controls.values(),BUNDLE/'config.json']
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in files});verify(sources)
    plans=[ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json',
           ROOT/'experiments/decision_gate_publication_20261018/artifacts/plan.json']
    frozen_plans={str(p.relative_to(ROOT)):sha(p) for p in plans}
    for path in plans:verify(json.loads(path.read_text())['sources'])
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),plans=frozen_plans,
                               previous_goal_turn_classification=spec['previous_goal_turn_classification']))
    data={v:load(p) for v,p in paths.items()};oldstates={v:load(p) for v,p in oldpaths.items()}
    learning={key:value for key,value in bundle['gate_spec'].items() if type(value) in (int,float)}
    learning['horizon']=72
    learning.update(training_problem='Fixed-source 72h normalized pair price return, unchanged 38bp proxy cost.',
                    purpose='Single predeclared 72h horizon extension of retained model form.')
    save(OUT/'learning.json',learning)
    models={}
    for pair,path in zip(spec['pairs'],model_paths):
        record=json.loads(path.read_text());assert record['horizon']==72
        assert record['last_training_maturity']<record['first_calibration_source']<record['fit_available']
        assert record['last_calibration_maturity']<record['fit_available']
        models[tuple(pair),72]=record
    gated={};direct={};preparation=[]
    for name,path in reference_paths.items():
        ref=load(path);times=ref['market_times'];assert all(m['fit_available']<times[0] for m in models.values())
        features=design(times,ref['close'],ref['volume'],ref['trades'],bundle['trading']['symbols'],spec['pairs'])
        raw=frozen_predictions(features,models,spec['pairs'],72)
        target,maturity=labels(times,ref['close'],features,72)
        terminal=np.zeros_like(features['fresh']);terminal[:-72]=features['fresh'][72:]
        mean_spec={**learning,'candidates':['direct'],'diagnostic_variants':[]}
        means,mean_stamps,mean_records=direct_rolling(times,features,target,maturity,raw,terminal,mean_spec)
        actions,scores,available,stamps,gate_records=rolling(times,features,target,maturity,raw,terminal,learning)
        assert all(r['last_maturity']<r['fit_time'] for r in [*mean_records,*gate_records])
        np.savez_compressed(OUT/f'{name}_forecasts.npz',times=times,target=target,maturity=maturity,
            prediction=means['direct'],unit=features['vol']*np.sqrt(72),economic_action=actions,
            economic_available=available,decision_score=scores,fit_stamps=stamps,mean_fit_stamps=mean_stamps,
            fresh=features['fresh'],terminal_fresh=terminal)
        save(OUT/f'{name}_fits.json',dict(direct=mean_records,gate=gate_records))
        for venue in (['reference'] if name=='reference' else ['par','par_extended']):
            baseline=map_state(data[venue]['times'],ref,features,means['direct'],features['pairs'],source_hours=1,lifetime_hours=1)
            state=map_gate(data[venue]['times'],times,actions,available,baseline)
            for key in baseline:np.testing.assert_array_equal(baseline[key],state[key])
            direct[venue]=baseline;gated[venue]=state
            np.savez_compressed(OUT/f'{venue}_direct_state.npz',**baseline)
            np.savez_compressed(OUT/f'{venue}_gate_state.npz',**state)
        preparation.append(dict(source=name,direct_fits=len(mean_records),gate_fits=len(gate_records),
            strict_maturity=True,expert_available_before_source=True,
            maximum_solver_gap=max(r['solver']['gap'] for r in gate_records)))
        print('FIT',name,len(mean_records),len(gate_records),'models',flush=True)
    save(OUT/'preparation.json',preparation)
    oldruns={key:readrun(path) for key,path in controls.items()};exact=[]
    def cfg(venue,new):
        value=copy.deepcopy(cfgs['reference' if venue=='reference' else 'par'])
        value['conditional_completion']={'risk_aversion':0.}
        if new:value['funding_forecast_hours']=72
        return value
    def window(venue,phase):
        for scenario in ['base','stress']:
            old=oldruns[venue,phase,scenario]
            control=plain(replay(data[venue],oldstates[venue],None,cfg(venue,False),bundle['trading'],
                                 spec['retained'],scenario,old['start'],old['end']))
            assert control==old,(venue,phase,scenario)
            exact.append(dict(venue=venue,phase=phase,scenario=scenario))
            for arm,state in [(spec['diagnostic'],direct[venue]),(spec['candidate'],gated[venue])]:
                run=plain(replay(data[venue],state,None,cfg(venue,True),bundle['trading'],arm,
                                 scenario,old['start'],old['end']))
                with gzip.open(OUT/'runs'/f'{venue}-{phase}-{arm}-{scenario}.json.gz','wt') as handle:
                    json.dump(run,handle,allow_nan=False)
                rows.append(dict(venue=venue,phase=phase,scenario=scenario,arm=arm,metrics=run['metrics']))
                print('RUN',venue,phase,scenario,arm,'PnL',round(run['metrics']['net_pnl'],6),
                      'Q',run['metrics']['financial_score'],'pairs',run['metrics']['completed_pairs'],flush=True)
    window('reference','validation')
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in rows};checks=[]
    for sc in ['base','stress']:
        new=lookup['reference','validation',spec['candidate'],sc];old=lookup['reference','validation',spec['retained'],sc]
        checks.append(dict(scenario=sc,pnl_difference=new['net_pnl']-old['net_pnl'],
            q_difference=new['financial_score']-old['financial_score'],no_risk_breach=new['risk_breach_bars']==0,
            above_natives=all(new['net_pnl']>lookup['reference','validation',arm,sc]['net_pnl'] and
                new['financial_score']>lookup['reference','validation',arm,sc]['financial_score'] for arm in ['first_strategy','second_strategy'])))
    promote=all(c['pnl_difference']>=-1e-8 and c['q_difference']>=-1e-12 and c['above_natives'] and c['no_risk_breach'] for c in checks)
    promote&=any(c['pnl_difference']>1e-8 and c['q_difference']>1e-12 for c in checks)
    selection=dict(selected=spec['candidate'] if promote else spec['retained'],promote_72h=bool(promote),checks=checks,
                   frozen_before_later_replays=True,historical_independence=False)
    save(OUT/'selection.json',selection);choice_hash=sha(OUT/'selection.json')
    for venue,phase in windows:
        if (venue,phase)!=('reference','validation'):window(venue,phase)
    assert sha(OUT/'selection.json')==choice_hash;verify(sources);verify(frozen_plans)
    save(OUT/'results.json',dict(rows=rows,windows=windows,selection=selection,preparation=preparation,
        new_account_replays=28,exact_retained_controls=exact,protected=protected(),
        tests_passed=3,orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False))
    print('DONE 28 new accounts, 14 exact retained controls; selection',selection['selected'],flush=True)


if __name__=='__main__':main()
