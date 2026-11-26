"""Frozen chronological comparison of residual liquidation queues on two fixed universes."""
import copy
import gzip
import json
import time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.complementary_alpha_20261017.research import protected
from experiments.conditional_completion_pairs_20261017.research import verify
from experiments.dense_history_pairs_20261017.research import load
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as PAR_INPUT,save,sha
from experiments.formation_graph_pairs_20261018.research import plain,readrun
from .portfolio import replay

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'
GATE=HERE.parent/'decision_gate_pairs_20261017/artifacts'
GRAPH=HERE.parent/'formation_graph_pairs_20261018/artifacts'
STABILITY=HERE.parent/'lifecycle_stability_pairs_20261018/artifacts'


def main():
    OUT.mkdir(exist_ok=False);(OUT/'runs').mkdir()
    spec=json.loads((HERE/'protocol.json').read_text())
    tests=(HERE/'test_results.txt').read_text()
    assert 'Ran 12 tests' in tests and tests.rstrip().endswith('OK')
    parent=json.loads((GRAPH/'results.json').read_text());windows=parent['windows']
    retained=spec['universes']['two'];arms={'first_strategy','second_strategy',*spec['universes'].values()}
    rows=[r for r in parent['rows'] if r['arm'] in arms]
    cfgs={v:json.loads((ROOT/f'experiments/hourly_refresh_pairs_20261017/{v}_protocol.json').read_text()) for v in ['reference','par']}
    cfgs['par_extended']=copy.deepcopy(cfgs['par'])
    graph_spec=json.loads((GRAPH.parent/'protocol.json').read_text())
    statepaths={v:{'two':GATE/f'{v}_gate_state.npz','graph':GRAPH/f'{v}_state.npz'} for v in cfgs}
    paths={}
    for v,phase in [*windows,['reference','continuous']]:
        for sc in ['base','stress']:
            for universe,arm in spec['universes'].items():
                folder=GRAPH if universe=='graph' else STABILITY if phase=='continuous' else GATE
                paths[v,phase,sc,universe]=folder/'runs'/f'{v}-{phase}-{arm}-{sc}.json.gz'
    for sc in ['base','stress']:
        for arm in ['first_strategy','second_strategy',retained]:
            path=STABILITY/'runs'/f'reference-continuous-{arm}-{sc}.json.gz'
            run=readrun(path)
            rows.append(dict(venue='reference',phase='continuous',arm=arm,scenario=sc,metrics=run['metrics']))
    refpath=ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz'
    extpath=ROOT/'experiments/historical_par_extension_20261017/artifacts_v3/inputs.npz'
    prior_manifest=HERE.parent/'opportunity_switch_pairs_20261018/artifacts/conclusion_frozen.json'
    sources=json.loads(prior_manifest.read_text())['sources'].copy()
    files=[prior_manifest,*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',GRAPH/'conclusion_frozen_v2.json',
           GRAPH/'results.json',GRAPH.parent/'protocol.json',refpath,extpath,PAR_INPUT/'inputs.npz',
           *paths.values(),*[p for v in statepaths.values() for p in v.values()]]
    files += [STABILITY/'runs'/f'reference-continuous-{arm}-{sc}.json.gz' for arm in ['first_strategy','second_strategy'] for sc in ['base','stress']]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in files});verify(sources)
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(prospective)
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),
         previous_goal_turn_classification=spec['previous_goal_turn_classification']))
    ref,_,trading=inputs(cfgs['reference'])
    data=dict(reference=ref,par=load(PAR_INPUT/'inputs.npz'),par_extended=load(extpath))
    states={v:{u:load(p) for u,p in paths_.items()} for v,paths_ in statepaths.items()}
    original={k:readrun(path) for k,path in paths.items()};exact=[]

    def record(v,phase,sc,u,enabled):
        c=copy.deepcopy(cfgs[v]);c['conditional_completion']={'risk_aversion':0.}
        if u=='graph':c['dense_pairs']=graph_spec['pairs']
        if enabled:c['residual_queue']=True
        arm=spec['candidates' if enabled else 'universes'][u];old=original[v,phase,sc,u]
        result=plain(replay(data[v],states[v][u],None,c,trading,arm,sc,old['start'],old['end']))
        if not enabled:
            for key in old:assert result[key]==old[key],(v,phase,sc,u,key)
            exact.append(dict(venue=v,phase=phase,scenario=sc,universe=u))
            print('EXACT',v,phase,sc,u,flush=True)
        else:
            with gzip.open(OUT/'runs'/f'{v}-{phase}-{arm}-{sc}.json.gz','wt') as f:json.dump(result,f,allow_nan=False)
            m=result['metrics'];rows.append(dict(venue=v,phase=phase,arm=arm,scenario=sc,metrics=m))
            print('RUN',v,phase,sc,u,'PnL',round(m['net_pnl'],6),'Q',m['financial_score'],
                  'handoffs',m['residual_handoffs'],'flat',m['residual_episodes_closed'],flush=True)

    def window(v,phase):
        for sc in ['base','stress']:
            for u in spec['universes']:
                record(v,phase,sc,u,False);record(v,phase,sc,u,True)

    window('reference','validation')
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in rows}
    ranking=[]
    for u,arm in spec['candidates'].items():
        checks=[]
        for sc in ['base','stress']:
            m=lookup['reference','validation',arm,sc]
            for baseline in dict.fromkeys([retained,spec['universes'][u]]):
                b=lookup['reference','validation',baseline,sc]
                checks.append(dict(scenario=sc,baseline=baseline,pnl_difference=m['net_pnl']-b['net_pnl'],
                    q_difference=m['financial_score']-b['financial_score'],no_risk_breach=m['risk_breach_bars']==0,
                    above_originals=all(m['net_pnl']>lookup['reference','validation',n,sc]['net_pnl'] and
                      m['financial_score']>lookup['reference','validation',n,sc]['financial_score'] for n in ['first_strategy','second_strategy'])))
        ok=(all(c['pnl_difference']>=-1e-8 and c['q_difference']>=-1e-12 and c['no_risk_breach'] and c['above_originals'] for c in checks)
            and any(c['pnl_difference']>1e-8 and c['q_difference']>1e-12 for c in checks))
        ranking.append(dict(universe=u,arm=arm,qualifies=bool(ok),checks=checks,
            minimum_pnl_advantage=min(x['pnl_difference'] for x in checks),minimum_q_advantage=min(x['q_difference'] for x in checks)))
    ranking.sort(key=lambda r:(r['minimum_pnl_advantage'],r['minimum_q_advantage'],r['arm']),reverse=True)
    eligible=[r for r in ranking if r['qualifies']]
    selection=dict(selected=eligible[0]['arm'] if eligible else retained,ranking=ranking,
                   frozen_before_later_replays=True,historical_independence=False)
    save(OUT/'selection.json',selection);frozen_selection=sha(OUT/'selection.json')
    for v,phase in [*windows,['reference','continuous']]:
        if (v,phase)!=('reference','validation'):window(v,phase)
    assert sha(OUT/'selection.json')==frozen_selection;verify(sources);verify(prospective)
    save(OUT/'results.json',dict(rows=rows,windows=[*windows,['reference','continuous']],selection=selection,
        new_account_replays=28,exact_controls=exact,tests_passed=12,protected=protected(),
        prospective_sources=len(prospective),orders_sent=0,live_strategy_changed=False,
        reserved_raw_read=False,historical_independence=False))
    print('DONE 28 new accounts and 28 exact no-queue controls',flush=True)


if __name__=='__main__':main()
