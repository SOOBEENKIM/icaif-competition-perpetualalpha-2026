"""One frozen model-filter ablation with a paired execution factorial."""
import copy
import gzip
import json
import time
from pathlib import Path
import numpy as np
from experiments.residual_queue_pairs_20261018.portfolio import replay
from experiments.residual_queue_pairs_20261018.research import (
    ROOT,PAR_INPUT,GRAPH,GATE,STABILITY,inputs,load,plain,readrun,save,sha,verify,protected)

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'
QUEUE=HERE.parent/'residual_queue_pairs_20261018/artifacts'
SECTOR=HERE.parent/'decision_sector_pairs_20261018/artifacts'


def main():
    OUT.mkdir(exist_ok=False);(OUT/'runs').mkdir()
    spec=json.loads((HERE/'protocol.json').read_text())
    parent=json.loads((QUEUE/'results.json').read_text());windows=parent['windows']
    arms={'first_strategy','second_strategy',spec['retained'],spec['gated_control'],spec['queued_control']}
    rows=[r for r in parent['rows'] if r['arm'] in arms]
    cfgs={v:json.loads((ROOT/f'experiments/hourly_refresh_pairs_20261017/{v}_protocol.json').read_text()) for v in ['reference','par']}
    cfgs['par_extended']=copy.deepcopy(cfgs['par'])
    graph_spec=json.loads((GRAPH.parent/'protocol.json').read_text())
    formation=json.loads((GRAPH/'formation.json').read_text())
    paths={(v,phase,sc,queue):(QUEUE if queue else GRAPH)/'runs'/f"{v}-{phase}-{spec['queued_control' if queue else 'gated_control']}-{sc}.json.gz"
           for v,phase in windows for sc in ['base','stress'] for queue in [False,True]}
    rawpaths={v:SECTOR/f'{v}_ungated_state.npz' for v in cfgs}
    gatedpaths={v:GRAPH/f'{v}_state.npz' for v in cfgs}
    refpath=ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz'
    extpath=ROOT/'experiments/historical_par_extension_20261017/artifacts_v3/inputs.npz'
    prior_manifest=QUEUE/'conclusion_frozen.json'
    sources=json.loads(prior_manifest.read_text())['sources'].copy()
    files=[*HERE.glob('*.py'),HERE/'protocol.json',prior_manifest,QUEUE/'results.json',
        GRAPH/'formation.json',GRAPH.parent/'protocol.json',refpath,extpath,PAR_INPUT/'inputs.npz',
        *paths.values(),*rawpaths.values(),*gatedpaths.values()]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in files});verify(sources)
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(prospective)
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),
        previous_goal_turn_classification=spec['previous_goal_turn_classification']))
    ref,_,trading=inputs(cfgs['reference']);data=dict(reference=ref,par=load(PAR_INPUT/'inputs.npz'),par_extended=load(extpath))
    gated={v:load(p) for v,p in gatedpaths.items()};raw={};state_checks=[]
    mask=np.asarray(formation['mask'],dtype=bool)
    for v,path in rawpaths.items():
        state=load(path)
        assert 'economic_action' not in state and 'economic_available' not in state
        for k in state:np.testing.assert_array_equal(state[k],gated[v][k],err_msg=v+':'+k)
        original_directions=state['directions'].copy();state['directions'][:,~mask]=0
        np.testing.assert_array_equal(state['directions'][:,mask],original_directions[:,mask])
        assert [[trading['symbols'][int(j)] for j in ids] for ids in state['pairs']]==graph_spec['pairs']
        raw[v]=state;np.savez_compressed(OUT/f'{v}_direct_state.npz',**state)
        state_checks.append(dict(venue=v,common_fields_exact_before_mask=True,selected_directions_exact=True,
                                 excluded_directions_zero=True,selected_pairs=formation['selected_pairs']))
    originals={k:readrun(p) for k,p in paths.items()};exact=[]

    def record(v,phase,sc,queue,direct):
        c=copy.deepcopy(cfgs[v]);c['conditional_completion']={'risk_aversion':0.};c['dense_pairs']=graph_spec['pairs']
        if queue:c['residual_queue']=True
        old=originals[v,phase,sc,queue]
        arm=spec['candidate' if queue else 'diagnostic'] if direct else spec['queued_control' if queue else 'gated_control']
        run=plain(replay(data[v],raw[v] if direct else gated[v],None,c,trading,arm,sc,old['start'],old['end']))
        if not direct:
            assert run==old,(v,phase,sc,queue)
            exact.append(dict(venue=v,phase=phase,scenario=sc,queue=queue))
            print('EXACT',v,phase,sc,queue,flush=True)
        else:
            with gzip.open(OUT/'runs'/f'{v}-{phase}-{arm}-{sc}.json.gz','wt') as f:json.dump(run,f,allow_nan=False)
            m=run['metrics'];rows.append(dict(venue=v,phase=phase,arm=arm,scenario=sc,metrics=m))
            print('RUN',v,phase,sc,'queue',queue,'PnL',round(m['net_pnl'],6),'Q',m['financial_score'],'pairs',m['completed_pairs'],flush=True)

    def window(v,phase):
        for sc in ['base','stress']:
            for queue in [False,True]:
                record(v,phase,sc,queue,False);record(v,phase,sc,queue,True)

    window('reference','validation')
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in rows};checks=[]
    for sc in ['base','stress']:
        new=lookup['reference','validation',spec['candidate'],sc]
        for control in [spec['retained'],spec['queued_control']]:
            old=lookup['reference','validation',control,sc]
            checks.append(dict(scenario=sc,control=control,pnl_difference=new['net_pnl']-old['net_pnl'],
                q_difference=new['financial_score']-old['financial_score'],no_risk_breach=new['risk_breach_bars']==0,
                above_originals=all(new['net_pnl']>lookup['reference','validation',n,sc]['net_pnl'] and
                    new['financial_score']>lookup['reference','validation',n,sc]['financial_score'] for n in ['first_strategy','second_strategy'])))
    selected=(all(x['pnl_difference']>=-1e-8 and x['q_difference']>=-1e-12 and x['no_risk_breach'] and x['above_originals'] for x in checks)
              and any(x['pnl_difference']>1e-8 and x['q_difference']>1e-12 for x in checks))
    selection=dict(selected=spec['candidate'] if selected else spec['retained'],promote_direct=bool(selected),
                   checks=checks,frozen_before_later_replays=True,historical_independence=False)
    save(OUT/'selection.json',selection);selection_hash=sha(OUT/'selection.json')
    for v,phase in windows:
        if (v,phase)!=('reference','validation'):window(v,phase)
    assert sha(OUT/'selection.json')==selection_hash;verify(sources);verify(prospective)
    save(OUT/'results.json',dict(rows=rows,windows=windows,selection=selection,state_checks=state_checks,
        new_account_replays=28,exact_controls=exact,reused_queue_tests=12,protected=protected(),
        prospective_sources=len(prospective),orders_sent=0,live_strategy_changed=False,
        reserved_raw_read=False,historical_independence=False,new_model_fits=0))
    print('DONE 28 new direct accounts, 28 exact gated controls; 0 new model fits',flush=True)


if __name__=='__main__':main()
