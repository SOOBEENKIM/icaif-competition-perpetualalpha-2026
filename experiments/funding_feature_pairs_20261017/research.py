"""Funding input ablation, original baselines, frozen-before-later selection."""
import copy,gzip,json,time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.complementary_alpha_20261017.research import protected
from experiments.hourly_refresh_pairs_20261017.policy import replay
from experiments.dense_history_pairs_20261017.research import load
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,save,sha,ts
from .model import infer

HERE=Path(__file__).resolve().parent
PARENT=ROOT/'experiments/rolling_feature_pairs_20261017'
HOURLY=ROOT/'experiments/hourly_refresh_pairs_20261017'


def main():
    out=HERE/'artifacts';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    spec=json.loads((HERE/'protocol.json').read_text())
    direct=json.loads((PARENT/'protocol.json').read_text())
    originals={venue:json.loads((HOURLY/f'{venue}_protocol.json').read_text()) for venue in ['reference','par']}
    cfgs=copy.deepcopy(originals)
    for cfg in cfgs.values():cfg['dense_pairs']=spec['pairs']
    reserved=ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json'
    sources=json.loads(reserved.read_text())['sources'].copy()
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in [*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',reserved,PARENT/'artifacts/results.json']})
    for venue in originals:
        path=PARENT/f'artifacts/{venue}_direct_24_state.npz';sources[str(path.relative_to(ROOT))]=sha(path)
    for rel,digest in sources.items():assert sha(ROOT/rel)==digest,rel
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),
        previous_goal_turn='progress_sector_extension_rejected_16_replays',
        historical_independence=False,orders_sent=0))
    rd,_,p=inputs(originals['reference']);data={'reference':rd,'par':load(INPUT/'inputs.npz')}
    references={'reference':load(ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz'),
                'par':load(ROOT/cfgs['par']['reference_run']/'inputs.npz')}
    parent=json.loads((PARENT/'artifacts/results.json').read_text())
    controls=['first_strategy','second_strategy',*[f'{base}_dense_hourlyhedge_{model}_24' for base in ['first','second'] for model in ['direct','bias']]]
    rows=[r for r in parent['rows'] if r['arm'] in controls];assert len(rows)==48
    states={};preparation=[]
    for venue,reference in references.items():
        state,records,arrays=infer(data[venue]['times'],reference,p['symbols'],spec['pairs'],direct)
        old=load(PARENT/f'artifacts/{venue}_direct_24_state.npz')
        for key in ['beta','vol']:
            np.testing.assert_allclose(state[key][:,:2],old[key],atol=1e-14,rtol=1e-12,equal_nan=True)
        for key in ['signal_times']:np.testing.assert_array_equal(state[key][:,:2],old[key])
        np.testing.assert_array_equal(state['source_prices'],old['source_prices'])
        np.savez_compressed(out/f'{venue}_state.npz',**state)
        np.savez_compressed(out/f'{venue}_forecasts.npz',**arrays)
        save(out/f'{venue}_fit_audit.json',records);states[venue]=state
        preparation.append(dict(venue=venue,matched_original_pair_risk_units=2,daily_fits=len(records),
            available_sources_per_pair=state['directions'].sum(0).tolist()))
        print('PREPARED',venue,preparation[-1],flush=True)
    save(out/'preparation.json',preparation)
    verified=[]
    for venue,phase,base,scenario in [('reference','validation','first','base'),('reference','comparison','second','stress'),
                                       ('par','validation','first','base'),('par','comparison','second','stress')]:
        arm=f'{base}_dense_hourlyhedge_direct_24'
        with gzip.open(PARENT/'artifacts/runs'/f'{venue}-{phase}-{arm}-{scenario}.json.gz','rt') as f:old=json.load(f)
        control_state=load(PARENT/f'artifacts/{venue}_direct_24_state.npz')
        run=replay(data[venue],control_state,None,originals[venue],p,arm,scenario,old['start'],old['end'])
        for key in ['fills','events','metrics','snapshots']:assert run[key]==old[key],(venue,phase,key)
        verified.append(dict(venue=venue,phase=phase,arm=arm,scenario=scenario))
        print('EXACT',venue,phase,arm,scenario,flush=True)
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in rows}
    arms=[f'{base}_dense_hourlyhedge_funding_24' for base in ['first','second']]
    def record(venue,phase,arm,scenario):
        cfg=cfgs[venue];start=ts(cfg['validation_start'] if phase=='validation' else cfg['comparison_start'])
        end=ts(cfg['comparison_start'] if phase=='validation' else cfg['end'])
        run=replay(data[venue],states[venue],None,cfg,p,arm,scenario,start,end)
        with gzip.open(out/'runs'/f'{venue}-{phase}-{arm}-{scenario}.json.gz','wt') as f:
            json.dump(run,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        m=run['metrics'];lookup[venue,phase,arm,scenario]=m
        matched=arm.replace('_funding_24','_direct_24');old=lookup[venue,phase,matched,scenario]
        rows.append(dict(venue=venue,phase=phase,arm=arm,scenario=scenario,metrics=m,matched=matched,
            delta_two_pair_pnl=m['net_pnl']-old['net_pnl'],delta_two_pair_q=m['financial_score']-old['financial_score']))
        print(venue,phase,arm,scenario,'PnL',round(m['net_pnl'],3),'Q',round(m['financial_score'],8),
              'pairs',m['completed_pairs'],'delta_two',round(m['net_pnl']-old['net_pnl'],3),flush=True)
    for arm in arms:
        for scenario in cfgs['reference']['scenarios']:record('reference','validation',arm,scenario)
    ranking=[]
    for arm in arms:
        checks=[];margins=[]
        for scenario in cfgs['reference']['scenarios']:
            m=lookup['reference','validation',arm,scenario]
            checks.extend([m['net_pnl']>0,m['completed_pairs']>=3,m['risk_breach_bars']==0])
            for base in ['first_strategy','second_strategy']:
                b=lookup['reference','validation',base,scenario]
                checks.extend([m['net_pnl']>b['net_pnl'],m['financial_score']>b['financial_score']])
                margins.append(m['net_pnl']-b['net_pnl'])
        native_eligible=all(checks)
        direct_checks=[]
        for scenario in cfgs['reference']['scenarios']:
            m=lookup['reference','validation',arm,scenario]
            b=lookup['reference','validation',arm.replace('_funding_24','_direct_24'),scenario]
            direct_checks.extend([m['net_pnl']>b['net_pnl'],m['financial_score']>b['financial_score']])
        ranking.append(dict(arm=arm,eligible=native_eligible and all(direct_checks),
            native_eligible=native_eligible,improves_price_only_both_scenarios=all(direct_checks),
            minimum_native_pnl_advantage=min(margins)))
    ranking.sort(key=lambda r:(-r['eligible'],-r['minimum_native_pnl_advantage'],r['arm']))
    selection=dict(selected=ranking[0]['arm'],eligible=ranking[0]['eligible'],ranking=ranking,
        frozen_before_later_account_replay=True,historical_independence=False)
    save(out/'selection.json',selection);selected_hash=sha(out/'selection.json')
    for venue,phase in [('reference','comparison'),('par','validation'),('par','comparison')]:
        for arm in arms:
            for scenario in cfgs[venue]['scenarios']:record(venue,phase,arm,scenario)
    assert sha(out/'selection.json')==selected_hash
    for rel,digest in sources.items():assert sha(ROOT/rel)==digest,rel
    save(out/'results.json',dict(rows=rows,selection=selection,new_account_replays=16,
        exact_control_replays=verified,reused_accounts=48,changed_features=['funding_24','funding_168'],
        protected=protected(),orders_sent=0,historical_independence=False,automatic_live_replacement=False))
    print('DONE 16 new accounts; selected',selection['selected'],'eligible',selection['eligible'],flush=True)


if __name__=='__main__':main()
