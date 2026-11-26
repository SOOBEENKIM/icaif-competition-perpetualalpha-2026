"""Matched daily-direct horizons; selection frozen before later account replays."""
import gzip,json,time
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
    spec=json.loads((HERE/'protocol.json').read_text());direct=json.loads((PARENT/'protocol.json').read_text())
    cfgs={v:json.loads((HOURLY/f'{v}_protocol.json').read_text()) for v in ['reference','par']}
    reserved=ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json'
    sources=json.loads(reserved.read_text())['sources'].copy()
    paths=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',reserved,
           ROOT/'experiments/sector_pairs_20261017/model.py',PARENT/'artifacts/results.json']
    paths += [ROOT/x['path'] for x in spec['prior_audit']]
    paths += [PARENT/f'artifacts/{v}_direct_24_state.npz' for v in cfgs]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),
        previous_goal_turn='progress_funding_features_rejected_16_replays',historical_independence=False,orders_sent=0))
    rd,_,p=inputs(cfgs['reference']);data={'reference':rd,'par':load(INPUT/'inputs.npz')}
    refs={'reference':load(ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz'),
          'par':load(ROOT/cfgs['par']['reference_run']/'inputs.npz')}
    parent=json.loads((PARENT/'artifacts/results.json').read_text())
    controls=['first_strategy','second_strategy',*[f'{base}_dense_hourlyhedge_{kind}_24'
              for base in ['first','second'] for kind in ['direct','bias']]]
    rows=[r for r in parent['rows'] if r['arm'] in controls];assert len(rows)==48
    states={};preparation=[]
    for venue,reference in refs.items():
        for h in [spec['control_horizon'],*spec['horizons_hours']]:
            state,records,arrays=infer(data[venue]['times'],reference,p['symbols'],spec['pairs'],direct,h)
            if h==24:
                old=load(PARENT/f'artifacts/{venue}_direct_24_state.npz')
                for k in ['mu','beta','vol']:np.testing.assert_allclose(state[k],old[k],atol=1e-14,rtol=1e-12,equal_nan=True)
                for k in ['directions','signal_times','source_prices']:np.testing.assert_array_equal(state[k],old[k])
            else:
                np.savez_compressed(out/f'{venue}_{h}_state.npz',**state)
                np.savez_compressed(out/f'{venue}_{h}_forecasts.npz',**arrays)
                save(out/f'{venue}_{h}_fit_audit.json',records)
            states[venue,h]=state
            preparation.append(dict(venue=venue,horizon=h,daily_fits=len(records),
                available_pair_sources=int(state['directions'].sum())))
            print('PREPARED',preparation[-1],flush=True)
    save(out/'preparation.json',preparation)
    verified=[]
    for venue,phase,base,sc in [('reference','validation','first','base'),('reference','comparison','second','stress'),
                              ('par','validation','first','base'),('par','comparison','second','stress')]:
        arm=f'{base}_dense_hourlyhedge_direct_24'
        with gzip.open(PARENT/'artifacts/runs'/f'{venue}-{phase}-{arm}-{sc}.json.gz','rt') as f:old=json.load(f)
        run=replay(data[venue],states[venue,24],None,cfgs[venue],p,arm,sc,old['start'],old['end'])
        for k in ['fills','events','metrics','snapshots']:assert run[k]==old[k],(venue,phase,k)
        verified.append(dict(venue=venue,phase=phase,arm=arm,scenario=sc));print('EXACT',verified[-1],flush=True)
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in rows}
    arms=[f'{base}_dense_hourlyhedge_direct_{h}' for base in ['first','second'] for h in spec['horizons_hours']]
    def record(venue,phase,arm,sc):
        cfg=cfgs[venue];h=int(arm.rsplit('_',1)[1])
        start=ts(cfg['validation_start'] if phase=='validation' else cfg['comparison_start'])
        end=ts(cfg['comparison_start'] if phase=='validation' else cfg['end'])
        run=replay(data[venue],states[venue,h],None,cfg,p,arm,sc,start,end)
        for event in run['events']:
            if event['event']=='pair_intent':assert event['horizon']==h
        with gzip.open(out/'runs'/f'{venue}-{phase}-{arm}-{sc}.json.gz','wt') as f:
            json.dump(run,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        m=run['metrics'];lookup[venue,phase,arm,sc]=m
        matched=arm.rsplit('_',1)[0]+'_24';old=lookup[venue,phase,matched,sc]
        rows.append(dict(venue=venue,phase=phase,arm=arm,scenario=sc,metrics=m,matched=matched,
            delta_direct24_pnl=m['net_pnl']-old['net_pnl'],delta_direct24_q=m['financial_score']-old['financial_score']))
        print(venue,phase,arm,sc,'PnL',round(m['net_pnl'],3),'Q',round(m['financial_score'],8),
            'pairs',m['completed_pairs'],'delta24',round(m['net_pnl']-old['net_pnl'],3),flush=True)
    for arm in arms:
        for sc in cfgs['reference']['scenarios']:record('reference','validation',arm,sc)
    ranking=[]
    for arm in arms:
        checks=[];margins=[];control_checks=[]
        for sc in cfgs['reference']['scenarios']:
            m=lookup['reference','validation',arm,sc]
            checks.extend([m['net_pnl']>0,m['completed_pairs']>=3,m['risk_breach_bars']==0])
            for base in ['first_strategy','second_strategy']:
                b=lookup['reference','validation',base,sc]
                checks.extend([m['net_pnl']>b['net_pnl'],m['financial_score']>b['financial_score']]);margins.append(m['net_pnl']-b['net_pnl'])
            old=lookup['reference','validation',arm.rsplit('_',1)[0]+'_24',sc]
            control_checks.extend([m['net_pnl']>old['net_pnl'],m['financial_score']>old['financial_score']])
        ranking.append(dict(arm=arm,eligible=all(checks) and all(control_checks),native_eligible=all(checks),
            improves_direct24_both_scenarios=all(control_checks),minimum_native_pnl_advantage=min(margins)))
    ranking.sort(key=lambda r:(-r['eligible'],-r['minimum_native_pnl_advantage'],r['arm']))
    selection=dict(selected=ranking[0]['arm'],eligible=ranking[0]['eligible'],ranking=ranking,
        frozen_before_later_account_replay=True,historical_independence=False)
    save(out/'selection.json',selection);selected_hash=sha(out/'selection.json')
    for venue,phase in [('reference','comparison'),('par','validation'),('par','comparison')]:
        for arm in arms:
            for sc in cfgs[venue]['scenarios']:record(venue,phase,arm,sc)
    assert sha(out/'selection.json')==selected_hash
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path
    save(out/'results.json',dict(rows=rows,selection=selection,new_account_replays=32,
        exact_control_replays=verified,reused_control_accounts=48,protected=protected(),
        orders_sent=0,historical_independence=False,automatic_live_replacement=False))
    print('DONE 32 accounts; selected',selection['selected'],'eligible',selection['eligible'],flush=True)


if __name__=='__main__':main()
