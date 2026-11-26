"""Frozen causal tree calibration and full after-cost account comparisons."""
import gzip,json,time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.complementary_alpha_20261017.research import protected
from experiments.hourly_refresh_pairs_20261017.policy import replay
from experiments.dense_history_pairs_20261017.research import load
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,save,sha,ts
from .model import calibrate,mapped_states

HERE=Path(__file__).resolve().parent
PARENT=ROOT/'experiments/daily_tree_pairs_20261017'
HOURLY=ROOT/'experiments/hourly_refresh_pairs_20261017'


def provenance(arrays,records):
    by_fit={(r['pair_index'],r['fit_time']):r for r in records};n=0
    for i,k in np.argwhere(np.isfinite(arrays['prediction_tree'])):
        fitted=int(arrays['fit_stamps'][i,k]);r=by_fit[k,fitted]
        assert r['last_maturity']<fitted<=arrays['times'][i]
        assert r['last_source']+24*3600==r['last_maturity']
        n+=1
    return dict(checked_predictions=n,fit_records=len(records),
        historical_source_time_reconstruction=True,actually_live_broadcast_in_past=False)


def main():
    out=HERE/'artifacts';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    spec=json.loads((HERE/'protocol.json').read_text())
    cfgs={v:json.loads((HOURLY/f'{v}_protocol.json').read_text()) for v in ['reference','par']}
    sources=json.loads((PARENT/'artifacts/frozen.json').read_text())['sources'].copy()
    sources.update(json.loads((PARENT/'artifacts/trained_model_manifest.json').read_text())['sources'])
    reserved=ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json'
    sources.update(json.loads(reserved.read_text())['sources'])
    paths=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',reserved,
           PARENT/'artifacts/results.json',PARENT/'artifacts/trained_model_manifest.json']
    paths += [ROOT/x['path'] for x in spec['prior_audit']]
    for venue in cfgs:
        paths += [PARENT/f'artifacts/{venue}_forecasts.npz',PARENT/f'artifacts/{venue}_fit_audit.json',
                  *[PARENT/f'artifacts/{venue}_{kind}_24_state.npz' for kind in ['direct','tree','mix']]]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),
        previous_goal_turn='progress_daily_tree_mixture_32_replays_rejected',historical_independence=False,orders_sent=0))
    rd,_,p=inputs(cfgs['reference']);data={'reference':rd,'par':load(INPUT/'inputs.npz')}
    parent=json.loads((PARENT/'artifacts/results.json').read_text());rows=parent['rows'].copy();assert len(rows)==80
    states={};preparation=[]
    for venue in cfgs:
        arrays=load(PARENT/f'artifacts/{venue}_forecasts.npz')
        original=load(PARENT/f'artifacts/{venue}_tree_24_state.npz')
        audit=provenance(arrays,json.loads((PARENT/f'artifacts/{venue}_fit_audit.json').read_text()))
        forecasts,records=calibrate(arrays,spec)
        computed=mapped_states(data[venue]['times'],None,original,forecasts,arrays)
        for kind,state in computed.items():
            np.savez_compressed(out/f'{venue}_{kind}_24_state.npz',**state);states[venue,kind]=state
        np.savez_compressed(out/f'{venue}_forecasts.npz',**arrays,**{f'prediction_{k}':v for k,v in forecasts.items()})
        save(out/f'{venue}_calibration_audit.json',records)
        preparation.append(dict(venue=venue,provenance=audit,calibrations=len(records),
            available_pair_sources={k:int(v['directions'].sum()) for k,v in computed.items()}))
        print('PREPARED',preparation[-1],flush=True)
    save(out/'preparation.json',preparation)
    verified=[]
    for venue,phase,base,sc in [('reference','validation','first','base'),('reference','comparison','second','stress'),
                              ('par','validation','first','base'),('par','comparison','second','stress')]:
        arm=f'{base}_dense_hourlyhedge_tree_24'
        with gzip.open(PARENT/'artifacts/runs'/f'{venue}-{phase}-{arm}-{sc}.json.gz','rt') as f:old=json.load(f)
        control=load(PARENT/f'artifacts/{venue}_tree_24_state.npz')
        run=replay(data[venue],control,None,cfgs[venue],p,arm,sc,old['start'],old['end'])
        for key in ['fills','events','metrics','snapshots']:assert run[key]==old[key],(venue,phase,key)
        verified.append(dict(venue=venue,phase=phase,arm=arm,scenario=sc));print('EXACT',verified[-1],flush=True)
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in rows}
    arms=[f'{base}_dense_hourlyhedge_{kind}_24' for base in ['first','second'] for kind in spec['candidates']]
    def record(venue,phase,arm,sc):
        cfg=cfgs[venue];kind=arm.split('_')[-2]
        start=ts(cfg['validation_start'] if phase=='validation' else cfg['comparison_start'])
        end=ts(cfg['comparison_start'] if phase=='validation' else cfg['end'])
        run=replay(data[venue],states[venue,kind],None,cfg,p,arm,sc,start,end)
        with gzip.open(out/'runs'/f'{venue}-{phase}-{arm}-{sc}.json.gz','wt') as f:
            json.dump(run,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        m=run['metrics'];lookup[venue,phase,arm,sc]=m
        raw_kind={'ctree':'tree','cmix':'mix'}[kind];matched=arm.replace(f'_{kind}_24',f'_{raw_kind}_24')
        direct=arm.replace(f'_{kind}_24','_direct_24');old=lookup[venue,phase,matched,sc];baseline=lookup[venue,phase,direct,sc]
        rows.append(dict(venue=venue,phase=phase,arm=arm,scenario=sc,metrics=m,matched=matched,
            delta_raw_pnl=m['net_pnl']-old['net_pnl'],delta_raw_q=m['financial_score']-old['financial_score'],
            delta_direct_pnl=m['net_pnl']-baseline['net_pnl'],delta_direct_q=m['financial_score']-baseline['financial_score']))
        print(venue,phase,arm,sc,'PnL',round(m['net_pnl'],3),'Q',round(m['financial_score'],8),
              'pairs',m['completed_pairs'],'delta_direct',round(m['net_pnl']-baseline['net_pnl'],3),flush=True)
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
            b=lookup['reference','validation',arm.rsplit('_',2)[0]+'_direct_24',sc]
            control_checks.extend([m['net_pnl']>b['net_pnl'],m['financial_score']>b['financial_score']])
        ranking.append(dict(arm=arm,eligible=all(checks) and all(control_checks),native_eligible=all(checks),
            improves_direct_both_scenarios=all(control_checks),minimum_native_pnl_advantage=min(margins)))
    ranking.sort(key=lambda r:(-r['eligible'],-r['minimum_native_pnl_advantage'],r['arm']))
    selection=dict(selected=ranking[0]['arm'],eligible=ranking[0]['eligible'],ranking=ranking,
        frozen_before_later_account_replay=True,historical_independence=False)
    save(out/'selection.json',selection);selected_hash=sha(out/'selection.json')
    for venue,phase in [('reference','comparison'),('par','validation'),('par','comparison')]:
        for arm in arms:
            for sc in cfgs[venue]['scenarios']:record(venue,phase,arm,sc)
    assert sha(out/'selection.json')==selected_hash
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path
    save(out/'results.json',dict(rows=rows,selection=selection,new_account_replays=32,exact_control_replays=verified,
        reused_control_accounts=80,protected=protected(),orders_sent=0,historical_independence=False,automatic_live_replacement=False))
    print('DONE 32 accounts; selected',selection['selected'],'eligible',selection['eligible'],flush=True)


if __name__=='__main__':main()
