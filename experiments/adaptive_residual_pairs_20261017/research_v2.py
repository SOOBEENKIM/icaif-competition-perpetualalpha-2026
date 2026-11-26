"""Predeclared residual-calibration arms with unchanged costed execution."""
import gzip,json,time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs as reference_inputs
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,ts,save,sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.dense_history_pairs_20261017.model import design,map_state
from experiments.dense_history_pairs_20261017.portfolio import replay
from experiments.dense_history_pairs_20261017.research import load,forecast_audit
from .model_v2 import build

HERE=Path(__file__).resolve().parent;PREVIOUS=HERE.parent/'dense_history_pairs_20261017'


def main():
    out=HERE/'artifacts_v2';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    cfgs={v:json.loads((HERE/f'{v}_protocol.json').read_text()) for v in ['reference','par']}
    spec=json.loads((HERE/'model_protocol.json').read_text())
    old_frozen=json.loads((PREVIOUS/'artifacts/frozen.json').read_text())
    for rel,digest in old_frozen['sources'].items():assert sha(ROOT/rel)==digest,rel
    old_results=json.loads((PREVIOUS/'artifacts/results.json').read_text());old_by_key={}
    for row in old_results['rows']:old_by_key[tuple(row[k] for k in ['venue','phase','arm','scenario'])]=row
    paths=[*HERE.glob('*.py'),*HERE.glob('*protocol.json'),PREVIOUS/'artifacts/frozen.json',PREVIOUS/'artifacts/results.json',
        *[PREVIOUS/n for n in ['portfolio.py','policy.py','completion.py','model.py','research.py']],
        *[ROOT/f'experiments/shared_execution_pairs_20261017/{n}' for n in ['account_v2.py','proposals.py']],
        INPUT/'inputs.npz',ROOT/cfgs['reference']['reference_run']/'inputs.npz',
        ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',ROOT/'experiments/risk_preserving_20261016/artifacts/targets.npz']
    models={}
    for pair in spec['pairs']:
        for h in spec['horizons']:
            path=ROOT/spec['model_directory']/('_'.join(pair)+f'_{h}.json');paths.append(path)
            record=json.loads(path.read_text());assert record['fit_available']==ts(spec['model_available'])
            models[tuple(pair),h]=record
    for venue,cfg in cfgs.items():
        previous=json.loads((PREVIOUS/f'{venue}_protocol.json').read_text())
        excluded={'controls','candidates','experiment','selection'}
        assert {k:v for k,v in cfg.items() if k not in excluded}=={k:v for k,v in previous.items() if k not in excluded}
        for phase in ['validation','comparison']:
            for arm in cfg['controls']:
                for scenario in cfg['scenarios']:
                    paths.append(PREVIOUS/f'artifacts/runs/{venue}-{phase}-{arm}-{scenario}.json.gz')
        for kind in ['zero','ridge']:
            for h in spec['horizons']:paths.append(PREVIOUS/f'artifacts/{venue}_{kind}_{h}_state.npz')
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),
        historical_independence=False,orders_sent=0,previous_goal_turn='progress_dense_forecasts_replayed_and_costs_attributed'))
    rd,_,protocol=reference_inputs(cfgs['reference']);pd=load(INPUT/'inputs.npz');data={'reference':rd,'par':pd}
    references={'reference':load(ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz'),
        'par':load(ROOT/cfgs['par']['reference_run']/'inputs.npz')}
    states={};fits=[];forecast_audits={}
    for venue,r in references.items():
        cfg=cfgs[venue];assert r['market_times'][0]>ts(spec['model_available'])
        d=design(r['market_times'],r['close'],r['volume'],r['trades'],protocol['symbols'],cfg['dense_pairs'])
        for h in spec['horizons']:
            forecasts,records=build(r,d,models,cfg['dense_pairs'],h,spec)
            fits.extend(dict(venue=venue,horizon=h,**x) for x in records)
            for kind,pp in forecasts.items():
                state=map_state(data[venue]['times'],r,d,pp,d['pairs'],cfg['dense_policy']['decision_hours'],cfg['dense_policy']['source_lifetime_hours'])
                if kind in ['zero','ridge']:
                    old=load(PREVIOUS/f'artifacts/{venue}_{kind}_{h}_state.npz')
                    for k in old:np.testing.assert_array_equal(old[k],state[k],err_msg=f'{venue}-{kind}-{h}-{k}')
                else:
                    states[venue,kind,h]=state;np.savez_compressed(out/f'{venue}_{kind}_{h}_state.npz',**state)
            for phase in ['validation','comparison']:
                begin=ts(cfg['validation_start'] if phase=='validation' else cfg['comparison_start'])
                end=ts(cfg['comparison_start'] if phase=='validation' else cfg['end'])
                forecast_audits[f'{venue}_{phase}_{h}']=forecast_audit(r,d,forecasts,h,begin,end)
            print('CALIBRATE',venue,h,'decisions',len(records),'last maturity',max(x['last_maturity'] for x in records),flush=True)
    save(out/'fits.json',fits)
    rows=[];runs={};reused=[]
    def control_rows(venue,phase):
        cfg=cfgs[venue]
        for arm in cfg['controls']:
            for scenario in cfg['scenarios']:
                key=(venue,phase,arm,scenario);row=old_by_key[key]
                path=PREVIOUS/'artifacts/runs'/('-'.join(key)+'.json.gz')
                with gzip.open(path,'rt') as f:old=json.load(f)
                assert old['metrics']==row['metrics']
                rows.append(row);runs[key]=old
                reused.append(dict(venue=venue,phase=phase,arm=arm,scenario=scenario,path=str(path.relative_to(ROOT)),sha256=sha(path)))
    def record(venue,phase,arm,scenario):
        cfg=cfgs[venue];_,_,kind,h=arm.split('_');state=states[venue,kind,int(h)]
        start=cfg['validation_start'] if phase=='validation' else cfg['comparison_start'];end=cfg['comparison_start'] if phase=='validation' else cfg['end']
        run=replay(data[venue],state,None,cfg,protocol,arm,scenario,ts(start),ts(end));key=(venue,phase,arm,scenario);runs[key]=run
        with gzip.open(out/'runs'/('-'.join(key)+'.json.gz'),'wt') as f:
            json.dump(run,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        rows.append(dict(venue=venue,phase=phase,arm=arm,scenario=scenario,metrics=run['metrics']))
        m=run['metrics'];print('-'.join(key),'PnL',round(m['net_pnl'],3),'Q',round(m['financial_score'],7),
            'pairs',m['completed_pairs'],'aborts',m['aborted_pairs'],flush=True)
    control_rows('reference','validation')
    for arm in cfgs['reference']['candidates']:
        for scenario in cfgs['reference']['scenarios']:record('reference','validation',arm,scenario)
    ranking=[]
    for arm in cfgs['reference']['candidates']:
        checks=[];differences=[]
        for scenario in cfgs['reference']['scenarios']:
            m=runs['reference','validation',arm,scenario]['metrics']
            checks.extend([m['net_pnl']>0,m['completed_pairs']>=3,m['risk_breach_bars']==0])
            for base in ['first_strategy','second_strategy']:
                b=runs['reference','validation',base,scenario]['metrics'];differences.append(m['net_pnl']-b['net_pnl'])
                checks.extend([m['net_pnl']>b['net_pnl'],m['financial_score']>b['financial_score']])
        ranking.append(dict(arm=arm,eligible=all(checks),minimum_pnl_advantage=min(differences)))
    ranking.sort(key=lambda x:(-x['eligible'],-x['minimum_pnl_advantage'],x['arm']))
    selection=dict(selected=ranking[0]['arm'],eligible=ranking[0]['eligible'],ranking=ranking,
        frozen_before_later_account_comparison=True,historical_independence=False)
    save(out/'selection.json',selection);selection_hash=sha(out/'selection.json')
    for venue,phase in [('reference','comparison'),('par','validation'),('par','comparison')]:
        control_rows(venue,phase)
        for arm in cfgs[venue]['candidates']:
            for scenario in cfgs[venue]['scenarios']:record(venue,phase,arm,scenario)
    assert sha(out/'selection.json')==selection_hash
    for rel,digest in sources.items():assert sha(ROOT/rel)==digest,rel
    save(out/'control_reuse.json',reused);save(out/'forecast_audits.json',forecast_audits)
    save(out/'results.json',dict(rows=rows,selection=selection,actual_replays=len(rows)-len(reused),reused_control_accounts=len(reused),
        frozen_forecasts_match_previous_exactly=True,execution_source_unchanged=True,protected=protected(),
        orders_sent=0,automatic_live_replacement=False,historical_independence=False))
    print('DONE',len(rows)-len(reused),'new replays;',len(reused),'verified reused controls; eligible',selection['eligible'],flush=True)


if __name__=='__main__':main()
