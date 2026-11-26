"""Separate older training, frozen selection, and reused-history account tests."""
import gzip,json,time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs as reference_inputs
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,ts,save,sha
from experiments.complementary_alpha_20261017.research import protected
from .model import design,labels,fit,predict,map_state
from .portfolio import replay

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'shared_execution_pairs_20261017'
OLDER=HERE.parent/'older_reference_data_20261017'


def load(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}


def older_pair(pair):
    legs=[load(OLDER/(symbol+'USDT.npz')) for symbol in pair]
    times=np.intersect1d(legs[0]['market_times'],legs[1]['market_times'])
    if not np.all(np.diff(times)==3600):raise ValueError('Missing older source hour; do not interpolate')
    indexes=[np.searchsorted(leg['market_times'],times) for leg in legs]
    return dict(market_times=times,**{key:np.column_stack([leg[key][ix] for leg,ix in zip(legs,indexes)])
        for key in ['close','volume','trades']})


def one_pair(d,pair_ids):
    # Never reuse column numbers from another pair universe or venue.
    identities=[tuple(int(x) for x in p) for p in d['pairs']]
    k=identities.index(tuple(pair_ids))
    return dict(names=d['names'],pairs=[list(pair_ids)],
        **{key:d[key][:,k:k+1].copy() for key in ['x','beta','vol','fresh']})


def forecast_audit(reference,d,predictions,h,start,end):
    y,maturity=labels(reference['market_times'],reference['close'],d,h)
    t=reference['market_times'];valid=(t>=start)&(maturity<end)&((t-start)%(h*3600)==0)
    actual=y*d['vol']*np.sqrt(h);result=[]
    for k,ids in enumerate(d['pairs']):
        rows=valid&np.isfinite(actual[:,k])&d['fresh'][:,k]
        for kind,pp in predictions.items():
            use=rows&np.isfinite(pp[:,k]);yy=actual[use,k];estimate=pp[use,k]
            if not len(yy):continue
            result.append(dict(pair_ids=list(ids),kind=kind,rows=len(yy),
                mean_forecast_bps=float(estimate.mean()*1e4),mean_actual_bps=float(yy.mean()*1e4),
                rmse_bps=float(np.sqrt(np.mean((estimate-yy)**2))*1e4),
                directional_accuracy=float(np.mean(np.sign(estimate)==np.sign(yy))),
                mean_absolute_forecast_bps=float(np.mean(abs(estimate))*1e4),
                max_absolute_forecast_bps=float(np.max(abs(estimate))*1e4)))
    return dict(rows=result,nonoverlapping_within_pair=True,not_independent_across_pairs=True,
        no_account_costs_or_funding=True,not_used_for_model_selection=True)


def main():
    out=HERE/'artifacts';out.mkdir(exist_ok=False);(out/'runs').mkdir();(out/'models').mkdir()
    cfgs={v:json.loads((HERE/f'{v}_protocol.json').read_text()) for v in ['reference','par']}
    spec=json.loads((HERE/'model_protocol.json').read_text())
    paths=[*HERE.glob('*.py'),*HERE.glob('*protocol.json'),INPUT/'inputs.npz',
        ROOT/cfgs['reference']['reference_run']/'inputs.npz',
        ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',ROOT/'experiments/risk_preserving_20261016/artifacts/targets.npz',
        *[PREVIOUS/n for n in ['account_v2.py','portfolio_v2.py','proposals.py','completion.py']],
        ROOT/'experiments/adaptive_mean_pairs_20261017/signals.py',ROOT/'experiments/pairs_forecast_20261017/model.py',
        ROOT/'experiments/persistent_intent_pairs_20261017/signals.py',ROOT/'experiments/quote_anchored_pairs_20261017/pricing.py',
        OLDER/'manifest.json',*[OLDER/(s+'USDT.npz') for s in ['BTC','ETH','HYPE']],
        *[ROOT/f'experiments/persistent_intent_pairs_20261017/artifacts/{v}_immediate_state.npz' for v in cfgs]]
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),orders_sent=0,historical_independence=False))
    models={};model_records=[]
    for pair in spec['pairs']:
        r=older_pair(pair);d=design(r['market_times'],r['close'],r['volume'],r['trades'],pair,[pair])
        for h in spec['horizons']:
            rec,tree,cal=fit(r['market_times'],r['close'],d,h,spec,ts(spec['train_end']),ts(spec['calibration_end']))
            models[tuple(pair),h]=(rec,tree);stem='_'.join(pair)+f'_{h}'
            save(out/'models'/f'{stem}.json',rec);tree.save_model(out/'models'/f'{stem}.ubj')
            np.savez_compressed(out/'models'/f'{stem}_calibration.npz',**cal)
            model_records.append(dict(pair=pair,**rec))
            print('MODEL',stem,'train',rec['training_rows'],'calibration',rec['calibration_rows'],
                'slope',round(rec['calibration_slope'],6),'intercept',round(rec['calibration_intercept'],6),flush=True)
    save(out/'training_manifest.json',model_records)
    rd,_,protocol=reference_inputs(cfgs['reference']);pd=load(INPUT/'inputs.npz')
    data={'reference':rd,'par':pd}
    references={'reference':load(ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz'),
        'par':load(ROOT/cfgs['par']['reference_run']/'inputs.npz')}
    original={v:load(ROOT/f'experiments/persistent_intent_pairs_20261017/artifacts/{v}_immediate_state.npz') for v in cfgs}
    states={};prediction_audits={}
    for venue,r in references.items():
        cfg=cfgs[venue];dd=design(r['market_times'],r['close'],r['volume'],r['trades'],protocol['symbols'],cfg['dense_pairs'])
        for h in spec['horizons']:
            forecasts={kind:np.full(dd['beta'].shape,np.nan) for kind in ['zero','ridge','calibrated']}
            for k,pair in enumerate(cfg['dense_pairs']):
                ids=[protocol['symbols'].index(s) for s in pair];single=one_pair(dd,ids)
                rec,tree=models[tuple(pair),h];pp=predict(rec,tree,single)
                assert rec['fit_available']<ts(cfg['validation_start'])
                for kind in forecasts:forecasts[kind][:,k]=pp[kind]
            for kind,pp in forecasts.items():
                states[venue,kind,h]=map_state(data[venue]['times'],r,dd,pp,dd['pairs'],
                    cfg['dense_policy']['decision_hours'],cfg['dense_policy']['source_lifetime_hours'])
                np.savez_compressed(out/f'{venue}_{kind}_{h}_state.npz',**states[venue,kind,h])
            for phase in ['validation','comparison']:
                begin=ts(cfg['validation_start'] if phase=='validation' else cfg['comparison_start'])
                end=ts(cfg['comparison_start'] if phase=='validation' else cfg['end'])
                prediction_audits[f'{venue}_{phase}_{h}']=forecast_audit(r,dd,forecasts,h,begin,end)
    # Raw future error diagnostics remain in memory until account selection is frozen.
    runs={};rows=[];arms=cfgs['reference']['controls']+cfgs['reference']['candidates']
    def record(venue,phase,arm,scenario):
        cfg=cfgs[venue];state=original[venue]
        if '_dense_' in arm:
            _,_,kind,h=arm.split('_');state=states[venue,kind,int(h)]
        start=cfg['validation_start'] if phase=='validation' else cfg['comparison_start']
        end=cfg['comparison_start'] if phase=='validation' else cfg['end']
        r=replay(data[venue],state,None,cfg,protocol,arm,scenario,ts(start),ts(end))
        if arm in cfg['controls']:
            with gzip.open(PREVIOUS/f'artifacts_v2/runs/{venue}-{phase}-{arm}-{scenario}.json.gz','rt') as f:old=json.load(f)
            for key in ('fills','snapshots','metrics'):assert old[key]==r[key],(venue,phase,arm,scenario,key)
        label=f'{venue}-{phase}-{arm}-{scenario}';runs[label]=r
        with gzip.open(out/'runs'/f'{label}.json.gz','wt') as f:
            json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        rows.append(dict(venue=venue,phase=phase,arm=arm,scenario=scenario,metrics=r['metrics']))
        m=r['metrics'];print(label,'PnL',round(m['net_pnl'],3),'Q',round(m['financial_score'],7),
            'pairs',m['completed_pairs'],'aborts',m['aborted_pairs'],flush=True)
    for arm in arms:
        for scenario in cfgs['reference']['scenarios']:record('reference','validation',arm,scenario)
    ranking=[]
    for arm in cfgs['reference']['candidates']:
        checks=[];diff=[]
        for scenario in cfgs['reference']['scenarios']:
            m=runs[f'reference-validation-{arm}-{scenario}']['metrics']
            checks.extend([m['net_pnl']>0,m['completed_pairs']>=3,m['risk_breach_bars']==0])
            for base in arms[:2]:
                b=runs[f'reference-validation-{base}-{scenario}']['metrics'];diff.append(m['net_pnl']-b['net_pnl'])
                checks.extend([m['net_pnl']>b['net_pnl'],m['financial_score']>b['financial_score']])
        ranking.append(dict(arm=arm,eligible=all(checks),minimum_pnl_advantage=min(diff)))
    ranking.sort(key=lambda a:(-a['eligible'],-a['minimum_pnl_advantage'],a['arm']))
    selection=dict(selected=ranking[0]['arm'],eligible=ranking[0]['eligible'],ranking=ranking,
        frozen_before_later_reference_and_par=True,historical_independence=False)
    save(out/'selection.json',selection);selection_hash=sha(out/'selection.json')
    for arm in arms:
        for scenario in cfgs['reference']['scenarios']:record('reference','comparison',arm,scenario)
    for phase in ['validation','comparison']:
        for arm in arms:
            for scenario in cfgs['par']['scenarios']:record('par',phase,arm,scenario)
    assert sha(out/'selection.json')==selection_hash
    for rel,digest in sources.items():assert sha(ROOT/rel)==digest,rel
    save(out/'forecast_audits.json',prediction_audits)
    save(out/'results.json',dict(rows=rows,selection=selection,actual_replays=len(runs),existing_controls_exact=True,
        protected=protected(),orders_sent=0,automatic_live_replacement=False,historical_independence=False,
        model_training_end=spec['train_end'],model_calibration_end=spec['calibration_end']))
    print('DONE',len(runs),'eligible',selection['eligible'],flush=True)


if __name__=='__main__':main()
