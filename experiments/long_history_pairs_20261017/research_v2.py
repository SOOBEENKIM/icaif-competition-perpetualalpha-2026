import gzip,json,time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs as reference_inputs
from experiments.event_value_pairs_20261017.model import source_design
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,ts,save,sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.shared_execution_pairs_20261017.portfolio_v2 import replay
from .model_v2 import feature_at,training_rows,fit,predict,map_state

HERE=Path(__file__).resolve().parent;PREVIOUS=HERE.parent/'shared_execution_pairs_20261017'


def forecast_audit(reference,raw,events,model,cutoff,until):
    rows=[]
    for e in events:
        if e['time']<cutoff or e['maturity'] is None or e['maturity']>=until:continue
        i=int(np.searchsorted(reference['market_times'],e['time']));x,unit=feature_at(reference,raw,i,e['pair_index'])
        estimates={kind:predict(model,x,unit,e['expected_full_convergence'],kind) for kind in ['scalar','ridge']}
        rows.append(dict(time=e['time'],maturity=e['maturity'],actual=e['realized_price_return'],full=e['expected_full_convergence'],predictions=estimates))
    metrics={}
    if rows:
        actual=np.array([r['actual'] for r in rows]);full=np.array([r['full'] for r in rows])
        metrics['zero']=dict(rmse=float(np.sqrt(np.mean(actual**2))),bias=float(-actual.mean()))
        metrics['full']=dict(rmse=float(np.sqrt(np.mean((full-actual)**2))),bias=float(np.mean(full-actual)))
        for kind in ['scalar','ridge']:
            valid=[r for r in rows if r['predictions'][kind]['method']!='unavailable']
            if valid:
                y=np.array([r['actual'] for r in valid]);p=np.array([r['predictions'][kind]['ratio']*r['full'] for r in valid])
                metrics[kind]=dict(events=len(valid),rmse=float(np.sqrt(np.mean((p-y)**2))),bias=float(np.mean(p-y)))
    return dict(rows=rows,metrics=metrics,not_an_account_backtest=True,not_used_for_parameter_selection=True)


def main():
    out=HERE/'artifacts_v2';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    cfgs={v:json.loads((HERE/f'{v}_protocol.json').read_text()) for v in ['reference','par']}
    spec=json.loads((HERE/'model_protocol.json').read_text());older=ROOT/spec['input_directory']
    paths=[*HERE.glob('*.py'),*HERE.glob('*protocol.json'),INPUT/'inputs.npz',ROOT/cfgs['reference']['reference_run']/'inputs.npz',
        ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',ROOT/'experiments/risk_preserving_20261016/artifacts/targets.npz',
        *[PREVIOUS/n for n in ['account_v2.py','portfolio_v2.py','proposals.py','completion.py']],
        ROOT/'experiments/adaptive_mean_pairs_20261017/signals.py',ROOT/'experiments/event_value_pairs_20261017/model.py',
        ROOT/'experiments/quote_anchored_pairs_20261017/pricing.py',older/'summary.json',
        *[ROOT/f'experiments/persistent_intent_pairs_20261017/artifacts/{v}_immediate_state.npz' for v in cfgs]]
    for pair in spec['pairs']:
        stem='_'.join(pair);paths.extend(older/(stem+suffix) for suffix in ['_reference.npz','_state.npz','_events.json'])
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),orders_sent=0,historical_independence=False))
    models={};early={};diagnostics={};training_manifest=[]
    for pair in spec['pairs']:
        stem='_'.join(pair)
        with np.load(older/f'{stem}_reference.npz') as z:reference={k:z[k].copy() for k in z.files}
        with np.load(older/f'{stem}_state.npz') as z:raw={k:z[k].copy() for k in z.files}
        events=json.loads((older/f'{stem}_events.json').read_text());at=ts(spec['training_cutoff']);cut=ts(spec['source_audit_cutoff'])
        rows=training_rows(reference,raw,events,at);models[tuple(pair)]=fit(rows,at,spec)
        past=training_rows(reference,raw,events,cut);early[stem]=fit(past,cut,spec)
        diagnostics[stem]=forecast_audit(reference,raw,events,early[stem],cut,at)
        # Appending arbitrarily bad future labels cannot change the pre-cut model.
        extra=dict(events[-1]);extra['time']=int(reference['market_times'][-1]);extra['maturity']=at+1;extra['realized_price_return']=-9999.
        assert fit(training_rows(reference,raw,events+[extra],at),at,spec)==models[tuple(pair)]
        record=dict(pair=pair,events=len(rows['y']),first_source=int(min(rows['source_times'])) if len(rows['y']) else None,
            last_maturity=models[tuple(pair)]['last_maturity'],fit_time=at,venue='Binance USD-M',no_Hyperliquid_labels=True)
        training_manifest.append(record)
        print('MODEL',stem,'n',len(rows['y']),'scalar',models[tuple(pair)]['scalar'],'ridge',models[tuple(pair)]['ridge'] is not None,flush=True)
    save(out/'models.json',{'_'.join(k):v for k,v in models.items()});save(out/'early_models.json',early)
    save(out/'source_forecast_audit.json',diagnostics);save(out/'training_manifest.json',training_manifest)
    rd,_,protocol=reference_inputs(cfgs['reference'])
    with np.load(INPUT/'inputs.npz') as z:pd={k:z[k].copy() for k in z.files}
    with np.load(ROOT/cfgs['reference']['reference_run']/'inputs.npz') as z:ref={k:z[k].copy() for k in ['market_times','close']}
    with np.load(ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz') as z:original_ref={k:z[k].copy() for k in ['market_times','close']}
    references={'reference':original_ref,'par':ref}
    data={'reference':rd,'par':pd};states={};mapped={};signal_audits={}
    for venue,d in data.items():
        ref=references[venue];raw=source_design(ref,cfgs[venue],protocol['symbols'])
        with np.load(ROOT/f'experiments/persistent_intent_pairs_20261017/artifacts/{venue}_immediate_state.npz') as z:states[venue]={k:z[k].copy() for k in z.files}
        for kind in ['universe','scalar','ridge']:
            mapped[venue,kind],signal_audits[venue+'_'+kind]=map_state(states[venue],d['times'],ref,raw,models,kind,cfgs[venue]['pairs'])
    save(out/'signal_audits.json',signal_audits)
    runs={};rows=[];arms=cfgs['reference']['controls']+cfgs['reference']['candidates']
    def record(venue,phase,arm,scenario):
        cfg=cfgs[venue];state=states[venue];original_arm=arm
        if '_crypto_' in arm or '_long_' in arm:
            kind='universe' if '_crypto_' in arm else arm.rsplit('_',1)[-1]
            state=mapped[venue,kind];original_arm=('first' if arm.startswith('first') else 'second')+'_plus_pairs'
        start=cfg['validation_start'] if phase=='validation' else cfg['comparison_start'];end=cfg['comparison_start'] if phase=='validation' else cfg['end']
        r=replay(data[venue],state,None,cfg,protocol,original_arm,scenario,ts(start),ts(end));r['arm']=arm
        if arm in cfg['controls'][:5]:
            with gzip.open(PREVIOUS/f'artifacts_v2/runs/{venue}-{phase}-{arm}-{scenario}.json.gz','rt') as f:old=json.load(f)
            for key in ('fills','snapshots'):assert old[key]==r[key],(venue,phase,arm,scenario,key)
        label=f'{venue}-{phase}-{arm}-{scenario}';runs[label]=r
        with gzip.open(out/'runs'/f'{label}.json.gz','wt') as f:json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        rows.append(dict(venue=venue,phase=phase,arm=arm,scenario=scenario,metrics=r['metrics']))
        m=r['metrics'];print(label,'PnL',round(m['net_pnl'],3),'Q',round(m['financial_score'],7),'pairs',m['completed_pairs'],flush=True)
    for arm in arms:
        for scenario in cfgs['reference']['scenarios']:record('reference','validation',arm,scenario)
    ranking=[]
    for arm in cfgs['reference']['candidates']:
        checks=[];diff=[];universe_control=arm.split('_')[0]+'_crypto_pairs'
        for scenario in cfgs['reference']['scenarios']:
            m=runs[f'reference-validation-{arm}-{scenario}']['metrics'];checks.extend([m['net_pnl']>0,m['completed_pairs']>=3,m['risk_breach_bars']==0])
            for base in [*arms[:2],universe_control]:
                b=runs[f'reference-validation-{base}-{scenario}']['metrics']
                if base in arms[:2]:diff.append(m['net_pnl']-b['net_pnl'])
                checks.extend([m['net_pnl']>b['net_pnl'],m['financial_score']>b['financial_score']])
        ranking.append(dict(arm=arm,eligible=all(checks),minimum_pnl_advantage=min(diff)))
    ranking.sort(key=lambda a:(-a['eligible'],-a['minimum_pnl_advantage'],a['arm']))
    selection=dict(selected=ranking[0]['arm'],eligible=ranking[0]['eligible'],ranking=ranking,frozen_before_later_reference_and_par=True,historical_independence=False)
    save(out/'selection.json',selection);selection_hash=sha(out/'selection.json')
    for arm in arms:
        for scenario in cfgs['reference']['scenarios']:record('reference','comparison',arm,scenario)
    for phase in ['validation','comparison']:
        for arm in arms:
            for scenario in cfgs['par']['scenarios']:record('par',phase,arm,scenario)
    assert sha(out/'selection.json')==selection_hash
    for rel,digest in sources.items():assert sha(ROOT/rel)==digest,rel
    save(out/'results.json',dict(rows=rows,selection=selection,actual_replays=len(runs),existing_controls_exact=True,training_ends_before_March2026=True,
        protected=protected(),orders_sent=0,automatic_live_replacement=False,historical_independence=False))
    print('DONE',len(runs),'eligible',selection['eligible'],flush=True)


if __name__=='__main__':main()
