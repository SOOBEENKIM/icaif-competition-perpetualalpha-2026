import gzip,json,time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs as reference_inputs
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,ts,save,sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.shared_execution_pairs_20261017.portfolio_v2 import replay
from .model import source_design,completed_events,fit_at,map_state

HERE=Path(__file__).resolve().parent;PREVIOUS=HERE.parent/'shared_execution_pairs_20261017'


def main():
    out=HERE/'artifacts';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    cfgs={v:json.loads((HERE/f'{v}_protocol.json').read_text()) for v in ('reference','par')}
    spec=json.loads((HERE/'model_protocol.json').read_text())
    paths=[*HERE.glob('*.py'),*HERE.glob('*protocol.json'),INPUT/'inputs.npz',ROOT/cfgs['reference']['reference_run']/'inputs.npz',
        ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',ROOT/'experiments/risk_preserving_20261016/artifacts/targets.npz',
        *[PREVIOUS/n for n in ['account_v2.py','portfolio_v2.py','proposals.py','completion.py']],
        ROOT/'experiments/adaptive_mean_pairs_20261017/signals.py',ROOT/'experiments/persistent_intent_pairs_20261017/signals.py',
        ROOT/'experiments/quote_anchored_pairs_20261017/pricing.py',
        *[ROOT/f'experiments/persistent_intent_pairs_20261017/artifacts/{v}_immediate_state.npz' for v in cfgs]]
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),orders_sent=0,historical_independence=False))
    rd,_,protocol=reference_inputs(cfgs['reference'])
    with np.load(INPUT/'inputs.npz') as z:pd={k:z[k].copy() for k in z.files}
    with np.load(ROOT/cfgs['reference']['reference_run']/'inputs.npz') as z:ref={k:z[k].copy() for k in ['market_times','close']}
    raw=source_design(ref,cfgs['reference'],protocol['symbols']);events=completed_events(ref,raw,cfgs['reference'])
    fits=[fit_at(events,ts(at),spec) for at in spec['fit_dates']]
    # Independent prefix calculation: changing/removing future price paths must
    # not alter any mature labels or the model available at September 1.
    cutoff=ts('2026-09-01');mask=ref['market_times']<cutoff;prefix={k:v[mask] for k,v in ref.items()}
    pr=source_design(prefix,cfgs['reference'],protocol['symbols']);pe=completed_events(prefix,pr,cfgs['reference'])
    assert fit_at(pe,cutoff,spec)==fit_at(events,cutoff,spec)
    for key in ['beta','mean','sd','z','delayed','confirmed','price_recovery']:
        np.testing.assert_array_equal(pr[key],raw[key][mask])
    save(out/'events.json',events);save(out/'fits.json',fits)
    print('EVENTS',len(events),'mature',sum(e['maturity'] is not None for e in events),flush=True)
    for f in fits:print('FIT',f['time'],'n',f['events'],{k:v['ratio'] for k,v in f['groups'].items()},flush=True)
    data={'reference':rd,'par':pd};states={};mapped={};signal_audits={}
    for venue in data:
        with np.load(ROOT/f'experiments/persistent_intent_pairs_20261017/artifacts/{venue}_immediate_state.npz') as z:
            states[venue]={k:z[k].copy() for k in z.files}
        for mode in ('confirmed','calibrated'):
            mapped[venue,mode],signal_audits[venue+'_'+mode]=map_state(states[venue],data[venue]['times'],ref,raw,fits,mode)
    save(out/'signal_audits.json',signal_audits)
    runs={};rows=[];arms=cfgs['reference']['controls']+cfgs['reference']['candidates']
    def record(venue,phase,arm,scenario):
        cfg=cfgs[venue];state=states[venue];original_arm=arm
        if arm in cfg['candidates']:
            mode='confirmed' if 'confirmed' in arm else 'calibrated';state=mapped[venue,mode]
            original_arm=('first' if arm.startswith('first') else 'second')+'_plus_pairs'
        start=cfg['validation_start'] if phase=='validation' else cfg['comparison_start']
        end=cfg['comparison_start'] if phase=='validation' else cfg['end']
        r=replay(data[venue],state,None,cfg,protocol,original_arm,scenario,ts(start),ts(end));r['arm']=arm
        if arm in cfg['controls']:
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
        checks=[];diff=[]
        for scenario in cfgs['reference']['scenarios']:
            m=runs[f'reference-validation-{arm}-{scenario}']['metrics'];checks.extend([m['net_pnl']>0,m['completed_pairs']>=3,m['risk_breach_bars']==0])
            for base in arms[:2]:
                b=runs[f'reference-validation-{base}-{scenario}']['metrics'];diff.append(m['net_pnl']-b['net_pnl'])
                checks.extend([m['net_pnl']>b['net_pnl'],m['financial_score']>b['financial_score']])
        ranking.append(dict(arm=arm,eligible=all(checks),minimum_pnl_advantage=min(diff)))
    ranking.sort(key=lambda r:(-r['eligible'],-r['minimum_pnl_advantage'],r['arm']))
    selection=dict(selected=ranking[0]['arm'],eligible=ranking[0]['eligible'],ranking=ranking,frozen_before_later_reference_and_par=True,historical_independence=False)
    save(out/'selection.json',selection);selection_hash=sha(out/'selection.json')
    for arm in arms:
        for scenario in cfgs['reference']['scenarios']:record('reference','comparison',arm,scenario)
    for phase in ('validation','comparison'):
        for arm in arms:
            for scenario in cfgs['par']['scenarios']:record('par',phase,arm,scenario)
    assert sha(out/'selection.json')==selection_hash
    for rel,digest in sources.items():assert sha(ROOT/rel)==digest,rel
    save(out/'results.json',dict(rows=rows,selection=selection,actual_replays=len(runs),all_controls_exact=True,
        prefix_calibration_check=True,protected=protected(),orders_sent=0,automatic_live_replacement=False,historical_independence=False))
    print('DONE',len(runs),'eligible',selection['eligible'],flush=True)


if __name__=='__main__':main()
