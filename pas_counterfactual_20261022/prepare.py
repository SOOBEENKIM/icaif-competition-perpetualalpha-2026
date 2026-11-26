import copy,json,hashlib
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from pas_economic_allocation import model
from pas_adaptive_development.study import ROOT,save,sha
from strategy_repairs.model import build as base_build
from strategy_final_risk.model import final_mix
from replay_lab.common_comparison.models import Market

OUT=ROOT/'reports/pas_counterfactual_20261022'
START=1792411200
END=1792584000


def prepare():
    from competition_live.profile import load_profile,verify_deployment
    from competition_live.candidate import build_candidate
    from experiments.published_pair_lifecycle_20261017.signals import FRAME_KEYS
    OUT.mkdir(exist_ok=True);profile=load_profile()
    if (OUT/'source_freeze.json').exists():raise FileExistsError('Frozen run already exists')
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),start=START,end=END,
        interval_kst='2026-10-19 21:00:00 through 2026-10-21 21:00:00',
        candidate='budget_lifecycle',model_refit=False,model_horizon_hours=168,
        target='48-hour PAS strategy counterfactual; not a matured168h forecast accuracy test',
        arms=['incumbent','size_only','budget_lifecycle'],primary_scenario='base_1pct',
        scenarios={'base_1pct':dict(fee_bps=9.,impact_bps=10.,participation=.01,receipt_multiplier=1.),
                   'stress_025pct':dict(fee_bps=12.,impact_bps=20.,participation=.0025,receipt_multiplier=.5),
                   'base_10pct':dict(fee_bps=9.,impact_bps=10.,participation=.1,receipt_multiplier=1.)},
        intent='Freeze candidate and scenarios before this replay. No PAS labels used to fit coefficients, choose thresholds or change horizons.',
        causality='Only data available by each source timestamp. Decisions use completed candles before publication time. Execution only subsequent candle. Actual per-event PAS funding is a cash outcome, not a signal input.',
        fill_model='Conditional marketable IOC: next-minute candle high for buys/low for sells plus impact, up to a fixed share of that minute traded volume. An observed trade is not resting book depth. No inferred queue priority.',
        engine_repairs=['Do not impose historical trade volume as a resting-depth prerequisite.',
            'Exchange minimum applies to submitted order, not partial fill value.',
            'Static prior-close limit is retained and separately diagnosed; no future-price-derived order limit.',
            'Snapshot and funding valuation retain missing/stale prices; do not future-backfill.'],
        limitations=['No historical L2/mark/oracle tape for whole interval. Trades include actual incumbent orders; volume is endogenous.',
            'Only two completed daily returns; volatility/Q are very unstable and candle-valued conditional quantities.',
            'This period was previously inspected; not an untouched holdout.',
            'The candidate rule was developed later; as-of simulation does not erase human selection knowledge.',
            '168h predictions issued Oct19-21 do not mature within this evaluation window.'],
        live_original=verify_deployment()['files'],orders_sent=0)
    fitfile=ROOT/'reports/pas_economic_allocation_20261022/fit.json'
    fit=json.loads(fitfile.read_text());p['fit_sha256']=sha(fitfile)
    assert fit['last_training_maturity']<START and fit['last_calibration_maturity']<START
    p['model_maturity']={k:fit[k] for k in ('last_training_maturity','last_calibration_maturity')}
    if (OUT/'protocol.json').exists():save(OUT/'preparation_attempt_protocol.json',json.loads((OUT/'protocol.json').read_text()))
    save(OUT/'protocol.json',p)
    prior=json.loads((ROOT/'reports/pas_adaptive_development_20261021/protocol.json').read_text())
    data=dict(np.load(ROOT/prior['pointer']['inputs'],allow_pickle=False))
    targets=dict(np.load(ROOT/'reports/strategy_repairs_20261021/final_risk_targets.npz'))
    default=dict(np.load(ROOT/'reports/pas_adaptive_development_20261021/targets.npz'))
    x,u,carry=model.features(data);pred=model.predict(x,u,fit)*fit['attenuation']
    sources=data['market_times']
    oldtargets=dict(np.load(ROOT/prior['pointer']['run']/'targets.npz',allow_pickle=False))
    _,parts=base_build(data,oldtargets['chronos_forecast'],oldtargets['frozen_regime_probability'],
        dict(scope='legacy',short=False,cross_exit=False,risk=True,original_size=False))
    market=Market(sources,data['close'],data['volume'],data['trades'],data['funding'],data['observed'],tuple(profile['trading']['symbols']))
    weights,trace=final_mix(market,parts['ungated_sized_trend'],.5*parts['reversion_weight'],float(parts['final_scale']),
        oldtargets['frozen_regime_probability'],oldtargets['chronos_forecast'])
    ix=np.searchsorted(sources,targets['times'])
    np.testing.assert_allclose(weights[ix],targets['final_mix_risk'],atol=1e-10,rtol=0)
    targets=dict(times=sources,final_mix_risk=weights,protected=trace['protected'],sigma=parts['sigma'])
    assert np.array_equal(sources,default['times'])
    # Verify actual publication times and target values, rather than backdate
    # the current regenerated signal to the historical decision boundary.
    publications={}
    for file in (ROOT/'practice_live/updates').glob('*/signals.json'):
        r=json.loads(file.read_text());source=int(datetime.fromisoformat(r['input_completed_bar_utc']).timestamp())
        at=datetime.fromisoformat(r['created_at_utc']).timestamp()
        if source<START-43200 or source>=END:continue
        if source not in publications or at<publications[source][0]:publications[source]=(at,file)
    adapter=build_candidate(profile,lambda *a:None)['signals']
    records=[];pair=[];audit=[]
    for i,source in enumerate(sources):
        source=int(source)
        if source<START-43200 or source>=END:continue
        if source<START and source%86400:continue  # midnight initialization
        if source not in publications:raise ValueError('Missing publication: '+str(source))
        at,file=publications[source];at=max(at,source+120)
        old=np.load(file.parent/'targets.npz',allow_pickle=False)
        assert int(old['times'][-1])==source
        np.testing.assert_allclose(old['default'][-1],default['incumbent'][i],atol=1e-10,rtol=0)
        frame=dict(source_time=source,published_time=at,weights=default['incumbent'][i].tolist(),retention=[1.]*10,
            sigma=targets['sigma'][i].tolist(),volume=data['volume'][i].tolist(),trades=data['trades'][i].tolist(),
            source_hash=sha(file.parent/'inputs.npz'))
        event=dict(channel='strategySignal',received_ms=int(at*1000)+1,data=dict(frame=frame,
            incumbent=frame['weights'],candidate=frame['weights'],
            publication_sha256=hashlib.sha256(json.dumps({k:frame[k] for k in FRAME_KEYS},sort_keys=True,allow_nan=False).encode()).hexdigest()))
        prefix={k:v[:i+1].copy() for k,v in data.items()}
        forecasts=adapter.ingest(event,prefix)
        if source<START:continue
        record=dict(source_time=source,time=int(at*1000)+1,weights=default['incumbent'][i].tolist(),
            size_only=default['size_only'][i].tolist(),sigma=frame['sigma'],volume=frame['volume'],trades=frame['trades'],
            anchor=(targets['final_mix_risk'][i]*2*1.44117372816997).tolist(),protected=targets['protected'][i].tolist(),
            forecast=pred[i].tolist(),carry=carry[i].tolist())
        records.append(record);pair.append(dict(source=source,publication_time=at,forecasts=forecasts))
        audit.append(dict(source=source,publication=at,actual_publication_file=str(file.relative_to(ROOT)),
            training_labels_last=fit['last_training_maturity'],calibration_labels_last=fit['last_calibration_maturity'],
            pair_daily_fit=adapter.gate_day))
    save(OUT/'events.json',records);save(OUT/'pair_forecasts.json',pair);save(OUT/'availability.json',audit)
    np.savez_compressed(OUT/'candidate_inputs.npz',times=[r['source_time'] for r in records],
        forecast=[r['forecast'] for r in records],carry=[r['carry'] for r in records],
        anchor=[r['anchor'] for r in records],protected=np.array([r['protected'] for r in records],bool))
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in [fitfile,ROOT/prior['pointer']['inputs'],
        ROOT/'reports/strategy_repairs_20261021/final_risk_targets.npz',OUT/'protocol.json',OUT/'events.json',OUT/'pair_forecasts.json',
        * (ROOT/'pas_economic_allocation').glob('*.py')]})
    print(json.dumps(dict(sources=len(records),pair_actions=[dict(source=r['source'],actions=[f.get('economic_action') for f in r['forecasts'] if f['model']=='decision_gate']) for r in pair])))


if __name__=='__main__':prepare()
