import argparse
import copy
import gzip
import json
from datetime import datetime
import numpy as np
from .protocol import ROOT,OUT,freeze,save,sha,native,ts
from . import model


def prepare():
    p=freeze()
    if (OUT/'models.json').exists(): raise FileExistsError('Experiment already prepared')
    from competition_live.profile import load_profile,BUNDLE
    from replay_lab.common_comparison.models import Market,covariance
    from replay_lab.common_comparison.engine import project
    assert sha(ROOT/p['pointer']['inputs'])==p['input_hash']
    d=dict(np.load(ROOT/p['pointer']['inputs'],allow_pickle=False))
    previous=dict(np.load(ROOT/p['pointer']['run']/'targets.npz',allow_pickle=False))
    trading=load_profile()['trading']
    market=Market(d['market_times'],d['close'],d['volume'],d['trades'],d['funding'],d['observed'],tuple(p['symbols']))
    sigma=np.tile(np.eye(10)*.0005**2,(len(d['market_times']),1,1))
    for i in range(168,len(sigma)):sigma[i]=covariance(market,i,trading['risk'])[1]
    fitted=model.train(d,p);cal=model.calibrate(d,fitted,p)
    f,mu=model.predict(d,fitted);mu*=cal['scale']
    # Build July targets solely to freeze the exposure control before later results.
    allocation_p=dict(p,calibration_end=p['calibration_start'])
    w=model.weights(d,mu,sigma,allocation_p,trading)
    incumbent=previous['default'].copy()
    mask=(d['market_times']>=p['calibration_start'])&(d['market_times']<p['calibration_end'])
    denominator=np.mean(abs(incumbent[mask]).sum(1))
    exposure_scale=float(np.mean(abs(w[mask]).sum(1))/denominator) if denominator else 0.
    control=incumbent*exposure_scale
    size=incumbent*1.44117372816997
    for a in (control,size):
        for i in range(168,len(a)): a[i]=project(a[i],trading['initial'],sigma[i],trading['risk'])
    save(OUT/'models.json',dict(model=fitted,calibration=cal,exposure_control_scale=exposure_scale))
    np.savez_compressed(OUT/'targets.npz',times=d['market_times'],sigma=sigma,
        incumbent=incumbent,size_only=size,directional=w,exposure_control=control,
        forecast=mu,unit=f['unit'])
    # Strong causality checks: poison held-out prices, delete future rows, refit.
    poisoned={k:v.copy() for k,v in d.items()}
    poisoned['close'][d['market_times']>=p['train_end']]*=7
    altered=model.train(poisoned,p)
    for key in ('beta','center','scale'): np.testing.assert_array_equal(fitted[key],altered[key])
    poisoned={k:v.copy() for k,v in d.items()}
    poisoned['close'][d['market_times']>=p['calibration_end']]*=7
    assert model.calibrate(poisoned,fitted,p)==cal
    n=len(d['market_times'])-48
    prefix={k:v[:n] for k,v in d.items()}
    pf,pm=model.predict(prefix,fitted)
    np.testing.assert_allclose(pm*cal['scale'],mu[:n],equal_nan=True,atol=1e-13)
    pw=model.weights(prefix,pm*cal['scale'],sigma[:n],allocation_p,trading)
    np.testing.assert_allclose(pw,w[:n],atol=1e-13)
    b=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False))
    ix=np.searchsorted(d['market_times'],b['times'])
    np.testing.assert_allclose(incumbent[ix],b['first'],atol=1e-12)
    save(OUT/'causality.json',dict(fit_future_poison_pass=True,calibration_future_poison_pass=True,
        prediction_and_allocation_prefix_invariant=True,incumbent_matches_frozen_bundle=True,
        training_labels_before_train_end=fitted['last_maturity']<p['train_end'],
        calibration_labels_before_evaluation=cal['last_maturity']<p['calibration_end'],
        historical_independence=False))
    # Forecast errors are evaluated only after model and calibration are saved.
    diagnostics=[]
    for name,start,end in [('August',p['evaluation_start'],ts('2026-09-01T12:00:00Z')),
                            ('later',ts('2026-09-01T12:00:00Z'),p['evaluation_end'])]:
        ii,jj,y=model.label_rows(d,f,start,end)
        actual=y*f['unit'][ii,jj];pred=mu[ii,jj]
        diagnostics.append(dict(period=name,rows=len(y),mae_bps=float(np.mean(abs(actual-pred))*10000),
            zero_mae_bps=float(np.mean(abs(actual))*10000),
            direction_accuracy=float(np.mean(np.sign(pred)==np.sign(actual)))))
    save(OUT/'forecast_diagnostics.json',diagnostics)
    save(OUT/'frozen_sources.json',{str(x.relative_to(ROOT)):sha(x) for x in
        [* (ROOT/'pas_allocation_study').glob('*.py'),OUT/'protocol.json',OUT/'models.json',OUT/'targets.npz']})
    print(json.dumps(dict(training_rows=fitted['rows'],calibration=cal,exposure_scale=exposure_scale,
        diagnostics=diagnostics),default=native),flush=True)


def check():
    from competition_live.profile import verify_deployment
    p=freeze()
    for rel,digest in json.loads((OUT/'frozen_sources.json').read_text()).items():
        if sha(ROOT/rel)!=digest:raise ValueError('Frozen source changed: '+rel)
    assert verify_deployment()['files']==p['deployment_hashes']


def summarize(r,arm,phase,scenario):
    from competition_live.scoring import evaluate
    m=r['metrics'];curve=np.array([s['equity']+50000 for s in r['snapshots']])
    samples=[dict(time=s['time'],equity=s['equity']+50000) for s in r['snapshots'] if s['time']%86400==43200]
    sc=evaluate(samples,trading_days=len({f['time']//86400 for f in r['fills']}))
    return dict(arm=arm,phase=phase,scenario=scenario,net_pnl=m['net_pnl'],
        terminal_pnl=m['pnl_after_liquidation_reserve'],**sc,
        max_drawdown=float(-np.min(curve/np.maximum.accumulate(curve)-1)),
        mean_notional=float(np.mean([s['gross']*s['equity'] for s in r['snapshots']])),
        fees=m['fees'],funding=m['funding'],execution_cost=m['execution_cost'],
        base_pnl=m['base_owned_pnl'],pair_pnl=m['pair_owned_pnl'],completed_pairs=m['completed_pairs'],
        fills=m['fills'],risk_breach_bars=m['risk_breach_bars'],accounting_error=m['maximum_accounting_error'],
        symbols=sorted({f['symbol'] for f in r['fills']}),
        short_entries=sum(f['signed_qty']<0 and not f['reduce_only'] for f in r['fills']),
        timing_pass=all(f['source_available']<=f['decision_time']<f['time'] for f in r['fills']))


def historical():
    check();p=freeze()
    from competition_live.profile import load_profile,BUNDLE
    from experiments.decision_gate_pairs_20261017.portfolio import replay
    profile=load_profile();d0=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False))
    states=dict(np.load(BUNDLE/'models/reference_gate_state.npz',allow_pickle=False))
    weights=dict(np.load(OUT/'targets.npz',allow_pickle=False))
    ix=np.searchsorted(weights['times'],d0['times'])
    expected=json.loads((BUNDLE/'expected.json').read_text())['runs']
    windows=[r for r in expected if r['venue']=='reference' and r['arm']==profile['candidate']]
    rows=[]
    for window in windows:
        for arm in p['reference_arms']:
            dest=OUT/'reference'/f"{arm}_{window['phase']}_{window['scenario']}.json.gz"
            if dest.exists():
                with gzip.open(dest,'rt') as f:r=json.load(f)
            else:
                d=dict(d0);d['first']=weights[arm][ix]
                cfg=copy.deepcopy(profile['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
                r=replay(d,states,None,cfg,profile['trading'],profile['candidate'],window['scenario'],window['start'],window['end'])
                dest.parent.mkdir(exist_ok=True)
                with gzip.open(dest,'wt') as f:json.dump(r,f,default=native)
            row=summarize(r,arm,window['phase'],window['scenario'])
            assert row['timing_pass'] and row['accounting_error']<1e-6
            rows.append(row);save(OUT/'reference_results.json',rows)
            print(json.dumps(row),flush=True)
    check()


def pas():
    check();p=freeze()
    from competition_live.profile import load_profile
    from pas_strategy_audit.data import load
    from strategy_recheck.pas import simulate
    from replay_lab.common_comparison.engine import project
    from .venue import basis_forecast,net_forecast
    trading=load_profile()['trading'];cfg,d=load(OUT/'pas')
    z=dict(np.load(OUT/'targets.npz',allow_pickle=False))
    history=dict(np.load(ROOT/p['pointer']['inputs'],allow_pickle=False))
    published={}
    for path in (ROOT/'practice_live/updates').glob('*/signals.json'):
        event=json.loads(path.read_text());t=ts(event['input_completed_bar_utc']);at=ts(event['created_at_utc'])*1000
        if p['pas_start']<=t<=p['pas_end']:published[t]=min(published.get(t,at),at)
    basis=[];rates=[]
    for j,s in enumerate(p['symbols']):
        rows=json.loads((OUT/'pas/raw'/f'pas_hourly_{s}.json').read_text())['data'];pairs=[]
        for b in rows:
            at=int(b['t']//1000+3600);k=np.searchsorted(z['times'],at)
            if k<len(z['times']) and z['times'][k]==at and b['n']>0 and float(b['v'])>0:
                pairs.append((at,float(np.log(float(b['c'])/history['close'][k,j]))))
        basis.append(pairs)
        rows=json.loads((OUT/'pas/raw'/f'pas_funding_{s}.json').read_text())['data']
        rates.append(sorted((r['time']/1000,float(r['fundingRate'])) for r in rows))
    events={a:[] for a in p['pas_arms']};trace=[];held=np.zeros(10)
    for i,t in enumerate(z['times']):
        t=int(t)
        if not p['pas_start']<=t<p['pas_end'] or t%21600:continue
        if t not in published:raise ValueError('Missing original publication availability')
        decision_at=max(published[t],(t+120)*1000)
        minute=int(np.searchsorted(d['times'],decision_at,side='right')-1)
        for arm in p['pas_arms']:
            if arm=='venue_aware':
                net=np.full(10,np.nan)
                for j,s in enumerate(p['symbols']):
                    series=basis[j]
                    correction,detail=basis_forecast([x[0] for x in series],[x[1] for x in series],t-3600,p)
                    known_rates=[r for at,r in rates[j] if at<t and at>=t-25*3600]
                    # Anchor at the LAST known PAS price, never at future execution.
                    price=d['close'][minute,j]
                    fresh=np.isfinite(price) and d['age_ms'][minute,j]<=300000
                    # AR uses an older prefix; translate its delta to current observed basis.
                    current_basis=float(np.log(price/history['close'][i,j])) if fresh else 0.
                    old_basis=detail.get('basis',current_basis)
                    current_correction=(1-detail.get('phi',1.)**24)*(detail.get('center',current_basis)-current_basis)
                    current_correction=float(np.clip(current_correction,-abs(correction)-abs(current_basis-old_basis),abs(correction)+abs(current_basis-old_basis)))
                    net[j],carry=net_forecast(z['forecast'][i,j],current_correction,known_rates,
                        reference_price=history['close'][i,j],venue_price=price if fresh else 1.,basis_now=current_basis)
                    if not fresh:net[j]=np.nan
                    trace.append(dict(source=t,symbol=s,reference_forecast=float(z['forecast'][i,j]),
                        funding_forecast=carry,net_forecast=float(net[j]) if fresh else None,
                        decision_time=decision_at,current_venue_price=float(price) if fresh else None,
                        current_basis=current_basis if fresh else None,current_correction=current_correction,
                        quote_source='last completed trade candle proxy; not historical L2',basis_model=detail))
                held=model.allocate(net,z['unit'][i],held,p)
                risk=float(np.sqrt(max(24*held@z['sigma'][i]@held,0)))
                if risk>p['daily_target_risk']:held*=p['daily_target_risk']/risk
                held=project(held,trading['initial'],z['sigma'][i],trading['risk'])
                w=held.copy()
            else:w=z[arm][i]
            events[arm].append(dict(time=decision_at,source_time=t,weights=w.tolist(),sigma=z['sigma'][i].tolist(),
                volume=history['volume'][i].tolist(),trades=history['trades'][i].tolist()))
    save(OUT/'pas_decision_trace.json',trace);save(OUT/'pas_targets.json',events)
    rows=[]
    for arm,target in events.items():
        for scenario,assumptions in p['scenarios'].items():
            r=simulate(cfg,d,target,trading,assumptions)
            r['metrics']['pair_overlay']='Base-only conditional PAS comparison; no hypothetical pair profitability asserted'
            row=dict(arm=arm,scenario=scenario,**r['metrics'])
            with gzip.open(OUT/'pas'/f'{arm}_{scenario}.json.gz','wt') as f:json.dump(r,f,default=native)
            rows.append(row);save(OUT/'pas_results.json',rows);print(json.dumps(row),flush=True)
    save(OUT/'pas_data_limitations.json',dict(
        candles='Retrospectively downloaded completed trade candles; two-minute source availability assumption for PAS features, original publication times for external signals.',
        funding='Actual PAS settled rates, prior trade-close proxy for unavailable historical settlement oracle. Stress halves funding receipts.',
        marks='Historical PAS mark prices absent. Daily score is a trade-close valuation proxy, not official mark-equity Q.',
        execution='Next-minute adverse high/low plus impact, caps on both past and future realized volume. Historical L2/queue missing; not proven live fills.',
        independent=False,orders_sent=0))
    check()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','historical','pas'])
    globals()[parser.parse_args().stage]()
