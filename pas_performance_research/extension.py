"""Later PAS development extension, frozen before fetching outcome candles.

Only public data requests. The live producer, execution state and code remain
untouched. Candle extrema fills deliberately differ from the trade-tape model.
"""
import copy,gzip,hashlib,json,sys,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .study import ROOT,BASE,save,sha,live_files
from .funding import forecast,filtered_anchor

OUT=ROOT/'reports/pas_performance_research_20261022/extension'
START=1792411200
END=1792666800 # Oct22 11:00UTC, the last frozen completed reference hour.
SPLIT=1792584000
SYMBOLS=['BTC','ETH','HYPE','SP500','EWY','EWJ','BRENT','GOLD','USDEUR','USDJPY']

def freeze():
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    pointer=json.loads((ROOT/'practice_live/latest.json').read_text())
    # Explicit immutable pointer; do not follow future live data publications.
    run=ROOT/'practice_live/updates/20261022T110017645155Z'
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),start=START,split=SPLIT,end=END,
        candidate='venue_funding',arms=[BASE,'venue_funding'],input=str((run/'inputs.npz').relative_to(ROOT)),
        model_inputs=str((run/'targets.npz').relative_to(ROOT)),live_files=live_files(),
        development=True,independent_test=False,model_refit=False,orders_sent=0,live_changed=False,
        rationale='Candidate frozen before fetching later PAS candles; developer has already seen some aggregate live outcomes, so not claimed untouched. Earlier48h, later23h and continuous71h all retained.',
        fill='Completed next-minute high for buys/low for sells plus adverse impact, capped by1% or0.25% of actual minute volume. No same-minute fill, no inferred queue/L2. Opposite directions within one bar treated pessimistically.',
        funding='Actual PAS historical rates with60s observation lag for forecasts. Settlement valuation uses known trade price as oracle proxy; report sensitivity. No live contemporaneous receipt record.',
        pair='Rebuild retained pair signals as-of. Base-only engine permitted only if every economic gate is zero; otherwise mark extension incomplete rather than silently omit pairs.',
        scenarios=dict(base_1pct=dict(fee_bps=9.,impact_bps=10.,participation=.01,receipt_multiplier=1.),
          stress_025pct=dict(fee_bps=12.,impact_bps=20.,participation=.0025,receipt_multiplier=.5),
          no_funding_receipts=dict(fee_bps=12.,impact_bps=20.,participation=.0025,receipt_multiplier=0.)))
    save(OUT/'protocol.json',p)
    save(OUT/'dataset.json',dict(start_ms=START*1000,cutoff_ms=END*1000,warmup_start_ms=(START-21600)*1000,symbols=SYMBOLS))
    save(OUT/'freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),Path(__file__).with_name('funding.py'),OUT/'protocol.json',run/'inputs.npz',run/'targets.npz']})

def verify():
    for f,h in json.loads((OUT/'freeze.json').read_text()).items():assert sha(ROOT/f)==h,f
    assert live_files()==json.loads((OUT/'protocol.json').read_text())['live_files']

def fetch():
    verify()
    for s in SYMBOLS:
        for kind in ['candles','funding']:
            dest=OUT/'raw'/f'pas_{kind}_{s}.json'
            if dest.exists():continue
            body=dict(type='candleSnapshot',req=dict(coin='pas:'+s,interval='1m',startTime=(START-21600)*1000,endTime=END*1000)) if kind=='candles' else dict(type='fundingHistory',coin='pas:'+s,startTime=(START-21600)*1000,endTime=END*1000)
            req=urllib.request.Request('https://api.hyperliquid-testnet.xyz/info',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            # No retries on rate limit: prioritize the existing live worker.
            with urllib.request.urlopen(req,timeout=25) as r:data=json.load(r)
            if not isinstance(data,list):raise ValueError('Invalid public response')
            save(dest,dict(request=body,received_utc=datetime.now(timezone.utc).isoformat(),data=data))
            print(s,kind,len(data),flush=True)
            time.sleep(12 if kind=='candles' else 4)

def prepare():
    from competition_live.profile import load_profile
    from competition_live.candidate import build_candidate
    from pas_economic_allocation import model
    from pas_economic_allocation.anchored import RATIO
    from universe_repair.core import calculate
    from experiments.published_pair_lifecycle_20261017.signals import FRAME_KEYS
    verify();p=json.loads((OUT/'protocol.json').read_text());profile=load_profile()
    d=dict(np.load(ROOT/p['input']));z=dict(np.load(ROOT/p['model_inputs']));t=d['market_times']
    anchor,gate,parts=calculate(d,z['chronos_forecast'],z['frozen_regime_probability'],BASE)
    fit=json.loads((ROOT/'reports/pas_economic_allocation_20261022/fit.json').read_text());x,u,carry=model.features(d)
    price=model.predict(x,u,fit)*fit['attenuation']
    # Prefix data revision test against the first-wave historical input.
    oldp=json.loads((ROOT/'reports/universe_repair_20261022/protocol.json').read_text())['input']['inputs']
    old=dict(np.load(ROOT/oldp));ix=np.searchsorted(t,old['market_times'])
    revisions={k:int(np.count_nonzero(~np.isclose(d[k][ix],old[k],equal_nan=True))) for k in ['close','funding','volume']}
    publications={}
    for f in (ROOT/'practice_live/updates').glob('*/signals.json'):
        r=json.loads(f.read_text());source=int(datetime.fromisoformat(r['input_completed_bar_utc']).timestamp());at=datetime.fromisoformat(r['created_at_utc']).timestamp()
        if START-43200<=source<END and (source not in publications or at<publications[source][0]):publications[source]=(at,f)
    adapter=build_candidate(profile,lambda *a:None)['signals'];records=[];pair=[]
    for i,source in enumerate(t):
        source=int(source)
        if not START-43200<=source<END or (source<START and source%86400):continue
        if source not in publications:raise ValueError('Missing observed publication:'+str(source))
        at,file=publications[source];at=max(at,source+120)
        frame=dict(source_time=source,published_time=at,weights=(anchor[i]*RATIO*2).tolist(),retention=[1.]*10,
            sigma=parts['sigma'][i].tolist(),volume=d['volume'][i].tolist(),trades=d['trades'][i].tolist(),source_hash=sha(file.parent/'inputs.npz'))
        # The retained pair expert sees the original frozen incumbent frame,
        # as it does in live production, not our alternative core weights.
        historical=dict(np.load(file.parent/'targets.npz'))
        frame['weights']=historical['default'][-1].tolist()
        event=dict(channel='strategySignal',received_ms=int(at*1000)+1,data=dict(frame=frame,incumbent=frame['weights'],candidate=frame['weights'],
            publication_sha256=hashlib.sha256(json.dumps({k:frame[k] for k in FRAME_KEYS},sort_keys=True,allow_nan=False).encode()).hexdigest()))
        fs=adapter.ingest(event,{k:v[:i+1].copy() for k,v in d.items()})
        if source<START:continue
        pair.append(dict(source=source,forecasts=fs))
        records.append(dict(source_time=source,time=int(at*1000)+1,weights=frame['weights'],sigma=frame['sigma'],volume=frame['volume'],trades=frame['trades'],
            anchor=(anchor[i]*RATIO*2).tolist(),protected=gate['protected'][i].tolist(),forecast=price[i].tolist(),carry=carry[i].tolist()))
    save(OUT/'events.json',records);save(OUT/'pair_forecasts.json',pair);save(OUT/'prefix_revisions.json',revisions)
    np.savez_compressed(OUT/'candidate_inputs.npz',times=[r['source_time'] for r in records],forecast=[r['forecast'] for r in records],carry=[r['carry'] for r in records],anchor=[r['anchor'] for r in records],protected=np.array([r['protected'] for r in records],bool))
    nonzero=[dict(source=r['source'],pair=f['pair'],action=f.get('economic_action')) for r in pair for f in r['forecasts'] if f['model']=='decision_gate' and (not f['economic_available'] or f['economic_action']!=0)]
    save(OUT/'pair_audit.json',dict(nonzero=nonzero,can_use_base_only=not nonzero,events=len(records)))
    print('prepared',len(records),'nonzero_pair',len(nonzero),'revisions',revisions,flush=True)

def run():
    from competition_live.profile import load_profile
    from pas_strategy_audit.data import load
    from pas_counterfactual_20261022.engine import simulate
    from pas_economic_allocation.lifecycle import LifecycleController
    verify();p=json.loads((OUT/'protocol.json').read_text());profile=load_profile()
    if not json.loads((OUT/'pair_audit.json').read_text())['can_use_base_only']:raise ValueError('Cannot omit active pair expert')
    cfg,d=load(OUT);events=json.loads((OUT/'events.json').read_text());f=dict(np.load(OUT/'candidate_inputs.npz'))
    tape=[];funding=[];obs=[]
    for j,s in enumerate(SYMBOLS):
        for b in json.loads((OUT/'raw'/f'pas_candles_{s}.json').read_text())['data']:
            known=int(b['t'])+60000
            if float(b['v'])>0 and b['n']>0 and START*1000<=known<=END*1000:
                tape.append(dict(time=known,coin='pas:'+s,qty=float(b['v']),vwap=float(b['c']),buy_price=float(b['h']),sell_price=float(b['l'])))
        for r in json.loads((OUT/'raw'/f'pas_funding_{s}.json').read_text())['data']:
            obs.append(dict(known=r['time']/1000+60,symbol=s,rate=float(r['fundingRate'])))
            funding.append(dict(time=r['time'],coin='pas:'+s,rate=float(r['fundingRate']),cash_per_unit=None))
    venue_carry,_=forecast(f['times'],obs,SYMBOLS)
    rows=[]
    for phase,start,end in [('earlier48h',START,SPLIT),('later23h',SPLIT,END),('continuous71h',START,END)]:
        for scenario,sc in p['scenarios'].items():
            for arm in p['arms']:
                cc=copy.deepcopy(cfg);cc.update(start_ms=start*1000,cutoff_ms=end*1000)
                if arm==BASE:anchor,guard,carry=f['anchor'],f['protected'],f['carry']
                else:
                    carry=venue_carry;anchor,guard=filtered_anchor(f['anchor'],f['protected'],f['forecast'],carry,arm)
                c=LifecycleController(f['times'],f['forecast'],carry,anchor,guard,5.)
                r=simulate(cc,d,copy.deepcopy(events),tape,funding,profile['trading'],sc,controller=c)
                row=dict(arm=arm,phase=phase,scenario=scenario,**r['metrics']);rows.append(row)
                file=OUT/'pas'/f'{arm}_{phase}_{scenario}.json.gz';file.parent.mkdir(exist_ok=True)
                with gzip.open(file,'wt') as fp:json.dump(r,fp,default=lambda x:x.tolist(),allow_nan=False)
                save(OUT/'results.json',rows);print(json.dumps({k:row[k] for k in ['arm','phase','scenario','net_pnl','Q','risk_breach_events']}),flush=True)
    verify()

if __name__=='__main__':globals()[sys.argv[1]]()
