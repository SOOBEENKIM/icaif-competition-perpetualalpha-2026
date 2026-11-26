"""First-open evaluation of the eight-hour interval reserved in protocol.json.

Only the originally frozen fixed_entry_1h candidate and incumbent. No model
selection. Retrospective collection and proxy books do NOT make this an
independent actual-execution test. Both begin with identical flat50k sleeves.
"""
import csv,gzip,hashlib,json,sys,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .protocol import ROOT,OUT,save,sha

DEST=OUT/'reserved_eight_hours'
REF=ROOT/'practice_live/updates/20261025T020008252480Z'
START=int(datetime(2026,10,24,18,tzinfo=timezone.utc).timestamp())
END=int(datetime(2026,10,25,2,tzinfo=timezone.utc).timestamp())


def freeze():
    from competition_live.profile import load_profile
    assert not (DEST/'protocol.json').exists()
    old=json.loads((OUT/'protocol.json').read_text())
    for f,h in old['candidate_hashes'].items():assert sha(ROOT/f)==h
    with (ROOT/'reports/top10_live_review_20261024/funding.csv').open(encoding='utf-8-sig') as f:
        wallets=sorted({r['address'] for r in csv.DictReader(f)})
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),parent=str(OUT/'protocol.json'),
        primary=old['primary'],control='incumbent',start=START,end=END,symbols=load_profile()['trading']['symbols'],
        scenarios=old['scenarios'],source=str(REF.relative_to(ROOT)),wallets_for_funding_cash_only=wallets,
        comparison='Flat50kown+50kidle reserve atOct24 18UTC forboth, no preperiodP&L copied. Separate from exact-inventory prospective evaluation. Eight hours cannot produce official volatility or Q.',
        timing='Reconstructed decision inputs use the original published reference snapshot and frozen models. Assume additional120seconds computation after the later of original reference publication andsource+120seconds. Exact old budget-packet reception was not archived; this is explicitly a latency assumption.',
        exposure='Interval reserved at02:54UTC before detailed outcomes were fetched; some aggregate account observations were indirectly known. Not analyst-blind and not actualL2. Do not retune using outcomes and relabel this interval independent.',
        code_hash=sha(__file__),candidate_hashes=old['candidate_hashes'],reference_hashes={str(f.relative_to(ROOT)):sha(f) for f in [REF/'inputs.npz',REF/'targets.npz']},
        orders_sent=0,live_changed=False))


def check():
    p=json.loads((DEST/'protocol.json').read_text());assert sha(__file__)==p['code_hash']
    for f,h in {**p['candidate_hashes'],**p['reference_hashes']}.items():assert sha(ROOT/f)==h,f
    return p


def request(body,path):
    if path.exists():return json.loads(path.read_text())['data']
    req=urllib.request.Request('https://api.hyperliquid-testnet.xyz/info',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req,timeout=30) as response:data=json.load(response)
            if not isinstance(data,list):raise ValueError('Unexpected public response')
            break
        except Exception:
            if attempt==3:raise
            time.sleep(8*(attempt+1))
    save(path,dict(request=body,received_utc=datetime.now(timezone.utc).isoformat(),data=data));time.sleep(4)
    return data


def collect():
    p=check()
    save(DEST/'first_open.json',dict(at=datetime.now(timezone.utc).isoformat(),scope='Public candle/funding outcomes only; no orders'))
    for s in p['symbols']:
        for kind in ['candles','funding']:
            body=(dict(type='candleSnapshot',req=dict(coin='pas:'+s,interval='1m',startTime=(START-86400)*1000,endTime=END*1000-1))
                  if kind=='candles' else dict(type='fundingHistory',coin='pas:'+s,startTime=(START-21600)*1000,endTime=END*1000))
            rows=request(body,DEST/'raw'/f'pas_{kind}_{s}.json');print(s,kind,len(rows),flush=True)
    for addr in p['wallets_for_funding_cash_only']:
        rows=request(dict(type='userFunding',user=addr,startTime=START*1000,endTime=END*1000+999),DEST/'funding_wallets'/f'{addr}.json')
        if len(rows)>=500:raise ValueError('Funding response potentially truncated; page before using')
        print('funding wallet',addr,len(rows),flush=True)
    check()


def assemble():
    p=check();groups={}
    for path in (DEST/'funding_wallets').glob('*.json'):
        for row in json.loads(path.read_text())['data']:
            d=row['delta'];coin=d.get('coin','');q=float(d.get('szi',0));cash=float(d.get('usdc',0));rate=float(d.get('fundingRate',0));at=row['time']
            if q and rate and coin.startswith('pas:') and START*1000<at<=END*1000+999:
                groups.setdefault((coin,at//3600000),[]).append((q,cash,rate,at))
    units={};audit=[]
    for key,rows in groups.items():
        q,cash,rate,at=np.asarray(rows).T
        if np.ptp(rate)>1e-12:continue
        unit=float(q@cash/(q@q));bound=float(1e-6*np.abs(q).sum()/(q@q));oracle=-unit/rate[0];err=float(np.max(abs(cash-q*unit)))
        if oracle<=0 or bound/max(abs(unit),1e-30)>.001 or err>1e-6+bound*max(abs(q))+1e-10:continue
        units[key]=dict(cash_per_unit=unit,rounding_unit_bound=bound,oracle=oracle)
        audit.append(dict(coin=key[0],hour=key[1],observations=len(rows),maximum_cash_residual=err,**units[key]))
    funding=[]
    for s in p['symbols']:
        for r in json.loads((DEST/'raw'/f'pas_funding_{s}.json').read_text())['data']:
            funding.append(dict(time=r['time'],coin='pas:'+s,rate=float(r['fundingRate']),**units.get(('pas:'+s,r['time']//3600000),{})))
    save(DEST/'funding.json',funding);save(DEST/'settlement_cash_audit.json',audit)
    save(DEST/'dataset.json',dict(start_ms=START*1000,cutoff_ms=END*1000,warmup_start_ms=(START-86400)*1000,symbols=p['symbols']))


def prepare():
    from competition_live.profile import load_profile
    from competition_live.candidate import build_candidate
    from pas_economic_allocation import model
    from pas_economic_allocation.anchored import RATIO
    from universe_repair.core import calculate,REVISION
    from experiments.published_pair_lifecycle_20261017.signals import FRAME_KEYS
    from competition_live.runtime import fingerprint
    from candidate_handoff.entry_policy import summarize_candles,NAME
    p=check();profile=load_profile();d=dict(np.load(REF/'inputs.npz'));z=dict(np.load(REF/'targets.npz'));times=d['market_times']
    anchor,gate,parts=calculate(d,z['chronos_forecast'],z['frozen_regime_probability'],REVISION)
    fit=json.loads((ROOT/'reports/pas_economic_allocation_20261022/fit.json').read_text());x,u,carry=model.features(d);price=model.predict(x,u,fit)*fit['attenuation']
    publications={}
    for path in (ROOT/'practice_live/updates').glob('*/signals.json'):
        r=json.loads(path.read_text());source=int(datetime.fromisoformat(r['input_completed_bar_utc']).timestamp());at=datetime.fromisoformat(r['created_at_utc']).timestamp()
        if START-43200<=source<END and (source not in publications or at<publications[source][0]):publications[source]=(at,path)
    raw={s:json.loads((DEST/'raw'/f'pas_candles_{s}.json').read_text())['data'] for s in p['symbols']}
    adapter=build_candidate(profile,lambda *args:None)['signals'];packets=[];audit=[]
    for i,source in enumerate(times):
        source=int(source)
        if not START-43200<=source<END or source not in publications:continue
        at,path=publications[source];available=max(at,source+120)+120
        saved=dict(np.load(path.parent/'inputs.npz'));target=dict(np.load(path.parent/'targets.npz'))
        for field in ['close','funding','volume','trades']:
            np.testing.assert_array_equal(saved[field],d[field][:i+1],err_msg=f'As-of source revised:{source}:{field}')
        digest=sha(path.parent/'inputs.npz')
        f=dict(source_time=source,published_time=available,weights=target['default'][-1].tolist(),retention=[1.]*10,
               sigma=parts['sigma'][i].tolist(),volume=d['volume'][i].tolist(),trades=d['trades'][i].tolist(),source_hash=digest)
        event=dict(channel='strategySignal',received_ms=int(available*1000)+1,data=dict(frame=f,incumbent=f['weights'],candidate=f['weights'],
            publication_sha256=hashlib.sha256(json.dumps({k:f[k] for k in FRAME_KEYS},sort_keys=True,allow_nan=False).encode()).hexdigest()))
        forecasts=adapter.ingest(event,{k:v[:i+1].copy() for k,v in d.items()})
        audit.append(dict(source=source,reference_published_at=at,assumed_budget_available=available,
            nonzero_pair_actions=[dict(pair=fs['pair'],action=fs.get('economic_action')) for fs in forecasts if fs['model']=='decision_gate' and fs.get('economic_action')]))
        if source<START:continue
        available+=.001
        history=summarize_candles(raw,p['symbols'],source);history['received_at']=available
        frame=dict(source_time=source,source_hash=digest,symbols=p['symbols'],anchor_weights=(anchor[i]*RATIO*2).tolist(),protected=gate['protected'][i].tolist(),
            price_forecast_168h=price[i].tolist(),reference_funding_forecast_168h=carry[i].tolist(),sigma=parts['sigma'][i].tolist(),volume=d['volume'][i].tolist(),trades=d['trades'][i].tolist(),
            risk_aversion=5.,execution_policy=NAME,venue_history=history)
        body=dict(candidate='budget_lifecycle',profile_hash=fingerprint(profile),published_at=available,frame=frame,
            original_signal=dict(publication=event,forecasts=forecasts,profile_hash=fingerprint(profile),candidate=profile['candidate'],published_at=available),
            core_revision=REVISION,execution_policy=NAME,universe_audit=[dict(symbol=s,considered=True) for s in p['symbols']],
            research_timing_assumption='Original reference publication plus120scompute; not old live-budget reception')
        packets.append(dict(body=body,sha256=fingerprint(body)))
    save(DEST/'packets.json',packets);save(DEST/'signal_audit.json',audit)
    save(DEST/'input_manifest.json',{str(f.relative_to(ROOT)):sha(f) for f in [DEST/'packets.json',DEST/'funding.json',*(DEST/'raw').glob('*.json'),*(DEST/'funding_wallets').glob('*.json')]})
    check();print('Prepared',len(packets),'reserved-period packets',flush=True)


def job(arm,scenario):
    from competition_live.profile import load_profile
    from pas_strategy_audit.data import load
    from .engine import run
    p=check();cfg,d=load(DEST);source=json.loads((DEST/'packets.json').read_text());funding=json.loads((DEST/'funding.json').read_text())
    r=run(d,load_profile(),p['scenarios'][scenario],funding,source,START,END,arm=arm)
    row=dict(arm=arm,scenario=scenario,**r['metrics'],fills=len(r['fills']),risk_violations=len(r['audit']['risk_violations']))
    assert row['Q'] is None and row['sigma'] is None
    with gzip.open(DEST/f'{arm}_{scenario}.json.gz','wt') as fp:json.dump(dict(row=row,result=r),fp,allow_nan=False,default=lambda x:x.tolist())
    return row


def evaluate():
    check();rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,s) for a in ['incumbent','fixed_entry_1h'] for s in ['base','stress','no_receipts']]):
            r=f.result();rows.append(r);save(DEST/'results.json',rows);print(r,flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
