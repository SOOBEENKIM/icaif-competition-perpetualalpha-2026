"""Post-freeze continuation of the existing one-hour execution candidate.

No signing keys, order endpoint, model refit, or live configuration writes.
"""
import csv
import gzip
import json
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import numpy as np
from pas_cashflow_research.data import ROOT, OUT as PRIOR, SYMBOLS, START, read, save, sha
from pas_cashflow_research.strict_execution import simulate
from pas_cashflow_research.runner import persist
from pas_strategy_audit.data import load

OUT = ROOT / 'reports/execution_revision_20261024'
OLD = PRIOR / 'sealed_extension'
REF = ROOT / 'practice_live/updates/20261024T120003369438Z'
OLD_END = int(datetime(2026, 10, 23, 13, tzinfo=timezone.utc).timestamp())
BOUNDARY = int(datetime(2026, 10, 23, 15, tzinfo=timezone.utc).timestamp())
END = int(datetime(2026, 10, 24, 12, tzinfo=timezone.utc).timestamp())

def freeze():
    assert not (OUT / 'protocol.json').exists()
    original = read(PRIOR / 'nominee_freeze.json')
    for p, h in original['files'].items():
        assert sha(ROOT / p) == h, p
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), start=START,
        prior_end=OLD_END, new_interval_start=BOUNDARY, end=END,
        candidate='incumbent_ttl1h', baseline='incumbent', model_refit=False,
        definition='Identical pullback_any3 and signed all-market targets. Normal intent lifetime900->3600seconds, with persistent reduction-only leftovers. Unchanged risk, capital and fee limits.',
        original_definition_frozen=original['created_utc'],
        exposure='Post-freeze chronological continuation, not pristine analyst-blind data. The fixed candidate predates these outcomes; aggregate peer/account results throughOct24have already been inspected. No reselection/tuning using this continuation.',
        comparison='Continuous inventory fromOct19, no reset at boundary. 50k own sleeve plus50k idle reserve for every arm; teammate returns excluded. Both previous full path and new21h increment reported.',
        acceptance='Positive candidate netPnL and higher netPnL/Q on full path, plus higher new-interval netPnL and terminal-cost-adjusted increment in base,stress,no_receipts. No timing,accounting,risk violations. Check larger/saturated flow sensitivity. Fewer than2new completed daily returns: no independent new-intervalQ.',
        scenarios=read(PRIOR/'protocol.json')['fees'],
        sensitivity=dict(larger_flow_base=dict(fee_bps=9.,impact_bps=10.,participation=.10,receipt_multiplier=1.),
            larger_flow_stress=dict(fee_bps=12.,impact_bps=20.,participation=.025,receipt_multiplier=.5),
            saturation_base=dict(fee_bps=9.,impact_bps=10.,participation=1.,receipt_multiplier=1.),
            saturation_stress=dict(fee_bps=12.,impact_bps=20.,participation=1.,receipt_multiplier=.5)),
        input=str(REF.relative_to(ROOT)), orders_sent=0, live_changed=False)
    save(OUT/'protocol.json', protocol)
    files=[__file__,OUT/'protocol.json',REF/'inputs.npz',REF/'targets.npz',
        ROOT/'reports/top10_live_review_20261024/funding.csv',
        ROOT/'.runtime/pas_budget_lifecycle/execution_policy.json',
        ROOT/'reports/pas_economic_allocation_20261022/fit.json']
    files += [ROOT/p for p in original['files']]
    hashes={str(__import__('pathlib').Path(p).relative_to(ROOT)):sha(p) for p in files}
    save(OUT/'freeze.json',hashes)
    print('Frozen candidate and21hcontinuation; prior hashes verified',flush=True)

def verify():
    for p,h in read(OUT/'freeze.json').items():
        assert sha(ROOT/p)==h,p

def collect():
    verify()
    for symbol in SYMBOLS:
        for kind in ['candles','funding']:
            path=OUT/'raw'/f'pas_{kind}_{symbol}.json'
            if path.exists():continue
            body=(dict(type='candleSnapshot',req=dict(coin='pas:'+symbol,interval='1m',startTime=OLD_END*1000,endTime=END*1000-1))
                  if kind=='candles' else dict(type='fundingHistory',coin='pas:'+symbol,startTime=OLD_END*1000,endTime=END*1000))
            req=urllib.request.Request('https://api.hyperliquid-testnet.xyz/info',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=30) as response:obj=json.load(response)
            assert isinstance(obj,list)
            obj=[r for r in obj if (r['t']+60000 if kind=='candles' else r['time'])<=END*1000]
            save(path,dict(request=body,received_utc=datetime.now(timezone.utc).isoformat(),data=obj))
            print(symbol,kind,len(obj),flush=True)
            time.sleep(4)
    verify()

def assemble():
    verify();groups={}
    with (ROOT/'reports/top10_live_review_20261024/funding.csv').open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            x=json.loads(row['delta']);at=int(row['time']);q=float(x['szi']);cash=float(x['usdc']);rate=float(x['fundingRate'])
            if OLD_END*1000<at<=END*1000 and q and rate and x['coin'].startswith('pas:'):
                groups.setdefault((x['coin'],at//3600000),[]).append((q,cash,rate,at))
    units={};audit=[]
    for key,rows in groups.items():
        q,cash,rates,at=np.asarray(rows).T
        if np.ptp(rates)>1e-12:continue
        unit=float(q@cash/(q@q));bound=float(1e-6*np.abs(q).sum()/(q@q));price=-unit/rates[0];res=float(np.max(abs(cash-q*unit)))
        if price<=0 or bound/max(abs(unit),1e-30)>.001 or res>1e-6+bound*max(abs(q))+1e-10:continue
        units[key]=dict(cash_per_unit=unit,rounding_unit_bound=bound,oracle=price,available=int(max(at))+60000)
        audit.append(dict(coin=key[0],hour=key[1],observations=len(rows),residual=res,**units[key]))
    save(OUT/'oracle_reconstruction.json',audit)
    oldfund=read(OLD/'dataset/funding.json');fund=[];counts=[]
    for symbol in SYMBOLS:
        bars={};rates={};revisions=0
        for source in [OLD/'dataset/raw',OUT/'raw']:
            for b in read(source/f'pas_candles_{symbol}.json')['data']:
                if b['t']+60000>END*1000:continue
                if b['t'] in bars:
                    revisions+=int(any(bars[b['t']][k]!=b[k] for k in ['o','h','l','c','v','n']))
                else:bars[b['t']]=b
            for r in read(source/f'pas_funding_{symbol}.json')['data']:
                if r['time']<=END*1000:rates.setdefault(r['time'],r)
        prior={r['time']:r for r in oldfund if r['coin']=='pas:'+symbol}
        for r in rates.values():
            fund.append(prior.get(r['time'],dict(coin='pas:'+symbol,time=r['time'],rate=float(r['fundingRate']),**units.get(('pas:'+symbol,r['time']//3600000),{}))))
        save(OUT/'dataset/raw'/f'pas_candles_{symbol}.json',dict(data=sorted(bars.values(),key=lambda x:x['t'])))
        save(OUT/'dataset/raw'/f'pas_funding_{symbol}.json',dict(data=sorted(rates.values(),key=lambda x:x['time'])))
        counts.append(dict(symbol=symbol,bars=len(bars),funding=len(rates),overlap_revisions=revisions))
    save(OUT/'dataset/dataset.json',dict(start_ms=START*1000,cutoff_ms=END*1000,warmup_start_ms=(START-21600)*1000,symbols=SYMBOLS))
    save(OUT/'dataset/funding.json',sorted(fund,key=lambda r:(r['time'],r['coin'])))
    save(OUT/'dataset/data_audit.json',counts)

def prepare():
    from pas_performance_research import extension
    verify();directory=OUT/'dataset'
    save(directory/'protocol.json',dict(input=str((REF/'inputs.npz').relative_to(ROOT)),model_inputs=str((REF/'targets.npz').relative_to(ROOT))))
    extension.OUT=directory;extension.END=END;extension.verify=verify;extension.prepare()
    assert read(directory/'pair_audit.json')['can_use_base_only'],'Active pair requires full replay'
    new=dict(np.load(directory/'candidate_inputs.npz'));old=dict(np.load(OLD/'dataset/candidate_inputs.npz'));ix=np.searchsorted(new['times'],old['times'])
    checks={k:bool(np.allclose(new[k][ix],v,equal_nan=True,rtol=1e-11,atol=1e-12)) for k,v in old.items()}
    save(OUT/'input_prefix_checks.json',checks);assert all(checks.values()),checks
    save(OUT/'input_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in [*directory.glob('*.json'),directory/'candidate_inputs.npz',*(OUT/'raw').glob('*.json')]})

def job(arm,scenario,sensitivity=False):
    from competition_live.profile import load_profile
    from pas_performance_research.selected import controller
    verify();directory=OUT/'dataset';cfg,d=load(directory);p=load_profile()['trading'];protocol=read(OUT/'protocol.json')
    sc=protocol['sensitivity' if sensitivity else 'scenarios'][scenario];f=dict(np.load(directory/'candidate_inputs.npz'))
    c=controller(f['times'],f['forecast'],f['carry'],f['anchor'],f['protected'],price_times=d['times']/1000,prices=d['close'],age=d['age_ms']/1000)
    tape=[]
    for symbol in SYMBOLS:
        for b in read(directory/'raw'/f'pas_candles_{symbol}.json')['data']:
            at=b['t']+60000
            if b['n']>0 and float(b['v'])>0 and cfg['start_ms']<=at<=cfg['cutoff_ms']:
                tape.append(dict(time=at,coin='pas:'+symbol,qty=float(b['v']),vwap=float(b['c']),buy_price=float(b['h']),sell_price=float(b['l'])))
    improved=arm=='incumbent_ttl1h'
    r=simulate(cfg,d,read(directory/'events.json'),tape,read(directory/'funding.json'),p,sc,controller=c,entry_lifetime=3600 if improved else 900,sticky_exits=improved)
    row=dict(arm=arm,scenario=scenario,phase='continuous120h',**r['metrics'])
    a=next(x for x in r['snapshots'] if x['time']==BOUNDARY*1000);b=r['snapshots'][-1];cost=(sc['fee_bps']+sc['impact_bps'])/10000
    row.update(tail_pnl=b['equity']-a['equity'],tail_liquidation_adjusted_pnl=b['equity']-b['notional']*cost-a['equity']+a['notional']*cost,
        tail_start_notional=a['notional'],tail_end_notional=b['notional'],tail_fees=b['fees']-a['fees'],tail_funding=b['funding']-a['funding'],tail_fills=sum(x['time']>BOUNDARY for x in r['fills']))
    if not sensitivity:
        with gzip.open(OLD/'evaluation/runs'/f'{arm}_extended97h_{scenario}.json.gz','rt') as fp:old=json.load(fp)
        checks={}
        for field in ['fills','daily','decisions','snapshots']:
            prefix=[x for x in r[field] if (x['time'] if field in ['fills','daily'] else x['time']/1000)<=OLD_END]
            checks[field]=prefix==old[field]
        row['path_prefix_checks']=checks
        assert all(checks.values()),checks
    r['controller_trace']=c.trace
    persist(OUT/('sensitivity' if sensitivity else 'evaluation'),row,r)
    return row

def run(sensitivity=False):
    dest=OUT/('sensitivity' if sensitivity else 'evaluation');assert not (dest/'started.json').exists()
    verify();save(dest/'started.json',dict(created_utc=datetime.now(timezone.utc).isoformat()))
    scenarios=read(OUT/'protocol.json')['sensitivity' if sensitivity else 'scenarios'];rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for task in as_completed([pool.submit(job,a,s,sensitivity) for a in ['incumbent','incumbent_ttl1h'] for s in scenarios]):
            row=task.result();rows.append(row);save(dest/'results.json',rows)
            print({k:row[k] for k in ['arm','scenario','net_pnl','Q','tail_pnl','tail_liquidation_adjusted_pnl','mean_notional']},flush=True)
    verify()

if __name__=='__main__':
    if sys.argv[1]=='sensitivity':run(True)
    else:globals()[sys.argv[1]]()
