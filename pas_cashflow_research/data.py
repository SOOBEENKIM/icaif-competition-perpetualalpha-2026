import csv, hashlib, json, time, urllib.request
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/pas_cashflow_research_20261023'
OLD=ROOT/'reports/pas_performance_research_20261022/extension'
REF=ROOT/'practice_live/updates/20261023T080007449138Z'
SYMBOLS=['BTC','ETH','HYPE','SP500','EWY','EWJ','BRENT','GOLD','USDEUR','USDJPY']
START=1792411200
DEV_END=1792742400
TEST_END=1792756800

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n')

def freeze():
    from pas_performance_research.study import live_files
    if (OUT/'protocol.json').exists():raise FileExistsError('Preserve experiment plan')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),start=START,
        development_selection_end=START+48*3600,development_end=DEV_END,uninspected_pas_tail_end=TEST_END,
        tail_scope='Four-hour PAS outcome extension only. Do not call it a long independent test or compute standalone Q from fewer than2daily returns. Fetch tail only after nominee and code hashes saved.',
        source=str(REF.relative_to(ROOT)),initial=50000,reserved=50000,
        baseline='Current pullback_any3, all_market_signed_core_v1, original active pair expert when nonzero. Verify zero economic pair gates before using base-only event engine.',
        candidates=['funding_basket72','funding_pairs72','basis_funding_basket72','trend_funding_basket72'],
        hypothesis='PAS settlement cash and venue-relative prices may contain economically different information from long reference-price regression. Attribute funding, price and execution costs separately. Symmetric all10 or all45 without named-asset whitelists.',
        controls='Same50k sleeve+50k reserve, original absolute gross/net/asset/daily-risk limits and fees/volume/paid execution. Six-hour strategic updates; mandatory risk reductions bypass cadence. Current inventory matters for turnover cost.',
        selection='Use first48h development results to choose maximum minimum base/stress PnL advantage with positive PnL and no risk/accounting/timing failures. All candidates retained. Later44h development sensitivity, then freeze nominee before new4h extension. Previously seen history never called independent.',
        funding='Past6h/24h estimates with60s settlement observation delay; expected receipts50%, payments100%,48h half-life beyond current rate. Actual hourly cash uses reconstructed oracle per-unit payments where available, conservative price proxy otherwise; reconstruction errors bounded from published cash rounding.',
        fees=dict(base=dict(fee_bps=9.,impact_bps=10.,participation=.01,receipt_multiplier=1.),stress=dict(fee_bps=12.,impact_bps=20.,participation=.0025,receipt_multiplier=.5),no_receipts=dict(fee_bps=12.,impact_bps=20.,participation=.0025,receipt_multiplier=0.)),
        live_hashes=live_files(),orders_sent=0,live_changed=False,external_2022_status='Already exposed and failed prior candidate; not used as PAS training targets or retuned test.'))
    save(OUT/'inputs_before_download.json',{str(p.relative_to(ROOT)):sha(p) for p in [REF/'inputs.npz',REF/'targets.npz',ROOT/'reports/market_candidate_audit_20261023/funding.csv',OUT/'protocol.json']})

def verify():
    from pas_performance_research.study import live_files
    for p,h in read(OUT/'inputs_before_download.json').items():assert sha(ROOT/p)==h,p
    assert live_files()==read(OUT/'protocol.json')['live_hashes']

def collect(tail=False):
    verify()
    if tail:assert (OUT/'nominee_freeze.json').exists(),'Seal nominee before opening tail'
    end=TEST_END if tail else DEV_END
    begin=DEV_END if tail else read(OLD/'protocol.json')['end']-3600
    dest=OUT/('tail_raw' if tail else 'raw')
    for symbol in SYMBOLS:
        for kind in ['candles','funding']:
            path=dest/f'pas_{kind}_{symbol}.json'
            if path.exists():continue
            body=dict(type='candleSnapshot',req=dict(coin='pas:'+symbol,interval='1m',startTime=begin*1000,endTime=end*1000-1)) if kind=='candles' else dict(type='fundingHistory',coin='pas:'+symbol,startTime=begin*1000,endTime=end*1000)
            req=urllib.request.Request('https://api.hyperliquid-testnet.xyz/info',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=30) as response:obj=json.load(response)
            if not isinstance(obj,list):raise ValueError('Unexpected public data')
            # Do not preserve or inspect responses after the declared boundary.
            obj=[x for x in obj if (x['t']+60000<=end*1000 if kind=='candles' else x['time']<=end*1000)]
            save(path,dict(request=body,received_utc=datetime.now(timezone.utc).isoformat(),data=obj))
            print(symbol,kind,len(obj),flush=True);time.sleep(4)
    verify()

def oracle_units():
    groups={}
    with (ROOT/'reports/market_candidate_audit_20261023/funding.csv').open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            x=json.loads(row['delta']);t=int(row['time']);q=float(x['szi']);cash=float(x['usdc']);rate=float(x['fundingRate'])
            if t>DEV_END*1000 or not q or not rate or not x['coin'].startswith('pas:'):continue
            groups.setdefault((x['coin'],t//3600000),[]).append((q,cash,rate,t))
    result={};audit=[]
    for (coin,hour),rows in groups.items():
        rates=np.array([r[2] for r in rows]);q=np.array([r[0] for r in rows]);cash=np.array([r[1] for r in rows])
        if np.ptp(rates)>1e-12:continue
        unit=float(q@cash/(q@q));bound=float(1e-6*np.abs(q).sum()/(q@q));price=-unit/rates[0]
        residual=float(np.max(abs(cash-q*unit)))
        if price<=0 or bound/max(abs(unit),1e-30)>.001 or residual>1e-6+bound*max(abs(q))+1e-10:continue
        result[(coin,hour)]=dict(cash_per_unit=unit,rounding_unit_bound=bound,oracle=price,available=max(r[3] for r in rows)+60000)
        audit.append(dict(coin=coin,hour=hour,oracle=price,observations=len(rows),relative_bound=bound/abs(unit),maximum_cash_residual=residual))
    save(OUT/'oracle_reconstruction.json',audit)
    return result

def assemble(tail=False):
    verify();end=TEST_END if tail else DEV_END;directory=OUT/('tail' if tail else 'development');units=oracle_units()
    counts=[];funding=[]
    for symbol in SYMBOLS:
        merged={};rates={};revisions=0
        sources=[OLD/'raw',OUT/'raw']+([OUT/'tail_raw'] if tail else [])
        for source in sources:
            for b in read(source/f'pas_candles_{symbol}.json')['data']:
                if b['t']+60000>end*1000:continue
                if b['t'] in merged:
                    prior=merged[b['t']]
                    if any(prior[k]!=b[k] for k in ['o','h','l','c','v','n']):revisions+=1
                    continue # Preserve first collected bar on overlapping histories.
                merged[b['t']]=b
            for r in read(source/f'pas_funding_{symbol}.json')['data']:
                if r['time']<=end*1000:rates.setdefault(r['time'],r)
        bars=sorted(merged.values(),key=lambda x:x['t']);ff=sorted(rates.values(),key=lambda x:x['time'])
        assert len({r['time']//3600000 for r in ff})==len(ff),'Repeated funding interval'
        for r in ff:
            coin='pas:'+symbol;rec=units.get((coin,r['time']//3600000),{})
            funding.append(dict(coin=coin,time=r['time'],rate=float(r['fundingRate']),**rec))
        save(directory/'raw'/f'pas_candles_{symbol}.json',dict(data=bars));save(directory/'raw'/f'pas_funding_{symbol}.json',dict(data=ff))
        counts.append(dict(symbol=symbol,bars=len(bars),active=sum(b['n']>0 and float(b['v'])>0 for b in bars),funding=len(ff),overlap_revisions=revisions))
    save(directory/'dataset.json',dict(start_ms=START*1000,cutoff_ms=end*1000,warmup_start_ms=(START-21600)*1000,symbols=SYMBOLS))
    save(directory/'funding.json',sorted(funding,key=lambda r:(r['time'],r['coin'])))
    save(directory/'data_audit.json',dict(symbols=counts,oracle_units=len(units),not_realized_exchange_orders=True))
    verify()

if __name__=='__main__':
    import sys
    globals()[sys.argv[1]](True) if len(sys.argv)>2 and sys.argv[2]=='tail' else globals()[sys.argv[1]]()
