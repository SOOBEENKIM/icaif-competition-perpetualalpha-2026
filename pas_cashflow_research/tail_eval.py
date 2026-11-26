"""One frozen PAS outcome extension. Public read-only collection, no order client."""
import copy,csv,gzip,json,sys,time,urllib.request
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .data import ROOT,OUT,OLD,START,DEV_END,SYMBOLS,read,save,sha
from .wave4 import verify as prior_verify
from .strict_execution import simulate
from .policy_v2 import VenueController
from .wave3 import predictions
from .runner import persist
from pas_strategy_audit.data import load

HERE=OUT/'sealed_extension'
NEW_START=DEV_END+3600   #09 UTC: discard the previously observed08:14 interval.
END=DEV_END+5*3600      #13 UTC: last completed hour when protocol was written.
REF=ROOT/'practice_live/updates/20261023T130014868332Z'
ARMS={'incumbent':('incumbent',900,False),'incumbent_ttl1h':('incumbent',3600,True),'reversion12_quote3h':('pas_reversion12',10800,True)}

def freeze():
    prior_verify()
    assert not (OUT/'nominee_freeze.json').exists(),'Nominees already sealed'
    latest={}
    with (ROOT/'reports/market_candidate_audit_20261023/funding.csv').open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if int(row['time'])>DEV_END*1000:continue
            x=json.loads(row['delta']);key=(row['address'],x['coin'])
            if key not in latest or int(row['time'])>latest[key][0]:latest[key]=(int(row['time']),abs(float(x['szi'])))
    wallets=set()
    for symbol in SYMBOLS:
        ranked=sorted([(q,addr) for (addr,coin),(t,q) in latest.items() if coin=='pas:'+symbol],reverse=True)
        wallets.update(addr for q,addr in ranked[:2])
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),end=END,new_outcome_start=NEW_START,
        original_tail_start=DEV_END,original_tail_end=DEV_END+4*3600,
        exposure_amendment='The old market audit had outcomes through08:14UTC. Exclude08-09from new-outcome claims. Extend endpoint to13UTC, the latest completed hour at protocol creation, before downloading prices. This is a short uninspected PAS outcome extension, not a new multi-day independent sample.',
        arms=ARMS,primary='reversion12_quote3h',execution_control='incumbent_ttl1h',
        primary_reason='Full92h development improves netPnL and Q under base and stress with near incumbent exposure. This does NOT satisfy the original first48h selection rule: report that failure explicitly. No revision of the original protocol or claim of unbiased model selection.',
        hypothesis='Quarter reversal of the past12h PAS return, all10 eligible, covariance/carry/cost constrained. Precommitted economic entry limits persist3h, reductions remain eligible until superseded or filled. Retain the original pair expert only when its audited economic gate is nonzero; stop if base-only assumption fails.',
        code_review_corrections='All arms exclude minute bars overlapping decision time. Forced risk exits cannot be blocked by historical entry-price bands. Frozen wave1-4 results retained, corrected development rerun reported separately.',
        boundary='No resetting balances or positions at09UTC. Replay from competition start, compare incremental equity09-13UTC, include earlier inventory and transaction cash. Show12UTCscore separately from13UTCmarkedPnL. Fewer than2new daily returns: no standalone new-tail volatility/Q.',
        success='Primary must exceed incumbent netPnL and completed-dayQ on the extended whole path in base and stress, and have positive new-tail PnL at least as high as incumbent after conservative end-cost adjustment in both. Failure is recorded without changing parameters or replacing the primary with a post-hoc winner.',
        reference=str(REF.relative_to(ROOT)),funding_wallets=sorted(wallets),
        costs=read(OUT/'protocol.json')['fees'],orders_sent=0,live_changed=False)
    save(HERE/'protocol.json',protocol)
    files=list((ROOT/'pas_cashflow_research').glob('*.py'))+[HERE/'protocol.json',REF/'inputs.npz',REF/'targets.npz']
    files += [ROOT/p for p in ['competition_live/scoring.py','competition_live/config.json','pas_performance_research/extension.py','pas_strategy_audit/data.py','pas_performance_research/selected.py','pas_economic_allocation/policy.py','replay_lab/common_comparison/engine.py','experiments/shared_execution_pairs_20261017/account_v2.py']]
    save(OUT/'nominee_freeze.json',dict(created_utc=protocol['created_utc'],files={str(p.relative_to(ROOT)):sha(p) for p in files},primary=protocol['primary']))
    print('Frozen',protocol['primary'],'uninspected outcome hours',NEW_START,END,'funding-wallets',len(wallets),flush=True)

def verify():
    prior_verify()
    for p,h in read(OUT/'nominee_freeze.json')['files'].items():assert sha(ROOT/p)==h,p

def request(path,body):
    if path.exists():return
    req=urllib.request.Request('https://api.hyperliquid-testnet.xyz/info',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=30) as r:obj=json.load(r)
    assert isinstance(obj,list),'Unexpected public response'
    save(path,dict(request=body,received_utc=datetime.now(timezone.utc).isoformat(),data=obj))
    print(path.name,len(obj),flush=True);time.sleep(4)

def collect():
    verify()
    for symbol in SYMBOLS:
        request(HERE/'raw'/f'pas_candles_{symbol}.json',dict(type='candleSnapshot',req=dict(coin='pas:'+symbol,interval='1m',startTime=DEV_END*1000,endTime=END*1000-1)))
        request(HERE/'raw'/f'pas_funding_{symbol}.json',dict(type='fundingHistory',coin='pas:'+symbol,startTime=DEV_END*1000,endTime=END*1000))
    for addr in read(HERE/'protocol.json')['funding_wallets']:
        request(HERE/'raw'/f'user_funding_{addr}.json',dict(type='userFunding',user=addr,startTime=DEV_END*1000,endTime=END*1000))
    verify()

def funding_units():
    groups={}
    for p in sorted((HERE/'raw').glob('user_funding_*.json')):
        for row in read(p)['data']:
            at=int(row['time']);x=row['delta'];q=float(x['szi']);cash=float(x['usdc']);rate=float(x['fundingRate'])
            if not DEV_END*1000<at<=END*1000 or not q or not rate or not x['coin'].startswith('pas:'):continue
            groups.setdefault((x['coin'],at//3600000),[]).append((q,cash,rate,at))
    result={};audit=[]
    for (coin,hour),rows in groups.items():
        rates=np.array([r[2] for r in rows]);q=np.array([r[0] for r in rows]);cash=np.array([r[1] for r in rows])
        if np.ptp(rates)>1e-12:continue
        unit=float(q@cash/(q@q));bound=float(1e-6*np.abs(q).sum()/(q@q));price=-unit/rates[0];res=float(np.max(abs(cash-q*unit)))
        if price<=0 or bound/max(abs(unit),1e-30)>.001 or res>1e-6+bound*max(abs(q))+1e-10:continue
        rec=dict(cash_per_unit=unit,rounding_unit_bound=bound,oracle=price,available=max(r[3] for r in rows)+60000)
        result[(coin,hour)]=rec;audit.append(dict(coin=coin,hour=hour,observations=len(rows),residual=res,**rec))
    save(HERE/'oracle_reconstruction.json',audit);return result

def assemble():
    verify();units=funding_units();oldfund=read(OUT/'development/funding.json');fund=[];counts=[]
    for symbol in SYMBOLS:
        merged={};rates={};revisions=0
        for src in [OUT/'development/raw',HERE/'raw']:
            for b in read(src/f'pas_candles_{symbol}.json')['data']:
                if b['t']+60000>END*1000:continue
                if b['t'] in merged:
                    if any(merged[b['t']][k]!=b[k] for k in ['o','h','l','c','v','n']):revisions+=1
                    continue
                merged[b['t']]=b
            for r in read(src/f'pas_funding_{symbol}.json')['data']:
                if r['time']<=END*1000:rates.setdefault(r['time'],r)
        bars=sorted(merged.values(),key=lambda b:b['t']);ff=sorted(rates.values(),key=lambda r:r['time'])
        save(HERE/'dataset/raw'/f'pas_candles_{symbol}.json',dict(data=bars));save(HERE/'dataset/raw'/f'pas_funding_{symbol}.json',dict(data=ff))
        old={r['time']:r for r in oldfund if r['coin']=='pas:'+symbol}
        for r in ff:
            coin='pas:'+symbol
            fund.append(old.get(r['time'],dict(coin=coin,time=r['time'],rate=float(r['fundingRate']),**units.get((coin,r['time']//3600000),{}))))
        counts.append(dict(symbol=symbol,bars=len(bars),funding=len(ff),revisions=revisions))
    save(HERE/'dataset/dataset.json',dict(start_ms=START*1000,cutoff_ms=END*1000,warmup_start_ms=(START-21600)*1000,symbols=SYMBOLS))
    save(HERE/'dataset/funding.json',sorted(fund,key=lambda r:(r['time'],r['coin'])))
    save(HERE/'dataset/data_audit.json',counts);verify()

def prepare():
    from pas_performance_research import extension
    verify();directory=HERE/'dataset'
    save(directory/'protocol.json',dict(input=str((REF/'inputs.npz').relative_to(ROOT)),model_inputs=str((REF/'targets.npz').relative_to(ROOT))))
    extension.OUT=directory;extension.END=END;extension.verify=verify;extension.prepare()
    assert read(directory/'pair_audit.json')['can_use_base_only'],'Do not omit a newly active incumbent pair'
    arrays,audits=predictions(directory);np.savez_compressed(HERE/'forecasts.npz',**arrays)
    prior=dict(np.load(OUT/'wave3/forecasts.npz'));ix=np.searchsorted(arrays['times'],prior['times'])
    assert np.array_equal(arrays['times'][ix],prior['times'])
    checks={k:bool(np.allclose(arrays[k][ix],v,equal_nan=True,rtol=1e-11,atol=1e-12)) for k,v in prior.items()}
    new=dict(np.load(directory/'candidate_inputs.npz'));old=dict(np.load(OUT/'development/candidate_inputs.npz'));ix=np.searchsorted(new['times'],old['times'])
    checks.update({'incumbent_'+k:bool(np.allclose(new[k][ix],v,equal_nan=True,rtol=1e-11,atol=1e-12)) for k,v in old.items()})
    save(HERE/'prefix_checks.json',checks);assert all(checks.values()),checks
    save(HERE/'input_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in [*directory.glob('*.json'),directory/'candidate_inputs.npz',HERE/'forecasts.npz',*list((HERE/'raw').glob('*.json'))]})

def job(arm,scenario,development=False):
    from competition_live.profile import load_profile
    from pas_performance_research.selected import controller as incumbent
    verify();directory=OUT/'development' if development else HERE/'dataset';cfg,d=load(directory)
    sc=read(HERE/'protocol.json')['costs'][scenario];p=load_profile()['trading'];model,ttl,sticky=ARMS[arm]
    f=dict(np.load(directory/'candidate_inputs.npz'));z=dict(np.load(OUT/'wave3/forecasts.npz' if development else HERE/'forecasts.npz'))
    if model=='incumbent':c=incumbent(f['times'],f['forecast'],f['carry'],f['anchor'],f['protected'],price_times=d['times']/1000,prices=d['close'],age=d['age_ms']/1000)
    else:c=VenueController(z['times'],np.nan_to_num(z[model],nan=0.),z['carry12'],z['fresh']&np.isfinite(z[model]),model,horizon=12,receipt=.5,impact_bps=sc['impact_bps'])
    tape=[]
    for symbol in SYMBOLS:
        for b in read(directory/'raw'/f'pas_candles_{symbol}.json')['data']:
            at=b['t']+60000
            if b['n']>0 and float(b['v'])>0 and cfg['start_ms']<=at<=cfg['cutoff_ms']:
                tape.append(dict(time=at,coin='pas:'+symbol,qty=float(b['v']),vwap=float(b['c']),buy_price=float(b['h']),sell_price=float(b['l'])))
    r=simulate(cfg,d,read(directory/'events.json'),tape,read(directory/'funding.json'),p,sc,controller=c,entry_lifetime=ttl,economic_limits=model!='incumbent',sticky_exits=sticky)
    phase='corrected92h' if development else 'extended97h';row=dict(arm=arm,scenario=scenario,phase=phase,**r['metrics'])
    if not development:
        a=next(s for s in r['snapshots'] if s['time']==NEW_START*1000);b=r['snapshots'][-1];unitcost=(sc['fee_bps']+sc['impact_bps'])/10000
        row.update(new_outcome_start=NEW_START,new_outcome_end=END,tail_pnl=b['equity']-a['equity'],tail_liquidation_adjusted_pnl=b['equity']-b['notional']*unitcost-a['equity']+a['notional']*unitcost,
            tail_fees=b['fees']-a['fees'],tail_funding=b['funding']-a['funding'],tail_fills=sum(x['time']>NEW_START for x in r['fills']),tail_start_notional=a['notional'],tail_end_notional=b['notional'])
        prior=gzip.open(HERE/'development/runs'/f'{arm}_corrected92h_{scenario}.json.gz','rt')
        with prior:previous=json.load(prior)
        for field in ['fills','daily','decisions','snapshots']:
            prefix=[x for x in r[field] if (x['time'] if field in ['fills','daily'] else x['time']/1000)<=DEV_END]
            assert prefix==previous[field],f'Changed pre-tail {field}: {arm}/{scenario}'
        row['path_prefix_identical']=True
    r['controller_trace']=c.trace;persist(HERE/('development' if development else 'evaluation'),row,r);return row

def run(development=False):
    verify();dest=HERE/('development' if development else 'evaluation');assert not (dest/'started.json').exists(),'Do not silently rerun'
    save(dest/'started.json',dict(time=datetime.now(timezone.utc).isoformat()));rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,s,development) for a in ARMS for s in ['base','stress','no_receipts']]):
            row=f.result();rows.append(row);save(dest/'results.json',rows)
            print({k:row[k] for k in ['arm','scenario','net_pnl','Q']+([] if development else ['tail_pnl','tail_liquidation_adjusted_pnl'])},flush=True)
    verify()

if __name__=='__main__':
    if sys.argv[1]=='development':run(True)
    else:globals()[sys.argv[1]]()
