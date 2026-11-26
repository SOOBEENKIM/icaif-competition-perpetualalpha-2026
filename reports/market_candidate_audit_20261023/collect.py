"""Read-only public snapshots. Sequential pacing, no retries; stops on rate limit."""
import datetime as dt, json, time, sys, copy
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent; RAW=OUT/'raw'; OLD=ROOT/'reports/market_peer_refresh_20261022_2348/raw'
KST=dt.timezone(dt.timedelta(hours=9)); USER='0x125778065a578d598d88bddaa3fdd635280d0e47'; START=1792411200000
SESSION=requests.Session(); LAST=0

def now(): return dt.datetime.now(KST).isoformat()
def save(p,d): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def api(req):
 global LAST
 time.sleep(max(0,1.8-(time.monotonic()-LAST))); sent=now(); LAST=time.monotonic()
 r=SESSION.post('https://api.hyperliquid-testnet.xyz/info',json=req,timeout=30)
 if r.status_code!=200:
  save(RAW/'error.json',dict(request=req,status=r.status_code,at=now(),text=r.text[:1000]));r.raise_for_status()
 return dict(request=req,sent_kst=sent,received_kst=now(),response=r.json())
def public(name):
 url='https://d23m2l9bjnbgul.cloudfront.net/public/'+name+'.json'
 r=SESSION.get(url,timeout=30);r.raise_for_status();d=dict(url=url,received_kst=now(),headers=dict(r.headers),response=r.json());save(RAW/(name+'.json'),d);return d

def merged_history(old,typ,a,end):
 # Overlap 1 second with previously complete coverage; merge by content identity.
 start=max(START,int(old.get('end_ms',START))-1000)
 pages=[];rows=[];complete=False;limit=2000 if typ=='userFillsByTime' else 500
 for _ in range(5):
  req=dict(type=typ,user=a,startTime=start,endTime=end)
  if typ=='userFillsByTime':req['aggregateByTime']=False
  page=api(req);pages.append(page);rs=page['response']
  if not isinstance(rs,list):raise ValueError((typ,rs))
  rows+=rs
  if len(rs)<limit:complete=True;break
  latest=max(x['time'] for x in rs)
  if latest<=start:break
  start=latest # inclusive, deduplicated below
 combined={json.dumps(r,sort_keys=True):r for r in old.get('rows',[])+rows}
 return dict(rows=sorted(combined.values(),key=lambda r:r['time']),pages=pages,pagination_complete=bool(old.get('pagination_complete',True)) and complete,retention_limit_possible=old.get('retention_limit_possible',False),incremental_rows=len(rows),end_ms=end,prior_source=old.get('prior_source'))

def wallet(a,manifest):
 dst=RAW/'wallets'/(a+'.json')
 if dst.exists():return
 prior=json.loads((OLD/'wallets'/(a+'.json')).read_text());w=dict(address=a,grants=manifest['grant_wallets'][a],started_kst=now(),prior_source=str(OLD/'wallets'/(a+'.json')))
 ex=dict(address=a,mode=api(dict(type='userAbstraction',user=a)));save(RAW/'extra'/(a+'.json'),ex)
 w['positions']=api(dict(type='clearinghouseState',user=a,dex='pas'))
 w['spot']=api(dict(type='spotClearinghouseState',user=a))
 w['orders']=api(dict(type='frontendOpenOrders',user=a,dex='pas'))
 end=int(time.time()*1000); w['history_end_ms']=end
 for key,typ in [('fills','userFillsByTime'),('funding','userFunding'),('ledger','userNonFundingLedgerUpdates')]:
  old=copy.deepcopy(prior[key]);old['end_ms']=prior['history_end_ms'];old['prior_source']=w['prior_source']
  w[key]=merged_history(old,typ,a,end)
 w['completed_kst']=now();save(dst,w)
 usdp=next((x['total'] for x in w['spot']['response']['balances'] if x['coin']=='USDP'),'0')
 print(json.dumps(dict(address=a,completed=w['completed_kst'],USDP=usdp,positions=len(w['positions']['response']['assetPositions']),fills=len(w['fills']['rows']),newfillrows=w['fills']['incremental_rows'])),flush=True)

def main():
 manifest=json.loads((OLD/'manifest.json').read_text());save(RAW/'manifest.json',dict(created_kst=now(),phase_start_ms=START,grant_wallets=manifest['grant_wallets'],roster_source=str(OLD/'manifest.json'),scope='47 previously verified grant recipients; no claim to exhaust all newer registrations'))
 public('leaderboard');public('chart');print('OFFICIAL_READY',flush=True)
 wallet(USER,manifest)
 if not (RAW/'market.json').exists():save(RAW/'market.json',api(dict(type='metaAndAssetCtxs',dex='pas')))
 market=json.loads((RAW/'market.json').read_text())['response'];symbols=[x['name'] for x in market[0]['universe']];end=int(time.time()*1000)
 for coin in symbols:
  p=RAW/'markets'/(coin.replace(':','_')+'.json')
  if p.exists():continue
  d=dict(coin=coin,book=api(dict(type='l2Book',coin=coin)),candles=api(dict(type='candleSnapshot',req=dict(coin=coin,interval='1h',startTime=START,endTime=end))),recent_trades=api(dict(type='recentTrades',coin=coin)),funding_history=api(dict(type='fundingHistory',coin=coin,startTime=START,endTime=end)));save(p,d)
 r=SESSION.post('https://api.hyperliquid.xyz/info',json={'type':'metaAndAssetCtxs'},timeout=30);r.raise_for_status();save(RAW/'mainnet_market.json',dict(received_kst=now(),response=r.json()))
 print('MARKETS_READY',flush=True)
 # Previously profitable, mixed-direction, high-turnover and losing comparators first.
 first=['0xcd71d468ddc1bc65cac67cf6e078b260bb0128be','0xc7d3adfe9a5164a041bedf222f1ded32f2c34b18','0xdd0da0d42b4d19c8ebe86e8f3d33e41c2ef75f84','0x790899ee522f4646c40bf11ef3b5c8d8fc0ac387','0xde13132f77965e429b01a4b019eaa23061e9b3a6','0x06a336acf9d4ed8fae9e60c453fff4ecda409a11']
 for a in dict.fromkeys(first+list(manifest['grant_wallets'])):wallet(a,manifest)
 public('leaderboard');public('chart');print('COMPLETE',now(),flush=True)
if __name__=='__main__':main()
