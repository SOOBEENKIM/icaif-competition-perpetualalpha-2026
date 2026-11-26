"""Public, unsigned audit capture. Fixed cutoff; no exchange endpoint or credentials."""
import json,time,urllib.request,pathlib,datetime
P=pathlib.Path(__file__).resolve().parents[1]/'reports/pas_strategy_audit_20261020'
P.mkdir(exist_ok=True);(P/'raw').mkdir(exist_ok=True)
START=1792411200000
MAP={'BTC':'BTC','ETH':'ETH','HYPE':'HYPE','SP500':'xyz:SP500','EWY':'xyz:EWY','EWJ':'xyz:EWJ','BRENT':'xyz:BRENTOIL','GOLD':'xyz:GOLD','USDEUR':'xyz:EUR','USDJPY':'xyz:JPY'}
TN='https://api.hyperliquid-testnet.xyz/info';MN='https://api.hyperliquid.xyz/info'
W='0x125778065a578d598D88bddaA3FdD635280d0E47'
manifest=P/'dataset.json'
if manifest.exists(): cfg=json.loads(manifest.read_text())
else:
 cfg={'start_ms':START,'warmup_start_ms':START-6*3600000,'cutoff_ms':int(time.time()//60)*60000,'symbols':list(MAP),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};manifest.write_text(json.dumps(cfg,indent=2))
assert START < cfg['cutoff_ms'] <= time.time()*1000
def fetch(name,url,body=None):
 path=P/'raw'/f'{name}.json'
 if path.exists():
  cached=json.loads(path.read_text())
  if cached['url']!=url or cached['request']!=body:raise ValueError('Cache request mismatch: '+name)
  return cached
 a=time.time();req=urllib.request.Request(url,data=json.dumps(body).encode() if body else None,headers={'Content-Type':'application/json','User-Agent':'SFINX-readonly-audit/1'})
 try:
  with urllib.request.urlopen(req,timeout=30) as f: data=json.load(f)
  obj={'url':url,'request':body,'started_ms':round(a*1000),'received_ms':round(time.time()*1000),'data':data}
  path.write_text(json.dumps(obj));print(name,'ok',len(data) if isinstance(data,(dict,list)) else '',flush=True)
  time.sleep(1.6)
  return obj
 except Exception as e:print(name,'ERROR',str(e),flush=True);raise
if __name__=='__main__':
 fetch('leaderboard','https://d23m2l9bjnbgul.cloudfront.net/public/leaderboard.json')
 fetch('chart','https://d23m2l9bjnbgul.cloudfront.net/public/chart.json')
 for key,body in [('spot',{'type':'spotClearinghouseState','user':W}),('positions',{'type':'clearinghouseState','user':W,'dex':'pas'}),('fills',{'type':'userFillsByTime','user':W,'startTime':START,'endTime':cfg['cutoff_ms']}),('user_funding',{'type':'userFunding','user':W,'startTime':START,'endTime':cfg['cutoff_ms']})]:fetch(key,TN,body)
 for key,url,dex in [('pas_context',TN,'pas'),('main_context',MN,''),('xyz_context',MN,'xyz')]:fetch(key,url,{'type':'metaAndAssetCtxs','dex':dex})
 for s,m in MAP.items():
  fetch('pas_candles_'+s,TN,{'type':'candleSnapshot','req':{'coin':'pas:'+s,'interval':'1m','startTime':cfg['warmup_start_ms'],'endTime':cfg['cutoff_ms']-1}})
  fetch('pas_funding_'+s,TN,{'type':'fundingHistory','coin':'pas:'+s,'startTime':cfg['warmup_start_ms'],'endTime':cfg['cutoff_ms']})
  fetch('pas_book_'+s,TN,{'type':'l2Book','coin':'pas:'+s})
  fetch('main_book_'+s,MN,{'type':'l2Book','coin':m})
 print('complete',cfg,flush=True)
