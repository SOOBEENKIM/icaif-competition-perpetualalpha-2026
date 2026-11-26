"""Read-only public wallet discovery carried forward from prior practice audit."""
import concurrent.futures,json,time,urllib.request
from pathlib import Path
from .data import ROOT,OUT
WALLETS=sorted(p.stem for p in (ROOT/'reports/wallet_leaderboard_20261016/raw/wallets').glob('0x*.json'))
URL='https://api.hyperliquid-testnet.xyz/info'
def post(body):
 req=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=20) as f:return json.load(f)
def one(w):
 try:
  d=post({'type':'spotClearinghouseState','user':w});b=next((x for x in d['balances'] if x['coin']=='USDP'),None)
  return {'wallet':w,'at_ms':int(time.time()*1000),'usdp_total':float(b['total']) if b else 0.}
 except Exception as e:return {'wallet':w,'error':str(e)}
def main():
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,WALLETS))
 (OUT/'peer_balances.json').write_text(json.dumps(rows,indent=2))
 candidates=sorted([x for x in rows if 100100<x.get('usdp_total',0)<110000],key=lambda x:-x['usdp_total'])[:3]
 result=[]
 for c in candidates:
  w=c['wallet'];d={**c,'identity':'Unmatched public wallet; balance alone is not verified team identity or net return.'}
  for key,body in [('positions',{'type':'clearinghouseState','user':w,'dex':'pas'}),('fills',{'type':'userFillsByTime','user':w,'startTime':1792411200000}),('funding',{'type':'userFunding','user':w,'startTime':1792411200000}),('ledger',{'type':'userNonFundingLedgerUpdates','user':w,'startTime':1792411200000})]:
   d[key]=post(body);time.sleep(.3)
  result.append(d)
 (OUT/'peer_details.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({'scanned':len(rows),'errors':sum('error'in x for x in rows),'funded_wallets':sum(x.get('usdp_total',0)>0 for x in rows),'positive_balance_candidates':candidates,'identities_confirmed':False},indent=2))
if __name__=='__main__':main()
