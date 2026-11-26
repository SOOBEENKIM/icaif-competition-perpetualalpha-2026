"""Download existing historical public candles only; no wallet or trading API."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,time,urllib.request,hashlib
import numpy as np

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts';RAW=OUT/'raw'
MAPPING={'BTC':'BTC','ETH':'ETH','HYPE':'HYPE','SP500':'xyz:SP500','EWY':'xyz:EWY','EWJ':'xyz:EWJ',
         'BRENT':'xyz:BRENTOIL','GOLD':'xyz:GOLD','USDEUR':'xyz:EUR','USDJPY':'xyz:JPY'}
def ts(s):return int(np.datetime64(s,'s').astype('int64'))
def save(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def fetch(item):
    venue,symbol,coin=item;path=RAW/f'{venue}_{symbol}_5m.json'
    if path.exists():return venue,symbol,json.loads(path.read_text())
    url='https://api.hyperliquid-testnet.xyz/info' if venue=='par' else 'https://api.hyperliquid.xyz/info'
    payload=dict(type='candleSnapshot',req=dict(coin=coin,interval='5m',startTime=ts('2026-09-30T00:00:00')*1000,endTime=ts('2026-10-16T12:00:00')*1000-1))
    errors=[]
    for attempt in range(3):
        sent=time.time()
        try:
            req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=25) as f:rows=json.load(f)
            if not isinstance(rows,list):raise ValueError('Unexpected candle response')
            o=dict(url=url,request=payload,sent_at=sent,received_at=time.time(),response=rows,prior_errors=errors)
            save(path,o);return venue,symbol,o
        except Exception as e:
            errors.append(str(e))
            if attempt==2:raise
            time.sleep(1+attempt)

def main():
    RAW.mkdir(parents=True,exist_ok=True)
    items=[('par',s,'par:'+s) for s in MAPPING]+[('reference',s,c) for s,c in MAPPING.items()]
    results=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for venue,symbol,o in pool.map(fetch,items):
            r=o['response'];result=dict(venue=venue,symbol=symbol,rows=len(r),first=int(r[0]['t']) if r else None,
                last=int(r[-1]['T']) if r else None,active=sum(float(x['v'])>0 for x in r))
            results.append(result);print(result,flush=True)
    save(OUT/'collection.json',dict(at_unix=time.time(),rows=results,orders_sent=0,
        sources={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in RAW.glob('*.json')}))

if __name__=='__main__':main()
