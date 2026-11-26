"""Fixed-cutoff read-only public history; never call an exchange endpoint."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,time,urllib.request
from pathlib import Path
import numpy as np
from experiments.par_direct_forecast_20261017.collect import MAPPING
from experiments.complementary_alpha_20261017.research import protected

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'artifacts';RAW=OUT/'raw'
def ts(s):return int(np.datetime64(s,'s').astype('int64'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):
    Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
def load(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}

def fetch(job):
    filename,url,payload=job;path=RAW/filename
    if path.exists():
        old=json.loads(path.read_text());assert old['request']==payload and old['url']==url
        return filename,len(old['response'])
    errors=[]
    for attempt in range(3):
        sent=time.time()
        try:
            req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=25) as response:rows=json.load(response)
            if not isinstance(rows,list):raise ValueError('Unexpected public history response')
            if payload['type']=='fundingHistory' and len(rows)>=500:raise ValueError('Unexpected pagination in 26h funding request')
            save(path,dict(url=url,request=payload,sent_at=sent,received_at=time.time(),response=rows,prior_errors=errors))
            return filename,len(rows)
        except Exception as exc:
            errors.append(str(exc))
            if attempt==2:raise
            time.sleep(1+attempt)

def main():
    cfg=json.loads((HERE/'protocol.json').read_text());OUT.mkdir(exist_ok=True);RAW.mkdir(exist_ok=True)
    frozen=OUT/'precollection.json'
    sources={str(p.relative_to(ROOT)):sha(p) for p in [*HERE.glob('*.py'),HERE/'protocol.json']}
    if frozen.exists():assert json.loads(frozen.read_text())['sources']==sources
    else:save(frozen,dict(at_unix=time.time(),sources=sources,protected=protected(),protocol=cfg))
    start,end=ts(cfg['fetch_start'])*1000,ts(cfg['end'])*1000;jobs=[]
    for venue in ['par','reference']:
        url='https://api.hyperliquid-testnet.xyz/info' if venue=='par' else 'https://api.hyperliquid.xyz/info'
        for symbol,reference_coin in MAPPING.items():
            coin='par:'+symbol if venue=='par' else reference_coin
            jobs.append((f'{venue}_{symbol}_5m.json',url,dict(type='candleSnapshot',req=dict(coin=coin,interval='5m',startTime=start,endTime=end-1))))
            if venue=='par':jobs.append((f'funding_{symbol}.json',url,dict(type='fundingHistory',coin=coin,startTime=start,endTime=end)))
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(fetch,jobs))
    assert json.loads(frozen.read_text())['sources']==sources
    save(OUT/'collection.json',dict(at_unix=time.time(),requests=results,sources={str(p.relative_to(ROOT)):sha(p) for p in RAW.glob('*.json')},orders_sent=0))
    print('COLLECTED',len(results),'public requests',sum(n for _,n in results),'rows',flush=True)

if __name__=='__main__':main()
