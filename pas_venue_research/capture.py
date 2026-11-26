"""Bounded unsigned PAS/mainnet data collection, separate from the live worker."""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path
import threading
import time
import requests
import websocket

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/pas_venue_research_20261021'
SYMBOLS=('BTC','ETH','HYPE','SP500','EWY','EWJ','BRENT','GOLD','USDEUR','USDJPY')
REFERENCES=dict(zip(SYMBOLS,('BTC','ETH','HYPE','xyz:SP500','xyz:EWY','xyz:EWJ','xyz:BRENTOIL','xyz:GOLD','xyz:EUR','xyz:JPY')))
URLS={'pas':'https://api.hyperliquid-testnet.xyz/info','main':'https://api.hyperliquid.xyz/info'}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':')))


def query(job):
    name,network,payload=job
    allowed={'metaAndAssetCtxs','l2Book','candleSnapshot','fundingHistory','perpDexLimits',
        'userNonFundingLedgerUpdates','clearinghouseState','spotClearinghouseState','userFillsByTime','userFees','perpDexs'}
    assert network in URLS and payload['type'] in allowed
    begin=time.time()
    response=requests.post(URLS[network],json=payload,timeout=20)
    response.raise_for_status()
    row=dict(request=payload,endpoint=URLS[network],started=begin,received=time.time(),response=response.json())
    save(OUT/'raw'/f'{name}.json',row)
    return name


def initial():
    OUT.mkdir(parents=True,exist_ok=True)
    profile=json.loads((ROOT/'competition_live/config.json').read_text())
    now=int(time.time()*1000);start=profile['start_unix']*1000
    jobs=[('pas_meta','pas',dict(type='metaAndAssetCtxs',dex='pas')),
          ('main_meta','main',dict(type='metaAndAssetCtxs')),
          ('xyz_meta','main',dict(type='metaAndAssetCtxs',dex='xyz')),
          ('limits','pas',dict(type='perpDexLimits',dex='pas')),
          ('own_fees','pas',dict(type='userFees',user=profile['team_wallet'])),
          ('own_ledger','pas',dict(type='userNonFundingLedgerUpdates',user=profile['team_wallet'],startTime=start,endTime=now))]
    for symbol in SYMBOLS:
        for network,coin in [('pas','pas:'+symbol),('main',REFERENCES[symbol])]:
            jobs += [(f'{network}_book_{symbol}',network,dict(type='l2Book',coin=coin)),
                     (f'{network}_candles_{symbol}',network,dict(type='candleSnapshot',req=dict(
                         coin=coin,interval='5m',startTime=now-10*86400000,endTime=now))),
                     (f'{network}_funding_{symbol}',network,dict(type='fundingHistory',
                         coin=coin,startTime=now-10*86400000,endTime=now))]
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(query,job):job[0] for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:results.append(dict(name=future.result(),ok=True))
            except Exception as error:results.append(dict(name=futures[future],ok=False,error=str(error)))
    for name,url in [
        ('leaderboard','https://d23m2l9bjnbgul.cloudfront.net/public/leaderboard.json'),
        ('chart','https://d23m2l9bjnbgul.cloudfront.net/public/chart.json'),
        ('guide','https://perpetual-alpha26.github.io/docs/'),
        ('about','https://perpetual-alpha26.github.io/about'),
        ('leaderboard_page','https://perpetual-alpha26.github.io/leaderboard')]:
        begin=time.time();response=requests.get(url,timeout=20);response.raise_for_status()
        save(OUT/'raw'/f'{name}.json',dict(url=url,started=begin,received=time.time(),
             response=response.json() if name in ('leaderboard','chart') else response.text))
    save(OUT/'initial_manifest.json',dict(at=time.time(),results=results,orders_sent=0,
         sources={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (OUT/'raw').glob('*.json')}))
    print(json.dumps(dict(completed=len(results),errors=[r for r in results if not r['ok']])),flush=True)


def stream(seconds):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'synchronized_public.jsonl.gz'
    if path.exists():raise FileExistsError('Preserve completed capture')
    stop=threading.Event();lock=threading.Lock();begin=time.time();counts={};errors=[]
    writer=gzip.open(path,'wt',compresslevel=3)
    def emit(network,message):
        with lock:
            writer.write(json.dumps(dict(network=network,received=time.time(),**message),separators=(',',':'))+'\n')
            key=network+':'+message.get('channel','status');counts[key]=counts.get(key,0)+1
    def worker(network):
        url=URLS[network].replace('https:','wss:').replace('/info','/ws')
        coins=['pas:'+s for s in SYMBOLS] if network=='pas' else list(REFERENCES.values())
        sock=None
        try:
            sock=websocket.create_connection(url,timeout=5)
            for coin in coins:
                for kind in (('l2Book','activeAssetCtx','trades') if network=='pas' else ('l2Book','activeAssetCtx')):
                    sock.send(json.dumps(dict(method='subscribe',subscription=dict(type=kind,coin=coin))))
            last_ping=time.time();last={}
            while time.time()<begin+seconds and not stop.is_set():
                try:raw=sock.recv()
                except websocket.WebSocketTimeoutException:raw=None
                now=time.time()
                if now-last_ping>20:sock.send('{"method":"ping"}');last_ping=now
                if not raw:continue
                message=json.loads(raw);channel=message.get('channel');data=message.get('data')
                if channel not in ('l2Book','activeAssetCtx','trades'):continue
                if channel!='trades':
                    key=(channel,data['coin'])
                    if now-last.get(key,0)<.5:continue
                    last[key]=now
                emit(network,message)
        except Exception as error:
            errors.append(dict(network=network,at=time.time(),error=str(error)))
            emit(network,dict(channel='capture_error',data=str(error)))
        finally:
            if sock is not None:sock.close()
    threads=[threading.Thread(target=worker,args=(network,),daemon=True) for network in URLS]
    for thread in threads:thread.start()
    while any(t.is_alive() for t in threads):
        for thread in threads:thread.join(timeout=.5)
        save(OUT/'capture_health.json',dict(at=time.time(),started=begin,deadline=begin+seconds,
            counts=counts,errors=errors,orders_sent=0))
    writer.close()
    save(OUT/'capture_manifest.json',dict(start=begin,end=time.time(),duration_seconds=seconds,
        errors=errors,counts=counts,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        orders_sent=0,signer_loaded=False))
    print(json.dumps(dict(counts=counts,errors=errors)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=('initial','stream'))
    parser.add_argument('--seconds',type=int,default=900)
    args=parser.parse_args()
    if args.kind=='initial':initial()
    else:stream(args.seconds)
