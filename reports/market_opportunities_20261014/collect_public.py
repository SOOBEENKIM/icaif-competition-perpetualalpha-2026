"""Bounded public market-data research capture. No account keys or exchange actions."""
import argparse, collections, concurrent.futures, datetime, gzip, json, pathlib
import threading, time, urllib.request
import websocket

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / 'raw'
RAW.mkdir(parents=True, exist_ok=True)
COINS = ['BTC','ETH','HYPE','SP500','EWY','EWJ','BRENT','GOLD','USDEUR','USDJPY']
REFERENCES = dict(zip(COINS, ['BTC','ETH','HYPE','xyz:SP500','xyz:EWY','xyz:EWJ','xyz:BRENTOIL','xyz:GOLD','xyz:EUR','xyz:JPY']))
LOCK = threading.Lock()
SPENT = []
COUNTS = collections.Counter()
ERRORS = []

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def reserve(weight):
    while True:
        with LOCK:
            now = time.monotonic()
            SPENT[:] = [(t,w) for t,w in SPENT if now-t < 61]
            if sum(w for t,w in SPENT)+weight <= 700:
                SPENT.append((now,weight)); return
        time.sleep(.5)

def fetch(endpoint, payload):
    reserve(100 if payload['type'] in ('candleSnapshot','fundingHistory') else 20)
    start = time.time_ns()//1000000
    for attempt in range(3):
        try:
            req = urllib.request.Request(endpoint+'/info', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req, timeout=20) as r: result=json.load(r)
            return dict(request=payload, sent_ms=start, received_ms=time.time_ns()//1000000, response=result)
        except Exception as e:
            error = str(e)
            if attempt < 2: time.sleep(attempt+1)
    return dict(request=payload, sent_ms=start, received_ms=time.time_ns()//1000000, error=error)

def histories(end):
    start = int(datetime.datetime(2026,9,25,tzinfo=datetime.timezone.utc).timestamp()*1000)
    jobs=[]
    for coin in COINS:
        jobs.append(('funding_'+coin, 'https://api.hyperliquid-testnet.xyz', dict(type='fundingHistory',coin='par:'+coin,startTime=start,endTime=end)))
        for label,endpoint,c in [('par','https://api.hyperliquid-testnet.xyz','par:'+coin),('reference','https://api.hyperliquid.xyz',REFERENCES[coin])]:
            jobs.append(('candles_'+label+'_'+coin,endpoint,dict(type='candleSnapshot',req=dict(coin=c,interval='1h',startTime=start,endTime=end))))
    def job(args):
        name,endpoint,payload=args
        result=fetch(endpoint,payload)
        (RAW/(name+'.json')).write_text(json.dumps(result))
        print(json.dumps(dict(history=name,rows=len(result.get('response',[])),error=result.get('error'))),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: list(pool.map(job,jobs))

def stream(label, endpoint, coins, channels, deadline):
    path=RAW/(label+'_stream.jsonl.gz')
    if path.exists(): raise RuntimeError('Capture already exists: '+str(path))
    with gzip.open(path,'wt',encoding='utf-8') as out:
        while time.monotonic()<deadline:
            ws=None
            try:
                ws=websocket.create_connection(endpoint+'/ws',timeout=8)
                for coin in coins:
                    for channel in channels:
                        ws.send(json.dumps(dict(method='subscribe',subscription=dict(type=channel,coin=coin))))
                last_ping=time.monotonic(); last_flush=last_ping
                while time.monotonic()<deadline:
                    if time.monotonic()-last_ping>20:
                        ws.send(json.dumps(dict(method='ping')));last_ping=time.monotonic()
                    try: message=ws.recv()
                    except websocket.WebSocketTimeoutException: continue
                    if not message: raise RuntimeError('WebSocket closed')
                    received=time.time_ns()//1000000
                    data=json.loads(message)
                    out.write(json.dumps(dict(received_ms=received,message=data),separators=(',',':'))+'\n')
                    with LOCK: COUNTS[label+':'+data.get('channel','unknown')]+=1
                    if time.monotonic()-last_flush>5:out.flush();last_flush=time.monotonic()
            except Exception as e:
                item=dict(stream=label,time=utc(),error=str(e))
                with LOCK: ERRORS.append(item)
                print(json.dumps(item),flush=True)
                time.sleep(2)
            finally:
                if ws is not None: ws.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=int,default=900);args=parser.parse_args()
    start=time.time_ns()//1000000; deadline=time.monotonic()+args.seconds
    spec=dict(started_ms=start,started_utc=utc(),planned_seconds=args.seconds,read_only=True,keys_read=False,
              practice_venue='par',reference_markets=REFERENCES,notes='Public live observation for hypothesis discovery, not a prospective strategy profitability test.')
    (RAW/'capture_manifest.json').write_text(json.dumps(spec,indent=2))
    tasks=[threading.Thread(target=histories,args=(start,)),
           threading.Thread(target=stream,args=('par','wss://api.hyperliquid-testnet.xyz',['par:'+c for c in COINS],['l2Book','trades','activeAssetCtx'],deadline)),
           threading.Thread(target=stream,args=('reference','wss://api.hyperliquid.xyz',list(REFERENCES.values()),['bbo','activeAssetCtx'],deadline))]
    for thread in tasks:thread.start()
    while any(thread.is_alive() for thread in tasks):
        for thread in tasks:thread.join(timeout=1)
        with LOCK: status=dict(spec,updated_utc=utc(),counts=dict(COUNTS),errors=list(ERRORS))
        (RAW/'capture_status.json').write_text(json.dumps(status,indent=2))
    status.update(finished_ms=time.time_ns()//1000000,finished_utc=utc())
    (RAW/'capture_status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status),flush=True)

if __name__=='__main__':main()
