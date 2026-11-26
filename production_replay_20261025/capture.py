"""Public PAS feed only. No signer, account mutation, or order endpoint."""
import argparse,gzip,json,time,shutil
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import websocket

SYMBOLS=['BTC','ETH','HYPE','SP500','EWY','EWJ','BRENT','GOLD','USDEUR','USDJPY']
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/production_replay_20261025/capture'

def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value));tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=21600);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'start.json').exists():raise FileExistsError('Do not merge captures silently')
    start=time.time();end=start+args.seconds
    save(OUT/'start.json',dict(start=start,end=end,venue='pas',symbols=SYMBOLS,orders_sent=0,
        sampling='At most one book/context per coin per second. Every received trade recorded. Gaps/reconnects explicit.',
        purpose='Build actual depth/mark history while historical development continues; not itself a profitability test.'))
    sock=None;writer=None;bucket=None;epoch=0;last_ping=0;health=0;last_emit={};counts=Counter();errors=[]
    def emit(channel,data,throttle=True):
        nonlocal writer,bucket
        now=time.time();coin=data.get('coin','') if isinstance(data,dict) else ''
        if throttle and channel in ('l2Book','activeAssetCtx'):
            key=(channel,coin)
            if now-last_emit.get(key,0)<1:return
            last_emit[key]=now
        b=datetime.fromtimestamp(now,timezone.utc).strftime('%Y%m%d_%H')
        if b!=bucket:
            if writer:writer.close()
            writer=gzip.open(OUT/(b+'.jsonl.gz'),'at',compresslevel=3);bucket=b
        writer.write(json.dumps(dict(received_ms=int(now*1000),epoch=epoch,channel=channel,data=data),separators=(',',':'))+'\n')
        counts[channel]+=1
    try:
        while time.time()<end and not (OUT/'STOP').exists():
            if shutil.disk_usage(OUT).free<2*1024**3:raise RuntimeError('Disk reserve')
            try:
                if sock is None:
                    sock=websocket.create_connection('wss://api.hyperliquid-testnet.xyz/ws',timeout=5)
                    epoch+=1;last_emit.clear();emit('connection',dict(state='connected'))
                    for symbol in SYMBOLS:
                        for channel in ['l2Book','activeAssetCtx','trades']:
                            sock.send(json.dumps(dict(method='subscribe',subscription=dict(type=channel,coin='pas:'+symbol))))
                if time.time()-last_ping>20:sock.send(json.dumps(dict(method='ping')));last_ping=time.time()
                try:msg=json.loads(sock.recv())
                except websocket.WebSocketTimeoutException:continue
                if msg.get('channel') in ['l2Book','activeAssetCtx','trades']:
                    emit(msg['channel'],msg['data'])
                elif msg.get('channel')=='error':raise RuntimeError(str(msg['data']))
                if time.time()-health>30:
                    if writer:writer.flush()
                    save(OUT/'health.json',dict(at=time.time(),end=end,counts=dict(counts),epochs=epoch,errors=errors[-10:],orders_sent=0))
                    print(dict(seconds=int(time.time()-start),counts=dict(counts)),flush=True);health=time.time()
            except (websocket.WebSocketException,OSError,ValueError) as e:
                errors.append(dict(time=time.time(),error=type(e).__name__+': '+str(e)))
                emit('connection',dict(state='disconnected',error=errors[-1]['error']))
                if sock:sock.close()
                sock=None;time.sleep(2)
    finally:
        if writer:writer.close()
        if sock:sock.close()
        save(OUT/'finished.json',dict(at=time.time(),counts=dict(counts),errors=errors,orders_sent=0))

if __name__=='__main__':main()
