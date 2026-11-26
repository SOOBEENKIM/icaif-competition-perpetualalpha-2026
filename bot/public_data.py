"""Credential-free public capture and historical funding queries for par."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time

import requests
import websocket

from .market_data import SYMBOLS

URL = 'https://api.hyperliquid-testnet.xyz/info'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def info(payload):
    if payload.get('type') not in ('meta', 'fundingHistory'):
        raise ValueError('Only public metadata and funding queries allowed')
    r = requests.post(URL, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


def funding(start, end, output):
    def get(coin):
        result = []
        cursor = start
        while cursor < end:
            rows = info(dict(type='fundingHistory', coin='par:'+coin, startTime=cursor, endTime=end))
            if not isinstance(rows, list):
                raise ValueError('Unexpected funding response')
            if not rows:
                break
            result.extend(rows)
            nxt = max(int(r['time']) for r in rows) + 1
            if nxt <= cursor:
                raise ValueError('Funding pagination failed to advance')
            cursor = nxt
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = [row for part in pool.map(get, SYMBOLS) for row in part]
    write(Path(output), dict(start_ms=start, end_ms=end, fetched_ms=time.time_ns()//1000000, records=records))
    print(json.dumps(dict(records=len(records), output=output)), flush=True)


def capture(seconds, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise RuntimeError('Refuse to overwrite capture')
    meta = info(dict(type='meta', dex='par'))
    write(out/'metadata.json', meta)
    actual = {r['name'].removeprefix('par:') for r in meta['universe'] if not r.get('isDelisted')}
    if not set(SYMBOLS) <= actual:
        raise ValueError('Competition universe changed')
    start = time.time_ns()//1000000
    manifest = dict(start_ms=start, duration_seconds=seconds, symbols=SYMBOLS,
                    purpose='Blind public observation; no model, account, credential or order access',
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), actual_orders_sent=0)
    write(out/'manifest.json', manifest)
    stop = threading.Event()
    counts, errors, last_rx = {}, [], {}
    lock = threading.Lock()
    tape = (out/'events.jsonl').open('x',buffering=1)
    def log(channel, data):
        with lock:
            t = time.time_ns()//1000000
            tape.write(json.dumps(dict(received_ms=t,venue='par',channel=channel,data=data),separators=(',',':'))+'\n')
            counts[channel] = counts.get(channel,0)+1
            last_rx[channel] = t
    def worker():
        while not stop.is_set():
            ws = None
            try:
                ws = websocket.create_connection('wss://api.hyperliquid-testnet.xyz/ws',timeout=2)
                log('connected',{})
                for coin in SYMBOLS:
                    for kind in ('l2Book','activeAssetCtx','trades'):
                        ws.send(json.dumps(dict(method='subscribe',subscription=dict(type=kind,coin='par:'+coin))))
                ping = time.monotonic()
                while not stop.is_set():
                    if time.monotonic()-ping>=15:
                        ws.send('{"method":"ping"}')
                        ping=time.monotonic()
                    try:
                        opcode,frame=ws.recv_data_frame(control_frame=True)
                    except websocket.WebSocketTimeoutException:
                        continue
                    if opcode==websocket.ABNF.OPCODE_CLOSE:
                        raise RuntimeError('WebSocket closed')
                    if opcode!=websocket.ABNF.OPCODE_TEXT:
                        continue
                    message=json.loads(frame.data)
                    if message.get('channel')=='error':
                        raise RuntimeError(str(message.get('data')))
                    log(message['channel'],message.get('data'))
            except Exception as exc:
                if not stop.is_set():
                    error=dict(time_ms=time.time_ns()//1000000,error=str(exc))
                    with lock:
                        errors.append(error)
                    log('disconnected',error)
                    stop.wait(2)
            finally:
                if ws:
                    ws.close()
        log('stopped',{})
    thread=threading.Thread(target=worker)
    thread.start()
    print(json.dumps(manifest),flush=True)
    try:
        while time.time_ns()//1000000 < start+seconds*1000:
            with lock:
                state=dict(manifest,finished=False,now_ms=time.time_ns()//1000000,
                           counts=dict(counts),last_rx=dict(last_rx),errors=list(errors))
            write(out/'status.json',state)
            stop.wait(5)
    finally:
        stop.set()
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError('Public capture failed to stop')
        tape.close()
        write(out/'status.json',dict(manifest,finished=True,end_ms=time.time_ns()//1000000,counts=counts,errors=errors))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='action',required=True)
    f=sub.add_parser('funding')
    f.add_argument('--start',type=int,required=True)
    f.add_argument('--end',type=int,required=True)
    f.add_argument('--output',required=True)
    c=sub.add_parser('capture')
    c.add_argument('--seconds',type=int,required=True)
    c.add_argument('--output',required=True)
    args=p.parse_args()
    if args.action=='funding':
        funding(args.start,args.end,args.output)
    elif 60<=args.seconds<=86400:
        capture(args.seconds,args.output)
    else:
        raise ValueError('Capture must be bounded to 60–86400 seconds')
