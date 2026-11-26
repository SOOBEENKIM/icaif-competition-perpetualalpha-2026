"""Read-only cutoff capture for the user's 15:30 KST performance report."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
import time

import websocket

from scripts.refresh_public_wallets import api, history

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = OUT / 'raw'
END = int(datetime.fromisoformat('2026-10-16T15:30:00+09:00').timestamp()*1000)
USER = '0x125778065a578d598D88bddaA3FdD635280d0E47'


def save(name, value):
    path = RAW / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def main():
    RAW.mkdir(exist_ok=True)
    db = sqlite3.connect('file:' + str(ROOT/'practice_live/live_runtime/state.sqlite3') + '?mode=ro', uri=True)
    anchor = json.loads(db.execute("select value from kv where key='anchor'").fetchone()[0])
    records = [json.loads(r) for r, in db.execute("select raw from records where kind='fills'")]
    db.close()
    start = min(f['time'] for f in records)
    save('measurement.json', {'start_ms': start, 'end_ms': END, 'initial_equity': anchor['initial_equity'],
         'team_wallet': USER, 'start_definition': 'Immediately before the first actual fill; include that fill and its fee.',
         'cutoff_definition': 'Fills/funding/capital flows timestamped at or before the cutoff; last public context received at or before cutoff.',
         'orders_sent': 0})
    meta = api({'type': 'metaAndAssetCtxs', 'dex': 'par'})
    save('metadata.json', meta)
    coins = [m['name'] for m in meta['response'][0]['universe']]
    latest, counts, errors = {}, {}, []
    stop = threading.Event()

    def account_window():
        samples = []
        while not stop.is_set() and time.time()*1000 < END + 5000:
            now = time.time()*1000
            if now >= END - 15000:
                sample = api({'type': 'clearinghouseState', 'user': USER, 'dex': 'par'})
                samples.append(sample)
                save('account_cutoff_window.json', samples)
                stop.wait(.5)
            else:
                stop.wait(min(2, (END-15000-now)/1000))
        save('open_orders_after_cutoff.json', api({'type': 'frontendOpenOrders', 'user': USER, 'dex': 'par'}))
        save('spot_after_cutoff.json', api({'type': 'spotClearinghouseState', 'user': USER}))

    worker = threading.Thread(target=account_window)
    worker.start()
    ws, health_at, last_ping = None, 0., 0.
    with (RAW/'contexts.jsonl').open('a') as stream:
        while time.time()*1000 < END+5000:
            try:
                if ws is None:
                    ws = websocket.create_connection('wss://api.hyperliquid-testnet.xyz/ws', timeout=1)
                    for coin in coins:
                        ws.send(json.dumps({'method': 'subscribe', 'subscription': {'type': 'activeAssetCtx', 'coin': coin}}))
                if time.time()-last_ping >= 20:
                    ws.send(json.dumps({'method': 'ping'}));last_ping = time.time()
                message = json.loads(ws.recv())
                received = int(time.time()*1000)
                if message.get('channel') == 'activeAssetCtx':
                    row = {'received_ms': received, 'data': message['data']}
                    stream.write(json.dumps(row,separators=(',', ':'))+'\n')
                    coin = message['data']['coin']
                    counts[coin] = counts.get(coin, 0)+1
                    if received <= END:
                        latest[coin] = row
                elif message.get('channel') == 'error':
                    raise ValueError(message['data'])
            except websocket.WebSocketTimeoutException:
                pass
            except Exception as exc:
                errors.append({'time': time.time(), 'error': str(exc)})
                if ws:
                    ws.close()
                ws = None;time.sleep(1)
            if time.time()-health_at >= 10:
                stream.flush()
                save('capture_health.json', {'state':'capturing','at':time.time(),'cutoff_ms':END,
                     'counts':counts,'last_context_times':{c:r['received_ms'] for c,r in latest.items()},'recent_errors':errors[-10:]})
                health_at = time.time()
        if ws:
            ws.close()
    stop.set();worker.join(timeout=120)
    if worker.is_alive():
        raise RuntimeError('Account cutoff worker did not finish')
    save('cutoff_contexts.json', latest)
    for kind, name in [('userFillsByTime','fills'),('userFunding','funding'),('userNonFundingLedgerUpdates','transfers')]:
        result = history(kind, USER, {}, END)
        save(name+'.json', result)
        if not result['pagination_complete']:
            raise RuntimeError('Incomplete account history: '+name)
    account = json.loads((ROOT/'practice_live/live_runtime/account_latest.json').read_text())
    save('bot_account_after_cutoff.json', account)
    save('capture_health.json', {'state':'complete','at':time.time(),'cutoff_ms':END,'counts':counts,
         'context_age_at_cutoff_ms':{c:END-r['received_ms'] for c,r in latest.items()},'orders_sent':0})
    print(json.dumps({'state':'cutoff_capture_complete','end_ms':END,'orders_sent':0}),flush=True)


if __name__ == '__main__':
    main()
