"""Read-only, resumable public Hyperliquid testnet wallet snapshot.

Run with --output and --cache directories. Never accesses keys or /exchange.
History pages and prior snapshots remain available for audit.
"""
import argparse
import concurrent.futures
import datetime as dt
import json
import math
from pathlib import Path
import threading
import time
import urllib.request

USER = '0x125778065a578d598d88bddaa3fdd635280d0e47'
PRIORITY = ['0xdd0da0d42b4d19c8ebe86e8f3d33e41c2ef75f84',
            '0xffac3f6faa48163bfe7c6ea6207bdbab34139c3f',
            '0x06a336acf9d4ed8fae9e60c453fff4ecda409a11',
            '0x74ca48ffc4ce36072057609f6d2e1732fece948a', USER]
START = 1788220800000
LOCK = threading.Lock()
SPENT = []

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)

def reserve(weight):
    while True:
        with LOCK:
            t = time.monotonic()
            SPENT[:] = [s for s in SPENT if t - s[0] < 61]
            if sum(s[1] for s in SPENT) + weight <= 900:
                slot = [t, weight]
                SPENT.append(slot)
                return slot
        time.sleep(.5)

def api(payload):
    base = 2 if payload['type'] in ('clearinghouseState', 'spotClearinghouseState') else 20
    extra = 100 if payload['type'] in ('userFillsByTime', 'userFunding', 'candleSnapshot') else 0
    for attempt in range(3):
        slot = reserve(base + extra)
        sent = now()
        try:
            req = urllib.request.Request('https://api.hyperliquid-testnet.xyz/info',
                data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.load(response)
            with LOCK:
                slot[1] = base + (math.ceil(len(data) / 20) if extra and isinstance(data, list) else 0)
            return {'request': payload, 'sent_at_utc': sent, 'received_at_utc': now(), 'response': data}
        except Exception as exc:
            error = str(exc)
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    return {'request': payload, 'sent_at_utc': sent, 'received_at_utc': now(), 'error': error}

def rowkey(row, kind):
    if kind == 'userFillsByTime':
        return row['coin'], row['time'], row['tid']
    return json.dumps(row, sort_keys=True)

def history(kind, address, previous, end):
    field = {'userFillsByTime': 'fills', 'userFunding': 'funding',
             'userNonFundingLedgerUpdates': 'ledger'}[kind]
    cached = previous.get(field, {})
    usable = cached.get('pagination_complete', False)
    cursor = max(START, previous['history_end_ms'] - 1) if usable else START
    rows = {rowkey(r, kind): r for r in cached.get('rows', []) if r['time'] <= end}
    pages, returned = [], set()
    complete, note = False, ''
    for _ in range(80):
        payload = {'type': kind, 'user': address, 'startTime': cursor, 'endTime': end}
        if kind == 'userFillsByTime':
            payload['aggregateByTime'] = False
        page = api(payload)
        pages.append(page)
        result = page.get('response')
        if not isinstance(result, list):
            note = 'request failed'
            break
        before = len(returned)
        for row in result:
            if row['time'] <= end:
                k = rowkey(row, kind)
                returned.add(k)
                rows[k] = row
        limit = 2000 if kind == 'userFillsByTime' else 500
        if len(result) < limit:
            complete = True
            break
        nxt = max(r['time'] for r in result)
        if nxt <= cursor and len(returned) == before:
            note = 'pagination stalled'
            break
        cursor = nxt
    else:
        note = 'pagination bound reached'
    return {'pages': pages, 'rows': sorted(rows.values(), key=lambda r: r['time']),
            'pagination_complete': complete, 'note': note,
            'cached_history_included': bool(cached.get('rows')),
            'incremental_from_ms': max(START, previous['history_end_ms'] - 1) if usable else START,
            'distinct_api_rows': len(returned),
            'retention_limit_possible': kind == 'userFillsByTime' and len(returned) >= 10000}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--capture', type=Path)
    args = parser.parse_args()
    root = args.output / 'raw'
    wallets_dir = root / 'wallets'
    wallets_dir.mkdir(parents=True, exist_ok=True)
    sources, old = {}, {}
    def add(address, source):
        if isinstance(address, str) and address.startswith('0x') and len(address) == 42 and int(address, 16):
            sources.setdefault(address.lower(), set()).add(source)
    for path in (args.cache / 'wallets').glob('*.json'):
        value = json.loads(path.read_text())
        old[value['address']] = value
        add(value['address'], 'previous public snapshot')
    add(USER, 'user supplied address')
    if args.capture and args.capture.exists():
        for line in args.capture.open():
            event = json.loads(line)
            if event.get('venue') == 'par' and event.get('channel') == 'trades':
                for trade in event.get('data', []):
                    for address in trade.get('users', []):
                        add(address, '2026-10-15 public websocket trade')
    if (root / 'market_now.json').exists():
        market = json.loads((root / 'market_now.json').read_text())
    else:
        market = api({'type': 'metaAndAssetCtxs', 'dex': 'par'})
        if not isinstance(market.get('response'), list):
            raise RuntimeError(market)
        market['queried_at_ms'] = int(time.time() * 1000)
        save(root / 'market_now.json', market)
    end = market['queried_at_ms']
    if not (root / 'spot_meta.json').exists():
        save(root / 'spot_meta.json', api({'type': 'spotMeta'}))
    spot_meta = json.loads((root / 'spot_meta.json').read_text())['response']
    token_names = {t['index']: t['name'] for t in spot_meta['tokens']}
    usdpr_pairs = [u['name'] for u in spot_meta['universe']
                   if [token_names[t] for t in u['tokens']] == ['USDPR', 'USDC']]
    coins = [u['name'] for u in market['response'][0]['universe']] + usdpr_pairs
    recent_path = root / 'recent_trades.json'
    if not recent_path.exists():
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            recent = dict(zip(coins, pool.map(lambda c: api({'type': 'recentTrades', 'coin': c}), coins)))
        save(recent_path, recent)
    recent = json.loads(recent_path.read_text())
    for response in recent.values():
        for trade in response.get('response', []):
            for address in trade.get('users', []):
                add(address, 'refreshed public recentTrades')
    def collect(address):
        path = wallets_dir / (address + '.json')
        if path.exists():
            return json.loads(path.read_text())
        previous = old.get(address, {})
        value = {'address': address, 'discovery_sources': sorted(sources[address]),
                 'history_start_ms': START, 'history_end_ms': end, 'started_at_utc': now(),
                 'cache_source': str(args.cache / 'wallets' / (address + '.json')) if previous else None}
        value['fills'] = history('userFillsByTime', address, previous, end)
        value['ledger'] = history('userNonFundingLedgerUpdates', address, previous, end)
        value['par_fill_count'] = sum(f['coin'].startswith('par:') for f in value['fills']['rows'])
        if value['par_fill_count'] or address == USER:
            value['funding'] = history('userFunding', address, previous, end)
            value['positions'] = api({'type': 'clearinghouseState', 'user': address, 'dex': 'par'})
            value['orders'] = api({'type': 'frontendOpenOrders', 'user': address, 'dex': 'par'})
            value['spot'] = api({'type': 'spotClearinghouseState', 'user': address})
        else:
            value['funding'] = {'rows': [], 'pagination_complete': True,
                                'note': 'no observed par fills; funding not queried'}
        value['completed_at_utc'] = now()
        save(path, value)
        print(json.dumps({'address': address, 'par_fills': value['par_fill_count'],
              'incremental_fills': value['fills']['distinct_api_rows'],
              'gross_closed_pnl': sum(float(f.get('closedPnl', 0)) for f in value['fills']['rows'] if f['coin'].startswith('par:')),
              'errors': [v['note'] for v in (value['fills'], value['ledger'], value['funding']) if not v['pagination_complete']]}), flush=True)
        return value
    processed, rounds = set(), []
    for wave in range(8):
        todo = sorted(set(sources) - processed,
                      key=lambda a: (PRIORITY.index(a) if a in PRIORITY else 100, a))
        if not todo:
            break
        print(f'Discovery wave {wave+1}: {len(todo)} candidates; cutoff {end}', flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(collect, todo))
        processed.update(todo)
        for value in results:
            for row in value['ledger']['rows']:
                delta = row['delta']
                if delta.get('token') == 'USDPR':
                    add(delta.get('user'), 'USDPR public ledger graph')
                    add(delta.get('destination'), 'USDPR public ledger graph')
        rounds.append({'wave': wave+1, 'new_candidates': len(todo), 'known_after_wave': len(sources)})
        save(root / 'manifest.json', {'history_start_ms': START, 'history_end_ms': end,
             'candidate_count': len(sources), 'processed_count': len(processed), 'rounds': rounds,
             'sources': {a: sorted(s) for a,s in sources.items()},
             'unprocessed': sorted(set(sources)-processed), 'usdpr_spot_pairs': usdpr_pairs,
             'cache_directory': str(args.cache), 'completed_at_utc': now()})
    print(f'Collection complete: {len(processed)} wallets', flush=True)

if __name__ == '__main__':
    main()
