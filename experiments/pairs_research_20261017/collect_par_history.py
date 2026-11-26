"""Collect public par history with receipts; no trading SDK or wallet credentials.

Candles are traded-price observations, never historical order books. This is a
coverage audit, not a new fitted strategy or an execution-profitability claim.
"""
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENDPOINT = 'https://api.hyperliquid-testnet.xyz/info'
PUBLIC_BOARD = 'https://d23m2l9bjnbgul.cloudfront.net/public/practice.json'
ALLOWED = {'metaAndAssetCtxs', 'candleSnapshot', 'fundingHistory', 'recentTrades', 'l2Book'}
HOUR = 3_600_000


def stamp(value):
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def iso(value):
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000, timezone(timedelta(hours=9))).isoformat(timespec='seconds')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def fetch(payload=None):
    if payload is not None:
        assert payload['type'] in ALLOWED
    url = ENDPOINT if payload is not None else PUBLIC_BOARD
    attempts = []
    for attempt in range(4):
        sent = time.time()
        request = urllib.request.Request(url, data=json.dumps(payload).encode() if payload else None,
                                         headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                body = json.load(response)
                return dict(url=url, request=payload, sent_at=sent, received_at=time.time(),
                            response=body, prior_attempts=attempts)
        except (urllib.error.URLError, TimeoutError) as exc:
            attempts.append(dict(at=sent, error=str(exc)))
            if attempt == 3:
                return dict(url=url, request=payload, sent_at=sent, received_at=time.time(),
                            error=str(exc), prior_attempts=attempts)
            time.sleep(2 ** attempt)


def candle_stats(rows, coin, interval, cutoff):
    step = {'1h': HOUR, '5m': 300_000}[interval]
    by_time = {}
    for row in rows:
        assert row['s'] == coin and row['i'] == interval
        o, h, low, c = (float(row[k]) for k in ('o', 'h', 'l', 'c'))
        assert 0 < low <= min(o, c) <= max(o, c) <= h
        assert float(row['v']) >= 0 and int(row['n']) >= 0
        assert int(row['T']) == int(row['t']) + step - 1
        if row['t'] in by_time:
            assert row == by_time[row['t']], 'Conflicting duplicate candle'
        by_time[row['t']] = row
    closed = [r for _, r in sorted(by_time.items()) if r['T'] < cutoff]
    active = [r for r in closed if float(r['v']) > 0 and int(r['n']) > 0]
    first = closed[0]['t'] if closed else None
    last = closed[-1]['T'] if closed else None
    expected = int((last + 1 - first) // step) if closed else 0
    return dict(returned=len(rows), unique=len(by_time), completed=len(closed), with_trades=len(active),
                first_kst=iso(first), last_closed_kst=iso(last),
                missing_intervals_within_observed_span=expected - len(closed),
                active_fraction_of_observed_span=len(active) / expected if expected else None,
                total_reported_trades=sum(int(r['n']) for r in closed),
                unfinished_candles=len(by_time) - len(closed),
                latest_closed_age_minutes=(cutoff - last) / 60_000 if last is not None else None)


def main():
    cutoff = int(time.time() * 1000)
    out = HERE / 'par_history' / datetime.fromtimestamp(cutoff / 1000, timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=False)
    starts = {'1h': stamp('2026-08-01T00:00:00+00:00'),
              '5m': stamp('2026-10-13T00:00:00+00:00'),
              'funding': stamp('2026-08-01T00:00:00+00:00')}
    save(out / 'plan.json', dict(cutoff_ms=cutoff, cutoff_kst=iso(cutoff), starts=starts,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        purpose='Public data coverage audit, not a strategy-selection experiment',
        live_orders_sent=0, sealed_forward_data_accessed=False,
        candle_policy='Only T < cutoff is complete; missing bars are not forward-filled',
        current_snapshot_policy='Requests completed after the cutoff are context only, not historical inputs'))
    meta = fetch({'type': 'metaAndAssetCtxs', 'dex': 'par'})
    save(out / 'metadata.json', meta)
    assert 'response' in meta, meta
    universe = meta['response'][0]['universe']
    symbols = [u['name'] for u in universe]
    assert all(s.startswith('par:') for s in symbols)
    board = fetch()
    save(out / 'practice_leaderboard.json', board)
    rows, failures = [], []
    for coin in symbols:
        label = coin.split(':')[1]
        row = {'coin': coin}
        for interval in ('1h', '5m'):
            receipt = fetch(dict(type='candleSnapshot', req=dict(coin=coin, interval=interval,
                              startTime=starts[interval], endTime=cutoff)))
            save(out / f'candles_{label}_{interval}.json', receipt)
            if isinstance(receipt.get('response'), list):
                row[interval] = candle_stats(receipt['response'], coin, interval, cutoff)
            else:
                row[interval] = {'error': receipt.get('error', 'unexpected response')}
                failures.append(f'{coin}:{interval}')
            time.sleep(0.8)
        pages, funding, cursor = [], {}, starts['funding']
        complete = False
        for _ in range(32):
            receipt = fetch(dict(type='fundingHistory', coin=coin, startTime=cursor, endTime=cutoff))
            pages.append(receipt)
            page = receipt.get('response')
            if not isinstance(page, list):
                failures.append(f'{coin}:funding')
                break
            for value in page:
                assert value['coin'] == coin and cursor <= value['time'] <= cutoff
                if value['time'] in funding:
                    assert funding[value['time']] == value
                funding[value['time']] = value
            if len(page) < 500:
                complete = True
                break
            next_cursor = max(v['time'] for v in page) + 1
            assert next_cursor > cursor
            cursor = next_cursor
            time.sleep(0.8)
        assert complete or failures, 'Funding pagination exceeded the safety bound'
        ordered = [funding[t] for t in sorted(funding)]
        save(out / f'funding_{label}.json', dict(pages=pages, rows=ordered, pagination_complete=complete))
        row['funding'] = dict(observations=len(ordered), pagination_complete=complete,
                             first_kst=iso(ordered[0]['time']) if ordered else None,
                             last_kst=iso(ordered[-1]['time']) if ordered else None,
                             gaps_over_one_hour=sum(b['time'] - a['time'] > HOUR + 10_000
                                                    for a, b in zip(ordered, ordered[1:])))
        for kind in ('recentTrades', 'l2Book'):
            receipt = fetch(dict(type=kind, coin=coin))
            save(out / f'{kind}_{label}.json', receipt)
            if 'error' in receipt:
                failures.append(f'{coin}:{kind}')
            if kind == 'recentTrades' and isinstance(receipt.get('response'), list):
                trades = [t for t in receipt['response'] if t['time'] <= cutoff]
                row[kind] = dict(returned=len(receipt['response']), before_cutoff=len(trades),
                    first_kst=iso(min(t['time'] for t in trades)) if trades else None,
                    last_kst=iso(max(t['time'] for t in trades)) if trades else None,
                    complete_historical_trade_tape=False)
            time.sleep(0.8)
        rows.append(row)
        save(out / 'coverage_progress.json', rows)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    manifest = dict(cutoff_ms=cutoff, cutoff_kst=iso(cutoff), collected_through_kst=iso(int(time.time() * 1000)),
                    rows=rows, failures=failures, live_orders_sent=0,
                    historical_order_book_available=False,
                    limitations=['Candle close is a trade price, not a synchronized executable quote',
                        'recentTrades is a limited recent tape, not the complete market history',
                        'Funding history has rates and premium, not historical oracle prices',
                        'Current books and asset contexts cannot reconstruct past liquidity',
                        'Live strategies started on different dates, so their realized returns are not matched experiments'],
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(out.glob('*.json'))})
    save(out / 'manifest.json', manifest)
    print('OUTPUT', out, flush=True)


if __name__ == '__main__':
    main()
