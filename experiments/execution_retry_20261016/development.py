"""Fixed historical execution probes. Deliberately not called an OOS strategy test."""
import bisect
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .engine import Paper
from practice_live.refresh import load_inputs
from replay_lab.common_comparison.models import Market, covariance
from replay_lab.ema_rebuild.engine import Controller

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P = json.loads((HERE / 'protocol.json').read_text())


def save(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))


def events_from_history():
    events, manifest, books_seen = [], [], set()
    for path in sorted((ROOT / 'reports').rglob('events.jsonl')):
        counts = defaultdict(int)
        for line in path.open():
            row = json.loads(line)
            if row.get('venue') != 'par' or row['channel'] not in ('l2Book', 'activeAssetCtx'):
                continue
            if row['channel'] == 'l2Book':
                key = row['data']['coin'], row['data']['time']
                if key in books_seen:
                    continue
                books_seen.add(key)
            events.append(row)
            counts[row['channel']] += 1
        manifest.append({'path': str(path.relative_to(ROOT)), 'counts': dict(counts),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    events.sort(key=lambda r: r['received_ms'])
    return events, manifest


def cached_inputs():
    directory = HERE / 'frozen_inputs'
    directory.mkdir(exist_ok=True)
    if not (directory / 'targets.npz').exists():
        pointer = json.loads((ROOT / 'practice_live/latest.json').read_text())
        run = ROOT / pointer['run']
        for name in ('inputs.npz', 'targets.npz', 'signals.json', 'verification.json'):
            (directory / name).write_bytes((run / name).read_bytes())
        (directory / 'pointer.json').write_text(json.dumps(pointer, indent=2))
    z = load_inputs(directory / 'inputs.npz')
    target = np.load(directory / 'targets.npz')
    metadata = json.loads((HERE / 'raw/meta.json').read_text())['response'][0]
    symbols = tuple(m['name'].split(':')[1] for m in metadata['universe'])
    m = Market(z['market_times'], z['close'], z['volume'], z['trades'], z['funding'], z['observed'], symbols)
    return m, target, {x['name']: x for x in metadata['universe']}


def simulate(events, start, targets, meta, fund, scenario, identifier):
    end = start + (P['development_probe']['holding_seconds'] + P['development_probe']['exit_observation_seconds']) * 1000
    close = start + P['development_probe']['holding_seconds'] * 1000
    arms = [Paper(name, P, scenario, meta) for name in ('baseline', 'candidate')]
    books, contexts = {}, {}
    selected = [e for e in events if start - 60000 <= e['received_ms'] <= end]
    initialized, closing = False, False
    completion, desired = {}, sum(abs(q) for q in targets.values())
    fund_times = funding_times(fund, start, end)
    for event in selected:
        now = event['received_ms']
        if initialized:
            while fund_times and fund_times[0] <= now:
                when = fund_times.pop(0)
                for arm in arms:
                    arm.charge_funding(when, fund.get(when, {}), contexts)
        coin = event['data']['coin']
        if event['channel'] == 'l2Book':
            if coin in books and event['data']['time'] <= books[coin]['data']['time']:
                continue
            books[coin] = event
            for arm in arms:
                arm.depth.update(coin, event['data'])
        else:
            contexts[coin] = {'received_ms': now, 'ctx': event['data']['ctx']}
        if now < start:
            continue
        if not initialized:
            for arm in arms:
                arm.request_target(targets, start, start + 900000, identifier)
            initialized = True
        if not closing and now >= close:
            for arm in arms:
                arm.apply_pending(now, books)
                arm.request_target({c: 0. for c in targets}, close, end, identifier + ':exit', terminal=True)
            closing = True
        for arm in arms:
            arm.tick(now, books, contexts)
            if not arm.equity_series or now - arm.equity_series[-1]['time'] >= 5000:
                arm.snapshot(now, contexts)
            if now >= start + 180000 and arm.name not in completion:
                denominator = sum(abs(q) * float(contexts[c]['ctx']['markPx']) for c, q in targets.items())
                numerator = sum(min(abs(arm.qty.get(c, 0)), abs(q)) * float(contexts[c]['ctx']['markPx']) for c, q in targets.items())
                completion[arm.name] = 100 * numerator / denominator if denominator else 100.
    results, details = [], {}
    for arm in arms:
        if arm.snapshot(end, contexts) is None:
            arm.fatal.append('Stale terminal valuation')
        for coin in targets:
            times = [e['received_ms'] for e in selected if e['channel'] == 'l2Book' and e['data']['coin'] == coin]
            gaps = np.diff([start] + [t for t in times if t >= start] + [end])
            if max(gaps, default=end-start) > 90000:
                arm.fatal.append('Book coverage gap over 90 seconds: ' + coin)
        result = arm.result(contexts)
        result.update(episode=identifier, start_ms=start, end_ms=end,
                      target_completion_at_3m_percent=completion.get(arm.name),
                      evaluation_kind='historical_execution_probe_not_strategy_backtest')
        results.append(result)
        details[arm.name] = {'fills': arm.fills, 'attempts': arm.attempts, 'funding_events': arm.funding_events,
                             'equity': arm.equity_series, 'target_qty': targets}
    return results, details


def funding_times(fund, start, end):
    """Keep actual settlement milliseconds; fundingHistory is not exactly :00."""
    times = sorted(t for t in fund if start < t <= end)
    expected = set(range((start // 3600000 + 1) * 3600000, end, 3600000))
    actual = {t // 3600000 * 3600000 for t in times}
    if not expected <= actual:
        raise ValueError('Missing settled funding hour; cannot assume zero')
    return times


def main():
    events, manifest = events_from_history()
    market, saved, meta = cached_inputs()
    baseline = json.loads((ROOT / 'replay_lab/output/tail_calibration/20261014-v1/baseline.json').read_text())['protocol']
    coins = [f'par:{s}' for s in market.symbols]
    fund = defaultdict(dict)
    for path in (HERE / 'raw').glob('funding_*.json'):
        for row in json.loads(path.read_text())['response']:
            fund[int(row['time'])][row['coin']] = float(row['fundingRate'])
    clock = [e['received_ms'] for e in events]
    btc = [e['received_ms'] for e in events if e['channel'] == 'l2Book' and e['data']['coin'] == 'par:BTC']
    segments = []
    for t in btc:
        if not segments or t - segments[-1][-1] > 90000:
            segments.append([t, t])
        else:
            segments[-1][-1] = t
    rows, detail, skipped = [], {}, []
    for lo, hi in segments:
        start = lo + 60000
        while start + 1500000 <= hi:
            ident = str(start)
            before = events[max(0, bisect.bisect_left(clock, start) - 1000):bisect.bisect_right(clock, start)]
            contexts, books = {}, {}
            for event in before:
                if event['channel'] == 'l2Book':
                    books[event['data']['coin']] = event
                else:
                    contexts[event['data']['coin']] = event
            i = int(np.searchsorted(saved['times'], start // 1000 // 21600 * 21600, side='right') - 1)
            raw = saved['default'][i]
            required = [c for c, w in zip(coins, raw) if abs(w) > 1e-10]
            missing = [c for c in required if c not in contexts or c not in books
                       or max(start - contexts[c]['received_ms'], start - books[c]['received_ms']) > 15000]
            if missing:
                skipped.append({'start_ms': start, 'reason': 'missing_or_stale_nonzero_target_market', 'coins': missing})
                start += 1800000
                continue
            j = int(np.searchsorted(market.times, start // 1000, side='right') - 1)
            prices = market.close[j].copy()
            for k, coin in enumerate(coins):
                if coin in contexts:
                    prices[k] = float(contexts[coin]['data']['ctx']['markPx'])
            live_protocol = json.loads(json.dumps(baseline))
            live_protocol['execution']['decimals'] = [meta[c]['szDecimals'] for c in coins]
            sigma = covariance(market, j, baseline['risk'])[1]
            plan = Controller().propose(int(saved['times'][i]), prices, np.zeros(10), 1e6, raw, sigma,
                                        market.volume[j], market.trades[j], live_protocol, buffer=.2)
            targets = {c: q for c, q in zip(coins, plan['target_qty']) if q}
            if not targets:
                skipped.append({'start_ms': start, 'reason': 'no_nonzero_target'})
            else:
                for name, scenario in P['scenarios'].items():
                    result, data = simulate(events, start, targets, meta, fund, scenario, ident)
                    for row in result:
                        row['scenario'] = name
                    rows.extend(result)
                    detail[ident + ':' + name] = data
            start += 1800000
    summary, matched = [], {}
    for scenario in P['scenarios']:
        paired = {r['episode'] for r in rows if r['scenario'] == scenario}
        paired = {e for e in paired if all(not r['fatal'] and not r['pending_orders']
                  for r in rows if r['scenario'] == scenario and r['episode'] == e)}
        matched[scenario] = sorted(paired)
        for arm in ('baseline', 'candidate'):
            selected = [r for r in rows if r['scenario'] == scenario and r['arm'] == arm]
            valid = [r for r in selected if r['episode'] in paired]
            summary.append({'scenario': scenario, 'arm': arm, 'episodes': len(selected), 'valid': len(valid),
                            'average_probe_pnl': float(np.mean([r['mark_net_pnl'] for r in valid])) if valid else None,
                            'average_target_completion_percent': float(np.mean([r['target_completion_at_3m_percent'] for r in valid])) if valid else None,
                            'average_fees': float(np.mean([r['fees'] for r in valid])) if valid else None,
                            'fully_closed_episodes': sum(r['flat'] for r in valid),
                            'daily_volatility': None})
    output = {'created_at': time.time(), 'kind': 'development_execution_probes', 'protocol_sha256': hashlib.sha256((HERE / 'protocol.json').read_bytes()).hexdigest(),
              'sources': manifest, 'segments_ms': segments, 'skipped': skipped, 'summary': summary, 'episodes': rows,
              'matched_valid_episodes': matched,
              'decision': 'HOLD_EXISTING', 'reasons': ['No fresh held-out full-strategy evaluation yet',
                 'Short disconnected captures cannot establish daily volatility or full holding-cycle profitability'],
              'production_strategy_changed': False, 'orders_sent': 0}
    save('development_results.json', output)
    save('development_details.json', detail)
    with (HERE / 'development_episodes.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'summary': summary, 'skipped': skipped, 'decision': output['decision']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
