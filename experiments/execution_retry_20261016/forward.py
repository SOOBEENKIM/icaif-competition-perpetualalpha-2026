"""Bounded read-only capture, then ONE registry-controlled paper evaluation.

This file never instantiates an exchange client and never reads wallet settings.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time

import numpy as np
import websocket

from .engine import Paper
from .development import funding_times
from practice_live.runner import signal_inputs
from replay_lab.common_comparison.engine import risk_breaches
from replay_lab.ema_rebuild.engine import Controller
from replay_lab.evaluation_control.registry import Registry
from scripts.refresh_public_wallets import api

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAPTURE = HERE / 'forward_capture'
USER = '0x125778065a578d598d88bddaa3fdd635280d0e47'
P = json.loads((HERE / 'protocol.json').read_text())
META = {m['name']: m for m in json.loads((HERE / 'raw/meta.json').read_text())['response'][0]['universe']}
COINS = tuple(META)


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def capture(plan):
    """Raw data only. Health file deliberately contains no comparison PnL."""
    CAPTURE.mkdir(exist_ok=True)
    lock, stop = threading.Lock(), threading.Event()
    counts, last, writers, errors = Counter(), {}, {}, []
    end = plan['test_end']
    signal_hour = -1

    def emit(channel, data):
        now = time.time()
        if now >= end:
            return
        row = {'received_ms': int(now * 1000), 'venue': 'par', 'channel': channel, 'data': data}
        bucket = datetime.fromtimestamp(now, timezone.utc).strftime('%Y%m%d_%H')
        encoded = json.dumps(row, default=convert, separators=(',', ':'), allow_nan=False) + '\n'
        with lock:
            for key in list(writers):
                if key != bucket:
                    writers.pop(key).close()
            if bucket not in writers:
                writers[bucket] = gzip.open(CAPTURE / (bucket + '.jsonl.gz'), 'at', encoding='utf-8')
            writers[bucket].write(encoded)
            counts[channel] += 1
            if isinstance(data, dict) and 'coin' in data:
                last[channel + ':' + data['coin']] = now

    def signals():
        nonlocal signal_hour
        while not stop.is_set() and time.time() < end:
            now = time.time()
            if int(now // 3600) * 3600 != signal_hour:
                try:
                    value = signal_inputs(now)
                    emit('strategySignal', value)
                    signal_hour = value['hour']
                except Exception as exc:
                    with lock:
                        errors.append({'time': time.time(), 'source': 'signal', 'error': str(exc)})
                        del errors[:-20]
            stop.wait(10)

    signal_worker = threading.Thread(target=signals, daemon=True)
    signal_worker.start()
    socket, reconnects, last_ping, last_health = None, 0, 0., 0.
    try:
        while time.time() < end:
            if shutil.disk_usage(HERE).free < 5 * 1024 ** 3:
                raise RuntimeError('Less than 5GB free; capture stopped without changing live bot')
            if socket is None:
                try:
                    socket = websocket.create_connection('wss://api.hyperliquid-testnet.xyz/ws', timeout=5)
                    for coin in COINS:
                        for kind in ('l2Book', 'activeAssetCtx', 'activeAssetData', 'trades'):
                            subscription = {'type': kind, 'coin': coin}
                            if kind == 'activeAssetData':
                                subscription['user'] = USER
                            socket.send(json.dumps({'method': 'subscribe', 'subscription': subscription}))
                    reconnects += 1
                except Exception as exc:
                    if socket:
                        socket.close()
                    socket = None
                    errors.append({'time': time.time(), 'source': 'connect', 'error': str(exc)})
                    time.sleep(2)
            if socket is not None:
                try:
                    if time.time() - last_ping >= 20:
                        socket.send(json.dumps({'method': 'ping'}));last_ping = time.time()
                    message = json.loads(socket.recv())
                    channel = message.get('channel')
                    if channel in ('l2Book', 'activeAssetCtx', 'activeAssetData', 'trades'):
                        emit(channel, message['data'])
                    elif channel == 'error':
                        raise RuntimeError(str(message.get('data')))
                except websocket.WebSocketTimeoutException:
                    pass
                except Exception as exc:
                    errors.append({'time': time.time(), 'source': 'websocket', 'error': str(exc)})
                    socket.close();socket = None
            if time.time() - last_health >= 15:
                with lock:
                    for writer in writers.values():
                        writer.flush()
                    save(CAPTURE / 'health.json', {'state': 'capturing_raw_only', 'at': time.time(),
                         'test_start': plan['test_start'], 'test_end': end, 'counts': dict(counts),
                         'last_received': last, 'signal_hour': signal_hour, 'reconnects': reconnects,
                         'recent_errors': errors[-20:], 'orders_sent': 0, 'partial_results_inspected': False})
                last_health = time.time()
    finally:
        stop.set();signal_worker.join(timeout=20)
        if socket:
            socket.close()
        with lock:
            for writer in writers.values():
                writer.close()
    save(CAPTURE / 'health.json', {'state': 'capture_finished', 'at': time.time(), 'test_end': end,
                                 'counts': dict(counts), 'orders_sent': 0})


def recorded_events():
    """Merge hour files, receipt-ordering only; no exchange-time backdating."""
    for path in sorted(CAPTURE.glob('*.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            # Two producer threads can write in a different order in the same ms.
            rows = [json.loads(line) for line in stream]
        yield from sorted(rows, key=lambda x: x['received_ms'])


def boundaries(series, start_ms, end_ms, step_ms):
    result, j = {}, 0
    for boundary in range((start_ms // step_ms + 1) * step_ms, end_ms + 1, step_ms):
        while j + 1 < len(series) and series[j + 1]['time'] <= boundary:
            j += 1
        if series and 0 <= boundary - series[j]['time'] <= 15000:
            result[boundary] = series[j]['oracle_equity']
    return result


def daily_metrics(series, start_ms, end_ms):
    days = boundaries(series, start_ms, end_ms, 86400000)
    returns = [days[t] / days[t - 86400000] - 1 for t in sorted(days) if t - 86400000 in days]
    return {'daily_returns': returns, 'complete_days': len(returns),
            'daily_volatility_percent': float(np.std(returns, ddof=0) * 100) if len(returns) >= 2 else None,
            'daily_sample_std_percent': float(np.std(returns, ddof=1) * 100) if len(returns) >= 2 else None}


def completed_cycles(fills):
    qty, count = defaultdict(float), 0
    for fill in fills:
        coin = fill['coin'];old = qty[coin];qty[coin] += fill['qty']
        if abs(old) > 1e-10 and abs(qty[coin]) <= 1e-10:
            count += 1
    return count


def paired_interval(base, candidate, start, end):
    b, c = [boundaries(a.equity_series, start, end, 21600000) for a in (base, candidate)]
    times = sorted(set(b) & set(c))
    differences = [(c[t] - c[t - 21600000] - b[t] + b[t - 21600000]) / 1e6
                   for t in times if t - 21600000 in b and t - 21600000 in c]
    n = len(differences)
    if n < 8:
        return {'blocks': n, 'interval95': None}
    rng = np.random.default_rng(20261016)
    d = np.array(differences)
    # Circular moving blocks preserve dependence within each adjacent pair.
    starts = rng.integers(0, n, size=(10000, (n + 1) // 2))
    indexes = np.stack((starts, (starts + 1) % n), axis=-1).reshape(10000, -1)[:, :n]
    means = np.mean(d[indexes], axis=1)
    return {'blocks': n, 'mean_difference': float(d.mean()),
            'interval95': np.quantile(means, [.025, .975]).tolist()}


def evaluate_data(plan):
    start, end = plan['test_start'] * 1000, plan['test_end'] * 1000
    fund = defaultdict(dict)
    for coin in COINS:
        response = api({'type': 'fundingHistory', 'coin': coin, 'startTime': start, 'endTime': end})
        save(CAPTURE / ('funding_' + coin.replace(':', '_') + '.json'), response)
        if not isinstance(response.get('response'), list):
            raise ValueError('Settled funding download failed: ' + coin)
        for row in response['response']:
            fund[int(row['time'])][coin] = float(row['fundingRate'])
    payment_times = funding_times(fund, start, end)
    arms = {(scenario, name): Paper(name, P, config, META)
            for scenario, config in P['scenarios'].items() for name in ('baseline', 'candidate')}
    books, contexts, active, signal_value = {}, {}, {}, None
    decisions, peaks, halted = {k: set() for k in arms}, {k: 1e6 for k in arms}, {k: False for k in arms}
    last_seen, max_gaps, errors = {}, defaultdict(int), set()
    decision_log = []
    last_snapshot, last_risk = -1, -1
    for event in recorded_events():
        now, channel, data = event['received_ms'], event['channel'], event['data']
        if now > end:
            break
        if now >= start:
            while payment_times and payment_times[0] <= now:
                when = payment_times.pop(0)
                for arm in arms.values():
                    arm.charge_funding(when, fund[when], contexts)
        if channel == 'strategySignal':
            signal_value = data
        elif channel in ('l2Book', 'activeAssetCtx', 'activeAssetData'):
            coin = data['coin'];key = channel + ':' + coin
            if key in last_seen and now >= start:
                max_gaps[key] = max(max_gaps[key], now - max(start, last_seen[key]))
            last_seen[key] = now
            if channel == 'l2Book':
                if coin in books and data['time'] <= books[coin]['data']['time']:
                    continue
                books[coin] = event
                for arm in arms.values():
                    arm.depth.update(coin, data)
            elif channel == 'activeAssetCtx':
                contexts[coin] = {'received_ms': now, 'ctx': data['ctx']}
            else:
                active[coin] = event
        if now < start:
            continue
        fresh = all(c in contexts and now - contexts[c]['received_ms'] <= 15000 for c in COINS)
        minute = now // 60000
        same_hour = signal_value is not None and signal_value['hour'] == now // 3600000 * 3600
        if fresh and same_hour and minute != last_risk:
            inputs = signal_value
            prices = np.array([float(contexts[c]['ctx']['markPx']) for c in COINS])
            protocol = json.loads(json.dumps(inputs['protocol']))
            protocol['execution']['decimals'] = [META[c]['szDecimals'] for c in COINS]
            sigma = np.array(inputs['sigma'])
            for key, arm in arms.items():
                eq = arm.equity(contexts);peaks[key] = max(peaks[key], eq)
                halted[key] |= eq / peaks[key] - 1 <= -protocol['risk']['drawdown_halt']
                qty = np.array([arm.qty.get(c, 0.) for c in COINS])
                forced = halted[key] or bool(risk_breaches(qty, eq, prices, sigma, protocol['risk']))
                if forced:
                    errors.add('Observed strategy risk breach or drawdown halt')
                normal = inputs['hour'] % 21600 == 0 and now / 1000 - inputs['hour'] <= 900
                decision = int(minute * 60) if forced else inputs['hour']
                identity = ('risk' if forced else 'normal', decision)
                if (not forced and not normal) or identity in decisions[key] or arm.pending:
                    continue
                proposal = Controller().propose(decision, prices, qty, eq, np.array(inputs['weights']), sigma,
                           np.array(inputs['volume']), np.array(inputs['trades']), protocol, buffer=.2, halted=halted[key])
                if proposal is None:
                    continue
                deadline = min(inputs['hour'] * 1000 + 3600000, now + 60000) if forced else (inputs['hour'] + 900) * 1000
                target = dict(zip(COINS, proposal['target_qty']))
                arm.request_target(target, now, deadline, f'{identity[0]}:{decision}')
                decisions[key].add(identity)
                decision_log.append({'scenario': key[0], 'arm': key[1], 'time': now, 'identity': identity,
                                     'target': target, 'before': qty.tolist(), 'raw_weights': inputs['weights'],
                                     'input_hash': inputs['input_hash']})
            last_risk = minute
        for arm in arms.values():
            arm.tick(now, books, contexts, active)
        # Dense enough to use the last causal observation within 15s of UTC boundaries.
        if now - last_snapshot >= 5000:
            for arm in arms.values():
                arm.snapshot(now, contexts)
            last_snapshot = now
    while payment_times:
        when = payment_times.pop(0)
        for arm in arms.values():
            arm.charge_funding(when, fund[when], contexts)
    for coin in COINS:
        for channel in ('l2Book', 'activeAssetCtx'):
            key = channel + ':' + coin
            max_gaps[key] = max(max_gaps[key], end - last_seen.get(key, start))
            if max_gaps[key] > 90000:
                errors.add('Data gap longer than 90 seconds: ' + key)
    expected_rounds = set(range((plan['test_start'] // 21600 + 1) * 21600, plan['test_end'], 21600))
    results, details = {}, {}
    for key, arm in arms.items():
        if not arm.equity_series or end - arm.equity_series[-1]['time'] > 15000:
            errors.add('Stale terminal valuation: ' + str(key))
        if {t for kind, t in decisions[key] if kind == 'normal'} != expected_rounds:
            errors.add('Missing normal decision: ' + str(key))
        result = arm.result(contexts)
        values = np.array([1e6] + [x['oracle_equity'] for x in arm.equity_series])
        result.update(daily_metrics(arm.equity_series, start, end))
        result.update(oracle_return_percent=result['oracle_net_pnl'] / 1e6 * 100,
                      oracle_drawdown_percent=float(np.min(values / np.maximum.accumulate(values) - 1) * 100),
                      completed_cycles=completed_cycles(arm.fills),
                      average_gross_percent=float(np.mean([r['gross'] for r in arm.equity_series]) * 100),
                      turnover=sum(abs(f['qty']) * f['px'] for f in arm.fills) / 1e6)
        label = ':'.join(key);results[label] = result
        details[label] = {'fills': arm.fills, 'attempts': arm.attempts, 'funding': arm.funding_events,
                          'equity': arm.equity_series}
    comparisons = {}
    for scenario in P['scenarios']:
        b, c = results[scenario + ':baseline'], results[scenario + ':candidate']
        interval = paired_interval(arms[scenario, 'baseline'], arms[scenario, 'candidate'], start, end)
        gates = {
            'complete_data_and_reconciliation': not errors and not b['fatal'] and not c['fatal'] and not b['pending_orders'] and not c['pending_orders'],
            'positive_net_return_and_above_baseline': c['oracle_net_pnl'] > max(0., b['oracle_net_pnl']),
            'minimum_three_complete_days': min(b['complete_days'], c['complete_days']) >= 3,
            'volatility_no_higher': c['daily_volatility_percent'] is not None and b['daily_volatility_percent'] is not None and c['daily_volatility_percent'] <= b['daily_volatility_percent'],
            'drawdown_no_worse': c['oracle_drawdown_percent'] >= b['oracle_drawdown_percent'],
            'minimum_four_completed_cycles': c['completed_cycles'] >= 4,
            'block_interval_above_zero': interval['interval95'] is not None and interval['interval95'][0] > 0,
        }
        comparisons[scenario] = {'gates': gates, 'paired_blocks': interval,
                                 'exposure_increase_percent_points': c['average_gross_percent'] - b['average_gross_percent']}
    passed = all(all(c['gates'].values()) for c in comparisons.values())
    save(HERE / 'forward_details.json', {'arms': details, 'decisions': decision_log})
    return {'decision': 'ELIGIBLE_FOR_REVIEW_ONLY' if passed else 'HOLD_EXISTING',
            'orders_sent': 0, 'production_strategy_changed': False, 'test_start': plan['test_start'], 'test_end': plan['test_end'],
            'results': results, 'comparisons': comparisons, 'errors': sorted(errors), 'maximum_data_gaps_ms': dict(max_gaps),
            'limitations': ['Counterfactual depth and actual-wallet capacity proxy do not guarantee venue execution.',
                            'Ending open positions are marked; no liquidation price is invented.',
                            'This window cannot supply the minimum three complete UTC daily returns.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['collect-and-evaluate', 'evaluate'])
    args = parser.parse_args()
    reservation = json.loads((HERE / 'forward_reservation.json').read_text())
    plan = reservation['plan']
    try:
        Registry._verify_artifacts(plan)
        if args.action == 'collect-and-evaluate' and time.time() < plan['test_end']:
            capture(plan)
        result = Registry().evaluate(reservation['trial_id'], evaluate_data)
        save(HERE / 'forward_result.json', result)
        print(json.dumps({'decision': result['decision'], 'orders_sent': 0}), flush=True)
    except Exception as exc:
        save(HERE / 'forward_failure.json', {'decision': 'HOLD_EXISTING', 'at': time.time(),
             'reason': str(exc), 'orders_sent': 0, 'production_strategy_changed': False})
        raise


if __name__ == '__main__':
    main()
