"""Read-only incumbent review: actual fills, current account and code evidence.

Does not open the reserved execution experiment's forward data or results.
"""
from collections import defaultdict
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import numpy as np

from practice_live.runner import signal_inputs

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LIVE = ROOT / 'practice_live/live_runtime'


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))


def main():
    account = json.loads((LIVE / 'account_latest.json').read_text())
    snapshot = datetime.fromisoformat(account['received_at_utc'])
    db = sqlite3.connect('file:' + str(LIVE / 'state.sqlite3') + '?mode=ro', uri=True)
    db.execute('BEGIN')
    rounds = [json.loads(v) for v, in db.execute("SELECT value FROM kv WHERE key LIKE 'round:%' ORDER BY key")]
    fills = [json.loads(v) for v, in db.execute("SELECT raw FROM records WHERE kind='fills'")]
    orders = [{'status': s, 'plan': json.loads(p), 'result': json.loads(r) if r else None}
              for s, p, r in db.execute('SELECT status,plan,result FROM orders ORDER BY updated')]
    db.rollback();db.close()
    inputs = signal_inputs(snapshot.timestamp())
    coins = [m['name'] for m in account['meta']['universe']]
    qty, prices, oracle, components = [], [], [], []
    for coin, ctx in zip(coins, account['contexts']):
        relevant = [f for f in fills if f['coin'] == coin]
        q = sum(float(f['sz']) * (1 if f['side'] == 'B' else -1) for f in relevant)
        assert abs(q - account['accounting']['qty'][coin]) < 1e-9
        mark, op = float(ctx['markPx']), float(ctx['oraclePx'])
        pnl = sum((mark - float(f['px'])) * float(f['sz']) * (1 if f['side'] == 'B' else -1) for f in relevant)
        fee = sum(float(f['fee']) for f in relevant)
        if relevant or q:
            components.append({'coin': coin, 'qty': q, 'mark': mark, 'oracle': op,
                               'notional': q * mark, 'price_pnl': pnl, 'fees': fee,
                               'oracle_minus_mark_pnl': q * (op-mark)})
        qty.append(q);prices.append(mark);oracle.append(op)
    ledger = account['accounting']
    price_pnl = sum(c['price_pnl'] for c in components)
    fees = sum(c['fees'] for c in components)
    assert abs(fees-ledger['fees']) < 1e-6
    assert abs(price_pnl-fees+ledger['funding']-ledger['net_pnl']) < 1e-6
    w = np.array(qty)*np.array(prices)/ledger['equity']
    variance = float(w @ inputs['sigma'] @ w)
    contributions = w*(inputs['sigma'] @ w)/variance if variance else w*0
    for c in components:
        c['share_of_gross_notional_percent'] = abs(c['notional']) / sum(abs(x['notional']) for x in components)*100
        c['model_variance_contribution_percent'] = float(contributions[coins.index(c['coin'])]*100)
    timings = []
    for path in (ROOT / 'practice_live/updates').glob('*/signals.json'):
        s = json.loads(path.read_text())
        if datetime.fromisoformat(s['created_at_utc']) >= datetime.fromisoformat(rounds[0]['created_at_utc']):
            timings.append({'at': s['created_at_utc'], 'seconds': s['elapsed_inference_seconds']})
    latest = json.loads((ROOT / 'practice_live/latest.json').read_text())
    latest_signals = json.loads((ROOT / latest['run'] / 'signals.json').read_text())
    source_files = ['practice_live/runner.py','practice_live/live_logic.py','practice_live/live_store.py',
                    'practice_live/infer.py','replay_lab/tail_calibration/agent.py',
                    'replay_lab/competition_rebuild/causal_risk.py','replay_lab/competition_rebuild/candidate_agent.py',
                    'replay_lab/v10_revision/core.py','replay_lab/reversion_blend/model.py','replay_lab/ema_rebuild/engine.py']
    evidence = {
        'snapshot_kst': snapshot.astimezone(ZoneInfo('Asia/Seoul')).isoformat(),
        'actual_account': ledger, 'price_pnl': price_pnl, 'components': components,
        'oracle_net_pnl': ledger['net_pnl']+sum(c['oracle_minus_mark_pnl'] for c in components),
        'gross_exposure_percent': float(abs(w).sum()*100),
        'model_daily_volatility_percent_not_realized_std': float(np.sqrt(24*variance)*100),
        'round_count': len(rounds), 'orders': len(orders), 'matches': len(fills),
        'close_matches': sum('Close' in f['dir'] for f in fills),
        'normal_rounds': rounds, 'all_order_statuses': [o['status'] for o in orders],
        'inference_runs_after_first_entry': len(timings),
        'inference_seconds_median': float(np.median([x['seconds'] for x in timings])),
        'inference_seconds_max': max(x['seconds'] for x in timings),
        'latest_signal_assets': latest_signals['assets'],
        'mechanism_check': {'kind': 'synthetic calculation, not live attribution', 'base_weight': .04,
                            'risk_filter_half_weight': .02, 'renormalized_weight': .038},
        'limitations': ['One live holding episode per asset; no live exit observed.',
                        'Cannot identify initial order-size binding constraint without contemporaneous L2/capacity records.',
                        'Price movement after a fill is not causal evidence of a strategy defect.',
                        'Risk-normalization cancellation is a verified mechanism, not proven attribution of current loss.'],
        'production_changed': False, 'orders_sent_by_review': 0, 'reserved_forward_results_read': False,
        'source_hashes': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files},
    }
    save('evidence.json', evidence)
    save('actual_orders_and_fills.json', {'orders': orders, 'fills': fills})
    save('inference_timings.json', timings)
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('normal_rounds','latest_signal_assets','source_hashes')},indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
