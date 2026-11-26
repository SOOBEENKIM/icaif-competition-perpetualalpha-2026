"""Read-only review of the first live practice decisions. No client or keys."""
from collections import defaultdict
from datetime import datetime
from decimal import Decimal as D
import hashlib
import html
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LIVE = ROOT / 'practice_live/live_runtime'


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False))


def kst(ms):
    return datetime.fromtimestamp(ms / 1000, ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')


def main():
    db = sqlite3.connect(f'file:{LIVE}/state.sqlite3?mode=ro', uri=True)
    db.execute('BEGIN')
    rounds = [json.loads(v) for _, v in db.execute("SELECT key,value FROM kv WHERE key LIKE 'round:%' ORDER BY key")]
    records = {kind: [json.loads(r) for r, in db.execute('SELECT raw FROM records WHERE kind=?', (kind,))]
               for kind in ('fills', 'funding', 'transfers')}
    orders = [{'cloid': c, 'status': s, 'plan': json.loads(p), 'result': json.loads(r)}
              for c, s, p, r in db.execute('SELECT cloid,status,plan,result FROM orders ORDER BY updated')]
    db.rollback()
    db.close()
    account = json.loads((LIVE / 'account_latest.json').read_text())
    save('account_snapshot.json', account)
    save('execution_snapshot.json', {'rounds': rounds, 'orders': orders, 'records': records})
    by_oid = defaultdict(list)
    for fill in records['fills']:
        by_oid[fill['oid']].append(fill)
    order_rows = []
    for order in orders:
        oid = order['result']['order']['order']['oid']
        fills = by_oid[oid]
        size = sum(D(f['sz']) for f in fills)
        notional = sum(D(f['sz']) * D(f['px']) for f in fills)
        fees = sum(D(f['fee']) for f in fills)
        assert size == D(order['plan']['size'])
        assert all(D(f['px']) <= D(order['plan']['price']) for f in fills)
        order_rows.append({'oid': oid, 'coin': order['plan']['coin'],
                           'time_kst': kst(min(f['time'] for f in fills)),
                           'size': str(size), 'vwap': str(notional / size),
                           'notional': str(notional), 'fees': str(fees),
                           'fee_bps': str(fees / notional * 10000),
                           'fill_count': len(fills), 'submitted_order_fully_filled': True})
    symbols = [m['name'] for m in account['meta']['universe']]
    marks = {s: D(c['markPx']) for s, c in zip(symbols, account['contexts'])}
    price_pnl = sum((marks[f['coin']] - D(f['px'])) * D(f['sz']) for f in records['fills'])
    fees = sum(D(f['fee']) for f in records['fills'])
    # Snapshot funding may include a newer payment than the read transaction;
    # use the snapshot's own total for this exact valuation reconciliation.
    funding = D(str(account['accounting']['funding']))
    net = price_pnl - fees + funding
    assert abs(net - D(str(account['accounting']['net_pnl']))) < D('.000001')
    notional = sum(D(r['notional']) for r in order_rows)
    first_round = rounds[0]
    first_notional = sum(D(r['notional']) for r in order_rows[:2])
    first_target = sum(D(str(w)) for w in first_round['raw_weights']) * D(str(first_round['equity']))
    initial = []
    for r in order_rows[:2]:
        index = symbols.index(r['coin'])
        target = D(str(first_round['target_qty'][index]))
        initial.append({'coin': r['coin'], 'target_size': str(target), 'sent_and_filled_size': r['size'],
                        'target_quantity_completion_percent': str(D(r['size']) / target * 100)})
    changes = []
    for coin in ('par:BTC', 'par:HYPE'):
        a, b = [r for r in order_rows if r['coin'] == coin]
        changes.append({'coin': coin, 'first_vwap': a['vwap'], 'second_vwap': b['vwap'],
                        'second_price_change_percent': str((D(b['vwap']) / D(a['vwap']) - 1) * 100)})
    findings = [
        {'status': 'observed', 'finding': 'Only three decision rounds, two entry rounds, four submitted orders and eleven matches; no position reductions or exits yet.',
         'implication': 'Eleven matches are not eleven independent strategy trials. Profitability, daily volatility and exit quality are not established.'},
        {'status': 'observed', 'finding': 'All four submitted IOC quantities filled in full. Initial orders were much smaller than the strategy targets.',
         'implication': 'Measure target tracking separately from order fill ratio. Calling this a partially filled submitted order would be incorrect.'},
        {'status': 'code_confirmed', 'finding': 'The runner permits one attempt per coin/phase/normal round, and marks the round complete even with a residual target. Next normal attempt is six hours later.',
         'implication': 'Test bounded child orders with refreshed quotes and unique IDs within a fixed validity window; compare fees, slippage and time to target before adoption.'},
        {'status': 'observed', 'finding': 'First and second round raw weights were identical. Most exposure was bought at the second round at higher prices.',
         'implication': 'The large second buy was target catch-up, not evidence of a stronger new forecast. Its higher price alone does not prove it was a bad decision.'},
        {'status': 'measurement_gap', 'finding': 'The execution journal stores submitted orders/results, but does not persist the order-generation L2 book or activeAssetData limits.',
         'implication': 'The exact limiting input behind the initial small size is not established by this journal. Do not invent executable prices or quote slippage.'},
        {'status': 'observed', 'finding': 'All eleven fills were taker entry fills, approximately 9bp each. Exit fees and exit liquidity remain unobserved.',
         'implication': 'Backtests must include both entry and exit fees, spread, finite depth, unfilled quantities, funding and position reconciliation.'},
        {'status': 'code_confirmed', 'finding': 'Current historical simulations use next-hour reference closes and proxy participation assumptions; live execution uses the par book immediately after a decision.',
         'implication': 'Reference-market backtest gains do not by themselves establish executable par-market gains.'},
        {'status': 'not_established', 'finding': 'No corrected candidate or fresh held-out comparison was run for this review.',
         'implication': 'This review supports targeted experiments, not a claim that another strategy is ready to replace the current one.'},
    ]
    gates = [
        'Freeze the existing baseline and the small candidate set before evaluating them. Change execution first while holding trading signals and risk caps constant.',
        'Use identical starting capital, periods and risk limits; report matched-risk comparisons so exposure increases are not confused with better signals.',
        'Use chronological development and held-out periods with adequate separation for label horizons. Previously inspected periods remain development evidence.',
        'Replay the available real par quotes/trades/funding. Separate exact replay, proxy fills and unavailable counterfactual liquidity; never fill arbitrary size at a historical best price.',
        'A proposed promotion criterion is positive held-out net return above the baseline, daily volatility and drawdown no worse, and no advantage that disappears under predeclared adverse costs/latency.',
        'Evaluate paired performance differences over independent time blocks, across market conditions and completed position lifecycles. Insufficient data means provisional trial, not confirmed superiority.',
        'Confirm bounded exposure, no duplicate orders, actual exits, recovery and accounting in forward practice; record the transition and its costs separately.',
    ]
    review = {'snapshot_kst': datetime.fromisoformat(account['received_at_utc']).astimezone(ZoneInfo('Asia/Seoul')).isoformat(),
              'round_count': len(rounds), 'order_count': len(order_rows), 'fill_count': len(records['fills']),
              'exit_fill_count': sum(f['side'] == 'A' for f in records['fills']),
              'first_target_notional_approx': str(first_target), 'first_fill_notional': str(first_notional),
              'initial_target_notional_completion_approx_percent': str(first_notional / first_target * 100),
              'initial_target_completion': initial, 'orders': order_rows, 'entry_price_changes': changes,
              'total_entry_notional': str(notional), 'entry_fees': str(fees),
              'effective_entry_fee_bps': str(fees / notional * 10000),
              'price_pnl': str(price_pnl), 'net_funding': str(funding), 'net_pnl': str(net),
              'first_two_raw_weight_vectors_identical': rounds[0]['raw_weights'] == rounds[1]['raw_weights'],
              'findings': findings, 'proposed_evaluation_gates_not_official_rules': gates,
              'new_backtest_executed': False, 'strategy_changed': False, 'orders_sent_by_review': 0,
              'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [ROOT / 'practice_live/runner.py', ROOT / 'practice_live/live_logic.py', ROOT / 'replay_lab/ema_rebuild/engine.py']}}
    save('review.json', review)
    print(json.dumps({k: v for k, v in review.items() if k not in ('findings', 'proposed_evaluation_gates_not_official_rules', 'source_hashes')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
