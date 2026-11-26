"""Compare the public snapshot with the independent live-bot fill journal."""
import json
import sqlite3
from decimal import Decimal as D
from pathlib import Path

import build_public_wallet_leaderboard as m


root = Path(__file__).resolve().parents[1]
project = root.parents[1]
wallet = json.loads((root / 'raw/wallets' / f'{m.USER}.json').read_text())
market = json.loads((root / 'raw/market_now.json').read_text())
end = market['queried_at_ms']
assets = list(zip(market['response'][0]['universe'], market['response'][1]))
oracle = {u['name']: D(c['oraclePx']) for u, c in assets}
mark = {u['name']: D(c['markPx']) for u, c in assets}
fills, issues = m.ordered_fills([
    f for f in wallet['fills']['rows']
    if f['coin'].startswith('par:') and f['time'] <= end
])
funding = [
    f for f in wallet['funding']['rows']
    if f['time'] <= end and f['delta']['coin'].startswith('par:')
]
flows, costs, flow_issues = m.cashflows(wallet, end, ['@2900'])
oracle_result, _, accounting_issues = m.reconstruct(fills, funding, costs, oracle)
mark_result, _, _ = m.reconstruct(fills, funding, costs, mark)
database = project / 'practice_live/live_runtime/state.sqlite3'
connection = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
live_fills = [json.loads(row[0]) for row in connection.execute(
    "SELECT raw FROM records WHERE kind='fills'")]
live_fills = [f for f in live_fills if f['time'] <= end]
live_funding = [json.loads(row[0]) for row in connection.execute(
    "SELECT raw FROM records WHERE kind='funding'")]
live_funding = [f for f in live_funding if f['time'] <= end]
connection.close()
assert {(f['coin'], f['tid']) for f in fills} == {
    (f['coin'], f['tid']) for f in live_fills
}
assert sum(D(f['delta']['usdc']) for f in funding) == sum(
    D(f['delta']['usdc']) for f in live_funding
)
independent = sum((
    D(f['sz']) * (1 if f['side'] == 'B' else -1)
    * (oracle[f['coin']] - D(f['px'])) - D(f['fee'])
    for f in live_fills
), D(0)) + sum((D(f['delta']['usdc']) for f in live_funding), D(0))
assert independent == oracle_result['net']
assert not issues + flow_issues + accounting_issues
check = {
    'cutoff_kst': m.date(end),
    'fill_count': len(fills),
    'order_count': len({f['oid'] for f in fills}),
    'capital': str(sum(v for _, v, _ in flows)),
    'oracle_net_pnl': str(oracle_result['net']),
    'oracle_return_percent': str(oracle_result['net'] / 10000),
    'mark_net_pnl': str(mark_result['net']),
    'fees': str(oracle_result['fees']),
    'funding': str(oracle_result['funding']),
    'independent_live_fill_ids_match': True,
    'independent_funding_match': True,
    'independent_executed_cashflows_match': True,
}
(root / 'user_reconciliation.json').write_text(
    json.dumps(check, ensure_ascii=False, indent=2)
)
print(json.dumps(check, ensure_ascii=False))
