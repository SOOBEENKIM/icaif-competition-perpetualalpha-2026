"""Validate and regenerate this frozen public-data snapshot without network access."""
import collections
import contextlib
import csv
import io
import json
import math
import pathlib
import runpy
from decimal import Decimal as D
from html.parser import HTMLParser

ROOT = pathlib.Path(__file__).resolve().parent
before = json.loads((ROOT / 'analysis.json').read_text())['leaderboard']
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path(str(ROOT / 'build_leaderboard.py'))
analysis = json.loads((ROOT / 'analysis.json').read_text())
assert analysis['leaderboard'] == before, 'Regeneration must preserve IDs and results'

# Partial close and reversal use exchange settlement and retain the remaining basis.
def fill(t, side, quantity, price, start, closed='0'):
    return dict(time=t, coin='par:X', side=side, sz=quantity, px=price,
                startPosition=start, closedPnl=closed, fee='0', feeToken='USDPR')

fs = [fill(1, 'B', '3', '10', '0'), fill(2, 'A', '1', '12', '3', '2'),
      fill(3, 'A', '3', '9', '2', '-2'), fill(4, 'B', '1', '8', '-1', '1')]
equity = ns['equity']
assert equity(2, fs, [], [(0, D(100), 'deposit')], [], {'par:X': D(11)})[0] == 104
assert equity(3, fs, [], [(0, D(100), 'deposit')], [], {'par:X': D(8)})[0] == 101
assert equity(4, fs, [], [(0, D(100), 'deposit')], [], {})[0] == 101
# Same-millisecond fills may arrive reversed: follow position transitions.
same = [fill(1, 'A', '1', '12', '2', '2'), fill(1, 'B', '2', '10', '0')]
ordered, errors = ns['ordered_fills'](same)
assert not errors and [f['startPosition'] for f in ordered] == ['0', '2']

ranked = [r for r in analysis['leaderboard'] if r['rank70'] is not None]
assert len(analysis['leaderboard']) == 47 and len(ranked) == 42
assert len(analysis['unfilled_active']) == 1
assert [r['rank70'] for r in ranked] == list(range(1, 43))
assert all(r['jury_score30'] is None and r['official_total_score100'] is None
           for r in analysis['leaderboard'])
low, high = min(r['period_return_percent'] for r in ranked), max(r['period_return_percent'] for r in ranked)
vlow, vhigh = min(r['daily_volatility_percent_proxy'] for r in ranked), max(r['daily_volatility_percent_proxy'] for r in ranked)
for r in ranked:
    assert not r['history_issues']
    expected = 40 * (r['period_return_percent'] - low) / (high - low)
    expected += 30 * (vhigh - r['daily_volatility_percent_proxy']) / (vhigh - vlow)
    assert math.isclose(r['score70_proxy'], expected, abs_tol=1e-10)
    assert 0 <= r['return_score40'] <= 40 and 0 <= r['risk_score30_proxy'] <= 30
assert all(a['score70_proxy'] >= b['score70_proxy'] for a, b in zip(ranked, ranked[1:]))

csv_counts = {}
for path in ROOT.glob('*.csv'):
    with path.open(encoding='utf-8-sig', newline='') as f:
        csv_counts[path.name] = sum(1 for _ in csv.DictReader(f))
assert csv_counts['wallet_inventory.csv'] == 88
assert csv_counts['leaderboard.csv'] == 47
with (ROOT / 'fills.csv').open(encoding='utf-8-sig', newline='') as f:
    fills = list(csv.DictReader(f))
assert len(fills) == 48326
groups = collections.defaultdict(list)
seen = set()
for f in fills:
    assert int(f['time_ms']) <= ns['END']
    key = (f['coin'], f['time_ms'], f['tid'])
    assert (f['address'], key) not in seen
    seen.add((f['address'], key))
    groups[key].append(f)
assert len(groups) == 24163
for group in groups.values():
    assert len(group) == 2 and {r['side'] for r in group} == {'A', 'B'}
    a, b = group
    assert a['counterparty_address'] == b['address']
    assert b['counterparty_address'] == a['address']
    assert a['px'] == b['px'] and a['sz'] == b['sz']

coverage = json.loads((ROOT / 'raw' / 'candle_coverage_validation.json').read_text())
hour_counts = collections.Counter((coin, int(tm) // 3600000 * 3600000)
                                  for coin, tm, tid in groups)
for check in coverage['checks']:
    assert check['expected'] == check['observed']
    assert hour_counts[(check['coin'], check['hour_start_ms'])] == check['expected']
assert len(coverage['checks']) == coverage['all_completed_coin_hours'] == 3281
assert coverage['all_observed_trades'] == 24115 and coverage['mismatches'] == 0

for path in (ROOT / 'raw' / 'wallets').glob('*.json'):
    wallet = json.loads(path.read_text())
    for name in ['fills', 'ledger']:
        assert wallet[name]['pagination_complete'], (path.name, name)
    assert isinstance(wallet['positions']['response'], dict)
    assert isinstance(wallet['orders']['response'], list)

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'href' in attrs:
            self.links.append(attrs['href'])
        if 'id' in attrs:
            self.ids.add(attrs['id'])

document = Links()
document.feed((ROOT / 'report.html').read_text())
for link in document.links:
    if '://' not in link and not link.startswith('#'):
        assert (ROOT / link).is_file(), link
assert all(r['wallet_id'] in document.ids for r in analysis['leaderboard'])

result = dict(status='passed', reproducible_without_network=True,
              ranked_wallets=len(ranked), all_trading_wallets=47,
              unique_executions=len(groups), both_sides_matched=True,
              completed_coin_hours_validated=3281,
              completed_hour_executions=24115, csv_row_counts=csv_counts,
              caveat='Validation checks internal consistency and observed candle counts; it does not establish official ranking or unbounded historical completeness.')
(ROOT / 'validation.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
