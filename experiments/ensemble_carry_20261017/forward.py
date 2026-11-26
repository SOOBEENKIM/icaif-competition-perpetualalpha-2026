"""Registered additive comparison using an existing sealed public capture.

Read the capture only after its original release. No early outcomes, duplicate
capture or live strategy changes. Evaluation never has a signing client.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import json
from pathlib import Path
import time
import numpy as np
from experiments.cost_horizon_20261016 import forward as capture
from experiments.cost_horizon_20261016.research import save, sha
from experiments.paired_venue_20261017.forward import paper_metrics
from experiments.paired_funding_20261017.probe import settlement_clock
from experiments.paired_funding_20261017.quote import request
from experiments.complementary_alpha_20261017.research import protected
from .engine import Account

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; DEST = HERE / 'artifacts'
DATA = ROOT / 'experiments/paired_venue_20261017/artifacts/forward'
ARMS = ('first_strategy', 'second_strategy', 'second_scaled_control', 'second_plus_carry')


def completed_pairs(events):
    count = 0; opened = False
    for event in events:
        if event['event'] == 'entry_intent': opened = False
        elif event['event'] == 'balanced_open': opened = True
        elif event['event'] == 'flat_confirmed':
            count += int(opened); opened = False
    return count


def assert_ready(plan, now=None):
    now = time.time() if now is None else now
    if now < plan['outcome_release_unix'] + 60:
        raise ValueError('Original outcomes are sealed or the capture is still flushing')
    if sha(DATA / 'plan.json') != plan['capture_plan_sha256']:
        raise ValueError('The registered source capture plan changed')
    for path, digest in plan['sources'].items():
        if sha(ROOT / path) != digest: raise ValueError('Frozen dependency changed: ' + path)


def prepare():
    if DEST.exists(): raise FileExistsError('Comparison already registered')
    old = json.loads((DATA / 'plan.json').read_text())
    if time.time() >= old['paper_start_unix']: raise ValueError('Register before the common start')
    diagnostic_path = ROOT / 'experiments/paired_funding_20261017/artifacts/additive_risk_diagnostic.json'
    diagnostic = json.loads(diagnostic_path.read_text())
    row = next(r for r in diagnostic['rows'] if r['baseline'] == 'candidate')
    scale = row['combined_target_daily_sigma'] / row['baseline_target_daily_sigma']
    p = {k: deepcopy(old[k]) for k in ('paper_start_unix', 'capture_end_unix', 'outcome_release_unix',
        'metadata', 'coins', 'trading_protocol', 'paired_protocol', 'end_liquidation_seconds')}
    sources = dict(old['sources'])
    sources.update(json.loads((ROOT / 'experiments/paired_funding_20261017/forward_probe/plan.json').read_text())['sources'])
    sources[str(diagnostic_path.relative_to(ROOT))] = sha(diagnostic_path)
    for f in HERE.glob('*.py'): sources[str(f.relative_to(ROOT))] = sha(f)
    p.update(created_at=time.time(), arms=ARMS, overlay_coins=['par:EWY', 'par:SP500'],
        direction=-1, horizon_hours=72, initial_equity_each=1e6, sources=sources,
        scale_control=scale, scale_control_source=str(diagnostic_path.relative_to(ROOT)),
        scale_control_meaning='Second strategy multiplied by the pre-registration current-snapshot predicted risk ratio. This is a scale-only control, not exact realized-risk matching over the future path.',
        capture_directory=str(DATA.relative_to(ROOT)), capture_plan_sha256=sha(DATA / 'plan.json'),
        current_source_checks=protected(),
        allocation='Retain second-strategy controller. Add only EWY-short/SP500-long carry with then-published 168h hedge ratio when both assets are unused. Baseline conflict or budget demand closes the complete pair before baseline reclaims its assets.',
        entry='Only this fixed pair/direction, one pair at a time. Every five minutes: 72h forecast, positive funding haircut 50%, basis convergence 50%, paid entry/exit spreads and fees, >5bp net expected edge, correlation >=0.3, visible two-way depth, global partial-fill risk corners. No entry after expiry would exceed common liquidation boundary.',
        execution='One cash/inventory ledger and executor per account. Scoped pair cancellations. New received arrival books, 1/3 second latency, 5 second grace, independent partial legs, paid repair/closure. Shared one-second submission cadence and global reservations for all four accounts.',
        risk='All candidate and baseline target/pending-fill combinations share the original global caps; reserve 50bp potential friction on remaining turnover. Freeze new increases if reservation becomes invalid. Stale risk data or breached actual limits flatten the account; in-flight orders are resolved, never erased.',
        financing='Funding uses actual settlement history and pre-settlement observed oracle. Positive funding forecasts are uncertain. Orders are paper only.',
        comparison='Same seven days, initial 1M equity, fees, latency, books, funding and risk caps. Compare PnL and Q against first, second and predeclared scale-only control. Report realized volatility, MDD, turnover, exposure and funding attribution.',
        success_rule='Positive fully closed PnL, higher PnL and Q than all three controls under both costs, >=2 completed balanced pairs, >=5 completed daily returns, no fatal errors/pending/residual holdings. Finite-period improvement is not guaranteed future profit and never causes automatic deployment.',
        independence='Registered before first observation. Shares the existing seven-day capture, so it is one comparison on that period, not an independent second replication. Earlier short funding probe results cannot change this candidate.',
        source_outcomes_inspected=False, live_orders_sent=0, automatic_live_replacement=False)
    DEST.mkdir(); save(DEST / 'plan.json', p)
    print(json.dumps(dict(registered=True, scale_control=scale, start=p['paper_start_unix'],
                         release=p['outcome_release_unix'], sources=len(sources), orders_sent=0)))


def comparison_checks(rows):
    index = {(r['arm'], r['scenario']): r for r in rows}; checks = []
    for scenario in ('base', 'stress'):
        c = index['second_plus_carry', scenario]
        cm = c['metrics']['mark_equity']
        for name in ARMS[:-1]:
            b = index[name, scenario]; bm = b['metrics']['mark_equity']
            gates = dict(positive_pnl=c['mark_net_pnl'] > 0, higher_pnl=c['mark_net_pnl'] > b['mark_net_pnl'],
                higher_Q=cm['financial_score'] is not None and bm['financial_score'] is not None and cm['financial_score'] > bm['financial_score'],
                enough_days=min(cm['completed_days'], bm['completed_days']) >= 5,
                completed_pairs=c['completed_pairs'] >= 2, no_fatal_errors=not c['fatal'] and not b['fatal'],
                flat=c['flat'] and b['flat'], no_pending=c['pending_orders'] == b['pending_orders'] == 0)
            checks.append(dict(scenario=scenario, baseline=name, gates=gates, passed=all(gates.values())))
    return checks


def evaluate(plan):
    assert_ready(plan)
    if (DEST / 'results.json').exists(): return
    start, end = [int(plan[k] * 1000) for k in ('paper_start_unix', 'capture_end_unix')]
    rates = defaultdict(dict)
    def get_rates(c): return request(dict(type='fundingHistory', coin=c, startTime=start, endTime=end))
    with ThreadPoolExecutor(max_workers=3) as pool: receipts = list(pool.map(get_rates, plan['coins']))
    save(DEST / 'funding_receipts.json', receipts)
    for coin, receipt in zip(plan['coins'], receipts):
        rows = receipt['response']
        if not isinstance(rows, list) or len(rows) >= 500: raise ValueError('Incomplete funding history')
        for r in rows:
            if int(r['time']) % 3600000 >= 1000: raise ValueError('Unexpected settlement time')
            rates[int(r['time']) // 3600000 * 3600000][coin] = float(r['fundingRate'])
    accounts = {(n, s): Account(n, s, plan) for n in ARMS for s in plan['paired_protocol']['scenarios']}
    books, contexts, signal = {}, {}, None
    settlements = settlement_clock(start, end); cursor = 0; last_snapshot = -1
    capture.DEST = DATA
    for event in capture.events():
        now = event['received_ms']
        if now > end: break
        while cursor < len(settlements) and settlements[cursor] <= now:
            at = settlements[cursor]
            for a in accounts.values(): a.paper.charge_funding(at, rates[at], contexts)
            cursor += 1
        channel, d = event['channel'], event['data']
        if channel == 'l2Book':
            books[d['coin']] = dict(received_ms=now, data=d)
            for a in accounts.values(): a.paper.depth.update(d['coin'], d)
        elif channel == 'activeAssetCtx': contexts[d['coin']] = dict(received_ms=now, ctx=d['ctx'])
        elif channel == 'strategySignal': signal = d
        for a in accounts.values(): a.step(now, books, contexts, signal)
        if now >= start and now - last_snapshot >= 10000:
            for a in accounts.values(): a.paper.snapshot(now, contexts)
            last_snapshot = now
    rows = []
    for (name, scenario), a in accounts.items():
        metrics = paper_metrics(a.paper, start, end); result = a.paper.result(contexts)
        result.update(arm=name, scenario=scenario, metrics=metrics,
            balanced_opens=sum(e['event'] == 'balanced_open' for e in a.pair.events) if a.pair else 0,
            completed_pairs=completed_pairs(a.pair.events) if a.pair else 0,
            mean_gross=float(np.mean([r['gross'] for r in a.paper.equity_series])),
            risk_rejections=len(a.hub.events))
        with gzip.open(DEST / f'result_{name}_{scenario}.json.gz', 'wt') as f:
            json.dump(dict(summary=result, fills=a.paper.fills, equity=a.paper.equity_series,
                funding=a.paper.funding_events, attempts=a.paper.attempts, baseline_decisions=a.events,
                risk_decisions=a.hub.events, pair_events=a.pair.events if a.pair else []), f, allow_nan=False,
                default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist())
        rows.append(result)
    checks = comparison_checks(rows)
    save(DEST / 'results.json', dict(rows=rows, checks=checks, paper_improvement=all(c['passed'] for c in checks),
        same_future_period_as_other_registered_trial=True, independent_second_replication=False,
        exact_realized_risk_matching=False, live_orders_sent=0, automatic_live_replacement=False))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'evaluate']); args = parser.parse_args()
    if args.action == 'prepare': prepare()
    else: evaluate(json.loads((DEST / 'plan.json').read_text()))


if __name__ == '__main__': main()
