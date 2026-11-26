"""Sealed, public-data, seven-day comparison. Never signs or places orders.

The new trial has its own capture directory. Previously reserved experiment
outcomes are not opened. Raw-only capture finishes before evaluation starts.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import time
from pathlib import Path
import numpy as np
from experiments.cost_horizon_20261016 import forward as capture_module
from experiments.cost_horizon_20261016.research import SecondController
from experiments.performance_attribution_20261016.live_signal import read_publication
from experiments.basis_carry_20261017.score import score
from experiments.execution_retry_20261016.engine import Paper
from experiments.pairs_research_20261017.research import save, sha
from experiments.complementary_alpha_20261017.research import protected
from replay_lab.ema_rebuild.engine import Controller
from replay_lab.common_comparison.engine import risk_breaches
from .engine import quote_pair, PairExecution
from .arrival import broker

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = HERE / 'artifacts/forward'
NAMES = ('first_strategy', 'second_strategy', 'basis_carry_pair_72h')
_CACHE = {}


def published_pairs():
    frame, obj = read_publication()
    if frame.source_time not in _CACHE:
        cfg = json.loads((HERE / 'protocol.json').read_text())
        path = ROOT / obj['source_run'] / 'inputs.npz'
        if sha(path) != frame.source_hash:
            raise ValueError('Reference input hash differs from publication')
        with np.load(path, allow_pickle=False) as z:
            if int(z['market_times'][-1]) != frame.source_time:
                raise ValueError('Reference data and signal clock mismatch')
            returns = np.diff(np.log(z['close'][-169:]), axis=0)
        from strategy.agent import TAIL
        symbols = json.loads((TAIL / 'baseline.json').read_text())['protocol']['symbols']
        covariance = np.cov(returns.T, ddof=0)
        pairs = []
        for a, b in cfg['pairs']:
            i, j = symbols.index(a), symbols.index(b)
            beta = float(np.clip(covariance[i, j] / max(covariance[j, j], 1e-10), .25, 4.))
            correlation = float(covariance[i, j] / max(np.sqrt(covariance[i, i] * covariance[j, j]), 1e-10))
            vol = float(np.sqrt(max(covariance[i, i] + beta**2 * covariance[j, j] - 2*beta*covariance[i, j], 1e-8)) / (1 + beta))
            pairs.append(dict(coins=['par:' + a, 'par:' + b], beta=beta, correlation=correlation, hourly_vol=vol))
        _CACHE.clear(); _CACHE[frame.source_time] = pairs
    obj = dict(obj); obj['frame'] = dict(obj['frame'], paired_metrics=_CACHE[frame.source_time])
    return frame, obj


def prepare():
    DEST.mkdir(exist_ok=False)
    cfg = json.loads((HERE / 'protocol.json').read_text())
    metadata = capture_module.info({'type': 'metaAndAssetCtxs', 'dex': 'par'})[0]
    meta = {r['name']: r for r in metadata['universe']}
    from strategy.agent import TAIL
    trading = json.loads((TAIL / 'baseline.json').read_text())['protocol']
    start = (int(time.time()) - 43200) // 86400 * 86400 + 43200 + 86400
    sources = {str(p.relative_to(ROOT)): sha(p) for p in [*HERE.glob('*.py'), HERE / 'protocol.json']}
    # These dependency manifests contain source digests only, not trial outcomes.
    baseline = json.loads((ROOT / 'experiments/cost_horizon_20261016/artifacts/forward/plan.json').read_text())
    sources.update(baseline['sources'])
    for p in [ROOT / 'experiments/cost_horizon_20261016/forward.py', ROOT / 'experiments/pairs_forecast_20261017/policy.py',
              ROOT / 'experiments/basis_carry_20261017/score.py']:
        sources[str(p.relative_to(ROOT))] = sha(p)
    _, obj = published_pairs()
    p = dict(created_at=time.time(), capture_start_unix=time.time(), paper_start_unix=start,
        capture_end_unix=start + 7 * 86400, arms=NAMES, metadata=meta, coins=['par:' + s for s in trading['symbols']],
        trading_protocol=trading, paired_protocol=cfg, sources=sources,
        execution_protocol=broker(meta, cfg['scenarios']['base']).protocol,
        initial_equity_each=1e6, selection='72h is the only tested horizon with any positive historical expected net edge; this is not realized-profit validation.',
        historical_positive_edge_dependent_on_funding_persistence=True,
        first_publication=obj['frame']['source_time'], protected=protected(),
        no_automatic_promotion=True, orders_sent=0,
        arrival_rule='Fill only after a book received at/after simulated arrival, with server time at/after submission; absence within five seconds is an unfilled order. Same rule for every arm.',
        end_liquidation_seconds=600, new_entry_cutoff='No entry whose 72h horizon extends past the common liquidation boundary.',
        outcome_release_unix=start + 7 * 86400,
        success_rule=dict(positive_net_pnl=True, higher_net_pnl_and_Q_than_both_baselines_in_both_scenarios=True,
            minimum_completed_pairs=2, minimum_daily_returns=5, no_fatal_errors=True, no_residual_positions=True),
        metric='Mark-equity net return and 12:00 UTC sample daily returns; Q is a paper estimate, not the official leaderboard. Oracle-equity sensitivity also retained.',
        limitations='Displayed L2 is counterfactual executable liquidity. No actual maker queue or own impact is assumed. Funding forecasts may fail. A seven-day study cannot establish long-term profitability. Practice DEX inactivity/gaps invalidate outcomes.')
    save(DEST / 'plan.json', p)
    print(json.dumps(dict(prepared=True, paper_start_unix=start, end_unix=p['capture_end_unix'], sources=len(sources), orders_sent=0)))


def current_snapshots(now, books, contexts, cfg):
    result = {}
    for c, b in books.items():
        ctx = contexts.get(c)
        if ctx is None or now - b['received_ms'] > cfg['maximum_book_age_ms'] or now - ctx['received_ms'] > cfg['maximum_context_age_ms']:
            continue
        if not -1000 <= now - b['data']['time'] <= cfg['maximum_book_age_ms']:
            continue
        levels = b['data']['levels']
        if not all(levels): continue
        bid, ask = [float(x[0]['px']) for x in levels]
        oracle = float(ctx['ctx']['oraclePx'])
        if min(bid, ask, oracle) <= 0 or bid >= ask: continue
        result[c] = dict(book=b['data'], ctx=ctx['ctx'], oracle=oracle, mid=(bid + ask) / 2)
    return result


def paper_metrics(paper, start, end):
    series = paper.equity_series
    if not series: raise ValueError('No equity observations')
    if series[0]['time'] - start > 15000: paper.fatal.append('Missing initial equity observation')
    if end - series[-1]['time'] > 15000: paper.fatal.append('Missing terminal equity observation')
    gaps = [b['time'] - a['time'] for a, b in zip(series, series[1:])]
    if gaps and max(gaps) > 60000: paper.fatal.append('Equity coverage gap above 60 seconds')
    samples = [dict(time=start, mark_equity=1e6, oracle_equity=1e6)]
    times = np.array([r['time'] for r in series])
    for at in range(start + 86400000, end + 1, 86400000):
        i = int(np.searchsorted(times, at, side='right') - 1)
        if i >= 0 and 0 <= at - times[i] <= 15000:
            samples.append(dict(time=at, mark_equity=series[i]['mark_equity'], oracle_equity=series[i]['oracle_equity']))
        else: paper.fatal.append('Missing 12:00 UTC daily equity sample')
    spans = np.diff([s['time'] for s in samples]) / 86400000
    output = {}
    for field in ('mark_equity', 'oracle_equity'):
        values = np.array([s[field] for s in samples])
        dr = values[1:] / values[:-1] - 1
        R = series[-1][field] / 1e6 - 1
        output[field] = score(float(R), dr, spans, trading_days=len({(f['time'] - 43200000)//86400000 for f in paper.fills}), start_sample=True)
    output['daily'] = samples
    return output


def evaluate(plan):
    if time.time() < plan['outcome_release_unix']:
        raise ValueError('New forward outcomes remain sealed until the registered end')
    for path, digest in plan['sources'].items():
        if sha(ROOT / path) != digest: raise ValueError('Frozen dependency changed: ' + path)
    if (DEST / 'results.json').exists(): return
    start, end = int(plan['paper_start_unix'] * 1000), int(plan['capture_end_unix'] * 1000)
    cfg, p, coins = plan['paired_protocol'], plan['trading_protocol'], plan['coins']
    def get_rates(c):
        r = capture_module.info({'type': 'fundingHistory', 'coin': c, 'startTime': start, 'endTime': end})
        if not isinstance(r, list) or len(r) >= 500: raise ValueError('Incomplete funding history')
        return c, r
    rates = defaultdict(dict)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for c, rows in pool.map(get_rates, coins):
            save(DEST / ('funding_' + c.replace(':', '_') + '.json'), rows)
            for row in rows:
                at = int(row['time']); hour = at // 3600000 * 3600000
                if at - hour >= 1000: raise ValueError('Unexpected funding settlement timestamp')
                rates[hour][c] = float(row['fundingRate'])
    arms = {(name, scenario): broker(plan['metadata'], sc) for name in NAMES for scenario, sc in cfg['scenarios'].items()}
    paired = {key: PairExecution(arm, cfg) for key, arm in arms.items() if key[0] == NAMES[-1]}
    books, contexts = {}, {}; signal = None
    last_risk, last_snapshot = -1, -1
    decisions, last_pair_bar, control_state, exit_attempt = set(), {}, {}, {}
    peaks = {k: 1e6 for k in arms}; halted = {k: False for k in arms}
    funding_times = list(range(start + 3600000, end + 1, 3600000)); cursor = 0
    capture_module.DEST = DEST
    for event in capture_module.events():
        now = event['received_ms']
        if now > end: break
        while cursor < len(funding_times) and funding_times[cursor] <= now:
            at = funding_times[cursor]
            for arm in arms.values(): arm.charge_funding(at, rates[at], contexts)
            cursor += 1
        channel, d = event['channel'], event['data']
        if channel == 'l2Book':
            books[d['coin']] = dict(received_ms=now, data=d)
            for arm in arms.values(): arm.depth.update(d['coin'], d)
        elif channel == 'activeAssetCtx': contexts[d['coin']] = dict(received_ms=now, ctx=d['ctx'])
        elif channel == 'strategySignal': signal = d
        if now < start: continue
        terminal = now >= end - plan['end_liquidation_seconds'] * 1000
        if now - last_risk >= 30000 and signal and all(c in contexts and now - contexts[c]['received_ms'] <= 15000 for c in coins):
            frame = signal['frame']; covariance = np.array(frame['sigma'])
            prices = np.array([float(contexts[c]['ctx']['markPx']) for c in coins]); hour = now // 3600000 * 3600
            fresh_signal = 0 <= hour - frame['source_time'] <= cfg['maximum_signal_age_seconds']
            for key, arm in arms.items():
                name, scenario = key; equity = arm.equity(contexts)
                qty = np.array([arm.qty.get(c, 0.) for c in coins])
                peaks[key] = max(peaks[key], equity); halted[key] |= equity / peaks[key] - 1 <= -p['risk']['drawdown_halt']
                forced = halted[key] or bool(risk_breaches(qty, equity, prices, covariance, p['risk'])) or not fresh_signal
                if name == NAMES[-1]:
                    manager = paired[key]
                    if terminal or forced: manager.close(now, 'evaluation_end' if terminal else 'account_or_stale_signal_risk')
                    bar = now // 300000
                    snapshots = current_snapshots(now, books, contexts, cfg)
                    if manager.state is None and not arm.pending and not forced and not terminal and last_pair_bar.get(key) != bar:
                        # Do not open a trade whose planned expiry falls outside the observation window.
                        if now + 72 * 3600000 > end - plan['end_liquidation_seconds'] * 1000: continue
                        choices = []
                        for pair in frame['paired_metrics']:
                            if pair['correlation'] < cfg['minimum_correlation'] or any(c not in snapshots for c in pair['coins']): continue
                            for direction in (-1, 1):
                                proposal = quote_pair(pair['coins'], snapshots, plan['metadata'], pair['beta'], direction, 72, cfg, cfg['scenarios'][scenario])
                                if proposal and proposal['predicted_net_bps'] > cfg['entry_buffer_bps']:
                                    trial = np.zeros(len(coins))
                                    for c, q in zip(proposal['coins'], proposal['target_qty']): trial[coins.index(c)] = q
                                    if not risk_breaches(trial, equity, prices, covariance, p['risk']): choices.append(proposal)
                        last_pair_bar[key] = bar
                        if choices: manager.enter(max(choices, key=lambda r: r['predicted_net_pnl']), now)
                elif not arm.pending:
                    boundary = now // 21600000 * 21600000
                    normal = fresh_signal and now - boundary <= 900000 and frame['source_time'] == hour and (key, boundary) not in decisions
                    if terminal:
                        if now >= exit_attempt.get(key, 0):
                            arm.request_target({c: 0. for c in coins}, now, end - 1000, f'terminal:{now}')
                            exit_attempt[key] = now + 30000
                    elif normal or forced:
                        when = now // 1000 if forced else boundary // 1000
                        if name == 'first_strategy': control = Controller(**control_state.get(key, {})); raw = signal['incumbent']
                        else:
                            control = SecondController([when], [frame['retention']]); raw = signal['candidate']
                            if key in control_state: control.controller.last_decision = control_state[key]['last_decision']
                        action = control.propose(when, prices, qty, equity, np.array(raw), covariance,
                            np.array(frame['volume']), np.array(frame['trades']), p, buffer=.2, halted=bool(halted[key] or not fresh_signal))
                        if action is not None:
                            control_state[key] = dict(last_decision=control.last_decision)
                            arm.request_target(dict(zip(coins, action['target_qty'])), now, now + 900000, f'{name}:{when}')
                            if normal: decisions.add((key, boundary))
            last_risk = now
        for key, arm in arms.items():
            if key in paired: paired[key].tick(now, books, contexts)
            else: arm.tick(now, books, contexts)
        if now - last_snapshot >= 10000:
            for arm in arms.values(): arm.snapshot(now, contexts)
            last_snapshot = now
    rows = []
    for key, arm in arms.items():
        name, scenario = key
        metrics = paper_metrics(arm, start, end)
        result = arm.result(contexts)
        result.update(arm=name, scenario=scenario, metrics=metrics, completed_pairs=sum(e['event'] == 'flat_confirmed' for e in paired[key].events) if key in paired else None,
            mean_gross=float(np.mean([r['gross'] for r in arm.equity_series])))
        with gzip.open(DEST / f'result_{name}_{scenario}.json.gz', 'wt') as f:
            json.dump(dict(summary=result, equity=arm.equity_series, fills=arm.fills, attempts=arm.attempts,
                funding=arm.funding_events, events=paired[key].events if key in paired else []), f, allow_nan=False)
        rows.append(result)
    index = {(r['arm'], r['scenario']): r for r in rows}; checks = []
    for scenario in cfg['scenarios']:
        c = index[NAMES[-1], scenario]
        for baseline in NAMES[:2]:
            b = index[baseline, scenario]; cm, bm = c['metrics']['mark_equity'], b['metrics']['mark_equity']
            gates = dict(positive_net_pnl=c['mark_net_pnl'] > 0,
                higher_pnl=c['mark_net_pnl'] > b['mark_net_pnl'], higher_Q=cm['financial_score'] is not None and bm['financial_score'] is not None and cm['financial_score'] > bm['financial_score'],
                enough_days=min(cm['completed_days'], bm['completed_days']) >= 5, completed_pairs=c['completed_pairs'] >= 2,
                no_fatal_errors=not c['fatal'] and not b['fatal'], flat=not c['residual_positions'] and not b['residual_positions'],
                no_pending=c['pending_orders'] == b['pending_orders'] == 0)
            checks.append(dict(scenario=scenario, baseline=baseline, gates=gates, passed=all(gates.values())))
    save(DEST / 'results.json', dict(rows=rows, checks=checks, paper_improvement=all(c['passed'] for c in checks),
        independent_after_registration=True, live_orders_sent=0, automatic_promotion=False,
        required_next_step='Review complete raw fills, exposure and uncertainty; this experiment never changes the live strategy.'))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'capture', 'evaluate']); args = parser.parse_args()
    if args.action == 'prepare': prepare(); return
    plan = json.loads((DEST / 'plan.json').read_text())
    if args.action == 'capture':
        capture_module.DEST = DEST
        capture_module.read_publication = published_pairs
        capture_module.capture(plan)
    evaluate(plan)


if __name__ == '__main__': main()
