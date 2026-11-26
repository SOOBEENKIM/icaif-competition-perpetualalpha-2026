"""Fixed omitted-pair investigation, then a fresh executable-cost snapshot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from experiments.performance_attribution_20261016.live_signal import read_publication
from experiments.cost_horizon_20261016.research import save, sha
from experiments.paired_funding_20261017.quote import request
from experiments.paired_venue_20261017.forward import current_snapshots
from experiments.ensemble_carry_20261017.engine import interval_risk
from replay_lab.common_comparison.engine import risk_breaches
from strategy.agent import TAIL
from .quote import signed_beta, hedge_diagnostic, signed_quote

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; OUT = HERE / 'artifacts'
COINS = ['par:USDEUR', 'par:USDJPY']


def prepare():
    if OUT.exists(): raise FileExistsError('Already prepared: do not reselect on outcomes')
    frame, publication = read_publication()
    source = ROOT / publication['source_run'] / 'inputs.npz'
    if sha(source) != frame.source_hash: raise ValueError('Published reference inputs changed')
    trading = json.loads((TAIL / 'baseline.json').read_text())['protocol']
    ids = [trading['symbols'].index(c.split(':')[1]) for c in COINS]
    with np.load(source) as z:
        if int(z['market_times'][-1]) != frame.source_time: raise ValueError('Input time mismatch')
        prices = z['close'][-169:, ids].copy(); observed = z['observed'][-169:, ids].copy()
        times = z['market_times'][-169:].copy()
    if not np.all(np.diff(times) == 3600) or not observed.all(): raise ValueError('Incomplete hourly reference window')
    returns = np.diff(np.log(prices), axis=0); diagnostic = hedge_diagnostic(returns)
    rng = np.random.default_rng(20261017); samples = []
    # Seven circular 24h blocks. Descriptive stability only, not future coverage.
    for _ in range(500):
        indexes = np.concatenate([(start + np.arange(24)) % len(returns) for start in rng.integers(0, len(returns), 7)])
        try: samples.append(signed_beta(returns[indexes])['beta'])
        except ValueError: pass
    diagnostic.update(bootstrap_replicates=len(samples), circular_block_hours=24,
        descriptive_beta_quantiles=np.quantile(samples, [.025, .5, .975]).tolist(),
        bootstrap_negative_share=float(np.mean(np.array(samples) < 0)),
        bootstrap_is_not_a_forecast_guarantee=True, past_nonzero_moves=np.count_nonzero(returns, axis=0).tolist())
    cfg = json.loads((ROOT / 'experiments/paired_venue_20261017/protocol.json').read_text())
    old = json.loads((ROOT / 'experiments/ensemble_carry_20261017/artifacts/plan.json').read_text())
    sources = dict(old['sources'])
    for f in HERE.glob('*.py'): sources[str(f.relative_to(ROOT))] = sha(f)
    p = dict(created_at=time.time(), coins=COINS, signed_beta=diagnostic['beta'], direction_candidates=[-1, 1],
        horizons_hours=[24, 72], lookback_hours=168, absolute_correlation_minimum=.3,
        absolute_beta_bounds=[.25, 4], quote_protocol=cfg, trading_protocol=trading,
        source_run=publication['source_run'], source_time=frame.source_time,
        source_file=str(source.relative_to(ROOT)), source_sha256=sha(source),
        paired_covariance=diagnostic['covariance'], sources=sources,
        hypothesis='Previously excluded negative-correlation FX pair can hedge common return risk with same-sign positions. Test executable basis/funding economics, not generic mean reversion.',
        limitations='All legs settle linearly in USDPR; USDJPY contract quote units are not an unleveraged foreign-currency cash balance. Negative beta implies same-side positions and nonzero dollar net exposure. Covariance does not prove cointegration or convergence.',
        success='Only a prospective study candidate if both costs show >5bp expected net edge and current plus all partial-fill states satisfy the existing risk caps. A favorable quote cannot prove realized PnL/Q improvement.',
        metadata_sources=['https://perpetual-alpha26.github.io/docs/', 'https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding'],
        raw_reserved_outcomes_read=False, actual_pnl=None, actual_Q=None, live_orders_sent=0)
    OUT.mkdir(); save(OUT / 'plan.json', p); save(OUT / 'hedge_diagnostic.json', diagnostic)
    np.savez_compressed(OUT / 'reference_window.npz', times=times, close=prices, observed=observed, returns=returns)
    print(json.dumps(dict(prepared=True, beta=diagnostic['beta'], correlation=diagnostic['correlation'],
        bootstrap_quantiles=diagnostic['descriptive_beta_quantiles'], sources=len(sources))))


def evaluate_quote():
    if (OUT / 'results.json').exists(): raise FileExistsError('Single snapshot already evaluated')
    p = json.loads((OUT / 'plan.json').read_text()); cfg = p['quote_protocol']
    for path, digest in p['sources'].items():
        if sha(ROOT / path) != digest: raise ValueError('Frozen source changed: ' + path)
    if sha(ROOT / p['source_file']) != p['source_sha256']: raise ValueError('Reference history changed')
    payloads = [dict(type='metaAndAssetCtxs', dex='par')] + [dict(type='l2Book', coin=c) for c in COINS]
    with ThreadPoolExecutor(max_workers=3) as pool: receipts = list(pool.map(request, payloads))
    now = int(time.time() * 1000); meta, ctx = receipts[0]['response']
    metadata = {r['name']: r for r in meta['universe']}
    contexts = {m['name']: dict(received_ms=int(receipts[0]['received_at'] * 1000), ctx=c) for m, c in zip(meta['universe'], ctx)}
    books = {r['response']['coin']: dict(received_ms=int(r['received_at'] * 1000), data=r['response']) for r in receipts[1:]}
    snapshots = current_snapshots(now, books, contexts, cfg)
    diagnostic = json.loads((OUT / 'hedge_diagnostic.json').read_text())
    input_gates = dict(fresh_quotes=all(c in snapshots for c in COINS),
        fresh_hedge_input=0 <= now / 1000 - p['source_time'] <= 21600,
        negative_beta=p['signed_beta'] < 0, admissible_beta=.25 <= abs(p['signed_beta']) <= 4,
        sufficient_correlation=abs(diagnostic['correlation']) >= p['absolute_correlation_minimum'])
    rows = []
    if all(input_gates.values()):
        prices = np.array([float(snapshots[c]['ctx']['markPx']) for c in COINS])
        for direction in p['direction_candidates']:
            for horizon in p['horizons_hours']:
                for scenario, cost in cfg['scenarios'].items():
                    q = signed_quote(COINS, snapshots, metadata, p['signed_beta'], direction, horizon, cfg, cost)
                    if q is None:
                        rows.append(dict(direction=direction, horizon_hours=horizon, scenario=scenario, proposal=None, eligible=False))
                        continue
                    target = np.array(q['target_qty']); covariance = np.array(p['paired_covariance'])
                    risk = p['trading_protocol']['risk']
                    breach = risk_breaches(target, 1e6, prices, covariance, risk)
                    partial = interval_risk({}, dict(zip(COINS, target)), [], COINS, prices, 1e6, covariance, risk)
                    rows.append(dict(direction=direction, horizon_hours=horizon, scenario=scenario,
                        proposal=q, risk_breaches=breach, partial_fill_risk_breaches=partial,
                        eligible=q['predicted_net_bps'] > cfg['entry_buffer_bps'] and not breach and not partial))
    groups = [(d, h) for d in p['direction_candidates'] for h in p['horizons_hours']
              if all(any(r['direction'] == d and r['horizon_hours'] == h and r['scenario'] == s and r['eligible'] for r in rows) for s in cfg['scenarios'])]
    result = dict(at_ms=now, receipts=receipts, input_gates=input_gates, rows=rows,
        eligible_in_both_costs=groups, actual_pnl=None, actual_Q=None, realized_improvement_proven=False,
        independent_future_performance_evaluated=False, live_orders_sent=0, no_automatic_live_replacement=True)
    save(OUT / 'results.json', result)
    print(json.dumps(dict(file=str(OUT / 'results.json'), input_gates=input_gates,
        rows=[{k: r[k] for k in ('direction','horizon_hours','scenario','eligible')} | dict(
            predicted_net_bps=r['proposal']['predicted_net_bps'] if r['proposal'] else None,
            predicted_net_pnl=r['proposal']['predicted_net_pnl'] if r['proposal'] else None) for r in rows],
        eligible_in_both_costs=groups), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'quote']); args = parser.parse_args()
    if args.action == 'prepare': prepare()
    else: evaluate_quote()
