"""Cash-conserving episode attribution and descriptive paired sensitivity."""
import bisect
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.par_direct_forecast_20261017.data import ROOT, save, sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.hourly_refresh_pairs_20261017.research import control_directory

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'hourly_refresh_pairs_20261017'


def load(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def summary(returns):
    x = np.asarray(returns)
    r = np.prod(1 + x, axis=-1) - 1
    sigma = np.std(x, axis=-1, ddof=1) * np.sqrt(x.shape[-1])
    q = (np.maximum(r, 0) + .005) / (np.abs(r) + 1.5 * sigma + .05)
    return dict(return_fraction=r, volatility_horizon=sigma, financial_score=q)


def circular_indices(n, block, count, seed):
    if n < 2 or block < 1:
        raise ValueError('Need at least two observations and a positive block size')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(count, (n + block - 1)//block))
    return ((starts[..., None] + np.arange(block)) % n).reshape(count, -1)[:, :n]


def paired_resampling(candidate, baselines, settings):
    x = np.asarray(candidate)
    assert all(len(v) == len(x) for v in baselines.values())
    rows = []
    for block in settings['circular_block_days']:
        idx = circular_indices(len(x), block, settings['samples'], settings['seed'] + block)
        chosen = summary(x[idx])
        for name, values in baselines.items():
            base = summary(np.asarray(values)[idx])
            row = dict(baseline=name, block_days=block, samples=len(idx))
            for k in ['return_fraction', 'financial_score']:
                delta = chosen[k] - base[k]
                row[k + '_difference_quantiles'] = np.quantile(delta, settings['quantiles']).tolist()
            row['fraction_resamples_pnl_and_q_above'] = float(np.mean((chosen['return_fraction'] > base['return_fraction']) & (chosen['financial_score'] > base['financial_score'])))
            rows.append(row)
    return rows


def episode_paths(run, d, scenario):
    entries = [e for e in run['events'] if e['event'] == 'pair_intent']
    starts = [e['time'] for e in entries]
    times = np.asarray(d['times'])
    fills = defaultdict(list)
    for f in run['owner_fills']['pair']:
        fills[f['time']].append(f)
    changes = np.zeros((len(entries), len(run['snapshots'])))
    components = np.zeros((len(entries), 4))
    for column, (previous, current) in enumerate(zip(run['snapshots'], run['snapshots'][1:]), 1):
        now = current['time']
        i = int(np.searchsorted(times, now)); j = int(np.searchsorted(times, previous['time']))
        assert times[i] == now and times[j] == previous['time']
        q = np.asarray(previous['pair_qty']); physical = np.asarray(previous['qty'])
        prices = d['par_close'][i]
        move = float(q @ (prices - d['par_close'][j]))
        funding = 0.
        if now % 3600 == 0:
            rates = d['funding'][i]
            rates = np.where(-physical*d['reference_close'][i]*rates > 0,
                             rates*scenario['funding_receipt_multiplier'], rates)
            funding = -float(q @ (d['reference_close'][i]*rates))
        cost = sum(f['signed_qty']*(f['price'] - prices[f['asset']]) for f in fills[now])
        fees = sum(f['fee'] for f in fills[now])
        k = bisect.bisect_left(starts, now) - 1
        pnl = move - cost - fees + funding
        if k < 0:
            assert abs(pnl) < 1e-10 and not fills[now]
        else:
            changes[k, column] = pnl
            components[k] += [move, -cost, -fees, funding]
    cumulative = np.cumsum(changes, axis=1)
    if len(entries):
        assert abs(cumulative[:, -1].sum() - run['metrics']['pair_owned_pnl']) < 1e-6
        np.testing.assert_allclose(cumulative[:, -1], components.sum(axis=1), atol=1e-8, rtol=0)
    else:
        assert abs(run['metrics']['pair_owned_pnl']) < 1e-10
    return entries, cumulative, components


def main():
    out = HERE / 'diagnostic_artifacts'; out.mkdir(exist_ok=False)
    spec = json.loads((HERE/'protocol.json').read_text())
    cfg = json.loads((PRIOR/'reference_protocol.json').read_text())
    for rel, digest in json.loads((PRIOR/'artifacts/frozen.json').read_text())['sources'].items():
        assert sha(ROOT/rel) == digest, rel
    paths = [HERE/'analysis.py', HERE/'protocol.json', HERE/'test_results.txt', PRIOR/'artifacts/frozen.json', PRIOR/'artifacts/selection.json', PRIOR/'artifacts/attribution.json']
    runpaths = {}
    for phase in spec['phases']:
        for sc in spec['scenarios']:
            for arm in [spec['candidate'], *spec['baselines']]:
                parent = PRIOR/'artifacts' if arm == spec['candidate'] else control_directory(arm)
                path = parent/'runs'/f'reference-{phase}-{arm}-{sc}.json.gz'
                paths.append(path); runpaths[phase, sc, arm] = path
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    save(out/'frozen.json', dict(at_unix=time.time(), sources=hashes, protected=protected(), post_evaluation_only=True))
    d, _, _ = inputs(cfg)
    attribution = json.loads((PRIOR/'artifacts/attribution.json').read_text())
    rows = []
    for phase in spec['phases']:
        for sc in spec['scenarios']:
            run = load(runpaths[phase, sc, spec['candidate']])
            bases = {name:load(runpaths[phase, sc, name]) for name in spec['baselines']}
            days = np.array([s['time'] for s in run['daily']])
            returns = np.array([s['return_'] for s in run['daily']])
            br = {name:np.array([s['return_'] for s in b['daily']]) for name, b in bases.items()}
            for b in bases.values():
                assert [s['time'] for s in b['daily']] == days.tolist()
            for source in [run, *bases.values()]:
                calc = summary([s['return_'] for s in source['daily']])
                for k, value in calc.items():
                    assert abs(value - source['metrics'][k]) < 1e-10, k
            entries, cumulative, components = episode_paths(run, d, cfg['scenarios'][sc])
            old = next(x['episodes'] for x in attribution['rows'] if (x['venue'], x['phase'], x['arm'], x['scenario']) == ('reference', phase, spec['candidate'], sc))
            np.testing.assert_allclose(cumulative[:, -1], [e['net_pnl'] for e in old], atol=1e-7, rtol=0)
            episodes = [dict(**e, components=dict(zip(['price_move', 'execution_cost', 'fees', 'funding'], components[i]))) for i, e in enumerate(old)]
            best = int(np.argmax(cumulative[:, -1]))
            equity = np.array([s['equity'] for s in run['snapshots']])
            stripped = equity - cumulative[best]
            daily_mask = np.array([s['time'] % 86400 == 43200 for s in run['snapshots']])
            stripped_daily = stripped[daily_mask]
            stripped_stats = summary(stripped_daily[1:]/stripped_daily[:-1] - 1)
            stripped_stats['net_pnl'] = float(stripped[-1] - stripped[0])
            normalization = []
            for name, base_returns in br.items():
                scale = float(np.std(base_returns, ddof=1)/np.std(returns, ddof=1))
                scaled = summary(returns*scale)
                baseline = summary(base_returns)
                normalization.append(dict(baseline=name, hindsight_scale=scale, **scaled,
                    net_pnl=float(scaled['return_fraction']*1e6),
                    pnl_advantage=float((scaled['return_fraction']-baseline['return_fraction'])*1e6),
                    q_advantage=float(scaled['financial_score']-baseline['financial_score'])))
            gains = sum(max(e['net_pnl'], 0) for e in episodes)
            rows.append(dict(phase=phase, scenario=sc, candidate=spec['candidate'], metrics=run['metrics'],
                episodes=episodes, best_episode=best, best_episode_share_of_positive_episode_pnl=float(max(episodes[best]['net_pnl'],0)/gains) if gains else None,
                without_best_episode=stripped_stats, ex_post_risk_normalized=normalization,
                paired_resampling=paired_resampling(returns, br, spec['daily_resampling'])))
            print(phase, sc, 'pair PnL', run['metrics']['pair_owned_pnl'], 'best', episodes[best]['net_pnl'],
                  'account without best', stripped_stats['net_pnl'], 'risk normalized', [(n['baseline'],n['pnl_advantage']) for n in normalization], flush=True)
    for rel, digest in hashes.items():
        assert sha(ROOT/rel) == digest, rel
    save(out/'results.json',dict(rows=rows, protected=protected(), orders_sent=0, historical_independence=False,
                                actual_account_replays=0, frozen_strategy_selection_unchanged=True,
                                descriptive_sensitivity_not_tradable_counterfactual=True))


if __name__ == '__main__':
    main()
