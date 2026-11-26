"""Frozen reference signals with actual par candles/volume/funding rates.

This is a transfer diagnostic. Hourly trade closes are NOT executable quotes,
funding notionals use close-price proxies, and figures are not wallet returns.
No parameters are fitted to the par outcomes or promoted to the live bot.
"""
import copy
import gzip
import json
import time
from pathlib import Path

import numpy as np

from .research import HERE, ROOT, save, sha
from .policy import PairsController
from .partial import PartialController
from experiments.cost_horizon_20261016.research import data, SecondController
from experiments.performance_attribution_20261016.engine import simulate
from experiments.basis_carry_20261017.score import score
from experiments.complementary_alpha_20261017.research import protected
from replay_lab.common_comparison.models import Market, funding_bucket
from replay_lab.ema_rebuild.engine import Controller


def score_proxy(run):
    """Include the partial day in R, never in completed-day sample volatility."""
    snapshots = run['snapshots']
    assert snapshots[0]['time'] % 86400 == 43200
    daily = [s for s in snapshots if s['time'] % 86400 == 43200]
    values = np.array([s['equity'] for s in daily])
    times = np.array([s['time'] for s in daily])
    returns = values[1:] / values[:-1] - 1
    spans = np.diff(times) / 86400
    partial_return = snapshots[-1]['equity'] / values[-1] - 1
    R = snapshots[-1]['equity'] / values[0] - 1
    assert abs(np.prod(1 + returns) * (1 + partial_return) - 1 - R) < 1e-10
    td = len({(f['time'] - 43200) // 86400 for f in run['fills']})
    result = score(float(R), returns, spans, trading_days=td, start_sample=True)
    result.update(partial_day_return=float(partial_return),
        daily=[dict(time=int(t), return_=float(r), days_covered=float(s))
               for t, r, s in zip(times[1:], returns, spans)],
        valuation='par trade-close proxy; not oracle equity, not an official submitted score')
    return result


def main():
    captures = sorted((HERE / 'par_history').glob('*/manifest.json'))
    capture = captures[-1].parent
    manifest = json.loads((capture / 'manifest.json').read_text())
    assert not manifest['failures']
    for name, digest in manifest['files'].items():
        assert sha(capture / name) == digest, name
    out = HERE / 'par_replay'
    out.mkdir(exist_ok=False)
    (out / 'runs').mkdir()
    inp, arrays, protocol, reference = data()
    protocol = copy.deepcopy(protocol)
    protocol['scenarios']['execution_stress'] = dict(impact_bps=20., participation=.0025, adverse_funding_bps_hour=0.)
    cfg = json.loads((HERE / 'protocol.json').read_text())
    selected = json.loads((HERE / 'partial_artifacts/selection.json').read_text())['selected']
    family, window = selected.rsplit('_', 1)
    forms = json.loads((HERE / f'partial_artifacts/formations_{window}.json').read_text())
    strict = json.loads((HERE / 'artifacts/formations_1440.json').read_text())
    multiplier = json.loads((ROOT / 'experiments/performance_attribution_20261016/artifacts/budget_frozen.json').read_text())['multiplier']
    # A distinct execution market: missing data stay NaN / unavailable. The
    # reference market remains unchanged and supplies the frozen predictors.
    close = np.full_like(reference.close, np.nan)
    volume = np.zeros_like(close)
    trades = np.zeros_like(close)
    rates = np.zeros_like(close)
    observed = np.zeros_like(close, dtype=bool)
    index = {int(t): i for i, t in enumerate(reference.times)}
    alignment = []
    for j, symbol in enumerate(reference.symbols):
        candles = json.loads((capture / f'candles_{symbol}_1h.json').read_text())['response']
        for candle in candles:
            assert (int(candle['T']) + 1) % 3_600_000 == 0
            available = (int(candle['T']) + 1) // 1000
            if available not in index or candle['T'] >= manifest['cutoff_ms']:
                continue
            i = index[available]
            close[i, j] = float(candle['c'])
            volume[i, j] = float(candle['v'])
            trades[i, j] = int(candle['n'])
        funding = json.loads((capture / f'funding_{symbol}.json').read_text())
        assert funding['pagination_complete']
        for event in funding['rows']:
            t = funding_bucket(event['time'])
            if t in index:
                i = index[t]
                rates[i, j] = float(event['fundingRate'])
                observed[i, j] = True
        mask = np.isfinite(close[:, j])
        alignment.append(dict(symbol=symbol, overlapping_closed_hours=int(mask.sum()),
            traded_hours=int(np.sum(mask & (trades[:, j] > 0))),
            first_close_unix=int(reference.times[np.flatnonzero(mask)[0]]),
            last_close_unix=int(reference.times[np.flatnonzero(mask)[-1]])))
    available = np.flatnonzero(np.isfinite(close).all(axis=1) & observed.all(axis=1))
    # Start on the first available 12:00 UTC valuation boundary, independent of
    # outcomes. End at the last already-frozen signal time, never extrapolate.
    first = next(int(i) for i in available if reference.times[i] % 86400 == 43200)
    last = int(available[-1])
    assert np.isfinite(close[first:last + 1]).all()
    assert observed[first:last + 1].all()
    market = Market(reference.times, close, volume, trades, rates, observed, reference.symbols)
    frozen_files = [Path(__file__), HERE / 'policy.py', HERE / 'models.py', HERE / 'partial.py', HERE / 'protocol.json',
        capture / 'manifest.json', HERE / 'artifacts/formations_1440.json',
        HERE / f'partial_artifacts/formations_{window}.json', HERE / 'partial_artifacts/selection.json',
        ROOT / 'experiments/risk_preserving_20261016/artifacts/inputs.npz',
        ROOT / 'experiments/risk_preserving_20261016/artifacts/targets.npz']
    plan = dict(created_at=time.time(), start_unix=int(reference.times[first]), end_unix=int(reference.times[last]),
        data_cutoff_kst=manifest['cutoff_kst'], alignment=alignment, protocol=protocol,
        selected_partial=selected, sources={str(p.relative_to(ROOT)): sha(p) for p in frozen_files},
        protected=protected(), independent_evidence=False, live_orders_sent=0,
        signal_source='Existing frozen reference-market signals, covariance and pair formation models; no par fitting',
        valuation='Actual par hourly trade close, NOT historical mark/oracle or executable quote',
        funding='Actual par rates, but notional uses par trade-close proxy because oracle history is unavailable',
        execution='One-hour delay, two independent legs, min(source, execution) observed volume participation cap',
        policy='No model selection on these outcomes; same capital and shared caps, not equal realized exposure',
        limitations=['Inputs end before the latest capture, so this is not a through-today signal replay',
            'Historical L2, queue, intrahour prices and oracle valuations are unavailable',
            'Historical volume participation is an assumption, not proof of liquidity at an order time',
            'Baseline marks residual positions; paired policy attempts to close near the end; estimated exit cost is reported separately'])
    save(out / 'plan.json', plan)
    rows = []
    for scenario in ('base', 'execution_stress'):
        for name in ('first', 'second', 'cointegration_ou_1440', 'partial_' + selected):
            if name == 'first':
                weights, controller = arrays['incumbent'], Controller()
            elif name == 'second':
                weights = arrays['candidate'] * multiplier
                controller = SecondController(reference.times, arrays['retention'])
            elif name.startswith('cointegration'):
                weights = np.zeros_like(arrays['incumbent'])
                controller = PairsController(reference, strict, cfg, 'ou', protocol['scenarios'][scenario], last)
            else:
                weights = np.zeros_like(arrays['incumbent'])
                controller = PartialController(reference, forms, cfg, family, protocol['scenarios'][scenario], last)
            result = simulate(market, weights, arrays['sigma'], protocol, first, last, scenario, controller)
            result.update(policy=name, score_proxy=score_proxy(result), evidence='par trade-close proxy replay')
            if isinstance(controller, PairsController):
                result['events'] = controller.events
                result['metrics']['pair_entries'] = sum(e['event'] == 'entry_intent' for e in controller.events)
                result['metrics']['partial_entry_aborts'] = sum(e['event'] == 'partial_entry_abort' for e in controller.events)
            result['gates']['all_funding_rates_observed'] = bool(observed[first:last + 1].all())
            result['gates']['fills_require_both_observed_traded_hours'] = all(
                trades[index[f['time']], f['asset']] > 0 and trades[index[f['decision_time']], f['asset']] > 0
                for f in result['fills'])
            gross = float(abs(np.array(result['metrics']['final_qty'])) @ close[last])
            exit_cost = gross * (protocol['execution']['fee_bps'] + protocol['scenarios'][scenario]['impact_bps']) / 10000
            result['metrics']['estimated_residual_exit_cost'] = exit_cost
            result['metrics']['pnl_after_estimated_exit_cost'] = result['metrics']['net_pnl'] - exit_cost
            with gzip.open(out / 'runs' / f'{name}-{scenario}.json.gz', 'wt') as f:
                json.dump(result, f, allow_nan=False, default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist())
            rows.append({k: result[k] for k in ('policy', 'scenario', 'metrics', 'score_proxy', 'gates', 'independent_accounting_audit')})
            print(name, scenario, 'PnL', result['metrics']['net_pnl'], 'fills', result['metrics']['fills'], 'gates', result['gates'], flush=True)
    for name, digest in plan['sources'].items():
        assert sha(ROOT / name) == digest
    save(out / 'results.json', dict(rows=rows, protected_after=protected(), live_orders_sent=0,
        candidate_replaces_live=False, execution_assumptions_confirmed=False,
        numerical_validation_passed=all(all(r['gates'].values()) for r in rows)))


if __name__ == '__main__':
    main()
