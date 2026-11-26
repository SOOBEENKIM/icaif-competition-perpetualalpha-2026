"""Re-run the actual selected historical pair engine with fixed baseline ablations."""
import copy
import gzip
import json
import numpy as np

from competition_live.profile import BUNDLE, load_profile
from competition_live.scoring import evaluate
from experiments.decision_gate_pairs_20261017.portfolio import replay
from .protocol import OUT, PRIOR, freeze


def native(x):
    return x.tolist() if hasattr(x, 'tolist') else x.item()


def main():
    spec = freeze()
    p = load_profile()
    data = dict(np.load(BUNDLE / 'data/reference.npz', allow_pickle=False))
    state = dict(np.load(BUNDLE / 'models/reference_gate_state.npz', allow_pickle=False))
    weights = dict(np.load(PRIOR / 'reference_targets.npz', allow_pickle=False))
    ids = np.searchsorted(weights['times'], data['times'])
    np.testing.assert_array_equal(weights['times'][ids], data['times'])
    np.testing.assert_allclose(weights['incumbent_published'][ids], data['first'], rtol=1e-11, atol=1e-13)
    expected = json.loads((BUNDLE / 'expected.json').read_text())
    windows = [x for x in expected['runs'] if x['venue'] == 'reference'
               and x['arm'] == p['candidate'] and x['phase'] in spec['reference_windows']]
    out = OUT / 'reference'
    out.mkdir(exist_ok=True)
    rows = []
    for window in windows:
        for variant, definition in spec['variants'].items():
            name = f"{variant}_{window['phase']}_{window['scenario']}"
            summary_path = out / (name + '.json')
            if summary_path.exists():
                rows.append(json.loads(summary_path.read_text()))
                continue
            d = dict(data)
            d['first'] = weights[definition['target']][ids] * definition['multiplier']
            cfg = copy.deepcopy(p['configs']['reference'])
            cfg['conditional_completion'] = {'risk_aversion': 0.}
            result = replay(d, state, None, cfg, p['trading'], p['candidate'],
                            window['scenario'], window['start'], window['end'])
            m = result['metrics']
            samples = [dict(time=int(s['time']), equity=s['equity'] + 50000.)
                       for s in result['snapshots'] if s['time'] % 86400 == 43200]
            team = evaluate(samples, trading_days=len({f['time'] // 86400 for f in result['fills']}))
            row = dict(variant=variant, phase=window['phase'], scenario=window['scenario'],
                       start=window['start'], end=window['end'], net_pnl=m['net_pnl'],
                       team_return=team['return_fraction'], Q=team['Q'], score=team['score'],
                       volatility=team['volatility'], daily_returns=team['completed_returns'],
                       **{k: m[k] for k in ('fees', 'funding', 'price_move_pnl', 'execution_cost',
                          'max_drawdown', 'mean_gross', 'maximum_gross', 'fills', 'completed_pairs',
                          'risk_breach_bars', 'maximum_accounting_error', 'pnl_after_liquidation_reserve')})
            row['mean_notional'] = float(np.mean([s['gross'] * s['equity'] for s in result['snapshots']]))
            row['short_opening_fills'] = sum(f['signed_qty'] < 0 and not f['reduce_only'] for f in result['fills'])
            row['traded_symbols'] = sorted({f['symbol'] for f in result['fills']})
            row['timing_pass'] = all(f['source_available'] <= f['decision_time'] < f['time']
                                     for f in result['fills'])
            assert row['maximum_accounting_error'] < 1e-6 and row['timing_pass']
            with gzip.open(out / (name + '.json.gz'), 'wt') as f:
                json.dump(result, f, default=native)
            summary_path.write_text(json.dumps(row, indent=2))
            rows.append(row)
            (OUT / 'reference_comparison.json').write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
    (OUT / 'reference_comparison.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
