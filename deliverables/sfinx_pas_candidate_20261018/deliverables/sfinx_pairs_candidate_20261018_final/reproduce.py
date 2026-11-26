"""Offline, source-contained reproduction of the retained SFINX candidate."""
import argparse
import copy
import csv
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    return json.loads(json.dumps(value, allow_nan=False,
        default=lambda v: v.tolist() if hasattr(v, 'tolist') else v.item()))


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def verify(root, manifest):
    for name, expected in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError('Bundle content changed: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cached-models', action='store_true',
                        help='Skip refitting paired models; still verify frozen input/model hashes and account replay.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'manifest.json').read_text())
    verify(root, manifest)
    out = args.output.resolve()
    if out.exists(): raise FileExistsError('Use a new output directory; no archived result is overwritten')
    out.mkdir(parents=True); (out/'runs').mkdir(); (out/'models').mkdir()

    def offline_guard(event, values):
        if event in {'socket.connect', 'socket.getaddrinfo', 'urllib.Request', 'subprocess.Popen', 'os.system'}:
            raise RuntimeError('This reproducer is offline: ' + event)
    sys.addaudithook(offline_guard)
    sys.path.insert(0, str(root/'source'))
    import numpy as np
    from experiments.dense_history_pairs_20261017.model import design, labels, map_state
    from experiments.adaptive_residual_pairs_20261017.model import frozen_predictions
    from experiments.rolling_feature_pairs_20261017.model import rolling as direct_rolling
    from experiments.decision_gate_pairs_20261017.model import rolling, map_gate
    from experiments.decision_gate_pairs_20261017.portfolio import replay

    def load(path):
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}

    specification = json.loads((root/'config.json').read_text())
    expected = json.loads((root/'expected.json').read_text())
    data = {venue: load(root/f'data/{venue}.npz') for venue in specification['venues']}
    states = {venue: load(root/f'models/{venue}_gate_state.npz') for venue in specification['venues']}
    model_checks = []
    if not args.cached_models:
        spec = specification['gate_spec']; pairs = specification['pairs']
        models = {(tuple(pair), 24): json.loads((root/'models'/('_'.join(pair)+'_24.json')).read_text()) for pair in pairs}
        for source, venues in [('reference', ['reference']), ('par_reference', ['par', 'par_extended'])]:
            reference = load(root/f'data/{source}_training.npz'); times = reference['market_times']
            started = time.perf_counter()
            features = design(times, reference['close'], reference['volume'], reference['trades'],
                              specification['trading']['symbols'], pairs)
            raw = frozen_predictions(features, models, pairs, 24)
            y, maturity = labels(times, reference['close'], features, 24)
            terminal = np.zeros_like(features['fresh']); terminal[:-24] = features['fresh'][24:]
            direct_spec = {**spec, 'candidates': ['direct'], 'diagnostic_variants': []}
            predictions, _, direct_records = direct_rolling(times, features, y, maturity, raw, terminal, direct_spec)
            actions, scores, available, stamps, records = rolling(times, features, y, maturity, raw, terminal, spec)
            assert all(record['last_maturity'] < record['fit_time'] for record in records)
            for venue in venues:
                baseline = map_state(data[venue]['times'], reference, features, predictions['direct'], features['pairs'],
                                     source_hours=1, lifetime_hours=1)
                state = map_gate(data[venue]['times'], times, actions, available, baseline)
                assert set(state) == set(states[venue])
                for key in state:
                    np.testing.assert_array_equal(state[key], states[venue][key], err_msg=venue+':'+key)
                states[venue] = state
                np.savez_compressed(out/'models'/f'{venue}_gate_state.npz', **state)
            save(out/'models'/f'{source}_gate_fits.json', records)
            model_checks.append(dict(source=source, gate_fits=len(records), exact_all_mapped_fields=True,
                strictly_mature_labels=True, seconds=time.perf_counter()-started,
                maximum_solver_gap=max(r['solver']['gap'] for r in records)))
            print('REFIT', source, len(records), 'gate fits; mapped forecasts/actions exact', flush=True)

    rows = []
    for record in expected['runs']:
        venue, arm, scenario = record['venue'], record['arm'], record['scenario']
        cfg = copy.deepcopy(specification['configs']['reference' if venue=='reference' else 'par'])
        cfg['conditional_completion'] = {'risk_aversion': 0.}
        common = plain(replay(data[venue], states[venue], None, cfg, specification['trading'], arm,
                              scenario, record['start'], record['end']))
        archive_cfg = copy.deepcopy(cfg)
        archive_state = states[venue]
        if record['engine'] == 'shared_execution_pairs_20261017.portfolio':
            archive_cfg = specification['archive_configs'][venue]
            archive_state = load(root/f'models/{venue}_legacy_state.npz')
        elif record['engine'] == 'lifecycle_stability_pairs_20261018.portfolio':
            archive_cfg.update(lifecycle_kind='filter', lifecycle_models={})
        if record['engine'] == 'decision_gate_pairs_20261017.portfolio':
            run = common
        else:
            original_replay = importlib.import_module('experiments.'+record['engine']).replay
            run = plain(original_replay(data[venue], archive_state, None, archive_cfg, specification['trading'],
                                        arm, scenario, record['start'], record['end']))
        # Includes physical and owner fills, funding, all decisions, orders,
        # snapshots, metrics and daily returns, not just rounded headline PnL.
        actual_hash = digest(run)
        if actual_hash != record['canonical_run_sha256']:
            save(out/(record['id']+'_mismatch.json'), dict(expected=record, actual=run))
            raise AssertionError('Full archived account differs: ' + record['id'])
        # Older engine generations logged different rejected order intentions.
        # Reproduce those exactly above, then explicitly check the actual
        # financial path under the common candidate execution engine as well.
        financial_fields = ['metrics','fills','owner_fills','snapshots','funding_events','daily','risk_breaches','internal_transfers']
        financial_exact = all(common[field] == run[field] for field in financial_fields)
        if not financial_exact:
            save(out/(record['id']+'_common_engine_mismatch.json'), dict(archived=run, common=common))
            raise AssertionError('Common-engine financial path differs: ' + record['id'])
        with gzip.open(out/'runs'/(record['id']+'.json.gz'), 'wt') as handle:
            json.dump(run, handle, allow_nan=False)
        row = {key:record[key] for key in ['id','venue','phase','arm','scenario']}
        row.update(metrics=run['metrics'], exact_full_account=True, canonical_run_sha256=actual_hash,
            common_engine_financial_path_exact=financial_exact,
            common_engine_order_log_exact=common['orders']==run['orders'],
            common_engine_decision_log_exact=common['decisions']==run['decisions'])
        rows.append(row)
        print('EXACT', record['id'], 'PnL', round(run['metrics']['net_pnl'],6),
              'Q', run['metrics']['financial_score'], flush=True)
    result = dict(rows=rows, refitted_models=not args.cached_models, model_checks=model_checks,
        exact_accounts=len(rows), expected_accounts=len(expected['runs']),
        historical_independence=False, actual_par_live_pnl=False, network_used=False,
        orders_sent=0, live_strategy_changed=False,
        baseline_target_rule=specification['baseline_target_rule'], limits=specification['limits'])
    save(out/'results.json', result)
    with (out/'comparison.csv').open('w') as handle:
        columns=['venue','phase','arm','scenario','net_pnl','roi','financial_score','volatility_horizon',
                 'max_drawdown','fees','funding','completed_pairs','residual_gross']
        writer=csv.DictWriter(handle, fieldnames=columns);writer.writeheader()
        for row in rows:
            writer.writerow({key:row.get(key,row['metrics'].get(key)) for key in columns})
    verify(root,manifest)
    print('COMPLETE',len(rows),'exact accounts; refit',not args.cached_models,'; no orders',flush=True)


if __name__=='__main__':
    main()
