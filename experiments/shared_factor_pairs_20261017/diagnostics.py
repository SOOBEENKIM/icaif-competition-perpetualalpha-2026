"""Post-evaluation diagnostics; never reselect on later outcomes."""
import gzip
import json
import time
import numpy as np
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.par_direct_forecast_20261017.data import ROOT, save, sha
from experiments.dense_history_pairs_20261017.research import load
from .research import HERE, OUT, PARENT, MATCHED, ARMS, verify, path_for


def main():
    cfg = json.loads((ROOT / 'experiments/hourly_refresh_pairs_20261017/reference_protocol.json').read_text())
    d, _, _ = inputs(cfg)
    a = load(OUT / 'reference_issued_forecasts.npz')
    actual = a['target'] * a['unit']; times = a['times']
    valid = (a['fresh'] & a['terminal_fresh'] & np.isfinite(actual)
             & np.isfinite(a['prediction_direct']) & np.isfinite(a['prediction_combined']))
    forecasts = {'zero': np.zeros_like(actual), **{k: a[f'prediction_{k}'] for k in ['direct', *ARMS]}}
    prediction_rows, accounts, entry_rows, component_rows = [], [], [], []
    for phase in ['validation', 'comparison']:
        with gzip.open(path_for('reference', phase, 'base'), 'rt') as f: parent = json.load(f)
        daily = (times >= parent['start']) & (a['maturity'] <= parent['end']) & (times % 86400 == 43200)
        mask = valid & daily[:, None]
        for kind, pred in forecasts.items():
            y, v = actual[mask], pred[mask]
            prediction_rows.append(dict(phase=phase, model=kind, pair_days=len(y),
                rmse_bps=float(np.sqrt(np.mean((v-y)**2))*10000), mean_actual_bps=float(y.mean()*10000),
                mean_prediction_bps=float(v.mean()*10000), directional_accuracy=float(np.mean(y*v>0)) if kind != 'zero' else None))
        btc_mask = daily & np.isfinite(a['btc_prediction']) & np.isfinite(a['btc_target'])
        for kind, pred, target, mask_ in [('btc', a['btc_prediction'], a['btc_target'], btc_mask),
                ('residual', a['component_residual'], a['residual_target'], mask)]:
            y, v = target[mask_], pred[mask_]
            component_rows.append(dict(phase=phase, component=kind, samples=len(y),
                rmse_bps=float(np.sqrt(np.mean((v-y)**2))*10000), zero_rmse_bps=float(np.sqrt(np.mean(y*y))*10000),
                mean_prediction_bps=float(v.mean()*10000), mean_actual_bps=float(y.mean()*10000)))
        for kind, arm in {'direct': MATCHED, **ARMS}.items():
            for sc in ['base', 'stress']:
                path = (PARENT if kind == 'direct' else OUT) / 'runs' / f'reference-{phase}-{arm}-{sc}.json.gz'
                with gzip.open(path, 'rt') as f: run = json.load(f)
                audit = episode_attribution(run, d, cfg['scenarios'][sc])
                intents = [e for e in run['events'] if e['event'] == 'pair_intent']
                for e in audit['episodes']:
                    intent = next(v for v in intents if v['time'] == e['time'])
                    e['expected_price_bps'] = intent['expected_price_bps']; e['direction'] = intent['direction']
                    source = intent['forecast_source']; j = int(np.searchsorted(times, source)); k = intent['pair_index']
                    if j < len(times) and times[j] == source and valid[j, k] and a['maturity'][j] <= parent['end']:
                        entry_rows.append(dict(phase=phase, model=kind, scenario=sc, time=e['time'], pair=e['pair'],
                            source_prediction_bps=float(forecasts[kind][j,k]*intent['direction']*10000),
                            source_realized_24h_bps=float(actual[j,k]*intent['direction']*10000), includes_fill=e['fills']>0,
                            note='Fixed-source 24h price label, not the delayed fill or renewed exit return.'))
                totals = {key: sum(e[key] for e in audit['episodes']) for key in ['price_move', 'execution_cost', 'fees', 'funding', 'net_pnl']}
                accounts.append(dict(phase=phase, model=kind, scenario=sc, base_owned_pnl=run['metrics']['base_owned_pnl'], totals=totals, **audit))
    books = json.loads((OUT / 'books/results.json').read_text())
    diagnostics = dict(at_unix=time.time(), prediction_checks=prediction_rows, component_checks=component_rows,
        account_attribution=accounts, entry_source_labels=entry_rows, book_summary=books['summary'],
        interpretation='Only combined was selectable before later evaluation. Market/residual arms diagnose the same actual pair hedge, not independently tradable market-neutral component portfolios. Nonoverlapping source days share BTC across pairs. RMSE is diagnostic, not an automatic veto of observed after-cost PnL improvement.',
        orders_sent=0, live_strategy_changed=False, reserved_raw_read=False, historical_independence=False)
    save(OUT / 'diagnostics.json', diagnostics)
    for row in prediction_rows: print('FORECAST', row, flush=True)
    for row in component_rows: print('COMPONENT', row, flush=True)
    for row in accounts:
        if row['phase'] == 'comparison': print('ACCOUNT', row['model'], row['scenario'], row['totals'], flush=True)
    verify(json.loads((OUT / 'frozen.json').read_text())['sources'])
    verify(json.loads((OUT / 'books/frozen.json').read_text())['sources'])
    verify(json.loads((ROOT / 'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'])
    paths = [HERE / 'diagnostics.py', OUT / 'diagnostics.json', OUT / 'results.json', OUT / 'books/results.json']
    save(OUT / 'diagnostics_frozen.json', dict(at_unix=time.time(), sources={str(p.relative_to(ROOT)): sha(p) for p in paths}, post_evaluation_only=True))
    print('DONE diagnostics', flush=True)


if __name__ == '__main__': main()
