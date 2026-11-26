"""Prediction and ledger diagnostics; never used to reselect a historical winner."""
import gzip
import json
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.complementary_alpha_20261017.research import protected
from experiments.dense_history_pairs_20261017.research import load
from experiments.par_direct_forecast_20261017.data import ROOT, save, sha, ts
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution

HERE = Path(__file__).resolve().parent
OUT = HERE/'artifacts'
HOURLY = HERE.parent/'hourly_refresh_pairs_20261017'


def main():
    results = json.loads((OUT/'results.json').read_text())
    spec = json.loads((HERE/'protocol.json').read_text())
    for rel, digest in json.loads((OUT/'frozen.json').read_text())['sources'].items():
        assert sha(ROOT/rel) == digest, rel
    cfg = json.loads((HOURLY/'reference_protocol.json').read_text())
    a = load(OUT/'reference_issued_forecasts.npz')
    rows = []
    for phase in ['validation', 'comparison']:
        start = ts(cfg['validation_start'] if phase == 'validation' else cfg['comparison_start'])
        end = ts(cfg['comparison_start'] if phase == 'validation' else cfg['end'])
        window = (a['times'] >= start) & (a['maturity'] <= end) & (a['times'] % 86400 == 0)
        for k, pair in enumerate(cfg['dense_pairs']):
            valid = window & a['fresh'][:, k] & a['terminal_fresh'][:, k] & np.isfinite(a['target'][:, k])
            valid &= np.isfinite(a['issued'][:, k]).all(-1)
            y = a['target'][valid, k]
            predictions = {kind: a['issued'][valid, k, j] for j, kind in enumerate(spec['experts'])}
            predictions.update({kind: a[f'prediction_{kind}'][valid, k] for kind in spec['candidates']})
            predictions['zero'] = np.zeros(len(y))
            for kind, prediction in predictions.items():
                row = dict(phase=phase, pair=pair, model=kind, samples=len(y),
                           normalized_rmse=float(np.sqrt(np.mean((prediction-y)**2))),
                           clipped_target_rmse=float(np.sqrt(np.mean((prediction-np.clip(y, -4, 4))**2))),
                           sign_accuracy=float(np.mean(np.sign(prediction) == np.sign(y))),
                           mean_abs_prediction=float(np.mean(abs(prediction))),
                           mean_realized_normalized_return=float(np.mean(y)),
                           average_weights=None)
                if kind in spec['candidates']:
                    row['average_weights'] = a[f'weights_{kind}'][valid, k].mean(0).tolist()
                rows.append(row)
    save(OUT/'forecast_diagnostics.json', dict(rows=rows, nonoverlapping_sources=True,
         fair_common_source_mask=True, post_evaluation_only=True, not_used_for_selection=True,
         normalization='Each prediction/target uses its original source volatility and fixed beta.',
         not_trading_pnl=True, historical_independence=False))
    d, _, _ = inputs(cfg); attributions = []
    for r in results['rows']:
        if r['venue'] != 'reference' or r['arm'].split('_')[-2] not in spec['candidates']:
            continue
        name = '-'.join(r[k] for k in ['venue', 'phase', 'arm', 'scenario'])
        with gzip.open(OUT/'runs'/f'{name}.json.gz', 'rt') as f:
            run = json.load(f)
        audit = episode_attribution(run, d, cfg['scenarios'][r['scenario']])
        attributions.append(dict(phase=r['phase'], arm=r['arm'], scenario=r['scenario'], **audit))
    save(OUT/'attribution.json', dict(rows=attributions, not_used_for_selection=True, orders_sent=0))
    print('FORECAST DIAGNOSTICS', len(rows), 'LEDGER ATTRIBUTIONS', len(attributions))
    for r in rows:
        if r['pair'] == ['BTC', 'HYPE'] and r['model'] in ['bias', 'equal', 'imse', 'stack']:
            print(r)
    print(protected())


if __name__ == '__main__':
    main()
