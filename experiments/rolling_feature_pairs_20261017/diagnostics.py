"""Nonoverlap forecast scoring and independent account cash attribution."""
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
ART = HERE/'artifacts'
HOURLY = HERE.parent/'hourly_refresh_pairs_20261017'


def main():
    result = json.loads((ART/'results.json').read_text())
    spec = json.loads((HERE/'protocol.json').read_text())
    for rel, digest in json.loads((ART/'frozen.json').read_text())['sources'].items():
        assert sha(ROOT/rel) == digest, rel
    cfg = json.loads((HOURLY/'reference_protocol.json').read_text())
    a = load(ART/'reference_issued_forecasts.npz')
    kinds = [*spec['experts'], *spec['candidates'], *spec['diagnostic_variants']]
    rows = []
    for phase in ['validation', 'comparison']:
        start = ts(cfg['validation_start'] if phase == 'validation' else cfg['comparison_start'])
        end = ts(cfg['comparison_start'] if phase == 'validation' else cfg['end'])
        window = (a['times'] >= start) & (a['maturity'] <= end) & (a['times'] % 86400 == 0)
        for k, pair in enumerate(cfg['dense_pairs']):
            mask = window & a['fresh'][:, k] & a['terminal_fresh'][:, k] & np.isfinite(a['target'][:, k])
            for kind in kinds:
                mask &= np.isfinite(a[f'prediction_{kind}'][:, k])
            target = a['target'][mask, k]
            for kind in [*kinds, 'zero']:
                pred = np.zeros(len(target)) if kind == 'zero' else a[f'prediction_{kind}'][mask, k]/a['unit'][mask, k]
                rows.append(dict(phase=phase, pair=pair, model=kind, samples=len(target),
                    normalized_rmse=float(np.sqrt(np.mean((pred-target)**2))),
                    sign_accuracy=float(np.mean(np.sign(pred) == np.sign(target))),
                    mean_abs_normalized_prediction=float(np.mean(abs(pred))),
                    mean_abs_return_prediction=float(np.mean(abs(pred)*a['unit'][mask, k]))))
    save(ART/'forecast_diagnostics.json', dict(rows=rows, nonoverlapping_diagnostic_sources=True,
        overlapping_training_sources=True, same_common_diagnostic_mask=True, post_evaluation_only=True,
        not_used_for_selection=True, not_account_returns=True))
    d, _, _ = inputs(cfg); attribution = []
    new_kinds = [*spec['candidates'], *spec['diagnostic_variants']]
    for row in result['rows']:
        if row['venue'] != 'reference' or row['arm'].split('_')[-2] not in new_kinds:
            continue
        key = '-'.join(row[k] for k in ['venue', 'phase', 'arm', 'scenario'])
        with gzip.open(ART/'runs'/f'{key}.json.gz', 'rt') as f:
            run = json.load(f)
        audit = episode_attribution(run, d, cfg['scenarios'][row['scenario']])
        attribution.append(dict(phase=row['phase'], arm=row['arm'], scenario=row['scenario'], **audit))
    save(ART/'attribution.json', dict(rows=attribution, not_used_for_selection=True, orders_sent=0))
    print('FORECAST DIAGNOSTICS', len(rows), 'ATTRIBUTION', len(attribution))
    for row in rows:
        if row['pair'] == ['BTC', 'HYPE'] and row['model'] in ['bias', *new_kinds]:
            print(row)
    print(protected())


if __name__ == '__main__':
    main()
