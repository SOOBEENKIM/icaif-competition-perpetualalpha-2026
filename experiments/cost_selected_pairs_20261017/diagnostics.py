"""Audit proxy supervision separately from independently executed account PnL."""
import gzip
import json
from pathlib import Path
from collections import Counter
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
    a = load(ART/'reference_issued_forecasts.npz'); rows = []; selections = []
    audit = json.loads((ART/'reference_weight_audit.json').read_text())
    for phase in ['validation', 'comparison']:
        start = ts(cfg['validation_start'] if phase == 'validation' else cfg['comparison_start'])
        end = ts(cfg['comparison_start'] if phase == 'validation' else cfg['end'])
        window = (a['times'] >= start) & (a['maturity'] <= end) & (a['times'] % 86400 == 0)
        for k, pair in enumerate(cfg['dense_pairs']):
            ix = np.flatnonzero(window & np.isfinite(a['payoff'][:, k]).all(-1))
            for kind in [*spec['experts'], *spec['candidates']]:
                chosen = np.full(len(ix), spec['experts'].index(kind), dtype=int) if kind in spec['experts'] else a[f'selected_{kind}'][ix, k]
                payoff = np.zeros(len(ix)); active = np.zeros(len(ix), bool)
                for j, (source, model) in enumerate(zip(ix, chosen)):
                    if model >= 0:
                        payoff[j] = a['payoff'][source, k, model]
                        active[j] = a['actions'][source, k, model] != 0
                rows.append(dict(phase=phase, pair=pair, model=kind, nonoverlap_opportunities=len(ix),
                    selected_opportunities=int(np.sum(chosen >= 0)), active_proxy_trades=int(active.sum()),
                    mean_source_gross_return=float(payoff.mean()), sum_source_gross_returns=float(payoff.sum()),
                    profitable_proxy_trades=int(np.sum(payoff > 0)), losing_proxy_trades=int(np.sum(payoff < 0)),
                    selected_experts=dict(Counter(spec['experts'][j] if j >= 0 else 'none' for j in chosen)),
                    not_account_return=True))
            decisions = [r for r in audit if r['pair_index'] == k and start <= r['time'] < end]
            for kind in spec['candidates']:
                choices = [r['choices'][kind] for r in decisions]
                selections.append(dict(phase=phase, pair=pair, model=kind, decisions=len(choices),
                    choices=dict(Counter(spec['experts'][r['selected']] if r['selected'] >= 0 else 'none' for r in choices)),
                    positive_selection_minimum_active_rows=min((r['active_rows'][r['selected']] for r in choices if r['selected'] >= 0), default=None)))
    save(ART/'proxy_diagnostics.json', dict(rows=rows, hourly_selections=selections, post_evaluation_only=True,
        not_account_pnl_or_roi=True, note='Proxy trade at fixed source unit gross is only a model selection label; actual delayed/partial execution is evaluated in separate full-account replays.'))
    d, _, _ = inputs(cfg); attribution = []
    for row in result['rows']:
        if row['venue'] != 'reference' or row['arm'].split('_')[-2] not in spec['candidates']:
            continue
        key = '-'.join(row[k] for k in ['venue', 'phase', 'arm', 'scenario'])
        with gzip.open(ART/'runs'/f'{key}.json.gz', 'rt') as f:
            run = json.load(f)
        account = episode_attribution(run, d, cfg['scenarios'][row['scenario']])
        attribution.append(dict(phase=row['phase'], arm=row['arm'], scenario=row['scenario'], **account))
    save(ART/'attribution.json', dict(rows=attribution, orders_sent=0, not_used_for_selection=True))
    print('PROXY DIAGNOSTICS', len(rows), 'ATTRIBUTED ACCOUNTS', len(attribution))
    for row in selections:
        print(row)
    print(protected())


if __name__ == '__main__':
    main()
