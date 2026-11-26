"""Predeclared mechanism/capacity audit on previously used quote tapes.

No return from a 24/72-hour trade is inferred from a one-hour tape.
"""
import json
import time
from pathlib import Path
import numpy as np
from experiments.basis_carry_20261017.probe import load_tape, asof
from experiments.cost_horizon_20261016.research import data
from experiments.pairs_forecast_20261017.model import design
from experiments.pairs_research_20261017.research import save, sha
from experiments.complementary_alpha_20261017.research import protected
from .engine import quote_pair

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    out = HERE / 'artifacts'; out.mkdir(exist_ok=False)
    cfg = json.loads((HERE / 'protocol.json').read_text())
    prior = json.loads((ROOT / 'experiments/basis_carry_20261017/protocol.json').read_text())
    save(out / 'frozen.json', dict(at=time.time(), sources={str(f.relative_to(ROOT)): sha(f) for f in [*HERE.glob('*.py'), HERE / 'protocol.json']}, protected=protected(),
        historical_data_reused=True, actual_execution_model='L2 depth, not candle-volume percentage'))
    _, _, _, m = data()
    forecast_cfg = json.loads((ROOT / 'experiments/pairs_forecast_20261017/protocol.json').read_text())
    features = design(m, forecast_cfg)
    source = ROOT / 'experiments/pairs_research_20261017/par_history/20261016T173824Z/metadata.json'
    metadata = json.loads(source.read_text())
    if isinstance(metadata, dict): metadata = metadata['response']
    meta = {r['name']: r for r in metadata[0]['universe']}
    rows, coverage = [], []
    freshness = dict(book_max_age_ms=cfg['maximum_book_age_ms'], context_max_age_ms=cfg['maximum_context_age_ms'])
    # Read the exact allowlist from the earlier declared, already-used study.
    tapes = prior['tapes']
    for path in tapes:
        streams, receipt = load_tape(ROOT / path); coverage.append(receipt)
        for now in range((receipt['start_ms'] // 300000 + 1) * 300000, receipt['end_ms'] + 1, 300000):
            i = int(np.searchsorted(m.times, now // 1000, side='right') - 1)
            if i < 0 or now / 1000 - m.times[i] > 3600: continue
            snapshots = {f'par:{s}': asof(stream, now, freshness) for s, stream in streams.items()}
            for k, pair in enumerate(cfg['pairs']):
                coins = ['par:' + s for s in pair]
                if any(not snapshots.get(c) for c in coins): continue
                correlation = float(features['x'][i, k, features['names'].index('correlation_168')])
                if correlation < cfg['minimum_correlation']: continue
                for horizon in cfg['horizons_hours']:
                    for scenario, sc in cfg['scenarios'].items():
                        choices = [quote_pair(coins, snapshots, meta, features['beta'][i, k], direction, horizon, cfg, sc) for direction in (-1, 1)]
                        choices = [p for p in choices if p is not None]
                        if not choices: continue
                        p = max(choices, key=lambda p: p['predicted_net_bps'])
                        rows.append(dict(time=now, tape=path, scenario=scenario, correlation=correlation,
                            proposed=p, passes_expected_edge=p['predicted_net_bps'] > cfg['entry_buffer_bps']))
    summaries = []
    for horizon in cfg['horizons_hours']:
        for scenario in cfg['scenarios']:
            rs = [r for r in rows if r['scenario'] == scenario and r['proposed']['horizon_hours'] == horizon]
            edge = [r['proposed']['predicted_net_bps'] for r in rs]
            passing = [r for r in rs if r['passes_expected_edge']]
            summaries.append(dict(horizon_hours=horizon, scenario=scenario, evaluated=len(rs),
                positive_forecast_count=len(passing), maximum_predicted_bps=max(edge) if edge else None,
                median_predicted_bps=float(np.median(edge)) if edge else None,
                passing_pairs=sorted(set('/'.join(r['proposed']['coins']) for r in passing)),
                completed_horizon_observations=0, realized_pnl=None, official_Q=None))
    save(out / 'results.json', dict(summaries=summaries, decisions=rows, coverage=coverage,
        orders_sent=0, protected_after=protected(), historical_independence=False,
        limitation='Forecast/capacity diagnostic only. Every 24/72h exit falls outside its short tape. No realized strategy return is estimated.'))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__': main()
