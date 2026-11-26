import copy
import gzip
import json
import time
from pathlib import Path
import numpy as np
from experiments.cost_horizon_20261016.research import data, ts, SecondController
from experiments.performance_attribution_20261016.engine import simulate
from experiments.q_execution_20261017.metrics import official_metrics, paired_uncertainty
from experiments.complementary_alpha_20261017.research import protected
from experiments.pairs_research_20261017.research import save, sha
from .policy import HorizonMatchedController

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'artifacts'
PREVIOUS = ROOT / 'experiments/pairs_forecast_20261017'


def main():
    OUT.mkdir(exist_ok=False); (OUT / 'runs').mkdir()
    plan = json.loads((HERE / 'protocol.json').read_text())
    cfg = json.loads((PREVIOUS / 'protocol.json').read_text())
    _, arrays, p, m = data(); p = copy.deepcopy(p)
    p['scenarios']['execution_stress'] = dict(impact_bps=20., participation=.0025, adverse_funding_bps_hour=0.)
    pred = np.load(PREVIOUS / 'artifacts/models/predictions.npz', allow_pickle=False)
    np.testing.assert_array_equal(pred['times'], m.times)
    deps = [*HERE.glob('*.py'), HERE / 'protocol.json', PREVIOUS / 'artifacts/models/predictions.npz']
    save(OUT / 'frozen.json', dict(at=time.time(), sources={str(f.relative_to(ROOT)): sha(f) for f in deps}, protected=protected(), independent=False))
    rows, runs = [], {}
    def prior(name, phase, scenario):
        with gzip.open(PREVIOUS / 'artifacts/runs' / f'{phase}-{name}-{scenario}.json.gz', 'rt') as f:
            return json.load(f)
    def replay(name, phase, scenario, baseline_scale=None):
        first, last = [m.exact(ts(s)) for s in plan[phase]]
        if baseline_scale is None:
            control = HorizonMatchedController(m, pred[name], pred['beta'], pred['vol'], cfg, 72, p['scenarios'][scenario], last)
            weights = np.zeros_like(arrays['incumbent'])
        else:
            control = SecondController(m.times, arrays['retention'])
            mult = json.loads((ROOT / 'experiments/performance_attribution_20261016/artifacts/budget_frozen.json').read_text())['multiplier']
            weights = arrays['candidate'] * mult * baseline_scale
        result = simulate(m, weights, arrays['sigma'], p, first, last, scenario, control)
        result.update(policy=name, phase=phase, official=official_metrics(result))
        if baseline_scale is None:
            result['events'] = control.events
            result['metrics']['balanced_open'] = sum(e['event'] == 'balanced_open' for e in control.events)
        key = f'{phase}-{name}-{scenario}'
        with gzip.open(OUT / 'runs' / f'{key}.json.gz', 'wt') as f:
            json.dump(result, f, allow_nan=False, default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist())
        rows.append({k: result[k] for k in ('policy', 'phase', 'scenario', 'metrics', 'official', 'gates')})
        runs[key] = result
        print(key, 'PnL', round(result['metrics']['net_pnl'], 2), 'Q', round(result['official']['financial_score'], 6), flush=True)
        return result
    ranking = []
    for name in plan['arms']:
        advantage, qad, eligible = [], [], []
        for scenario in ('base', 'execution_stress'):
            r = replay(name, 'selection', scenario)
            baselines = [prior(n, 'selection', scenario) for n in ('first', 'second')]
            advantage.append(r['metrics']['net_pnl'] - max(b['metrics']['net_pnl'] for b in baselines))
            qad.append(r['official']['financial_score'] - max(b['official']['financial_score'] for b in baselines))
            eligible.append(r['metrics']['net_pnl'] > 0 and r['metrics']['balanced_open'] >= 4 and all(r['gates'].values()))
        ranking.append(dict(name=name, eligible=all(eligible) and min(advantage) > 0 and min(qad) > 0,
                            minimum_pnl_advantage=min(advantage), minimum_q_advantage=min(qad)))
    ranking.sort(key=lambda r: (-r['eligible'], -r['minimum_pnl_advantage'], -r['minimum_q_advantage'], r['name']))
    selected = ranking[0]['name']
    scale = float(np.clip(runs[f'selection-{selected}-base']['official']['daily_sample_std'] / prior('second', 'selection', 'base')['official']['daily_sample_std'], .25, 4.))
    frozen = dict(at=time.time(), selected=selected, ranking=ranking, baseline_risk_matched_scale=scale, historical_independence=False)
    save(OUT / 'selection.json', frozen)
    comparisons = []
    for scenario in ('base', 'execution_stress'):
        r = replay(selected, 'comparison', scenario)
        matched = replay('second_selection_risk_matched', 'comparison', scenario, scale)
        for name, b in [('first', prior('first', 'comparison', scenario)), ('second', prior('second', 'comparison', scenario)), ('second_risk_matched', matched)]:
            comparisons.append(dict(scenario=scenario, baseline=name,
                pnl_advantage=r['metrics']['net_pnl']-b['metrics']['net_pnl'],
                q_advantage=r['official']['financial_score']-b['official']['financial_score'],
                uncertainty=paired_uncertainty(r['official'], b['official'])))
    save(OUT / 'results.json', dict(rows=rows, selection=frozen, comparisons=comparisons, protected_after=protected(),
        independent_holdout=False, live_orders_sent=0, development_improvement=ranking[0]['eligible'] and all(c['pnl_advantage'] > 0 and c['q_advantage'] > 0 for c in comparisons)))
    print('COMPLETED', len(rows), 'runs', flush=True)


if __name__ == '__main__': main()
