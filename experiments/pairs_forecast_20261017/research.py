"""One registered candidate batch, frozen selection, then historical comparison."""
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
from replay_lab.ema_rebuild.engine import Controller
from .model import fit_predict, design
from .policy import ForecastController

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'artifacts'


def main():
    OUT.mkdir(exist_ok=False); (OUT / 'runs').mkdir()
    cfg = json.loads((HERE / 'protocol.json').read_text())
    inp, arrays, p, m = data(); p = copy.deepcopy(p)
    p['scenarios']['execution_stress'] = dict(impact_bps=20., participation=.0025, adverse_funding_bps_hour=0.)
    multiplier = json.loads((ROOT / 'experiments/performance_attribution_20261016/artifacts/budget_frozen.json').read_text())['multiplier']
    sources = [*HERE.glob('*.py'), HERE / 'protocol.json',
        ROOT / 'experiments/risk_preserving_20261016/artifacts/inputs.npz',
        ROOT / 'experiments/risk_preserving_20261016/artifacts/targets.npz']
    save(OUT / 'frozen.json', dict(at=time.time(), sources={str(f.relative_to(ROOT)): sha(f) for f in sources},
        protected=protected(), prior_failed_candidates=12, historical_independence=False, live_orders_sent=0))
    forecasts, d, trained = fit_predict(m, cfg, OUT / 'models')
    rows, runs = [], {}
    def replay(name, phase, scenario, scale=1.):
        first, last = [m.exact(ts(s)) for s in cfg[phase]]
        if name == 'first': weights, control = arrays['incumbent'], Controller()
        elif name in ('second', 'second_selection_risk_matched'):
            weights = arrays['candidate'] * multiplier * scale
            control = SecondController(m.times, arrays['retention'])
        else:
            horizon = int(name.rsplit('_', 1)[1]); weights = np.zeros_like(arrays['incumbent'])
            control = ForecastController(m, forecasts[name], d['beta'], d['vol'], cfg, horizon, p['scenarios'][scenario], last)
        result = simulate(m, weights, arrays['sigma'], p, first, last, scenario, control)
        result.update(policy=name, phase=phase, official=official_metrics(result))
        if isinstance(control, ForecastController):
            result['events'] = control.events
            result['metrics'].update(pair_entries=sum(e['event'] == 'entry_intent' for e in control.events),
                balanced_open=sum(e['event'] == 'balanced_open' for e in control.events),
                unhedged_aborts=sum(e['event'] == 'unhedged_abort' for e in control.events))
            result['gates']['past_only_fit'] = all(0 < trained[m.exact(e['time']), e['pair']] <= e['time']
                for e in control.events if e['event'] == 'entry_intent')
        result['metrics']['estimated_residual_exit_cost'] = float(abs(np.array(result['metrics']['final_qty'])) @ m.close[last]) * (9 + p['scenarios'][scenario]['impact_bps']) / 10000
        key = f'{phase}-{name}-{scenario}'
        with gzip.open(OUT / 'runs' / f'{key}.json.gz', 'wt') as f:
            json.dump(result, f, allow_nan=False, default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist())
        rows.append({k: result[k] for k in ('policy', 'phase', 'scenario', 'metrics', 'official', 'gates')})
        runs[key] = result
        print(key, 'PnL', round(result['metrics']['net_pnl'], 2), 'Q', round(result['official']['financial_score'], 6),
              'opens', result['metrics'].get('balanced_open'), 'gates', all(result['gates'].values()), flush=True)
        return result
    names = [f'{model}_{h}' for h in cfg['horizons'] for model in cfg['models']]
    for name in ['first', 'second', *names]:
        for scenario in ('base', 'execution_stress'): replay(name, 'selection', scenario)
    ranking = []
    for name in names:
        differences, score_differences, passes = [], [], []
        for scenario in ('base', 'execution_stress'):
            r = runs[f'selection-{name}-{scenario}']
            comparisons = [runs[f'selection-{n}-{scenario}'] for n in ('first', 'second')]
            differences.append(r['metrics']['net_pnl'] - max(x['metrics']['net_pnl'] for x in comparisons))
            score_differences.append(r['official']['financial_score'] - max(x['official']['financial_score'] for x in comparisons))
            passes.append(all(r['gates'].values()) and r['metrics']['net_pnl'] > 0 and r['metrics']['balanced_open'] >= 4)
        ranking.append(dict(name=name, eligible=all(passes) and min(differences) > 0 and min(score_differences) > 0,
                            minimum_pnl_improvement=min(differences), minimum_q_improvement=min(score_differences)))
    ranking.sort(key=lambda r: (-r['eligible'], -r['minimum_pnl_improvement'], -r['minimum_q_improvement'], r['name']))
    selected = ranking[0]['name']
    nominee = runs[f'selection-{selected}-base']; base = runs['selection-second-base']
    scale = float(np.clip(nominee['official']['daily_sample_std'] / max(base['official']['daily_sample_std'], 1e-12), .25, 4.))
    selection = dict(selected=selected, eligible=ranking[0]['eligible'], ranking=ranking,
                     frozen_at=time.time(), comparison_not_read=True, second_risk_matched_scale=scale)
    save(OUT / 'selection.json', selection)
    comparisons = []
    for phase in ('comparison', 'recent'):
        for scenario in ('base', 'execution_stress'):
            for name in ('first', 'second', selected, 'second_selection_risk_matched'):
                replay(name, phase, scenario, scale=scale if name == 'second_selection_risk_matched' else 1.)
            c, b = [runs[f'{phase}-{name}-{scenario}'] for name in (selected, 'second')]
            comparisons.append(dict(phase=phase, scenario=scenario, pnl_change=c['metrics']['net_pnl']-b['metrics']['net_pnl'],
                q_change=c['official']['financial_score']-b['official']['financial_score'],
                uncertainty=paired_uncertainty(c['official'], b['official'])))
    frozen = json.loads((OUT / 'frozen.json').read_text())
    for name, digest in frozen['sources'].items(): assert sha(ROOT / name) == digest
    save(OUT / 'results.json', dict(selection=selection, rows=rows, comparisons=comparisons, protected_after=protected(),
        historical_independence=False, live_orders_sent=0, live_replacement=False,
        development_improvement=selection['eligible'] and all(r['pnl_change'] > 0 and r['q_change'] > 0
            for r in comparisons if r['phase'] == 'comparison')))
    print('COMPLETED', len(rows), 'runs', flush=True)


if __name__ == '__main__':
    main()
