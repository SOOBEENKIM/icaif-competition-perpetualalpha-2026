"""Risk-scale replays and episode sensitivity of frozen decision-gate results."""
import copy
import gzip
import json
import time
from pathlib import Path

import numpy as np

from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.complementary_alpha_20261017.research import protected
from experiments.conditional_completion_pairs_20261017.research import verify
from experiments.decision_gate_pairs_20261017.portfolio import replay
from experiments.hourly_pairs_robustness_20261017.analysis import (
    load, summary, episode_paths, paired_resampling,
)
from experiments.hourly_refresh_pairs_20261017.research import control_directory
from experiments.par_direct_forecast_20261017.data import ROOT, save, sha, ts
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution

HERE = Path(__file__).resolve().parent
OUT = HERE / 'artifacts'
GATE = HERE.parent / 'decision_gate_pairs_20261017/artifacts'
CASH = HERE.parent / 'conditional_completion_pairs_20261017/artifacts'


def main():
    OUT.mkdir(exist_ok=False)
    (OUT / 'runs').mkdir()
    spec = json.loads((HERE / 'protocol.json').read_text())
    tests = (HERE / 'test_results.txt').read_text()
    assert 'Ran 4 tests' in tests and tests.rstrip().endswith('OK')
    cfg_path = HERE.parent / 'hourly_refresh_pairs_20261017/reference_protocol.json'
    cfg = json.loads(cfg_path.read_text())
    cfg['conditional_completion'] = {'risk_aversion': 0.}
    candidate, retained = spec['candidate'], spec['retained']
    arms = [*spec['baselines'], retained, candidate]
    runpaths = {}
    for phase in spec['phases']:
        for sc in spec['scenarios']:
            for arm in arms:
                directory = GATE if arm == candidate else CASH if arm == retained else control_directory(arm)
                runpaths[phase, sc, arm] = directory / 'runs' / f'reference-{phase}-{arm}-{sc}.json.gz'
    previous = json.loads((GATE / 'frozen.json').read_text())['sources']
    paths = [HERE / 'research.py', HERE / 'protocol.json', HERE / 'test_results.txt', cfg_path,
             GATE / 'frozen.json', GATE / 'results.json', GATE / 'conclusion.json',
             GATE / 'reference_gate_state.npz', *runpaths.values(),
             HERE.parent / 'hourly_pairs_robustness_20261017/analysis.py',
             HERE.parent / 'hourly_pairs_robustness_20261017/test_analysis.py',
             HERE.parent / 'risk_neutral_pairs_20261017/attribution.py']
    sources = {**previous, **{str(p.relative_to(ROOT)): sha(p) for p in paths}}
    prospective_path = HERE.parent / 'published_pair_lifecycle_20261017/artifacts/plan.json'
    prospective = json.loads(prospective_path.read_text())['sources']
    sources[str(prospective_path.relative_to(ROOT))] = sha(prospective_path)
    verify(sources)
    verify(prospective)
    save(OUT / 'frozen.json', dict(at_unix=time.time(), sources=sources,
         protected=protected(), prospective_sources=len(prospective),
         previous_goal_turn=spec['previous_goal_turn'], historical_independence=False))
    d, _, p = inputs(cfg)
    with np.load(GATE / 'reference_gate_state.npz') as z:
        state = {k: z[k].copy() for k in z.files}
    original = {key: load(path) for key, path in runpaths.items()}
    count = 0
    controls = []

    def simulate(arm, scale, phase, sc, label):
        nonlocal count
        data = dict(d)
        if arm in spec['baselines']:
            key = 'first' if arm == 'first_strategy' else 'second'
            data[key] = d[key] * scale
        else:
            assert scale == 1.
        # cash_direct is a control without the gate keys, exactly as its saved run.
        de = state if arm != retained else {k: v for k, v in state.items() if k not in ('economic_action', 'economic_available')}
        start = cfg['validation_start'] if phase == 'validation' else cfg['comparison_start']
        end = cfg['comparison_start'] if phase == 'validation' else cfg['end']
        run = replay(data, de, None, cfg, p, arm, sc, ts(start), ts(end))
        count += 1
        with gzip.open(OUT / 'runs' / f'{label}.json.gz', 'wt') as f:
            json.dump(run, f, allow_nan=False, default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist())
        print('RUN', label, 'scale', scale, 'PnL', round(run['metrics']['net_pnl'], 4),
              'sigma', round(run['metrics']['volatility_horizon'], 8), flush=True)
        return run

    replayed = {}
    for (phase, sc, arm), old in original.items():
        run = simulate(arm, 1., phase, sc, f'control-{phase}-{arm}-{sc}')
        for key in ('events', 'fills', 'metrics', 'snapshots', 'daily', 'owner_fills'):
            assert run[key] == old[key], (phase, sc, arm, key)
        replayed[phase, sc, arm] = run
        controls.append(dict(phase=phase, scenario=sc, arm=arm, exact=True))

    target = original['validation', 'base', candidate]['metrics']['volatility_horizon']
    fit, fitted_runs = [], {}
    for arm in spec['baselines']:
        control = replayed['validation', 'base', arm]
        cache = {1.: (control, f'control-validation-{arm}-base')}

        def evaluate(scale):
            if scale not in cache:
                label = f'fit-{arm}-{len(cache):02d}'
                cache[scale] = (simulate(arm, scale, 'validation', 'base', label), label)

        def closest():
            return min(cache, key=lambda scale: (abs(cache[scale][0]['metrics']['volatility_horizon'] - target), scale))

        for scale in spec['scale_fit']['grid']:
            evaluate(scale)
        for _ in range(spec['scale_fit']['refinement_rounds']):
            best = closest()
            keys = sorted(cache)
            i = keys.index(best)
            additions = []
            if i:
                additions.append((keys[i - 1] + best) / 2)
            if i < len(keys) - 1:
                additions.append((keys[i + 1] + best) / 2)
            for scale in additions:
                evaluate(scale)
        best = closest()
        run, label = cache[best]
        achieved = run['metrics']['volatility_horizon']
        fit.append(dict(arm=arm, scale=best, target=target, achieved=achieved,
                        relative_error=achieved / target - 1, path=label,
                        trace=[dict(scale=s, sigma=r['metrics']['volatility_horizon'], path=lab)
                               for s, (r, lab) in cache.items()]))
        fitted_runs['validation', 'base', arm] = run
    save(OUT / 'calibration.json', dict(fit=fit, fit_end=cfg['comparison_start'],
         frozen_before_stress_and_later_replays=True, criterion='August sigma distance only',
         historical_independence=False))
    calibration_hash = sha(OUT / 'calibration.json')
    for f in fit:
        arm, scale = f['arm'], f['scale']
        for phase, sc in [('validation', 'stress'), ('comparison', 'base'), ('comparison', 'stress')]:
            fitted_runs[phase, sc, arm] = simulate(arm, scale, phase, sc, f'scaled-{phase}-{arm}-{sc}')
    assert sha(OUT / 'calibration.json') == calibration_hash
    rows, episodes = [], []
    for phase in spec['phases']:
        for sc in spec['scenarios']:
            run = original[phase, sc, candidate]
            bases = {arm: original[phase, sc, arm] for arm in [*spec['baselines'], retained]}
            bases.update({arm + '_august_vol_matched': fitted_runs[phase, sc, arm] for arm in spec['baselines']})
            days = [s['time'] for s in run['daily']]
            for source in [run, *bases.values()]:
                assert [s['time'] for s in source['daily']] == days
                calc = summary([s['return_'] for s in source['daily']])
                for key, value in calc.items():
                    assert abs(value - source['metrics'][key]) < 1e-10, key
            comparison = []
            for arm, base in bases.items():
                b, m = base['metrics'], run['metrics']
                comparison.append(dict(arm=arm, metrics=b, candidate_pnl_advantage=m['net_pnl'] - b['net_pnl'],
                    candidate_q_advantage=m['financial_score'] - b['financial_score'],
                    candidate_sigma_difference=m['volatility_horizon'] - b['volatility_horizon']))
            rows.append(dict(phase=phase, scenario=sc, candidate=run['metrics'], comparisons=comparison,
                paired_resampling=paired_resampling([s['return_'] for s in run['daily']],
                  {arm: [s['return_'] for s in base['daily']] for arm, base in bases.items()}, spec['daily_resampling'])))
            for arm in [candidate, retained]:
                source = original[phase, sc, arm]
                entries, cumulative, components = episode_paths(source, d, cfg['scenarios'][sc])
                attribution = episode_attribution(source, d, cfg['scenarios'][sc])
                details = attribution['episodes']
                np.testing.assert_allclose(cumulative[:, -1], [e['net_pnl'] for e in details], atol=1e-7, rtol=0)
                best = int(np.argmax(cumulative[:, -1])) if entries else None
                stripped = np.array([s['equity'] for s in source['snapshots']])
                if best is not None:
                    stripped = stripped - cumulative[best]
                daily_mask = np.array([s['time'] % 86400 == 43200 for s in source['snapshots']])
                stripped_daily = stripped[daily_mask]
                sensitivity = summary(stripped_daily[1:] / stripped_daily[:-1] - 1)
                sensitivity['net_pnl'] = float(stripped[-1] - stripped[0])
                gains = sum(max(e['net_pnl'], 0.) for e in details)
                episodes.append(dict(phase=phase, scenario=sc, arm=arm, episodes=details,
                    best_episode_index=best,
                    best_episode_share_of_positive_pair_pnl=max(details[best]['net_pnl'], 0.) / gains if gains else None,
                    without_best_episode=sensitivity,
                    accounting_sensitivity_only=True))
    verify(sources)
    verify(prospective)
    save(OUT / 'results.json', dict(rows=rows, episodes=episodes, fit=fit,
         exact_controls=controls, actual_account_replays=count, tests_passed=4,
         protected=protected(), prospective_sources=len(prospective), historical_independence=False,
         orders_sent=0, live_strategy_changed=False, reserved_raw_read=False,
         frozen_candidate_and_selection_unchanged=True))
    save(HERE / 'work_state.json', dict(status='progress_decision_gate_robustness_completed',
         actual_account_replays=count, exact_controls=len(controls), tests_passed=4,
         goal_complete=False, no_progress_streak=0, orders_sent=0, live_strategy_changed=False,
         reserved_raw_read=False, historical_independence=False))
    print('DONE', count, 'actual replays', len(controls), 'exact controls', flush=True)


if __name__ == '__main__':
    main()
