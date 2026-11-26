"""Post-result sensitivity for the preselected direct refit; no reselection."""
import json
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.complementary_alpha_20261017.research import protected
from experiments.hourly_pairs_robustness_20261017.analysis import load, summary, episode_paths, paired_resampling
from experiments.hourly_refresh_pairs_20261017.research import control_directory
from experiments.par_direct_forecast_20261017.data import ROOT, save, sha

HERE = Path(__file__).resolve().parent
ART = HERE/'artifacts'
HOURLY = HERE.parent/'hourly_refresh_pairs_20261017'


def main():
    selection = json.loads((ART/'selection.json').read_text())
    assert selection['selected'] == 'first_dense_hourlyhedge_direct_24'
    selection_hash = sha(ART/'selection.json')
    cfg = json.loads((HOURLY/'reference_protocol.json').read_text())
    d, _, _ = inputs(cfg); rows = []
    attribution = json.loads((ART/'attribution.json').read_text())['rows']
    settings = dict(circular_block_days=[2, 7], samples=5000, seed=20261017, quantiles=[.025, .5, .975])
    for phase in ['validation', 'comparison']:
        for scenario in ['base', 'stress']:
            arm = selection['selected']; run = load(ART/'runs'/f'reference-{phase}-{arm}-{scenario}.json.gz')
            bases = {}
            for base in ['first_strategy', 'second_strategy', 'first_dense_hourlyhedge_bias_24']:
                parent = HOURLY/'artifacts' if '_hourlyhedge_' in base else control_directory(base)
                bases[base] = load(parent/'runs'/f'reference-{phase}-{base}-{scenario}.json.gz')
            days = [s['time'] for s in run['daily']]
            returns = np.array([s['return_'] for s in run['daily']])
            baseline_returns = {}
            for name, b in bases.items():
                assert [s['time'] for s in b['daily']] == days
                baseline_returns[name] = np.array([s['return_'] for s in b['daily']])
            for source in [run, *bases.values()]:
                for key, value in summary([s['return_'] for s in source['daily']]).items():
                    assert abs(value-source['metrics'][key]) < 1e-10
            entries, paths, components = episode_paths(run, d, cfg['scenarios'][scenario])
            old = next(r['episodes'] for r in attribution if (r['phase'], r['arm'], r['scenario']) == (phase, arm, scenario))
            np.testing.assert_allclose(paths[:, -1], [e['net_pnl'] for e in old], atol=1e-7, rtol=0)
            best = int(np.argmax(paths[:, -1])); best_pnl = float(paths[best, -1])
            equity = np.array([s['equity'] for s in run['snapshots']]); stripped = equity-paths[best]
            daily = stripped[[s['time'] % 86400 == 43200 for s in run['snapshots']]]
            without = dict(net_pnl=float(stripped[-1]-stripped[0]), **summary(daily[1:]/daily[:-1]-1))
            risk = []
            for name, br in baseline_returns.items():
                factor = float(np.std(br, ddof=1)/np.std(returns, ddof=1))
                m = summary(returns*factor); b = summary(br)
                risk.append(dict(baseline=name, hindsight_scale=factor,
                    pnl_difference=float((m['return_fraction']-b['return_fraction'])*1e6),
                    q_difference=float(m['financial_score']-b['financial_score'])))
            gains = float(np.maximum(paths[:, -1], 0).sum())
            rows.append(dict(phase=phase, scenario=scenario, metrics=run['metrics'], episodes=old,
                best_episode_pnl=best_pnl, best_share_of_positive_pnl=max(best_pnl, 0)/gains if gains else None,
                account_without_best_episode=without, ex_post_risk_normalized=risk,
                paired_resampling=paired_resampling(returns, baseline_returns, settings)))
            print(phase, scenario, 'best', best_pnl, 'share', rows[-1]['best_share_of_positive_pnl'],
                  'without', without['net_pnl'], 'risk', risk)
    assert sha(ART/'selection.json') == selection_hash
    save(ART/'robustness.json', dict(rows=rows, settings=settings, post_evaluation_only=True,
        actual_account_replays=0, descriptive_not_tradable_counterfactual=True,
        not_selection_adjusted_inference=True, not_probability_of_future_success=True,
        protected=protected(), historical_independence=False, orders_sent=0))


if __name__ == '__main__':
    main()
