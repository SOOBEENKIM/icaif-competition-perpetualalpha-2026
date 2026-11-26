"""Read-only strategy audit. Writes only beside this script; never sends orders.

Run from the repository root with .venv/bin/python reports/strategy_rigor_audit_20261023/audit.py.
The synthetic case tests an optimization claim, not market profitability.
"""
import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# Match the live producer's import order before the frozen bundle is registered.
from pas_economic_allocation.model import features, predict, rows, HORIZON
from pas_economic_allocation.policy import utility
from pas_performance_research.selected import controller
from competition_live.profile import load_profile, verify_deployment
from competition_live.runtime import private_load
from candidate_handoff.entry_policy import configured_policy


def read(relative):
    return json.loads((ROOT / relative).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False) + '\n')


def synthetic_case():
    n = 10
    protocol = load_profile()['trading']
    mu = np.zeros(n)
    mu[0] = .025
    # A PSD covariance compatible with 75% diagonal shrinkage. This is NOT
    # an estimate from market outcomes; both final position costs are nonzero.
    raw_sigma = np.eye(n) * 1e-8
    raw_sigma[:2, :2] = .0004 * np.array([[1., .96], [.96, 1.]])
    sigma = .25 * raw_sigma + .75 * np.diag(np.diag(raw_sigma))
    history = np.full((3, n), 99.)
    history[-1] = 100.
    c = controller([0], mu[None], np.zeros((1, n)), np.zeros((1, n)),
                   np.zeros((1, n), bool), price_times=np.array([-86400, -10800, 0]),
                   prices=history, age=np.zeros((3, n)))
    prices = np.full(n, 100.)
    equity = 50000.
    result = c.propose(0, prices, np.zeros(n), equity, np.zeros(n), sigma,
                       np.ones(n), np.ones(n), protocol)
    pre = np.array(c.trace[-1]['pre_entry_filter_qty']) * prices / equity
    post = np.array(result['target_qty']) * prices / equity
    args = (np.zeros(n), mu, 168 * sigma, 5., .0019, np.full(n, .0024))
    before, after = utility(pre, *args), utility(post, *args)
    assert before > 0 and after < 0
    assert pre[0] > 0 and pre[1] < 0 and post[0] == 0 and post[1] < 0
    assert abs(post[1]) * equity > protocol['execution']['minimum_order_value']
    assert np.linalg.eigvalsh(sigma).min() > 0
    # The first forecast is within the production clipping/attenuation range.
    fit = read('reports/pas_economic_allocation_20261022/fit.json')
    assert mu[0] < 2 * fit['attenuation'] * np.sqrt(168 * sigma[0, 0])
    return dict(kind='synthetic property counterexample; not historical/live PnL',
                claim_refuted='Final filtered target always has nonnegative incremental forecast utility',
                expected_returns=mu.tolist(), covariance=sigma.tolist(),
                covariance_diagonal_shrinkage=.75,
                prior_weights=np.zeros(n).tolist(), before_filter_weights=pre.tolist(),
                final_weights=post.tolist(), before_filter_utility=before,
                final_utility=after, hold_utility=0., equity=equity,
                trace=c.trace[-1], scope='Controller target only; no exchange or live scheduler call')


def calibration_audit():
    p = read('reports/pas_economic_allocation_20261022/protocol.json')
    fit = read('reports/pas_economic_allocation_20261022/fit.json')
    with np.load(ROOT / p['pointer']['inputs'], allow_pickle=False) as z:
        d = {k: z[k] for k in ['market_times', 'close', 'volume', 'trades', 'funding', 'observed']}
    mask = d['market_times'] < p['calibration_end']
    cal = {k: v[mask] for k, v in d.items()}
    x, unit, _ = features(cal)
    i, j, y = rows(cal, x, unit, p['calibration_start'], p['calibration_end'])
    raw = predict(x, unit, fit)[i, j] / unit[i, j]
    attenuated = raw * fit['attenuation']
    assert len(y) == fit['calibration_rows']
    assert np.isclose(np.mean(abs(raw-y)), fit['calibration_mae'])
    assert fit['last_training_maturity'] < p['train_end']
    assert fit['last_calibration_maturity'] < p['calibration_end']
    # Confirm that current feature/prediction definitions do not read suffix rows.
    k = len(cal['market_times']) - 48
    xp, up, _ = features({name: value[:k] for name, value in cal.items()})
    np.testing.assert_allclose(xp, x[:k], equal_nan=True, rtol=0, atol=1e-12)
    np.testing.assert_allclose(predict(xp, up, fit), predict(x, unit, fit)[:k],
                               equal_nan=True, rtol=0, atol=1e-12)
    return dict(training_rows=fit['rows'], calibration_rows=len(y),
                distinct_calibration_origins=len(np.unique(cal['market_times'][i])),
                attenuation=fit['attenuation'], raw_normalized_mae=float(np.mean(abs(raw-y))),
                deployed_attenuated_normalized_mae=float(np.mean(abs(attenuated-y))),
                zero_normalized_mae=float(np.mean(abs(y))),
                strict_maturity_checks=True, prediction_prefix_check=True,
                interpretation='Calibration reused to estimate attenuation; not a fresh test. Rows share dates across assets.',
                boundaries={k: datetime.fromtimestamp(p[k], timezone.utc).isoformat()
                            for k in ['train_start', 'train_end', 'calibration_start',
                                      'calibration_end', 'selection_start', 'selection_end']})


def main():
    runtime = ROOT / '.runtime/pas_budget_lifecycle'
    policy = read('.runtime/pas_budget_lifecycle/execution_policy.json')
    before = {f: sha(ROOT / f) for f in policy['source_sha256']}
    assert configured_policy(runtime) == 'pullback_any3'
    verify_deployment()
    active = read('.runtime/active_pas_strategy.json')
    status = read('.runtime/pas_budget_lifecycle/status.json')
    portfolio = private_load(runtime / 'coordinator.pkl')['portfolio']
    ownership = {}
    for name, ledger in portfolio.account.owners.items():
        fills = ledger.fills
        ownership[name] = dict(qty=ledger.qty.tolist(), fills=len(fills) if isinstance(fills, list) else int(fills),
                               realized=float(ledger.realized), fees=float(ledger.fees),
                               funding=float(ledger.funding))
    selected = read('reports/pas_performance_research_20261022/selected.json')
    ref = read('reports/pas_performance_research_20261022/all_reference_results.json')
    pas = read('reports/pas_performance_research_20261022/all_pas_results.json')
    keep = {'all_market_signed_core_v1', 'pullback_any3'}
    ref_rows = [r for r in ref if r['wave'] == 'wave7' and r['arm'] in keep and r['phase'] == 'continuous']
    pas_rows = [r for r in pas if r['wave'] == 'wave7' and r['arm'] in keep and r.get('phase') == 'continuous71h']
    for r in ref_rows + pas_rows:
        expected = (max(r['return_fraction'], 0.) + .005) / (abs(r['return_fraction']) + 1.5*r['volatility'] + .05)
        assert np.isclose(expected, r['Q'])
    save('synthetic_counterexample.json', synthetic_case())
    save('calibration_audit.json', calibration_audit())
    save('evidence.json', dict(
        audited_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Read-only code and evidence review; no refit, no backtest selection, no strategy switch, no exchange mutation.',
        active={k: active[k] for k in ['core_revision', 'execution_policy', 'execution_revision', 'team_capital', 'our_capital']},
        status_at_utc=datetime.fromtimestamp(status['at'], timezone.utc).isoformat(),
        ownership=ownership, model_paths=dict(weekly='reports/pas_economic_allocation_20261022/fit.json',
            regime='replay_lab/output/tail_calibration/20261014-v1/regime_fit.json',
            pair='deliverables/sfinx_pairs_candidate_20261018_final/models'),
        compared_to='all_market_signed_core_v1 only; not all past strategies',
        research_reference=ref_rows, research_pas=pas_rows,
        selection=dict(waves=len(selected['catalog']),
                       distinct_arm_names=len({a for w in selected['catalog'] for a in w['arms']}),
                       wave_arm_combinations=sum(len(w['arms']) for w in selected['catalog']),
                       reference_rows=len(ref), pas_rows=len(pas), independent_test=selected['independent_test']),
        uncertainty=read('reports/pas_performance_research_20261022/uncertainty.json'),
        audited_source_hashes=before,
        imported_controller=inspect.getfile(controller), imported_utility=inspect.getfile(utility)))
    assert before == {f: sha(ROOT / f) for f in before}
    assert configured_policy(runtime) == 'pullback_any3'
    verify_deployment()
    save('verification.json', dict(live_source_hashes_unchanged=True,
          verified_source_count=len(before), deployment_manifest_verified=True,
          synthetic_counterexample_reproduced=True, calibration_metrics_recomputed=True,
          score_arithmetic_checked=True, orders_sent_by_audit=0))
    print(json.dumps(read(str((OUT / 'verification.json').relative_to(ROOT))), ensure_ascii=False))


if __name__ == '__main__':
    main()
