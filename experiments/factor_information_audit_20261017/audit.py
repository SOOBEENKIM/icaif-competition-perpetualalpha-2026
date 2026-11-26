"""Reproducible information/reparameterization audit, not a trading strategy."""
import hashlib
import html
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.dense_history_pairs_20261017.model import design

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'artifacts'
RAW = ROOT / 'experiments/raw_return_pairs_20261017/artifacts'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()) + '\n')


def verify(sources):
    for path, digest in sources.items():
        assert sha(ROOT / path) == digest, path


def ridge_identity():
    rng = np.random.default_rng(20261017)
    x = np.c_[np.ones(400), rng.normal(size=(400, 8))]
    w = np.exp2(-np.arange(400)[::-1] / 120)
    w /= w.sum()
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 12 / len(x)
    operator = np.linalg.solve(x.T @ (w[:, None] * x) + penalty, x.T * w)
    market = rng.normal(size=len(x))
    target = 0.4 * market + x[:, 2] + rng.normal(size=len(x))
    loading = 0.4
    direct = operator @ target
    decomposed = loading * (operator @ market) + operator @ (target - loading * market)
    np.testing.assert_allclose(direct, decomposed, atol=1e-14, rtol=1e-13)
    probe = np.c_[np.ones(100), rng.normal(size=(100, 8))]
    error = float(np.max(abs(probe @ direct - probe @ decomposed)))
    return dict(max_prediction_error=error, passed=True,
                conditions='Identical linear operator and constant loading; no separate clipping. Variable-loading or different-feature models need a separate empirical test.')


def reconstruction(ref, symbols, pairs):
    f = design(ref['market_times'], ref['close'], ref['volume'], ref['trades'], symbols, pairs)
    logs = pd.DataFrame(np.log(ref['close']))
    sigma = logs.diff().rolling(168).std(ddof=0).to_numpy()
    rows = []
    start = int(np.datetime64('2026-08-01T12:00:00', 's').astype('int64'))
    end = int(np.datetime64('2026-10-15T12:00:00', 's').astype('int64'))
    for k, (a, b) in enumerate(f['pairs']):
        beta = f['beta'][:, k]
        for horizon in (1, 6, 24, 72, 168, 336):
            changes = logs.diff(horizon).to_numpy()
            spread_scale = f['vol'][:, k] * np.sqrt(horizon)
            common_scale = np.maximum((sigma[:, a] + sigma[:, b]) * np.sqrt(horizon), 1e-8)
            s = (changes[:, a] - beta * changes[:, b]) / (1 + beta)
            c = changes[:, a] + changes[:, b]
            valid = (ref['market_times'] >= start) & (ref['market_times'] < end)
            valid &= np.isfinite(f['x'][:, k]).all(1) & f['fresh'][:, k]
            recovered = s + beta / (1 + beta) * c
            np.testing.assert_allclose(recovered[valid], changes[valid, a], atol=1e-14, rtol=1e-12)
            si = f['names'].index(f'spread_return_{horizon}')
            ci = f['names'].index(f'common_return_{horizon}')
            clipped_recovery = f['x'][:, k, si] * spread_scale + beta / (1 + beta) * f['x'][:, k, ci] * common_scale
            clipped = (abs(s / spread_scale) > 8) | (abs(c / common_scale) > 8)
            rows.append(dict(pair=pairs[k], horizon=horizon, source_rows=int(valid.sum()),
                clipped_rows=int((valid & clipped).sum()),
                max_unclipped_reconstruction_error=float(np.max(abs(recovered[valid] - changes[valid, a]))),
                max_clipped_reconstruction_error_bps=float(np.max(abs(clipped_recovery[valid] - changes[valid, a]))) * 1e4))
    cut = len(ref['close']) - 48
    changed = {k: v.copy() for k, v in ref.items()}
    changed['close'][cut:] *= np.linspace(1.1, 1.8, ref['close'].shape[1])
    altered = design(changed['market_times'], changed['close'], changed['volume'], changed['trades'], symbols, pairs)
    for key in ('x', 'beta', 'vol', 'fresh'):
        np.testing.assert_array_equal(f[key][:cut], altered[key][:cut])
    return dict(rows=rows, future_perturbation_passed=True,
                meaning='No additional raw market source is introduced by explicit BTC features. Source-dependent scales/loading, clipping, and different feature sets mean this is not proof that normalized ridge models are prediction-equivalent.')


def main():
    OUT.mkdir(exist_ok=False)
    main_sources = json.loads((RAW / 'frozen.json').read_text())['sources']
    prospective = json.loads((ROOT / 'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources']
    protected = json.loads((ROOT / 'experiments/cost_horizon_20261016/artifacts/protocol_frozen.json').read_text())
    sources = dict(main_sources)
    sources.update(prospective)
    for group in ('active_protected_sources', 'reserved_dependencies'):
        sources.update(protected[group])
    paths = [Path(__file__), HERE / 'protocol.json', RAW / 'results.json', RAW / 'diagnostics.json',
             RAW / 'conclusion.json', ROOT / 'experiments/joint_leg_pairs_20261017/artifacts/results.json']
    sources.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    verify(sources)
    save(OUT / 'frozen.json', dict(at_unix=time.time(), sources=sources, reserved_raw_read=False))
    with np.load(ROOT / 'experiments/risk_preserving_20261016/artifacts/inputs.npz') as z:
        ref = {k: z[k] for k in ['market_times', 'close', 'volume', 'trades']}
    symbols = ['BTC', 'ETH', 'HYPE', 'SP500', 'EWY', 'EWJ', 'BRENT', 'GOLD', 'USDEUR', 'USDJPY']
    cfg = json.loads((ROOT / 'experiments/hourly_refresh_pairs_20261017/reference_protocol.json').read_text())
    identities = ridge_identity()
    feature_audit = reconstruction(ref, symbols, cfg['dense_pairs'])
    diagnostics = json.loads((RAW / 'diagnostics.json').read_text())
    cost_rows = []
    for r in diagnostics['account_attribution']:
        if r['phase'] != 'comparison':
            continue
        totals = r['totals']
        calculated = totals['price_move'] - totals['execution_cost'] - totals['fees'] + totals['funding']
        assert abs(calculated - totals['net_pnl']) < 1e-6
        cost_rows.append(dict(model=r['model'], scenario=r['scenario'], **totals,
                             base_owned_pnl=r['base_owned_pnl']))
    rows = json.loads((RAW / 'results.json').read_text())['rows']
    rows += [r for r in json.loads(paths[-1].read_text())['rows'] if r['arm'] == 'first_dense_hourlyhedge_joint_24']
    comparison = [r for r in rows if r['venue'] == 'reference' and r['phase'] == 'comparison']
    result = dict(at_unix=time.time(), fixed_loading_ridge_identity=identities, source_feature_audit=feature_audit,
        forecast_checks=[r for r in diagnostics['prediction_checks'] if r['phase'] == 'comparison'],
        cost_attribution=cost_rows, reference_comparison=comparison,
        conclusion='No stable replacement established. Forecast skill and after-cost entry selection are the outstanding weaknesses. Factor decomposition is an interpretation/representation hypothesis, not new raw information or established alpha. Do not run a large model grid merely by relabeling existing signals.',
        next_test='One shared BTC predictor with BTC-only features plus source-loading residual predictor may differ through representation/parameter sharing. First verify mathematical target identity, source-only loadings, strict label maturity, and later-period forecast/tail calibration. Preserve original hedge and execution. No presumption of improved PnL.',
        research_sources=[dict(url='https://arxiv.org/html/2510.11616v1', status='Read abstract, method and empirical setup. 500 U.S. equities, firm characteristics, cost-aware joint learning; not a crypto-par replication or evidence of our profitability.'),
                          dict(url='https://arxiv.org/abs/2412.11432', status='WITHDRAWN due to universe-selection error; excluded as positive performance evidence.')],
        source_files_verified=len(sources), adopted=False, orders_sent=0, live_strategy_changed=False,
        reserved_raw_read=False, historical_independence=False)
    verify(sources)
    save(OUT / 'results.json', result)
    report = ROOT / 'reports/factor_information_audit_20261017'
    report.mkdir(exist_ok=True)
    table = ''
    for r in comparison:
        m = r['metrics']
        table += f"<tr><td>{html.escape(r['arm'])}</td><td>{r['scenario']}</td><td>{m['net_pnl']:+,.2f}</td><td>{m['financial_score']:.6f}</td><td>{m['completed_pairs']}</td></tr>"
    page = '''<!doctype html><meta charset="utf-8"><title>개선 정체 원인 점검</title><style>body{font:16px system-ui;max-width:1100px;margin:35px auto;line-height:1.7}table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #bbb;padding:8px}pre{white-space:pre-wrap;background:#f5f5f5;padding:16px}</style>
<h1>개선 정체 원인 점검</h1><p>개선 후보는 구현했지만 비용 차감 후 안정적인 교체 우위를 입증하지 못했다. 아래는 초기 자본 1M, 기준시장 2026-09-01 12UTC부터 10-15 12UTC까지의 재사용 과거 자료 모의 계좌다. 대회 지갑의 실제 손익이나 독립 검증이 아니다.</p>
<table><tr><th>전략</th><th>비용 조건</th><th>전체 계좌 순손익</th><th>Q</th><th>완료 페어 수</th></tr>''' + table + '''</table>
<h2>막힌 지점</h2><p>최근 raw 후보의 24시간 예측 오차는 81.2620bp로 변화 없음 기준 80.8203bp보다 크다. 전체 오차만으로 모든 조건부 수익 기회를 부정할 수는 없지만, 예측력 개선의 증거도 아니다. 페어 부분 가격 손익 +3441.9475에서 체결 비용 2429.2634, 수수료 2171.5450, 펀딩 지급 170.2080을 빼면 -1329.0688이다. 이는 실제 거래소 체결이 아닌 기준시장 시뮬레이션이다.</p>
<p>기준 연구 후보가 첫째·둘째보다 높은 구간은 있으나 비용 스트레스에서 완료 페어가 하나이며, 기존 견고성 점검에서 특정 수익 거래의 영향이 컸다. 대회 par 자료에서는 완료 페어를 확보하지 못했다. 반복 사용한 기간을 새 독립 검증으로 부를 수 없다.</p>
<h2>다음 모델을 추가하기 전 중복 점검</h2><p>같은 입력·표본·가중치·릿지 규제와 고정된 시장 노출을 쓰면, 시장과 잔차를 각각 회귀한 뒤 합친 예측은 원래 목표를 직접 회귀한 예측과 같다. 수치 검사를 통과했다. 기존 특징의 비정규화된 공통 움직임과 가격차로 각 자산의 과거 수익률을 복원할 수 있다. 다만 현재 회귀에는 변동성 정규화·클리핑이 있으므로 실제 모델 전체가 동일하다는 뜻은 아니다. 별도 BTC 입력·공유 모델·시간에 따라 바뀌는 노출은 다른 가설이며 검증 대상이다.</p>
<p>즉, 시장/잔차 분해는 원인 분석에 유용하지만 그 자체로 새로운 수익 신호가 되지는 않는다. 예측과 비용 후 거래 선택이 나아지는지 확인하기 전에 모델 이름이나 실행 변형만 늘리는 것은 적절하지 않다.</p>
<h2>논문과 적용 범위</h2><p><a href="https://arxiv.org/html/2510.11616v1">Attention Factors for Statistical Arbitrage (ICAIF 2025)</a>는 대규모 미국 주식의 기업 특성과 잔차 신호 및 비용 후 거래 목표를 함께 학습한다. 우리 BTC 요인 릿지는 해당 논문의 재현이 아니며 데이터·시장·체결 조건이 다르다. <a href="https://arxiv.org/abs/2412.11432">factor-residual 재현 문서</a>는 종목 집합 선택 오류로 철회되었으므로 성과 근거에서 제외했다.</p>
<p>이번 작업은 진단과 중복 검증이다. 새 거래 모델 성능 개선, 신규 주문 또는 실거래 전략 교체를 주장하지 않는다.</p>'''
    page += '<h2>재현 검사</h2><pre>' + html.escape(json.dumps(dict(identity=identities, features=feature_audit, verified_files=len(sources)), ensure_ascii=False, indent=2)) + '</pre>'
    (report / 'report.html').write_text(page)
    save(HERE / 'work_state.json', dict(status='progress_factor_information_audit_completed', report=str((report / 'report.html').relative_to(ROOT)),
        performance_improvement_established=False, goal_complete=False, no_progress_streak=0,
        orders_sent=0, live_strategy_changed=False, reserved_raw_read=False))
    print(json.dumps(dict(status='DONE', files_verified=len(sources), identity=identities,
        source_reconstructions=len(feature_audit['rows']), future_perturbation=True,
        comparison_rows=len(comparison), report=str(report / 'report.html'))), flush=True)


if __name__ == '__main__':
    main()
