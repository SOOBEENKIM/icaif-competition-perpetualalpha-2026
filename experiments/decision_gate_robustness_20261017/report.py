"""Report fixed risk-scale controls without changing candidate selection."""
import csv
import html
import json
import time
from pathlib import Path

from experiments.complementary_alpha_20261017.research import protected
from experiments.conditional_completion_pairs_20261017.research import verify
from experiments.par_direct_forecast_20261017.data import ROOT, save, sha

HERE = Path(__file__).resolve().parent
OUT = HERE / 'artifacts'
DEST = ROOT / 'reports/decision_gate_robustness_20261017'


def main():
    result = json.loads((OUT / 'results.json').read_text())
    spec = json.loads((HERE / 'protocol.json').read_text())
    verify(json.loads((OUT / 'frozen.json').read_text())['sources'])
    prospective = json.loads((ROOT / 'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources']
    verify(prospective)
    assert result['actual_account_replays'] == 46
    assert len(result['exact_controls']) == 16
    assert all(r['exact'] for r in result['exact_controls'])
    labels = {'first_strategy': '기존 1번', 'second_strategy': '기존 2번',
              spec['retained']: '개선 페어 cash_direct', spec['candidate']: '진입 필터 decisiongate',
              'first_strategy_august_vol_matched': '기존 1번 × 1.625',
              'second_strategy_august_vol_matched': '기존 2번 × 1.9375'}
    matched = [c for r in result['rows'] for c in r['comparisons'] if c['arm'].endswith('_august_vol_matched')]
    all_above = all(c['candidate_pnl_advantage'] > 0 and c['candidate_q_advantage'] > 0 for c in matched)
    later = next(r for r in result['rows'] if (r['phase'], r['scenario']) == ('comparison', 'base'))
    ep = next(r for r in result['episodes'] if (r['phase'], r['scenario'], r['arm']) == ('comparison', 'base', spec['candidate']))
    aug = next(r for r in result['rows'] if (r['phase'], r['scenario']) == ('validation', 'base'))
    aug_second = next(r for r in aug['comparisons'] if r['arm'] == 'second_strategy_august_vol_matched')
    next_action = (
        'Retain decisiongate and cash_direct as improved reference controls. The later-period gain survives '
        'actual native risk-scale replays, while August advantage over the scaled second strategy is small. '
        'Study causal holding/renewal decisions next: train on strictly matured remaining-horizon, fixed-current-hedge '
        'cash opportunities with incremental exit/renewal costs, rather than reuse a 24-hour entry decision score '
        'as expected profit. Freeze the new holding rule before later replay, retain exact entry/execution controls, '
        'and evaluate both reference and the permitted historical par data. Do not filter specific losing dates '
        'or redefine the observed improvement as absent. No automatic live replacement.'
    )
    conclusion = dict(at_unix=time.time(), classification='progress_risk_scaled_improvement_and_concentration_measured',
        prior_turn='no_progress_status_restatement_revalidated',
        candidate=spec['candidate'], observed_reference_improvement_preserved=True,
        above_scaled_native_pnl_and_q_all_eight_comparisons=all_above,
        calibration=result['fit'], calibration_relative_error_max=max(abs(f['relative_error']) for f in result['fit']),
        later_base_candidate=later['candidate'],
        later_base_scaled_controls=[r for r in later['comparisons'] if r['arm'].endswith('_august_vol_matched')],
        august_advantage_over_scaled_second=aug_second['candidate_pnl_advantage'],
        later_best_pair_share_of_positive_pair_pnl=ep['best_episode_share_of_positive_pair_pnl'],
        later_account_without_best_episode=ep['without_best_episode'],
        limitation='Risk matching is approximate, fitted only on August base volatility. Same reused historical '
                   'periods, no independent future validation; later daily block sensitivity includes negative '
                   'PnL/Q differences. Episode subtraction is an accounting sensitivity, not an executable '
                   'counterfactual. No new par profitability evidence or submission ban.',
        actual_account_replays=46, exact_controls=16, tests_passed=4,
        protected=protected(), prospective_sources=len(prospective),
        next_action=next_action, orders_sent=0, live_strategy_changed=False, reserved_raw_read=False,
        historical_independence=False, goal_complete=False, no_progress_streak=0)
    save(OUT / 'conclusion.json', conclusion)
    DEST.mkdir(exist_ok=False)
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>페어 개선판: 위험 규모와 수익 집중도 검증</title>
<style>body{font:15px system-ui;max-width:1200px;margin:32px auto;padding:0 24px;line-height:1.65;color:#172b3a}table{border-collapse:collapse;font-size:13px;width:100%;margin-bottom:24px}td,th{border:1px solid #ccd4db;padding:7px;text-align:right}th:first-child,td:first-child{text-align:left}h1,h2{line-height:1.3}.note{padding:14px;background:#eef4f8}pre{white-space:pre-wrap}</style>
<h1>페어 개선판: 위험 규모와 수익 집중도 검증</h1>
<p class="note">기존 전략의 규모를 실제로 늘려도 개선 후보의 비용 차감 후 PnL·Q 우위는 유지됐다. 단, 8월에는 규모를 맞춘 기존 2번과 차이가 작고, 이후 성과는 소수 수익 거래에 집중된다. 과거 백테스트 개선과 반복 가능한 실거래 수익성은 별개다.</p>
<p>초기자본 1M. 기간은 8월 1일 12UTC~9월 1일 12UTC, 9월 1일 12UTC~10월 15일 12UTC. 이미 반복 사용한 외부 기준시장 자료이며 독립적인 미래 검증이나 실제 지갑 실적이 아니다. 기존 전략의 원래 목표 비중을 조정하여 주문부터 다시 계산했다. 수수료, 펀딩, 수량 반올림, 지연, 거래량 제한, 위험 한도는 그대로다. 저장된 수익을 단순 배율로 늘리지 않았다.</p>
<p>기본 조건: 편도 수수료 9bp, 가격 충격 10bp, 거래량 참여 한도 1%. 불리한 조건: 편도 수수료 12bp, 가격 충격 20bp, 참여 한도 0.25%, 양의 펀딩 수입 50% 인정. 같은 기간이라도 비용 조건에 따라 진입·체결 경로가 달라질 수 있다. 표의 PnL은 기간 말 보유 평가손익을 포함하며 모든 포지션을 실제 청산했다는 의미가 아니다.</p>
<h2>규모 결정</h2><p>8월 기본 조건의 decisiongate 기간 변동성에 가장 가까운 규모를 선택했다. PnL·Q는 선택에 쓰지 않았다. 두 규모를 기록·고정한 뒤 불리한 조건 및 9월 이후를 재실행했다. 원래 과거 결과를 모르는 상태에서 수행한 독립 선정이라는 뜻은 아니다.</p>
<table><tr><th>비교 전략</th><th>목표 규모 배율</th><th>8월 변동성 σ</th><th>개선판 대비 상대 오차</th></tr>'''
    for f in result['fit']:
        page += f"<tr><td>{labels[f['arm']]}</td><td>{f['scale']}</td><td>{100*f['achieved']:.6f}%</td><td>{100*f['relative_error']:+.4f}%</td></tr>"
    page += '</table><p>개선판의 8월 σ는 1.303028%다. 완전히 같은 위험이 아니며, 이후 기간의 위험도 다시 달라진다. σ는 일별 수익률의 표본 표준편차 × √완료 일수이며 연율 변동성이 아니다. Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05), 심사 전 점수는 0.8Q이다.</p>'
    csv_rows = []
    for row in result['rows']:
        period = '8월' if row['phase'] == 'validation' else '9월 1일~10월 15일'
        scenario = '기본' if row['scenario'] == 'base' else '불리한 체결'
        page += f'<h2>{period} · {scenario}</h2><table><tr><th>전략</th><th>순손익</th><th>수익률</th><th>기간 σ</th><th>Q</th><th>최대낙폭</th><th>평균 총노출</th></tr>'
        for arm, m in [(c['arm'], c['metrics']) for c in row['comparisons']] + [(spec['candidate'], row['candidate'])]:
            page += f"<tr><td>{labels[arm]}</td><td>{m['net_pnl']:+,.2f}</td><td>{100*m['roi']:+.6f}%</td><td>{100*m['volatility_horizon']:.6f}%</td><td>{m['financial_score']:.8f}</td><td>{100*m['max_drawdown']:.6f}%</td><td>{100*m['mean_gross']:.4f}%</td></tr>"
            csv_rows.append(dict(phase=row['phase'], scenario=row['scenario'], arm=arm,
                                **{k:m[k] for k in ['net_pnl','roi','volatility_horizon','financial_score','max_drawdown','mean_gross']}))
        page += '</table>'
    page += f'''<p>개선판은 규모 조정 기존 1·2번과의 8개 비교에서 모두 PnL·Q가 높다. 그러나 8월 기본 조건에서 규모 조정 2번 대비 순손익 차이는 {aug_second['candidate_pnl_advantage']:.2f}에 불과하다. 원래 2번과 비교한 큰 격차 전체를 새 모델의 예측력으로 돌릴 수 없다. 9월 이후 기본 조건에서는 규모 조정 1·2번보다 순손익이 각각 +3,172.21·+2,859.54 높다. 개선판의 해당 기간 σ는 두 규모 조정 비교 전략보다 낮지만 최대낙폭은 더 크다. 모든 위험 지표가 좋아졌다는 주장은 아니다.</p>
<h2>거래 집중도</h2><p>가장 크게 번 페어 거래의 가격 손익·체결 비용·수수료·펀딩을 시간별로 분리하여 관측 계좌에서 차감했다. 다른 거래와 자금 배분을 그대로 두는 회계 민감도이며, 그 거래를 실제로 금지했을 때의 전략 성과는 아니다. 양의 페어 손익 비중은 수익을 낸 페어들의 이익 합계가 분모이고, 전체 계좌 이익 비중이 아니다.</p>
<table><tr><th>기간·조건·후보</th><th>가장 큰 수익 거래의 양의 페어 손익 비중</th><th>그 거래 기여 제외 계좌 PnL</th><th>제외 후 Q</th></tr>'''
    for e in result['episodes']:
        share = e['best_episode_share_of_positive_pair_pnl']
        v = e['without_best_episode']
        page += f"<tr><td>{e['phase']} / {e['scenario']} / {labels[e['arm']]}</td><td>{100*share:.2f}%</td><td>{v['net_pnl']:+,.2f}</td><td>{v['financial_score']:.8f}</td></tr>"
    page += '''</table><p>9월 이후 기본 조건에서 가장 큰 수익 페어는 +3,983.03이며 이를 회계적으로 제외하면 decisiongate 계좌는 -130.26이다. 불리한 조건에서는 수익 페어 한 건의 기여 +3,774.54를 제외하면 -468.83이다. 이것은 실제로 관측한 개선을 취소하는 계산이 아니라, 그 개선이 여러 독립적인 성공에 분산되지 않았음을 보여준다.</p>
<h2>일별 경로 민감도</h2><p>2일·7일 원형 블록을 5,000번 재표집했고 모든 전략에 같은 날짜 인덱스를 적용했다. 아래는 9월 이후의 7일 블록 결과다. 2.5~97.5 백분위가 0을 포함하므로 흔들림이 크다. 여러 후보를 시험한 선택 효과를 보정하지 않았고 미래 승률·독립적인 유의성 검정으로 해석할 수 없다. 동일한 스트레스 계좌의 엄격한 초과 비율은 0이며 열등하다는 뜻이 아니다.</p>
<table><tr><th>조건·비교 대상</th><th>수익률 차이 2.5 / 50 / 97.5 백분위 (퍼센트포인트)</th><th>Q 차이 2.5 / 50 / 97.5 백분위</th></tr>'''
    for row in result['rows']:
        if row['phase'] != 'comparison':
            continue
        for r in row['paired_resampling']:
            if r['block_days'] != 7:
                continue
            roi = ' / '.join(f'{100*v:+.5f}' for v in r['return_fraction_difference_quantiles'])
            q = ' / '.join(f'{v:+.5f}' for v in r['financial_score_difference_quantiles'])
            page += f"<tr><td>{row['scenario']} / {labels[r['baseline']]}</td><td>{roi}</td><td>{q}</td></tr>"
    page += '''</table><h2>검증 및 다음 작업</h2><p>계좌 재실행 46개, 원래 계좌 경로 16개 정확 재현, 회계·블록 재표집 검사 4개를 통과했다. 일별 수익률로 R·σ·Q를 독립 재계산했고 두 방식의 거래별 손익 분해를 대조했다. 실제 운용 소스 373개, 예약 검증 의존 소스 345개, 향후 검증 소스 1,045개의 해시를 대조했다. 예약된 미래 원시 자료를 열지 않았고 실제 주문·운용 교체도 하지 않았다.</p>
<p>이번 진단은 기준시장 개선의 규모 효과와 집중도를 확인했다. 실제 par의 수익성에 관한 새 증거는 아니다. 다음 개발에서는 고정 24시간 진입 학습과 실제 보유 연장의 목표 차이를 분리한다. 이미 지급한 진입 비용을 다시 청구하지 않고 현재 헤지·남은 기간·추가 청산/연장 비용으로 보유 판단을 평가한다. 기존 개선 후보는 비교 대상으로 보존한다. 특정 손실 날짜를 사후 제외하거나 작은 표본을 공식 제출 금지 조건으로 바꾸지 않는다.</p>
<p><a href="../decision_gate_pairs_20261017/report.html">원래 진입 필터 비교</a></p></html>'''
    (DEST / 'report.html').write_text(page)
    with (DEST / 'results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    comparison = ROOT / 'experiments/pairs_development_status_20261017/comparisons/decision_gate_robustness.json'
    assert not comparison.exists()
    save(comparison, conclusion)
    save(HERE / 'work_state.json', dict(status=conclusion['classification'], report=str((DEST / 'report.html').relative_to(ROOT)),
         observed_reference_improvement_preserved=True, actual_account_replays=46, exact_controls=16,
         tests_passed=4, next_action=next_action, goal_complete=False, no_progress_streak=0,
         orders_sent=0, live_strategy_changed=False, reserved_raw_read=False, historical_independence=False))
    paths = [Path(__file__), OUT / 'results.json', OUT / 'calibration.json', OUT / 'conclusion.json',
             DEST / 'report.html', DEST / 'results.csv', comparison, *sorted((OUT / 'runs').glob('*.json.gz'))]
    save(OUT / 'conclusion_frozen.json', dict(at_unix=time.time(), sources={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    print('DONE', str(DEST / 'report.html'), 'all scaled comparisons above:', all_above, flush=True)


if __name__ == '__main__':
    main()
