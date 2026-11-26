"""Report daily feature refit, including failed gates and concentration."""
import json
from collections import Counter
from pathlib import Path
from experiments.complementary_alpha_20261017.research import protected
from experiments.dense_history_pairs_20261017.report import grid
from experiments.netted_shadow_pairs_20261017.report import document, export_csv
from experiments.par_direct_forecast_20261017.data import ROOT, save, sha

HERE = Path(__file__).resolve().parent
ART = HERE/'artifacts'


def main():
    results = json.loads((ART/'results.json').read_text())
    forecasts = json.loads((ART/'forecast_diagnostics.json').read_text())['rows']
    robustness = json.loads((ART/'robustness.json').read_text())['rows']
    books = json.loads((ART/'books/results.json').read_text())
    for path in [ART/'frozen.json', ART/'books/frozen.json']:
        for rel, digest in json.loads(path.read_text())['sources'].items():
            assert sha(ROOT/rel) == digest, rel
    guards = protected(); rows = results['rows']; assert len(rows) == 208
    lookup = {(r['venue'], r['phase'], r['arm'], r['scenario']): r['metrics'] for r in rows}
    account = []
    arms = [('first_strategy', '기존 첫째'), ('second_strategy', '기존 둘째'),
            ('first_dense_hourlyhedge_bias_24', '첫째＋이전 bias 페어'),
            ('first_dense_hourlyhedge_direct_24', '첫째＋직접 재학습'),
            ('first_dense_hourlyhedge_residual_24', '첫째＋잔차 재학습'),
            ('first_dense_hourlyhedge_scalar_24', '첫째＋일별 평균 보정 대조군')]
    for phase in ['validation', 'comparison']:
        for scenario in ['base', 'stress']:
            for arm, label in arms:
                m = lookup['reference', phase, arm, scenario]
                account.append([phase, scenario, label, f"{m['net_pnl']:+,.2f}",
                    f"{100*m['return_fraction']:+.5f}%", f"{m['financial_score']:.8f}",
                    f"{100*m['volatility_horizon']:.5f}%", f"{100*m['max_drawdown']:.5f}%",
                    m.get('completed_pairs', '—')])
    prediction = [[r['phase'], r['model'], r['samples'], f"{r['normalized_rmse']:.6f}",
                   f"{100*r['sign_accuracy']:.2f}%", f"{r['mean_abs_return_prediction']*10000:.3f}"]
                  for r in forecasts if r['pair'] == ['BTC', 'HYPE'] and r['model'] in ['bias', 'direct', 'residual', 'scalar']]
    concentration = []
    bootstrap = []
    risk = []
    for r in robustness:
        concentration.append([r['phase'], r['scenario'], f"{r['best_episode_pnl']:+,.2f}",
            f"{100*r['best_share_of_positive_pnl']:.2f}%", f"{r['account_without_best_episode']['net_pnl']:+,.2f}"])
        for b in r['paired_resampling']:
            bootstrap.append([r['phase'], r['scenario'], b['baseline'], b['block_days'],
                ' ~ '.join(f'{x*1e6:+,.2f}' for x in [b['return_fraction_difference_quantiles'][0], b['return_fraction_difference_quantiles'][-1]]),
                ' ~ '.join(f'{x:+.6f}' for x in [b['financial_score_difference_quantiles'][0], b['financial_score_difference_quantiles'][-1]])])
        for b in r['ex_post_risk_normalized']:
            risk.append([r['phase'], r['scenario'], b['baseline'], f"{b['hindsight_scale']:.6f}",
                         f"{b['pnl_difference']:+,.2f}", f"{b['q_difference']:+.6f}"])
    body = '''<h1>최근 자료의 페어별 계수 재학습 결과</h1>
<p class="note"><strong>직접 재학습 후보는 이후 구간의 기본·높은 비용 조건에서 이전 페어 후보보다 개선됐다.
그러나 8월 기본 성과는 낮고 수익 집중이 커, 실거래 교체를 검증한 결과는 아니다.</strong>
새 계좌 재현 48개, 기존 계좌 정확 대조 4개, 기존 예측 상태 정확 대조 8개, 구현 검사 8개를 완료했다.</p>
<h2>구현한 변화</h2>
<p>BTC/ETH와 BTC/HYPE 각각의 가격 특성에서 24시간 뒤 고정 헤지 상대수익을 예측한다.
매일 00:00 UTC에 최근 90일의 확정된 정답으로 계수를 다시 학습하고, 이후 매시간 새 특성에 적용한다.
직접 재학습은 최근 특성에서 수익률을 바로 예측하고,
잔차 재학습은 오래된 고정 모델의 오차를 특성별로 예측해 보정한다.</p>
<p>자료 주기나 평균 오차 보정의 효과를 따로 확인하기 위해, 같은 행·가중치·일별 갱신으로
평균 오차만 보정하는 scalar 대조군도 실행했다. 이 대조군은 이후 성적을 보고 선택하는 후보에 넣지 않았다.</p>
<p>학습 표본은 6시간 간격, 최소 200개, 최근 자료 가중치 반감기 30일,
계수 정규화 강도 1과 절편 사전 강도 12를 미리 고정했다.
24시간 정답을 6시간마다 만들므로 학습 표본은 겹친다. 이를 독립 표본 수로 해석하지 않는다.
예측 진단은 별도로 24시간 간격의 겹치지 않는 표본에서 계산했다.</p>
<p>학습 시각보다 정답 확정 시각이 엄격히 이전인 행만 사용한다. 특성 평균·표준편차도 그 학습 행에서만 계산한다.
모델 계수의 학습 시각과 실제 매시간 가격 기준 시각을 구분했고, 원래의 신호 수명과 24시간 만기를 바꾸지 않았다.
보유 연장·유료 헤지 조정·양쪽 독립 체결·계좌 위험 한도는 기존 매시간 bias 페어와 같다.</p>
<h2>동일 자본 계좌 결과</h2>
<p>초기 자본 1,000,000. validation: 2026-08-01 12:00~09-01 12:00 UTC.
comparison: 09-01 12:00~10-15 12:00 UTC. 기본 조건은 편도 수수료·가격 불이익 9bp·10bp,
stress는 12bp·20bp, 제한된 체결 수량, 펀딩 수령액 절반을 적용한다.
Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05), σ=완료 일별 수익률의 표본 표준편차×√n이다.</p>
<p>이 수치는 기존 전략과 페어를 결합한 계좌 전체의 기준시장 백테스트다.
기간을 이전 연구에서도 사용했으므로 독립적인 미래 검증이나 실제 par 지갑 수익으로 제시하지 않는다.</p>'''
    body += grid(['기간', '조건', '전략', '순손익', '수익률', 'Q', '기간 변동성 σ', '최대낙폭', '완료 페어'], account)
    body += '''<p>직접 재학습의 이후 기본 순손익은 +3,422.16, Q 0.12430431,
stress는 +2,104.98, Q 0.10853817이다. 이전 bias 페어의 +1,491.90·+1,031.14보다 각각 1,930.26·1,073.83 높다.
직접 재학습의 이후 기간 변동성도 이전과 비슷한 수준이다.
그러나 8월 기본은 +13,165.33으로 이전 +20,090.87보다 낮다.
8월 stress +20,739.62는 이전보다 높지만 변동성도 증가했다.</p>
<p>첫째 기반 직접 재학습은 두 기간·두 비용 조건에서 기존 첫째·둘째 전략의 PnL과 Q를 모두 넘었다.
다만 8월 stress의 완료 페어가 1개여서 사전에 정한 최소 3개 기준을 통과하지 못했다.
고정 선택은 first_dense_hourlyhedge_direct_24, eligible=false다.
이는 연구 후보 선정 기준이며 대회 참가 자격과 별개다. 이후 결과를 보고 선택을 바꾸지 않았다.</p>
<h2>예측 정확도와 계좌 손익은 다른 결과</h2>
<p>BTC/HYPE의 공통 시점 표본을 비교했다. RMSE는 각 시점의 변동성×√24로 정규화한 수익률 오차다.
예측 크기는 평균 절대 예측 수익률을 bp로 표시했다.</p>'''
    body += grid(['기간', '모델', '표본', 'RMSE', '방향 일치율', '평균 절대 예측 bp'], prediction)
    body += '''<p>직접 재학습의 이후 RMSE는 1.155083으로 이전 bias의 1.137738보다 나쁘고,
방향 일치율도 46.51%로 이전 55.81%보다 낮다. 계좌 성과가 좋아졌다는 이유로 전체 예측 정확도가 좋아졌다고 주장할 수 없다.
거래 조건을 통과한 일부 경로와 보유 과정이 손익에 큰 영향을 준다.</p>
<h2>수익 집중과 민감도</h2>
<p>최대 이익 페어 사건의 실제 현금 경로를 수수료·펀딩·평가손익과 함께 빼 보았다.
다른 모든 주문 경로를 고정한 사후 분해로, 그 거래를 건너뛰었을 때 실제로 운용 가능한 대체 백테스트는 아니다.</p>'''
    body += grid(['기간', '조건', '최대 이익 페어', '전체 양의 페어 손익 중 비중', '그 경로를 뺀 계좌 손익'], concentration)
    body += '''<p>이후 기본은 최대 이익 페어 하나가 양의 페어 손익의 약 75.71%, stress는 약 94.46%다.
그 경로를 빼면 계좌 손익이 각각 −560.87·−719.18이 된다. 수익이 한 거래에 많이 의존한다는 뜻이며,
모든 전략이 최대 이익 거래 없이도 이익이어야 한다는 새 선택 기준을 추가한 것은 아니다.</p>
<p>일별 수익률을 대조군과 같은 사후 변동성으로 맞춰 본 단순 진단에서도 기존 두 전략 대비 개선은 남았다.
이 배율은 미래를 본 값이므로 실제 적용 가능한 규모 정책이나 동일 사전 위험의 재현 결과로 취급하지 않는다.</p>'''
    body += grid(['기간', '조건', '대조군', '사후 수익률 배율', '손익 차이', 'Q 차이'], risk)
    body += '''<p>같은 날짜를 함께 뽑는 원형 블록 재표집도 실행했다. 블록 길이 2·7일, 각각 5,000회다.
모든 비교의 아래 95% 분위 범위는 손익과 Q 차이에서 0을 포함한다.
이미 여러 후보를 시험한 선택 과정을 보정한 통계 검정이 아니며, 미래 성공 확률로 해석할 수 없다.</p>'''
    body += grid(['기간', '조건', '대조군', '블록 일수', '손익 차이 2.5~97.5% 분위', 'Q 차이 2.5~97.5% 분위'], bootstrap)
    body += '''<h2>실제 par 검증 범위</h2>
<p>허용된 실제 과거 호가 5개에서 세 방식 모두 신규 진입 승인은 0개다.
각 방식의 기본·stress 합계 34개 시점에서, 136개 페어·방향 평가 중 125개가 전체 비용 미달,
10개가 수량 단위 불균형, 1개가 표시 깊이 부족이었다.
par 캔들 재현에서도 완료 페어는 없고, 작은 부분 체결 차이만 발생했다.
기준시장 개선을 실제 par의 수익 개선으로 바꿔 말하지 않는다.</p>
<h2>검증과 다음 단계</h2>
<p>독립적인 최소제곱 풀이와 계수 계산을 대조했고, 미래 자료 변경 시 과거 결과 불변,
미확정 정답 배제, 훈련 전용 정규화, 페어별 분리, 원래 예측 정확 재현,
모델 학습 시각과 매시간 신호 시각의 구분을 검사했다.</p>
<p><a href="https://onlinelibrary.wiley.com/doi/10.1155/cplx/1155228">2026년 최근 기간 회귀·경제적 성과 평가 연구</a>를 설계 참고로 사용했다.
이번 고정 조건의 일별 페어 회귀는 논문의 FX 임계값 탐색·다중 기간 앙상블을 복제한 것이 아니다.</p>
<p>이제 다시 선택기 후보를 늘리기보다, 고정한 직접 재학습 후보와 이전 bias 페어를
같은 실제 par 관측에서 진입부터 청산까지 비교할 수 있는지 확인해야 한다.
이미 예약된 공개 자료 수집의 계획 메타데이터만 확인했으며, 원시 미래 결과는 열지 않았다.
그 관측을 재사용할 수 있으면 중복 수집 없이 별도 비교를 사전에 등록하되,
실제 주문이나 자동 운용 교체는 하지 않는다. 실제 시장 중단·공백은 관측 결과의 한계로 남겨야 한다.</p>'''
    body += f"<p>주문 전송 0개. 활성 소스 {guards['active_protected_sources']}개와 예약 의존 파일 {guards['reserved_dependencies']}개가 그대로다.</p>"
    out = ROOT/'reports/rolling_feature_pairs_20261017'; out.mkdir(exist_ok=True)
    (out/'report.html').write_text(document('최근 페어 계수 재학습 결과', body))
    export_csv(out/'comparison.csv', rows)
    summary = dict(selection=results['selection'], new_account_replays=48, exact_control_replays=4,
        exact_expert_states=8, tests_passed=8, actual_book_entries=0, protected=guards,
        historical_independence=False, orders_sent=0, automatic_live_replacement=False)
    save(out/'summary.json', summary)
    save(HERE/'work_state.json', dict(**summary,
        status='progress_daily_feature_refit_later_pnl_q_improved_but_concentrated_no_par_entry',
        previous_goal_turn='progress_cost_selected_32_accounts_and_prior_model_audit',
        report=str((out/'report.html').relative_to(ROOT)), live_strategy_changed=False,
        findings=[
            'Daily per-pair direct/residual and matched scalar implemented. All base/stress first-direct reference cells beat both native PnL/Q; no choice changed after later replay.',
            'First direct later base +3422.16 Q .12430431, stress +2104.98 Q .10853817; improvements versus hourly bias24 +1930.26/+1073.83. August base underperforms hourly bias24 by 6925.55.',
            'August stress completed only one pair. Latest RMSE/direction accuracy is worse despite economic improvement. Largest positive episode concentration 75.71%/94.46% later; paired descriptive resampling intervals cross zero.',
            'Ex-post equal-volatility descriptive returns preserve native PnL/Q advantage, but are not tradable risk-matched controls.',
            'Zero accepted actual old-book entries and zero complete par candle pairs. Full actual par lifecycle and independent evidence remain absent.',
        ],
        next='Before the existing paired_venue capture paper_start=2026-10-17T12:00Z, inspect metadata/process liveness only and assess adding a separately frozen direct-refit versus hourly-bias/native comparison to that same capture. Do not read reserved raw outcomes, mutate existing plans/services, or start duplicate capture. Need matching whole-account independent-leg paper execution with paid holding renewal and actual funding plus source publications sufficient for causal model inference. If feasible implement/test/register prior to common start; otherwise report exact limitation and pursue a distinct later window without outcome peeking. Keep actual live unchanged. Do not call goal complete.',
        remaining=['Research sample gate fails; long-run robustness not established',
                   'No verified profitable actual par entry-to-exit lifecycle',
                   'Independent prospective comparison remains outstanding']))
    print(out/'report.html')


if __name__ == '__main__':
    main()
