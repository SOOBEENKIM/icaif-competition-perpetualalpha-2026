"""Summarize causal forecasts and after-cost performance without reselection."""
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
    result = json.loads((ART/'results.json').read_text())
    forecast = json.loads((ART/'forecast_diagnostics.json').read_text())['rows']
    book = json.loads((ART/'books/results.json').read_text())
    spec = json.loads((HERE/'protocol.json').read_text())
    for file in [ART/'frozen.json', ART/'books/frozen.json']:
        for rel, digest in json.loads(file.read_text())['sources'].items():
            assert sha(ROOT/rel) == digest, rel
    guards = protected()
    rows = result['rows']
    assert len(rows) == 128 and result['new_account_replays'] == 48
    selected = result['selection']
    lookup = {(r['venue'], r['phase'], r['arm'], r['scenario']): r for r in rows}
    labels = {'equal': '단순 평균', 'imse': '과거 오차 역수', 'stack': '오차 공분산 조합'}
    arms = [('first_strategy', '기존 첫째'), ('second_strategy', '기존 둘째'),
            ('first_dense_hourlyhedge_bias_24', '첫째＋이전 bias 페어')]
    arms += [(f'first_dense_hourlyhedge_{k}_24', '첫째＋'+labels[k]) for k in spec['candidates']]
    account = []
    for phase in ['validation', 'comparison']:
        for scenario in ['base', 'stress']:
            for arm, label in arms:
                m = lookup['reference', phase, arm, scenario]['metrics']
                account.append([phase, scenario, label, f"{m['net_pnl']:+,.2f}",
                                f"{100*m['return_fraction']:+.5f}%", f"{m['financial_score']:.8f}",
                                f"{100*m['volatility_horizon']:.5f}%", f"{100*m['max_drawdown']:.5f}%",
                                m.get('completed_pairs', '—')])
    prediction = []
    for r in forecast:
        if r['pair'] != ['BTC', 'HYPE'] or r['model'] not in ['bias', *spec['candidates']]:
            continue
        prediction.append([r['phase'], r['model'], r['samples'], f"{r['normalized_rmse']:.6f}",
                           f"{100*r['sign_accuracy']:.2f}%", f"{r['mean_abs_prediction']:.6f}",
                           ', '.join(f'{w:.3f}' for w in r['average_weights']) if r['average_weights'] else '—'])
    reasons = []
    for kind in spec['candidates']:
        for scenario in ['base', 'stress']:
            count = Counter()
            for row in book['summaries']:
                if row['model'] == kind and row['scenario'] == scenario:
                    count.update(row['counts'])
            reasons.append([kind, scenario, count['scheduled_decisions'], count['edge_below_full_cost'],
                            count['lot_imbalance'], count['visible_depth_below_minimum'], count['open_maturity_exit_unobserved']])
    body = '''<h1>당시 발행한 예측의 성적으로 비중을 갱신하는 페어 앙상블</h1>
<p class="note"><strong>구현과 비교를 완료했으나 기존 후보를 대체할 성능 개선은 확인하지 못했다.</strong>
새 계좌 백테스트 48개, 기존 계좌 정확 재현 4개, 원래 예측 상태 정확 재현 8개와 구현 검사 7개를 완료했다.
이후 기간의 기본 비용에서는 일부 개선됐지만 높은 비용 조건에서 이점을 잃었고, 짧은 실제 par 호가에서도 진입이 발생하지 않았다.</p>
<h2>기존 앙상블과 달라진 점</h2>
<p>이전 pairs_forecast 실험의 ridge·tree·GRU 잔차 앙상블은 고정 평균이었다.
이번에는 이미 구현한 24시간 ridge, 편향 보정, 기울기·편향 보정, 최근 평균의 네 예측을 사용했다.
각 모델이 당시 발표한 예측을 보관하고, 24시간 뒤 정답이 확정된 자료만으로 비중을 갱신한다.
오늘의 보정 계수로 과거 예측을 다시 계산하지 않는다. 72시간 예측을 24시간 예측과 섞지도 않는다.</p>
<p>세 후보는 단순 평균, 정규화된 과거 제곱오차의 역수 비중, 모델 간 오차의 중복까지 고려하는 제약 조합이다.
마지막 방식은 비중을 음수가 아닌 값으로 제한하고 합을 1로 둔 제곱오차 최소화이며, 균등 비중으로의 정규화를 적용했다.
최근 90일, 반감기 30일, 사전 표본 강도 12를 실행 전에 고정했다. 학습에는 24시간 간격으로 겹치지 않는 예측만 사용한다.
현재 시각과 정확히 같은 시각에 만기인 표본도 제외하며, 사용 가능한 표본이 12개 미만이면 균등 비중을 사용한다.</p>
<p>매시간 신호 갱신·진입·보유 연장·유료 헤지 조정·수수료·펀딩·각 다리의 독립 체결과 계좌 위험 한도는 이전 후보와 같다.
이번 실험에는 직전 수량 감축 완료 로직을 합치지 않았다. 예측 조합 자체의 효과를 비교했다.</p>
<h2>동일 자본의 계좌 손익</h2>
<p>초기 자본 1,000,000. validation: 2026-08-01 12:00~09-01 12:00 UTC.
comparison: 09-01 12:00~10-15 12:00 UTC. 기본 조건은 편도 수수료 9bp·가격 불이익 10bp,
stress는 12bp·20bp, 더 적은 체결 가능 수량, 펀딩 수령액 절반이다.
Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05), σ는 완료 일별 수익률의 표본 표준편차×√n이다.
아래 순손익은 페어만의 수익이 아니라 기존 전략과 페어를 함께 운용한 계좌 전체 값이다.</p>
<p>이 기간들은 앞선 연구에서도 사용했다. 이후 기간 실행 전에 선택 파일을 고정했지만,
이 결과를 독립적인 미래 검증이나 실제 par 지갑 실적으로 제시하지 않는다.</p>'''
    body += grid(['기간', '조건', '전략', '순손익', '수익률', 'Q', '기간 변동성 σ', '최대낙폭', '완료 페어'], account)
    body += '''<p>오차 역수 조합은 이후 기본 조건에서 순손익 +2,503.38, Q 0.11223224로,
이전 bias 페어의 +1,491.90, Q 0.09855552보다 좋아졌다.
그러나 같은 기간 stress에서는 −457.10으로 이전의 +1,031.14보다 나빴다.
8월 기본에서도 +14,692.50으로 이전 +20,090.87보다 낮았다.</p>
<p>8월 stress에서는 세 조합 모두 페어 체결을 추가하지 못해 기존 첫째 전략과 결과가 같았다.
따라서 PnL·Q 개선과 최소 완료 페어 기준을 통과한 후보는 없다.
고정된 진단용 선택은 first_dense_hourlyhedge_equal_24, eligible=false이며,
이후 기본 구간에서 더 좋은 결과가 보였다는 이유로 이를 바꾸지 않았다.</p>
<h2>예측 오차와 거래 이익을 따로 확인</h2>
<p>같은 시각·같은 정답을 가진 겹치지 않는 24시간 예측만 비교했다.
RMSE와 예측 크기는 각각 원래 시점의 페어 변동성×√24로 정규화한 값이다.
평균 비중 순서는 ridge, bias, affine, recent다. 표본 수는 계좌 체결 수가 아니다.</p>'''
    body += grid(['기간', '모델', '표본', 'RMSE', '방향 일치율', '평균 절대 예측', '평균 비중'], prediction)
    body += '''<p>BTC/HYPE의 이후 구간 RMSE는 bias 1.137738에서 오차 역수 조합 1.133790으로 소폭 줄었다.
방향 일치율은 둘 다 55.81%다. 평균 절대 예측은 0.287378에서 0.207231로 약 28% 줄었다.
하지만 높은 비용 조건의 거래 성과는 악화됐다. 8월에는 앙상블의 RMSE 자체도 더 나빴다.
따라서 오차 감소가 안정적으로 확인됐거나, 오차 감소가 곧 수익 개선이라고 결론 내릴 수 없다.</p>
<h2>실제 par 호가에서의 진입 검토</h2>
<p>허용된 과거 호가 기록 5개, 총 약 4.48시간만 사용했다.
각 조합·비용 조건에서 17개 판단 시점을 확인했고, 두 페어의 양방향을 평가했다.
호가를 통과한 신규 주문 계획은 0개다. 아래 사유 수는 시점마다 페어·방향별로 평가한 횟수이므로 판단 시점 수와 다르다.</p>'''
    body += grid(['조합', '조건', '판단 시점', '전체 비용 미달', '수량 단위 불균형', '최소 주문 깊이 미달', '양쪽 진입'], reasons)
    body += '''<p>기본 조건에서는 68개 페어·방향 평가 중 66개가 전체 비용 미달, 2개가 주문 수량 단위 불균형이었다.
실제 체결·장기 청산 성능을 새로 입증한 것은 없다. par 캔들·펀딩 계좌 재현에서도 새 조합의 완료 페어는 0개였다.
호가 검사는 현재 비용과 표시 수량만 검증하며, 미래 청산 유동성을 보장하지 않는다.</p>
<h2>논문과 검증 범위</h2>
<p><a href="https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2026.1749337/full">2026년 암호화폐 페어 딥러닝 연구</a>의
최근 예측 정확도에 따른 동적 비중 아이디어를 참고했다. 해당 논문은 거래 비용과 유동성 제약을 직접 모형화하지 않는다.
이번 구현은 기존 예측기에 대한 자체 조합 실험이며 논문의 DNN·LSTM·Johansen 모형을 재현한 것이 아니다.</p>
<p>독립적인 수치 최적화기와 비중 해를 대조하고, 미래 자료 변경 시 과거 예측·비중이 불변인지,
만기 경계를 엄격히 지키는지, 결측 표본을 배제하는지, 기존 예측이 그대로 재현되는지 검사했다.
원전략과의 정확 계좌 대조와 페어 손익 분해도 완료했다.</p>
<h2>다음 개발 판단</h2>
<p>이번 세 조합은 교체 후보로 채택하지 않는다. 다음에는 예측을 평균내서 오차를 줄이는 목적과
비용을 지불할 가치가 있는 방향·시점을 고르는 목적을 구분해야 한다.
가능한 다음 검증은 고정된 모델 중 하나를 선택하되, 결과가 확정된 과거의 비용 반영 의사결정 성적을 사용하는 방식이다.
단순히 축소된 예측을 사후 배수로 키우거나 유리했던 이후 기간을 다시 최종 검증으로 사용하지 않는다.
이 다음 방식은 아직 구현·평가하지 않았다.</p>'''
    body += f"<p>실제 주문 전송 0개, 실거래 교체 없음. 활성 소스 {guards['active_protected_sources']}개와 예약 의존 파일 {guards['reserved_dependencies']}개의 해시가 그대로다.</p>"
    out = ROOT/'reports/causal_expert_pairs_20261017'; out.mkdir(exist_ok=True)
    (out/'report.html').write_text(document('페어 예측 앙상블 검증', body))
    export_csv(out/'comparison.csv', rows)
    summary = dict(selection=selected, new_account_replays=48, exact_control_replays=4, exact_expert_states=8,
                   tests_passed=7, actual_book_accepted_entries=0, historical_independence=False,
                   protected=guards, orders_sent=0, automatic_live_replacement=False)
    save(out/'summary.json', summary)
    save(HERE/'work_state.json', dict(**summary,
        status='progress_causal_experts_48_accounts_7_tests_no_stress_or_par_improvement',
        previous_goal_turn='progress_retained_completion_report_and_integrity_audit_completed',
        report=str((out/'report.html').relative_to(ROOT)), live_strategy_changed=False,
        findings=[
            'Replaces static model averaging with original issued-prediction loss weighting and constrained stacking. Mature labels only; eight original expert states and four full controls exact.',
            'First imse later base gains 1011.48 over hourly bias24 but stress loses 1488.24 and August base loses 5398.38. All candidates fail frozen selection.',
            'BTC/HYPE later nonoverlap RMSE decreases slightly with imse, direction accuracy unchanged and forecast amplitude down about 28%; August RMSE worsens.',
            'All three candidates have zero accepted actual-book entry plans across five old tapes. Base 66 of 68 direction/pair probes are below full cost; no full actual par lifecycle proved.',
        ],
        next='Investigate causal cost-aware expert decisions rather than mean forecast compression. Keep hourly bias24 as explicit control. Derive learning target from strictly matured source-time decisions and known-at-source fee/funding/exit assumptions; separate proxy training utility from actual ledger evaluation. Freeze choices before runs and do not tune forecast magnitude on later outcomes. First inspect prior cost/decision ensemble attempts to avoid repeating them.',
        remaining=['Robust incremental PnL/Q improvement beyond hourly bias24 absent',
                   'Actual par profitable full lifecycle and independent prospective evidence absent']))
    print(out/'report.html')


if __name__ == '__main__':
    main()
