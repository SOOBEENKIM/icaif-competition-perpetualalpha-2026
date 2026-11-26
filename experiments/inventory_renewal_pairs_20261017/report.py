"""Publish complete renewal comparisons and reconciled cost attribution."""
import json
from pathlib import Path
from experiments.par_direct_forecast_20261017.data import ROOT, save, sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.netted_shadow_pairs_20261017.report import document, table, export_csv
from experiments.dense_history_pairs_20261017.report import grid

HERE = Path(__file__).resolve().parent
ART = HERE / 'artifacts'


def main():
    results = json.loads((ART / 'results.json').read_text())
    attribution = json.loads((ART / 'attribution.json').read_text())
    frozen = json.loads((ART / 'frozen.json').read_text())
    for rel, digest in frozen['sources'].items():
        assert sha(ROOT / rel) == digest, rel
    guards = protected()
    out = ROOT / 'reports/inventory_renewal_pairs_20261017'
    out.mkdir(exist_ok=True)
    selected_arms = [
        'first_strategy', 'second_strategy', 'first_dense_ridge_24',
        'first_dense_renew_ridge_24', 'first_dense_renew_bias_24',
        'first_dense_renew_affine_24',
    ]
    comparisons = []
    for phase in ['validation', 'comparison']:
        for arm in selected_arms:
            for scenario in ['base', 'stress']:
                row = next(x for x in results['rows'] if
                           (x['venue'], x['phase'], x['arm'], x['scenario']) ==
                           ('reference', phase, arm, scenario))
                m = row['metrics']
                comparisons.append([
                    phase, arm, scenario, f"{m['net_pnl']:+,.2f}",
                    f"{100 * m['return_fraction']:+.6f}", f"{m['financial_score']:.8f}",
                    f"{100 * m['volatility_horizon']:.6f}", f"{100 * m['max_drawdown']:.6f}",
                    m['completed_pairs'], m.get('renewals', 0),
                    f"{m['pnl_after_liquidation_reserve']:+,.2f}",
                ])
    costs = []
    rejections = []
    for row in attribution['rows']:
        if row['venue'] != 'reference' or row['phase'] != 'comparison' or row['arm'] not in selected_arms:
            continue
        d = row['delta']
        costs.append([row['arm'], row['scenario'], *[
            f'{d[k]:+,.2f}' for k in [
                'pair_price_change', 'pair_execution_cost_saved', 'pair_fees_saved',
                'pair_funding_change', 'base_owned_change', 'account_net_change',
            ]
        ]])
        rejections.append([
            row['arm'], row['scenario'], row['renewal_events'],
            row['rejected'].get('new_hedge_requires_resizing', 0),
            row['rejected'].get('forecast_exceeds_hard_holding_limit', 0),
        ])
    par = [x for x in attribution['rows'] if x['venue'] == 'par']
    assert sum(x['renewal_events'] for x in par) == 0
    assert max(abs(x['delta']['account_net_change']) for x in par) == 0
    body = '''<h1>페어 보유 연장: 비용 절감과 보유 경로 변화 비교</h1>
<p class="note"><strong>보유 연장은 일부 재사용 과거 구간의 PnL·Q를 개선했지만 교체 기준을 통과한 후보는 없다.</strong>
16개 후보의 128개 새 계좌 재현, 대조군 검증 재현 4개, 이미 검증된 대조군 144개를 비교했다.
실제 주문은 보내지 않았고 실거래 전략은 변경하지 않았다.</p>
<h2>이번에 구현한 변경</h2>
<p>기존 24시간·72시간 페어 예측을 그대로 두고 보유 종료 규칙만 수정했다.
만기 전 마지막 정규 6시간 판단에서 새 예측이 실제로 도착했고 현재 보유를 유지할 기대 현금 흐름이
즉시 유료 청산보다 충분히 좋으면 보유 만기를 연장한다. 신규 매수·매도 수량은 없으며 실제 보유 수량을 유지한다.
이미 지급한 진입 수수료를 다시 부과하거나, 청산 수수료를 사후에 빼서 가상의 수익을 만들지 않는다.</p>
<p>유지 가치는 현재 가격에서 새 예측 만기 가격까지의 수량별 현금 변화, 당시 알려진 펀딩,
미래 유료 청산 추정치를 포함한다. 즉시 청산도 같은 불리한 가격·틱·수수료 규칙으로 비교하며
현재 보유 명목금액의 5bp를 넘는 차이가 있어야 한다. 예상 펀딩 수령액은 절반만 인정한다.
예측 현금 흐름은 판단에만 사용하고 계좌 현금으로 기록하지 않는다.</p>
<p>최초 체결가·수량·변동성에 따른 손절 기준과 최초 체결 시점은 유지한다. 최대 보유시간은 168시간이다.
새 예측 만기가 이 제한을 넘으면 예측을 임의로 짧게 환산하지 않고 거절한다.
한쪽 미체결·헤지 보완·위험 초과·이미 시작한 청산을 보유 연장으로 취소하지 않는다.
새 예측의 헤지 수량 비율과 현재 비율의 차이가 5%를 넘으면 기존 수량만으로 연장하지 않는다.</p>
<h2>동일 조건의 기준시장 비교</h2>
<p>초기자본 1,000,000. validation은 2026-08-01 12:00~09-01 12:00 UTC,
comparison은 09-01 12:00~10-15 12:00 UTC다. base는 편도 수수료 9bp·불리한 체결 가격 10bp,
stress는 12bp·20bp, 거래량 참여율 1/4 및 펀딩 수령액 50%다.
이는 기준시장 백테스트이며 실제 par 지갑 수익이 아니다.</p>
<p>Q = (max(R,0)+0.005)/(abs(R)+1.5σ+0.05).
σ는 12:00 UTC 표본의 완료 일별 수익률 표본 표준편차에 √n을 곱한 기간 변동성이다.
심사 점수는 비교에 넣지 않았다. 종료 시 남은 포지션은 평가손익에 포함하며,
마지막 열은 추가 청산 비용 추정치를 차감한 값이다. 그 비용을 실제 체결한 청산으로 해석하면 안 된다.</p>
'''
    body += grid([
        '기간', '전략', '조건', '순손익', '수익률 %', 'Q', '기간 σ %', 'MDD %',
        '완료 페어', '보유 연장', '잔여 청산비 추정 차감 손익',
    ], comparisons)
    body += '''<p>이후 구간의 24시간 릿지 보유 연장은 기본 조건에서 기존 고정 만기 페어 후보의
−4,532.67을 +5,900.60으로 바꿨다. 첫째 원전략 +459.53, 둘째 +572.88보다 PnL·Q가 높았지만
기간 변동성과 최대낙폭도 두 원전략보다 컸다. 불리한 조건에서는 보유 연장이 없었고 −961.49로 원전략보다 나빴다.</p>
<p>bias 24시간의 이후 stress 순손익은 +7.60이지만 Q 0.08233749는 두 원전략보다 낮으며,
잔여 청산 비용 추정치를 빼면 −130.11이다. 작은 양의 평가손익만으로 개선을 인정하지 않는다.</p>
<h2>개선분은 어디서 왔는가</h2>
<p>다음은 같은 예측을 쓰는 고정 만기 후보 대비 차이다. 비용 절감 열의 양수는 비용이 줄었다는 뜻이다.
가격 변화는 실제로 달라진 보유 경로의 손익 차이이며 예측 정확도가 그만큼 개선됐다는 뜻은 아니다.
페어와 원전략 소유 원장을 합친 값이 물리 계좌의 손익 차이와 일치하도록 검증했다.</p>'''
    body += grid([
        '후보', '조건', '페어 가격손익 차이', '체결 비용 절감', '수수료 절감',
        '펀딩 차이', '원전략 원장 차이', '계좌 총 개선분',
    ], costs)
    body += '''<p>릿지 24시간 기본 조건의 총 개선분 +10,433.27 중 체결 비용·수수료 감소는
818.01+732.26=1,550.27이며 가격손익 차이는 +8,973.83이다.
따라서 비용 절감만으로 수익이 좋아졌다고 설명할 수 없다.
새로운 보유 경로가 이 구간에서 유리했던 영향이 더 크며, 다른 기간에서도 유리한지는 별도 문제다.</p>
<h2>통과 여부와 다음 수정 근거</h2>
<p>8월 base·stress 모두에서 양의 PnL, 기존 두 원전략과 같은 예측의 고정 만기 대조군보다 높은 PnL·Q,
실제로 청산까지 마친 페어 최소 3건, 위험 위반 없음이 필요하다. 16개 모두 실패했다.
이후 구간을 보기 전에 정한 진단 대표는 first_dense_renew_affine_24이며 eligible=false다.
이후 구간에서 좋아 보이는 릿지 24시간으로 대표를 사후 교체하지 않았다.</p>'''
    body += grid(['후보', '조건', '보유 연장 수', '새 헤지 수량 필요로 거절', '최대 보유시간 초과 거절'], rejections)
    body += '''<p>다음 검토 대상은 새 예측의 헤지 비율이 달라졌을 때 양쪽 전체를 청산·재진입하는 대신
필요한 수량 차이만 실제 비용을 내고 조정하는 방식이다. 현재 5% 제한을 단순히 풀지는 않는다.
수량 조정의 유리함과 규모 감소 효과를 구분하고, 한쪽만 체결된 상태·미완료 조정·기존 청산·전체 위험을 함께 처리해야 한다.
수량이 바뀌면 최초 손절 판단도 실제 누적 손익을 추적하도록 확장해야 한다. 이 다음 변경은 아직 구현·평가하지 않았다.</p>
<h2>par 자료와 증거의 범위</h2>
<p>10/01~10/16 실제 par 캔들·펀딩 재현 64개 새 후보 계좌에서는 보유 연장이 한 번도 발생하지 않았고
고정 만기 대조군 대비 손익 차이가 모두 0이었다. 일부 거래가 있었더라도 이번 변경의 개선 증거는 아니다.
캔들 거래량 기반의 조건부 체결 모형이며 전체 기간의 실제 주문장 체결·청산 검증이 아니다.</p>
<p>허용된 기존 호가 기록 다섯 개는 6시간 신호의 1시간 진입 유효기간과 겹치지 않는다.
이를 맞추려고 신호 시각을 사후 이동하거나 없는 미래 호가를 만들지 않았다.
과거 비교 기간은 여러 실험에서 재사용한 개발 자료다. 시점별 미래 정보 누출을 막았다는 사실과
독립적인 새 기간에서 성능이 입증됐다는 주장은 구분한다.</p>
<h2>구현 검증</h2>
<p>보유 수량·주문 수 보존, 최초 손절 유지, 미완료 청산 유지, 미래 자료 변경 시 과거 결정 불변,
현재·미래 유료 청산 비교, 펀딩 단일 반영 등을 포함한 14개 검사를 통과했다.
기존 네 대조군을 다시 실행해 체결·잔고·지표·사건 기록의 정확한 일치를 확인했고,
나머지 대조군의 원본 파일 해시와 고정 예측 배열도 검증했다.
모든 신규 후보의 에피소드 손익 분해 오차는 6.78×10⁻¹¹ 이하다.
보호된 실거래 소스 373개와 예약 의존 파일 345개는 그대로다.</p>
<h2>논문과 구현의 관계</h2>
<p><a href="https://link.springer.com/article/10.1007/s10614-025-11163-0">Kitapbayev·Leung의 유한기간 페어 거래 최적정지 연구</a>
출판사 초록은 거래 비용을 포함해 진입·청산을 연계해 다룬다. 이번 구현은 그 문제의식에 따른 자체 비교 규칙이며,
해당 논문의 자유경계 최적정지 해법을 재현한 것은 아니다. 논문의 이론적 결과가 우리 시장 수익성을 보장하지 않는다.</p>'''
    for venue in ['reference', 'par']:
        body += '<h2>' + ('기준시장 전체 결과' if venue == 'reference' else 'par 조건부 체결 전체 결과') + '</h2>'
        body += table([x for x in results['rows'] if x['venue'] == venue])
    (out / 'report.html').write_text(document('페어 보유 연장 비교', body))
    export_csv(out / 'comparison.csv', results['rows'])
    summary = dict(
        selection=results['selection'], tests_passed=14,
        new_candidate_replays=results['new_candidate_replays'],
        control_verification_replays=results['control_verification_replays'],
        reused_control_accounts=results['reused_control_accounts'],
        maximum_ownership_error=max(x['metrics']['maximum_ownership_error'] for x in results['rows']),
        maximum_episode_accounting_error=attribution['maximum_episode_accounting_error'],
        par_renewals=0, par_maximum_pnl_change=0,
        protected=guards, orders_sent=0, automatic_live_replacement=False, historical_independence=False,
    )
    save(out / 'summary.json', summary)
    save(HERE / 'work_state.json', dict(
        status='progress_inventory_renewal_compared_and_cost_attribution_complete_no_qualified_replacement',
        previous_goal_turn='progress_recent_mature_residual_calibration_compared_costly_renewals_identified',
        report=str((out / 'report.html').relative_to(ROOT)), authoritative_results='artifacts',
        selection=results['selection'], new_candidate_replays=128, control_verification_replays=4,
        reused_control_accounts=144, tests_passed=14, orders_sent=0, live_strategy_changed=False,
        findings=[
            'Causal same-quantity renewal improves some reused reference-history results, but all 16 candidates fail August selection.',
            'Ridge 24h later base account PnL changes from -4532.674760 to +5900.599051; stress remains -961.486338.',
            'Its +10433.273811 account delta includes +8973.827420 pair price-path change, +1550.267353 fees/impact saved, -33.481616 funding, -57.339345 base-owned change.',
            'Improvement is mainly a changed holding path, not merely avoided transaction costs or a demonstrated forecast improvement.',
            'Ridge later stress has one renewal rejected because new hedge requires a quantity adjustment.',
            'All 64 par candidate replays have zero renewals and zero PnL difference to matched controls.',
            'No live orders or protected source changes; no reserved future raw data read.',
        ],
        next='Evaluate a causal paid minimum-delta hedge adjustment at renewal while preserving risk-budget exposure. Do not simply relax the hedge tolerance or shrink repeatedly. Seal independent leg limits and risk states, reconcile partial fills before accepting the new maturity, retain sticky exits and hard holding cap, and replace fixed-quantity stop accounting with actual episode PnL without resetting its original risk anchor. Freeze rules before replays and retain matched fixed-maturity and no-resize renewal controls.',
        remaining=['No candidate meets robust PnL/Q comparisons', 'Full par L2 lifecycle profitability unverified', 'Independent prospective improvement unverified'],
    ))
    print(json.dumps(summary, ensure_ascii=False))
    print(out / 'report.html')


if __name__ == '__main__':
    main()
