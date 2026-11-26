"""Publish all outcomes, including failed candidates and unchanged par paths."""
import json
from pathlib import Path
from experiments.par_direct_forecast_20261017.data import ROOT, save, sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.netted_shadow_pairs_20261017.report import document, table, export_csv
from experiments.dense_history_pairs_20261017.report import grid

HERE = Path(__file__).resolve().parent
ART = HERE/'artifacts'


def main():
    r = json.loads((ART/'results.json').read_text())
    a = json.loads((ART/'attribution.json').read_text())
    frozen = json.loads((ART/'frozen.json').read_text())
    for rel, digest in frozen['sources'].items():
        assert sha(ROOT/rel) == digest, rel
    guards = protected()
    out = ROOT/'reports/paid_rehedge_pairs_20261017'; out.mkdir(exist_ok=True)
    arms = ['first_strategy', 'second_strategy', 'first_dense_renewhedge_bias_24',
            'first_dense_renew_ridge_24', 'first_dense_renewhedge_ridge_24',
            'first_dense_renew_affine_72', 'first_dense_renewhedge_affine_72']
    comparisons = []
    for phase in ['validation', 'comparison']:
        for arm in arms:
            for scenario in ['base', 'stress']:
                m = next(x['metrics'] for x in r['rows'] if
                         (x['venue'],x['phase'],x['arm'],x['scenario']) == ('reference',phase,arm,scenario))
                comparisons.append([phase,arm,scenario,f"{m['net_pnl']:+,.2f}",
                                    f"{100*m['return_fraction']:+.6f}",f"{m['financial_score']:.8f}",
                                    f"{100*m['volatility_horizon']:.6f}",f"{100*m['max_drawdown']:.6f}",
                                    m['completed_pairs'],m.get('hedge_adjustments_completed',0),
                                    m.get('hedge_adjustments_failed',0),f"{m['pnl_after_liquidation_reserve']:+,.2f}"])
    costs = []
    for row in a['rows']:
        if row['venue']=='reference' and row['phase']=='comparison' and row['arm'] in [
                'first_dense_renewhedge_ridge_24','first_dense_renewhedge_bias_72','first_dense_renewhedge_affine_72']:
            d = row['delta']
            costs.append([row['arm'],row['scenario'],*[f'{d[k]:+,.2f}' for k in [
                'pair_price_change','pair_execution_cost_saved','pair_fees_saved',
                'pair_funding_change','base_owned_change','account_net_change']]])
    par = [x for x in a['rows'] if x['venue']=='par']
    assert sum(x['submitted_adjustments'] for x in par)==0
    assert max(abs(x['delta']['account_net_change']) for x in par)==0
    body = '''<h1>페어 보유 연장 시 필요한 수량만 유료 조정</h1>
<p class="note"><strong>새 수량 조정 후보는 일부 이후 구간의 PnL·Q를 개선했지만, 전체 교체 기준을 통과하지 못했다.</strong>
17개 검사를 통과한 뒤 128개 새 후보 계좌를 실행했다. 기존 대조군 272개를 재사용하고 4개를 다시 실행해 정확한 일치를 확인했다.
실제 주문은 0건이며 실거래 전략은 변경하지 않았다.</p>
<h2>구현</h2><p>새 예측의 헤지 비율이 현재 보유와 5% 이상 다르면 기존 보유 연장판은 연장을 거절했다.
새 후보는 현재 명목금액과 원래 페어 위험 예산 중 작은 금액을 유지하면서 새 비율의 수량을 계산하고 차이만 거래한다.
규모를 반복해서 줄여 손실을 줄이는 방식과 구분하기 위해, 더 작은 수량을 탐색하지 않는다.
수량 반올림·최소 주문금액·두 주문 중 하나만 체결된 상태의 위험 한도를 만족하지 못하면 조정을 거절한다.</p>
<p>조정 주문 비용, 알려진 펀딩, 미래 유료 청산을 포함한 보유 가치를 즉시 유료 청산과 비교한다.
남은 경제적 여유를 양쪽 주문의 독립적인 지정가 한도로 나누고, 이미 지급한 진입 수수료를 중복 부과하지 않는다.
다음 관측에서 두 주문은 각각 체결된다. 전혀 체결되지 않으면 기존 만기를 유지한다.
일부라도 체결됐는데 새 헤지·규모 허용범위를 맞추지 못하거나 현재 기대 가치가 사라졌으면 실제 비용을 내고 청산한다.
새 만기는 조정 완료를 확인한 뒤에만 반영한다. 신호가 나온 시각에서 만기를 계산하며 체결이 늦었다고 만기를 더 늦추지 않는다.</p>
<p>최초 진입가·수량만 계속 사용하면 추가한 노출의 손실을 빠뜨릴 수 있다. 조정 이후의 손절은
원래 진입 위험 한도를 유지한 채 실제 소유 원장의 누적 가격 손익으로 판단한다.
줄인 수량의 실현손익과 늘린 수량의 평가손익이 모두 반영된다. 최초 위험 기준과 168시간 최대 보유시간은 초기화하지 않는다.</p>
<h2>검증 범위</h2><p>17개 검사는 독립 부분체결과 유료 철수, 주문장 수량 제한, 알 수 없는 주문 결과의 재제출 차단,
한쪽만 체결될 때의 위험 한도, 최초 손절 기준 보존, 실제 수량의 펀딩, 미래 변경 시 과거 결정 불변을 포함한다.
주문장 검사는 합성 자료의 기능 검사이며 실제 par 수익 검증이 아니다.
실제 주문 결과를 모르면 조정을 완료하거나 미체결로 단정하지 않고 확인 대기 상태로 남긴다.</p>
<h2>동일 조건의 기준시장 결과</h2><p>초기자본 1,000,000. validation은 2026-08-01 12:00~09-01 12:00 UTC,
comparison은 09-01 12:00~10-15 12:00 UTC. base는 편도 수수료 9bp·불리한 체결 가격 10bp,
stress는 12bp·20bp, 거래량 참여율 1/4과 펀딩 수령액 50%다.
이는 기준시장 조건부 백테스트이며 실제 대회 지갑 수익이 아니다.</p>
<p>Q=(max(R,0)+0.005)/(abs(R)+1.5σ+0.05).
σ는 12:00 UTC 표본에서 완료 일별 수익률의 표본 표준편차에 √n을 곱한 기간 변동성이다.
마지막 열은 종료 시 남은 보유의 추정 청산 비용을 추가 차감한 값이며 실제 청산 체결을 뜻하지 않는다.</p>'''
    body += grid(['기간','전략','조건','순손익','수익률 %','Q','기간 σ %','MDD %',
                  '완료 페어','조정 완료','조정 후 철수','추정 잔여 청산비 차감 손익'],comparisons)
    body += '''<p>이후 구간의 affine 72시간 수량 조정판은 기본 조건 +2,915.03·Q 0.11469737,
불리한 조건 +1,056.53·Q 0.09401647로, 그 구간의 첫째·둘째 원전략보다 두 지표가 높았다.
하지만 8월에는 기본 +8,538.53, 불리한 조건 +8,805.72로 원전략보다 나빴고 완료 거래 수도 부족했다.
이후 결과를 보고 이 후보를 선정한 것으로 바꾸지 않는다.</p>
<p>반대로 릿지 24시간의 이후 기본 조건은 직전 보유 연장판 +5,900.60에서 +4,111.51로 나빠졌다.
수량 조정 5회 중 3회는 완료했고 2회는 체결 후 다시 청산했다.
헤지 조정을 더 자주 한다는 사실 자체가 성능 개선은 아니다.</p>
<h2>직전 보유 연장판 대비 손익 차이</h2>
<p>비용 절감 열의 양수는 비용 감소, 음수는 추가 비용이다. 가격손익 차이는 달라진 실제 보유 경로의 효과다.
원전략·페어 원장 합과 물리 계좌 차이를 대조했으며 최대 에피소드 손익 오차는 6.78×10⁻¹¹ 이하다.</p>'''
    body += grid(['후보','조건','페어 가격손익 차이','체결 비용 절감','수수료 절감','펀딩 차이','원전략 원장 차이','계좌 총 차이'],costs)
    body += '''<p>릿지 24시간 기본 조건의 악화 −1,789.09는 추가 비용 약 42.70보다 가격손익 경로 차이 −1,752.16이 더 크다.
따라서 다음 수정에서 수수료만 줄이면 해결된다고 가정하지 않는다.
affine 72시간 stress는 가격손익이 −2,293.90 나빠졌지만 거래 비용·수수료 절감과 펀딩 변화가 이를 상쇄했다.</p>
<h2>선정과 par 결과</h2><p>8월 두 비용 조건에서 양의 수익, 기존 두 원전략보다 높은 PnL·Q,
고정 만기 및 직전 보유 연장 대조군보다 나쁘지 않은 결과, 최소 3개 실제 완료 페어, 위험 위반 없음이 필요하다.
직전 연장 대조군보다 최소 한 조건에서는 PnL·Q가 엄격히 개선되어야 한다.
16개 모두 실패했다. 이후 결과 계산 전 고정한 진단 대표는 first_dense_renewhedge_bias_24이며 eligible=false다.</p>
<p>실제 par 캔들·펀딩을 이용한 새 후보 64개에서는 수량 조정 주문이 0개였고 직전 대조군과 손익 차이가 모두 0이었다.
조정 판단까지 간 두 경우는 보유가 너무 작아 수량을 반올림한 뒤 명목금액을 보존할 수 없어 거절됐다.
캔들 거래량 기반 체결 제약의 결과이지 실제 주문장에 유동성이 없다는 증거는 아니다.</p>
<p>허용된 기존 호가 기록 다섯 개는 현재의 6시간 신호·1시간 진입 유효기간과 겹치지 않는다.
따라서 실제 호가에서 이 후보의 진입부터 만기 청산까지 수익성을 확인했다고 주장할 수 없다.
보호된 실거래 소스 373개·예약 의존 파일 345개는 변경하지 않았고 예약 미래 원자료를 열지 않았다.</p>
<h2>다음 작업</h2><p>수량 조정 기능을 더 복잡하게 하기 전에, <strong>24/72시간 고정 만기 예측을 1시간 뒤 진입 판단에서 버리는 이유가 타당한지</strong>
검증한다. 예측은 6시간마다 갱신되므로 다음 갱신 전까지 동일한 원본 예측을 현재 가격·남은 만기·비용으로 재평가하는 후보를 비교할 수 있다.
과거 주문장 시간에 맞춰 신호 시각을 옮기지 않고, 새 예측으로 원본을 바꾸거나 미래에 알게 된 체결 정보를 사용하지 않는다.
기존 1시간 유효기간과 다음 6시간 갱신까지의 유효기간을 규칙 고정 후 비교할 예정이다.
이는 아직 개선이 입증된 변경이 아니며, 호가 자료가 없는 만기 청산은 미검증으로 남긴다.</p>
<p>모든 과거 구간은 여러 실험에서 재사용한 개발 자료다. 이번의 일부 구간 개선을 독립적인 미래 검증으로 표현하지 않는다.
관련 직전 설계와 논문 맥락은 <a href="../inventory_renewal_pairs_20261017/report.html">보유 연장 보고서</a>에 기록했다.</p>'''
    for venue in ['reference','par']:
        body += '<h2>'+('기준시장 전체 결과' if venue=='reference' else 'par 조건부 전체 결과')+'</h2>'
        body += table([x for x in r['rows'] if x['venue']==venue])
    (out/'report.html').write_text(document('페어 수량 조정 비교',body))
    export_csv(out/'comparison.csv',r['rows'])
    summary = dict(selection=r['selection'],tests_passed=17,new_candidate_replays=128,
                   control_verification_replays=4,reused_control_accounts=272,
                   maximum_ownership_error=max(x['metrics']['maximum_ownership_error'] for x in r['rows']),
                   maximum_episode_accounting_error=a['maximum_episode_accounting_error'],
                   par_adjustment_orders=0,par_maximum_pnl_difference=0,
                   protected=guards,orders_sent=0,automatic_live_replacement=False,historical_independence=False)
    save(out/'summary.json',summary)
    save(HERE/'work_state.json',dict(
        status='progress_paid_delta_rehedge_implemented_tested_replayed_attributed_no_qualified_replacement',
        previous_goal_turn='progress_inventory_renewal_compared_and_cost_attribution_complete_no_qualified_replacement',
        authoritative_results='artifacts',report=str((out/'report.html').relative_to(ROOT)),
        tests_passed=17,new_candidate_replays=128,control_verification_replays=4,reused_control_accounts=272,
        selection=r['selection'],orders_sent=0,live_strategy_changed=False,
        findings=[
            'Paid minimum-delta hedging preserves target gross and original cumulative loss anchor; independent fills, unknown outcomes and sticky paid unwind are tested.',
            'Affine 72h later reference base/stress improves PnL and Q over both native strategies, but August does not; not eligible and not promoted.',
            'Ridge 24h later base worsens versus no-resize renewal, mainly due to altered price path rather than approximately 42.70 additional fees/impact.',
            'All 16 candidates fail the frozen August selection; first_dense_renewhedge_bias_24 remains only the diagnostic representative.',
            'All 64 par candidate accounts have zero adjustment orders and zero PnL difference to no-resize controls; two attempts at valuation are rejected by size rounding.',
            'No live configuration/source changes or orders; reserved prospective outcomes remain unopened.',
        ],
        next='Freeze and compare one-hour source validity versus validity until the next six-hour forecast refresh, for the same fixed 24/72h price forecasts. Preserve original source/terminal prices, remaining maturity and current executable cost revaluation; never shift source times to recorded-book windows. Diagnose actual par entry availability and exercise independent received-book packets where observed; missing full-horizon exits remain explicitly unverified. Keep original native, fixed-maturity, no-resize and paid-rehedge controls; no live replacement or prospective raw data access.',
        remaining=['No cross-period robust PnL/Q replacement','Actual complete par book lifecycle profitability unverified','No independent prospective improvement evidence'],
    ))
    print(json.dumps(summary,ensure_ascii=False)); print(out/'report.html')


if __name__=='__main__':
    main()
