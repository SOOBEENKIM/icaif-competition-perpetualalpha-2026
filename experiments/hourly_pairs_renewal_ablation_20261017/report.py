"""Report fixed-maturity ablation and retain the useful renewal mechanism."""
import json
from pathlib import Path
from experiments.par_direct_forecast_20261017.data import ROOT, save, sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.netted_shadow_pairs_20261017.report import document, table, export_csv
from experiments.dense_history_pairs_20261017.report import grid

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'hourly_refresh_pairs_20261017'


def main():
    result=json.loads((HERE/'artifacts/results.json').read_text())
    original=json.loads((PRIOR/'artifacts/results.json').read_text())
    spec=json.loads((HERE/'protocol.json').read_text())
    for rel,digest in json.loads((HERE/'artifacts/frozen.json').read_text())['sources'].items():
        assert sha(ROOT/rel)==digest,rel
    guards=protected();out=ROOT/'reports/hourly_pairs_renewal_ablation_20261017';out.mkdir(exist_ok=True)
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in original['rows']}
    comparison=[];costs=[];deltas=[]
    for row in result['rows']:
        m=row['metrics'];old=lookup[row['venue'],row['phase'],spec['source_candidate'],row['scenario']]
        if row['venue']=='reference':
            comparison.append([row['phase'],row['scenario'],f"{old['net_pnl']:+,.2f}",f"{m['net_pnl']:+,.2f}",
                f"{old['financial_score']:.8f}",f"{m['financial_score']:.8f}",old['completed_pairs'],m['completed_pairs'],
                old['aborted_pairs'],m['aborted_pairs']])
        delta={k:m[k]-old[k] for k in ['net_pnl','price_move_pnl','execution_cost','fees','funding','base_owned_pnl','pair_owned_pnl']}
        assert abs(delta['net_pnl']-delta['price_move_pnl']+delta['execution_cost']+delta['fees']-delta['funding'])<1e-6
        assert abs(delta['net_pnl']-delta['base_owned_pnl']-delta['pair_owned_pnl'])<1e-6
        deltas.append(dict(venue=row['venue'],phase=row['phase'],scenario=row['scenario'],**delta))
        if row['venue']=='reference':
            costs.append([row['phase'],row['scenario'],*[f"{delta[k]:+,.2f}" for k in [
                'price_move_pnl','execution_cost','fees','funding','net_pnl']]])
    body='''<h1>매시간 페어 예측에서 보유 연장을 제거한 비교</h1>
<p class="note"><strong>보유 연장을 제거하면 거래 횟수는 늘었지만 비용 차감 성과가 크게 나빠졌다.
현재 후보의 보유 연장 기능은 유지한다.</strong>
새 계좌 재현 8개와 기존 엔진 대조 재현 2개를 완료했다. 기존 6시간 예측 대조군의 체결·계좌 경로·이벤트·성과는 정확히 일치했다.</p>
<h2>실험에서 바꾼 것</h2><p>24시간 오차 보정 모델의 매시간 예측, 최초 진입 조건, 수수료, 체결 제약,
기존 원전략, 초기자본 1,000,000과 위험 한도를 유지했다.
보유 연장과 그에 필요한 헤지 수량 조정만 비활성화했다. 최초 예측의 생성 시각+24시간에 청산을 목표로 삼고,
실제 청산 주문에도 지연·수량 제한·비용을 적용했다. 시간 봉 경계에서 청산 주문을 준비하는 기존 고정 만기 엔진을 그대로 사용했다.
체결된 순간부터 새로 24시간을 주거나 청산 실패 포지션을 사라지게 처리하지 않았다.</p>
<h2>기준시장 결과</h2><p>validation: 2026-08-01 12:00~09-01 12:00 UTC.
comparison: 09-01 12:00~10-15 12:00 UTC. 기존 여러 연구에서 사용한 과거 자료이며 독립 검증이 아니다.
base의 편도 수수료·가격 불이익은 9bp·10bp, stress는 12bp·20bp와 더 작은 참여율 및 펀딩 수령액 절반이다.</p>'''
    body+=grid(['기간','조건','보유 연장 PnL','고정 만기 PnL','보유 연장 Q','고정 만기 Q','연장 완료 페어','고정 완료 페어','연장 진입 후 철수','고정 진입 후 철수'],comparison)
    body+='''<p>이후 기본 조건에서 고정 만기는 −6,412.28, 보유 연장은 +1,491.90이었다.
stress에서는 각각 −5,689.66, +1,031.14였다.
기본 조건에서 완료 페어는 4→15건으로 늘었다. 이 추가 거래가 더 좋은 성과로 이어지지 않았다.
8월 stress는 두 조건이 정확히 같았다. 실제로 연장이 없었던 경로에서는 변화가 없다는 대조 결과다.</p>
<h2>왜 차이가 났는가</h2><p>아래는 고정 만기에서 보유 연장을 뺀 차이다. 실행 비용·수수료의 양수는 고정 만기에서 추가로 지불한 금액이다.
수량 경로가 바뀌므로 이후 진입 시각과 보유 가격 변화도 달라진다.
이 표는 전체 경로의 분해이며 각 연장 결정을 독립적인 인과 효과로 해석하지 않는다.</p>'''
    body+=grid(['기간','조건','가격 변화 손익 차이','추가 실행 비용','추가 수수료','펀딩 차이','계좌 순손익 차이'],costs)
    body+='''<p>이후 기본 조건의 악화 −7,904.18 중 추가 실행 비용 3,107.59와 수수료 2,781.48을 합친 금액은 5,889.07이다.
가격 변화 손익도 2,054.69 줄었다. stress 악화는 가격 변화 손익 감소 5,351.93의 영향이 더 컸다.
따라서 수수료만으로 모든 차이를 설명하지 않는다.</p>
<h2>실제 par 자료와 해석 한계</h2><p>같은 8개 중 par 캔들·펀딩을 사용한 4개에서는 성과 차이가 0이었다.
이 모형에서 연장할 규모의 페어 보유가 만들어지지 않았기 때문이다. 실제 대회 주문장에서 연장·청산의 동등성이 입증됐다는 뜻은 아니다.
앞선 실제 주문장 진입 검사는 24시간 뒤 호가가 없어 아직 전체 생애 수익 검증을 제공하지 못한다.</p>
<h2>현재 개발 판단</h2><p>후보의 과거 PnL·Q 개선과 수익 집중도 문제를 함께 보존한다.
최소 거래 건수를 통과하려고 매일 강제 청산하거나 거래를 늘리지 않는다.
다음 수정 후보는 새 예측의 크기와 남은 비용에 따라 기존 보유 규모를 유지·감축하는 의사결정이다.
현재 구조는 비용을 넘는 예측이면 큰 목표 수량을 유지하는 경향이 있어, 예측 강도 변화와 실제 위험 배분 사이를 검토할 가치가 있다.
아직 그 수량 관리 변경을 구현하거나 성능이 좋아졌다고 주장하지 않는다.</p>
<p>이 실험은 사전 선정 규칙을 바꾸지 않았고 전략을 실거래에 연결하지 않았다.
<a href="../hourly_pairs_robustness_20261017/report.html">앞선 수익 집중도·규모 대조 실험</a> 및
<a href="../hourly_refresh_pairs_20261017/report.html">매시간 예측 후보의 최초 비교</a>와 함께 판단해야 한다.</p>'''
    body+=table(result['rows'])
    body+=f"<p>실제 주문 0개. 활성 소스 {guards['active_protected_sources']}개·예약 의존 파일 {guards['reserved_dependencies']}개는 그대로다.</p>"
    (out/'report.html').write_text(document('페어 보유 연장 제거 실험',body));export_csv(out/'comparison.csv',result['rows'])
    save(out/'summary.json',dict(deltas=deltas,new_account_replays=8,exact_control_replays=2,
        decision='retain renewal; fixed original maturity is worse in later reference cases',
        orders_sent=0,protected=guards,historical_independence=False,automatic_live_replacement=False))
    save(HERE/'work_state.json',dict(
        status='progress_renewal_ablation_8_accounts_2_exact_controls_retain_renewal',
        previous_goal_turn='progress_hourly_bias24_concentration_and_actual_scale_controls_verified',
        report=str((out/'report.html').relative_to(ROOT)),orders_sent=0,live_strategy_changed=False,
        new_account_replays=8,exact_control_replays=2,historical_independence=False,
        findings=[
            'No-renewal hourly bias24 later base -6412.28 and stress -5689.66; renewal +1491.90 and +1031.14 respectively.',
            'No-renewal increases completed base pairs from four to fifteen, with +5889.07 execution/fee cost and -2054.69 price-path contribution versus renewal.',
            'Stress degradation is chiefly worse price-path contribution, not fees alone.',
            'par candle/funding outcomes are unchanged because meaningful renewable inventories were absent.',
        ],
        next='Keep hourly bias24 and paid renewal. Investigate a cost-aware retained-position sizing policy linked to fresh forecast magnitude and causal portfolio covariance, including opportunity cost of holding the existing gross amount and independent leg execution. Read the current actual renewal decision paths and forecast calibration before specifying a small frozen candidate family. Preserve model/control hashes and current live strategy; do not tune specific losing dates or force more trades.',
        remaining=['Concentrated profits and sparse stress completions','No uniform risk-normalized advantage',
                   'Full actual par lifecycle and independent prospective evidence unverified']))
    print(out/'report.html');print('Frozen sources and protected files unchanged.')


if __name__=='__main__':main()
