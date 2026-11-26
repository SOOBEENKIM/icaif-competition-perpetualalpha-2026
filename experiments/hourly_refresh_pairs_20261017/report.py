"""Report measured hourly-inference results without revising frozen selection."""
import json
from pathlib import Path

from experiments.par_direct_forecast_20261017.data import ROOT, save, sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.netted_shadow_pairs_20261017.report import document, table, export_csv
from experiments.dense_history_pairs_20261017.report import grid

HERE = Path(__file__).resolve().parent
ART = HERE / 'artifacts'
CHOSEN = 'first_dense_hourlyhedge_bias_24'


def main():
    r = json.loads((ART / 'results.json').read_text())
    a = json.loads((ART / 'attribution.json').read_text())
    b = json.loads((ART / 'books/results.json').read_text())
    edge = json.loads((ART / 'books/edge_attribution.json').read_text())
    for path in [ART / 'frozen.json', ART / 'books/frozen.json', HERE / 'prepared/frozen.json']:
        for rel, digest in json.loads(path.read_text())['sources'].items():
            assert sha(ROOT / rel) == digest, rel
    assert r['selection']['selected'] == CHOSEN
    guards = protected()
    out = ROOT / 'reports/hourly_refresh_pairs_20261017'
    out.mkdir(exist_ok=True)
    lookup = {(x['venue'], x['phase'], x['arm'], x['scenario']): x['metrics'] for x in r['rows']}
    rows = []
    checks = []
    episode_rows = []
    for phase in ['validation', 'comparison']:
        for sc in ['base', 'stress']:
            m = lookup['reference', phase, CHOSEN, sc]
            failures = []
            if m['completed_pairs'] < 3:
                failures.append('완료 페어 3건 미만')
            for name in ['first_strategy', 'second_strategy']:
                base = lookup['reference', phase, name, sc]
                assert m['net_pnl'] > base['net_pnl']
                assert m['financial_score'] > base['financial_score']
            checks.append(dict(phase=phase, scenario=sc, beats_both_native_pnl_q=True,
                               completed_pairs=m['completed_pairs'], diagnostic_failures=failures,
                               volatility_above_both_natives=all(m['volatility_horizon'] > lookup['reference', phase, n, sc]['volatility_horizon'] for n in ['first_strategy', 'second_strategy'])))
            for name in ['first_strategy', 'second_strategy', CHOSEN]:
                v = lookup['reference', phase, name, sc]
                rows.append([phase, sc, name, f"{v['net_pnl']:+,.2f}",
                             f"{100*v['return_fraction']:+.6f}", f"{v['financial_score']:.8f}",
                             f"{100*v['volatility_horizon']:.6f}", f"{100*v['max_drawdown']:.6f}",
                             v['completed_pairs']])
            owned = next(x for x in a['rows'] if (x['venue'], x['phase'], x['arm'], x['scenario']) == ('reference', phase, CHOSEN, sc))
            closed = sum(e['net_pnl'] for e in owned['episodes'] if e['closed'])
            opened = sum(e['net_pnl'] for e in owned['episodes'] if not e['closed'])
            assert abs(closed + opened - m['pair_owned_pnl']) < 1e-6
            episode_rows.append([phase, sc, f"{m['base_owned_pnl']:+,.2f}",
                                 f"{m['pair_owned_pnl']:+,.2f}", f"{closed:+,.2f}", f"{opened:+,.2f}",
                                 f"{m['pnl_after_liquidation_reserve']:+,.2f}", m['aborted_pairs']])
    s = edge['summary']
    body = '''<h1>페어 예측 매시간 갱신: 기존 두 전략과 비용 차감 후 비교</h1>
<p class="note"><strong>24시간 오차 보정 후보는 두 과거 기간의 기본·불리한 조건 모두에서 기존 두 전략보다 PnL과 Q가 높았다.
그러나 완료 거래가 적고 변동성이 커져, 사전에 고정한 교체 후보 선정 기준은 통과하지 못했다.</strong>
새 계좌 재현 128개, 대조군 정확 일치 확인 4개, 재사용 대조군 528개를 비교했다.
이번 수정의 인과성·시간 정렬 검사 5개를 통과했고, 기존 6시간 예측 16종도 정확히 재현했다.</p>
<h2>무엇을 수정했나</h2><p>BTC/ETH와 BTC/HYPE 상대가격의 24·72시간 예측을 완성된 시간봉마다 갱신한다.
학습 가중치는 그대로 유지하고, 이미 만기가 끝난 관측만 최근 오차 보정에 사용한다.
후보는 원래 릿지, 평균 오차 보정, 절편·기울기 보정, 최근 평균 예측이다.
진입할 때 최대 1시간 전 예측과 현재 가격·비용을 비교한다. 예측 만기는 원래 생성 시각에서 계산한다.
보유 중 헤지 수량 조정에는 비용을 지불하며 양쪽 주문을 독립적으로 처리한다.
6시간 보유 연장 판단 규칙·계좌 자본·위험 한도는 직전 후보와 같다.</p>
<h2>기존 두 전략과 비교</h2><p>초기자본 1,000,000. validation: 2026-08-01 12:00~09-01 12:00 UTC.
comparison: 09-01 12:00~10-15 12:00 UTC. 모두 재사용된 기준시장 과거 자료다.
base: 편도 수수료 9bp, 불리한 가격 10bp. stress: 12bp·20bp, 더 작은 거래량 참여율, 펀딩 수령액 절반.
아래는 실제 par 지갑 수익이 아니라 체결 가정을 적용한 계좌 백테스트다.
기간 σ는 완료 일별 수익률의 표본 표준편차×√n이고, Q=(max(R,0)+0.005)/(abs(R)+1.5σ+0.05)다.</p>'''
    body += grid(['기간', '비용 조건', '전략', '순손익', '수익률 %', 'Q', '기간 σ %', '최대낙폭 %', '완료 페어'], rows)
    body += '''<p>24시간 오차 보정 후보(first_dense_hourlyhedge_bias_24)는 8월 기본 +20,090.87, stress +13,089.43이었다.
이후 기간 기본 +1,491.90, stress +1,031.14로 두 원전략의 PnL·Q를 모두 넘었다.
이는 관측된 개선이며, 개선이 전혀 없었다고 해석하지 않는다.</p>
<p>반면 8월 stress 완료 페어는 1건, 이후 stress는 2건이다. 사전에 정한 최소 3건 기준에 미달한다.
이 최소 건수 자체가 통계적 신뢰성을 보장하는 기준은 아니다. 모든 조건에서 기간 변동성이 두 원전략보다 높았다.
8월 결과로 고정한 후보 선택은 eligible=false로 보존했고, 이후 결과를 보고 선정 조건을 바꾸지 않았다.
이는 대회 참가 자격을 뜻하는 eligible와 구별되는 연구 후보 선정 여부다.</p>
<h2>손익의 출처와 남아 있는 포지션</h2><p>원전략과 페어가 하나의 실제 계좌를 공유하는 모형이다.
소유 전략별 수량·비용·현금·미실현손익 합계가 물리 계좌와 일치하는지 검사했다.
아래의 종료된 에피소드 손익에는 진입 후 철수도 포함된다. 미종료 손익은 실현 수익으로 표시하지 않는다.
마지막 열 직전의 청산비용 차감 값은 남은 수량의 추정 비용을 추가 차감한 진단이며, 미래 실제 청산 보장은 아니다.</p>'''
    body += grid(['기간', '비용', '원전략 소유 손익', '페어 소유 손익', '종료 에피소드 손익',
                  '미종료 평가손익', '계좌 손익−추정 잔여 청산비용', '진입 후 철수'], episode_rows)
    body += f'''<h2>실제 par 주문장 진입 검사</h2><p>기존 허용 호가 기록 다섯 개, 약 4.48시간을 사용했다.
모델·비용 조건별 진입 시뮬레이션 {len(b['trials'])}개에서 양쪽 보유 {s['known_paired_entry_probes']}개,
미체결 {b['aggregate_statuses'].get('entry_zero_fill', 0)}개였다. 고유 기록·페어·판단시각·방향 조합은
{s['unique_tape_pair_decision_directions']}개이며 서로 겹치므로 독립 거래 수로 해석하지 않는다.
수량과 가격은 실제 도착 주문장에서 읽고, 1초/3초 후 양쪽 체결을 각각 처리했다.</p>
<p>새 예측의 나이는 {s['source_age_min_hours']}~{s['source_age_max_hours']}시간이다.
양쪽 체결 {s['known_paired_entry_probes']}조건 중 {s['positive_above_buffer_without_reference_age']}조건은
과거 기준가격으로 돌아가는 성분을 빼도 비용과 진입 버퍼를 넘었다.
모델 예측 증분을 빼면 버퍼를 넘는 조건은 {s['positive_above_buffer_without_model']}개였다.
직전 6시간 유효기간 실험은 35개 중 14개가 기준가격 경과분 없이 버퍼를 넘었다.
조건 집합이 달라 이 비율 차이를 독립적인 성능 개선 검정으로 보지는 않는다.</p>
<p><strong>24·72시간 뒤 실제 청산 호가가 없으므로 이 진입 검사의 수익률과 Q는 산출하지 않는다.</strong>
par 과거 캔들·펀딩 계좌 모형의 이후 비교 구간에서는 이 후보의 추가 체결이 없어 기존 원전략과 같았다.
앞선 par 구간 기본 조건에서는 한 번의 진입 후 철수로 원전략보다 손익이 약 0.39 높았고, stress는 같았다.
이는 캔들 거래량으로 가정한 체결 가능성과 실제 주문장 진입 가능성이 다름을 보여준다.
짧은 주문장 검사 결과를 긴 기간의 실제 거래 수익으로 확대하지 않는다.</p>
<h2>현재 판단</h2><p>매시간 예측 갱신과 최근 오차 보정은 유지할 연구 후보로 남긴다.
다음 판단은 이 후보의 비용 차감 손익이 특정 에피소드나 미실현손익에 얼마나 집중되는지,
같은 위험 수준에서도 원전략보다 나은지를 확인하는 것이다. 이 결과를 보고 규칙을 바꾸면 다음 수정판으로 따로 기록한다.
여러 번 사용한 이 과거 자료의 결과를 독립적인 미래 검증으로 부르지 않는다.</p>
<p>실제 주문 전송 0개. 실거래 전략 자동 교체 없음.
활성 소스 {guards['active_protected_sources']}개·예약 의존 파일 {guards['reserved_dependencies']}개의 해시는 유지했다.</p>'''
    for venue in ['reference', 'par']:
        body += '<h2>' + ('기준시장 전체 비교' if venue == 'reference' else 'par 캔들·펀딩 모형 전체 비교') + '</h2>'
        body += table([x for x in r['rows'] if x['venue'] == venue])
    (out / 'report.html').write_text(document('매시간 페어 예측 개선 결과', body))
    export_csv(out / 'comparison.csv', r['rows'])
    summary = dict(selection=r['selection'], native_comparisons=checks, new_tests_passed=5,
                   six_hour_model_controls_exact=16, new_candidate_replays=128,
                   control_verification_replays=4, reused_control_accounts=528,
                   book_entry_probes=len(b['trials']), book_statuses=b['aggregate_statuses'],
                   book_edge_attribution=s, book_account_return=None, book_financial_score=None,
                   maximum_episode_accounting_error=a['maximum_episode_accounting_error'],
                   protected=guards, orders_sent=0, historical_independence=False,
                   automatic_live_replacement=False)
    save(out / 'summary.json', summary)
    save(HERE / 'work_state.json', dict(
        status='progress_hourly_source_bias24_improves_native_pnl_q_four_cells_but_sparse_stress_episodes_and_higher_risk',
        report=str((out / 'report.html').relative_to(ROOT)), authoritative_results='artifacts',
        **summary,
        findings=[
            'The frozen 24h bias-corrected hourly candidate beats both native strategies on PnL and Q in both historical periods and both cost scenarios.',
            'August stress has one completed pair; later stress has two. Frozen selection remains ineligible. This is not a claim of no historical improvement.',
            'Period volatility exceeds both natives in all four comparisons; same limits do not mean equal realized risk.',
            '31 old-book entry probes yield 30 paired holdings, one zero fill, and seven overlapping unique entry groups; no full-maturity exit or account ROI/Q is observed.',
            '23 of 30 paired-entry forecasts exceed the buffer without source-reference aging; zero do without model increment. This is forecast attribution, not realized alpha.',
            'Later par candle-volume replay adds no trades for bias24; earlier base has one aborted entry and only +0.39 improvement. Actual par whole-lifecycle profitability remains unverified.',
        ],
        next='Preserve hourly bias24 as a research candidate. Diagnose its episode concentration, closed versus marked-open PnL and risk-normalized native comparison before another strategy mutation. Keep the frozen selection and all results; no reserved future raw data or live changes.',
        remaining=['Sparse completed stress episodes', 'Higher realized risk than natives',
                   'Full par lifecycle profitability unverified', 'Independent prospective evidence absent']))
    print(json.dumps(summary, ensure_ascii=False))
    print(out / 'report.html')


if __name__ == '__main__':
    main()
