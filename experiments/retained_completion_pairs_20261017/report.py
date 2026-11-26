"""Report the frozen partial-reduction completion experiment without rerunning it."""
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.complementary_alpha_20261017.research import protected
from experiments.dense_history_pairs_20261017.report import grid
from experiments.netted_shadow_pairs_20261017.report import document, export_csv, table
from experiments.par_direct_forecast_20261017.data import ROOT, save, sha

HERE = Path(__file__).resolve().parent
ART = HERE / 'artifacts'


def main():
    results = json.loads((ART / 'results.json').read_text())
    paths = json.loads((ART / 'completion_paths.json').read_text())['rows']
    for rel, digest in json.loads((ART / 'frozen.json').read_text())['sources'].items():
        assert sha(ROOT / rel) == digest, rel
    guards = protected()
    rows = results['rows']
    new = [r for r in rows if '_utilitycomplete' in r['arm']]
    assert len(new) == 16 and len(rows) == 80
    assert len(results['exact_control_replays']) == 4
    assert sum(r['completion_submitted'] for r in new) == 2
    assert sum(r['completion_outcomes'].get('renewed', 0) for r in new) == 2
    unique = {(e['new_source'], tuple(e['pair'])) for r in paths for e in r['pending']}
    assert len(unique) == 1
    assert all(r['delta_pnl'] == 0 for r in new if r['venue'] == 'par')
    lookup = {(r['venue'], r['phase'], r['arm'], r['scenario']): r for r in rows}
    comparison = []
    for phase in ['validation', 'comparison']:
        for sc in ['base', 'stress']:
            for arm, label in [
                ('first_strategy', '기존 첫째 전략'),
                ('second_strategy', '기존 둘째 전략'),
                ('first_dense_hourlyhedge_bias_24', '첫째＋매시간 페어'),
                ('first_dense_utility64_bias_24', '첫째＋수량 조정 v1'),
                ('first_dense_utilitycomplete64_bias_24', '첫째＋부분 감축 완료 v2'),
            ]:
                m = lookup['reference', phase, arm, sc]['metrics']
                comparison.append([
                    phase, sc, label, f"{m['net_pnl']:+,.2f}",
                    f"{100*m['return_fraction']:+.5f}%", f"{m['financial_score']:.8f}",
                    f"{100*m['daily_sample_std']:.5f}%", f"{100*m['volatility_horizon']:.5f}%",
                    f"{100*m['max_drawdown']:.5f}%", m.get('completed_pairs', '—'),
                ])
    incremental = []
    for r in new:
        if r['venue'] != 'reference':
            continue
        m = r['metrics']
        incremental.append([
            r['phase'], r['arm'], r['scenario'], f"{m['net_pnl']:+,.2f}",
            f"{r['delta_pnl']:+,.2f}", f"{r['delta_hourly_pnl']:+,.2f}",
            r['completion_submitted'], r['completion_outcomes'].get('renewed', 0),
        ])
    event_rows = []
    for r in paths:
        for e in r['pending']:
            plan = e['completion_plan']
            outcome = next(x for x in r['outcomes'] if x['new_source'] == e['new_source'])
            event_rows.append([
                r['arm'], datetime.fromtimestamp(e['time'], timezone.utc).isoformat(),
                str(e['actual_qty']), str(e['original_target']), str(e['remaining_delta']),
                f"{plan['adjustment_cost_preview']:.4f}",
                f"{plan['utility_advantage']:.4f}", outcome['outcome'],
            ])
    body = '''<h1>페어 부분 감축 주문의 한 번 추가 실행: 검증 결과</h1>
<p class="note"><strong>부분 체결 처리 개선은 구현됐지만, 이전 매시간 페어 후보보다 일관된 성과 향상은 확인되지 않았다.</strong>
새 계좌 백테스트 16개, 기존 대조군 정확 재현 4개, 구현 검사 15개를 완료했다.
재사용 대조 결과 64개를 포함한 아래 수치는 실제 지갑 수익이나 실거래 주문이 아니다.</p>
<h2>이번 수정 범위</h2>
<p>매시간 갱신하는 24시간 예측, 진입 규칙, 원래 손실 기준과 보유 연장 조건을 유지했다.
위험 조정에 따른 수량 감축에서 한쪽만 체결되면, 현재 실제 보유 수량을 기준으로 남은 감축만 한 번 재평가한다.
원래 예측의 목표 가격·목표 수량은 바꾸지 않는다. 새 가격에서 예상 수익·펀딩·남은 주문 비용·청산 비용·계좌 전체 위험을 계산해,
즉시 유료 청산보다 현금 가치와 위험 조정 가치가 각각 보유 명목금액의 5bp 이상 유리할 때만 추가 주문한다.</p>
<p>첫 주문의 수수료는 이미 원장에 반영돼 두 대안의 공통 비용이다.
추가 주문의 지정가는 부분 체결을 확인한 뒤 내리는 새로운 판단이다. 최초 주문의 과거 체결을 소급해 바꾸지 않는다.
다음 실행 시점 안에만 완료할 수 있고, 이를 위해 원래 청산 기한을 연장하지 않는다.
추가 주문도 미체결·부분 체결이면 청산 상태를 유지한다. 결과 미확인이면 확인 전까지 중복 주문을 막는다.
동시에 들어온 원전략 주문 때문에 위험 한도를 넘는 경우에도 불균형 포지션의 청산을 요청한다.</p>
<h2>동일 자본 계좌 비교</h2>
<p>초기 자본 1,000,000. 기준시장 과거 가격을 이용한 계좌 재현이다.
validation은 2026-08-01 12:00~09-01 12:00 UTC,
comparison은 09-01 12:00~10-15 12:00 UTC다.
기본 조건은 편도 수수료 9bp, 가격 불이익 10bp이며,
stress는 12bp·20bp와 더 제한된 체결 수량, 펀딩 수령액 절반을 적용했다.
Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05), σ=완료된 일별 수익률의 표본 표준편차×√n이다.
표의 일별 표준편차와 기간 변동성을 구분했다. J=0 가정 점수는 0.8Q다.</p>
<p><strong>모두 반복 사용한 과거 자료의 개발 결과다.</strong> 선택 규칙은 이후 기간 실행 전에 고정했지만,
여러 후보에서 이미 사용한 기간이므로 독립적인 최종 검증으로 간주하지 않는다.</p>'''
    body += grid(['기간', '조건', '전략', '순손익', '수익률', 'Q', '일별 표준편차', '기간 변동성 σ', '최대낙폭', '완료 페어'], comparison)
    body += '''<h2>이번 수정의 추가 효과</h2>
<p>첫째 기반 후보의 8월 기본 순손익은 수량 조정 v1의 +16,793.19에서 +18,077.95로 1,284.76 증가했다.
그러나 더 이전 매시간 페어 후보의 +20,090.87보다 2,012.92 낮다.
이후 기간 기본은 +1,526.52로 v1과 같고, 매시간 후보보다 34.62 높다. stress에서는 이번 수정으로 인한 변화가 없다.
따라서 부분 체결 처리의 손실을 일부 회복했지만 이 수량 조정 방식을 전체 후보의 우승자로 채택하지 않는다.</p>'''
    body += grid(['기간', '후보', '조건', '순손익', 'v1 대비', '매시간 페어 대비', '추가 주문', '완료'], incremental)
    body += '''<p>첫째 기반 후보는 두 기간·두 비용 조건에서 기존 첫째·둘째 전략의 PnL과 Q를 모두 넘었다.
다만 8월 stress의 완료 페어가 1개여서 사전에 둔 연구 선정 기준인 최소 3개를 통과하지 못했다.
selection.eligible=false는 연구 후보 선정 기준이다. 대회 점수 집계 자격과 다른 값이다.
앞서 확인한 특정 거래에 대한 수익 집중과 위험 규모에 대한 민감성도 해소됐다고 볼 수 없다.</p>
<h2>한쪽 체결 이후의 경로</h2>
<p>BTC 0.6205, HYPE −297.33이 남은 상태에서 HYPE 146.41개만 매수해
원래 목표인 BTC 0.6205, HYPE −150.92로 감축을 마쳤다.
두 계좌 변형에서 같은 시장 사건을 재현했으므로 독립적인 성공 기회 두 건으로 세지 않는다.</p>'''
    body += grid(['후보', '확인·재판단 UTC', '부분 체결 후 수량', '원래 목표', '남은 변경 수량', '예상 추가 주문 비용', '위험 조정 가치 차이', '결과'], event_rows)
    body += '''<p>이후 다른 부분 감축은 남은 비용 차감 가치가 부족해 추가 주문을 거부하고 청산했다.
추가 주문을 허용하면 언제나 이익이 된다는 결과는 아니다. 표의 체결은 백테스트 실행 모형에서 확인된 체결이다.</p>
<h2>실제 par 자료에서 확인된 범위</h2>
<p>보유한 par 캔들·펀딩 자료의 8개 새 계좌 재현에서는 추가 감축 완료가 발생하지 않아 v1과 결과가 같다.
짧은 실제 호가 기록만으로 24시간 뒤 보유 조정과 최종 청산까지 검증할 수는 없다.
부분 체결·미확인 주문 상태 검사는 합성 입력을 포함한 실행 검사이며, 실제 par 장기간 수익 검증으로 취급하지 않는다.</p>
<h2>검사와 다음 개발 판단</h2>
<p>미래 자료 변경 시 이전 판단 불변, 실제 현금·수량 보존, 미확인 결과의 중복 방지,
추가 주문 미체결·두 번째 부분 체결·기한 초과 처리, 원전략 동시 주문과의 합산 위험,
이전 대조군 정확 재현을 포함한 15개 검사를 통과했다.
구현 검사 통과는 수익성 보장을 뜻하지 않는다.</p>
<p>다음 성능 연구는 이미 시험한 모델·앙상블을 먼저 대조한 뒤 예측 오차와 거래 손익의 관계를 개선하는 데 둔다.
특히 아직 정답이 확정되지 않은 예측을 학습에 쓰지 않고, 같은 예측 기간의 모델만 비교해야 한다.
이번 보고서는 새 예측 앙상블이 구현됐거나 개선됐다고 주장하지 않는다.</p>'''
    body += table(rows)
    body += f"<p>실제 주문 전송 0개, 실거래 코드 변경 없음. 활성 소스 {guards['active_protected_sources']}개와 예약 의존 파일 {guards['reserved_dependencies']}개의 해시가 그대로임을 확인했다.</p>"
    out = ROOT / 'reports/retained_completion_pairs_20261017'
    out.mkdir(exist_ok=True)
    (out / 'report.html').write_text(document('페어 부분 감축 완료 검증', body))
    export_csv(out / 'comparison.csv', rows)
    summary = dict(selection=results['selection'], new_account_replays=16,
                   exact_control_replays=4, reused_control_accounts=64, tests_passed=15,
                   completion_account_paths=2, unique_completion_market_events=1,
                   protected=guards, orders_sent=0, historical_independence=False,
                   automatic_live_replacement=False)
    save(out / 'summary.json', summary)
    save(HERE / 'work_state.json', dict(
        **summary,
        status='progress_partial_reduction_completion_16_accounts_15_tests_no_uniform_incremental_gain',
        report=str((out / 'report.html').relative_to(ROOT)), live_strategy_changed=False,
        findings=[
            'Completion works for one overlapping August event in two baseline variants; other exhausted-edge reductions remain paid unwinds.',
            'First-based August base recovers 1284.76 over utility v1, but remains 2012.92 below hourly bias24; later and stress do not improve over v1.',
            'First-based candidate beats both native strategies on PnL/Q in all four reused reference cells; stress August has only one completed pair, failing frozen research eligibility.',
            'Actual par candle/funding replays add no completions and show no incremental gain; full actual par lifecycle remains unverified.',
        ],
        next='Review existing forecast and ensemble experiments and primary sources before freezing a distinct forecast-level candidate. Use only matured, originally issued predictions for causal same-horizon expert calibration; do not retune on later reused results or claim independent validation. Retain hourly bias24 as an explicit control; do not alter live or frozen source files.',
        remaining=['Robust incremental performance beyond hourly bias24 not established',
                   'Trade concentration and risk-size sensitivity remain',
                   'Full actual par lifecycle and independent prospective evidence unverified'],
    ))
    print(out / 'report.html')
    print('Frozen and protected sources verified unchanged.')


if __name__ == '__main__':
    main()
