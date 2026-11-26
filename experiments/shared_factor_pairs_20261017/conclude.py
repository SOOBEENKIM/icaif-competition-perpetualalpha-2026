"""Retain the improved benchmark; summarize this completed, failed successor."""
import json
import time
from pathlib import Path
from .research import ROOT, HERE, OUT, ARMS, MATCHED, verify, save, sha


def main():
    result = json.loads((OUT / 'results.json').read_text())
    diagnostics = json.loads((OUT / 'diagnostics.json').read_text())
    books = json.loads((OUT / 'books/results.json').read_text())
    lookup = {(r['venue'], r['phase'], r['arm'], r['scenario']): r['metrics'] for r in result['rows']}
    differences = []
    for phase in ['validation', 'comparison']:
        for sc in ['base', 'stress']:
            new = lookup['reference', phase, ARMS['combined'], sc]
            old = lookup['reference', phase, MATCHED, sc]
            differences.append(dict(phase=phase, scenario=sc, candidate_pnl=new['net_pnl'], retained_pnl=old['net_pnl'],
                pnl_difference=new['net_pnl']-old['net_pnl'], candidate_q=new['financial_score'], retained_q=old['financial_score'],
                q_difference=new['financial_score']-old['financial_score']))
    episode = next(r for r in diagnostics['entry_source_labels'] if r['phase']=='comparison' and r['model']=='combined' and r['scenario']=='stress')
    assert episode['source_prediction_bps'] > 0 > episode['source_realized_24h_bps']
    for path in ['frozen.json', 'books/frozen.json', 'diagnostics_frozen.json', 'render_recovery_frozen.json']:
        verify(json.loads((OUT / path).read_text())['sources'])
    verify(json.loads((ROOT / 'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'])
    conclusion = dict(at_unix=time.time(), previous_goal_turn_classification='progress_development_status_clarified',
        classification='progress_shared_factor_model_and_comparisons_completed',
        retained_development_benchmark=MATCHED, retained_reference_improvement_still_valid=True,
        successor=ARMS['combined'], replace_retained=False,
        reason='Small later base-case gain, but earlier August stress PnL/Q below retained and later stress loss. Do not promote diagnostic-only arms or erase prior benchmark improvement.',
        reference_differences=differences, selection=result['selection'],
        stress_entry_diagnostic=episode,
        evidence_limits='Reused historical development periods. Source 24h label is not actual delayed-fill/renewed-exit PnL. Old disconnected par books show no accepted entry and do not cover full 24h exits. Internal sample-count flags do not prohibit competition submission.',
        new_account_replays=36, exact_control_replays=12, model_tests_passed=9,
        candidate_book_observations=408, exact_control_book_observations=books['exact_control_bound_observations'],
        reporting_recovery='Research exit 1 occurred only during HTML rendering of numpy.bool_ after all accounts and results were saved. render_saved exited 0 after verifying all 36 persisted accounts; no rerun or source alteration.',
        next_action='Continue from retained cash_direct. Audit how its squared-error forecast fit aligns with cost-gated entry and paid holding/exit outcomes; locate previous cost-sensitive experiments before proposing a new economic-loss model. Do not retune thresholds on the September 23 loss.',
        orders_sent=0, live_strategy_changed=False, reserved_raw_read=False, historical_independence=False,
        goal_complete=False, no_progress_streak=0)
    save(OUT / 'conclusion.json', conclusion)
    report = ROOT / 'reports/shared_factor_pairs_20261017/report.html'
    page = report.read_text()
    page += '''<h2>판정: 기존 개선판 유지</h2><p>새 combined 후보의 기준시장 Sep1–Oct15 기본 순손익은 +3557.7150으로 cash_direct의 +3422.1612보다 +135.5538 높지만, 불리한 체결 조건에서는 -2484.9002로 기존 +3305.7125보다 -5790.6127 나빴다. 8월에서도 불리한 조건의 PnL/Q는 기존 개선판보다 낮았다. 따라서 이번 변경으로 교체하지 않는다. 기존 개선판이 첫째·둘째보다 높았던 기준시장 백테스트 성과는 그대로 유효하다.</p>
<p>새 후보의 불리한 조건에서는 9월 18일의 수익 거래가 빠지고 9월 23일 BTC/HYPE 거래에 진입했다. 그 진입 원천 예측은 +84.65bp이나 같은 시점부터 24시간 뒤 고정 헤지 가격차 변화는 -141.03bp였다. 이 값은 실제 지연 체결·보유 연장 손익과 구분한다. 기록된 전체 거래의 페어 가격 손익 -1280.7782에 체결 비용·수수료·펀딩까지 반영하면 페어 순손익은 -2063.1711이다. 비용 증가만으로 설명되는 악화가 아니다.</p>
<p>분해 후 BTC와 잔차 예측의 비교 기간 오차도 각각 190.6983bp/55.0069bp이며 0 예측은 182.8175bp/53.6349bp다. 이는 추가 정보나 예측 개선을 확인하지 못했다는 진단이지 공식 제출 금지 기준이 아니다.</p>
<p>허용된 기존 par 호가 68방향/비용조건에서 combined의 진입은 0건이다. 무제한 최우선 호가 체결이라는 낙관적 가정에서도 가장 좋은 비용 차감 기대값은 기본 -2.5643bp, 불리한 조건 -18.5637bp였다. 이 짧고 끊긴 관측으로 시장 전체나 실제 24시간 거래 성과를 단정하지 않는다.</p>
<p>모델 검사 9개, 새 모의 계좌 36개, 기존 계좌 12개 재현을 완료했다. HTML의 NumPy 참/거짓 직렬화 오류만 저장된 JSON을 다시 읽어 복구했으며 모델·선정·계좌 결과는 바꾸거나 재실행하지 않았다. 실거래 주문 및 전략 교체는 없다.</p>'''
    report.write_text(page)
    dest = ROOT / 'experiments/pairs_development_status_20261017/comparisons'
    dest.mkdir(exist_ok=True)
    save(dest / 'shared_factor.json', conclusion)
    save(HERE / 'work_state.json', dict(status=conclusion['classification'], report=str(report.relative_to(ROOT)),
        retained_development_benchmark=MATCHED, replace_retained=False, new_account_replays=36, exact_controls=12,
        tests_passed=9, orders_sent=0, live_strategy_changed=False, reserved_raw_read=False,
        historical_independence=False, goal_complete=False, no_progress_streak=0, next_action=conclusion['next_action']))
    paths = [Path(__file__), OUT / 'conclusion.json', report, dest / 'shared_factor.json']
    save(OUT / 'conclusion_frozen.json', dict(at_unix=time.time(), sources={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    print('DONE: retained improved candidate, no new live strategy', flush=True)


if __name__ == '__main__': main()
