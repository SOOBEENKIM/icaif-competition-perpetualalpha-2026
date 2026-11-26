"""Seal the gate ablation, preserving the already improved submission candidate."""
import json
import time
from pathlib import Path
from html import escape
from html.parser import HTMLParser

from .research import ROOT, HERE, OUT, save, sha, verify, protected


def main():
    result = json.loads((OUT / 'results.json').read_text())
    audit = json.loads((OUT / 'audit.json').read_text())
    books = json.loads((OUT / 'books/results.json').read_text())
    spec = json.loads((HERE / 'protocol.json').read_text())
    sources = json.loads((OUT / 'audit_frozen.json').read_text())['sources'].copy()
    sources.update(json.loads((OUT / 'books/frozen.json').read_text())['sources'])
    prospective = json.loads((ROOT / 'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources']
    verify(sources); verify(prospective)
    assert not result['selection']['promote_direct']
    assert result['selection']['selected'] == spec['retained']
    assert len(result['exact_controls']) == 28 and audit['accounts_verified'] == 28
    assert books['exact_control_summaries'] == 10
    assert audit['risk_breach_bars'] == 0
    next_action = ('Implement an additive publication-to-received-book adapter for the retained two-pair decision gate. '
        'Preserve direct mean/terminal prices and use the learned gate only for new entries, never renewal or exit. '
        'Verify forecast parity, strict label maturity, receipt latency and missing-midnight behavior using allowed old '
        'data before registering any new prospective arm. Preserve the original sealed four-arm plan, reserved raw '
        'and live services. A later-start sidecar may compare first, second, ungated and gated with identical books; '
        'do not call it preregistered until its code/tests and future start are frozen.')
    conclusion = dict(at_unix=time.time(), classification='progress_gate_ablation_completed_and_accounted',
        previous_user_turn_classification='no_progress_status_and_submission_eligibility_clarification_only',
        retained_improved_candidate=spec['retained'], selection=result['selection'],
        new_account_replays=28, exact_controls=28, audited_accounts=28,
        new_model_fits=0, reused_queue_tests=12, gate_effects=audit['gate_effects'],
        old_book_candidate_statuses=books['statuses'], exact_book_control_summaries=10,
        full_strategy_quote_pnl=None, full_strategy_quote_q=None,
        next_action=next_action, protected=protected(), prospective_sources=len(prospective),
        orders_sent=0, live_strategy_changed=False, reserved_raw_read=False,
        historical_independence=False, goal_complete=False, no_progress_streak=0)
    save(OUT / 'conclusion.json', conclusion)
    report = ROOT / 'reports/graph_filter_ablation_20261018/report.html'
    report.parent.mkdir(exist_ok=False)
    page = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>페어 진입 선별 모델 기여 검증</title>',
        '<style>body{max-width:1160px;margin:36px auto;padding:0 18px;font:16px/1.65 sans-serif;color:#18263b}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{border:1px solid #cbd5e1;padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}.note{background:#eef4fc;padding:16px}</style>',
        '<h1>같은 페어·예측·주문 구조에서 진입 선별 모델의 기여</h1>',
        '<p class="note"><b>진입 선별 모델을 유지한다.</b> 고정 그래프와 잔량 처리 구조에서, 선별 모델을 제거하면 8월 기본 조건의 순손익이 22,028.45에서 14,237.79로, 9월–10월은 3,837.08에서 3,406.47로 낮아졌다. 불리한 조건에서는 동일했다. 첫째·둘째보다 개선된 기존 두 페어 후보를 보존한다.</p>',
        '<p>이 실험은 새로운 모델을 훈련하지 않는다. 기존 직접 수익률 예측, 베타, 위험, 가격, 규모, 비용을 유지하고 신규 진입의 학습 필터만 제거했다. 고정 페어는 BTC/ETH, BTC/HYPE, SP500/EWY, EWY/EWJ다. 잔량 분리 사용 여부도 교차 비교했다. 8월 결과로 먼저 선정 결정을 저장한 뒤 나머지 구간을 실행했으며, 모든 기간은 이미 사용한 개발 자료다.</p>',
        '<table><tr><th>구간 / 비용</th><th>선별 제거·잔량 분리 PnL</th><th>선별 유지·잔량 분리 PnL</th><th>선별 기여</th><th>Q 기여</th></tr>']
    lookup = {(r['venue'], r['phase'], r['arm'], r['scenario']): r['metrics'] for r in result['rows']}
    for row in audit['gate_effects']:
        if row['venue'] != 'reference': continue
        key = ('reference', row['phase'])
        direct = lookup[*key, spec['candidate'], row['scenario']]
        gated = lookup[*key, spec['queued_control'], row['scenario']]
        page.append(f"<tr><td>{escape(row['phase'])} / {row['scenario']}</td><td>{direct['net_pnl']:+,.2f}</td><td>{gated['net_pnl']:+,.2f}</td><td>{row['gated_minus_direct_pnl']:+,.2f}</td><td>{row['gated_minus_direct_q']:+.6f}</td></tr>")
    page += ['</table><p>validation: 8월 1일–9월 1일, comparison: 9월 1일–10월 15일, continuous: 8월 1일–10월 15일. 경계는 12:00 UTC이며 초기 자본은 각각 1,000,000이다. 수수료·모의 체결 비용·펀딩을 포함한 평가손익이고 종료 시 미청산 포지션을 포함한다. 겹치는 기간을 독립적인 반복 성공으로 세지 않는다.</p>',
        '<h2>어디서 차이가 발생했는가</h2><p>9월–10월 기본 조건에서 선별 모델의 계좌 순손익 기여는 +430.610593이다. BTC/HYPE 페어 기여 +429.768227과 기본 전략 +0.842366으로 대조된다. 이 페어의 수수료는 오히려 45.500416 늘었다. 따라서 단순한 거래 축소나 수수료 절감만으로 성과가 좋아졌다고 설명할 수 없다. 같은 기간 EWY/EWJ의 손익은 변하지 않았다.</p>',
        '<p>8월 기본 조건에서는 BTC/HYPE 기여 +8,533.102630, BTC/ETH 기여 −785.743256이다. 모든 페어에서 좋아진 것은 아니다. 불리한 체결 조건에서는 선별 유무의 계좌 경로 차이가 없었다. 두 구간 모두에서 불리한 조건까지 추가 개선했다고 주장하지 않는다.</p>',
        '<h2>실제 과거 par 호가 검사</h2><p>동일한 5개 과거 호가 구간에서 기존 그래프 통제의 10개 요약을 정확히 재현했다. 선별 유지와 제거 각각 4건이 양쪽 진입 후 보유 상태에 도달했다. 24시간 전략 청산은 관측되지 않았으므로 완결 거래 수익률과 Q는 미측정이다. 짧은 호가 검사를 장기 수익 증거로 쓰지 않았다.</p>',
        '<h2>검사와 다음 작업</h2><p>새 계좌 28개, 기존 통제 28개를 비교했다. 새 계좌의 수량·소유 원장·거래별 손익·수수료·펀딩, 잔량 감축 및 종목 중복 금지, 완료 거래 수를 감사했다. 위험 한도 초과는 0건이다. 신호의 공통 필드가 정확히 같고 제외된 페어가 새 진입에 사용되지 않는지도 확인했다.</p>',
        '<p>다음은 유지한 두 페어 개선 후보의 실제 수신 시각 기준 신호·체결 연결이다. 신규 진입에만 필터를 쓰고 보유·연장·청산에는 원래 예측을 유지하는지 검증한다. 기존 미래 검증 계획과 미공개 원자료를 보존하며, 새 평가를 등록할 경우 구현과 검사 이후의 시작 시각으로 별도 고정한다.</p>',
        '<p>실거래 주문 또는 기존 운용 전략의 변경은 없다. 이 보고서는 기준시장 백테스트에서 이미 확인한 개선을 취소하거나 제출 불가로 판정하는 문서가 아니다.</p></html>']
    report.write_text('\n'.join(page)); HTMLParser().feed(report.read_text())
    save(HERE / 'work_state.json', dict(status=conclusion['classification'],
        report=str(report.relative_to(ROOT)), next_action=next_action,
        retained_improved_candidate=spec['retained'], orders_sent=0,
        live_strategy_changed=False, reserved_raw_read=False, goal_complete=False, no_progress_streak=0))
    paths = [p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.log'] + [report]
    sources.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    verify(sources); verify(prospective)
    save(OUT / 'conclusion_frozen.json', dict(at_unix=time.time(), sources=sources,
        prospective_sources=len(prospective), protected=protected(), excluded='mutable runner .log files'))
    print('SEALED', len(sources), 'sources; retained', spec['retained'], flush=True)


if __name__ == '__main__':
    main()
