"""Record exact unchanged performance and the corrected liquidation diagnosis."""
import json
import time
from collections import Counter
from pathlib import Path
from html.parser import HTMLParser
import numpy as np
from .research import ROOT,HERE,OUT,save,sha,verify,protected


def main():
    result=json.loads((OUT/'results.json').read_text());audit=json.loads((OUT/'audit.json').read_text())
    spec=json.loads((HERE/'protocol.json').read_text())
    sources=json.loads((OUT/'audit_frozen.json').read_text())['sources'].copy();verify(sources)
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(prospective)
    assert audit['all_accounts_unchanged'] and audit['requests']==0
    assert len(result['exact_controls'])==28 and result['new_account_replays']==28
    reasons=Counter()
    for row in audit['account_paths']:reasons.update(row['reasons'])
    diagnosis=audit['refined_liquidation_diagnosis']
    data_path=ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz'
    with np.load(data_path) as z:
        i=int(np.searchsorted(z['market_times'],diagnosis['rival_entry_time']))
        assert int(z['market_times'][i])==diagnosis['rival_entry_time']
        residual_gross=float(abs(np.array(diagnosis['held_qty_at_rival']))@z['close'][i])
    next_design=dict(
      objective='Allow a new disjoint pair opportunity while residual inventory from an already-requested exit keeps liquidating, with the same physical account and global risk limits.',
      accounting='Separate active-pair ownership from residual-liquidation ownership. Transfers of ownership conserve physical cash/positions and incur no fictitious exchange fill, fee or volume. Physical execution, total fees and effective funding must sum exactly across owners.',
      residual='Old exits continue; residual quantity can only fall toward zero. No reversal, increase or cancellation of required liquidation. Unknown order outcomes still block new submissions until reconciled.',
      entries='Existing forecasts and economic gates only. New pair symbols must not overlap any residual symbol. Check combined physical exposure, pending orders and each independent close/new-leg outcome against unchanged limits. No extra starting capital or risk ceilings.',
      scope='Run both fixed two-pair and frozen graph universes. A single active paired trade plus a reduction-only residual queue; do not pretend the portfolio is flat or count residual as completed.',
      attribution='Episode attribution must follow owners rather than assuming all pair cash belongs to the latest entry. Existing account field equality remains required when feature is disabled.',
      evaluation='Freeze before replay; compare August first and save selection, then later/continuous reference and actual par data under the same base/stress assumptions. Treat all reused history as development. Verify received-book accounting and partial closure; no invented missing future quotes or automatic deployment.')
    save(OUT/'next_implementation.json',next_design)
    conclusion=dict(at_unix=time.time(),classification='progress_costed_switching_implemented_and_liquidation_bottleneck_verified',
       previous_goal_turn_classification=spec['previous_goal_turn_classification'],
       retained_improved_candidate=spec['universes']['two'],new_candidate_additional_gain=False,
       selection=result['selection'],new_account_replays=28,exact_control_replays=28,
       unchanged_candidate_account_paths=28,tests_passed=10,actual_switches=0,rejection_reasons=dict(reasons),
       refined_diagnosis=diagnosis,residual_gross_at_missed_opportunity=residual_gross,
       quote_switch_window_available=any(x['minimum_holding_window_observed'] for x in audit['old_book_coverage']),
       full_quote_switch_pnl=None,full_quote_switch_q=None,next_implementation=next_design,
       protected=protected(),prospective_sources=len(prospective),orders_sent=0,live_strategy_changed=False,
       reserved_raw_read=False,historical_independence=False,goal_complete=False,no_progress_streak=0)
    save(OUT/'conclusion.json',conclusion)
    report=ROOT/'reports/opportunity_switch_pairs_20261018/report.html';report.parent.mkdir(exist_ok=False)
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in result['rows']}
    parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>비용 반영 페어 교체 검증</title>',
      '<style>body{max-width:1100px;margin:36px auto;padding:0 18px;font:16px/1.65 sans-serif;color:#18263b}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #cbd5e1;padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}.note{background:#eef4fc;padding:16px}</style>',
      '<h1>비용 반영 기회 비교와 청산 확인 후 재진입</h1>',
      '<p class="note"><b>구현과 검사를 완료했지만 추가 PnL 개선은 없다.</b> 두 고정 페어 집합에서 신규 계좌 28개를 평가했다. 실제 교체 요청은 0건이며, 모든 체결·계좌 경로가 대응하는 기존판과 정확히 같다. 기존에 입증한 첫째·둘째 대비 개선 후보는 유지한다.</p>',
      '<h2>구현한 동작</h2><p>동일한 예측 시점·만기의 기존 보유 가치와 다른 페어 가치를 비교한다. 현재 청산 비용, 추가 청산 여유 10bp, 새 진입·미래 청산 비용, 펀딩과 기존 5bp 여유를 반영한다. 교체 및 청산 후 첫 재진입의 지정가는 기존 한도를 넓히지 않고 현재 예상 체결가격에서 추가 10bp 이내로 제한한다. 기진입 수수료는 남은 가치에서 다시 차감하지 않는다.</p>',
      '<p>교체 판단은 청산 요청만 만든다. 양쪽 청산을 실제로 확인한 뒤 새 예측·가격·수량·비용을 계산한다. 일부 수량이 남아 있거나 주문 결과가 미확인인 동안 새 페어를 산 것으로 처리하지 않는다. 기다리는 동안 기회가 없어지면 현금으로 남는다. 새 규칙은 모의 연구 엔진에만 구현했다.</p>',
      '<h2>9월 1일–10월 15일 기준시장 결과</h2><p>시작 자본 1,000,000, 비용·펀딩 차감 후 평가손익. 종료 시 미청산 포지션 평가액을 포함한다. 대회 시장 실거래 수익이 아니다.</p>',
      '<table><tr><th>전략</th><th>비용</th><th>PnL</th><th>Q</th><th>기간 σ</th><th>교체</th></tr>']
    labels=[('first_strategy','첫 번째'),('second_strategy','두 번째'),
       (spec['universes']['two'],'기존 개선 후보'),(spec['candidates']['two'],'기존 개선 후보＋교체'),
       (spec['universes']['graph'],'그래프 선정'),(spec['candidates']['graph'],'그래프 선정＋교체')]
    for sc in ['base','stress']:
        for arm,label in labels:
            m=lookup['reference','comparison',arm,sc]
            parts.append(f"<tr><td>{label}</td><td>{sc}</td><td>{m['net_pnl']:+,.2f}</td><td>{m['financial_score']:.6f}</td><td>{m['volatility_horizon']*100:.5f}%</td><td>{m.get('switch_requests','—')}</td></tr>")
    parts+=['</table><h2>앞선 원인 설명의 정정</h2>',
      '<p>이전 분석은 ETF 보유 92시간과 BTC/HYPE 기회의 겹침을 보고 보유 대 교체 판단의 부재를 다음 원인 후보로 잡았다. 전체 주문 상태를 확인하니, ETF는 9월 18일 05:00 UTC에 이미 만기 청산을 요청했다. 기존 두 페어 후보의 BTC/HYPE 진입은 14:00 UTC였고, ETF 잔량이 모두 정리된 시각은 16:00 UTC다. 즉 새 기회가 나타나기 9시간 전부터 청산 중이었으며 청산 완료에 총 11시간이 걸렸다.</p>',
      f'<p>첫 청산에서 EWY 0.224는 모두 체결됐지만 EWJ 0.438 중 0.020만 체결됐다. 이후 매 시간 소량씩 청산했다. 새 기회 시점의 EWJ 잔량은 0.245이며, 당시 평가금액은 {residual_gross:.6f}다. 손실 난 페어를 미리 알았다는 뜻이 아니며, 이 수치는 사후 원인 분석이다.</p>',
      '<p><b>실제 병목은 이미 청산 중인 잔량이 단일 페어 처리 흐름을 막는 구조였다.</b> 비교 모델을 붙이는 것만으로 해결되지 않는다. 기존 청산을 끝난 것으로 가정하거나 참여율을 사후 완화하면 개선을 조작하게 된다.</p>',
      '<h2>실제 과거 호가의 검증 범위</h2><p>허용된 5개 과거 호가 기록에 있던 4개 가상 진입의 체결 이후 관측은 약 14.1~43.8분이다. 규칙의 최소 1시간 보유 이후까지 기록이 이어지지 않는다. 따라서 이 자료로 실제 교체나 교체 후 24시간 거래의 수익을 주장하지 않았다. 미래 검증 자료는 열지 않았다.</p>',
      '<h2>검사와 다음 구현</h2><p>검사 10개: 비용에 따른 판단, 새 가격 한도 보존, 미래 자료 변경 불변성, 예측 만기·유효기간, 현재 청산 수량, 한쪽 청산의 위험, 주문 처리 중 재교체 차단, 비활성화 시 기존 엔진 일치, 실제 부분 청산 이후 재진입, 청산 도중 신호 소멸. 기존 계좌 재현 28개와 신규 비교 28개를 완료했다.</p>',
      '<p>다음 구현은 활성 페어와 청산 잔량의 소유·현금·수수료·펀딩을 분리한 원장이다. 잔량은 계속 감축하고, 잔량과 종목이 겹치지 않는 새 페어만 기존 계좌 위험 한도 안에서 평가한다. 부분 체결의 각 경우와 미확인 주문을 고려하며, 위험 한도·자본을 늘리거나 잔량을 지운 것으로 처리하지 않는다. 구현 후 기존 두 페어 및 그래프 선정판과 다시 비교한다.</p>',
      '<p>모든 평가기간은 이미 연구에 사용한 과거이며 새로운 독립 검증이 아니다. 8월의 선택 기록을 이후 평가 전에 고정했으며 이후 결과로 바꾸지 않았다. 실거래 주문·전략 교체는 수행하지 않았다.</p>',
      '<p><a href="https://arxiv.org/html/2502.04284v1">2025년 신호 유효기간과 거래 비용 연구</a>를 참고해 보유와 비용 있는 교체를 구분했다. 이번 규칙은 그 논문의 무한기간 최적 제어 해를 재현한 것이 아니라 기존 예측을 사용하는 유한기간 비교 규칙이다.</p></html>']
    report.write_text('\n'.join(parts));HTMLParser().feed(report.read_text())
    save(HERE/'work_state.json',dict(status=conclusion['classification'],report=str(report.relative_to(ROOT)),
       retained_improved_candidate=spec['universes']['two'],new_candidate_additional_gain=False,
       next_action=next_design['objective'],next_design=str((OUT/'next_implementation.json').relative_to(ROOT)),
       new_account_replays=28,exact_controls=28,tests_passed=10,orders_sent=0,live_strategy_changed=False,
       reserved_raw_read=False,historical_independence=False,goal_complete=False,no_progress_streak=0))
    # Redirected runner logs remain mutable through the final console print.
    # Seal source/data/results only, never that stream.
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.log']+[report,data_path]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in files});verify(sources);verify(prospective)
    save(OUT/'conclusion_frozen.json',dict(at_unix=time.time(),sources=sources,excluded='mutable runner .log files',
       protected=protected(),prospective_sources=len(prospective)))
    print('DONE',len(sources),'sources; 28 unchanged candidate paths; residual gross',residual_gross,flush=True)


if __name__=='__main__':main()
