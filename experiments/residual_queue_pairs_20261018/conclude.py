"""Document an execution improvement without claiming universal alpha promotion."""
import json
import time
from pathlib import Path
from html.parser import HTMLParser
from .research import ROOT,HERE,OUT,save,sha,verify,protected


def main():
    result=json.loads((OUT/'results.json').read_text());audit=json.loads((OUT/'audit.json').read_text())
    spec=json.loads((HERE/'protocol.json').read_text())
    books={kind:json.loads((OUT/kind/'results.json').read_text()) for kind in ['books','books_partial']}
    sources=json.loads((OUT/'audit_frozen.json').read_text())['sources'].copy()
    for kind in books:sources.update(json.loads((OUT/kind/'frozen.json').read_text())['sources'])
    verify(sources)
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(prospective)
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in result['rows']}
    comparisons=[]
    for phase in ['validation','comparison','continuous']:
        for sc in ['base','stress']:
            new=lookup['reference',phase,spec['candidates']['graph'],sc]
            matched=lookup['reference',phase,spec['universes']['graph'],sc]
            retained=lookup['reference',phase,spec['universes']['two'],sc]
            comparisons.append(dict(phase=phase,scenario=sc,candidate=new,matched_no_queue=matched,retained=retained,
                matched_pnl_gain=new['net_pnl']-matched['net_pnl'],matched_q_gain=new['financial_score']-matched['financial_score'],
                retained_pnl_difference=new['net_pnl']-retained['net_pnl'],retained_q_difference=new['financial_score']-retained['financial_score']))
    later=next(r for r in comparisons if r['phase']=='comparison' and r['scenario']=='stress')
    assert later['matched_pnl_gain']>0 and later['matched_q_gain']>0
    assert result['selection']['selected']==spec['universes']['two']
    assert audit['risk_breach_bars']==0
    next_action=('Keep the demonstrated two-pair decision gate as the performance incumbent and retain the residual queue '
      'as an independently tested execution improvement. The corrected fixed-graph stress path restores the missed BTC/HYPE '
      'entry; its ETF overlay remains negative in base conditions. Next compare the already available direct-return forecast '
      'without the decision gate on this SAME frozen graph and queue engine, against gated/no-queue controls and both original '
      'strategies. This is a single model-filter ablation, not a new grid or a claim that ungated means will win. Freeze before '
      'replaying; no graph reselection, known-loser removal, enlarged limits or reused-history independent-validation claim. '
      'Existing old quote tapes cannot establish a 24h strategy exit or live profitability.')
    conclusion=dict(at_unix=time.time(),classification='progress_residual_queue_implemented_and_matched_stress_pnl_q_improved',
       previous_goal_turn_classification=spec['previous_goal_turn_classification'],
       retained_improved_candidate=spec['universes']['two'],new_candidate=spec['candidates']['graph'],
       observed_execution_improvement=True,additional_uniform_improvement_over_retained=False,
       selection=result['selection'],reference_comparisons=comparisons,new_account_replays=28,exact_control_replays=28,
       tests_passed=12,episode_accounts_verified=audit['accounts_verified'],risk_breach_bars=0,
       concurrent_entry_accounts=audit['accounts_with_new_entry_during_residual'],
       concurrent_entry_independence='Two overlapping reference replays of the same historical opportunity, not two independent market events.',
       exact_old_book_entry_probes=books['books']['exact_entry_probes'],
       full_exit_diagnostics=books['books']['forced_exit_diagnostics'],
       partial_intent_handoff_diagnostics=books['books_partial']['post_handoff_diagnostics'],
       quote_scope='Forced 60-second full exits and deliberate half exits verify physical fills/fees/cash and ownership under received quotes. They are not the strategy holding rule or alpha evidence. Missing strategy-horizon exits remain unobserved.',
       next_action=next_action,protected=protected(),prospective_sources=len(prospective),orders_sent=0,
       live_strategy_changed=False,reserved_raw_read=False,historical_independence=False,goal_complete=False,no_progress_streak=0)
    save(OUT/'conclusion.json',conclusion)
    report=ROOT/'reports/residual_queue_pairs_20261018/report.html';report.parent.mkdir(exist_ok=False)
    page=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>잔량 분리 운용 검증</title>',
      '<style>body{max-width:1150px;margin:36px auto;padding:0 18px;font:16px/1.65 sans-serif;color:#18263b}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{border:1px solid #cbd5e1;padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}.note{background:#eef4fc;padding:16px}</style>',
      '<h1>청산 잔량 분리: 실제 실행 경로와 비용 차감 성과</h1>',
      '<p class="note"><b>그래프 후보의 불리한 체결 조건에서 PnL과 Q가 개선됐다.</b> 이후 구간 +2,774.08 → +3,308.53, Q 0.117679 → 0.124505다. 기존 최선의 두 페어 후보와 비교하면 기본 조건에서는 −15.68, 불리한 조건에서는 +2.82이므로 전체 우위로 채택하지 않는다.</p>',
      '<h2>구현</h2><p>실제 계좌는 하나다. 기본 전략, 활성 페어, 청산 잔량의 소유 원장을 분리했다. 이미 청산을 요청하고 한 번 이상 결과를 확인한 잔량만 옮긴다. 이동은 내부 회계 처리이며 실제 포지션·현금·수수료·체결 수는 변하지 않는다. 잔량의 주문 목표는 항상 0이고, 그 수량은 줄어들기만 한다.</p>',
      '<p>새 페어는 잔량과 종목이 겹치지 않아야 한다. 기존 예측·필터·주문 규모·가격 한도·자본·전체 위험 한도를 유지하며, 잔량이 그대로 남는 경우와 일부 또는 전부 청산되는 경우 각각에 새 한쪽 또는 양쪽 체결을 더해 위험을 검사한다. 미래의 청산 수량을 미리 아는 것으로 처리하지 않는다. 실제 노출은 달라질 수 있으므로 계좌 변동성과 Q를 함께 비교했다.</p>',
      '<h2>기준시장 결과</h2><p>시작 자본 1,000,000, 모의 체결 비용·수수료·펀딩 차감 후 평가손익이다. 미청산 잔량은 계속 평가액과 위험에 포함한다. 기간 σ는 일별 수익률 표본 표준편차 × √완료 일수다. 실제 par 실거래 수익은 아니다.</p>',
      '<table><tr><th>구간 / 비용</th><th>그래프 기존 PnL</th><th>잔량 분리 PnL</th><th>개선액</th><th>새 Q</th><th>새 σ</th><th>기존 최선 대비 PnL</th></tr>']
    for row in comparisons:
        m=row['candidate']
        page.append(f"<tr><td>{row['phase']} / {row['scenario']}</td><td>{row['matched_no_queue']['net_pnl']:+,.2f}</td><td>{m['net_pnl']:+,.2f}</td><td>{row['matched_pnl_gain']:+,.2f}</td><td>{m['financial_score']:.6f}</td><td>{m['volatility_horizon']*100:.5f}%</td><td>{row['retained_pnl_difference']:+,.2f}</td></tr>")
    page+=['</table><p>validation: 8월 1일–9월 1일; comparison: 9월 1일–10월 15일; continuous: 8월 1일–10월 15일. 모두 12:00 UTC 경계다. 연속 구간과 이후 구간은 겹치므로 독립적인 성공 두 번으로 세지 않는다.</p>',
      '<h2>무엇이 달라졌는가</h2><p>9월 18일 14:00 UTC, EWJ 청산 잔량 0.245(평가금액 23.63221)를 계속 보유·정리하면서 BTC/HYPE 진입을 요청했다. 이전 구조는 16:00까지 기다렸다. 새 구조의 BTC/HYPE 페어 손익 +3,774.538283은 기존 두 페어 후보와 정확히 같아졌고, EWY/EWJ 손익 +2.816001도 원장에 남았다. 잔량 손익이나 청산 비용을 지우지 않고 기회를 처리한 결과다.</p>',
      '<p>기본 조건에서는 EWY/EWJ 추가 거래가 여전히 −15.693199를 기록한다. 청산 처리를 개선했다고 예측 모델이 좋아진 것은 아니다. 8월 선정 규칙에서도 신규 후보를 채택하지 않았고, 이후 결과를 보고 그 결정을 수정하지 않았다.</p>',
      '<h2>회계와 체결 검사</h2><p>검사 12개, 신규 계좌 28개, 기존 계좌 28개 정확 재현을 완료했다. 모든 새 계좌에서 거래별 수량·수수료·펀딩·최종 손익 합계가 실제 계좌와 소유 원장에 맞는지 확인했다. 내부 이동을 실제 거래량으로 더하지 않고, 완료 거래는 잔량이 모두 사라진 뒤에만 셌다. 관측된 위험 한도 초과는 0건이다.</p>',
      '<p>허용된 실제 과거 호가의 진입 4건을 그대로 재현했다. 별도로 60초 후 전량 청산을 요청한 4건, 절반 청산을 요청해 잔량을 만든 4건을 검사했다. 잔량 이동 전후 실제 체결·현금·수수료가 원래 단일 원장 처리와 같았다. 절반 청산은 회계 경로를 시험하기 위한 의도적 요청이며 실제 유동성 부족으로 생긴 부분 체결이라 주장하지 않는다. 이 짧은 검사를 전략의 24시간 수익으로 사용하지 않았다.</p>',
      '<h2>판정과 다음 비교</h2><p>기존 두 페어 개선 후보를 유지한다. 잔량 분리 원장은 실행 개선으로 보존한다. 같은 고정 페어·같은 원장에서 기존 직접 수익률 예측과 진입 판단 필터의 기여를 한 가지 비교 실험으로 분리할 예정이다. 새로운 모델을 학습했다고 하거나, 손실 난 종목만 사후 삭제해 개선했다고 보고하지 않는다.</p>',
      '<p>모든 기간은 재사용 과거 자료이며 독립적인 미래 검증이 아니다. 미래 관측 파일을 열거나 실거래 코드·설정·전략을 변경하지 않았다.</p></html>']
    report.write_text('\n'.join(page));HTMLParser().feed(report.read_text())
    save(HERE/'work_state.json',dict(status=conclusion['classification'],report=str(report.relative_to(ROOT)),
       retained_improved_candidate=spec['universes']['two'],observed_execution_improvement=True,
       additional_uniform_improvement_over_retained=False,next_action=next_action,
       new_account_replays=28,exact_controls=28,tests_passed=12,orders_sent=0,live_strategy_changed=False,
       reserved_raw_read=False,historical_independence=False,goal_complete=False,no_progress_streak=0))
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.log']+[report]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in files});verify(sources);verify(prospective)
    save(OUT/'conclusion_frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),
       prospective_sources=len(prospective),excluded='mutable runner .log files'))
    print('DONE',len(sources),'sources; matched stress gain',later['matched_pnl_gain'],'Q gain',later['matched_q_gain'],flush=True)


if __name__=='__main__':main()
