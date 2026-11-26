"""Preserve the positive incumbent result and reject unsupported graph promotion."""
import html
import json
import time
from pathlib import Path
from .research import ROOT,HERE,OUT,GATE,STABILITY,readrun,save,sha,verify,protected


def main():
    spec=json.loads((HERE/'protocol.json').read_text())
    result=json.loads((OUT/'results.json').read_text())
    diagnosis=json.loads((OUT/'diagnostics.json').read_text())
    books=json.loads((OUT/'books/results.json').read_text())
    for path in [OUT/'frozen.json',OUT/'diagnostics_frozen.json',OUT/'books/frozen.json']:
        verify(json.loads(path.read_text())['sources'])
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources']
    verify(prospective)
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in result['rows']}
    comparisons=[]
    for phase in ['validation','comparison','continuous']:
        for sc in ['base','stress']:
            new=lookup['reference',phase,spec['candidate'],sc]
            if phase=='continuous':
                old=readrun(STABILITY/'runs'/f"reference-continuous-{spec['control']}-{sc}.json.gz")['metrics']
                natives={arm:readrun(STABILITY/'runs'/f'reference-continuous-{arm}-{sc}.json.gz')['metrics']
                         for arm in ['first_strategy','second_strategy']}
            else:
                old=lookup['reference',phase,spec['control'],sc]
                natives={arm:lookup['reference',phase,arm,sc] for arm in ['first_strategy','second_strategy']}
            comparisons.append(dict(phase=phase,scenario=sc,candidate=new,retained=old,natives=natives,
                pnl_difference=new['net_pnl']-old['net_pnl'],q_difference=new['financial_score']-old['financial_score'],
                above_native_pnl_q=all(new['net_pnl']>v['net_pnl'] and new['financial_score']>v['financial_score'] for v in natives.values())))
    next_action=('Retain the demonstrated two-pair decision gate. The fixed pre-period graph adds small ETF opportunities '
      'but does not improve incumbent PnL/Q; do not tune graph edges or choose by later losses. Investigate the one-active-pair '
      'capital-allocation constraint: independently assess a newly available pair against the remaining value of the current pair '
      'and paid exit/entry/repair costs, only using current forecasts and observable liquidity. A replacement must first confirm '
      'old inventory is flat, then recompute new size and opportunity; no free instantaneous swap, enlarged risk limit, known-loser '
      'rule, or automatic live replacement. Compare to both the two-pair incumbent and this fixed graph with no preemption, and '
      'treat any work on these dates as reused-history development. Do not claim the observed timing overlap proves profitable preemption.')
    conclusion=dict(at_unix=time.time(),classification='progress_preperiod_graph_and_opportunity_occupation_audit_completed',
        previous_goal_turn_classification=spec['previous_goal_turn_classification'],
        retained_improved_candidate=spec['control'],new_candidate=spec['candidate'],
        new_candidate_additional_gain=False,formation=result['formation'],selection=result['selection'],
        reference_comparisons=comparisons,all_reference_conditions_above_originals=all(r['above_native_pnl_q'] for r in comparisons),
        tests_passed=result['tests_passed'],new_account_replays=result['new_account_replays'],
        exact_controls=len(result['exact_controls']),post_hoc_exact_restoration_replays=diagnosis['extra_diagnostic_replays'],
        book_entries=len(books['trials']),book_statuses=books['statuses'],exact_book_summaries=books['exact_control_summaries'],
        complete_future_quote_exit_observed=False,old_book_full_trade_pnl=None,old_book_q=None,
        overlap_evidence=diagnosis['overlapping_control_entries'],next_action=next_action,
        protected=protected(),prospective_sources=len(prospective),orders_sent=0,live_strategy_changed=False,
        reserved_raw_read=False,historical_independence=False,goal_complete=False,no_progress_streak=0)
    save(OUT/'conclusion.json',conclusion)
    report=ROOT/'reports/formation_graph_pairs_20261018/report.html';report.parent.mkdir(exist_ok=False)
    parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>평가 전 페어 선정 검증</title>',
      '<style>body{max-width:1200px;margin:36px auto;padding:0 18px;font:16px/1.65 sans-serif;color:#18263b}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{padding:8px;border:1px solid #cbd5e1;text-align:right}th:first-child,td:first-child{text-align:left}code{background:#eef2f6}h2{margin-top:32px}.note{background:#eef4fc;padding:16px}</style>',
      '<h1>평가기간 이전 관계로 선정한 페어: 구현·비교 결과</h1>',
      '<p class="note"><b>기존 첫째·둘째 전략보다 높은 기준시장 백테스트 PnL/Q는 유지했지만, 이미 확보한 개선 후보를 추가로 앞서지 못했다.</b> 선정 결과는 기존 두 페어 진입 판단 모델 유지다. 실거래 전략은 변경하지 않았다.</p>',
      '<p>90일 형성 기간은 2026-05-03 12:00 UTC 이상, 2026-08-01 12:00 UTC 미만이다. 양 끝에 가격·거래량·체결 수가 관측된 연속 시간봉 수익률만 사용했다. 상관계수 0.3 이상, 관측 200개 이상인 연결에 최소 신장 숲을 적용했다. 규칙·소스는 계산 전에 고정하고, 8월 결과로 선정 기록을 저장한 뒤 이후 구간을 계산했다. 과거 결과를 이미 연구에 사용했으므로 새로운 독립 검증은 아니다.</p>',
      '<h2>페어 선정</h2><table><tr><th>페어</th><th>형성 관측</th><th>상관계수</th><th>선정</th></tr>']
    for edge in result['formation']['edges']:
        selected=edge['pair'] in result['formation']['selected_pairs']
        rho='—' if edge['correlation'] is None else f"{edge['correlation']:.6f}"
        parts.append(f"<tr><td>{'/'.join(edge['pair'])}</td><td>{edge['observations']}</td><td>{rho}</td><td>{'예' if selected else '아니오'}</td></tr>")
    parts+=['</table><p>상관관계는 공적분이나 평균회귀 수익성을 보장하지 않는다. 기존 예측 평균·위험 추정·진입 방향·보유 연장·청산 규칙은 유지하고 신규 진입 허용 여부만 바꿨다.</p>',
      '<h2>기준시장 계좌 비교</h2><p>각 구간 시작 자본 1,000,000. 모의 체결 비용·수수료·펀딩 차감 후 평가손익이며, 종료 시 남은 포지션은 평가가격으로 계산한다. 기간 σ는 일별 수익률 표본 표준편차 × √일수다. 연속 구간은 9월에 계좌를 초기화하지 않는다.</p>',
      '<table><tr><th>구간 / 비용</th><th>기존 개선판 PnL</th><th>신규 그래프 PnL</th><th>차이</th><th>기존 Q</th><th>신규 Q</th><th>신규 σ</th><th>완료 페어</th></tr>']
    for row in comparisons:
        a,b=row['candidate'],row['retained']
        parts.append(f"<tr><td>{row['phase']} / {row['scenario']}</td><td>{b['net_pnl']:+,.2f}</td><td>{a['net_pnl']:+,.2f}</td><td>{row['pnl_difference']:+,.2f}</td><td>{b['financial_score']:.6f}</td><td>{a['financial_score']:.6f}</td><td>{a['volatility_horizon']*100:.5f}%</td><td>{a['completed_pairs']}</td></tr>")
    parts+=['</table><p>validation: 8월 1일–9월 1일; comparison: 9월 1일–10월 15일; continuous: 8월 1일–10월 15일. 모두 12:00 UTC 경계다. 세 구간, 두 비용 조건에서 첫째·둘째 원본보다 PnL/Q가 높지만, 추가 개선은 아니다.</p>',
      '<h2>변화의 원인</h2><p>이후 기본 조건에서 추가 EWY/EWJ 페어 손익은 −15.693199이며, BTC/HYPE 페어 손익은 기존 개선판과 정확히 같다. 불리한 조건에서는 EWY/EWJ 자체는 +2.816001이지만 BTC/HYPE 페어 손익이 +3,774.538283에서 +3,237.408857로 줄었다. 전체 계좌 차이는 기본 −15.684944, 불리한 조건 −531.629245다. 계좌 자산 변화에 따른 기본 전략 수량 변화도 포함한다.</p>',
      '<p>불리한 조건의 9월 14일 20:00 UTC ETF 진입 기대 순이익은 0.063520이었다. 이 페어는 92시간 동안 단일 페어 자리를 사용했다. 같은 시간대 기존 개선판에는 기대 순이익 533.711126의 BTC/HYPE 진입이 있었다. 추가 ETF 진입 권한만 없앤 사후 원인 분리 실험에서 기본·불리한 두 조건의 모든 계좌 결과가 기존 개선판과 정확히 같아졌다. 이 원인 분리는 새 전략 선정이나 수익성 있는 교체 정책을 입증한 것이 아니다.</p>',
      '<h2>실제 과거 par 호가에서 확인한 범위</h2><p>허용된 5개 호가 기록에서 신규 조합은 EWY/EWJ 진입 4건(기본·불리한 조건별 가상 실험)을 만들었고, 모두 양쪽 체결 후 보유 상태에 도달했다. 같은 시각의 두 조건은 독립적인 시장 기회 두 개가 아니다. 기존 두 페어 모델의 10개 호가 집계는 정확히 재현했다. 기록이 단절되어 24시간 이후 청산은 관측하지 못했으므로 전체 거래 PnL/Q를 기입하지 않았다.</p>',
      '<h2>검사와 다음 작업</h2><p>형성 시점 이후 자료 변조, 양 끝 활동성, 부족 표본, 동률 처리·순환 방지, 누락 시간, 보유 예측 보존 검사 6개 통과. 신규 계좌 14개, 기존 계좌 26개 정확 재현, 사후 원인 분리 2개를 완료했다. 현재 실거래 코드 및 보류된 미래 관측 자료는 변경·열람하지 않았다.</p>',
      '<p>'+html.escape(next_action)+'</p>',
      '<h2>참고 연구와 적용 범위</h2><p><a href="https://www.nature.com/articles/s41599-025-05661-7.pdf">2025 페어 네트워크 연구</a>는 공적분 네트워크와 PMFG/TMFG를 다룬다. 이번 구현은 그 방법의 재현이 아니라 거래 대상 간 관계를 함께 고려한다는 동기만 차용한 단순 비교 실험이다. <a href="https://arxiv.org/abs/2005.03963">금융 수익률 상관관계 MST 연구</a>를 참고했으며, 여기서는 사전 고정한 Pearson 상관계수만 사용했다.</p></html>']
    report.write_text('\n'.join(parts))
    save(HERE/'work_state.json',dict(status=conclusion['classification'],report=str(report.relative_to(ROOT)),
        retained_improved_candidate=spec['control'],additional_gain=False,next_action=next_action,
        new_account_replays=14,exact_controls=26,post_hoc_restorations=2,tests_passed=6,
        orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False,
        goal_complete=False,no_progress_streak=0))
    sources=json.loads((OUT/'books/frozen.json').read_text())['sources'].copy()
    sources.update(json.loads((OUT/'diagnostics_frozen.json').read_text())['sources'])
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[report]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in files});verify(sources);verify(prospective)
    save(OUT/'conclusion_frozen.json',dict(at_unix=time.time(),sources=sources,
                                        prospective_sources=len(prospective),protected=protected()))
    print('DONE',len(sources),'sources; graph not promoted; retained PnL/Q improvement preserved',flush=True)


if __name__=='__main__':main()
