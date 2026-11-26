"""Finish the stability trial and document stronger continuous-path evidence."""
import csv
import json
import time
from pathlib import Path
from .research import ROOT,HERE,OUT,save,sha,verify,protected


def main():
    result=json.loads((OUT/'results.json').read_text());diag=json.loads((OUT/'diagnostics.json').read_text())
    books=json.loads((OUT/'books/results.json').read_text());risk=json.loads((OUT/'continuous_risk_results.json').read_text())
    activity=json.loads((OUT/'par_label_activity.json').read_text());spec=json.loads((HERE/'protocol.json').read_text())
    for path in ['frozen.json','diagnostics_frozen.json','books/frozen.json','continuous_risk_frozen.json','par_label_activity_frozen.json']:
        verify(json.loads((OUT/path).read_text())['sources'])
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(prospective)
    assert result['selection']['selected'] is None
    assert len(diag['exact_paths'])==14 and all(all(r['equal'].values()) for r in diag['exact_paths'])
    assert result['new_account_replays']==22 and len(result['exact_controls'])==24 and risk['new_replays']==4
    assert diag['verified_deleted_block_fits']==895 and books['trials']==[] and books['exact_prior_summaries']==20
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in result['rows']}
    all_above=all(r['candidate_pnl_difference']>0 and r['candidate_q_difference']>0 for r in risk['rows'])
    assert all_above
    next_action=(
        'Retain the demonstrated decisiongate strategy; stop changing lifecycle-veto confidence thresholds '
        'on the same named historical trades. The full and leave-week-out means cannot reliably identify '
        'a negative opportunity, and the robust rule makes no account change. Focus next on actual-par '
        'target validity: the original price-label learner requires source activity but often prices both '
        'entry and exit at inactive last-trade closes. Build separately declared execution-aware historical '
        'par outcomes that distinguish zero fills, partial inventory, paid completion and observed paid '
        'flat exits; keep censoring explicit and do not simply filter on future liquidity. Reuse permitted '
        'historical data and compare against both original par models and the retained reference control. '
        'Do not invent old L2, unseal forward raw data, or replace live automatically.'
    )
    continuous=[r for r in result['rows'] if r['venue']=='reference' and r['phase']=='continuous']
    conclusion=dict(at_unix=time.time(),classification='progress_stability_trial_and_continuous_risk_validation_completed',
        retained_improved_candidate=spec['control'],new_candidate_additional_gain=False,exact_incumbent_paths=14,
        daily_bundles=117,complete_bundles=89,successful_deleted_block_fits=895,tests_passed=8,
        new_portfolio_replays=26,exact_control_replays=24,continuous_results=continuous,
        continuous_risk_scaled_controls=risk['rows'],above_frozen_scale_native_pnl_q_all_four=all_above,
        actual_old_book_entries=0,par_label_activity=activity['rows'],
        protected=protected(),prospective_sources=len(prospective),next_action=next_action,
        orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False,
        goal_complete=False,no_progress_streak=0)
    save(OUT/'conclusion.json',conclusion)
    dest=ROOT/'reports/lifecycle_stability_pairs_20261018';dest.mkdir(exist_ok=False)
    labels={'first_strategy':'기존 1번','second_strategy':'기존 2번',spec['control']:'보존한 페어 개선판',
            spec['parent']:'앞선 청산손익 필터',spec['candidate']:'이번 주별 안정성 필터'}
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>주별 안정성 필터와 연속 운용 비교</title>
<style>body{max-width:1200px;margin:40px auto;font:16px/1.7 sans-serif;color:#172b3b;padding:0 18px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}</style>
<h1>주별 제외 재학습과 계좌 연속 운용 비교</h1>
<p><strong>보존한 페어 개선판을 유지한다.</strong> 새 안정성 필터는 기존 개선판과 14개 평가 조건에서 주문·체결·잔고·손익이 모두 같았다. 추가 수익을 만들었다고 주장하지 않는다. 대신 8월 1일부터 10월 15일까지 계좌를 초기화하지 않은 재평가와, 과거에 고정한 위험 규모를 적용한 비교에서도 보존한 개선판이 기존 1·2번보다 높은 PnL과 Q를 보였다.</p>
<p>초기자본 1M, 외부 기준시장 과거 자료와 모의 체결 비용·수수료·펀딩을 적용한 계좌 결과다. 기간 경계는 모두 12UTC(한국시간 21시)이며 미청산 포지션 평가손익을 포함한다. 실제 par 지갑 수익이나 독립 미래 검증 결과가 아니다. 실제 운용과 예약된 미래 평가 자료는 변경하지 않았다.</p>
<h2>새로 구현한 판단</h2><p>앞선 모델은 청산손익 추정이 음수이면 바로 진입을 막았다. 이번에는 각 적합 시점에 학습 가능한 자료에서 UTC 월요일 기준 한 주씩 제외해 재학습했다. 해당 주와 겹치는 거래는 시작 주가 달라도 모두 제외했다. 원래 모델과 모든 제외 재학습 모델이 음수를 예측할 때에만 기존 진입을 막는다. 최소 네 주와 모든 재학습의 표본 조건이 갖춰지지 않으면 기존 판단을 유지한다. 이는 학습 자료에 대한 부호 안정성 검사이며 수익 확률이나 보장된 신뢰구간이 아니다.</p>
<p>117개 일별 모델 묶음 중 89개가 모든 조건을 갖췄고, 주별 제외 재학습 895개를 완료했다. 8월 BTC/ETH의 원래 음수 추정은 제외 주에 따라 양수로 바뀌었다. 9월 18일 BTC/HYPE도 마찬가지였다. 이 규칙은 앞선 필터가 놓친 수익 진입을 복구했지만 앞선 필터가 피했던 손실 진입도 함께 허용하여, 최종 계좌 경로는 기존 개선판과 같아졌다.</p>'''
    csv_rows=[]
    for scenario in ['base','stress']:
        page+=f'<h2>8월 1일~10월 15일 연속 운용 · {"기본 비용" if scenario=="base" else "불리한 체결 조건"}</h2><table><tr><th>전략</th><th>순손익</th><th>수익률</th><th>기간 σ</th><th>Q</th><th>최대낙폭</th><th>평균 총노출</th></tr>'
        for arm,label in labels.items():
            m=lookup['reference','continuous',arm,scenario]
            page+=f"<tr><td>{label}</td><td>{m['net_pnl']:+,.2f}</td><td>{100*m['roi']:+.6f}%</td><td>{100*m['volatility_horizon']:.6f}%</td><td>{m['financial_score']:.8f}</td><td>{100*m['max_drawdown']:.6f}%</td><td>{100*m['mean_gross']:.4f}%</td></tr>"
            csv_rows.append(dict(phase='continuous',scenario=scenario,arm=arm,**{k:m[k] for k in ['net_pnl','roi','volatility_horizon','financial_score','max_drawdown','mean_gross']}))
        page+='</table>'
    page+='''<h2>투자 규모만 늘린 효과인가</h2><p>이전 연구에서 8월 변동성만으로 정해 보존한 배율 1.625배(기존 1번), 1.9375배(기존 2번)를 연속 계좌 전체에 그대로 적용했다. 현재 연속 기간의 PnL을 보고 배율을 새로 맞추지 않았다. 8월의 근사 위험 일치이며 이후 실제 변동성이 같도록 강제한 비교는 아니다.</p><table><tr><th>비교 전략</th><th>조건</th><th>순손익</th><th>기간 σ</th><th>Q</th><th>개선판 PnL 우위</th><th>개선판 Q 우위</th></tr>'''
    for r in risk['rows']:
        m=r['metrics'];label='기존 1번 ×1.625' if r['arm'].startswith('first') else '기존 2번 ×1.9375'
        page+=f"<tr><td>{label}</td><td>{r['scenario']}</td><td>{m['net_pnl']:+,.2f}</td><td>{100*m['volatility_horizon']:.6f}%</td><td>{m['financial_score']:.8f}</td><td>{r['candidate_pnl_difference']:+,.2f}</td><td>{r['candidate_q_difference']:+.8f}</td></tr>"
        csv_rows.append(dict(phase='continuous_risk_matched',scenario=r['scenario'],arm=r['arm'],**{k:m[k] for k in ['net_pnl','roi','volatility_horizon','financial_score','max_drawdown','mean_gross']}))
    page+='''</table><p>네 조건 모두 개선판의 PnL과 Q가 높았고 실현 기간 변동성도 조정된 비교 전략보다 낮았다. 그러나 기본 조건의 최대낙폭은 개선판 1.2195%로 조정된 기존 1·2번의 1.0672%·1.1448%보다 크다. 모든 위험 지표가 우월하다고 해석하지 않는다.</p>
<h2>기존 기간별 판정과 par</h2><p>새 후보는 8월 기본·불리한 조건에서 기존 개선판과 PnL·Q가 같아 추가 승격되지 않았다. 9월 이후 및 par의 여덟 조건에서도 같은 경로다. 이후 연속 운용 결과로 이 결정을 다시 고르지 않았다. 앞선 청산손익 필터는 연속 기본 조건에서 −107.42 낮고 불리한 조건에서는 +336.88 높아, 모든 조건에서 더 좋은 대안은 아니다.</p>
<p>허용된 과거 호가 테이프 다섯 개에서 기존 두 페어 모델과 새 필터 모두 진입 0개였다. 기존 호가 판단 요약 20개를 정확히 재현했다. 이로부터 실제 par에서 완료된 페어의 수익성을 주장하지 않는다.</p>
<h2>다음 연구 방향을 바꾼 자료 점검</h2><p>기존 par 직접 예측 모델을 확인하니 출발 시점 거래량은 요구하지만 학습 목표의 다음 봉 진입가와 미래 종료가에는 실제 거래 발생을 요구하지 않았다. 다음은 원래 학습 조건에 포함된 관측 중 양쪽 종목이 해당 봉에서 모두 거래된 개수다. 봉 거래가 있어도 실제 호가 잔량이나 우리 주문의 체결을 보장하지 않는다.</p>
<table><tr><th>예측 기간</th><th>적합 시점 UTC</th><th>원래 학습 행</th><th>진입 봉 양쪽 활동</th><th>종료 봉 양쪽 활동</th><th>두 끝점 모두 활동</th></tr>'''
    from datetime import datetime,timezone
    for r in activity['rows']:
        date=datetime.fromtimestamp(r['fit_time'],timezone.utc).strftime('%m-%d %H:%M')
        page+=f"<tr><td>{r['horizon']}시간</td><td>{date}</td><td>{r['original_training_rows']}</td><td>{r['next_bar_both_active']}</td><td>{r['terminal_both_active']}</td><td>{r['both_endpoints_active']}</td></tr>"
    page+='''</table><p>예를 들어 10월 8일의 24시간 모델은 학습 행 135개 중 종료 봉 양쪽 거래가 있었던 것이 5개이며 진입·종료 양끝 모두 활동한 것은 0개다. 10월 14일에는 342개 중 82개로 증가했다. 과거 체결 종가의 예측을 즉시 가능한 매매 수익으로 해석하기 어려운 이유를 구체적으로 확인한 것이다.</p>
<p>미래에 거래가 있었던 표본만 골라 학습한 뒤 그 유동성이 항상 있을 것처럼 추론하면 또 다른 편향이 생긴다. 다음에는 실제 par의 역사적 모의 주문 결과를 전량 미체결, 부분 체결, 비용을 지불한 청산 완료, 관측 종료까지 미청산으로 나누어 학습 목표를 만든다. 미래 호가를 꾸며내거나 관측되지 않은 청산을 이익으로 기록하지 않는다. 기존 자료를 사용하는 연구를 계속하되 이 부호 필터의 임계값만 반복 조절하지 않는다.</p>
<h2>검증</h2><p>새 계좌 재생 26개(구간별 후보 12, 연속 계좌 10, 고정 위험 배율 4), 기존 계좌 재현 24개, 검사 8개를 완료했다. 14개 후보 조건의 실제 주문·체결·의사결정·잔고·손익이 기존 개선판과 정확히 같다. 895개 제외 재학습에서 삭제 구간과 겹치는 정답이 남지 않았고 모든 학습 결과가 적합 시점 전에 끝났음을 확인했다. 보호 소스 373개, 예약 의존 345개, 향후 평가 소스 1,045개의 해시도 확인했다. 실제 주문은 보내지 않았다.</p>
<p><a href="../decision_gate_pairs_20261017/report.html">보존한 개선판</a> · <a href="../lifecycle_value_pairs_20261018/report.html">앞선 청산손익 학습 모델</a></p></html>'''
    (dest/'report.html').write_text(page)
    with (dest/'continuous_results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    state=dict(conclusion,report=str((dest/'report.html').relative_to(ROOT)));save(HERE/'work_state.json',state)
    cross=ROOT/'experiments/pairs_development_status_20261017/comparisons/lifecycle_stability_review.json'
    assert not cross.exists();save(cross,state)
    files=[Path(__file__),OUT/'conclusion.json',OUT/'results.json',OUT/'selection.json',OUT/'models.json',
           OUT/'diagnostics.json',OUT/'books/results.json',OUT/'continuous_risk_results.json',OUT/'par_label_activity.json',
           dest/'report.html',dest/'continuous_results.csv',cross,*sorted((OUT/'runs').glob('*.json.gz'))]
    save(OUT/'conclusion_frozen.json',dict(at_unix=time.time(),sources={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('DONE stability and continuous risk audit; retain incumbent; par label limitation quantified',flush=True)


if __name__=='__main__':main()
