"""Close the complete evaluation and retain every result, including failures."""
import csv
import json
import time
from pathlib import Path
from .research_v2 import ROOT,HERE,OUT,GATE,readrun,save,sha,verify,protected


def main():
    result=json.loads((OUT/'results.json').read_text())
    diagnostics=json.loads((OUT/'diagnostics.json').read_text())
    books=json.loads((OUT/'books/results.json').read_text())
    labels=json.loads((OUT/'labels.json').read_text())
    for file in ['frozen.json','evaluation_frozen_v2.json','diagnostics_frozen.json','books/frozen.json']:
        verify(json.loads((OUT/file).read_text())['sources'])
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources']
    verify(prospective)
    assert result['new_account_replays']==24 and len(result['exact_controls'])==12
    assert result['tests_passed']==8 and len(diagnostics['rows'])==8
    control='first_dense_hourlyhedge_decisiongate_24'
    candidate='first_dense_hourlyhedge_lifecycle_filter_24'
    assert result['selection']['selected']==candidate
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in result['rows']}
    checks=[];unchanged=[];paths=[]
    for phase in ['validation','comparison']:
        for sc in ['base','stress']:
            new=lookup['reference',phase,candidate,sc];old=lookup['reference',phase,control,sc]
            checks.append(dict(phase=phase,scenario=sc,pnl_difference=new['net_pnl']-old['net_pnl'],
                               q_difference=new['financial_score']-old['financial_score'],
                               sigma_difference=new['volatility_horizon']-old['volatility_horizon']))
    assert all(c['pnl_difference']>0 and c['q_difference']>0 for c in checks if c['phase']=='validation')
    assert all(c['pnl_difference']<0 and c['q_difference']<0 for c in checks if c['phase']=='comparison')
    for venue,phase in result['windows']:
        if venue=='reference':continue
        for sc in ['base','stress']:
            first=OUT/'runs'/f'{venue}-{phase}-{candidate}-{sc}.json.gz'
            second=GATE/'runs'/f'{venue}-{phase}-{control}-{sc}.json.gz'
            a,b=readrun(first),readrun(second)
            for field in ['metrics','snapshots','fills','owner_fills']:
                assert a[field]==b[field],(venue,phase,sc,field)
            unchanged.append(dict(venue=venue,phase=phase,scenario=sc));paths.extend([first,second])
    assert len(unchanged)==8 and books['exact_prior_trials']==4
    next_action=(
        'Retain decisiongate as the demonstrated incumbent; lifecycle_filter is an August-selected research '
        'alternative that remains above native first/second but underperforms decisiongate later. '
        'The new value estimate delays the September 18 winning entry by two hours, while removing the '
        'August BTC/ETH loss. Before another entry veto, measure the stability and uncertainty of the '
        'estimated negative value using only matured, time-blocked training episodes. A weak fitted '
        'negative mean should not silently be treated as a confidently negative opportunity. Freeze '
        'a separate uncertainty policy before evaluation; do not tune cutoffs on these named trades. '
        'Preserve the actual-par transfer limitation and do not replace live automatically.'
    )
    actual_training_replays=len(labels['rows'])-labels['counts'].get('unobserved_next_bar',0)
    conclusion=dict(at_unix=time.time(),classification='progress_completed_lifecycle_value_training_and_evaluation',
         retained_improved_candidate=control,august_selected_candidate=candidate,
         improvement_over_incumbent_in_august=True,improvement_over_incumbent_in_later_period=False,
         comparisons=checks,source_opportunities=len(labels['rows']),actual_virtual_training_replays=actual_training_replays,
         dataset_counts=labels['counts'],daily_value_models=result['daily_value_models'],
         new_portfolio_replays=24,exact_control_replays=12,tests_passed=8,
         par_filter_financial_paths_unchanged=unchanged,book_trials=len(books['trials']),exact_old_book_trials=4,
         no_new_full_par_lifecycle=True,forecast_diagnostics=diagnostics['forecasts'],
         implementation_recovery='First evaluation stopped on numpy-array versus JSON-list equality. Version 2 normalizes serialization only and reuses unchanged fitted models; all twelve controls then match exactly.',
         count_clarification='The original attempted_virtual_accounts field counts 367 source opportunities; five have no observable next bar and are not replayed. Actual virtual training replays: 362.',
         protected=protected(),prospective_sources=len(prospective),next_action=next_action,
         orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False,
         goal_complete=False,no_progress_streak=0)
    save(OUT/'conclusion.json',conclusion)
    dest=ROOT/'reports/lifecycle_value_pairs_20261018';dest.mkdir(exist_ok=False)
    labels_by_arm={'first_strategy':'기존 1번','second_strategy':'기존 2번',control:'유지한 개선판',
                   candidate:'새 청산손익 필터',
                   'first_dense_hourlyhedge_lifecycle_rank_24':'새 여섯 페어 순위판'}
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>실행 결과를 학습한 페어 거래 비교</title>
<style>body{max-width:1180px;margin:40px auto;font:16px/1.7 sans-serif;color:#182938;padding:0 18px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}</style>
<h1>실제 모의 청산손익을 학습한 페어 거래 후보</h1>
<p><strong>기존 개선판을 유지한다.</strong> 새 두 페어 필터는 8월에서 비용 차감 PnL과 Q를 높여 당시의 고정된 선정 기준을 통과했다. 그러나 9월 이후에는 두 지표가 낮아져 기존 개선판을 교체하지 않는다. 새 후보도 기존 1·2번보다 높은 성과지만 가장 좋은 기존 개선판을 대체할 추가 성과는 아니다.</p>
<p>초기자본 1M. 기준시장 과거 자료의 계좌 시뮬레이션이며 모의 체결 비용·수수료·펀딩과 종료 시 미청산 평가손익을 포함한다. 기간 경계는 12UTC(한국시간 21시)다. 반복 사용한 과거 개발 평가이며 독립 미래 검증으로 주장하지 않는다. 실제 운용은 변경하지 않았다.</p>
<h2>새로 구현한 것</h2><p>기존 모델은 고정 24시간 가격 변화로 진입을 판단했다. 이번에는 기존 실행 엔진의 부분 체결, 유료 잔량 보완·헤지 조정, 보유 연장, 손절과 청산을 그대로 적용한 가상 거래에서 페어 소유 장부의 순손익을 추출했다. 실제로 포지션이 0이 된 뒤에만 학습 가능한 정답으로 취급한다. 전량 미체결은 관측된 결과 0, 미청산은 아직 모르는 결과로 별도 보존한다.</p>
<p>367개 출발 기회 중 다음 봉이 없는 5개를 제외한 362개 가상 계좌를 재생했다. 청산하여 손익이 확정된 거래 188개(중도 철회·불균형 청산 포함), 전량 미체결 8개, 진입 조건 불충족 164개, 끝까지 관측되지 않은 열린 거래 2개다. 이들은 서로 겹치는 경로이며 196개의 독립 거래라는 뜻은 아니다.</p>
<p>일별 릿지 가치 모델 117개를 적합했다. 같은 페어의 보유 구간이 겹치면 가중치를 낮추고 최근 90일 중 해당 적합 시점 전에 끝난 거래만 사용했다. 원래 가격 예측·주문 수량·위험 한도·청산 규칙은 바꾸지 않고, 새 가치 추정이 양수인 기존 진입만 남긴다. 별도 여섯 페어 후보는 같은 추정으로 진입 순위를 매긴다.</p>
<p>두 종목의 비용을 구별해야 한다는 <a href="https://arxiv.org/html/2604.02035v1">2026년 연구의 Remark 2.2</a>, 의사결정 점수를 평균 수익 예측으로 간주할 수 없다는 <a href="https://arxiv.org/html/2609.33472v1">2026년 연구</a>와 연결된다. 논문의 RL 알고리즘이나 성과를 재현한 것이 아니라, 기존 SPO 방향 점수와 별도의 실제 모의 손익 목표를 분리한 구현이다.</p>'''
    csv_rows=[]
    for phase in ['validation','comparison']:
        for sc in ['base','stress']:
            period='8월 1일~9월 1일' if phase=='validation' else '9월 1일~10월 15일'
            page+=f'<h2>{period} · {"기본 비용" if sc=="base" else "불리한 체결 조건"}</h2><table><tr><th>전략</th><th>순손익</th><th>수익률</th><th>기간 σ</th><th>Q</th><th>최대낙폭</th><th>완료 페어</th></tr>'
            for arm,label in labels_by_arm.items():
                m=lookup['reference',phase,arm,sc]
                page+=f"<tr><td>{label}</td><td>{m['net_pnl']:+,.2f}</td><td>{100*m['roi']:+.6f}%</td><td>{100*m['volatility_horizon']:.6f}%</td><td>{m['financial_score']:.8f}</td><td>{100*m['max_drawdown']:.6f}%</td><td>{m['completed_pairs']}</td></tr>"
                csv_rows.append(dict(phase=phase,scenario=sc,arm=arm,**{k:m[k] for k in ['net_pnl','roi','volatility_horizon','financial_score','max_drawdown','completed_pairs']}))
            page+='</table>'
    page+='''<h2>손익 차이를 만든 판단</h2>
<p>8월에는 BTC/ETH의 손실 진입을 걸러 기본 비용 +431.36, 불리한 조건 +709.36의 계좌 개선이 관측됐다. 반면 9월 18일 BTC/HYPE 진입은 기존 14UTC에서 새 필터 16UTC로 두 시간 늦어졌다. 해당 페어 손익은 기본 +3,983.03에서 +3,441.54, 불리한 조건 +3,774.54에서 +3,237.41로 줄었다. 다른 거래와 기본 전략의 미세한 계좌 영향까지 합친 이후 기간 총손익 차이는 −538.63/−534.47이다. 특정 날짜를 알고 제외 조건을 수정하지 않았다.</p>
<p>이후 기간에 당시 발행한 예측으로 평가 가능한 가상 거래 50개에서 순손익 수익률 예측 RMSE는 259.82bp, 0 예측 기준은 231.93bp였다. 모델이 양수로 본 38개 기회의 평균 실제 수익률은 −69.91bp였다. 이 표본도 겹치는 가상 거래이며 실제 계좌 수익률로 합산하지 않는다. 청산 손익을 정답으로 바꾼 것만으로 예측력이 좋아졌다는 증거는 얻지 못했다.</p>
<h2>par와 재현 검사</h2><p>두 페어 필터의 과거 par 평가 8조건은 기존 개선판과 체결·보유·손익 경로가 정확히 같다. 여섯 페어 순위판도 새 par 성과를 만들지 못했다. 다섯 과거 호가 테이프에서 두 페어 필터 진입은 0개, 여섯 페어 순위판은 기존 여섯 페어와 같은 EWY/EWJ 진입 시험 4개를 만들었다. 24시간 뒤 청산 호가는 관측되지 않아 전체 거래 수익률이나 Q를 계산하지 않았다.</p>
<p>새 포트폴리오 계좌 24개, 기존 계좌 재현 12개, 의미 있는 검사 8개를 완료했다. 첫 비교 실행의 배열·목록 표현 차이 오류를 별도 재개 스크립트에서 정규화했고 학습된 모델은 그대로 사용했다. 최종 12개 대조에서는 주문·의사결정·체결·잔고·손익이 모두 정확히 같았다. 보호 소스 373개, 예약 의존 345개, 향후 평가 소스 1,045개를 확인했고 예약된 미래 원시 자료는 열지 않았다.</p>
<h2>다음 판단</h2><p>추가 필터의 음수 추정이 얼마나 안정적인지 먼저 측정한다. 추정이 약하게 음수라는 이유로 기존의 좋은 진입을 막는 문제를 구분해야 한다. 미래 결과를 사용하지 않는 시간 블록별 재학습으로 추정의 부호와 불확실성을 확인하고, 필요하면 별도 규칙을 미리 고정해 평가한다. 지금 확인한 두 거래에 맞춘 임계값 탐색이나 실제 전략 자동 교체는 하지 않는다.</p>
<p><a href="../decision_gate_pairs_20261017/report.html">보존한 개선판</a> · <a href="../decision_gate_robustness_20261017/report.html">기존 개선판의 위험 규모·성과 집중도 검사</a></p></html>'''
    (dest/'report.html').write_text(page)
    with (dest/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    state=dict(conclusion,report=str((dest/'report.html').relative_to(ROOT)))
    save(HERE/'work_state.json',state)
    cross=ROOT/'experiments/pairs_development_status_20261017/comparisons/lifecycle_value_review.json'
    assert not cross.exists();save(cross,state)
    files=[Path(__file__),OUT/'conclusion.json',OUT/'results.json',OUT/'selection.json',OUT/'labels.json',OUT/'models.json',
           OUT/'diagnostics.json',OUT/'books/results.json',dest/'report.html',dest/'results.csv',cross,*paths]
    save(OUT/'conclusion_frozen.json',dict(at_unix=time.time(),sources={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('DONE lifecycle evaluation; August gain, later decline; retain decisiongate',flush=True)


if __name__=='__main__':main()
