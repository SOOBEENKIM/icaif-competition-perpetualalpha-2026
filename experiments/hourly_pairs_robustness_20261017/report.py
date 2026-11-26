"""Publish robustness evidence; leave the tested candidate and selection intact."""
import json
from pathlib import Path
from experiments.par_direct_forecast_20261017.data import ROOT, save, sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.netted_shadow_pairs_20261017.report import document, table, export_csv
from experiments.dense_history_pairs_20261017.report import grid

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'hourly_refresh_pairs_20261017'


def main():
    spec=json.loads((HERE/'protocol.json').read_text())
    diagnostic=json.loads((HERE/'diagnostic_artifacts/results.json').read_text())
    scaled=json.loads((HERE/'scale_artifacts/results.json').read_text())
    original=json.loads((PRIOR/'artifacts/results.json').read_text())
    official=json.loads((HERE/'official_leaderboard.json').read_text())
    for folder in ['diagnostic_artifacts','scale_artifacts']:
        for rel,digest in json.loads((HERE/folder/'frozen.json').read_text())['sources'].items():
            assert sha(ROOT/rel)==digest,rel
    score=official['scoring']
    assert (score['lambda'],score['c'],score['epsilon'],score['w_financial'],score['w_jury'])==(1.5,.05,.005,.8,.2)
    guards=protected();out=ROOT/'reports/hourly_pairs_robustness_20261017';out.mkdir(exist_ok=True)
    lookup={(x['phase'],x['scenario'],x['arm']):x['metrics'] for x in original['rows'] if x['venue']=='reference'}
    concentration=[];normalization=[];bootstrap=[];comparisons=[];episodes=[]
    for row in diagnostic['rows']:
        phase,sc=row['phase'],row['scenario'];m=row['metrics'];best=row['episodes'][row['best_episode']]
        concentration.append([phase,sc,f"{m['net_pnl']:+,.2f}",f"{m['pair_owned_pnl']:+,.2f}",
            f"{best['net_pnl']:+,.2f}",f"{100*row['best_episode_share_of_positive_episode_pnl']:.2f}",
            f"{row['without_best_episode']['net_pnl']:+,.2f}",f"{row['without_best_episode']['financial_score']:.8f}"])
        for e in row['episodes']:
            if not e['fills']:continue
            episodes.append([phase,sc,'/'.join(e['pair']),e['status'],f"{(e['end']-e['time'])/3600:.1f}",
                 *[f"{e[k]:+,.2f}" for k in ['price_move','execution_cost','fees','funding','net_pnl']]])
        for n in row['ex_post_risk_normalized']:
            normalization.append([phase,sc,n['baseline'],f"{n['hindsight_scale']:.6f}",
                f"{n['net_pnl']:+,.2f}",f"{n['pnl_advantage']:+,.2f}",f"{n['q_advantage']:+.8f}"])
        for b in row['paired_resampling']:
            bootstrap.append([phase,sc,b['baseline'],b['block_days'],
                 ' / '.join(f'{100*x:+.4f}' for x in b['return_fraction_difference_quantiles']),
                 ' / '.join(f'{x:+.6f}' for x in b['financial_score_difference_quantiles'])])
        for base in spec['baselines']:
            sm=next(x['metrics'] for x in scaled['rows'] if (x['phase'],x['scenario'],x['arm'])==(phase,sc,base+'_august_vol_matched'))
            comparisons.append(dict(phase=phase,scenario=sc,baseline=base,
                candidate_pnl=m['net_pnl'],scaled_baseline_pnl=sm['net_pnl'],pnl_advantage=m['net_pnl']-sm['net_pnl'],
                q_advantage=m['financial_score']-sm['financial_score'],
                candidate_sigma=m['volatility_horizon'],scaled_baseline_sigma=sm['volatility_horizon']))
    fits=[[f['baseline'],f['scale'],f"{100*f['target']:.6f}",f"{100*f['achieved']:.6f}",f"{100*f['relative_error']:+.3f}"] for f in scaled['fit']]
    compare_rows=[[c['phase'],c['scenario'],c['baseline'],f"{c['candidate_pnl']:+,.2f}",f"{c['scaled_baseline_pnl']:+,.2f}",
                   f"{c['pnl_advantage']:+,.2f}",f"{c['q_advantage']:+.8f}",
                   f"{100*c['candidate_sigma']:.6f}",f"{100*c['scaled_baseline_sigma']:.6f}"] for c in comparisons]
    body='''<h1>24시간 페어 후보: 수익 집중도와 투자 규모 효과</h1>
<p class="note"><strong>이후 기간의 개선은 단순한 규모 확대만으로 설명되지 않았다. 다만 가장 큰 수익 거래 한 건에 크게 의존했다.</strong>
기존 후보의 예측·주문 규칙과 사전 선정 결과는 보존했다. 원전략의 규모 대조군을 실제로 28회 재실행했고,
손익·시간 배분·표본 재추출 검사 4개를 통과했다. 이 보고서는 교체 완료나 새 실거래 수익을 뜻하지 않는다.</p>
<h2>비교 기간과 기준</h2><p>자본 1,000,000. validation은 2026-08-01 12:00~09-01 12:00 UTC,
comparison은 09-01 12:00~10-15 12:00 UTC다. 기준시장 과거 자료를 여러 번 재사용한 개발 평가다.
후보는 first_dense_hourlyhedge_bias_24이며, 기존 첫째 전략에 매시간 갱신하는 24시간 페어 예측과 최근 오차 보정을 더했다.
기본 조건 수수료·가격 불이익은 편도 9bp·10bp, stress는 12bp·20bp 및 더 작은 체결 참여율·펀딩 수령액 절반이다.</p>
<h2>가장 큰 수익 거래에 대한 의존</h2><p>각 페어 에피소드의 가격 변화, 실행 비용, 수수료, 펀딩을 시각별로 복원해 기존 계좌 손익과 대조했다.
가장 수익이 큰 에피소드의 누적 기여를 계좌 경로에서 빼고 PnL·Q를 다시 계산했다.
나머지 주문·자본 배분은 그대로 두는 회계 민감도 분석이다. 그 거래를 실제로 생략하고 재운용한 결과는 아니다.</p>'''
    body+=grid(['기간','조건','원래 계좌 순손익','페어 소유 손익','최대 수익 에피소드','양의 에피소드 손익 중 비중 %',
                '그 에피소드 제외 계좌 손익','제외 후 Q'],concentration)
    body+='''<p>이후 기본 조건은 +1,491.90에서 −1,397.15, stress는 +1,031.14에서 −2,906.75로 바뀐다.
최대 에피소드는 각각 페어의 양의 에피소드 손익 중 약 71.46%, 95.97%를 차지했다.
수익 거래 한 건을 제외해 손실이 된다는 사실만으로 전략이 잘못됐다고 단정하지 않는다.
다만 현재 개선을 여러 독립 거래가 반복해서 만든 성과로 보기에는 근거가 약하다.
체결된 페어는 두 기간 모두 BTC/HYPE였다. BTC/ETH 후보는 이 계좌 성과를 분산하지 못했다.</p>
<h2>기존 전략 규모를 실제로 바꾼 대조군</h2><p>8월 기본 조건의 후보 기간 변동성에 가까워지도록 원전략 목표 수량 배수를 정했다.
고정 격자와 두 번의 인접 구간 세분화에서 변동성 차이만 최소화했다. PnL·Q는 배수 선택에 쓰지 않았다.
이 배수를 저장한 뒤 stress와 이후 기간에 그대로 적용했다. 수량 반올림, 실제 체결 모형, 수수료, 위험 한도를 모두 다시 계산했다.
배수 1.0에서는 기존 원전략의 체결·계좌 경로·판단 이벤트·성과가 정확히 재현됐다.</p>'''
    body+=grid(['원전략','고정 배수','목표 σ %','8월 달성 σ %','상대 오차 %'],fits)
    body+=grid(['기간','조건','원전략 대조군','후보 PnL','규모 대조군 PnL','후보 PnL 우위','후보 Q 우위','후보 σ %','대조군 σ %'],compare_rows)
    body+='''<p>첫째 전략 1.4375배, 둘째 전략 1.6875배를 이후 기간에도 그대로 적용했다.
이후 기본·stress에서 후보 PnL·Q가 두 규모 대조군보다 높았다. 8월 기본에서도 후보가 높았지만,
8월 stress에서는 규모 대조군이 후보보다 PnL·Q 모두 높았다.
8월에서만 변동성을 맞췄으므로 이후 기간까지 정확히 동일한 위험이라고 표현하지 않는다.
이 대조군은 단순 규모 효과를 확인하기 위한 것이며, 사용자에게 규모 확대를 권하거나 설정에 적용한 것이 아니다.</p>
<h2>변동성을 사후에 정확히 맞춘 계산</h2><p>각 기간·조건의 후보 일별 수익률에 원전략 표준편차/후보 표준편차를 곱하고 복리 수익률과 Q를 다시 계산했다.
이는 위험 대비 성과를 보는 설명용 계산이다. 미래를 알고 정한 배수이고, 주문 크기 변화에 따른 체결·비용을 재현하지 않으므로 운용 전략으로 해석하지 않는다.</p>'''
    body+=grid(['기간','조건','비교 원전략','사후 배수','정규화 계좌 손익','원전략 대비 손익 차이','Q 차이'],normalization)
    body+='''<p>이 계산에서도 8월 stress의 우위는 사라지고, 나머지 조건의 PnL 우위는 유지됐다.
따라서 변동성을 추가 부담한 효과가 일부 존재하며, 모든 조건에서 동일 위험 우위가 입증된 것은 아니다.</p>
<h2>일별 성과 경로의 민감도</h2><p>동일한 날짜 인덱스로 후보와 원전략을 함께 재추출했다.
2일·7일 연속 블록을 각각 5,000회 연결하고 복리 수익률·Q 차이를 계산했다.
아래는 차이의 2.5%/50%/97.5% 분위수다. 모든 구간이 0을 포함한다.
반복 개발과 후보 선택을 보정한 통계 검정이 아니며, 미래 성공 확률이나 독립 표본 수로 해석하지 않는다.</p>'''
    body+=grid(['기간','조건','원전략','블록 일수','수익률 차이 %p: 하/중/상','Q 차이: 하/중/상'],bootstrap)
    body+='<h2>체결 에피소드별 비용과 손익</h2>'
    body+=grid(['기간','조건','페어','종료 상태','관찰 시간 h','가격 변화','실행 비용','수수료','펀딩','순손익'],episodes)
    body+='''<h2>다음 구현 판단</h2><p>후보는 유지하되 즉시 교체 근거로 삼지 않는다.
단순히 거래 수를 늘리거나 가장 큰 수익 거래를 제외해도 이익이 나도록 맞추는 것을 새 목표로 삼지 않는다.
다음 수정은 BTC/HYPE의 보유 연장 과정에서 새 예측이 실제 남은 수익 기회를 설명하는지,
손실 에피소드에서도 오래 보유하게 된 이유를 분해한 뒤 정한다. 기존 원전략 대비 PnL·Q라는 목표는 그대로다.</p>
<p>최근 2026년 암호화폐 페어 연구도 동적 공적분과 DNN/LSTM 앙상블을 다루지만,
거래 비용·유동성 제한·체결 마찰을 명시적으로 모형화하지 않았다고 밝힌다.
따라서 논문의 예측 개선을 이 대회의 비용 차감 수익 근거로 바로 사용하지 않는다.
<a href="https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2026.1749337/full">원문: 모형 가정과 한계</a>.</p>'''
    body+=f'''<p><a href="https://perpetual-alpha26.github.io/leaderboard#practice">공식 리더보드</a>가 제공한
{official['generated_at']} 공개 JSON에서 λ=1.5, c=0.05, ε=0.005, 재무 80%·심사 20%,
표본 표준편차×√n을 다시 확인했다. 이 보고서의 Q는 그 식을 사용하지만 실제 대회 순위표는 아니다.</p>
<p>실제 주문 전송 0개, 실거래 전략 변경 없음. 활성 소스 {guards['active_protected_sources']}개와
예약 의존 파일 {guards['reserved_dependencies']}개의 해시는 유지됐고, 예약 미래 원자료는 열지 않았다.</p>'''
    (out/'report.html').write_text(document('페어 후보 수익 집중도·규모 효과',body))
    export_csv(out/'scale_comparison.csv',scaled['rows'])
    save(out/'summary.json',dict(controls=comparisons,fit=scaled['fit'],actual_scale_replays=scaled['actual_account_replays'],
         tests_passed=4,unit_scale_controls_exact=2,protected=guards,orders_sent=0,historical_independence=False,
         official_scoring=score,official_generated_at=official['generated_at'],candidate_selection_unchanged=True))
    save(HERE/'work_state.json',dict(
        status='progress_hourly_bias24_concentration_and_actual_scale_controls_verified',
        previous_goal_turn='progress_hourly_bias24_reference_pnl_q_improvement_found_not_qualified',
        report=str((out/'report.html').relative_to(ROOT)),actual_scale_replays=28,tests_passed=4,
        orders_sent=0,live_strategy_changed=False,historical_independence=False,
        findings=[
            'Later first/second scaled controls are both worse than bias24 on PnL and Q; August stress scaled controls are better. Fixed scales fit August base volatility, not returns.',
            'Removing the best episode as a fixed-flow accounting diagnostic changes later account PnL to -1397.15 base and -2906.75 stress.',
            'Positive episode profit concentrates 71.46% and 95.97% in the best later episode; all filled pairs are BTC/HYPE.',
            'Descriptive paired 2/7-day block resampling intervals include zero for every PnL/Q difference; not selection-adjusted inference.',
            'Exact hindsight variance normalization removes August stress outperformance, while retaining other PnL improvements; not a tradable sizing rule.',
            'Official public leaderboard scoring constants reconfirmed unchanged on its 2026-10-17T09:59 generation.',
        ],
        next='Audit hourly bias24 renewal decisions and realized remaining-horizon forecast errors, including profitable and losing episodes, before proposing a causal renewal/forecast-calibration revision. Focus on source terminal value, cost of continued holding, stale stop risk and baseline interaction. Do not tune out specific losing dates, force best-trade removal profitability, weaken frozen selection, read reserved raw data or change live deployment.',
        remaining=['Cross-period improvement sensitive to few episodes','No uniform equal-risk advantage',
                   'Full par lifecycle profitability and independent prospective evidence absent']))
    print(out/'report.html')
    print('All frozen source hashes and protected live files verified.')


if __name__=='__main__':main()
