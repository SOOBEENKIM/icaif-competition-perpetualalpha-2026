import html,json
from pathlib import Path
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.netted_shadow_pairs_20261017.report import document,table,export_csv
from experiments.dense_history_pairs_20261017.report import grid

HERE=Path(__file__).resolve().parent;ART=HERE/'artifacts_v2'


def main():
    r=json.loads((ART/'results.json').read_text());a=json.loads((ART/'attribution.json').read_text())
    patterns=json.loads((ART/'reentry_patterns.json').read_text());diag=json.loads((ART/'diagnostic_status.json').read_text())
    for folder in ['artifacts','artifacts_v2']:
        frozen=json.loads((HERE/folder/'frozen.json').read_text())
        for rel,digest in frozen['sources'].items():assert sha(ROOT/rel)==digest,rel
    guards=protected();out=ROOT/'reports/adaptive_residual_pairs_20261017';out.mkdir(exist_ok=True)
    comparison=[]
    for phase in ['validation','comparison']:
        for arm in ['first_strategy','second_strategy','first_dense_bias_24','first_dense_bias_72','first_dense_affine_72']:
            for scenario in ['base','stress']:
                x=next(x for x in r['rows'] if (x['venue'],x['phase'],x['arm'],x['scenario'])==('reference',phase,arm,scenario));m=x['metrics']
                comparison.append([phase,arm,scenario,f"{m['net_pnl']:+,.2f}",f"{m['financial_score']:.8f}",f"{100*m['volatility_horizon']:.6f}",m['completed_pairs']])
    costs=[];repeat_rows=[]
    for x in a['rows']:
        if x['venue']=='reference' and x['phase']=='comparison' and x['arm'] in ['first_dense_bias_24','first_dense_bias_72','first_dense_affine_72']:
            g=x['aggregate'];costs.append([x['arm'],x['scenario'],*[f'{g[k]:+,.2f}' for k in ['price_move','execution_cost','fees','funding','net_pnl']],
                f"{x['groups']['aborted_closed']['net_pnl']:+,.2f}"])
    for x in patterns['rows']:
        if x['venue']=='reference' and x['phase']=='comparison' and x['arm'] in ['first_dense_bias_24','first_dense_bias_72','first_dense_affine_72']:
            repeat_rows.append([x['arm'],x['scenario'],x['pattern_count'],sum(p['gap_seconds']==0 for p in x['patterns']),
                sum(p['new_source_available_at_previous_close_decision'] for p in x['patterns'])])
    body='''<h1>최근 예측 오차 보정과 반복 청산 비용 점검</h1>
<p class="note"><strong>최근 오차를 보정한 12개 후보도 교체 기준을 통과하지 못했다.</strong> 24시간 평균오차 보정은 8월 기본·불리한 조건에서 기존 두 전략보다 PnL·Q가 높았으나, 불리한 조건의 완료 페어가 1건이었고 이후 구간에서는 큰 손실을 냈다. 새 계좌 재현 96개와 검증된 기존 대조군 80개를 비교했다. 실거래 전략은 변경하지 않았다.</p>
<h2>무엇을 수정했는가</h2><p>직전 장기 Binance 학습 릿지 가중치는 고정했다. 같은 페어의 최근 Hyperliquid에서 만기가 이미 끝난 가격 예측만 골라 예측을 보정한다. 각 당시의 변동성 단위로 정규화한 24시간·72시간 목표이며, 90일 범위·30일 반감기의 가중치를 사용한다. 같은 페어의 학습 구간은 서로 겹치지 않게 만기 간격으로 골랐다. 최소 12개 관측과 12개 사전 관측에 해당하는 축소를 고정했고 결과에 따라 바꾸지 않았다.</p>
<ul><li><strong>bias:</strong> 장기 모델 예측에 최근의 평균 예측 오차를 축소해 더한다.</li>
<li><strong>affine:</strong> 예측의 절편과 배율을 함께 보정한다. 절편 0·배율 1을 사전 기준으로 두고 배율을 −2~2로 제한한다. 실제 과거 관계가 바뀌면 반대 방향도 추정할 수 있다.</li>
<li><strong>recent:</strong> 장기 모델 없이 최근에 만기가 끝난 페어 수익의 가중 평균만 사용한다. 복잡한 보정이 단순한 최근 평균보다 나은지 확인한다.</li></ul>
<p>세 방법을 첫째·둘째 전략 및 24/72시간에 결합했다. 0 가격 예측과 고정 릿지의 기존 결과도 대조군으로 포함한다. 진입 주기, 유효기간, 자본, 위험 한도, 두 주문의 독립 체결, 한쪽 미체결의 보완·철수, 청산 만기와 비용은 직전 실험과 같다. 온라인 보정은 그때까지 완성된 목표로만 갱신하는 고정된 알고리즘이며, 이후 결과를 미리 사용한 재학습이 아니다.</p>
<h2>학습 누출과 대조군 검증</h2><p>미래 목표·가격을 바꾸거나 자료를 중간에서 잘라도 그 이전의 예측·보정 계수는 같았다. 미성숙 목표를 보정기에 넘기면 실패하고, 표본이 부족하면 새 예측을 만들지 않는다. 새 보정 검사 7개를 통과했다. 기존 장기 예측 배열과 실행 소스가 동일함을 검증하고 기존 대조군 80개의 파일 해시·지표를 대조했다.</p>
<p>첫 시도는 기존 예측 배열의 엄격한 일치 검사에서 중단됐다. 곱셈 결합 순서를 바꿔 발생한 최대 1.74×10⁻¹⁸ 차이였으며 계좌 재현 전이었다. 기존 순서를 보존하는 v2로 수정했고, 실패한 코드·기록을 그대로 남겼다. 허용 오차를 넓혀 통과시키지 않았다.</p>
<h2>기준시장 결과</h2><p>초기자본 1,000,000. validation은 2026-08-01 12:00~09-01 12:00 UTC, comparison은 09-01 12:00~10-15 12:00 UTC. base: 편도 수수료 9bp·불리한 가격 10bp. stress: 12bp·20bp, 더 작은 체결 참여율, 펀딩 수령액 50%. 아래는 기준시장 백테스트이며 실제 par 지갑 수익이 아니다.</p>'''+grid(['기간','전략','조건','비용 차감 손익','Q','기간 σ %','완료 페어'],comparison)+'''
<p>선정 기준은 양의 수익, 기존 두 전략보다 높은 PnL과 Q, 최소 3개 완료 페어, 위험 한도 위반 없음이 두 비용 조건에서 모두 성립하는 것이다. 8월 결과로 진단 대표 first_dense_bias_24를 고정한 뒤 이후 계좌 결과를 계산했다. 자격은 false다. 이후 기간의 다른 후보 결과를 보고 대표를 바꾸지 않았다.</p>
<p>이후 불리한 조건의 affine 72시간은 손실 −20.64로 기존 첫째의 −457.10보다 작았지만 Q는 0.077617로 기존 0.084370보다 낮았다. 기간 변동성이 증가했기 때문이다. 손익 한 숫자의 개선만으로 목표를 달성했다고 판단하지 않는다.</p>
<h2>가격 변화와 비용을 분리한 결과</h2><p>다음은 계좌 전체가 아닌 페어 소유 원장의 이후 구간 손익이다. 실행 비용과 수수료 열의 양수는 차감액이다. 부분체결 철수 손익은 총손익에 이미 포함된 부분집합이므로 다시 더하지 않는다. 펀딩·수수료·보유 합계는 물리 계좌와 일치하도록 검증했다.</p>'''+grid(['후보','조건','가격 변화','실행 비용','수수료','펀딩','페어 순손익','부분체결 철수 손익'],costs)+'''
<p>24시간 bias의 기본 조건은 비용 전 가격 변화부터 −2,630.06으로 예측 문제가 남아 있다. 72시간 bias는 가격 변화 +3,633.47을 얻었으나 실행 비용 2,880.40과 수수료 2,577.96, 펀딩 지급을 빼면 −1,953.36이다. 손실을 전부 한쪽 미체결 탓으로 돌릴 수 없다. 정상적으로 양쪽을 보유한 거래에서도 비용을 넘지 못했다.</p>
<h2>다음 실행 개선의 구체적 근거</h2><p>만기 청산 직후 같은 페어·같은 방향으로 다시 진입한 패턴을 발견했다. 아래는 원래 기록의 기술적 집계이며, 청산을 생략했을 때의 수익을 계산한 결과가 아니다. 실제로 비용을 얼마 아낄 수 있을지는 새 계좌 재현이 필요하다.</p>'''+grid(['후보','조건','6시간 이내 같은 방향 재진입','청산 시각에 즉시 새 진입 의도','기존 청산 결정 때 새 신호를 알 수 있었던 수'],repeat_rows)+'''
<p>특히 bias·affine 72시간 기본 조건은 이런 패턴이 각각 8회다. 그러나 청산 시각에 나온 새 신호를 한 단계 전의 청산 결정에 사용하면 미래 정보 누출이다. 따라서 새 후보는 <strong>기존 만기 전에 실제로 도착한 새 예측</strong>만으로 보유 연장을 결정해야 한다. 단순히 과거 청산 수수료를 빼서 가상의 개선 수익을 만들 수 없다.</p>
<p>다음에는 만기 전 마지막 정규 판단에서 같은 페어·방향의 새 예측을 받고, 현재 보유 수량의 미래 예상 현금 흐름과 즉시 청산의 비용을 비교한다. 유지할 가치가 있으면 추가 수량 없이 보유를 연장하고, 원래 손절 기준과 최대 보유시간은 유지한다. 손절·위험 한도·미완료 청산을 연장으로 취소하지 않는다. 새 예측이 없다면 원래 청산을 따른다. 갱신을 새로운 완료 거래로 세지 않는다.</p>
<h2>대회 시장 자료와 연구의 한계</h2><p>par의 10/01~10/16 캔들·펀딩 재현에서는 대부분 추가 페어가 없거나 한쪽 체결 후 작은 비용을 내고 철수했고, 교체할 만한 개선은 확인하지 못했다. 캔들 거래량은 체결 대용치다. 기존 허용 호가 기록 다섯 개는 이번에도 동일한 진입 시간 창과 겹치지 않으므로 실제 호가 진입·만기 청산 성과를 검증했다고 주장하지 않는다. 예약된 미래 원자료는 열지 않았다.</p>
<p>평가 과거는 여러 실험에서 재사용한 개발 자료다. 과거 시점마다 원인을 분리한 검증은 가능하지만 독립적인 미래 개선 증거는 아니다. 실제 주문 0건, 자동 실거래 교체 없음. 보호된 실거래 소스 373개와 예약 의존 파일 345개는 그대로다.</p>
<h2>논문과 이번 구현의 관계</h2><p><a href="https://arxiv.org/html/2606.04574v2">2026년 암호화폐 다중 페어 연구</a>는 통계 신호, 실행 정책, 결정적 위험 한도의 분리를 다룬다. 이번 잔차 보정은 자체 비교 실험이며 해당 논문의 PPO-LSTM을 재현한 구현이 아니다. <a href="https://arxiv.org/abs/1911.10450">거래 비용을 포함한 페어 최적제어 연구</a>도 진입·청산 시점과 규모를 비용과 함께 다룬다. 이론적 동기가 우리 시장의 수익성 증거를 대신하지는 않는다.</p>'''
    for venue in ['reference','par']:body+='<h2>'+('기준시장 전체 결과' if venue=='reference' else 'par 조건부 체결 전체 결과')+'</h2>'+table([x for x in r['rows'] if x['venue']==venue])
    (out/'report.html').write_text(document('최근 오차 보정과 반복 비용 점검',body));export_csv(out/'comparison.csv',r['rows'])
    selected=next(x for x in patterns['rows'] if (x['venue'],x['phase'],x['arm'],x['scenario'])==('reference','comparison','first_dense_bias_72','base'))
    summary=dict(selection=r['selection'],actual_replays=r['actual_replays'],reused_control_accounts=r['reused_control_accounts'],tests_passed=7,
        source_and_forecast_controls_exact=True,maximum_ownership_error=max(x['metrics']['maximum_ownership_error'] for x in r['rows']),
        maximum_episode_accounting_error=diag['maximum_episode_accounting_error'],
        bias_72_later_base_same_direction_reentries=selected['pattern_count'],
        protected=guards,orders_sent=0,automatic_live_replacement=False,historical_independence=False)
    save(out/'summary.json',summary)
    save(HERE/'work_state.json',dict(status='progress_recent_mature_residual_calibration_compared_costly_renewals_identified',
        previous_goal_turn='progress_dense_model_account_replays_and_attribution',authoritative_results='artifacts_v2',
        report=str((out/'report.html').relative_to(ROOT)),actual_replays=96,reused_control_accounts=80,tests_passed=7,
        orders_sent=0,live_strategy_changed=False,selection=r['selection'],
        findings=['Bias 24h improves August PnL/Q in both cost scenarios, but only one completed stress pair; fails selection and loses later.',
            'Recent-domain residual calibration does not establish robust improvement; all 12 candidates retained.',
            'Bias 72h later base pair price move +3633.47 is outweighed by execution costs 2880.40 and fees 2577.96.',
            'Normal completed pair trades also fail after costs; partial-fill aborts are not the only cause.',
            'Bias and affine 72h later base each close then re-enter the same pair/direction eight times.',
            'New signal at close timestamp is not available one execution step before; naive removal of old closes would leak future information.',
            'Original control multiplication order restored after strict equality caught 1.74e-18 roundoff before any replay.'],
        next='Implement causal, inventory-preserving forecast renewal before fixed maturity. Use only a fresh signal available at the last normal six-hour decision before expiry, value existing fixed quantities against immediate paid close, preserve original risk-stop anchor and hard maximum holding time, never reset incomplete exits, count actual closures only. Compare original fixed-maturity controls under identical risk/costs; do not subtract fees retrospectively or use close-time signals early.',
        remaining=['No candidate meets PnL/Q and robustness requirements','Full par quote lifecycle profitability unverified','Independent prospective improvement unverified']))
    print(json.dumps(summary,ensure_ascii=False));print(out/'report.html')


if __name__=='__main__':main()
