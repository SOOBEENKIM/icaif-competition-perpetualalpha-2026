import gzip,html,json
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha,ts
from experiments.complementary_alpha_20261017.research import protected
from experiments.netted_shadow_pairs_20261017.report import document,table,export_csv

HERE=Path(__file__).resolve().parent


def date(t):return datetime.fromtimestamp(t,timezone.utc).strftime('%Y-%m-%d %H:%M')


def grid(headers,rows):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'


def main():
    out=ROOT/'reports/event_value_pairs_20261017';out.mkdir(exist_ok=True)
    r=json.loads((HERE/'artifacts/results.json').read_text());events=json.loads((HERE/'artifacts/events.json').read_text())
    repair_path=HERE/'par_repair_artifacts/results.json';repair=json.loads(repair_path.read_text()) if repair_path.exists() else None
    if repair is not None:r['rows']=[x for x in r['rows'] if x['venue']!='par']+repair['rows']
    fits=json.loads((HERE/'artifacts/fits.json').read_text());signals=json.loads((HERE/'artifacts/signal_audits.json').read_text())
    frozen=json.loads((HERE/'artifacts/frozen.json').read_text())
    for rel,digest in frozen['sources'].items():assert sha(ROOT/rel)==digest,rel
    guards=protected();calibration=[];pair_weights=[]
    for f in fits:
        for group,v in f['groups'].items():
            calibration.append([date(f['time']),group,f['events'],v['events'],'pool' if v['pooled'] else 'group',f"{v['unshrunk_ratio']:.6f}" if v['ratio'] is not None else 'N/A',f"{v['ratio']:.6f}" if v['ratio'] is not None else 'N/A'])
        prior=[e for e in events if e['maturity'] is not None and e['maturity']<f['time'] and e['time']>=f['time']-180*86400]
        den=sum(e['expected_full_convergence']**2 for e in prior)
        for pair in sorted(set(tuple(e['pair']) for e in prior)):
            selected=[e for e in prior if tuple(e['pair'])==pair]
            x=np.array([e['expected_full_convergence'] for e in selected]);y=np.array([e['realized_price_return'] for e in selected])
            pair_weights.append(dict(fit_time=f['time'],pair=list(pair),events=len(selected),squared_gap_weight=float(x@x/den),
                unshrunk_slope=float(x@y/(x@x)),mean_price_bps=float(y.mean()*10000),positive_price_events=int(sum(y>0))))
    save(HERE/'artifacts/pair_weight_audit.json',dict(rows=pair_weights,post_evaluation=True,not_a_new_selection=True))
    snapshot=[]
    for phase in ('validation','comparison'):
        for arm in ('first_strategy','second_strategy','first_plus_pairs','first_confirmed_pairs','first_calibrated_pairs'):
            for scenario in ('base','stress'):
                row=next(v for v in r['rows'] if (v['venue'],v['phase'],v['arm'],v['scenario'])==('reference',phase,arm,scenario));m=row['metrics']
                snapshot.append([phase,arm,scenario,f"{m['net_pnl']:+,.2f}",f"{m['financial_score']:.8f}",m['completed_pairs'],f"{m['pnl_after_liquidation_reserve']:+,.2f}"])
    recent_weights=[[date(a['fit_time']),'/'.join(a['pair']),a['events'],f"{100*a['squared_gap_weight']:.2f}%",f"{a['unshrunk_slope']:.4f}",f"{a['mean_price_bps']:+.2f}"] for a in pair_weights if a['fit_time']>=ts('2026-09-01')]
    body='''<h1>페어 가격 회복 확인과 과거 사례 기반 예상 이익 보정</h1>
<p class="note"><strong>두 수정 방식 모두 이번 채택 조건을 통과하지 못했다.</strong> 가격 이동 확인 조건은 8월의 이익 거래도 제외했고, 공동 보정 모델은 9월부터 추가 페어 진입을 막았다. 기존 두 전략보다 일관되게 개선되는 후보를 확보한 상태는 아니다. 72개 계좌 재현을 보존했으며 실거래 주문과 교체는 0건이다.</p>
<h2>실험을 시작한 근거</h2><p>이전 릿지·트리·GRU·앙상블의 고정기간 수익률 예측은 채택 조건을 통과하지 못했다. 이번에는 현재 사용 중인 3σ 되돌림 신호와 그 신호가 예상하는 평균 회복 금액을 직접 점검했다. 통계량이 시시각각 달라지는 Z 점수의 변화에는 가격 이동과 추정 기준의 이동이 함께 들어간다. 이를 확인한 것이 곧 기존 논문 구현 오류를 뜻하지는 않는다.</p>
<h2>미리 정한 두 수정안</h2><p><strong>confirmed:</strong> 기존 신호가 발생한 시점의 헤지 비율·평균·표준편차를 이전 가격에도 동일하게 적용해, 실제 가격 차이가 3σ 경계를 안쪽으로 통과했는지 확인한다. 기존 신호 중 일부만 남기며 새로운 진입 시점은 만들지 않는다. 청산 규칙과 비용은 같다.</p>
<p><strong>calibrated:</strong> 과거의 같은 되돌림 사건이 기존 고정 평균·손절·시간 종료 규칙으로 끝났을 때, 실제 가격 이동이 원래 예상한 평균 회복 금액의 얼마였는지를 추정한다. 매월 1일에 그 시각보다 먼저 종료된 사례만 쓴다. 같은 페어의 보유기간이 겹치는 사건은 제외하고, 아직 종료되지 않은 사례를 수익률 0으로 넣지 않는다.</p>
<p>사건별 목표는 신호 가격에서 종료 판단 다음 시간봉 가격까지의 고정 수량 기준 현금 변화다. 수수료·펀딩을 포함하지 않은 가격 경로 목표이며 실제 체결 수익이 아니다. 최종 거래 실행기는 현재 진입 가격으로 다시 평가하고 수수료·펀딩·한쪽 미체결·청산 비용을 별도로 적용한다. 따라서 펀딩을 두 번 빼지 않는다.</p>
<p>다섯 사전 지정 페어에서 최소 24개의 완료 사례를 요구했다. 가격 회복 확인 여부에 따른 집단에 12개 이상이면 해당 집단을 쓰고, 부족하면 전체를 썼다. 비율은 Σ(예상 회복×실제 가격 이동)/[Σ(예상 회복²)×(1+24/사례 수)]로 추정해 0~1로 제한했다. 미래 종료 가격의 원래 예상 변위를 이 비율만큼 줄이고, 실제 청산 조건인 원래 평균·손절 기준은 유지했다. 임계값이나 배율 탐색은 하지 않았다.</p>
<h2>당시 사용 가능했던 추정치</h2><p>전체 보유기간 중복 제거 사건은 74개이며, 자료 끝까지 종료된 것은 71개다. 아래는 각 학습 시각 이전에 종료된 사건만 사용한 결과다. 표본이 적고 여러 페어가 같은 시기에 움직일 수 있으므로 독립 표본이나 유의성의 근거로 보지 않는다. 9월·10월 보정치는 두 집단 모두 0이었다.</p>'''+grid(['학습 UTC','동일 기준 가격 회복','전체 완료 사례','사용 사례','범위','보정 전 비율','적용 비율'],calibration)+'''
<h2>기준시장 계좌 성과</h2><p>초기자본 1,000,000. validation: 8/01 12:00–9/01 12:00 UTC, comparison: 9/01 12:00–10/15 12:00 UTC. base는 편도 수수료 9bp와 불리한 가격 이동 10bp, stress는 12bp·20bp와 더 작은 거래량 참여율 및 양의 펀딩 수령액 50%다. 같은 기간·자본·위험 한도로 비교했다. 기준시장 결과는 실제 par 계좌 수익이 아니다.</p>'''+grid(['기간','전략','비용','순손익','Q','완료 페어','청산비용 추정 차감 후'],snapshot)+'''
<p>8월 가격 회복 확인 후보는 첫째 기준 +13,047.40보다 낮은 +11,942.96이었다. 이후 기본 조건 +2,231.27, 불리한 조건 +15.06은 원래 혼합판과 같았지만, 앞선 구간의 악화를 뒤집는 독립 증거는 아니다. 보정 후보는 8월 +13,007.59로 기존보다 조금 낮았고 이후에는 보정치 0으로 신규 페어를 열지 않아 원래 첫째 전략과 동일했다. 8월로 선정한 진단 대표 first_calibrated_pairs의 채택 자격은 false다.</p>
<h2>추정치를 추가로 점검한 결과</h2><p>공동 보정은 페어별 분포 차이를 충분히 다루지 못했다. 예상 회복 폭의 제곱이 추정에 주는 비중을 분해하면, 9월 학습에서 HYPE/EWJ가 약 72%, 10월에서 약 68%였다. 그 페어의 몇몇 큰 변동이 다른 페어의 판단에도 영향을 주는 구조다. 이는 이번 공통 배율 모델의 한계이며 모든 페어에 이익 기회가 없다는 증거가 아니다. 반대로 BTC/ETH의 완료 사례도 9월 12개, 10월 15개에 불과해 작은 표본으로 복잡한 모델의 우월성을 주장할 수 없다.</p>'''+grid(['학습 UTC','페어','완료 사례','전체 제곱 회복 폭 비중','보정 전 비율','평균 가격 이동 bp'],recent_weights)+'''
<p>위 진단은 평가 후의 원인 분석이다. 이 결과를 보고 같은 평가 구간에서 유리한 페어만 골라 성과를 다시 제시하지 않았다. 분포 차이를 반영하는 보정이나 더 긴 과거 자료는 별도 사전 규칙과 비교가 필요하다.</p>
<h2>par와 검증 범위</h2><p>전체 par 기간은 캔들·펀딩과 조건부 체결 모형으로 평가했다. 새 페어 체결은 모두 0건이어서 혼합 후보는 해당 기존 전략과 같은 성과였다. 실제 호가 전체 이력이 확보된 결과가 아니므로 0건을 실제 체결 불가능의 증거로 삼지 않는다. 앞서 관측한 호가 구간에는 원래 페어 신호가 없었고, 이번 후보는 그 신호를 늘리지 않았으므로 새 실제 호가 수익 근거를 만든 것이 아니다.</p>
<p>핵심 검사 7개 통과. 실제 입력을 9월 이전으로 잘라 다시 계산했을 때 과거 신호와 9월 학습 결과가 같았다. 학습 종료 시각과 같은 시각에 완성된 목표도 제외했다. 기존 다섯 대조군의 체결과 계좌 이력은 모든 시장·기간·비용 조건에서 정확히 같았다. 보호된 실거래 소스 373개와 예약 의존 파일 345개는 변경하지 않았다.</p>
<p>모든 자료는 이전 개발에서도 사용한 과거 자료다. 후보 대표를 8월 결과로 고정한 후 이후 계좌 재현을 수행한 순서는 지켰지만, 이를 독립적인 미사용 검증이라고 부르지 않는다. 완성된 미래 호가를 과거로 대체하지 않았으며 실제 주문도 보내지 않았다.</p>
<p>연구 근거: <a href="https://arxiv.org/html/2606.04574v2">Dynamic Multi-Pair Trading Strategy in Cryptocurrency Markets with Deep Reinforcement Learning, §3.6</a>의 시장 상태와 보유 상태를 구분하는 통계량·고정 헤지 설명을 참고했다. 이번 가격 회복 확인과 과거 사례 배율 보정은 자체 가설이며 해당 논문의 PPO/LSTM 재현이나 논문에서 입증된 개선이라고 주장하지 않는다.</p>
<h2>다음 작업</h2><p>추가 조건을 같은 작은 표본에 반복해서 붙이기 전에, 더 긴 공개 과거 자료를 확보할 수 있는지와 페어별 추정의 표본 수를 점검한다. 가능하다면 학습 구간만 늘리고 기존 비교 구간·체결 비용·두 기준 전략은 고정한다. par 호가에 관한 검증 범위는 별도로 유지한다.</p>'''
    if repair is not None:
        body='<p class="note"><strong>par 자료 연결 정정:</strong> 원본과 par의 페어 배열 순서 차이를 종목 조합 기준으로 수정하고 36개 계좌를 재평가했다. 기존 계좌값·체결·PnL·Q는 모두 동일했다. 신호 귀속과 par 결과의 최신 근거는 <code>par_repair_artifacts</code>이며 초기 매핑 기록은 보존한다.</p>'+body
    for venue in ('reference','par'):
        body+='<h2>'+('기준시장 전체 결과' if venue=='reference' else 'par 캔들 모형의 조건부 전체 결과')+'</h2>'+table([a for a in r['rows'] if a['venue']==venue])
    (out/'report.html').write_text(document('페어 사건별 예상 이익 보정 검증',body));export_csv(out/'comparison.csv',r['rows'])
    summary=dict(selection=r['selection'],actual_replays=72,tests_passed=7,prefix_calibration_check=True,
        events=len(events),mature_events=sum(e['maturity'] is not None for e in events),all_controls_exact=True,
        orders_sent=0,automatic_live_replacement=False,historical_independence=False,protected=guards,
        maximum_ownership_error=max(a['metrics']['maximum_ownership_error'] for a in r['rows']))
    summary['par_mapping_repair']=None if repair is None else dict(replays=repair['actual_replays'],account_outputs_unchanged=repair['all_old_metrics_fills_snapshots_unchanged'],authoritative_directory='experiments/event_value_pairs_20261017/par_repair_artifacts')
    save(out/'summary.json',summary)
    save(HERE/'work_state.json',dict(status='progress_event_calibration_and_price_confirmation_compared_not_adopted',
        previous_goal_turn='progress',report=str((out/'report.html').relative_to(ROOT)),actual_replays=72,tests_passed=7,
        orders_sent=0,live_strategy_changed=False,
        findings=['Prior ridge/tree/GRU/ensemble families already failed; this study tests original event-value hypothesis.',
            'Same-current-parameter recross condition removes some statistically driven signals but worsens August PnL.',
            'Monthly matured event calibration maps September/October expected displacement to zero; no PnL improvement.',
            'Only 74 nonoverlapping-per-pair price-path events, 71 mature; current full-convergence assumption is poorly calibrated.',
            'Pooled squared-gap estimator dominated by HYPE/EWJ (72% in Sep, 68% in Oct), not uniform evidence against all pairs.',
            'All five controls reproduced exactly; future-prefix refit invariant verified. No actual par pair fill evidence.'],
        next='Continue from experiments/long_history_pairs_20261017/work_state.json. Older public history has been acquired and per-pair sparse-event predictors evaluated; preserve the repaired par identity mapping and all earlier outcomes.',
        remaining=['No eligible candidate under declared selection and later stress criteria','Actual par order-book coverage insufficient for full-period profitability evidence','No independent prospective improvement evidence']))
    print(json.dumps(summary,ensure_ascii=False));print(out/'report.html')


if __name__=='__main__':main()
