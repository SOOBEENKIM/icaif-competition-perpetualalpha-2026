"""Report entry validity and actual quote feasibility without inventing exits."""
import json
from pathlib import Path
from experiments.par_direct_forecast_20261017.data import ROOT, save, sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.netted_shadow_pairs_20261017.report import document, table, export_csv
from experiments.dense_history_pairs_20261017.report import grid

HERE=Path(__file__).resolve().parent;ART=HERE/'artifacts'


def main():
    r=json.loads((ART/'results.json').read_text());a=json.loads((ART/'attribution.json').read_text())
    b=json.loads((ART/'books/results.json').read_text());edge=json.loads((ART/'books/edge_attribution.json').read_text())
    for path in [ART/'frozen.json',ART/'books/frozen.json']:
        for rel,digest in json.loads(path.read_text())['sources'].items():assert sha(ROOT/rel)==digest,rel
    guards=protected();out=ROOT/'reports/forecast_validity_pairs_20261017';out.mkdir(exist_ok=True)
    selected=[]
    for phase in ['validation','comparison']:
        for arm in ['first_strategy','second_strategy','first_dense_renewhedge_ridge_24','first_dense_refreshhedge_ridge_24',
                    'first_dense_refreshhedge_bias_24','first_dense_refreshhedge_affine_72']:
            for sc in ['base','stress']:
                m=next(x['metrics'] for x in r['rows'] if (x['venue'],x['phase'],x['arm'],x['scenario'])==('reference',phase,arm,sc))
                selected.append([phase,arm,sc,f"{m['net_pnl']:+,.2f}",f"{100*m['return_fraction']:+.6f}",
                                 f"{m['financial_score']:.8f}",f"{100*m['volatility_horizon']:.6f}",m['completed_pairs'],m['aborted_pairs']])
    book_rows=[]
    for t in b['trials']:
        book_rows.append([t['model'],t['horizon'],t['validity_hours'],t['scenario'],'/'.join(t['pair']),t['direction'],
                          t['source_age_hours'],t['status'],f"{t['residual_gross']:,.2f}",f"{t['known_fees']:,.4f}",
                          f"{t['plan']['expected_net_cash']:+,.4f}"])
    edge_rows=[]
    for row in edge['rows']:
        edge_rows.append([row['model'],row['horizon'],row['scenario'],row['source_age_hours'],*[f'{row[k]:+,.4f}' for k in [
            'planned_model_cash','reference_age_cash','venue_to_current_reference_cash','expected_funding_cash',
            'entry_fee','exit_cost_reserve','expected_net_cash','net_without_reference_age']]])
    body='''<h1>고정 만기 페어 예측의 진입 유효기간과 실제 par 호가</h1>
<p class="note"><strong>유효기간 확대만으로 안정적인 개선은 나오지 않았다. 그러나 실제 par 호가에서 진입 가능한 수량과 비용을 확인했다.</strong>
128개 새 계좌 재현, 대조군 확인 재현 4개, 기존 대조군 400개를 비교했다. 유효기간·계좌 검사 8개와 주문장 처리 검사 8개를 통과했다.
실제 거래소에 보낸 주문은 0개이며 실거래 전략은 변경하지 않았다.</p>
<h2>바꾼 부분</h2><p>24시간·72시간 만기 예측은 원래 6시간마다 갱신하면서 새 진입에는 1시간만 사용했다.
새 후보는 이미 도착했고 유효했던 동일 예측을 다음 6시간 갱신 전까지 현재 가격과 비용으로 다시 평가한다.
원래 예측 시각·가격·헤지 비율·변동성·모델 예측값은 그대로이며 진입이 늦어져도 만기를 뒤로 밀지 않는다.
새 시각의 예측이 없거나 유효하지 않으면 이전 예측을 부활시키지 않는다.
수량 조정·손절·세 번의 미체결 시도 제한·위험 한도는 직전 후보와 같다.</p>
<h2>기준시장 계좌 비교</h2><p>초기자본 1,000,000. validation은 2026-08-01 12:00~09-01 12:00 UTC,
comparison은 09-01 12:00~10-15 12:00 UTC다. base는 편도 수수료 9bp·불리한 가격 10bp,
stress는 12bp·20bp, 더 작은 거래량 참여율과 펀딩 수령액 절반이다.
아래는 재사용한 기준시장 과거 자료의 조건부 백테스트이며 실제 par 지갑 수익이 아니다.
σ는 완료 일별 수익률 표본 표준편차×√n, Q=(max(R,0)+0.005)/(abs(R)+1.5σ+0.05)다.</p>'''
    body+=grid(['기간','전략','조건','순손익','수익률 %','Q','기간 σ %','완료 페어','진입 후 철수'],selected)
    body+='''<p>24시간 릿지 확대판은 8월에 기본 +18,518.91·Q 0.250331,
불리한 조건 +19,857.56·Q 0.279754로 기존 두 원전략보다 좋았다.
그러나 이후 구간에서 기본 +1,456.10·Q 0.096208, 불리한 조건 −5,572.07·Q 0.070276이 되어 개선이 유지되지 않았다.</p>
<p>사전에 고정한 내부 선정 기준에는 원전략보다 높은 PnL·Q 외에 직전 1시간 후보보다 나쁘지 않을 것도 포함했다.
8월 릿지 확대판은 기본 조건에서 직전 후보보다 나빠 이 기준을 통과하지 못했다.
이를 원전략 비교까지 실패했다고 표현하지 않는다. 이후 구간 계산 전 고정한 진단 대표는 first_dense_refreshhedge_ridge_24이며 eligible=false다.
어느 해석으로도 이후 stress 손실은 교체를 뒷받침하지 않는다.</p>
<h2>실제 par 호가의 진입 결과</h2><p>허용된 기존 호가 기록 다섯 개, 총 약 4.48시간을 사용했다.
신호·판단 시각을 기록에 맞춰 옮기지 않고 원래 15분 판단 시각에 도착한 호가·자산 컨텍스트만 사용했다.
각 페어는 당시 보이는 수량과 독립 지정가 한도로 주문하고, 1초/3초 후 새로 도착한 주문장에서 각각 체결했다.
한쪽 미체결이면 확인 후 한 번만 보완하며 결과를 모르면 새 주문을 보내지 않는다.</p>
<p><strong>조건별 진입 시뮬레이션 37개 중 35개는 양쪽 보유, 2개는 미체결이었다.</strong>
모두 6시간 유효기간 조건이며 1시간 조건의 진입은 없었다.
35개는 같은 구간을 모델·만기·비용 조건별로 반복한 결과다. 기록·페어·판단시각·방향을 합친 고유 조합은 8개이고,
그 8개도 서로 겹치므로 독립 거래 표본이라고 볼 수 없다.</p>
<p>이 검사는 초기 보유 없는 1,000,000 계좌의 페어 진입 가능성 검사이며 원전략과 함께 운용한 포트폴리오 성과가 아니다.
양쪽이 체결된 후 남은 포지션을 청산된 수익으로 계산하지 않았다. 24/72시간 뒤 실제 청산 호가가 없어
이 자료의 계좌 수익률과 Q는 계산하지 않는다. 아래 기대 순현금은 모델 판단값이지 실현손익이 아니다.</p>'''
    body+=grid(['모델','만기 h','유효기간 h','조건','페어','방향','예측 나이 h','상태','남은 명목금액','확인된 수수료','예상 순현금'],book_rows)
    body+='''<h2>예상 수익의 출처 분리</h2><p>예상 가격손익을 세 부분으로 나눴다.
모델 증분은 원래 기준가격에서 예측 만기 가격까지의 변화다.
기준가격 경과분은 현재 기준시장에서 원래 기준가격으로 돌아가는 현금 변화다.
거래소 괴리는 현재 기준가격과 실제 진입 호가 사이의 차이다.
이 세 값에 예상 펀딩을 더하고 진입 수수료·예상 청산 비용을 빼면 원래 기대 순현금과 일치한다.
오라클이나 기준시장 가격을 실제 체결가로 사용하지 않았다.</p>
<p><strong>양쪽 체결 조건 35개 중 21개는 기준가격 경과분을 제외하면 5bp 진입 기준을 넘지 못했다.</strong>
예측 나이는 1.5~4.5시간이었다. 이는 청산 수익 증거가 아니라 오래된 예측을 유지하는 가정이
진입 판단에 큰 영향을 준다는 진단이다. 예를 들어 24시간 릿지의 한 조건은 모델 증분 −5.86에 기준가격 경과분 +12.74가 더해져 진입했다.</p>'''
    body+=grid(['모델','만기 h','조건','예측 나이 h','모델 증분','기준가격 경과분','현재 기준시장 괴리','예상 펀딩',
                '진입 수수료','청산 비용 추정','예상 순현금','기준가격 경과분 제외'],edge_rows)
    body+='''<h2>판단과 다음 변경</h2><p>유효기간을 늘리는 것만으로는 부족하다.
다음에는 매시간 이미 완성된 기준시장 자료로 같은 고정 모델을 다시 계산하고,
최근 1시간 예측으로 현재 par 호가의 남은 기회를 판단하는 후보를 비교한다.
추론 시각을 바꾸는 이유는 오래된 기준가격으로 되돌아간다는 가정에 덜 의존하기 위해서다.
모델 가중치·학습 기간은 유지하고, 예측 시각별 만기와 실제 체결 비용을 그대로 기록한다.
호가 기록에 맞춰 특정 시간만 선택하지 않고 모든 시간에 동일 규칙을 적용한다.
아직 이 다음 후보의 성능은 계산하지 않았다.</p>
<p>par 캔들·펀딩 계좌 재현은 추가 진입이 늘었지만 의미 있는 PnL·Q 개선을 보이지 않았다.
일부 판단에서 캔들 거래량 모형은 극소량만 허용한 반면 실제 주문장에는 그보다 큰 진입 수량이 있었다.
따라서 캔들 거래량을 실제 주문장 유동성의 증거로 해석하지 않는다.
장기 기준시장 평가와 짧은 실제 주문장 진입 검사를 구분하고, 없는 청산 호가를 보간해 성과를 만들지 않는다.</p>
<p>모든 과거 자료는 여러 실험에서 사용한 개발 자료다. 독립적인 미래 검증을 완료했다고 주장하지 않는다.
실거래 소스 373개·예약 의존 파일 345개는 그대로이며 예약 미래 원자료는 열지 않았다.</p>'''
    for venue in ['reference','par']:
        body+='<h2>'+('기준시장 전체 결과' if venue=='reference' else 'par 캔들·펀딩 조건부 전체 결과')+'</h2>'
        body+=table([x for x in r['rows'] if x['venue']==venue])
    (out/'report.html').write_text(document('예측 유효기간과 실제 호가 진입',body));export_csv(out/'comparison.csv',r['rows'])
    summary=dict(selection=r['selection'],source_validity_tests=8,book_tests=8,new_candidate_replays=128,
                 control_verification_replays=4,reused_control_accounts=400,
                 book_entry_probes=len(b['trials']),book_statuses=b['aggregate_statuses'],book_edge_attribution=edge['summary'],
                 book_account_return=None,book_financial_score=None,
                 maximum_episode_accounting_error=a['maximum_episode_accounting_error'],
                 protected=guards,orders_sent=0,automatic_live_replacement=False,historical_independence=False)
    save(out/'summary.json',summary)
    save(HERE/'work_state.json',dict(
        status='progress_forecast_validity_replayed_actual_book_entries_reconciled_stale_reference_component_identified',
        previous_goal_turn='progress_paid_delta_rehedge_implemented_tested_replayed_attributed_no_qualified_replacement',
        report=str((out/'report.html').relative_to(ROOT)),authoritative_results='artifacts',
        tests_passed=16,new_candidate_replays=128,control_verification_replays=4,reused_control_accounts=400,
        selection=r['selection'],book_entry_probes=37,paired_entries=35,zero_fills=2,orders_sent=0,live_strategy_changed=False,
        findings=[
            'Six-hour entry validity preserves original source prices, maturity and forecasts; missing new sources invalidate the old forecast.',
            'Ridge 24h improves August PnL/Q versus both native strategies, but not versus the one-hour matched base control; later stress loses 5572.07.',
            'Actual old received books produce 35 paired entries and 2 zero-fill probes across repeated model/scenario conditions; eight unique tape/pair/time/direction groups, not independent trades.',
            'Full 24/72h exit books are absent, so no account ROI/Q is reported for entry probes.',
            '21 of 35 paired-entry forecasts fail their original cost buffer if the source-to-current reference price component is removed; source age 1.5 to 4.5h.',
            'Predicted edge decomposition reconciles model increment, reference aging, venue difference, funding and costs. No projected cash is credited as profit.',
        ],
        next='Compare fresh hourly inference from the same frozen 24/72h models against one-hour validity/six-hour refresh and extended-validity controls. Use only completed reference bars and mature online calibration, globally fixed source cadence, fixed source maturity, independent paid execution and strict separation of book entry feasibility from unobserved full exits. Inspect existing hourly prediction artifacts before recomputing. Do not choose hours to align to old tapes, retrain on later outcomes, read reserved raw data or automatically deploy.',
        remaining=['No cross-period robust PnL/Q improvement','Actual full par book lifecycle profitability unverified','Independent prospective evidence absent']))
    print(json.dumps(summary,ensure_ascii=False));print(out/'report.html')


if __name__=='__main__':main()
