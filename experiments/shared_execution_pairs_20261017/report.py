import html,json,time
from pathlib import Path
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.netted_shadow_pairs_20261017.report import document,table,export_csv

HERE=Path(__file__).resolve().parent


def main():
    current=json.loads((HERE/'artifacts_v2/results.json').read_text());old=json.loads((HERE/'artifacts/results.json').read_text())
    for name in ('artifacts','artifacts_v2'):
        frozen=json.loads((HERE/name/'frozen.json').read_text())
        for rel,digest in frozen['sources'].items():assert sha(ROOT/rel)==digest,rel
    guards=protected();out=ROOT/'reports/shared_execution_pairs_20261017';out.mkdir(exist_ok=True)
    save(HERE/'artifacts/superseded.json',dict(reason='The first shared executor stopped every physical reversal at zero, even when the new pair leg was an opening order. This caused artificial incomplete pair legs. Preserve as diagnostic only; use account_v2 and artifacts_v2 for strategy comparisons.',replacement='artifacts_v2',orders_sent=0))
    overview=[]
    for row in current['rows']:
        if row['venue']=='reference' and row['phase']=='comparison' and row['arm']!='pairs_alone':
            m=row['metrics'];overview.append(dict(arm=row['arm'],scenario=row['scenario'],net_pnl=m['net_pnl'],Q=m['financial_score'],
                period_volatility=m['volatility_horizon'],mean_gross=m['mean_gross'],pnl_after_exit_reserve=m['pnl_after_liquidation_reserve']))
    rows=[]
    for r in overview:
        vals=[r['arm'],r['scenario'],f"{r['net_pnl']:+,.2f}",f"{r['Q']:.8f}",f"{100*r['period_volatility']:.4f}%",f"{100*r['mean_gross']:.3f}%",f"{r['pnl_after_exit_reserve']:+,.2f}"]
        rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in vals)+'</tr>')
    compact='<div class="scroll"><table><tr>'+''.join('<th>'+v+'</th>' for v in ['전략','비용 조건','순손익','Q','기간 변동성','평균 총노출','청산비용 추정 차감 후'])+'</tr>'+''.join(rows)+'</table></div>'
    body='''<h1>한 계좌에서 기존 전략과 페어 전략을 결합한 결과</h1>
<p class="note"><strong>실행 구조를 수정한 후보가 이후 기준시장 구간에서 기존 두 전략보다 PnL과 Q가 높았다.</strong> 다만 앞선 8월 선정 조건을 통과하지 못했고, 이후 구간의 불리한 비용 조건에서는 청산비용 여유도 부족하다. 따라서 검증된 교체 전략으로 채택하지 않는다. 아래 기준시장 결과는 실제 par 계좌 성과가 아니다. 실거래 주문·교체는 0건이다.</p>
<h2>이후 기준시장 비교: 2026-09-01 12:00–10-15 12:00 UTC</h2><p>초기자본 1,000,000, 같은 기간과 전역 위험 한도. base 편도 수수료 9bp·불리한 가격 변화 10bp; stress 12bp·20bp, 더 작은 거래량 참여율, 양의 펀딩 수령액 50% 반영. 표의 Q는 일별 12:00 UTC 계좌값으로 계산한 대회 재무 점수다. 마지막 포지션은 시가평가로 남기고 추가 청산비용 추정치를 별도로 차감했다.</p>'''+compact+'''
<p>8월 선정에서 미리 고정한 진단 대표는 <code>first_plus_pairs</code>다. 이후 기본 조건에서 +2,231.27 / Q 0.11888188, 불리한 조건에서 +15.06 / Q 0.08555826이었다. 이후 첫째 전략보다 기간 변동성은 조금 낮았지만, 평균 총노출은 4.18%에서 7.45%로 늘었다. 같은 자본·최대 위험 한도 안에서 페어를 추가한 결과이며, 수익 개선 전부를 동일 노출의 예측력 개선으로 해석하지 않는다. 노출/위험을 맞춘 별도 확대 대조군 비교는 다음 확인 대상이다.</p>
<p>불리한 비용 조건의 +15.06은 종료 포지션 청산비용 추정까지 차감하면 −120.61이다. 둘째 전략＋페어도 원래 두 전략보다 이후 PnL/Q는 높았지만 해당 조건의 순손익이 −24.38이었다. 이후 결과를 보고 선정 대표를 둘째 혼합으로 바꾸지 않았다.</p>
<h2>앞선 선정 구간의 실패 조건</h2><p>8월 첫째 전략＋페어의 base 순손익은 +13,591.66으로 원래 첫째 +13,047.40보다 높았지만 Q는 0.237737로 원래 0.240016보다 낮았다. stress 순손익은 +13,750.13, Q 0.241238로 높았으나 완료한 페어가 2개로 사전에 정한 최소 3개에 못 미쳤다. 모든 조건에서 두 기존 전략보다 높은 PnL/Q를 요구한 선정 자격은 false다. 재사용 자료이므로 이후 구간 또한 독립적인 미래 검증은 아니다.</p>
<h2>구현한 계좌 구조</h2><p>실제 계좌 원장은 하나이고 초기 현금은 1,000,000이다. 기존 전략의 소유 원장에는 초기 현금 1,000,000, 페어 원장에는 0을 배정한다. 두 원장의 수량·평가손익 합계가 실제 계좌와 같아야 하며, 현금을 두 번 더하지 않는다. 페어는 기존과 같은 최대 명목금액 120,000 및 전역 위험 한도를 사용한다. 각 종목만 먼저 체결되는 경우도 위험을 검사한다.</p>
<p>반대 방향의 전략별 주문은 두 전략이 미리 정한 가격 한도를 모두 허용하는 관측 가격에서 내부 소유권만 상쇄한다. 실제 거래소에는 남은 순수량만 보낸다. 실제 체결은 축소 요청, 페어 헤지, 기존 전략 순서로 배정하고 수수료는 체결된 물량에 한 번만 부과한다. 펀딩도 순포지션에 적용한 동일 유효 요율을 두 소유 원장에 배분해 합계를 보존한다. 페어의 가격 한도는 실제 순주문이 축소 주문인 경우에도 지킨다.</p>
<h2>발견하고 수정한 순포지션 전환 오류</h2><p>초기 합산 실행기는 실제 순포지션을 0까지만 줄이고 반대 방향 진입을 뒤로 미뤘다. 기존 전략이 롱을 보유한 상태에서 페어가 숏을 여는 경우에도 이 제한이 적용돼, 페어의 다른 종목만 더 많이 체결되고 불필요한 축소가 발생했다.</p>
<p>v2는 각 전략 자체의 청산 후 반대 방향 진입 순서는 보존하면서, 서로 다른 전략을 합친 비축소 주문이 실제 순포지션을 넘어 체결되는 경우를 처리한다. 원장에는 기존 포지션 종료와 반대편 진입을 나눠 기록하지만 같은 체결 가격을 적용하고, 전체 주문 수량에 수수료를 부과하며 거래량/호가 수량 예산은 한 번만 사용한다. 기존 두 전략 및 페어 단독의 실제 체결·계좌 이력은 모든 비교에서 이전 엔진과 완전히 같았다.</p>
<p>첫째 혼합의 이후 base 결과는 미수정 실행기 −56.01에서 수정 후 +2,231.27로, stress는 −1,686.48에서 +15.06으로 바뀌었다. 이는 전략 임계값을 조정한 개선이 아니라 실제 순주문의 체결 의미를 바로잡은 결과다. 미수정 결과는 삭제하지 않고 <code>artifacts/superseded.json</code>으로 비교용 사용을 중단한 이유를 기록했다.</p>
<p>거래소 근거: <a href="https://hyperliquid.gitbook.io/hyperliquid-docs/trading/order-types">공식 주문 유형 안내</a>는 reduce-only와 반대 포지션 개설을 구분한다. <a href="https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/error-responses">공식 오류 목록</a>도 포지션 전환과 reduce-only 제한을 별도로 다룬다. 실제 서버에 주문을 보내 이 수정의 체결을 확인한 것은 아니다.</p>
<h2>호가 기반 진입부터 청산까지</h2><p>호가 계획 생성기를 같은 계좌 실행자에 연결한 <code>BookPairSession</code>을 구현했다. 최신 수신 호가로 가격과 수량을 정하고, 미래 도착 호가는 주문의 사전 한도 안에서 각 종목별로 처리한다. 한쪽만 알려진 미체결이면 응답 확인 이후 한 번만 남은 현금 이익을 계산해 보완한다. 이미 채운 종목은 추가하지 않는다. 부분체결 비율이 어긋나면 축소하며, 청산 요청과 실제 청산 완료를 구분한다.</p>
<p>다음 호가가 관측되지 않으면 체결 0으로 가정하지 않고 상태를 미확정으로 남기며 새 주문을 막는다. 보완 기한을 넘기면 늦은 헤지를 하지 않고 기존 체결분의 청산을 요청한다. 펀딩 경계에서는 각 종목의 실제 체결 시각보다 먼저 보유했던 수량만 과금하도록 콜백을 연결했다. 이는 외부 주문 전송 기능이 없는 오프라인 실행기다.</p>
<h2>par와 검증 범위</h2><p>현재 전체 기간 par 재현은 여전히 캔들 체결 가정에 따른 조건부 결과다. 이번 두 혼합 후보의 추가 페어 체결은 0건으로 기존 전략 결과와 같았다. 기존 실제 호가 기록에는 해당 페어 신호 발생 시점이 겹치지 않아 전체 기간 호가 기반 PnL을 확인하지 못했다. 다른 날짜의 호가를 대체 사용하지 않았고, 예약 검증용 자료도 읽지 않았다. 따라서 위 기준시장 개선을 대회에서의 개선으로 간주하지 않는다.</p>
<p>v2 및 호가 세션 검사 25개, 펀딩 경계 검사 2개 통과. 두 실행기에서 각각 40개 계좌 재현을 보존했다. 기존 전략과 페어 단독 대조군의 체결·계좌 이력을 정확히 재현했으며, 계좌와 소유 원장 합계 오차는 1.63×10⁻⁹ 이하다. 보호된 실거래 파일 373개와 예약 의존 파일 345개가 변경되지 않았다.</p>'''
    for venue in ('reference','par'):
        body+='<h2>'+('기준시장 전체 결과' if venue=='reference' else 'par 캔들 모형의 조건부 전체 결과')+'</h2>'
        body+=table([r for r in current['rows'] if r['venue']==venue])
    body+='''<h2>남은 판정</h2><p>선정 구간의 Q 저하, 더 큰 노출의 기여, 불리한 청산비용, 실제 par 체결 관측 공백이 남아 있다. 다음에는 앞선 구간으로만 크기를 정한 노출/위험 대조군을 만들고, 조정한 규칙과 이후 평가를 분리한다. 지금은 교체 가능한 검증 완료 전략이 아니다.</p>'''
    (out/'report.html').write_text(document('공유 계좌 페어 전략 비교',body));export_csv(out/'comparison.csv',current['rows'])
    book_names=['book_session.py','book_plan.py','account_v2.py','book_lifecycle_settings.json','test_v2.py','test_book_funding.py']
    summary=dict(selection=current['selection'],reference_comparison=overview,latest_account_replays=40,earlier_superseded_replays=40,
        current_tests_passed=27,book_sources={str((HERE/n).relative_to(ROOT)):sha(HERE/n) for n in book_names},
        maximum_ownership_error=max(r['metrics']['maximum_ownership_error'] for r in current['rows']),
        maximum_accounting_error=max(r['metrics']['maximum_accounting_error'] for r in current['rows']),
        orders_sent=0,automatic_live_replacement=False,historical_independence=False,protected=guards)
    save(out/'summary.json',summary)
    save(HERE/'work_state.json',dict(status='progress_shared_executor_and_L2_lifecycle_implemented_reference_improvement_conditional',
        previous_goal_turn='progress',report=str((out/'report.html').relative_to(ROOT)),latest_replays=40,superseded_replays=40,tests_passed=27,
        orders_sent=0,live_strategy_changed=False,
        findings=['Independent strategy ownership and physical ledger conserve positions, fees, funding and equity.',
            'v1 stopped physical reversals at zero and corrupted pair hedge balance; v2 corrects this without changing standalone controls.',
            'Reference later window first_plus_pairs improves PnL and Q over both native controls in base/stress; base 2231.2734, stress 15.0561.',
            'Selection still fails: August base Q lower than first incumbent and stress has only 2 completed pairs.',
            'Stress terminal exit reserve reduces later first_plus_pairs PnL to -120.6098; mean gross exposure is higher.',
            'L2 session integrates entry, one known-outcome completion, partial repair, persistent exit, unknown-outcome blocking and funding timing.'],
        next='Evaluate scale-matched baseline controls with calibration restricted to August, quantify incremental pair contribution and fragile exit-cost margin. Preserve L2-backed evidence separately from explicit hypothetical-liquidity sensitivity, then improve selection/allocation based on pre-period information.',
        remaining=['No eligible selection under all declared conditions','No full-period observed par book profitability evidence','No independent prospective superiority evidence']))
    print('Report, latest summary, superseded diagnosis and continuation state written.')


if __name__=='__main__':main()
