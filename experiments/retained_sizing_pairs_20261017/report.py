"""Report utility sizing and actual partial-order failure paths."""
import gzip
import json
from pathlib import Path
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.netted_shadow_pairs_20261017.report import document,table,export_csv
from experiments.dense_history_pairs_20261017.report import grid

HERE=Path(__file__).resolve().parent;ART=HERE/'artifacts'


def main():
    r=json.loads((ART/'results.json').read_text())
    for rel,digest in json.loads((ART/'frozen.json').read_text())['sources'].items():assert sha(ROOT/rel)==digest,rel
    guards=protected();out=ROOT/'reports/retained_sizing_pairs_20261017';out.mkdir(exist_ok=True)
    selected=[];paths=[];failure_rows=[]
    for row in r['rows']:
        if '_utility' not in row['arm']:continue
        if row['venue']=='reference':
            m=row['metrics'];selected.append([row['phase'],row['arm'],row['scenario'],f"{m['net_pnl']:+,.2f}",
                f"{m['financial_score']:.8f}",f"{row['delta_pnl']:+,.2f}",row['utility_resize_plans'],row['utility_close_requests'],m['completed_pairs']])
        name=f"{row['venue']}-{row['phase']}-{row['arm']}-{row['scenario']}"
        with gzip.open(ART/'runs'/f'{name}.json.gz','rt') as f:run=json.load(f)
        for plan in [e for e in run['events'] if e['event']=='hedge_adjustment_planned' and e.get('utility_enabled')]:
            event=next(e for e in run['events'] if e.get('new_source')==plan['new_source'] and e['time']>plan['time'] and e.get('outcome') in ['renewed','unfilled','unwind'])
            fills=[f for f in run['owner_fills']['pair'] if plan['time']<f['time']<=event['time']]
            got=[sum(f['signed_qty'] for f in fills if f['symbol']==symbol) for symbol in plan['pair']]
            assert all(abs(a-b)<1e-8 for a,b in zip(got,event['actual_delta']))
            path=dict(venue=row['venue'],phase=row['phase'],arm=row['arm'],scenario=row['scenario'],
                decision_time=plan['time'],planned_delta=plan['delta'],actual_delta=got,
                current_gross=plan['current_gross'],target_gross=plan['target_gross'],
                expected_utility_gain=plan['utility_advantage'],utility_buffer=plan['utility_buffer_cash'],
                outcome=event['outcome'],reason=event['reason'],paid_first_packet_fees=sum(f['fee'] for f in fills))
            paths.append(path)
            failure_rows.append([row['phase'],row['arm'],row['scenario'],f"{plan['current_gross']:,.2f}",
                f"{plan['target_gross']:,.2f}",str(plan['delta']),str(got),event['reason']])
    assert len(paths)==4 and all(x['reason']=='partial_adjustment_hedge_mismatch' for x in paths)
    body='''<h1>남은 예측 수익과 위험에 따른 페어 보유 수량 조정</h1>
<p class="note"><strong>구현과 검증은 완료했지만, 기존 매시간 페어 후보보다 일관된 성과 개선은 나오지 않았다.</strong>
새 계좌 재현 32개, 기존 대조군 정확 재현 4개, 재사용 대조 계좌 32개를 비교했다.
부분 체결·미확인 주문·누적 손실 기준·미래 자료 누출 관련 검사 12개를 통과했다.</p>
<h2>수량 판단을 바꾼 방법</h2><p>기존 모델·진입 규칙과 비용 차감 보유 연장을 유지했다.
기존 연장 조건을 통과한 시점에 최초 배정액의 25·50·75·100%, 원래 연장 수량 유지, 전량 청산을 비교했다.
감축 후 예측이 강해지면 최초 배정액까지만 복원할 수 있다. 새 진입의 위험 한도나 최대 페어 배정액을 늘리지 않았다.</p>
<p>판단값은 예상 가격·펀딩 손익에서 현재 수량 변경 비용과 미래 청산 비용을 뺀 뒤,
현재 원전략 보유와의 공분산을 포함한 위험 부담을 차감한다.
위험 항은 γ×남은 시간×[변경 후 포트폴리오 분산−원전략만의 분산]/(2×계좌자산)이다.
자료의 공분산이 시간당 값이므로 남은 시간 수를 곱한다. γ=16·64를 실행 전에 고정했다.</p>
<p>새 수량은 원래 연장 수량보다 위험 조정 판단값이 현재 명목금액의 5bp 이상 좋아야 한다.
잔여 보유의 비용 차감 가치도 즉시 청산보다 좋아야 한다. 두 조건의 여유 중 작은 금액으로 각 주문의 지정가 한도를 정했다.
주문 도착 후 실제 수량·가격·위험을 다시 확인하고, 손실 기준의 원점을 초기화하지 않았다.</p>
<h2>기준시장 계좌 결과</h2><p>자본 1,000,000. validation: 2026-08-01 12:00~09-01 12:00 UTC.
comparison: 09-01 12:00~10-15 12:00 UTC. 기본 조건 편도 수수료·가격 불이익 9bp·10bp,
stress는 12bp·20bp, 더 작은 거래량 참여율, 펀딩 수령액 절반이다.
모두 재사용 과거 자료의 개발 결과다. 차이는 동일 원전략 기반의 기존 매시간 페어 후보 대비 값이다.</p>'''
    body+=grid(['기간','후보','조건','순손익','Q','기존 페어 후보 대비 손익','새 수량 조정 계획','위험 조정 청산 요청','완료 페어'],selected)
    body+='''<p>γ=16은 모든 조건에서 기존 페어 후보와 같은 체결 경로였다.
γ=64의 첫째 전략 기반 후보는 8월 기본 순손익이 +20,090.87에서 +16,793.19로 감소했다.
이후 기본에서는 +1,491.90에서 +1,526.52로 34.62 증가했지만, stress는 그대로였다.
이 작은 이후 개선만으로 강한 수량 조정이 낫다고 선택하지 않는다.
8월 단계의 고정 선택은 first_dense_utility16_bias_24, eligible=false다.
이는 사전 최소 완료 페어 기준을 포함한 연구 후보 선정 결과이며 대회 참가 자격과 별개다.</p>
<h2>새 감축 주문의 실제 재현 경로</h2><p>원전략 기반과 기간을 달리한 새 감축 계획 4개 모두 한쪽만 체결돼 원래의 전체 청산 규칙으로 넘어갔다.
아래 실제 수량은 이 백테스트 실행 모형에서 확인된 체결 수량이며 실제 거래소 주문 전송을 뜻하지 않는다.</p>'''
    body+=grid(['기간','후보','조건','감축 전 명목금액','목표 명목금액','목표 변경 수량','확인된 변경 수량','처리 이유'],failure_rows)
    body+='''<p>8월 첫째 전략 사례는 BTC/HYPE 명목금액 약 118,206을 59,994로 줄이려 했지만,
BTC 매도 약 0.6021개만 체결되고 HYPE 매수 146.41개는 체결되지 않았다.
현재 구현은 이 불균형에서 남은 감축을 재평가하지 않고 페어 전부를 비용을 지불해 청산한다.
따라서 수량 최적화의 기대 이득이 실제 균형 잡힌 감축으로 이어지지 않았다.</p>
<p>이를 해결하려고 미체결을 체결 처리하거나 최초 지정가 한도를 사후에 무제한 넓히지 않는다.
다음 후보는 확인된 부분 체결 이후 남은 감축 수량만 새 가격·남은 예측 가치·비용으로 평가하고,
이득이 남는 경우 한 번만 추가 주문하는 절차다. 이미 지불한 비용과 현재 불균형은 그대로 원장에 남긴다.
추가 주문이 실패하거나 결과가 미확인이면 각각 지속 청산 또는 확인 대기 상태로 처리해야 한다.
아직 이 추가 절차는 구현·검증하지 않았다.</p>
<h2>논문과의 관계</h2><p>유한 기간·거래 비용을 함께 다루는
<a href="https://link.springer.com/article/10.1007/s10614-025-11163-0">2026년 권호의 페어 최적 정지 연구</a>와
<a href="https://arxiv.org/abs/1911.10450">비례 거래 비용을 포함한 페어 수량 제어 연구</a>를 설계 배경으로 참고했다.
이번 구현은 고정 예측과 실제 계좌 제약에 맞춘 유한 후보 평균·분산 근사다.
논문의 자유경계나 HJB 해를 재현한 구현이라고 주장하지 않는다. 최종 평가는 논문 성과가 아닌 위 비용 차감 계좌 결과다.</p>
<h2>par 검증 범위</h2><p>par 캔들·펀딩 재현에서는 새 수량 조정이 생기지 않아 기존 결과와 같았다.
주문장 처리 검사는 합성 부분 체결·미확인 결과를 사용해 상태 처리가 맞는지 확인했다.
기존 짧은 실제 호가 기록에는 24시간 뒤 보유 연장 구간이 없어 새 로직의 실제 par 전체 생애 수익을 입증하지 못한다.</p>'''
    body+=table(r['rows'])
    body+=f"<p>주문 전송 0개, 실거래 변경 없음. 활성 소스 {guards['active_protected_sources']}개·예약 의존 파일 {guards['reserved_dependencies']}개는 그대로다.</p>"
    (out/'report.html').write_text(document('페어 보유 수량 조정 결과',body));export_csv(out/'comparison.csv',r['rows'])
    save(out/'summary.json',dict(selection=r['selection'],new_account_replays=32,exact_control_replays=4,
        tests_passed=12,utility_order_paths=paths,protected=guards,orders_sent=0,
        historical_independence=False,automatic_live_replacement=False))
    save(HERE/'work_state.json',dict(
        status='progress_costed_retained_sizing_implemented_32_accounts_12_tests_partial_reduction_unwind_bottleneck',
        previous_goal_turn='progress_renewal_ablation_8_accounts_2_exact_controls_retain_renewal',
        report=str((out/'report.html').relative_to(ROOT)),new_account_replays=32,exact_control_replays=4,tests_passed=12,
        orders_sent=0,live_strategy_changed=False,selection=r['selection'],historical_independence=False,
        findings=[
            'Gamma16 adds no orders and exactly preserves the prior hourly candidate.',
            'Gamma64 first-base August loses 3297.68 versus prior; later base gains only 34.62 and stress unchanged.',
            'All four new utility resize plans across base variants and periods become one-leg fills followed by full paid unwind; none is a completed retained-size adjustment.',
            'Risk penalty uses hourly covariance times remaining hours and includes the existing baseline cross term; costs and price-limit utility buffer are reconciled.',
            'Synthetic received-book partial/unknown outcomes, restoration cap, causal data use, and original cumulative stop checks passed; full actual par lifecycle remains absent.',
        ],
        next='Implement a separately frozen v2 with at most one paid completion of a confirmed partial risk-reducing resize. Only remaining reductions toward the original target may be sent; keep original forecast terminal/maturity and stop. Reprice at observed arrival, deduct remaining costs and actual risk, seal new limits. If no remaining utility or risk capacity, retain sticky unwind. Zero fill of completion must unwind existing imbalance, not retain old maturity as if harmless; unknown outcome must block retries. Test independently and compare gamma64 v2 to exact v1 and hourly controls. Do not alter frozen v1 or live files.',
        remaining=['No incremental robust improvement beyond hourly bias24','Partial reduction completion unimplemented',
                   'Full actual par lifecycle and independent prospective evidence unverified']))
    print(out/'report.html');print('Frozen and protected sources verified unchanged.')


if __name__=='__main__':main()
