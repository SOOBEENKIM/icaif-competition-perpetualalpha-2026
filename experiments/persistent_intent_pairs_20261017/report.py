import gzip,html,json,time
from pathlib import Path
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.netted_shadow_pairs_20261017.report import document,table,export_csv

HERE=Path(__file__).resolve().parent


def main():
    results=json.loads((HERE/'artifacts/results.json').read_text())
    audit=json.loads((HERE/'liquidity_artifacts/results.json').read_text())
    for folder in ('artifacts','liquidity_artifacts'):
        frozen=json.loads((HERE/folder/'frozen.json').read_text())
        for rel,digest in frozen['sources'].items():assert sha(ROOT/rel)==digest,rel
    baseline_equivalence=[]
    for phase in ('validation','comparison'):
        for sc in ('base','stress'):
            a=HERE/'artifacts/runs'/f'reference-{phase}-fixed_delayed_immediate_complete_168-{sc}.json.gz'
            b=ROOT/'experiments/hedge_completion_pairs_20261017/reference_artifacts/runs'/f'{phase}-fixed_delayed_complete_168-{sc}.json.gz'
            with gzip.open(a,'rt') as f:x=json.load(f)
            with gzip.open(b,'rt') as f:y=json.load(f)
            assert x['fills']==y['fills'] and x['snapshots']==y['snapshots']
            baseline_equivalence.append(dict(phase=phase,scenario=sc,fills_and_snapshots_identical=True))
    out=ROOT/'reports/persistent_intent_pairs_20261017';out.mkdir(exist_ok=True)
    lines=[]
    for r in audit['summary']:
        c=r['counts'];values=[r['symbol'],r['scenario'],r['notional'],c['valid_decision_books'],
            c['candle_source_below_minimum'],c['decision_and_arrival_depth_full'],c['candle_rejected_but_both_books_full'],c.get('arrival_unobserved',0)]
        lines.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    heading=['종목','조건','요청 명목금액','유효 호가 관측','캔들 기준 최소금액 미달','현재/지연 후 호가 모두 수량 충족','캔들 거절 중 호가 수량 충족','지연 후 미관측']
    liquidity_table='<div class="scroll"><table><tr>'+''.join('<th>'+v+'</th>' for v in heading)+'</tr>'+''.join(lines)+'</table></div>'
    body='''<h1>페어 진입 의도 전달과 체결 가정 점검</h1>
<p class="note"><strong>이번에 확인한 핵심 문제는 과거 체결량을 대기 호가 유동성처럼 사용한 진입 제한이다.</strong> 신호를 6시간 유지해도 par 캔들 모형에서는 거래가 0건이었다. 그러나 보유한 실제 호가 자료에서는 해당 거래량 제한이 최소 주문액 미달로 거절하는 상황에도 주문 수량이 표시된 경우가 많았다. 따라서 무거래 결과만으로 페어 수익성을 부정하거나, 반대로 무손실로 검증됐다고 판단하지 않는다. 실거래 변경과 주문은 0건이다.</p>
<h2>구현과 계좌 비교</h2><p>모의 계좌의 체결 후 보유량을 한 관측 늦게 복사하던 방식을 중단하고, 완성된 기준시장 자료의 진입 신호·베타·평균·표준편차·예상 청산 가격을 직접 보존했다. 주문 전 현재 par 가격과 알려진 펀딩으로 남은 기대 손익을 다시 계산한다. 신호 유효기간 1시간과 6시간을 사전에 정해 비교했으며, 평균 도달·중단 조건·만료 시 취소한다. 전량 미체결은 신호별 최대 3회만 재시도한다. 한쪽 체결 보완과 유료 축소/청산은 이전 실행 구조를 유지한다.</p>
<p>기준시장 1시간 후보는 이전 페어 엔진과 8월·9~10월, base/stress 네 비교에서 체결 및 계좌 이력이 완전히 같았다. 보유량 복사 방식의 추가 지연과 다른 주문 제한 때문에 사라졌던 성과를 복원했으며, 이는 새 수익성을 발견했다는 뜻은 아니다. 6시간 후보의 이후 기준시장 base 이익은 +1,772.72로 1시간 후보 +1,772.51과 거의 같았다. stress는 둘 다 +432.87이었다.</p>
<p>선정 구간인 8월에는 두 후보 모두 base +553.08, stress +1,206.49로 기존 첫째·둘째 전략보다 낮았다. 따라서 선정 대표는 1시간 후보지만 <strong>채택 자격은 false</strong>이다. 이후 기간에서 기준시장 PnL/Q가 높아도 이 판정을 뒤집지 않았다.</p>
<p>기준시장 validation: 2026-08-01 12:00–09-01 12:00 UTC; comparison: 09-01 12:00–10-15 12:00 UTC. par validation: 10-01 12:00–10-08 12:00 UTC; comparison: 10-08 12:00–10-16 12:00 UTC. 모든 계좌 초기자본 1,000,000. base 편도 수수료 9bp·가격 비용 10bp, stress 12bp·20bp와 더 작은 거래량 참여율 및 양의 펀딩 수령액 할인. 잔여 포지션을 무료로 청산하지 않는다. 모든 역사 자료는 개발에 재사용했으므로 독립적인 미래 검증이 아니다.</p>'''
    for venue in ('reference','par'):
        body+='<h2>'+('기준시장 — 대회 계좌 성과가 아님' if venue=='reference' else 'par — 캔들 체결 가정의 조건부 결과')+'</h2>'
        body+=table([r for r in results['rows'] if r['venue']==venue])
    body+='''<h2>par 진입이 사라진 지점</h2><p>6시간 후보의 서로 다른 신호는 10개, 15분 단위 관측은 180개였다. base에서는 177개가 양쪽 주문의 최소금액 10 미달, 나머지 3개가 반올림 후 헤지 비율 오차로 막혔다. stress는 180개 모두 최소금액 미달이었다. 이 숫자는 체결 가능한 주문의 수가 아니라 반복된 신호 점검의 수다. 실제 포지션이 없는 조건에서 계산한 진입 제한 점검이다.</p>
<h2>같은 시점의 실제 호가와 거래량 제한 비교</h2><p>기존에 공개돼 있던 다섯 호가 기록만 사용했다. 예약 검증용 원본은 열지 않았다. 1분마다 BTC·ETH·HYPE의 매수/매도 각각을 점검하고, 10 및 100 명목금액 요청을 미리 정했다. 최신 호가 15초·문맥 3초 제한, 주문 지연 base 1초/stress 3초, 지연 이후 새로 받은 호가, 10bp 가격 한도, stress 호가 수량 50%만 반영했다. 도착 호가가 없으면 미관측으로 남겼다.</p>
<p>예를 들어 BTC 100의 base 관측 538개 중 거래량 기준 거절은 300개였으며, 그중 293개는 당시와 지연 후 호가 모두에 요청 수량이 표시됐다. 매수/매도와 연속 관측이 포함되므로 독립적인 538회 실험도, 실제 체결 293건도 아니다. 같은 수량을 여러 번 체결했다고 합산하지 않았다. 취소·경쟁 주문으로 실제 체결은 달라질 수 있다.</p>'''+liquidity_table+'''
<p><strong>중요한 관측 공백:</strong> 이 호가 구간의 페어 관측 538개에서는 이번 6시간 신호가 한 번도 활성화되지 않았다. 따라서 다른 날짜의 호가를 신호 발생일에 옮겨 붙여 수익률을 계산하지 않았다. 이 점검은 체결 모형의 제한을 보여 주며, 이번 전략의 실제 호가 기반 수익성을 입증하지 않는다.</p>
<h2>호가 기반 진입 계획 모듈</h2><p><code>book_plan.py</code>를 추가했다. 양쪽의 현재 호가 수량으로 주문 크기를 정하고, 실제 매수/매도 호가를 따라 계산한 예상 진입 가격·현재 반대편 호가의 청산 비용·추가 불리한 가격 변화·수수료·펀딩을 차감한다. 고정된 예상 청산 가격에서 기대 현금 이익이 남아야만 독립적인 두 IOC 한도를 생성한다. 각 종목만 먼저 체결된 경우도 위험 한도를 검사한다. 캔들 체결량은 입력으로 사용하지 않는다. 두 주문을 동시에 완전 체결했다고 가정하지 않는다.</p>
<p>이 모듈은 읽기 전용 계획 생성기다. 지갑 서명·주문 전송 기능은 없으며, 실제 주문 수량 대조와 한쪽 체결 보완을 담당하는 실행기 연결은 남아 있다. 기존 전략과 동일 종목을 공유할 때의 소유 수량 합산도 아직 이 모듈에 통합하지 않았다. 미래 청산 호가가 현재와 같다고 보장하지 않는다.</p>
<h2>검증 범위와 다음 작업</h2><p>신호·계좌 테스트 10개, 호가 계획 테스트 7개 통과. 신규 계좌 재현 32개와 이전 페어 대조군 8개를 저장했다. 기존 두 전략 체결·계좌 이력은 이전 실험과 같았다. 원장 최대 오차는 1.87×10⁻⁹ 미만이다. 보호된 실거래 소스 373개와 예약 검증 의존 파일 345개가 그대로였다.</p>
<p>다음에는 이 호가 계획을 한 실행자와 실제 체결 대조 구조에 연결하고, 기존 전략 몫과 페어 몫을 합산하는 계좌 구조를 검증한다. 과거 호가가 없는 구간의 결과는 명시적인 체결 가정에 따른 민감도 결과로만 취급하며, 관측된 호가 구간의 체결 결과와 구분한다. 이번 실험만으로 기존 실거래를 교체할 근거는 확보되지 않았다.</p>
<p>자료 범위 참고: <a href="https://hyperliquid.gitbook.io/hyperliquid-docs/historical-data">Hyperliquid 공식 과거 자료 안내</a>는 L2/asset context 아카이브와 갱신 지연·누락 가능성을 설명한다. 해당 안내만으로 이번 testnet par 기간의 전체 호가가 확보됐다고 간주하지 않았으며 유료 아카이브 다운로드도 수행하지 않았다.</p>'''
    (out/'report.html').write_text(document('페어 진입 의도와 체결 가정',body));export_csv(out/'comparison.csv',results['rows'])
    guards=protected()
    book_sources={str((HERE/name).relative_to(ROOT)):sha(HERE/name) for name in ('book_plan.py','book_protocol.json','test_book_plan.py')}
    summary=dict(selection=results['selection'],actual_replays=32,previous_pair_controls=8,tests_passed=17,
        exact_reference_pair_reproduction=baseline_equivalence,funnels=audit['funnels'],liquidity_summary=audit['summary'],
        intent_book_coverage=audit['intent_coverage'],book_planner_sources=book_sources,orders_sent=0,
        automatic_live_replacement=False,historical_independence=False,protected=guards,
        maximum_accounting_error=max(r['metrics']['maximum_accounting_error'] for r in results['rows']))
    save(out/'summary.json',summary)
    save(HERE/'work_state.json',dict(status='diagnostic_complete_execution_model_limitation_verified',previous_goal_turn='progress',
        report=str((out/'report.html').relative_to(ROOT)),actual_replays=32,tests_passed=17,orders_sent=0,
        completed=['Causal intent transfer with frozen terminal price, 1h/6h expiry and zero-fill retries',
            'Native/reference-pair ledger reproduction', 'Quantified candle-flow gating against actual as-of/arrival L2',
            'Read-only L2 entry planner with independent leg limits and current exit-depth costs'],
        unfinished=['No candidate passes August selection versus both native baselines',
            'No active candidate signal occurs inside the old L2 tapes; full historical par book PnL remains unverified',
            'L2 planner not yet integrated with one execution owner and pair lifecycle',
            'Portfolio netting must preserve baseline intent and pair economic limits without shadow inventory copying',
            'No independent future superiority evidence; live strategy remains unchanged'],
        next='Integrate causal baseline and pair intentions into one execution ledger, preserving pair economics and L2 quantity limits. Keep factual book replay separate from stated-liquidity historical sensitivity.'))
    print('Report and work state written; preserved sources verified.')


if __name__=='__main__':main()
