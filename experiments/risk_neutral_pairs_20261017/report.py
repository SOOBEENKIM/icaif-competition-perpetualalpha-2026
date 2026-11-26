"""Publish scale controls, unsuccessful allocation candidate and attribution."""
import gzip,html,json
from datetime import datetime,timezone
from pathlib import Path
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.netted_shadow_pairs_20261017.report import document,table,export_csv

HERE=Path(__file__).resolve().parent


def grid(headers,rows):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</table></div>'


def main():
    result=json.loads((HERE/'artifacts/results.json').read_text());scale=json.loads((HERE/'scale_artifacts/results.json').read_text())
    attribution=json.loads((HERE/'artifacts/attribution.json').read_text())
    for directory in ('artifacts','scale_artifacts'):
        frozen=json.loads((HERE/directory/'frozen.json').read_text())
        for rel,digest in frozen['sources'].items():assert sha(ROOT/rel)==digest,rel
    guards=protected();out=ROOT/'reports/risk_neutral_pairs_20261017';out.mkdir(exist_ok=True)
    by={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in result['rows']}
    overview=[]
    for phase in ('validation','comparison'):
        for arm in ('first_strategy','second_strategy','first_plus_pairs','first_risk_neutral'):
            for scenario in ('base','stress'):
                m=by['reference',phase,arm,scenario]
                overview.append([phase,arm,scenario,f"{m['net_pnl']:+,.2f}",f"{m['financial_score']:.8f}",f"{100*m['volatility_horizon']:.4f}%",m['completed_pairs'],f"{m['pnl_after_liquidation_reserve']:+,.2f}"])
    calibration=[]
    for f in scale['fit']:
        calibration.append([f['baseline'],f['metric'],f"{f['scale']:.6f}",f"{f['relative_error']*100:+.4f}%"])
    episode_rows=[]
    for r in attribution['rows']:
        if r['phase']=='comparison' and r['arm'] in ('first_plus_pairs','first_risk_neutral'):
            for e in r['episodes']:
                episode_rows.append([r['arm'],r['scenario'],datetime.fromtimestamp(e['time'],timezone.utc).strftime('%m-%d %H:%M'),'/'.join(e['pair']),f"{e['net_pnl']:+,.2f}",f"{e['fees']:.2f}",f"{e['funding']:+.2f}",e.get('status','closed' if e['closed'] else 'open')])
    body='''<h1>페어 전략: 규모 대조군과 예상분산 상한 후보</h1>
<p class="note"><strong>수정 구현과 비교는 완료했지만 이 후보를 기존 실거래 전략으로 교체하지 않는다.</strong> 8월에서는 첫째 전략＋예상분산 상한 페어의 PnL·Q가 원래 두 전략보다 높았으나, 완료 페어 수가 사전 조건에 미달했다. 이후 불리한 비용 조건에서는 두 기존 전략보다 손익과 Q가 낮았다. 기준시장 과거 재현이며 par의 실제 거래 수익이 아니다.</p>
<h2>이번에 확인한 것</h2><p>이전 혼합판의 더 높은 투자 규모가 성과를 설명하는지 분리하려고, 8월 기본 조건의 평균 총노출 또는 평균 예측 일별 변동성이 같아지도록 기존 전략의 규모를 각각 맞췄다. 손익으로 배수를 고르지 않았으며, 8월에서 정한 배수를 고정한 뒤 이후 구간과 불리한 비용 조건을 평가했다. 60회 계좌 재현으로 네 대조군을 만들었다. 이후 구간까지 노출이 정확히 같도록 맞춘 것은 아니다.</p>'''+grid(['기존 전략','8월에 맞춘 항목','고정 배수','8월 항목 일치 오차'],calibration)+'''
<p>8월 첫째 혼합판은 기본 조건 +13,591.66 / Q 0.237737로, 총노출을 맞춘 첫째 전략 +14,660.20 / Q 0.251079보다 낮았다. 예상 위험을 맞춘 첫째 전략의 Q 0.243353보다도 낮았다. 이후 구간에서는 혼합판이 앞섰지만 평균 총노출 차이가 다시 발생했다. 이 비교로 모든 시장 구간에서 페어 신호가 우월하다고 결론내릴 수 없다.</p>
<h2>구현한 후보</h2><p>기존 신호와 한 계좌의 체결 실행기는 보존하고, 새 페어의 예정 수량만 제한했다. 현재 보유한 기존 전략과 미체결 목표 각각에 대해, 페어 추가 및 왕복 비용 추정 차감 후의 예상 수익률 분산이 기존 전략 단독보다 커지지 않도록 했다. 현재 시점까지 알려진 공분산만 사용한다. 연속 수량의 상한을 구한 뒤 종목별 실제 주문 단위로 반올림하고 다시 검사한다.</p>
<p>기존 명목금액 벡터 b, 총명목금액 1당 페어 벡터 u, 공분산 S, 계좌값 E, 왕복 비용률 k, 추가 규모 g에 대해 (b+gu)′S(b+gu)/(E−kg)² ≤ b′Sb/E²를 요구한다. 두 기존 포지션 상태 중 더 작은 상한에 0.999 여유를 적용한다. 현금 상태이거나 분산을 낮추지 않는 페어에는 추가 위험 예산을 임의로 만들지 않는다.</p>
<p>이 규칙은 진입 시의 계획에 관한 제약이다. 부분체결, 이후 가격·공분산 변화와 기존 전략의 재조정 때문에 실제 보유기간의 변동성이 항상 낮아진다는 뜻은 아니다. 한쪽 미체결 보완·축소, 비용, 전체 계좌 위험 한도와 종료 시 잔여 포지션 처리는 기존 실행기와 같다.</p>
<h2>기준시장 비교</h2><p>초기자본 1,000,000. validation: 8/01 12:00–9/01 12:00 UTC; comparison: 9/01 12:00–10/15 12:00 UTC. base는 편도 수수료 9bp·불리한 가격 이동 10bp, stress는 12bp·20bp 및 더 작은 체결 참여율과 펀딩 수령액 50% 가정이다. 모든 행은 같은 기간·자본·위험 한도를 공유한다. σ는 매일 12:00 UTC 수익률의 표본 표준편차×√n이다.</p>'''+grid(['기간','전략','비용','순손익','Q','기간 σ','완료 페어','종료 청산비용 추정 차감 후'],overview)+'''
<p>8월에 고정한 진단 대표는 <code>first_risk_neutral</code>이고 채택 자격은 false다. 8월 기본 조건 +13,978.55 / Q 0.243275, 불리한 조건 +13,504.58 / Q 0.240069로 원래 두 전략보다 높았다. 그러나 불리한 조건 완료 페어가 2개로 사전 최소 3개에 미달했다. 이후 기본 조건 +778.64 / Q 0.097347로 개선되었지만, 불리한 조건 −720.24 / Q 0.083906으로 첫째 −457.10 및 둘째 −492.97보다 나빴다. 결과를 보고 최소 거래 수 기준이나 기간을 바꾸지 않았다.</p>
<h2>어떤 진입이 달라졌는가</h2><p>아래는 평가가 끝난 뒤 실제 소유 원장으로 분해한 진단이다. 매 시점의 페어 보유량 가격 변화, 체결 비용, 수수료와 실제 적용 펀딩을 합했으며, 총합은 페어 소유 원장의 손익과 일치한다. 마감 시 열려 있는 페어는 시가평가이고 별도 청산비용은 위 계좌 표에 제시했다. 이 진단으로 평가 기간의 후보를 다시 선택하지 않았다.</p>'''+grid(['전략','비용','진입 의도 UTC','페어','페어 순손익','수수료','펀딩','마감 상태'],episode_rows)+'''
<p>이후 기본 조건에서 원래 혼합판의 페어 몫은 +1,772.51이었다. 분산 상한 후보는 +319.20으로 줄었다. 원래 혼합판에서 이익을 낸 첫 BTC/ETH 진입을 제외했고, 다른 진입의 수량도 크게 축소했다. 불리한 비용 조건에서는 상한 후보가 나중의 BTC/ETH 거래에서 −293.32를 기록해 페어 몫이 −263.49가 되었다. 위험을 낮춰 보이는 방향이라는 조건과 되돌림의 수익성은 서로 다른 문제다. 여러 신호가 같은 보유 기간에 겹치므로 이 차이를 개별 신호 하나의 독립적인 인과 효과로 보지는 않는다.</p>
<h2>par에 대해 확인한 범위</h2><p>par validation은 10/01 12:00–10/08 12:00 UTC, comparison은 10/08 12:00–10/16 12:00 UTC다. 전체 기간 캔들과 펀딩으로 계산하는 현재 체결 모형에서는 신규 페어 체결이 0건이며 혼합판 결과는 기존 전략과 같다. 과거의 관측 호가 점검에서는 캔들 거래량이 작아도 주문 가능한 호가가 있었으므로, 이 무체결을 실제 시장의 유동성 부재로 단정하지 않는다. 신호 시각과 겹치는 전체 과거 호가가 없어 par 수익 개선은 여전히 미확인이다.</p>
<h2>검증 및 해석</h2><p>이번 후보 56회 계좌 재현, 규모 대조군 60회로 총 116회다. 모든 기존 대조군의 체결과 계좌 이력이 이전 실행기와 정확히 같다. 새 위험 배분 검사 10개와 소유별 펀딩 귀속 검사 1개를 통과했다. 상쇄된 실제 순포지션이 0이어도 각 전략의 반대 펀딩을 누락하지 않는지를 검사했다. 보호된 실거래 소스 373개와 예약 검증 의존 파일 345개는 변경하지 않았다.</p>
<p>자료를 이미 여러 실험에서 사용했으므로 기간을 나눴더라도 독립적인 미사용 평가라고 부르지 않는다. 8월 이후 평가를 보기 전에 이번 대표를 저장한 순서는 지켰다. 선정 실패, 이후 비용 민감성, par 호가 공백, 독립 검증 부족을 모두 유지한다. 실제 주문 전송과 자동 교체는 0건이다.</p>
<p>연구 배경: <a href="https://arxiv.org/abs/2606.04574v2">Dynamic Multi-Pair Trading Strategy in Cryptocurrency Markets with Deep Reinforcement Learning</a>는 페어 선정과 실행 정책을 구분한다. 이번 실험은 자체 구현한 페어 신호와 배분 제약 비교이며 해당 논문의 PPO/LSTM 전체 재현이 아니다. <a href="https://perpetual-alpha26.github.io/leaderboard#practice">공식 리더보드</a>의 일별 샘플과 재무 점수 비교를 사용했다. 향후 규정 변경 시 점수를 재확인해야 한다.</p>
<h2>다음 수정의 근거</h2><p>공분산 기반 배분만으로 페어 수익성을 만들 수 없었다. 다음에는 동결한 평균으로 되돌아간다는 가격 목표 자체와 실제 보유시간·비용의 관계를 먼저 점검한다. 이미 실패한 모델을 다시 탐색하지 않도록 기존 실험 기록을 대조하고, 의사결정 당시 종료된 과거 사례만으로 추정 가능한 예상 손익을 검토한다. 과거 전체 호가가 없는 구간의 체결은 조건부 민감도로 분리한다.</p>'''
    for venue in ('reference','par'):
        body+='<h2>'+('기준시장 전체 28개 결과' if venue=='reference' else 'par 캔들 모형 전체 28개 결과')+'</h2>'+table([r for r in result['rows'] if r['venue']==venue])
    body+='<h2>고정 규모 대조군 전체 결과</h2>'+table(scale['rows'])
    (out/'report.html').write_text(document('페어 배분과 규모 효과 검증',body))
    export_csv(out/'comparison.csv',result['rows']);export_csv(out/'scale_comparison.csv',scale['rows'])
    summary=dict(selection=result['selection'],candidate_replays=56,scale_replays=60,tests_passed=11,
        maximum_ownership_error=max(r['metrics']['maximum_ownership_error'] for r in result['rows']),
        maximum_attribution_error=max(r['accounting_error'] for r in attribution['rows']),
        controls_exact=result['native_baselines_reproduced_exactly'],historical_independence=False,
        automatic_live_replacement=False,orders_sent=0,protected=guards,
        diagnostic_sources={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'attribution.py',HERE/'test_attribution.py',HERE/'report.py']})
    save(out/'summary.json',summary)
    save(HERE/'work_state.json',dict(status='progress_scale_audit_and_risk_budget_candidate_completed_not_adopted',report=str((out/'report.html').relative_to(ROOT)),
        previous_goal_turn='progress',candidate_replays=56,scale_replays=60,tests_passed=11,orders_sent=0,live_strategy_changed=False,
        findings=['August-only calibrated scale controls show hybrid is not consistently superior to larger baseline.',
            'New variance-budget candidate first_risk_neutral improves August PnL/Q but fails minimum completed-pair count.',
            'Later reference base +778.6417 improves both native baselines, stress -720.2412 underperforms both.',
            'Risk filter suppresses profitable and unprofitable entries differently; variance reduction is not predictive alpha.',
            '56 replays preserve all five controls exactly. Pair-episode attribution reconciles to owner equity.',
            'Par candle participation model still yields zero pair entries; no actual full-period book profitability claim.'],
        next='Review existing alpha-calibration experiments before another model. Diagnose frozen convergence targets against causally completed pre-entry examples and cost/holding-time outcomes. Preserve prior controls and separate actual observed L2 from hypothetical liquidity sensitivity.',
        remaining=['No candidate meeting all declared selection and later cost-stress conditions','No full-period observed par book profitability evidence','No independent prospective improvement evidence']))
    print(json.dumps(summary,ensure_ascii=False));print(out/'report.html')


if __name__=='__main__':main()
