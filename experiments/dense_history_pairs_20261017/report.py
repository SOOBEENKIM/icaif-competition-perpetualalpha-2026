"""Publish all fixed-maturity outcomes, including failed candidate families."""
import html,json
from pathlib import Path
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.netted_shadow_pairs_20261017.report import document,table,export_csv

HERE=Path(__file__).resolve().parent;ART=HERE/'artifacts'


def grid(headers,rows):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'


def main():
    r=json.loads((ART/'results.json').read_text());models=json.loads((ART/'training_manifest.json').read_text())
    attribution=json.loads((ART/'attribution.json').read_text());audit=json.loads((ART/'forecast_audits.json').read_text())
    book=json.loads((ART/'book_coverage.json').read_text());diag=json.loads((ART/'diagnostic_status.json').read_text())
    frozen=json.loads((ART/'frozen.json').read_text())
    for rel,digest in frozen['sources'].items():assert sha(ROOT/rel)==digest,rel
    guards=protected();out=ROOT/'reports/dense_history_pairs_20261017';out.mkdir(exist_ok=True)
    model_rows=[[ '/'.join(m['pair']),m['horizon'],m['training_rows'],m['calibration_rows'],
        f"{m['calibration_slope']:.6f}",f"{m['calibration_intercept']:+.6f}" ] for m in models]
    selected_rows=[]
    for phase in ['validation','comparison']:
        for arm in ['first_strategy','second_strategy','first_dense_ridge_24','first_dense_ridge_72','first_dense_calibrated_24']:
            for sc in ['base','stress']:
                row=next(x for x in r['rows'] if (x['venue'],x['phase'],x['arm'],x['scenario'])==('reference',phase,arm,sc))
                m=row['metrics'];selected_rows.append([phase,arm,sc,f"{m['net_pnl']:+,.2f}",f"{m['financial_score']:.8f}",m['completed_pairs'],m['aborted_pairs']])
    cost_rows=[]
    for x in attribution['rows']:
        if x['venue']!='reference' or x['phase']!='comparison' or x['arm'] not in ['first_dense_ridge_24','first_dense_ridge_72']:continue
        a=x['aggregate'];cost_rows.append([x['arm'],x['scenario'],*[f'{a[k]:+,.2f}' for k in ['price_move','execution_cost','fees','funding','net_pnl']]])
    forecast_rows=[]
    for h in [24,72]:
        for x in audit[f'reference_comparison_{h}']['rows']:
            pair='BTC/ETH' if x['pair_ids']==[0,1] else 'BTC/HYPE'
            forecast_rows.append([pair,h,x['kind'],x['rows'],f"{x['rmse_bps']:.2f}",f"{x['mean_forecast_bps']:+.2f}",f"{x['mean_actual_bps']:+.2f}"])
    body='''<h1>장기 자료·고정 만기 페어 예측 비교</h1>
<p class="note"><strong>이번 12개 후보 중 기존 두 전략보다 PnL과 Q를 일관되게 개선한 후보는 없다.</strong> 기본 비용의 8월 구간에서 좋아진 24시간 릿지 후보는 비용을 높이거나 이후 기간으로 옮기면 개선이 유지되지 않았다. 136개 계좌 재현을 모두 보존했으며, 실거래 전략은 교체하지 않았다.</p>
<h2>이번 수정</h2><p>희소한 3σ 진입 사건만 학습하던 방식을 벗어나 BTC/ETH, BTC/HYPE 각각의 고정 헤지 수익을 6시간마다 학습했다. 미래 24시간·72시간 가격 이동을 예측한다. 목표는 신호 시점 가격에서 만기 가격까지의 고정 수량 현금 변화이며, 아직 알 수 없는 진입 가격을 학습 목표의 기준점으로 쓰지 않는다. 예측 후 현재 거래 가격으로 남은 이익을 다시 계산한다.</p>
<p>종전처럼 중간에 새 예측이 뒤집혔다고 만기 전 청산하지 않는다. 만기 한 체결 단계 전에 청산 의도를 생성하되 유동성이 부족하면 완료될 때까지 유료 청산을 이어 간다. 실제 보유 수량의 손실이 진입 시점 변동성 기준 3σ√h를 넘으면 조기 종료하며, 기존 계좌 위험 한도도 유지한다. 실제 종료가 지연되거나 손절된 결과는 그대로 포함하므로 학습 목표 수익과 거래 손익이 같다고 가정하지 않는다.</p>
<p>원래 첫째·둘째 전략, 기존 페어 단독과 두 혼합 대조군을 보존했다. 새 후보는 각각의 기존 전략에 zero·ridge·calibrated 예측을 24시간·72시간으로 결합한 12개다. zero는 미래 가격 알파를 0으로 두고 현재 거래소 가격 차이와 알려진 펀딩만 남긴 대조군이다. calibrated는 릿지·깊이 2의 부스팅 트리 평균 예측을 과거 별도 기간에서 보정한다.</p>
<h2>학습과 보정 기간</h2><p>공식 Binance USD-M 월별 자료의 BTC/ETH 27,720개 공통 시간봉, BTC/HYPE 6,590개 공통 시간봉을 사용했다. 학습 목표 만기는 2026년 1월 1일 전, 보정 목표 만기는 3월 1일 전으로 제한했다. 1~2월은 학습 가중치 추정에서 분리하고, 보정에서는 같은 페어의 만기 구간이 겹치지 않게 표본을 골랐다. 학습의 6시간 간격 목표는 서로 겹치므로 아래 행 수를 독립 표본 수로 해석하면 안 된다.</p>
<p>표준화는 학습 자료에서만 추정했다. 학습한 모델은 Hyperliquid의 8~10월 결과로 재학습하지 않았다. 가격·수량·자산 배열은 식별자로 연결하며 두 거래소 가격을 이어 붙이지 않았다. 실제 par 미래 오라클이나 미래 호가를 사용하지 않았다. 미래의 par 상대 가격 괴리가 사라진다는 가정은 여전히 검증 대상이다.</p>'''+grid(['페어','만기 h','학습 행','보정 행','보정 기울기','보정 절편(표준화 단위)'],model_rows)+'''
<p>네 모델의 보정 기울기가 모두 0이었다. 분리한 1~2월에서 원래 혼합 예측의 양의 관계가 확인되지 않아 보정 후보는 과거 평균을 축소한 절편만 남겼다는 뜻이다. 이 사실을 숨기거나 보정 강도를 나중에 풀지 않았다. 릿지 단독은 사전에 선언한 별도 비교 후보로 유지했다.</p>
<h2>계좌 성과</h2><p>초기자본 1,000,000, 동일한 위험 한도. reference validation은 8/01 12:00~9/01 12:00 UTC, comparison은 9/01 12:00~10/15 12:00 UTC이다. 아래는 기준시장 백테스트이며 실제 대회 지갑 수익이 아니다. base는 편도 수수료 9bp·불리한 체결 가격 10bp, stress는 12bp·20bp와 더 낮은 거래량 참여율 및 펀딩 수령액 50%이다.</p>'''+grid(['기간','전략','조건','비용 차감 손익','Q','완료 페어','진입 중단'],selected_rows)+'''
<p>8월에 PnL·Q가 두 기존 전략보다 높고, 양의 수익·최소 3개 완료 페어·위험 한도 위반 없음이 기본/불리한 조건 모두에서 성립해야 한다는 선정 기준을 고정했다. 통과 후보는 없다. first_dense_calibrated_24는 진단 대표이며 교체 승인 후보가 아니다. 대표는 이후 기간과 par 계좌 재현 전에 고정했다. Hyperliquid 평가 과거는 여러 번 사용한 개발 자료이므로 독립적인 미래 검증이라고 부르지 않는다.</p>
<h2>손실 원인 분리</h2><p>아래는 이후 기준시장 구간의 페어 소유 원장만 분리한 결과다. 계좌 전체 성과에는 기존 전략의 수량·자본 반응도 들어가므로 두 값을 혼동하지 않는다. 양의 실행 비용·수수료는 손익에서 차감한다. 내부 상쇄는 거래소 수수료를 만들지 않고, 실제 외부 체결에만 부과한다. 펀딩도 물리 계좌와 소유 원장 합이 일치한다.</p>'''+grid(['후보','조건','가격 변화','실행 비용','수수료','펀딩','페어 순손익'],cost_rows)+'''
<h2>예측이 맞았는지 별도 점검</h2><p>다음은 이후 기준시장 구간에서 같은 페어의 목표 구간을 겹치지 않게 고른 가격 예측 진단이다. 페어 사이에는 공통 BTC 노출이 있으므로 전체 행이 독립적인 것은 아니다. 비용·펀딩 이전의 가격 예측 오차이며, 작은 RMSE가 수익성 보장을 뜻하지 않는다. BTC/HYPE의 릿지는 이후 기간에 평균 음의 이동을 예상했지만 실제 평균은 양수였고, 0 이동 대조군보다 RMSE도 컸다.</p>'''+grid(['페어','만기 h','모델','표본','RMSE bp','평균 예측 bp','평균 실제 bp'],forecast_rows)+'''
<h2>실제 par 자료의 범위</h2><p>par는 10/01~10/16의 실제 5분 캔들과 펀딩을 사용한 조건부 체결 재현이다. 두 주문과 청산은 독립적으로 처리하고 한쪽만 체결되면 알려진 다음 관측에서 한 번만 보완한다. 그때도 체결되지 않으면 비용을 내고 되돌린다. 거래량 대용치가 미체결을 만드는 경우는 실제 호가에 유동성이 없었다는 증거가 아니다.</p>
<p>기존 허용된 호가 기록 다섯 개의 전체 시간 범위와 이 후보의 6시간 간격·1시간 진입 유효기간을 대조했다. 겹치는 시간은 0초다. 따라서 이 후보의 실제 호가 진입·24/72시간 뒤 청산 수익을 확인했다고 주장할 수 없다. 관측 구간에 맞추려고 신호 시각을 이동하거나 과거에 없던 호가를 만들지 않았다. 예약된 미래 자료는 열지 않았다.</p>
<h2>검증과 다음 작업</h2><p>검사 11개 통과: 미래 가격 변경에 대한 학습·체결 불변, 신호 기준 목표, 자산 식별 보존, 만기 유지, 한쪽 가격 미체결 후 유료 보완, 거래량이 없는 보완 주문의 유료 철수, 기존 다섯 대조군의 결과 일치 등이다. 모든 136개 계좌 비교에서 기존 대조군의 체결·잔고·지표가 정확히 같았다. 실거래 소스 373개와 예약 의존 파일 345개는 변경하지 않았다.</p>
<p>다음 수정은 학습 자료를 늘렸다는 이유로 전이된 예측을 신뢰하는 대신, <strong>만기가 이미 지난 최근 Hyperliquid 관측으로 과거 학습 모델의 오차와 방향 안정성을 보정</strong>하는 것이다. 단순한 최근 상대 수익 및 0 예측 대조군을 함께 두고, 성과가 나쁜 페어를 사후 제거한 결과와 구분한다. 원래의 90일 재학습 실험과 달리 장기 모델은 고정하고, 당시 확인 가능한 잔차 보정만 추가한다. 구현·평가 전 규칙을 고정하고 이후 기간의 성과로 후보를 다시 고르지 않는다. 이 다음 후보의 성과는 아직 계산하지 않았다.</p>'''
    for venue in ['reference','par']:
        body+='<h2>'+('기준시장 전체 결과' if venue=='reference' else 'par 캔들·펀딩 모형 전체 결과')+'</h2>'+table([x for x in r['rows'] if x['venue']==venue])
    (out/'report.html').write_text(document('고정 만기 페어 예측 비교',body));export_csv(out/'comparison.csv',r['rows'])
    summary=dict(selection=r['selection'],actual_replays=r['actual_replays'],tests_passed=11,
        existing_controls_exact=r['existing_controls_exact'],all_calibration_slopes_zero=all(m['calibration_slope']==0 for m in models),
        maximum_ownership_error=max(x['metrics']['maximum_ownership_error'] for x in r['rows']),
        maximum_episode_accounting_error=diag['maximum_episode_accounting_error'],
        actual_book_entry_window_overlap_seconds=book['total_entry_window_overlap_seconds'],
        protected=guards,orders_sent=0,automatic_live_replacement=False,historical_independence=False)
    save(out/'summary.json',summary)
    save(HERE/'work_state.json',dict(status='progress_dense_model_and_fixed_maturity_replays_complete_no_candidate_qualified',
        report=str((out/'report.html').relative_to(ROOT)),actual_replays=r['actual_replays'],tests_passed=11,
        selection=r['selection'],orders_sent=0,live_strategy_changed=False,
        findings=['Dense older training produced trades; unlike sparse-event candidates, lack of signals is no longer the only issue.',
            'First plus 24h ridge improves August base PnL/Q, but fails August stress and loses in the later period.',
            'All chronological calibration slopes are zero; BTC/HYPE transferred forecasts have the wrong later-period average direction.',
            'Pair-owned price movement and actual execution/funding/fee contributions reconciled for all 96 candidate runs.',
            'Par candle replays have tiny fills or none, with no robust gain; this does not prove absence of exchange liquidity.',
            'Five existing L2 tape bounds have no overlap with fixed six-hour source plus one-hour entry windows.'],
        next='Freeze a past-maturity-only recent Hyperliquid residual calibration of fixed older-history models, with zero and recent-relative-return controls. Distinguish from prior 90-day full model refits. Evaluate same account/cost constraints and preserve failed outcomes; no result-driven schedule alignment to observed books.',
        remaining=['No PnL/Q improvement meeting required comparisons','Actual full-period par quote execution and exits unverified','No independent prospective improvement evidence']))
    print(json.dumps(summary,ensure_ascii=False));print(out/'report.html')


if __name__=='__main__':main()
