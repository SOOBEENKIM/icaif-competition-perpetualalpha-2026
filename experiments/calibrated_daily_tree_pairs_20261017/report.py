"""Calibration comparison, partial historical gains, no live promotion."""
import csv,html,json,re
from pathlib import Path
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha
from experiments.complementary_alpha_20261017.research import protected

HERE=Path(__file__).resolve().parent;ART=HERE/'artifacts'


def table(headers,rows):
    def row(values,tag):return '<tr>'+''.join(f'<{tag}>{html.escape(str(x))}</{tag}>' for x in values)+'</tr>'
    return '<table>'+row(headers,'th')+''.join(row(r,'td') for r in rows)+'</table>'


def main():
    result=json.loads((ART/'results.json').read_text());attrib=json.loads((ART/'attribution.json').read_text())
    forecasts=json.loads((ART/'forecast_diagnostics.json').read_text());calib=json.loads((ART/'calibration_diagnostics.json').read_text())
    flow=json.loads((ART/'entry_funnel.json').read_text());books=json.loads((ART/'books/results.json').read_text())
    plan=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())
    for manifest in [plan,json.loads((ART/'frozen.json').read_text()),json.loads((ART/'books/frozen.json').read_text())]:
        for path,digest in manifest['sources'].items():assert sha(ROOT/path)==digest,path
    test=(HERE/'test_results.txt').read_text();assert test.rstrip().endswith('OK')
    count=int(re.search(r'Ran (\d+) tests',test).group(1));guard=protected()
    assert len(result['rows'])==112 and not result['selection']['eligible']
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in result['rows']}
    def metrics(venue,phase,kind,sc):return lookup[venue,phase,f'first_dense_hourlyhedge_{kind}_24',sc]
    improvement=[]
    for phase in ['validation','comparison']:
        for sc in ['base','stress']:
            m=metrics('reference',phase,'cmix',sc)
            for base in ['first_strategy','second_strategy']:
                b=lookup['reference',phase,base,sc]
                improvement.append(dict(phase=phase,scenario=sc,baseline=base,
                    pnl_difference=m['net_pnl']-b['net_pnl'],q_difference=m['financial_score']-b['financial_score'],
                    volatility_difference=m['volatility_horizon']-b['volatility_horizon'],
                    drawdown_difference=m['max_drawdown']-b['max_drawdown']))
    assert all(r['pnl_difference']>0 and r['q_difference']>0 for r in improvement)
    new_books=[r for r in books['trials'] if r['model'] in ['ctree','cmix']];assert not new_books
    dest=ROOT/'reports/calibrated_daily_tree_pairs_20261017';dest.mkdir(exist_ok=True)
    exported=[]
    for r in result['rows']:
        exported.append({k:r[k] for k in ['venue','phase','arm','scenario']}|{k:r['metrics'].get(k) for k in
            ['net_pnl','return_fraction','financial_score','score','daily_sample_std','volatility_horizon','max_drawdown',
             'completed_pairs','pair_owned_pnl','fees','execution_cost','funding','mean_gross','residual_gross','pnl_after_liquidation_reserve']})
    with (dest/'account_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(exported[0]));writer.writeheader();writer.writerows(exported)
    body='''<h1>매일 학습한 트리의 수익 크기 보정</h1>
<p><strong>보정 혼합은 재사용 기준시장 네 조건에서 기존 첫째·둘째보다 PnL/Q가 높았다.
그러나 유지한 회귀보다 이후 성과가 낮고 실제 par 개선이 없어 실거래 교체를 보류한다.</strong>
새 계좌 재생 32개, 기존 트리 계좌 경로 대조 4개, 인과성·가용성 테스트 7개를 완료했다.</p>
<h2>이번 변경</h2>
<p>보정 트리(ctree)는 과거 예측에 대한 기울기를 0~1 범위로 제한하고 절편을 조정한다.
보정 혼합(cmix)은 유지한 회귀와 보정 트리를 50:50으로 합친다.
두 페어, 24시간 목표, 매매·규모·위험 한도·비용·갱신 규칙은 그대로다.</p>
<p>여기서 과거 예측은 <b>당시까지의 자료만 사용해 재현한 값</b>이다. 당시 실제 운영한 봇이 발행한 기록은 아니다.
각 예측의 24시간 목표가 완성된 뒤에만 보정에 사용한다. 현재 모델로 과거 훈련 자료에 다시 적합한 값은 사용하지 않는다.
90일 창의 하루 간격 표본, 30일 반감기, 최소 12개, 정규화 사전 강도 12를 고정했다.
미래 목표와 미래 종료 시점의 관측 상태를 바꿔도 현재 출력이 같아야 한다는 검사를 통과했다.</p>
<h2>동일 자본 기준시장 비교</h2>
<p>초기 자본 1,000,000. validation: 8월 1일 12시~9월 1일 12시 UTC;
comparison: 9월 1일 12시~10월 15일 12시 UTC. 이 기간은 이전 실험에서도 사용한 개발 자료다.
기본 편도 수수료 9bp·가격 불이익 10bp, stress는 12bp·20bp와 더 제한된 수량 및 펀딩 수령액 절반이다.
최종 보유분 평가손익을 포함하며, 잔존 명목금액과 청산 준비금 차감값은 CSV에 별도 기록했다.</p>'''
    labels=[('first_strategy','기존 첫째'),('second_strategy','기존 둘째'),
        *[(f'first_dense_hourlyhedge_{kind}_24',label) for kind,label in
          [('direct','첫째＋유지한 회귀'),('tree','첫째＋원래 트리'),('ctree','첫째＋보정 트리'),('cmix','첫째＋보정 혼합')]]]
    rows=[]
    for phase in ['validation','comparison']:
        for sc in ['base','stress']:
            for arm,label in labels:
                m=lookup['reference',phase,arm,sc]
                rows.append([phase,sc,label,f'{m["net_pnl"]:+,.2f}',f'{m["return_fraction"]*100:+.5f}%',
                    f'{m["financial_score"]:.8f}',f'{m["volatility_horizon"]*100:.5f}%',f'{m["max_drawdown"]*100:.5f}%',m['completed_pairs']])
    body+=table(['기간','조건','전략','순손익','수익률','Q','기간 σ','최대낙폭','완료 페어'],rows)
    body+='''<p>Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05), σ는 완료한 12시 UTC 일별 수익률의 표본 표준편차×√n이다.
보정 혼합의 8월 기본·stress 순손익은 +23,258.31·+22,946.80으로 좋아졌지만 완료 페어가 각각 2개여서 사전 최소 3개 기준에 못 미쳤다.
이후 +1,426.56·+1,915.04는 유지한 회귀의 +3,422.16·+2,104.98보다 낮다.
기존 두 전략보다 Q가 높아도 변동성과 최대낙폭이 모두 개선됐다는 뜻은 아니다.</p>
<h2>보정이 실제로 바꾼 것</h2>
<p>아래는 각 구간의 모든 시간별 보정 계수다. 수익을 본 뒤 계수를 선택하지 않았다.
기울기로 예측 크기를 낮추는 동시에 절편으로 최근 오차를 보정하므로 방향이 바뀔 수도 있다.</p>'''
    body+=table(['기간','페어','보정 표본 수 범위','기울기 최소/중앙/최대','절편 절대값 평균 bp','기울기×원예측 절대값 평균 bp'],[
        [r['phase'],'/'.join(r['pair']),str(r['rows_range']),'/'.join(f'{x:.3f}' for x in r['slope_quantiles']),
         f'{r["mean_abs_intercept_bp"]:.3f}',f'{r["mean_abs_slope_prediction_bp"]:.3f}'] for r in calib['rows']])
    body+='''<h2>동일 표본의 예측 오차</h2>
<p>하루 간격의 24시간 목표이며 시점별 변동성으로 정규화했다. 학습·보정과 분리한 진단이지만,
이미 여러 번 사용한 기간이므로 독립적인 통계적 발견으로 해석하지 않는다.</p>'''
    body+=table(['기간','페어','모델','표본','RMSE','0 예측 RMSE','방향 일치율','평균 절대 예측 bp'],[
        [r['phase'],'/'.join(r['pair']),r['kind'],r['samples'],f'{r["normalized_rmse"]:.5f}',f'{r["zero_rmse"]:.5f}',
         f'{100*r["sign_accuracy"]:.2f}%',f'{r["mean_abs_forecast_bp"]:.3f}'] for r in forecasts['rows'] if r['kind'] in ['direct','ctree','cmix']])
    body+='''<h2>이후 구간의 비용별 손익</h2>'''
    rows=[]
    for r in attrib['rows']:
        if r['phase']!='comparison' or r['kind'] not in ['ctree','cmix']:continue
        for pair,v in r['by_pair'].items():rows.append([r['kind'],r['scenario'],pair,*[f'{v[k]:+,.2f}' for k in
            ['price_move','execution_cost','fees','funding','net_pnl']]])
    body+=table(['모델','조건','페어','가격 손익','가격 불이익','수수료','펀딩','페어 순손익'],rows)
    body+='''<h2>par 거래 가능성</h2>
<p>새 보정 후보의 par 캔들 재생에서 완료 페어는 모두 0개였다.
다섯 개의 과거 실제 호가 기록에서도 보정 후보가 새로 진입한 사례는 0개였다.
원래 트리의 두 진입 사례만 다시 재현했으며 그 사례도 24시간 뒤 청산 결과는 없다.
아래는 계좌 위험·보유 상태·실제 지연 체결 전의 캔들 진입 진단이다.</p>'''
    body+=table(['모델','조건','유효 판단','수량·최소주문·헤지 통과','비용＋5bp 초과'],[
        [r['model'],r['scenario'],*[r['counts'].get(k,0) for k in
            ['valid_pair_decisions','volume_lot_minimum_and_hedge_pass','positive_buffered_edge_before_global_risk']]] for r in flow['rows']])
    body+='''<h2>판단과 다음 확인</h2>
<p>보정 혼합은 기존 두 전략 대비 역사적 개선을 보인 보조 연구 후보로 보관한다.
현재 유지한 회귀 후보와 실제 운용 전략을 교체하지 않는다. 더 많은 같은 기간 변형에 앞서,
예약 평가 시작 경계 이전의 추가 과거 par 캔들·펀딩 자료를 가져올 수 있는지 확인한다.
기존 예측 규칙과 후보 순위를 고정한 상태에서 새 날짜의 체결 가능성과 성과를 비교하며,
이미 본 기준시장 가격과의 연관 때문에 이것 역시 완전히 독립적이지 않다는 점을 구분한다.
예약 원자료를 미리 열거나 그 평가의 고정 파일을 변경하지 않는다.</p>
<p><a href="account_results.csv">전체 112개 계좌 비교 CSV</a></p>'''
    (dest/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>일별 트리 보정 비교</title><style>body{font:15px/1.6 system-ui;max-width:1280px;margin:32px auto;color:#183048}table{border-collapse:collapse;width:100%;font-size:12px;margin:18px 0}th,td{padding:7px;border:1px solid #d9e2eb;text-align:right}th{background:#eef3f7}th:first-child,td:first-child{text-align:left}</style>'+body+'</html>')
    save(HERE/'work_state.json',dict(status='progress_32_causal_tree_calibration_replays_secondary_historical_candidate_only',
        previous_goal_turn='progress_daily_tree_32_replays_rejected',new_account_replays=32,exact_control_replays=4,
        tests_passed=count,selection=result['selection'],cmix_native_comparisons=improvement,
        actual_book_new_candidate_entries=len(new_books),actual_book_control_entries=len(books['trials']),
        completed_actual_book_exits=0,orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,
        historical_independence=False,protected=guard,frozen_reserved_sources_unchanged=len(plan['sources']),
        report=str((dest/'report.html').relative_to(ROOT)),
        reference_first_pnl={kind:{phase:{sc:metrics('reference',phase,kind,sc)['net_pnl'] for sc in ['base','stress']}
            for phase in ['validation','comparison']} for kind in ['direct','tree','ctree','cmix']},
        findings=['Cmix beats native first/second PnL and Q in all four reused reference cells, but has fewer than 3 completed pairs in each cell.',
            'Cmix later +1426.56/+1915.04 remains below retained direct +3422.16/+2104.98; volatility/drawdown not uniformly improved.',
            'Current target and future terminal freshness cannot leak into current calibrated forecast; seven tests passed.',
            'Calibration uses causally reconstructed historical daily fits, not actual past live-broadcast forecasts.',
            'No new actual old-book entries; no par profitability proven.'],
        next='Before more model variants, audit reserved evaluation boundaries and fetch/append additional historical par candle/funding data only up to a fixed cutoff at or before 2026-10-17T12:00Z (new common prospective start). Do not read any reserved book raw or outcomes. Check older reserved plans too. Reuse fixed existing candidate/model configurations, preserve exact old input prefix and controls, distinguish newly fetched venue history from independent unseen market evidence. Current published input snapshot practice_live/updates/20261017T120009814481Z exists but has not been read in this turn. Preserve 1045 frozen sources and original raw release 2026-10-24T12:01Z.',
        goal_complete=False))
    paths=[HERE/'diagnostics.py',HERE/'book_attribution.py',HERE/'report.py',ROOT/'experiments/sector_pairs_20261017/diagnostics.py',
        *[ART/n for n in ['attribution.json','forecast_diagnostics.json','calibration_diagnostics.json','entry_funnel.json','books/results.json','book_value_attribution.json']]]
    save(ART/'diagnostics_manifest.json',dict(sources={str(p.relative_to(ROOT)):sha(p) for p in paths},orders_sent=0))
    print('REPORT',dest/'report.html','new_replays',32,'tests',count,'frozen_sources',len(plan['sources']))


if __name__=='__main__':main()
