"""Daily nonlinear result and the limits of direction-only evidence."""
import csv,html,json,re
from pathlib import Path
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha
from experiments.complementary_alpha_20261017.research import protected

HERE=Path(__file__).resolve().parent;ART=HERE/'artifacts'


def table(headers,rows):
    def row(values,tag):return '<tr>'+''.join(f'<{tag}>{html.escape(str(x))}</{tag}>' for x in values)+'</tr>'
    return '<table>'+row(headers,'th')+''.join(row(r,'td') for r in rows)+'</table>'


def main():
    result=json.loads((ART/'results.json').read_text());a=json.loads((ART/'attribution.json').read_text())
    forecast=json.loads((ART/'forecast_diagnostics.json').read_text());flow=json.loads((ART/'entry_funnel.json').read_text())
    books=json.loads((ART/'books/results.json').read_text());bv=json.loads((ART/'book_value_attribution.json').read_text())
    models=json.loads((ART/'trained_model_manifest.json').read_text())
    plan=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())
    for manifest in [plan,models,json.loads((ART/'frozen.json').read_text()),json.loads((ART/'books/frozen.json').read_text())]:
        for path,digest in manifest['sources'].items():assert sha(ROOT/path)==digest,path
    tests=(HERE/'test_results.txt').read_text();assert tests.rstrip().endswith('OK')
    count=int(re.search(r'Ran (\d+) tests',tests).group(1));guard=protected()
    assert not result['selection']['eligible'] and len(result['rows'])==80
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in result['rows']}
    def metrics(venue,phase,kind,scenario):return lookup[venue,phase,f'first_dense_hourlyhedge_{kind}_24',scenario]
    dest=ROOT/'reports/daily_tree_pairs_20261017';dest.mkdir(exist_ok=True)
    export=[]
    for r in result['rows']:
        export.append({k:r[k] for k in ['venue','phase','arm','scenario']}|{k:r['metrics'].get(k) for k in
            ['net_pnl','return_fraction','financial_score','score','daily_sample_std','volatility_horizon','max_drawdown',
             'completed_pairs','pair_owned_pnl','fees','execution_cost','funding','mean_gross','residual_gross','pnl_after_liquidation_reserve']})
    with (dest/'account_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(export[0]));writer.writeheader();writer.writerows(export)
    body='''<h1>일별 얕은 트리와 회귀·트리 혼합 비교</h1>
<p><strong>트리는 8월 성과를 개선했으나 이후 구간에서 유지하지 못했다. 두 새 후보 모두 채택하지 않는다.</strong>
새 계좌 재생 32개, 기존 회귀 계좌 경로 대조 4개, 인과성·혼합·저장 모델 테스트 6개를 완료했다.
실거래 주문과 전략 교체는 없다.</p>
<h2>구현</h2>
<p>BTC/ETH와 BTC/HYPE를 따로 학습한다. 매일 00시 UTC에 최근 90일 중 이미 24시간 목표가 완성된 자료만 사용한다.
입력은 기존 가격 신호 그대로이며, 30일 반감기·6시간 학습 표본·최소 200행·목표 범위 제한도 같다.
트리는 깊이 2, 80회, 학습률 0.04, 최소 자식 가중치 40, L2 10으로 기존 고정 트리의 설정을 가져왔다.
새로운 차이는 최근 자료로 매일 다시 학습하는 점이다. 혼합은 회귀 50%·트리 50%로 고정했다.</p>
<p>트리의 표본 가중치는 행 수 합계로 맞춘다. 최소 자식 가중치 40은 독립 표본 40개를 뜻하지 않으며,
겹치는 24시간 목표로 학습한다. 트리 잎의 규제와 회귀 계수의 규제는 동등한 값이 아니다.
변경 범위는 예측 학습기와 그 고정 혼합이며 매매 규칙·규모·위험 한도·비용은 유지했다.</p>
<h2>기준시장 계좌 비교</h2>
<p>초기 자본 1,000,000. validation: 8월 1일 12시~9월 1일 12시 UTC;
comparison: 9월 1일 12시~10월 15일 12시 UTC. 모두 재사용한 개발 자료다.
기본 편도 수수료 9bp·가격 불이익 10bp, stress는 12bp·20bp와 더 제한된 수량 및 펀딩 수령액 절반이다.
순손익에는 마지막 가격으로 평가한 잔존 포지션이 포함된다. 잔존 명목금액·청산 준비금 차감값은 CSV에 남겼다.</p>'''
    rows=[]
    for phase in ['validation','comparison']:
        for sc in ['base','stress']:
            for arm,label in [('first_strategy','기존 첫째'),('second_strategy','기존 둘째'),
                *[(f'first_dense_hourlyhedge_{kind}_24',label) for kind,label in
                  [('direct','첫째＋회귀 페어'),('tree','첫째＋트리 페어'),('mix','첫째＋혼합 페어')]]]:
                m=lookup['reference',phase,arm,sc]
                rows.append([phase,sc,label,f'{m["net_pnl"]:+,.2f}',f'{m["return_fraction"]*100:+.5f}%',
                    f'{m["financial_score"]:.8f}',f'{m["volatility_horizon"]*100:.5f}%',
                    f'{m["max_drawdown"]*100:.5f}%',m['completed_pairs']])
    body+=table(['기간','조건','전략','순손익','수익률','Q','기간 σ','최대낙폭','완료 페어'],rows)
    body+='''<p>Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05). σ는 완료한 12시 UTC 일별 수익률의 표본 표준편차×√n이다.
8월 트리는 두 비용 조건에서 기존 두 전략과 유지한 회귀보다 PnL/Q가 높았다.
하지만 stress 완료 페어가 2개여서 미리 정한 최소 3개 기준을 통과하지 못했다.
그 조건을 완화하더라도 이후 기간의 성과 악화가 남으므로 교체 근거가 되지 않는다.</p>
<h2>왜 방향을 맞혀도 수익이 나지 않았는가</h2>
<p>이후 BTC/HYPE의 일별 진단에서는 트리가 43개 중 27개의 방향을 맞혔다.
이것은 실제 체결 거래의 승률이 아니다. 비용 조건으로 선택된 진입과 갱신은 다른 시점·수량을 사용한다.
실제 계좌 경로의 해당 페어는 기본 조건에서 가격 이익 +1,395.48에 비해 가격 불이익·수수료가 합계 4,136.08이었다.
방향뿐 아니라 움직임의 크기와 거래 횟수가 비용을 감당해야 한다.</p>'''
    rows=[]
    for r in a['rows']:
        if r['phase']!='comparison' or r['kind']=='direct':continue
        for pair,v in r['by_pair'].items():rows.append([r['kind'],r['scenario'],pair,*[f'{v[k]:+,.2f}' for k in
            ['price_move','execution_cost','fees','funding','net_pnl']]])
    body+=table(['모델','조건','페어','가격 손익','가격 불이익','수수료','펀딩','페어 순손익'],rows)
    body+='''<h2>동일 표본 예측 오차</h2>
<p>24시간 목표가 겹치지 않는 하루 간격 표본이다. RMSE는 당시 페어 변동성으로 정규화한 목표 기준이다.
가격 0 변화 예측도 같이 제시한다. 방향 정확도가 높아도 오차의 크기와 순손익이 좋아진다고 볼 수 없다.</p>'''
    body+=table(['기간','페어','모델','표본','RMSE','0 예측 RMSE','방향 일치율','평균 절대 예측 bp'],[
        [r['phase'],'/'.join(r['pair']),r['kind'],r['samples'],f'{r["normalized_rmse"]:.5f}',
         f'{r["zero_rmse"]:.5f}',f'{100*r["sign_accuracy"]:.2f}%',f'{r["mean_abs_forecast_bp"]:.3f}'] for r in forecast['rows']])
    body+='''<h2>실제 par 자료</h2>
<p>10월 1~16일 캔들 재생의 새 후보들에서 완료 페어는 모두 0개였다.
이후 기본 조건 순손익은 트리 −89.32, 혼합 −87.93, 회귀 −87.89로 개선되지 않았다.
아래 기회 표는 계좌 보유·전역 위험·지연 체결 전에 보는 캔들 진단이며 실제 주문장 유동성을 뜻하지 않는다.</p>'''
    body+=table(['모델','조건','유효 판단','수량·최소주문·헤지 통과','비용＋5bp 초과'],[
        [r['model'],r['scenario'],*[r['counts'].get(k,0) for k in
            ['valid_pair_decisions','volume_lot_minimum_and_hedge_pass','positive_buffered_edge_before_global_risk']]] for r in flow['rows']])
    body+='''<p>다섯 개의 과거 실제 호가 기록에서는 트리 후보만 2개 조건 사례의 양쪽 진입이 가능했다.
실제 24시간 뒤 청산 호가가 없어 이 사례의 실현 수익이나 Q를 산정하지 않았다.
다음 표의 예상값은 원래 계획 수량 기준이며 실제 체결 후 수익이 아니다.</p>'''
    body+=table(['페어','조건','주문 전 예상 순현금','모델 증분 제거','현재 중간가격 앵커 가정','관측 진입 수수료'],[
        ['/'.join(r['pair']),r['scenario'],*[f'{r[k]:+.5f}' for k in
            ['expected_net_cash','without_model_increment','without_venue_basis_or_source_aging','known_entry_fees']]] for r in bv['rows']])
    body+='''<h2>다음 검토</h2>
<p>단순 50:50 혼합을 채택하지 않는다. 다음에는 트리의 예상 수익 크기를 이미 발행했던 예측과 나중에 확정된 실제 수익으로
보정할 수 있는지 시험한다. 같은 학습 표본에 다시 예측한 적합값을 보정에 쓰지 않고, 당시 사용 가능했던 모델의 예측만 사용해야 한다.
기존 고정 모델의 잔차·기울기 보정과 먼저 대조하고, 방법과 계수는 이후 평가 전에 고정한다.
이것도 개선이 보장되는 변경은 아니며, 기존 회귀 후보와 실제 운용을 유지한다.</p>
<p><a href="https://xgboost.readthedocs.io/en/stable/parameter.html">공식 XGBoost 문서</a>에서 가중치·초기값·규제의 의미를 확인했다.
설치된 3.2.0의 저장·재로딩과 실제 예측은 로컬 테스트로 확인했다. 이번 실험은 특정 논문의 성과를 재현했다고 주장하지 않는다.</p>
<p><a href="account_results.csv">전체 80개 계좌 비교 CSV</a></p>'''
    (dest/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>일별 트리 페어 비교</title><style>body{font:15px/1.6 system-ui;max-width:1280px;margin:32px auto;color:#183047}table{border-collapse:collapse;width:100%;font-size:12px;margin:18px 0}th,td{padding:7px;border:1px solid #d9e0e7;text-align:right}th{background:#edf2f7}th:first-child,td:first-child{text-align:left}</style>'+body+'</html>')
    save(HERE/'work_state.json',dict(status='progress_32_daily_tree_and_mix_replays_rejected_after_later_decay',
        previous_goal_turn='progress_32_matched_horizon_replays_rejected',new_account_replays=32,
        exact_control_replays=4,tests_passed=count,trained_tree_models=models['models'],selection=result['selection'],
        actual_book_scenario_cases=len(books['trials']),actual_book_groups=bv['source_tape_pair_groups'],
        completed_actual_book_exits=0,orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,
        protected=guard,frozen_reserved_sources_unchanged=len(plan['sources']),historical_independence=False,
        report=str((dest/'report.html').relative_to(ROOT)),
        reference_first_pnl={kind:{phase:{sc:metrics('reference',phase,kind,sc)['net_pnl'] for sc in ['base','stress']}
            for phase in ['validation','comparison']} for kind in ['direct','tree','mix']},
        findings=['Daily shallow tree improves August PnL/Q over native and direct in both costs but has only 2 stress completed pairs.',
            'Later tree +379.58/-1630.42 and equal mixture +2169.73/-5117.41 versus direct +3422.16/+2104.98: no adoption.',
            'BTC/HYPE later daily sign accuracy 27/43 is not a filled-trade win rate; selected pair price gains do not cover costs.',
            'Two old-book tree entry cases have no observed 24h exits; par candle completed pairs zero.'],
        next='Audit prior calibration variants, then test calibration of daily tree forecasts using strictly matured historical OUT-OF-SAMPLE issued tree predictions, never same-training fitted values. Keep horizon24, inputs, risk, account engine and costs fixed. Freeze calibration parameters/selection before later replay and compare with retained direct24 and raw tree. Keep all prospective sources/old models unchanged; no reserved raw read before original release and no live replacement.',
        goal_complete=False))
    paths=[HERE/'diagnostics.py',HERE/'book_attribution.py',HERE/'report.py',
        ROOT/'experiments/sector_pairs_20261017/diagnostics.py',
        *[ART/n for n in ['attribution.json','forecast_diagnostics.json','entry_funnel.json','books/results.json','book_value_attribution.json','trained_model_manifest.json']]]
    save(ART/'diagnostics_manifest.json',dict(sources={str(p.relative_to(ROOT)):sha(p) for p in paths},orders_sent=0))
    print('REPORT',dest/'report.html','new_replays',32,'tests',count,'models',models['models'],'frozen_sources',len(plan['sources']))


if __name__=='__main__':main()
