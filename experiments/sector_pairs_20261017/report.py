"""Report a rejected universe extension without relabeling open probes as PnL."""
import csv,html,json,re
from pathlib import Path
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha
from experiments.complementary_alpha_20261017.research import protected

HERE=Path(__file__).resolve().parent;ART=HERE/'artifacts'


def table(headers,rows):
    render=lambda row,tag:'<tr>'+''.join(f'<{tag}>'+html.escape(str(v))+f'</{tag}>' for v in row)+'</tr>'
    return '<table>'+render(headers,'th')+''.join(render(row,'td') for row in rows)+'</table>'


def main():
    data=json.loads((ART/'results.json').read_text());flow=json.loads((ART/'entry_funnel.json').read_text())
    attrib=json.loads((ART/'attribution.json').read_text());books=json.loads((ART/'books/results.json').read_text())
    value=json.loads((ART/'book_value_attribution.json').read_text())
    forecast=json.loads((ART/'forecast_diagnostics.json').read_text())
    for rel,digest in json.loads((ART/'frozen.json').read_text())['sources'].items():assert sha(ROOT/rel)==digest,rel
    for rel,digest in json.loads((ART/'books/frozen.json').read_text())['sources'].items():assert sha(ROOT/rel)==digest,rel
    test=(HERE/'test_results.txt').read_text();assert test.rstrip().endswith('OK')
    count=int(re.search(r'Ran (\d+) tests',test).group(1));guard=protected()
    destination=ROOT/'reports/sector_pairs_20261017';destination.mkdir(exist_ok=True)
    exported=[]
    for row in data['rows']:
        m=row['metrics'];exported.append({k:row[k] for k in ['venue','phase','arm','scenario']}|{
            k:m.get(k) for k in ['net_pnl','return_fraction','financial_score','volatility_horizon','max_drawdown','completed_pairs','fees','funding','pair_owned_pnl']})
    with (destination/'account_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(exported[0]));writer.writeheader();writer.writerows(exported)
    names={'first_strategy':'기존 첫째','second_strategy':'기존 둘째',
        'first_dense_hourlyhedge_direct_24':'첫째＋유지한 2페어 재학습',
        'first_dense_hourlyhedge_sector_24':'첫째＋확대한 6페어 재학습'}
    selected=[]
    for phase in ['validation','comparison']:
        for scenario in ['base','stress']:
            for arm,label in names.items():
                row=next(r for r in data['rows'] if (r['venue'],r['phase'],r['arm'],r['scenario'])==('reference',phase,arm,scenario));m=row['metrics']
                selected.append([phase,scenario,label,f'{m["net_pnl"]:+,.2f}',f'{m["return_fraction"]*100:+.5f}%',
                    f'{m["financial_score"]:.8f}',f'{m["volatility_horizon"]*100:.5f}%',f'{m["max_drawdown"]*100:.5f}%',m['completed_pairs']])
    body='''<h1>관련 자산 6페어 확대 실험: 채택하지 않음</h1>
<p><strong>같은 일별 재학습 모델의 페어 범위를 넓혔지만, 기존 두 페어 후보보다 계좌 성과가 나빠졌다.</strong>
새 계좌 재생 16개, 기존 계좌 경로 대조 4개, 기존 두 페어의 예측 보존 확인, 모델 테스트 4개를 완료했다.
실거래 주문과 전략 교체는 없다.</p>
<p>BTC/ETH, BTC/HYPE를 유지하고 ETH/HYPE, SP500/EWY, SP500/EWJ, EWY/EWJ를 추가했다.
후보별로 동시에 보유하는 페어는 여전히 하나다. 총 투자 규모·위험 한도·진입 조건·수수료·슬리피지·펀딩·유료 재조정 규칙은 변경하지 않았다.
매일 00시 UTC 학습, 90일 창, 30일 반감기, 6시간 표본, 최소 200개, 24시간 목표를 유지했다.</p>
<h2>동일 자본 기준시장 백테스트</h2>
<p>초기 자본 1M. validation: 8월 1일 12시~9월 1일 12시 UTC. comparison: 9월 1일 12시~10월 15일 12시 UTC.
이미 사용했던 과거 기간의 개발 비교이며 독립적인 미래 평가가 아니다. 기본 편도 수수료 9bp·가격 불이익 10bp,
stress는 12bp·20bp와 더 제한된 거래 수량 및 펀딩 수령액 절반을 사용했다.</p>'''
    body+=table(['기간','조건','전략','순손익','수익률','Q','기간 변동성','최대낙폭','완료 페어'],selected)
    body+='''<p>첫째 기반 확대 후보는 이후 기본 +419.96, stress −1,849.65로, 유지한 두 페어의 +3,422.16·+2,104.98보다 낮다.
8월 검증에서도 기본 +10,558.50·stress +7,391.63으로 기존 첫째 전략의 손익과 Q를 넘지 못했다.
선정 기록은 first_dense_hourlyhedge_sector_24, eligible=false다. 이후 결과를 보고 다른 후보로 바꾸지 않았다.</p>
<h2>손익 악화의 위치</h2><p>추가한 ETH/HYPE의 이후 가격 손익이 음수였다. 비용만 낮춰 해결됐다고 볼 수 없다.
아래는 페어 소유분 현금 경로를 수수료·가격 불이익·펀딩으로 대조한 사후 분해이며 후보 선정에 사용하지 않았다.</p>'''
    rows=[]
    for r in attrib['rows']:
        if r['phase']=='comparison' and r['arm'].startswith('first'):
            for pair,x in r['by_pair'].items():rows.append([r['scenario'],pair,*[f'{x[k]:+,.2f}' for k in ['price_move','execution_cost','fees','funding','net_pnl']]])
    body+=table(['조건','페어','가격 손익','가격 불이익 비용','수수료','펀딩','페어 순손익'],rows)
    body+='''<h2>실제 par 과거 캔들의 진입 제약</h2>
<p>10월 1일 00시~16일 12시 UTC. 각 시점에서 양방향 중 예상 순현금이 큰 쪽 하나를 센 값이다.
실제 계좌의 보유 여부·전역 위험·다음 봉의 체결·청산 전에 계산한 기회 진단이다.
거래량 없는 봉이 있다는 사실을 실제 주문장에 유동성이 없다는 뜻으로 해석하지 않는다.</p>'''
    body+=table(['모델','조건','유효 페어 판단','한쪽 이상 거래량 0','수량·최소주문·헤지 통과','비용＋5bp 초과'],[
        [r['model'],r['scenario'],*[r['counts'].get(k,0) for k in ['valid_pair_decisions','one_or_both_zero_volume','volume_lot_minimum_and_hedge_pass','positive_buffered_edge_before_global_risk']]] for r in flow['rows']])
    body+='''<p>기회 수는 늘었지만 실제 캔들 계좌 재생에서 확대 후보의 완료 페어는 모든 par 구간·비용 조건에서 0개였다.
확대 후보의 이후 par 순손익은 기본 −88.23, stress −58.23이다. 이 수치는 대회 지갑 실거래 수익이 아니다.</p>
<h2>과거 실제 호가에서 추가로 가능한 진입</h2>
<p>이미 관찰했던 다섯 개의 짧은 공개 호가 기록에 수정된 가격 한도·호가 수량 소비·독립 체결 실행기를 적용했다.
두 페어 후보는 진입 0건, 확대 후보는 네 조건 사례에서 양쪽 진입을 확인했다.
기본·stress가 겹치는 두 source/tape/pair 그룹이며 네 개의 독립적인 수익 기회가 아니다.
예측 만기 이후의 실제 청산 호가가 없어 이 사례들의 수익률·Q는 계산하지 않았다.</p>'''
    rows=[]
    for r in value['rows']:
        rows.append(['/'.join(r['pair']),r['scenario'],*[f'{r[k]:+.5f}' for k in ['expected_net_cash','without_model_increment','without_venue_basis_or_source_aging','known_entry_fees']]])
    body+=table(['페어','조건','주문 전 예상 순현금','모델 증분을 제거한 예상값','가격 괴리·신호 경과 효과 제거','관측된 진입 과정 수수료'],rows)
    body+='''<p>위 예상값은 원래 주문 계획 수량 기준이며 부분 체결·보정 뒤의 실현 손익이 아니다.
SP500/EWY의 두 사례는 모델 증가분만으로 비용을 넘지 못하고 기준시장과 현재 호가의 차이에도 의존한다.
EWY/EWJ는 이 효과를 제거해도 예상값이 양수지만 규모가 작고 실제 청산 결과는 없다.</p>
<h2>예측 정확도 진단</h2>
<p>각 페어의 정규화된 24시간 수익률을 하루 간격으로 비교했다. 학습 표본은 겹치지만 아래 진단 표본의 목표 기간은 겹치지 않는다.
정답 0 예측과 비교한 RMSE를 포함했다. 미래 구간 결과로 페어를 재선정하지 않았다.</p>'''
    body+=table(['기간','페어','표본','모델 RMSE','0 예측 RMSE','방향 일치율','평균 절대 예측 bp'],[
        [r['phase'],'/'.join(r['pair']),r['samples'],f'{r["normalized_rmse"]:.5f}',f'{r["zero_rmse"]:.5f}',f'{100*r["sign_accuracy"]:.2f}%',f'{r["mean_abs_return_bps"]:.3f}'] for r in forecast['rows']])
    body+='''<h2>논문과 이번 구현의 관계</h2>
<p><a href="https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2026.1749337/full">2026년 페어 예측 연구</a>는
페어 선택과 예측을 다루지만 거래 비용·유동성·실행 마찰을 명시적으로 모델링하지 않았다고 밝힌다.
이번 실험은 그 DNN/LSTM 구현의 재현이 아니다. 기존에 조사한 관련 자산 범위에 동일한 자체 회귀 모델을 적용하고 비용 후 계좌 성과로 검증한 확장 실험이다.</p>
<h2>판단과 다음 검토</h2>
<p>6페어 확대 후보를 채택하지 않는다. 유지한 2페어 후보와 현재 실거래 전략은 그대로다.
단순히 페어를 더 넣는 것보다 가격 예측의 정보 부족을 먼저 검토한다. 다음 변경 전에 기존 펀딩 활용 모델을 대조하고,
현재 가격 전용 일별 모델에 이미 확정된 과거 펀딩 정보를 추가하는 단일 변화가 유효한지 고정된 조건으로 시험한다.
펀딩을 가격 예측의 입력으로 사용하는 것과 실제 펀딩 현금을 성과에 더하는 것은 구분하고, 미래 정산 자료를 입력으로 사용하지 않는다.</p>
<p><a href="account_results.csv">전체 64개 계좌 비교 CSV</a></p>'''
    (destination/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>6페어 확대 검증</title><style>body{font:15px/1.6 system-ui;max-width:1200px;margin:32px auto;color:#1a2839}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}th,td{padding:7px;border:1px solid #d9e0e8;text-align:right}th{background:#edf2f7}th:first-child,td:first-child{text-align:left}</style>'+body+'</html>')
    save(HERE/'work_state.json',dict(status='progress_16_replays_universe_extension_rejected_and_cost_failure_localized',
        previous_goal_turn='progress_published_book_lifecycle_and_prospective_registration',new_account_replays=16,
        exact_control_replays=4,tests_passed=count,selection=data['selection'],actual_book_cases=4,actual_book_source_tape_pair_groups=2,
        completed_actual_book_exits=0,orders_sent=0,live_strategy_changed=False,protected=guard,
        frozen_reserved_sources_unchanged=1045,historical_independence=False,
        report=str((destination/'report.html').relative_to(ROOT)),
        next='Continue from retained two-pair daily direct, not the rejected sector expansion. Audit older funding-feature models before implementing a single funding-feature ablation of the daily per-pair price model; use only settled reference funding strictly before source and keep par funding as separately charged actual cash. Do not use later returns to filter ETH/HYPE or choose SP500/EWJ. Preserve new published_pair_lifecycle 1045 frozen sources and original sealed raw release 2026-10-24T12:01Z. No live replacement.',
        findings=['Base crypto candle opportunity funnel 2978 ->87 sized ->3 above costs; stress 44 sized and no above-cost opportunity. Not a book-liquidity estimate.',
            'Six-pair reference later first base +419.96, stress -1849.65; worse than retained two-pair +3422.16/+2104.98.',
            'Added ETH/HYPE later base price PnL -4113.16 and total pair PnL -5287.70; not explained by fees alone.',
            'Four old-book entry cases but only two overlapping source/tape/pair groups, no horizon exits. SP500/EWY relies partly on basis/aging assumptions.'],
        goal_complete=False))
    paths=[HERE/'diagnostics.py',HERE/'book_probe.py',HERE/'book_attribution.py',HERE/'report.py',
           *[ART/name for name in ['attribution.json','entry_funnel.json','forecast_diagnostics.json','book_value_attribution.json','books/results.json']]]
    save(ART/'diagnostics_manifest.json',dict(sources={str(p.relative_to(ROOT)):sha(p) for p in paths},orders_sent=0))
    print('REPORT',destination/'report.html','new_replays',16,'tests',count,'eligible',data['selection']['eligible'])


if __name__=='__main__':main()
