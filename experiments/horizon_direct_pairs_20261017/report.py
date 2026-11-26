"""Matched-horizon experiment: all costs, failed selection and quote limits."""
import csv,html,json,re
from pathlib import Path
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha
from experiments.complementary_alpha_20261017.research import protected

HERE=Path(__file__).resolve().parent;ART=HERE/'artifacts'


def table(headers,rows):
    def row(values,tag):return '<tr>'+''.join(f'<{tag}>{html.escape(str(x))}</{tag}>' for x in values)+'</tr>'
    return '<table>'+row(headers,'th')+''.join(row(r,'td') for r in rows)+'</table>'


def main():
    data=json.loads((ART/'results.json').read_text());a=json.loads((ART/'attribution.json').read_text())
    forecast=json.loads((ART/'forecast_diagnostics.json').read_text())
    books=json.loads((ART/'books/results.json').read_text());bv=json.loads((ART/'book_value_attribution.json').read_text())
    plan=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())
    for manifest in [plan,json.loads((ART/'frozen.json').read_text()),json.loads((ART/'books/frozen.json').read_text())]:
        for path,digest in manifest['sources'].items():assert sha(ROOT/path)==digest,path
    assert not data['selection']['eligible'];assert len(data['rows'])==80
    tests=(HERE/'test_results.txt').read_text();assert tests.rstrip().endswith('OK')
    count=int(re.search(r'Ran (\d+) tests',tests).group(1));guard=protected()
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in data['rows']}
    def metrics(venue,phase,h,scenario):return lookup[venue,phase,f'first_dense_hourlyhedge_direct_{h}',scenario]
    dest=ROOT/'reports/horizon_direct_pairs_20261017';dest.mkdir(exist_ok=True)
    export=[]
    for r in data['rows']:
        export.append({k:r[k] for k in ['venue','phase','arm','scenario']}|{k:r['metrics'].get(k) for k in
            ['net_pnl','return_fraction','financial_score','score','daily_sample_std','volatility_horizon','max_drawdown',
             'completed_pairs','pair_owned_pnl','fees','funding','mean_gross','renewals','residual_gross','pnl_after_liquidation_reserve']})
    with (dest/'account_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(export[0]));writer.writeheader();writer.writerows(export)
    body='''<h1>일별 페어 예측의 6·24·72시간 비교</h1>
<p><strong>6시간과 72시간 후보 모두 채택하지 않는다.</strong> 새 계좌 백테스트 32개, 기존 24시간 계좌 경로 대조 4개,
목표·청산·인과성 테스트 5개를 완료했다. 24시간 후보를 유지하며 실거래 전략을 교체하지 않았다.</p>
<h2>이전 실험과 다른 점</h2>
<p>기존에는 고정된 모델의 24·72시간 예측, 7일 par 자료로 학습한 6·24시간 모델, 갱신 제거 등을 시험했다.
이번에는 최근 유지한 <b>일별·페어별 직접 회귀</b>의 목표 기간을 6시간 또는 72시간으로 바꿨다.
BTC/ETH·BTC/HYPE 두 페어, 가격 입력, 90일 학습 창, 30일 반감기, 6시간 학습 표본, 규제 계수와 계좌 위험 한도는 같다.</p>
<p>각 시점의 목표는 source+h의 가격이다. 펀딩 예상 기간과 청산 시점을 h에 맞췄다.
기존 3σ×√h 손절 공식도 유지하므로 절대 손절 폭과 실제 보유 위험은 기간마다 다르다.
6시간 갱신 주기와 168시간 최대 보유는 그대로다. 6시간 후보는 원래 만기 전에 갱신 기회가 없을 수 있으며,
이를 유리하게 바꾸지 않았다. 실제 기준시장 체결 지연 1시간, par 캔들 지연 5분의 제약도 유지했다.</p>
<h2>동일 자본·비용의 기준시장 백테스트</h2>
<p>초기 자본 1,000,000. validation: 8월 1일 12시~9월 1일 12시 UTC.
comparison: 9월 1일 12시~10월 15일 12시 UTC. 모두 재사용한 개발 기간이다.
기본 편도 수수료 9bp·가격 불이익 10bp, stress는 12bp·20bp와 더 적은 체결 수량 및 펀딩 수령액 절반을 적용했다.
순손익에는 마지막 가격으로 평가한 보유분이 포함된다. 잔존 명목금액과 청산 준비금 차감값도 CSV에 남겼다.</p>'''
    rows=[]
    for phase in ['validation','comparison']:
        for sc in ['base','stress']:
            for arm,label in [('first_strategy','기존 첫째'),('second_strategy','기존 둘째'),
                *[(f'first_dense_hourlyhedge_direct_{h}',f'첫째＋{h}시간 페어') for h in (6,24,72)]]:
                m=lookup['reference',phase,arm,sc]
                rows.append([phase,sc,label,f'{m["net_pnl"]:+,.2f}',f'{m["return_fraction"]*100:+.5f}%',
                    f'{m["financial_score"]:.8f}',f'{m["volatility_horizon"]*100:.5f}%',
                    f'{m["max_drawdown"]*100:.5f}%',m['completed_pairs']])
    body+=table(['기간','조건','전략','순손익','수익률','Q','기간 σ','최대낙폭','완료 페어'],rows)
    body+='''<p>Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05), σ는 완료한 12시 UTC 일별 수익률의 표본 표준편차×√n이다.
8월의 기본·stress 양쪽에서 기존 첫째·둘째 및 유지한 24시간 후보를 넘는 변경은 없었다.
선정 파일의 6시간 후보는 실패한 후보 중의 진단용 선택이며 채택을 뜻하지 않는다.</p>
<h2>원인: 짧으면 예상 이익이 작고, 길면 가격 오차·비용이 커짐</h2>
<p>6시간 후보는 이후 기준시장 구간에서 페어 주문 의도 자체가 0건이었다. 계좌 결과는 기존 방향 전략과 동일했다.
하루의 일부만 거래하게 만드는 임의 제한이 아니라, 현재 가격 예측과 동일한 비용 조건에서 진입 제안을 만들지 못한 결과다.
72시간 후보는 더 많은 거래를 만들었지만 8월에는 비용 차감 전 가격 손익부터 손실이었고,
이후에는 양수인 가격 손익을 왕복 비용이 소진하는 사례도 있었다.</p>'''
    rows=[]
    for r in a['rows']:
        if r['horizon'] not in (6,72):continue
        for pair,v in r['by_pair'].items():
            rows.append([r['horizon'],r['phase'],r['scenario'],pair,*[f'{v[k]:+,.2f}' for k in
                ['price_move','execution_cost','fees','funding','net_pnl']]])
    body+=table(['시간','기간','조건','페어','가격 손익','가격 불이익','수수료','펀딩','페어 순손익'],rows)
    body+='''<h2>기간별 예측력</h2>
<p>각 기간 안에서는 목표가 겹치지 않도록 표본을 뽑았다. 기간별 표본과 정규화 단위가 다르므로
6시간 RMSE가 24시간 RMSE보다 작다는 사실만으로 더 좋은 모델이라고 해석하지 않는다.
동일 기간의 0 예측과 비교해야 한다. 72시간 표본은 특히 적다.</p>'''
    body+=table(['시간','기간','페어','표본','모델 RMSE','0 예측 RMSE','방향 일치율','평균 절대 예측 bp'],[
        [r['horizon'],r['phase'],'/'.join(r['pair']),r['samples'],f'{r["normalized_rmse"]:.5f}',
         f'{r["zero_rmse"]:.5f}',f'{100*r["sign_accuracy"]:.2f}%',f'{r["mean_abs_forecast_bp"]:.3f}'] for r in forecast['rows']])
    body+='''<h2>실제 par 캔들·호가에서의 결과</h2>
<p>par 캔들 계좌 재생에서 모든 새 후보의 완료 페어는 0개였다. 아래는 첫째 전략과 결합한 이후 구간(10월 8일 12시~16일 12시 UTC)이다.
주문 의도만 있고 체결되지 않거나 한쪽 체결 후 철회한 경로도 비용에 반영한다. 대회 지갑의 실거래 성과는 아니다.</p>'''
    body+=table(['시간','조건','순손익','Q','완료 페어'],[[h,sc,f'{metrics("par","comparison",h,sc)["net_pnl"]:+.5f}',
        f'{metrics("par","comparison",h,sc)["financial_score"]:.8f}',metrics('par','comparison',h,sc)['completed_pairs']]
        for h in (6,24,72) for sc in ('base','stress')])
    body+='''<p>다섯 개의 짧은 과거 실제 호가 기록에서 6·24시간 후보의 진입은 0건, 72시간 후보는 3개 조건 사례에서 양쪽 진입이 가능했다.
기본·stress가 겹치는 2개 source/tape/pair 그룹이며 세 개의 독립 수익 사례가 아니다.
그 뒤 72시간 실제 청산 호가가 없어 수익률이나 Q를 계산하지 않았다.</p>'''
    body+=table(['페어','조건','주문 전 예상 순현금','모델 증분 제거','현재 중간가격 기준으로 앵커 변경','관측 진입 수수료'],[
        ['/'.join(r['pair']),r['scenario'],*[f'{r[k]:+.5f}' for k in
         ['expected_net_cash','without_model_increment','without_venue_basis_or_source_aging','known_entry_fees']]] for r in bv['rows']])
    body+='''<p>위 값은 주문 전 계획 수량의 가정별 예상값이다. 실제 체결 후 청산 수익으로 읽으면 안 된다.
기준시장 가격을 par의 미래 청산 가격으로 연결하는 가정은 이 실험으로 입증되지 않았다.</p>
<h2>다음 작업</h2>
<p>기간 조정만으로 개선되지 않았다. 다음에는 기존 일별 24시간 모델의 가격 예측 자체에 비선형 관계를 추가할 가치가 있는지,
이미 시험한 고정 트리 모델과 대조한 후 판단한다. 같은 입력·목표·매매 규칙에서 일별로 학습한 얕은 트리와 회귀의 차이를 검증할 수 있다.
검증하지 않은 트리나 앙상블이 더 좋다고 가정하지 않는다. 현재 실거래와 예약 평가의 고정 파일은 유지한다.</p>
<p><a href="https://arxiv.org/abs/2502.04284">신호 지속성과 거래 비용 연구</a>는 예측과 보유 판단을 연결하는 동기다.
이번 구현은 해당 논문의 최적 제어 해법을 재현하지 않았으며 수익 보장의 근거로 인용하지 않는다.</p>
<p><a href="account_results.csv">전체 80개 계좌 비교 CSV</a></p>'''
    (dest/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>6·24·72시간 페어 비교</title><style>body{font:15px/1.6 system-ui;max-width:1280px;margin:32px auto;color:#183048}table{border-collapse:collapse;width:100%;font-size:12px;margin:18px 0}th,td{padding:7px;border:1px solid #d7e0e8;text-align:right}th{background:#eef3f7}th:first-child,td:first-child{text-align:left}</style>'+body+'</html>')
    save(HERE/'work_state.json',dict(status='progress_32_matched_horizon_replays_rejected',
        previous_goal_turn='progress_funding_features_rejected_16_replays',new_account_replays=32,
        exact_control_replays=4,tests_passed=count,selection=data['selection'],
        actual_book_scenario_cases=len(books['trials']),actual_book_groups=bv['source_tape_pair_groups'],
        completed_actual_book_exits=0,orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,
        protected=guard,frozen_reserved_sources_unchanged=len(plan['sources']),
        reference_first_pnl={str(h):{phase:{sc:metrics('reference',phase,h,sc)['net_pnl'] for sc in ['base','stress']}
            for phase in ['validation','comparison']} for h in (6,24,72)},
        historical_independence=False,report=str((dest/'report.html').relative_to(ROOT)),
        capture_process_verified_this_turn=dict(pid=3727561,state='S',expected_module_matched=True,health_at_unix=1792238739.5023656),
        findings=['Retained daily-direct 24h is distinct from prior frozen/pooled 6/24/72h experiments; new matched horizons did not improve.',
            '6h later reference pair intents zero; mean forecasts about 4bp BTC/ETH and 7bp BTC/HYPE.',
            '72h forecasts create more entries but fail after-cost historical PnL/Q criteria; actual old books only tiny open entry cases without 72h exit data.',
            'No frozen prospective dependency or existing live strategy changed.'],
        next='Retain daily-direct 24h. Audit prior fixed-tree/ensemble models, then consider a single daily per-pair nonlinear learner and fixed equal blend at the same 24h horizon/price inputs. Freeze all parameters before results, compare unchanged account/costs and original/native controls, and do not promote from later cherry-picked outcomes. Do not read reserved forward raw before 2026-10-24T12:01Z or edit frozen 1045 dependencies.',
        goal_complete=False))
    paths=[HERE/'diagnostics.py',HERE/'book_attribution.py',HERE/'report.py',
        *[ART/n for n in ['attribution.json','forecast_diagnostics.json','entry_forecast_diagnostics.json','books/results.json','book_value_attribution.json']]]
    save(ART/'diagnostics_manifest.json',dict(sources={str(p.relative_to(ROOT)):sha(p) for p in paths},orders_sent=0))
    print('REPORT',dest/'report.html','new_replays',32,'tests',count,'frozen_sources',len(plan['sources']))


if __name__=='__main__':main()
