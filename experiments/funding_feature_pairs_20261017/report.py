"""Persist the failed funding-input ablation and its measured cause."""
import csv,gzip,html,json,re
from pathlib import Path
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha
from experiments.complementary_alpha_20261017.research import protected

HERE=Path(__file__).resolve().parent;ART=HERE/'artifacts'


def table(headers,rows):
    def row(values,tag):
        return '<tr>'+''.join(f'<{tag}>{html.escape(str(x))}</{tag}>' for x in values)+'</tr>'
    return '<table>'+row(headers,'th')+''.join(row(r,'td') for r in rows)+'</table>'


def main():
    result=json.loads((ART/'results.json').read_text())
    attribution=json.loads((ART/'attribution.json').read_text())
    forecast=json.loads((ART/'forecast_diagnostics.json').read_text())
    flow=json.loads((ART/'entry_funnel.json').read_text())
    books=json.loads((ART/'books/results.json').read_text())
    plan=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())
    for manifest in [plan,json.loads((ART/'frozen.json').read_text()),json.loads((ART/'books/frozen.json').read_text())]:
        for path,digest in manifest['sources'].items():assert sha(ROOT/path)==digest,path
    assert not result['selection']['eligible']
    assert result['new_account_replays']==16 and len(result['exact_control_replays'])==4
    assert not books['trials']
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in result['rows']}
    par_controls=[]
    for row in result['rows']:
        if row['venue']=='par' and '_funding_' in row['arm']:
            old=lookup[row['venue'],row['phase'],row['matched'],row['scenario']]
            difference={k:[v,old.get(k)] for k,v in row['metrics'].items() if v!=old.get(k)}
            assert set(difference)<= {'orders'}
            new_path=ART/'runs'/('-'.join(row[k] for k in ['venue','phase','arm','scenario'])+'.json.gz')
            old_path=ROOT/'experiments/rolling_feature_pairs_20261017/artifacts/runs'/('-'.join(
                [row['venue'],row['phase'],row['matched'],row['scenario']])+'.json.gz')
            with gzip.open(new_path,'rt') as f:new_run=json.load(f)
            with gzip.open(old_path,'rt') as f:old_run=json.load(f)
            for k in ['fills','owner_fills','snapshots']:assert new_run[k]==old_run[k],k
            par_controls.append(dict(phase=row['phase'],arm=row['arm'],scenario=row['scenario'],
                metric_differences=difference,fills_owner_fills_snapshots_exact=True))
    save(ART/'par_control_differences.json',dict(rows=par_controls,orders_sent=0))
    values={phase:{sc:lookup['reference',phase,'first_dense_hourlyhedge_funding_24',sc]['net_pnl']
                   for sc in ['base','stress']} for phase in ['validation','comparison']}
    tests=(HERE/'test_results.txt').read_text();assert tests.rstrip().endswith('OK')
    test_count=int(re.search(r'Ran (\d+) tests',tests).group(1));guard=protected()
    dest=ROOT/'reports/funding_feature_pairs_20261017';dest.mkdir(exist_ok=True)
    exported=[]
    for r in result['rows']:
        exported.append({k:r[k] for k in ['venue','phase','arm','scenario']}|{k:r['metrics'].get(k) for k in
            ['net_pnl','return_fraction','financial_score','score','daily_sample_std','volatility_horizon',
             'max_drawdown','completed_pairs','fees','funding','pair_owned_pnl','residual_gross','pnl_after_liquidation_reserve']})
    with (dest/'account_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(exported[0]));writer.writeheader();writer.writerows(exported)
    names={'first_strategy':'기존 첫째','second_strategy':'기존 둘째',
           'first_dense_hourlyhedge_direct_24':'첫째＋가격 전용 페어',
           'first_dense_hourlyhedge_funding_24':'첫째＋펀딩 입력 페어'}
    body='''<h1>페어 모델의 과거 펀딩 입력 추가: 채택하지 않음</h1>
<p><strong>펀딩 입력을 추가한 후보는 일부 조건에서 개선됐지만 사전 기준을 통과하지 못했다.</strong>
16개 새 계좌 백테스트, 4개 기존 계좌 경로의 완전 대조, 6개 테스트를 완료했다.
실거래 주문·전략 교체는 없으며, 예약한 미래 평가의 원자료도 열지 않았다.</p>
<h2>변경 범위</h2>
<p>BTC/ETH와 BTC/HYPE의 24시간 가격 수익률을 예측하는 일별 회귀 모델에,
이미 정산된 최근 24시간·168시간의 두 종목 간 펀딩 차이를 추가했다.
판단 시점 t에서는 [t-h, t)의 관측값만 사용하고, 누락된 정산은 0으로 채우지 않는다.
모델 목표는 계속 가격 수익률이다. 매매에서 발생하는 펀딩 현금은 기존 실행기에서 별도로 한 번만 반영한다.</p>
<p>기존에 시험한 pooled 모델에도 펀딩 입력이 있었다. 이번 변경은 최근 유지한 일별·페어별 가격 모델에
펀딩 입력을 추가하는 대조 실험이다. 기존 가격 입력, 학습 창, 24시간 목표, 위험·규모·매매·비용 규칙은 고정했다.</p>
<h2>동일 자본 기준시장 비교</h2>
<p>초기 자본 1,000,000. validation: 8월 1일 12시~9월 1일 12시 UTC.
comparison: 9월 1일 12시~10월 15일 12시 UTC. 이미 사용한 과거 자료의 개발 비교다.
기본 편도 수수료 9bp·가격 불이익 10bp; stress는 12bp·20bp, 더 적은 체결 수량과 펀딩 수령액 절반을 사용한다.
잔존 포지션은 마지막 가격으로 평가하므로 아래 순손익 전체가 청산을 마친 실현손익은 아니다.
잔존 명목금액과 청산 비용 준비금 차감값은 CSV에 별도로 포함했다.</p>'''
    rows=[]
    for phase in ['validation','comparison']:
        for sc in ['base','stress']:
            for arm,label in names.items():
                r=next(x for x in result['rows'] if (x['venue'],x['phase'],x['scenario'],x['arm'])==('reference',phase,sc,arm));m=r['metrics']
                rows.append([phase,sc,label,f'{m["net_pnl"]:+,.2f}',f'{m["return_fraction"]*100:+.5f}%',
                    f'{m["financial_score"]:.8f}',f'{m["daily_sample_std"]*100:.5f}%',
                    f'{m["volatility_horizon"]*100:.5f}%',f'{m["max_drawdown"]*100:.5f}%',m['completed_pairs']])
    body+=table(['기간','조건','전략','순손익','수익률','Q','일별 수익률 표본 표준편차','기간 σ','최대낙폭','완료 페어'],rows)
    body+='''<p>Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05). σ는 12시 UTC에 구분한 완료 일별 수익률의 표본 표준편차×√n이다.
심사 점수를 모르는 비교에서는 0.8Q를 사용한다. 이는 대회 공식 지갑 순위가 아니다.
8월 두 비용 조건에서 기존 두 전략과 유지한 가격 모델을 모두 넘는다는 선정 기준을 충족하지 못했다.
이후 stress의 개선 한 칸만 선택하여 채택하지 않았다.</p>
<h2>손익이 악화한 원인</h2>
<p>8월 기본 조건에서 페어 소유분 가격 손익은 BTC/HYPE −7,494.28, BTC/ETH −5,067.33이었다.
가격 불이익과 수수료를 빼기 전부터 손실이므로 비용만의 문제로 설명할 수 없다.
펀딩 수익도 이를 상쇄하지 못했다. 아래는 후보 선정 뒤 작성한 원인 분해다.</p>'''
    rows=[]
    for r in attribution['rows']:
        if not r['arm'].startswith('first'):continue
        for pair,v in r['by_pair'].items():
            rows.append([r['phase'],r['scenario'],pair,*[f'{v[k]:+,.2f}' for k in
                ['price_move','execution_cost','fees','funding','net_pnl']]])
    body+=table(['기간','조건','페어','가격 손익','가격 불이익','수수료','펀딩','페어 순손익'],rows)
    body+='''<h2>예측력 대조</h2>
<p>하루 간격의 겹치지 않는 24시간 목표를 동일 표본에서 비교했다.
학습에 사용한 6시간 간격 목표는 서로 겹친다. 표의 RMSE는 각 시점의 페어 변동성으로 정규화한 목표 기준이다.
펀딩 입력의 예측 오차 개선은 일관되지 않았고, 개선된 오차가 계좌 수익 개선을 보장하지도 않았다.</p>'''
    body+=table(['기간','페어','표본','가격 모델 RMSE','펀딩 모델 RMSE','0 예측 RMSE','평균 예측 변화 bp'],[
        [r['phase'],'/'.join(r['pair']),r['samples'],f'{r["price_only"]["normalized_rmse"]:.5f}',
         f'{r["funding"]["normalized_rmse"]:.5f}',f'{r["zero_rmse"]:.5f}',f'{r["forecast_change_bps"]:.3f}'] for r in forecast['rows']])
    body+='''<h2>실제 par 자료에서는</h2>
<p>10월 1~16일 par 캔들 계좌 재생에서 두 모델의 체결·보유·손익 경로는 모든 비교 구간·비용 조건에서 동일했다.
다만 이후 기본 조건에서 펀딩 후보의 모의 주문 수는 각 367개에서 369개로 늘었다. 추가한 양쪽 주문은 미체결이었다.
이후 구간 순손익은 기본 −87.89, stress −57.84이고 완료 페어는 0개다.
다섯 개의 기존 공개 호가 기록에서도 양 모델 모두 새로운 진입 사례가 0개였다.
대회 지갑에서 실거래한 결과나 실제 호가로 24시간 보유·청산을 완료한 결과는 아니다.</p>'''
    body+=table(['모델','조건','유효 페어 판단','수량·최소주문·헤지 통과','비용＋5bp 초과'],[
        [r['model'],r['scenario'],*[r['counts'].get(k,0) for k in
         ['valid_pair_decisions','volume_lot_minimum_and_hedge_pass','positive_buffered_edge_before_global_risk']]] for r in flow['rows']])
    body+='''<p>이 진입 가능성 표는 실제 계좌 보유·위험 한도·지연 체결·청산 전의 캔들 진단이다.
거래량 0을 주문장 유동성 부재로 해석하지 않는다. 예상 종가와 실제 par 청산 가격의 연결은 여전히 검증 과제다.</p>
<h2>논문·검증 범위</h2>
<p><a href="https://arxiv.org/abs/2512.22476">AutoQuant의 2026년 개정판 초록</a>은 실행 시점·펀딩 가시성·비용과 평가 기간 재사용의 영향을 구분한다.
이번 실험은 그 방법의 재현이 아니며, 이 논문을 펀딩의 가격 예측력이나 수익 보장의 근거로 사용하지 않았다.
펀딩 입력의 효용은 별도의 가설로 시험했고 이번 결과로 채택하지 않는다.</p>
<p>기존 2페어 가격 모델과 실제 운용 전략을 유지한다. 다음 변경 전에는 이미 시험한 예측 기간·청산 정책을 대조하고,
가격 예측 실패와 갱신·보유 결정의 영향을 분리해 새로운 가설을 정한다.
재사용 기간에 수익이 좋아지는 후보만 반복 선택하여 독립적인 개선으로 주장하지 않는다.</p>
<p><a href="account_results.csv">전체 64개 계좌 비교 CSV</a></p>'''
    (dest/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>펀딩 입력 페어 비교</title><style>body{font:15px/1.6 system-ui;max-width:1250px;margin:32px auto;color:#182c3d}table{border-collapse:collapse;width:100%;font-size:12px;margin:18px 0}th,td{border:1px solid #d6dee6;padding:7px;text-align:right}th{background:#eef3f8}td:first-child,th:first-child{text-align:left}</style>'+body+'</html>')
    save(HERE/'work_state.json',dict(status='progress_funding_feature_ablation_rejected_16_replays',
        previous_goal_turn='progress_sector_pairs_rejected_16_replays',new_account_replays=16,
        exact_control_replays=4,tests_passed=test_count,selection=result['selection'],actual_book_entries=0,
        completed_actual_book_exits=0,orders_sent=0,live_strategy_changed=False,
        reserved_raw_read=False,frozen_reserved_sources_unchanged=len(plan['sources']),protected=guard,
        historical_independence=False,report=str((dest/'report.html').relative_to(ROOT)),
        first_funding_reference_pnl=values,
        findings=['Funding input original daily model ablation, not first funding use. Price-only label, actual funding cash separate.',
            'First-based validation PnL in both scenarios is worse than retained price-only.',
            'Later basic PnL worse and stress PnL better than retained price-only: no consistent improvement.',
            'All par fills, owner fills and account snapshots exactly matched. Later base simulated orders +2 per baseline, unfilled; zero actual old-book entries.',
            'Validation pair price loss -12561.61223 before execution fees; not a funding cost-only issue.'],
        next='Retain two-pair daily direct and frozen forward plan. Audit already-tested horizon/holding variants before defining a distinct hypothesis; isolate low forecast skill versus stale renewal/holding value. Do not select funding candidate from its single later stress improvement. No live replacement or reserved raw access before original release.',
        goal_complete=False))
    paths=[HERE/'diagnostics.py',HERE/'report.py',ROOT/'experiments/sector_pairs_20261017/diagnostics.py',
           ROOT/'experiments/risk_neutral_pairs_20261017/attribution.py',
           *[ART/name for name in ['attribution.json','entry_funnel.json','forecast_diagnostics.json','books/results.json','par_control_differences.json']]]
    save(ART/'diagnostics_manifest.json',dict(sources={str(p.relative_to(ROOT)):sha(p) for p in paths},orders_sent=0))
    print('REPORT',dest/'report.html','new_replays',16,'tests',test_count,'frozen_sources',len(plan['sources']))


if __name__=='__main__':main()
