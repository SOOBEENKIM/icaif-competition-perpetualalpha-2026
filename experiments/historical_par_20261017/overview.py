"""Consolidated, nonselective report of the historical replay and its revisions."""
import csv
import gzip
import html
import json
import numpy as np
from .data import HERE,OUT,ROOT,save

NAMES={'first_strategy':'기존 첫 번째','second_strategy':'기존 두 번째','pairs_carry':'펀딩 페어',
       'pairs_basis_carry':'가격 괴리＋펀딩 페어','pairs_ridge_basis_carry':'릿지＋가격 괴리＋펀딩 페어',
       'second_plus_basis_carry':'두 번째＋가격 괴리·펀딩'}


def main():
    finite=json.loads((OUT/'results.json').read_text())
    continuous=json.loads((OUT/'continuous/results.json').read_text())
    repaired=json.loads((OUT/'repair/results.json').read_text())
    dest=ROOT/'reports/historical_par_20261017'
    # Retain the initial finite-horizon report separately, never hide its results.
    if not (dest/'finite_horizon.html').exists():
        (dest/'finite_horizon.html').write_text((dest/'report.html').read_text())
    combined=[r for r in continuous['rows'] if r['arm'] in ('first_strategy','second_strategy')]+repaired['rows']
    comparison=[]
    for phase in ('full','later'):
        for scenario in ('participation_1pct','participation_10pct','stress_10pct'):
            baseline=next(r['metrics'] for r in combined if r['phase']==phase and r['scenario']==scenario and r['arm']=='second_strategy')
            for row in combined:
                if row['phase']!=phase or row['scenario']!=scenario or row['arm']=='second_strategy':continue
                c=row['metrics']
                comparison.append(dict(phase=phase,scenario=scenario,arm=row['arm'],
                    pnl_delta=c['net_pnl']-baseline['net_pnl'],Q_delta=c['financial_score']-baseline['financial_score'],
                    mean_gross_ratio=c['mean_gross']/baseline['mean_gross'] if baseline['mean_gross'] else None,
                    positive_pnl=c['net_pnl']>0,completed_pairs=c['completed_pairs']))
    # Inspect rather than infer why the incumbents coincide under the tight cap.
    a=json.load(gzip.open(OUT/'runs/full-first_strategy-base.json.gz','rt'))
    b=json.load(gzip.open(OUT/'runs/full-second_strategy-base.json.gz','rt'))
    diagnostic=dict(fills_identical=a['fills']==b['fills'],different_target_decisions=sum(x['target_qty']!=y['target_qty'] for x,y in zip(a['decisions'],b['decisions'])),
        orders=len(a['orders']),partial_or_unfilled=sum(o['status']!='FILLED' for o in a['orders']))
    models=json.loads((OUT/'models.json').read_text())
    # Forecast loss on nonoverlapping 72-hour decision blocks, each leg's actual
    # next-close par movement. Funding is assessed separately in the replay.
    with np.load(OUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
    with np.load(OUT/'signals.npz') as z:s={k:z[k].copy() for k in z.files}
    cfg=json.loads((HERE/'protocol.json').read_text())
    from strategy.agent import TAIL
    symbols=json.loads((TAIL/'baseline.json').read_text())['protocol']['symbols']
    forecast=[]
    for k,pair in enumerate(cfg['pairs']):
        j=[symbols.index(x) for x in pair];actual=[];pred=[]
        for i in range(12,len(d['times'])-73,72):
            be=s['beta'][i,k];pr=s['ridge'][i,k]
            if not np.isfinite(pr) or not .1<=be<=10:continue
            rr=d['close'][i+73,j]/d['close'][i+1,j]-1
            actual.append((rr[0]-be*rr[1])/(1+be));pred.append(pr)
        actual=np.array(actual);pred=np.array(pred)
        if len(actual):forecast.append(dict(pair=pair,time_blocks=len(actual),MSE=float(np.mean((actual-pred)**2)),
            zero_MSE=float(np.mean(actual**2)),direction_accuracy=float(np.mean(np.sign(actual)==np.sign(pred)))))
    save(dest/'summary.json',dict(rows=combined,comparisons=comparison,capacity_diagnostic=diagnostic,
        forecast_diagnostics=forecast,all_models_purged=all(m['last_label_maturity']<m['fit_time'] for m in models),
        total_historical_replays=118,unique_regression_checks=9,live_orders_sent=0,
        automatic_replacement=False,improvement_established=False,historical_independence=False))
    with (dest/'comparison.csv').open('w',newline='') as f:
        cols=['phase','scenario','arm','net_pnl','roi','daily_sample_std','volatility_horizon','financial_score','score','max_drawdown','fees','funding','price_move_pnl','execution_cost','fills','completed_pairs','aborted_pairs','mean_gross','residual_gross','pnl_after_liquidation_reserve']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in combined:
            w.writerow({k:r[k] if k in ('phase','scenario','arm') else r['metrics'][k] for k in cols})
    body='''<!doctype html><meta charset="utf-8"><title>10월 1~16일 실제 par 과거 백테스트</title><style>body{font:16px/1.6 system-ui;color:#14202e;margin:32px}p{max-width:1150px}table{border-collapse:collapse;font-size:13px;margin:16px 0}td,th{border:1px solid #c8d1db;padding:8px;white-space:nowrap;text-align:right}td:first-child{text-align:left}th{background:#eaf0f7}.note{padding:16px;background:#fff1c7}a{color:#185ca8}</style>
<h1>10월 1~16일: 실제 par 과거 자료 백테스트</h1>
<p><b>미래 관측을 기다리지 않고 완료한 역사적 비교입니다. 기존 두 전략보다 안정적으로 더 수익성 있는 교체 후보를 확보했다는 결론은 아직 아닙니다.</b></p>
<p>2026년 10월 1일 21시~16일 21시 한국시간, 15일. 각 계좌는 1,000,000 USDPR 현금에서 시작합니다. 종료일에는 보유 포지션을 평가하며, 미청산 수량과 추정 청산 비용도 별도로 공개합니다. 아래 수정판은 양쪽 부분 체결 시 과다 체결된 쪽을 줄여 비율을 복원합니다. 한쪽만 체결되거나 두 시간 안에 조정되지 않으면 청산합니다.</p>
<p class="note">실제 par 시간봉·거래량·펀딩률을 사용했습니다. 호가 이력 전체가 없으므로 체결가는 다음 시간봉 종가＋불리한 가격을 적용한 추정입니다. 평가가격은 체결 종가이며 공식 mark와 다를 수 있고, 펀딩의 oracle 가격은 기준시장 가격을 대용했습니다. 아래 수치는 실제 계좌 수익률이나 공식 제출 점수가 아닙니다.</p>
<h2>무엇을 확인했는가</h2>
<ul><li>기존 첫째·둘째의 신호가 저장된 경로와 정확히 일치합니다. 1% 거래량 제한에서 결과가 같은 것은 목표 수량 연결 오류가 아니라 체결량 제한 때문입니다. 최초 비교에서 목표가 다른 판단은 56회, 주문 253개 중 미체결·부분 체결은 250개였습니다.</li>
<li>릿지는 10월 1·8·15일 00:00 UTC에 재학습합니다. 8월 1일 이후 행 중 다음 시간봉부터 72시간 뒤의 결과가 학습 시점 전에 확정된 행만 사용하고, 스케일러도 그 행으로만 학습합니다. 미래를 잘라 다시 계산해 이전 예측이 동일한지 검증했습니다.</li>
<li>페어 헤지는 과거 168시간 기준시장 수익률로 계산합니다. 펀딩은 직전 시간까지의 마지막 값과 24시간 평균의 혼합이며, 양수 예상 수입은 절반으로 줄입니다. 괴리 후보는 가격 괴리의 50% 수렴을 가정합니다. 수렴 보장이나 학습된 확률이 아닙니다.</li>
<li>초기 후보 선정은 앞 7일, 이후 평가는 뒤 8일로 나누었으나, 모두 이미 프로젝트에서 사용한 과거입니다. 이어진 종료 처리·부분 체결 수정은 결과를 본 뒤의 개발 실험이므로 독립 검증이라고 부르지 않습니다. 파라미터 선택의 사후성을 숨기지 않습니다.</li></ul>'''
    for phase,title in [('full','전체 15일'),('later','뒤 8일: 별도 현금 계좌로 시작')]:
        for scenario,label in [('participation_1pct','거래량 1% 이내 체결 가정'),('participation_10pct','거래량 10% 이내 체결 가정'),('stress_10pct','10% 체결 가정＋비용·펀딩 스트레스')]:
            rows=[r for r in combined if r['phase']==phase and r['scenario']==scenario]
            body+=f'<h2>{title} — {label}</h2><table><tr><th>전략</th><th>순손익 USDPR</th><th>수익률 %</th><th>일별 표준편차 %</th><th>기간 σ %</th><th>Q</th><th>MDD %</th><th>평균 투자 명목금액 비율 %</th><th>완료 페어</th><th>중단 페어</th><th>잔여 포지션 USDPR</th><th>청산 비용 추정 차감 손익</th></tr>'
            for r in rows:
                m=r['metrics'];values=[NAMES[r['arm']],f"{m['net_pnl']:+.4f}",f"{100*m['roi']:+.6f}",f"{100*m['daily_sample_std']:.6f}",f"{100*m['volatility_horizon']:.6f}",f"{m['financial_score']:.8f}" if m['eligible'] else '점수 제외: 거래 없음',f"{100*m['max_drawdown']:.6f}",f"{100*m['mean_gross']:.6f}",m['completed_pairs'],m['aborted_pairs'],f"{m['residual_gross']:.2f}",f"{m['pnl_after_liquidation_reserve']:+.4f}"]
                body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
            body+='</table>'
    body+='''<h2>해석</h2><p>가격 괴리＋펀딩 페어 수정판은 10% 체결 가정에서 약 +2.08 USDPR이나, 스트레스에서는 약 −1.32 USDPR이고 완료 페어는 한 번입니다. 뒤 8일의 스트레스에서는 거래가 없었습니다. 기존 전략보다 현저히 적은 금액을 거래했으므로 손실·변동성이 작다는 사실만으로 예측력이 우수하다고 판정할 수 없습니다. 혼합판도 기존 둘째 대비 약 +2.03 USDPR 증가에 그쳤고 전체 손실을 뒤집지 못했습니다.</p>
<p>기준시장 가격·거래량을 그대로 쓰는 비교도 같은 기간에 실행했습니다. 첫째 기본 비용 −2,617.25, 둘째 −2,061.39 USDPR였습니다. 따라서 이번 구간 부진을 대회 시장의 거래량 제한이나 수수료만으로 설명할 수는 없습니다.</p>
<h2>회계·위험·실행 검사</h2><p>9개 핵심 검사: 다음 시간 체결, 0 거래량 차단, 기존 보유에만 펀딩 부과, oracle 대용 가격 적용, 한쪽 체결 중단, 청산 유동성 부족 표시, 신호가 없는 혼합판의 기존판 일치, 일별 복리·독립 손익 재계산 및 미래 절단 일치, 부분 체결 축소·시간초과 처리 등을 포함합니다. 상속된 공통 테스트까지 수정판에서 16개 테스트 실행이 통과했습니다. 수정판의 동작과 원본의 차이는 저장된 diff에서 볼 수 있습니다.</p>
<p>기본 편도 수수료 9bp＋불리한 가격 10bp. 스트레스 편도 수수료 12bp＋가격 20bp, 받는 펀딩 50%, 내는 펀딩 전액. 체결량은 직전과 다음 시간의 실제 거래량 중 작은 값의 1% 또는 10%로 제한했습니다. 10%도 실제 체결 가능량을 입증한 값은 아닙니다. 원금·위험 한도는 동일하나 실현된 투자 규모는 다르므로 표에 평균 노출을 병기했습니다.</p>
<p>순손익 = 가격 변화 손익 − 체결 가격 불리함 − 수수료 ＋ 펀딩. 매 시간 독립 계산과 대조했습니다. 12:00 UTC 일별 수익률 표본 표준편차 × √15가 전체 구간 σ이며 Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05). J를 가정하지 않았습니다. 미체결 계좌는 Q가 수학적으로 0.1이어도 순위 대상이 아닙니다.</p>
<p><a href="finite_horizon.html">최초 종료 전 청산형 백테스트 전체 결과</a> · <a href="comparison.csv">최종 비교 CSV</a> · <a href="summary.json">검증·비교 JSON</a> · <a href="https://perpetual-alpha26.github.io/docs/">공식 수수료·펀딩 문서</a> · <a href="https://perpetual-alpha26.github.io/leaderboard#practice">공식 점수 정의</a></p>
<p>실거래 주문 제출 0건. 운용 중인 전략과 예약된 관측 코드는 변경하지 않았습니다. 본대회 이후의 새 관측은 추가 검증 수단이며, 이 과거 백테스트를 대신하는 대기 조건이 아닙니다.</p>'''
    (dest/'report.html').write_text(body)
    save(HERE/'work_state.json',dict(status='historical_backtest_completed_improvement_not_established',
        user_correction_applied=True,historical_window_KST=['2026-10-01 21:00','2026-10-16 21:00'],
        do_not_wait_for_october_20_or_24_to_backtest=True,backtests=118,orders_sent=0,
        next_research='Improve predictive price alpha on chronological historical training/validation, compare with both incumbents under identical execution assumptions; short current quote expectations alone are insufficient.'))
    print('Consolidated report:',dest/'report.html')


if __name__=='__main__':main()
