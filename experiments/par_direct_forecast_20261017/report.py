"""Report every failed candidate and keep execution assumptions distinguishable."""
import csv,html,json,gzip
import numpy as np
from .data import HERE,OUT,ROOT,save,sha,ts
from .model import design,labels
from strategy.agent import TAIL

def main():
    direct=json.loads((OUT/'results.json').read_text());basis=json.loads((OUT/'basis/results.json').read_text())
    dest=ROOT/'reports/par_direct_forecast_20261017';dest.mkdir(exist_ok=True)
    cfg=json.loads((HERE/'protocol.json').read_text());p=json.loads((TAIL/'baseline.json').read_text())['protocol']
    with np.load(OUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
    with np.load(OUT/'predictions.npz') as z:pr={k:z[k].copy() for k in z.files}
    de=design(d,cfg,p['symbols']);diagnostics=[]
    for name in [k for k in pr if '_' in k]:
        h=int(name.split('_')[-1]);y,mature=labels(d,de,h);f=pr[name]
        use=((d['times']>=ts(cfg['comparison_start']))&(mature<=ts(cfg['end']))&(d['times']%900==0))[:,None]
        use=use&de['fresh']&np.isfinite(y)&np.isfinite(f)&(de['beta']>=.1)&(de['beta']<=10)&(de['correlation']>=.3)
        strong=use&(abs(f)*10000>43);n=int(strong.sum())
        diagnostics.append(dict(model=name,eligible_decisions=int(use.sum()),strong_decisions=n,
            mean_directional_gross_bps=float(np.mean(np.sign(f[use])*y[use])) if use.any() else None,
            strong_signal_mean_price_return_minus_38bp=float(np.mean(np.sign(f[strong])*y[strong]-38)) if n else None,
            excludes_funding_and_size_constraints=True,overlapping_labels_not_independent=True))
    quotes=[];base=ROOT/'experiments/pairs_research_20261017/par_history/20261016T173824Z'
    for symbol in ('BTC','ETH','HYPE','SP500','EWY','EWJ'):
        path=base/f'l2Book_{symbol}.json';o=json.loads(path.read_text())['response']
        bid,ask=float(o['levels'][0][0]['px']),float(o['levels'][1][0]['px']);mid=(bid+ask)/2
        quotes.append(dict(symbol=symbol,time=o['time'],spread_bps=(ask-bid)/mid*10000,
            bid_top_notional=bid*float(o['levels'][0][0]['sz']),ask_top_notional=ask*float(o['levels'][1][0]['sz']),
            source_sha256=sha(path),post_comparison_snapshot_not_historical_execution=True))
    save(dest/'diagnostics.json',dict(forecast=diagnostics,snapshot_quotes=quotes))
    fields=['family','phase','arm','scenario','net_pnl','roi','daily_sample_std','volatility_horizon','financial_score','score','fees','funding','price_move_pnl','execution_cost','mean_gross','fills','completed_pairs','aborted_pairs','hedge_repairs','residual_gross','pnl_after_liquidation_reserve']
    with (dest/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for family,r in [('direct',direct),('basis_and_entry_limit',basis)]:
            for row in r['rows']:
                w.writerow({k:family if k=='family' else row[k] if k in ('phase','arm','scenario') else row['metrics'][k] for k in fields})
    body='''<!doctype html><meta charset="utf-8"><title>실제 par 페어 직접 예측과 괴리 분해</title><style>body{font:16px/1.6 system-ui;color:#182538;margin:30px}p{max-width:1100px}table{border-collapse:collapse;font-size:13px;margin:18px 0}td,th{border:1px solid #cbd2dd;padding:8px;white-space:nowrap;text-align:right}td:first-child{text-align:left}th{background:#edf1f5}.note{background:#fff1ca;padding:14px}a{color:#1461a4}</style>
<h1>실제 par 상대가격 예측: 역사적 비교 결과</h1>
<p><b>가격 예측 목표와 체결 처리를 수정해 비교했으나, 이번 두 계열 모두 기존 전략보다 나은 교체 후보 선정 요건을 통과하지 못했습니다.</b> 운용 전략 자동 교체와 실거래 주문은 없었습니다.</p>
<p>10월 1~16일 대회 10종목과 기준시장 5분봉을 확보했습니다. 10월 8일 21시~12일 21시 한국시간은 후보 선정, 12일 21시~16일 21시는 이후 비교입니다. 매 구간 원금 1,000,000 USDPR 현금 계좌에서 시작했습니다. 이미 반복 사용한 과거 구간이므로 독립적인 새 검증이라고 부르지 않습니다.</p>
<h2>바꾼 내용</h2><ul><li>기준시장 수익률 대신 실제 par 두 종목의 6·24시간 상대수익률을 학습합니다. 목표는 다음 5분봉 체결 시점부터 측정합니다. 릿지, 작은 XGBoost, 동일 비중 혼합을 비교했습니다.</li>
<li>10월 8·10·12·14일 12:00 UTC마다 이전 7일의 이미 확정된 예측 결과만 사용합니다. 표준화·클리핑 경계도 학습 자료에서 계산합니다. 6시간 모델 학습 행 수는 150→166→204→472개, 24시간은 135→128→156→342개이며, 겹치는 시계열 자료입니다.</li>
<li>5분씩 늦게 양쪽 주문을 각각 체결하고, 과다 체결된 쪽을 줄여 비율을 복원합니다. 10분 안에 복원되지 않거나 한쪽만 체결되면 청산합니다. 기존 전략의 주문도 같은 실행기로 15분까지 재시도합니다.</li>
<li>후속 계열은 par/기준시장 가격 괴리의 AR(1)만 학습하고, 주문 도착 전에 이미 사라진 괴리를 수익으로 계산하지 않습니다. 동시에 신규 진입 가격을 제출 시점에서 10bp 이상 불리하게 옮겨 체결하지 못하도록 제한했습니다. 기준 전략에도 같은 제한을 적용했습니다.</li></ul>
<p class="note">두 계열의 실행 가정이 다르므로 아래 표는 각 계열 안에서 비교해야 합니다. 둘째 계열은 괴리 모델과 가격 제한을 함께 바꾼 후보이며 개별 변경의 효과를 분리 입증한 실험은 아닙니다. 실제 호가 이력 전체가 없어 체결가는 5분봉 기반 추정입니다. 펀딩률은 실제 par 값이고 oracle 가격은 기준시장 가격을 대용했습니다. 현물·선물 가격 괴리를 무위험 차익거래로 간주하지 않습니다.</p>'''
    for family,result in [('직접 가격 예측',direct),('괴리 분해＋진입 가격 제한',basis)]:
        selection=result['selection'];body+=f'<h2>{family}</h2><p>선정 요건 충족: <b>{selection["eligible"]}</b>. 적격 후보가 없어 진단 목적으로 고정한 후보: {html.escape(selection["selected"])}. 이후 구간 결과를 보고 선정 후보를 교체하지 않았습니다.</p>'
        for phase,label in [('validation','앞 4일: 선정'),('comparison','뒤 4일: 비교')]:
            for scenario,scname in [('base','기본 비용'),('stress','불리한 비용·유동성')]:
                body+=f'<h3>{label} — {scname}</h3><table><tr><th>전략</th><th>순손익 USDPR</th><th>수익률 %</th><th>기간 σ %</th><th>Q</th><th>수수료</th><th>펀딩</th><th>완료 페어</th><th>중단 페어</th><th>평균 명목 비중 %</th><th>잔여 명목금액</th></tr>'
                for row in result['rows']:
                    if row['phase']!=phase or row['scenario']!=scenario:continue
                    m=row['metrics'];values=[row['arm'],f"{m['net_pnl']:+.4f}",f"{100*m['roi']:+.6f}",f"{100*m['volatility_horizon']:.6f}",f"{m['financial_score']:.8f}" if m['eligible'] else '거래 없음: 점수 제외',f"{m['fees']:.4f}",f"{m['funding']:+.4f}",m['completed_pairs'],m['aborted_pairs'],f"{100*m['mean_gross']:.6f}",f"{m['residual_gross']:.4f}"]
                    body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
                body+='</table>'
    body+='''<h2>왜 실패했는가</h2><p>직접 예측 계열에서 비용 문턱을 넘는다고 예측한 신호를 별도로 확인했습니다. 뒤 구간의 강한 신호들에 대해 실제 이후 가격 변화에서 왕복 38bp를 차감하면 평균은 모든 모델에서 음수였습니다. 중복 보유 구간을 포함한 진단이며 계좌 수익률이나 독립 표본 검정은 아닙니다. 한쪽 체결 문제만 고치면 되는 상태는 아닙니다.</p>
<p>24시간 tree 모델은 겹치지 않는 3개 구간에서 6페어 모두 0수익률 예측보다 제곱오차가 작았습니다. 그러나 관측이 매우 적고, 비용을 넘는 신호를 고르는 능력·체결 가능성·계좌 순손익 개선으로 이어지지 않았습니다. 예측 오차가 개선됐다는 이유만으로 거래 전략을 채택하지 않았습니다.</p>
<p>첫 괴리 AR 추정의 반감기는 BTC 약 10.2분, ETH 3.0분, HYPE 2.1분, SP500 1.4분, EWY 3.9분입니다. 모형 추정치이며 확정된 물리적 법칙은 아닙니다. 5분 뒤 남는 괴리를 예측하면 비용 문턱을 넘는 진입은 거의 없어졌습니다.</p>
<h2>실행 가정 점검</h2><p>기본 편도 수수료 9bp＋불리한 가격 10bp, 거래량 참여 10%. 스트레스는 12bp＋20bp, 참여 5%, 받는 펀딩 절반·내는 펀딩 전액입니다. 참여율·가격 비용은 가정이며 실측 체결 가능량이 아닙니다. 저장된 10월 17일 02시 38분 무렵 한국시간의 한 호가 스냅샷에서 6종목 스프레드는 약 5.8~8.4bp였습니다. 이 값은 비교 기간 종료 후의 한 시점이므로 과거 전체 실행 비용으로 사용하지 않았습니다.</p>
<p>첫째·둘째가 같은 결과를 낸 것은 이 체결 가정에서 목표 수량보다 체결 가능량이 작은 경우가 지배하기 때문입니다. 같은 전략이라는 뜻이 아닙니다. 자료 주기·체결 가정이 다른 지난 15일 백테스트의 숫자와 직접 섞어 해석하면 안 됩니다.</p>
<h2>검증과 근거</h2><p>공통 검사 6개 및 괴리·진입 제한 추가 검사까지 8개 통과: 다음 봉 체결, 0 거래량 차단과 재시도, 기존 보유에 대한 시간별 펀딩, 부분 체결 축소·청산, 신호 없는 혼합판의 기존판 일치, 독립 손익 재계산, 미래 절단 시 특징·모델·예측 일치, 고정 진입 가격 제한. 원본과 후보 코드는 별도 보관하며 운용·예약 코드 해시는 변경되지 않았습니다.</p>
<p>참고 연구: <a href="https://arxiv.org/abs/2502.04284">신호 소멸과 다기간 거래 비용</a>, <a href="https://arxiv.org/abs/2606.04574">동적 다중 페어와 실행 모형</a>. 이 구현은 해당 아이디어를 적용한 별도 모델입니다. PPO/LSTM 논문 전체나 보고 수익률의 재현을 주장하지 않습니다.</p>
<p><a href="comparison.csv">전체 비교 CSV</a> · <a href="diagnostics.json">가격 예측·실제 호가 스냅샷 진단</a></p>'''
    (dest/'report.html').write_text(body)
    # Verify frozen evaluated source files after reporting; no mutation of results.
    for sub in ('','basis'):
        frozen=json.loads((OUT/sub/'frozen.json').read_text())
        for path,h in frozen['sources'].items():assert sha(ROOT/path)==h,path
    save(HERE/'work_state.json',dict(status='progress',previous_turn='historical actual-par backtest and hedge repair completed',
        this_turn='Collected complete Oct1-16 actual-par/reference 5m history; implemented direct par relative-return models, purged rolling fits, 5m shared executor, basis AR decomposition and fixed entry limit. Ran 64 historical comparisons; no eligible replacement.',
        model_selection_frozen_before_later_replays=True,performance_improvement_established=False,
        next_action='Calibrate executable spread/depth/arrival behavior from unsealed historical L2 before interpreting candle execution costs. Keep existing model losses, do not select cheaper costs solely for positive PnL. Investigate whether a patient maker entry and taker hedge can monetize the limited 24h predictive edge, with adverse selection and actual leg/exit constraints.',
        goal_complete=False,orders_sent=0,wait_for_future_data_required=False))
    print('REPORT',dest/'report.html',flush=True)

if __name__=='__main__':main()
