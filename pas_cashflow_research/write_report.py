"""Render preserved experiments; no data fetching, fitting or live mutation."""
import gzip,html,json,math
from datetime import datetime,timezone,timedelta
from pathlib import Path
from collections import defaultdict
import numpy as np
from .data import ROOT,OUT,read,save,sha

LABEL={'incumbent':'현행 전략 비교군 · 의도15분','incumbent_ttl1h':'실행 수정 비교군 · 의도1시간','reversion12_quote3h':'새 평균회귀 후보 · 가격조건3시간'}

def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def pct(x):return f'{100*x:.6f}%'
def num(x):return '—' if x is None else f'{x:,.2f}'
def runfile(arm,scenario):
    with gzip.open(OUT/'sealed_extension/evaluation/runs'/f'{arm}_extended97h_{scenario}.json.gz','rt') as f:return json.load(f)

def main():
    rows=read(OUT/'sealed_extension/evaluation/results.json');by={(r['arm'],r['scenario']):r for r in rows}
    liq=read(OUT/'liquidity_audit/results.json');allrows=[]
    for name in ['wave1','wave2','wave3','wave4','sealed_extension/development','sealed_extension/evaluation','liquidity_audit']:
        allrows.extend(read(OUT/name/'results.json'))
    details=[]
    for arm in LABEL:
        for scenario in ['base','stress','no_receipts']:
            r=runfile(arm,scenario);m=r['metrics'];realized=sum(f['realized'] for f in r['fills'])
            symbols=[]
            for symbol in r['metrics']['traded_symbols']:
                fills=[f for f in r['fills'] if f['symbol']==symbol]
                symbols.append(dict(symbol=symbol,fills=len(fills),buy_units=sum(max(0,f['signed_qty']) for f in fills),sell_units=sum(max(0,-f['signed_qty']) for f in fills),realized_price_pnl=sum(f['realized'] for f in fills),fees=sum(f['fee'] for f in fills),flat_episodes=sum(f['position_before']!=0 and f['position_after']==0 for f in fills)))
            details.append(dict(arm=arm,scenario=scenario,realized_price_pnl=realized,unrealized_price_pnl=m['net_pnl']-realized+m['fees']-m['funding'],fees=m['fees'],funding=m['funding'],symbols=symbols,daily=r['daily']))
    save(OUT/'attribution.json',details)
    decisions=dict(primary='reversion12_quote3h',primary_new_interval_pass=False,
        execution_control='incumbent_ttl1h',execution_control_higher_extended_pnl_Q_all_three_scenarios=True,
        control_higher_new_interval_pnl_all_three_scenarios=True,control_larger_exposure_and_volatility=True,
        execution_control_is_not_a_new_alpha_model=True,post_exposure_liquidity_sensitivity_all_four_higher=True,
        original_first48h_selection_pass=False,new_outcome_hours=4,completed_daily_returns_total=4,
        independent_test_score_available=False,orders_sent_by_research=0,financial_ranking_not_inferred=True,
        note='The originally nominated newalpha failed its new-outcome criterion. The preregistered execution control improved. Retain the failure and identify this as a small, uninspected outcome extension after extensive development, not robust selection-adjusted generalization.')
    for scenario in ['base','stress','no_receipts']:
        a,b=by['incumbent',scenario],by['incumbent_ttl1h',scenario]
        assert b['net_pnl']>a['net_pnl'] and b['Q']>a['Q'] and b['tail_pnl']>a['tail_pnl']
    save(OUT/'decision.json',decisions)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
    colors={'incumbent':'#7b8793','incumbent_ttl1h':'#1769c2','reversion12_quote3h':'#bd651f'}
    for col,sc in enumerate(['base','stress']):
        for arm in LABEL:
            r=runfile(arm,sc);ss=r['snapshots'];t=np.array([s['time']/1000 for s in ss]);eq=np.array([s['equity'] for s in ss]);dates=[datetime.fromtimestamp(v,timezone.utc) for v in t]
            axes[0,col].plot(dates,eq-100000,label=arm,color=colors[arm],lw=1.5)
            start=1792746000;ix=t>=start;v=eq[np.searchsorted(t,start)]
            axes[1,col].plot(np.array(dates)[ix],eq[ix]-v,color=colors[arm],lw=1.5)
        axes[0,col].set_title(sc+' · full97h (USDP)');axes[1,col].set_title(sc+' · new09–13UTC increment (USDP)')
        for ri,ax in enumerate(axes[:,col]):
            ax.grid(alpha=.2);ax.axhline(0,color='#777',lw=.7)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M' if ri==0 else '%H:%M',tz=timezone.utc))
            ax.tick_params(axis='x',labelsize=9)
    axes[0,0].legend(fontsize=8);fig.savefig(OUT/'equity.png',dpi=155);plt.close(fig)
    contents=[]
    contents.append('<h1>PAS 실행 개선 검증 결과</h1><p class="lead">새로 만든 평균회귀 후보보다, 기존 신호의 주문 의도를 유지하는 실행 수정 비교군이 새 PAS 구간에서 더 좋은 결과를 냈습니다. 아래 숫자는 공개 PAS 자료로 재생한 조건부 모의거래 결과입니다.</p>')
    contents.append('<p>전체 비교: <b>2026-10-19 21:00–10-23 22:00 KST ·97시간</b>. 신규 미평가 결과 구간: <b>10-23 18:00–22:00 KST ·4시간</b>. 우리 전략50,000USDP+비교용 유휴자금50,000USDP. 팀원의 실거래 손익을 합산하지 않았습니다.</p>')
    contents.append('<h2>같은 자본·위험 한도에서의 결과</h2>')
    tab=[]
    for sc in ['base','stress']:
        for arm in LABEL:
            r=by[arm,sc]
            tab.append([sc,LABEL[arm],num(r['net_pnl']),num(r['pnl_after_liquidation_reserve']),pct(r['return_fraction']),pct(r['volatility']),f"{r['Q']:.6f}",f"{r['score']:.6f}",num(r['mean_notional']),pct(r['max_drawdown'])])
    contents.append(table(['조건','전략','22시 순손익 USDP','종료 비용충당 후','21시 완료일 R','기간 σ','Q','0.8Q(J=0)','평균 보유 명목','최대낙폭'],tab))
    contents.append('<p>R·σ·Q는 한국시간21시(12UTC)의 일별 표본4개 수익률로 계산했습니다. 22시 순손익과 시점이 다릅니다. σ는 일별 표본표준편차×√4이며, 일별 표준편차 자체는 이 표의 σ/2입니다. 공식 순위는 주최 측의 실제 계좌 표본과 심사점수로 산정하므로 이 모의 Q를 실제 순위로 바꾸지 않습니다. <a href="https://perpetual-alpha26.github.io/leaderboard">공식 산정 안내</a></p>')
    contents.append('<h2>후보를 고정한 뒤 연 구간</h2>')
    contents.append(table(['조건','전략','신규4시간 손익','청산비용충당 변화 반영','신규구간 체결조각'],[[sc,LABEL[a],num(by[a,sc]['tail_pnl']),num(by[a,sc]['tail_liquidation_adjusted_pnl']),by[a,sc]['tail_fills']] for sc in ['base','stress','no_receipts'] for a in LABEL]))
    contents.append('<p><b>평균회귀 주후보는 새 구간 기준을 통과하지 못했습니다.</b> 실행 수정 비교군은 데이터를 열기 전에 함께 고정돼 있었으며 기본·스트레스·펀딩수취0조건에서 기존판보다 높은 신규구간 손익을 냈습니다. 이를 주후보 성공으로 바꾸어 쓰지 않습니다. 기본 조건에서 실행 수정 비교군의 신규 체결은0건으로, 앞서 확보한 보유분의 이후 가격 변화가 신규구간 이익을 만들었습니다. 스트레스 조건에서는34개 체결조각이 있었습니다.</p>')
    contents.append('<img src="equity.png" alt="PAS replay equity and new-interval incremental equity">')
    contents.append('<h2>무엇을 바꿨고, 무엇이 이익을 만들었나</h2><ol><li><b>실행 수정 비교군:</b> 현행 all_market_signed_core_v1+pullback_any3의 방향·수량 판단,6시간 판단 주기,전10종목 평가를 유지합니다. 주문 의도 유효시간15분을1시간으로 늘리고 축소 요청은 만료 후에도 다음 판단까지 유지합니다. 리스크 감소와 거래비용·부분체결·최소주문수량은 계속 적용됩니다.</li><li><b>새 평균회귀 주후보:</b> 지난12시간 PAS 수익률의−25%를 예상 반대 움직임으로 두고, 과거 변동성으로 예측을 제한합니다. 펀딩·비용·공분산·보유수량을 넣어 양방향 비중을 정하고 가격 조건을3시간 유지합니다. 새로 학습된 대형모델이나 페어트레이딩으로 부르지 않습니다.</li><li><b>성과의 성격:</b> 실행 수정판은 평균 노출과 변동성이 모두 커졌습니다. 같은 자본·상한 안에서 더 많이 보유한 효과가 포함됩니다. 방향 예측력 자체가 좋아졌다는 증거는 아닙니다. 평균회귀판은 개발 구간에서8개 종목별 포지션 청산을 마쳤으나 신규구간 우위가 재현되지 않았습니다.</li></ol>')
    contents.append('<h2>체결량 가정을 바꾼 추가 점검</h2>')
    contents.append(table(['조건','전략','순손익 USDP','Q','평균 보유 명목'],[[r['scenario'],LABEL[r['arm']],num(r['net_pnl']),f"{r['Q']:.6f}",num(r['mean_notional'])] for r in sorted(liq,key=lambda x:(x['scenario'],x['arm']))]))
    contents.append('<p>기본1%·스트레스0.25%의 분봉 거래량 참여 가정 외에10%·2.5%, 그리고100%상한 포화 가정을 검사했습니다.100%는 실제 확보 가능한 유동성 주장이 아니라 만료 효과가 낮은 참여율에만 의존하는지 보는 민감도 검사입니다. 이 점검은 신규구간 결과를 본 <b>후</b> 수행했으며 독립검증으로 세지 않습니다.</p>')
    contents.append('<h2>검증 기록과 해석 범위</h2><ul><li>총197회 재생: 개발171회, 체결 시각 수정 후 개발9회, 고정된 이후 구간9회, 유동성 민감도8회. 실패 결과를 포함해 모두 보존했습니다. 초기48시간을 기준으로 한 원래 선정 조건은 통과하지 못했습니다.</li><li>새 구간을 읽기 전 코드와 후보를 해시로 고정했습니다. 기존 공개 감사에서08:14UTC까지 확인했던 사실을 반영해 신규 구간 시작을09UTC로 미뤘고,끝은 동결 시점의 최신 완료시간13UTC로 정했습니다. 기존 보유분과 손익을 이어 재생했으며 초기화를 하지 않았습니다.</li><li>과거 입력·예측·체결·일별자산·분별자산 경로가 신규 데이터 추가 후에도 앞 구간에서 동일함을 검사했습니다. 판단과 겹치는1분봉은 체결에 쓰지 않았습니다.</li><li>실제 PAS 분봉 고가/저가에 불리한 충격비용을 더하고 실제 거래량 일부만 체결 가능하다고 가정했습니다. 기본 편도수수료9bp+충격10bp,스트레스12bp+20bp입니다. 호가 대기열이나 당시 깊이를 복원한 실체결 결과는 아닙니다.</li><li>실제 시간별 펀딩과 지갑 지급액에서 복원한 단위당 정산금으로 비용을 반영했습니다. 최종 비교9회에서 펀딩가격 대체 추정 사용0건,수량·현금 보존 오차는1e-6USDP미만,리스크 상한 위반0건입니다.</li><li>체결 종가로 평가한 모의계좌이며 공식 mark-price 계좌와 다릅니다.4시간의 신규 관측만으로 장기간 일반화나 심사점수J를 입증하지 않습니다. 모든 결과는 같은 모의 실행기 안의 비교입니다.</li><li>실거래의 시간별 위험보정·동일지갑 순포지션 처리·실제 장부 인계는 별도 동작입니다. 이 실험의 개별 전략 원장을 공식 팀 계좌의 실현성과로 부르지 않습니다.</li></ul>')
    op=OUT/'entry_reconciliation_applied/result.json'
    contents.append('<h2>실제 실행자에서 발견한 오류</h2><p>21:10KST경 ETH평균 진입가 대조에서 실행자가 중단됐습니다. 실제 체결 가중평균2393.307702…와 API표시2393.3의 차이에10.6278ETH를 곱한0.08186USDP가 고정 허용오차0.05를 넘었습니다. 공개 체결의 연속 수량은 모두 일치했습니다.</p><p>종목별szDecimals에서 계산한 표시 정밀도로 대조하도록 고쳤으며 원장의 정확한 평균 진입가는 보존합니다. 수량·현금·메타데이터 불일치는 계속 거부합니다. 표시값으로 평가가격을 역산하던 부분도 정확한 원장 진입가를 사용하게 수정했습니다. 관련 검사58개를 통과했습니다. API의 평균진입가 절삭 형식은 이번 실제 응답과 체결 원장을 대조해 확인한 동작입니다.</p>')
    if op.exists():
        r=read(op);contents.append('<p>복구 대조: 원장수량오차'+str(r['conservation']['conservation']['quantity'])+',원장자산보존오차'+str(r['conservation']['conservation']['equity'])+'USDP. 미확인주문'+str(r['pending_orders'])+'건. 팀원 보유 ETH·BRENT와 우리 보유분을 유지하고 만료된 미제출 의도만 제거했습니다.</p>')
    activation=OUT/'operational_recovery.json'
    if activation.exists():contents.append('<pre>'+html.escape(json.dumps(read(activation),ensure_ascii=False,indent=2))+'</pre>')
    contents.append('<h2>원본·코드·재현 근거</h2>')
    links=[('연구 검증 영수증','research_verification_receipt.json'),('후보 동결','nominee_freeze.json'),('신규구간 노출·평가기준','sealed_extension/protocol.json'),('신규구간 결과','sealed_extension/evaluation/results.json'),('정정된 개발 결과','sealed_extension/development/results.json'),('모든 비교군 해석','decision.json'),('실현·미실현 기여','attribution.json'),('체결량 민감도','liquidity_audit/results.json'),('기존 소스 보존','baseline_source_archive/'),('실행 오류 재현','entry_precision_red.log'),('실행 오류 수정 검사','entry_precision_green.log')]
    contents.append('<ul>'+''.join('<li><a href="'+p+'">'+label+'</a></li>' for label,p in links)+'</ul>')
    contents.append('<p>실험 코드: '+html.escape(str(ROOT/'pas_cashflow_research'))+'<br>실제 실행 오류 수정: candidate_handoff/entry_precision.py,bridge.py. 성능 후보를 실제 주문 실행자로 교체하지 않았습니다.</p>')
    style='body{font:15px/1.7 system-ui,sans-serif;color:#172b3a;background:#f6f8fb;margin:0}main{max-width:1240px;margin:36px auto;background:white;padding:36px;border-radius:16px}h1{font-size:30px}h2{margin-top:36px;font-size:21px}.lead{font-size:18px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}th,td{border-bottom:1px solid #dfe6ec;padding:10px;text-align:right;white-space:nowrap}th{background:#edf3fa}td:nth-child(2),th:nth-child(2){text-align:left}img{max-width:100%}a{color:#1769c2}pre{white-space:pre-wrap;background:#f6f8fb;padding:16px}li{margin:8px 0}'
    (OUT/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PAS 실행 개선 검증</title><style>'+style+'</style><main>'+''.join(contents)+'</main></html>')
    save(OUT/'report_manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),files={p.name:sha(p) for p in [OUT/'report.html',OUT/'equity.png',OUT/'decision.json',OUT/'attribution.json']}))
    print(OUT/'report.html')

if __name__=='__main__':main()
