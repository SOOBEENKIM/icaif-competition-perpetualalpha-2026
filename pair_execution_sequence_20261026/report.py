"""Reproducible research report; saved simulation results only."""
import base64,csv,gzip,html,json
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from .study import ROOT,OUT,PARENT,save

def read(p):
 return json.load(gzip.open(p,'rt')) if str(p).endswith('.gz') else json.loads(p.read_text())
def fmt(v,n=2):return '—' if v is None else f'{v:+,.{n}f}'
def table(head,rows):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(c))+'</th>' for c in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def kst(t):return datetime.fromtimestamp(t,timezone(timedelta(hours=9))).strftime('%m/%d %H:%M:%S')
def run(arm,sc,ph):
 p=OUT/'runs'/f'execution_repaired_{sc}_{ph}.json.gz' if arm=='final' else PARENT/'quotes'/f'{arm}_{sc}_{ph}.json.gz'
 return read(p)

def main():
 audit=read(OUT/'final_audit.json');assert audit['live_sources_unchanged']
 rows=read(OUT/'results.json');repair=read(ROOT/'reports/pair_execution_repair_20261026/results.json')
 comparisons=[];allrows=[];decomp=[];trades=[]
 for ph in [0,45]:
  for sc in ['base','stress']:
   rr=[r for r in rows if r['phase']==ph and r['scenario']==sc]
   a=next(r for r in rr if r['arm']=='incumbent');b=next(r for r in rr if r['arm']=='execution_repaired')
   rec=dict(scenario=sc,polling_phase_seconds=ph,incumbent_net=a['net_pnl'],candidate_net=b['net_pnl'],
    net_delta=b['net_pnl']-a['net_pnl'],incumbent_liquidated=a['net_after_terminal_exit'],
    candidate_liquidated=b['net_after_terminal_exit'],liquidated_delta=b['net_after_terminal_exit']-a['net_after_terminal_exit'])
   comparisons.append(rec)
   for r in [a,next(r for r in rr if r['arm']=='pas_spread_reversion'),
             next(r for r in repair if r['phase']==ph and r['scenario']==sc and r['arm']=='execution_repaired'),b]:
    allrows.append(dict(scenario=sc,phase=ph,version=['incumbent','new_forecast_old_execution','continuation_after_both_legs','receipt_sized_sequence'][len(allrows)%4],
     net_pnl=r['net_pnl'],terminal_liquidation_pnl=r['net_after_terminal_exit'],fees=r['fees'],funding=r['funding'],fills=r['fills'],valid=r['valid']))
   v=run('final',sc,ph);old=run('incumbent',sc,ph)
   decomp.append([sc,ph,fmt(v['price_pnl']),fmt(-v['fee_increment']),fmt(v['funding_increment']),fmt(v['net_pnl']),
     f"{old['snapshots'][-1]['notional']:,.2f} → {v['snapshots'][-1]['notional']:,.2f}"])
   for f in v['fills']:
    trades.append([sc,ph,kst(f['time']),f['symbol'],'매수' if f['signed_qty']>0 else '매도',f"{abs(f['signed_qty']):g}",f"{f['price']:,.5f}",f"{f['fee']:.5f}"])
 for name,data in [('comparisons.csv',comparisons),('all_versions.csv',allrows)]:
  with (OUT/name).open('w',newline='') as fp:
   writer=csv.DictWriter(fp,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
 summary=dict(conclusion='Development PnL improved in all four quote conditions; terminal-exit robustness fails in one. Not qualified for live promotion.',
  comparisons=comparisons,official_Q=None,independent_test=False,live_changed=False,
  completed_successful_pair_roundtrips=0,validation_window_hours=21900/3600,capital=50000,
  implementation='Fixed all45 PAS spread reversion forecasts plus receipt-sized execution; unchanged EMA core.',
  tests=22,tests_passed=True,audit='final_audit.json')
 save(OUT/'summary.json',summary)
 plt.style.use('seaborn-v0_8-whitegrid')
 fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
 for ax,(ph,sc) in zip(axes.flat,[(0,'base'),(0,'stress'),(45,'base'),(45,'stress')]):
  for arm,color,label in [('incumbent','#65758b','Incumbent'),('pas_spread_reversion','#d69244','New signal / old execution'),('final','#009979','Receipt-sized execution')]:
   v=run(arm,sc,ph);ss=v['snapshots'];x=[datetime.fromtimestamp(s['time'],timezone.utc) for s in ss]
   ax.plot(x,[s['equity']-ss[0]['equity'] for s in ss],color=color,label=label,lw=1.6)
  ax.set_title(f'{sc.capitalize()} | polling offset {ph}s');ax.set_ylabel('Net PnL (USDP)');ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=timezone.utc))
  ax.axhline(0,color='#333',lw=.5)
 axes[0,0].legend(fontsize=8);axes[1,0].set_xlabel('25 October 2026 UTC');axes[1,1].set_xlabel('25 October 2026 UTC')
 fig.suptitle('PAS recorded-book replay | same starting inventory | development data',fontsize=12)
 fig.tight_layout();fig.savefig(OUT/'pnl_paths.png',dpi=160);fig.savefig(OUT/'pnl_paths.pdf');plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4));x=np.arange(4);w=.35
 ax.bar(x-w/2,[r['net_delta'] for r in comparisons],w,label='After entry fees + funding',color='#009979')
 ax.bar(x+w/2,[r['liquidated_delta'] for r in comparisons],w,label='Including terminal depth exit',color='#526ca6')
 ax.set_xticks(x,[f"{r['scenario']} / {r['polling_phase_seconds']}s" for r in comparisons]);ax.axhline(0,color='#333',lw=.8)
 ax.set_ylabel('Improvement over incumbent (USDP)');ax.legend(fontsize=9);fig.tight_layout()
 fig.savefig(OUT/'cost_robustness.png',dpi=160);fig.savefig(OUT/'cost_robustness.pdf');plt.close(fig)
 diagnostic=read(PARENT/'selection.json')['all_candidates']
 models={'basis_reversion':'PAS–기준시장 괴리 회귀','relative_trend':'외부 상대 추세 지속','ema_basis_blend':'괴리 회귀 + 상대 추세 50:50','relative_reversal':'외부 상대 추세의 반전','pas_spread_reversion':'PAS 두 가격의 스프레드 회귀'}
 native=[]
 for phase in ['development','later','switch']:
  for sc in ['base','stress','no_receipts']:
   a=read(PARENT/'rows'/f'incumbent_{sc}_{phase}.json');b=read(OUT/'native/rows'/f'pas_spread_reversion_{sc}_{phase}.json')
   native.append([phase,sc,fmt(a['net_pnl'],5),fmt(b['net_pnl'],5),fmt(b['net_pnl']-a['net_pnl'],5),b['pair_fills'],b['completed_pairs'],b['funding_proxy_events']])
 m=read(ROOT/'reports/ema_pairs_pas_20261026/current_market.json');market=[]
 for meta,ctx in zip(m['meta']['universe'],m['contexts']):
  sym=meta['name'].split(':')[-1];book=m['books'][sym]['book']['levels'];bid=float(book[0][0]['px']);ask=float(book[1][0]['px'])
  market.append([sym,fmt((float(ctx['markPx'])/float(ctx['oraclePx'])-1)*10000),f"{(ask/bid-1)*10000:.2f}",fmt(float(ctx['funding'])*10000,4)])
 comparison_table=table(['조건 / 시작 지연','기존 순손익','최종 후보 순손익','개선액','기존 전량 청산 후','후보 전량 청산 후','청산 후 개선액'],
  [[r['scenario']+' / '+str(r['polling_phase_seconds'])+'초',*[fmt(r[k]) for k in ['incumbent_net','candidate_net','net_delta','incumbent_liquidated','candidate_liquidated','liquidated_delta']]] for r in comparisons])
 doc=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EMA + PAS 페어 전략 검증 · 2026-10-26</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;background:#f4f6f9;color:#17253b;margin:0}}main{{max-width:1160px;margin:auto;padding:36px 24px}}h1{{font-size:30px;line-height:1.35}}h2{{margin-top:40px;font-size:23px}}h3{{font-size:18px}}.card{{background:white;border:1px solid #dfe5eb;border-radius:12px;padding:20px;margin:20px 0}}.warn{{border-left:5px solid #bd7421}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;background:#fff;font-size:14px}}th,td{{padding:10px 12px;text-align:right;border-bottom:1px solid #e2e7ee;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{background:#e9eff5}}img{{max-width:100%;display:block;margin:20px auto}}a{{color:#135eab}}code{{font-size:13px;background:#e9eef5;padding:2px 4px}}.muted{{color:#56657a}}li{{margin:7px 0}}details{{margin:20px 0}}summary{{cursor:pointer;font-weight:650}}</style>
<main><p class="muted">SFINX · 2026-10-26 · 오프라인 개발 평가 · 단위 USDP</p>
<h1>EMA 전략에 PAS 페어 전략을 결합하고,<br>실제 체결 순서까지 고쳐 비교한 결과</h1>
<div class="card"><b>최종 후보의 평가손익은 네 조건 모두 개선됐음. 전량 청산 비용까지 포함한 개선은 4개 중 3개 조건에서 확인됨.</b>
<p>같은 보유분·자본·위험 한도로 비교해 기존판보다 순손익이 9.29~13.88 USDP 개선됐음. 그러나 가장 불리한 한 조건의 청산 후 손익은 1.13 USDP 악화됨. 평가 구간 전체는 여전히 손실이며, 완결된 수익성 있는 페어 거래를 다수 확보한 결과가 아님. 이번 결과만으로 실전 교체 적격이라고 판단하지 않았음.</p>
<p>실운용 전략·서비스·주문 설정은 변경하지 않았음. 아래 거래는 모두 모의 체결임.</p></div>
<h2>1. 동일 조건 비교 결과</h2>
<p><b>기록된 PAS 실제 호가: 한국시간 10/25 18:00–10/26 00:05, 총 6시간 5분.</b> 기존판과 후보에 당시 우리 전략의 동일 보유분과 현금을 인계했음. 최초 평가자산 50,068.48 USDP, 운용 기준자본 50,000 USDP임. 팀원 전략의 독립 손익을 합친 팀 전체 실적표가 아님.</p>
{comparison_table}
<p>‘순손익’은 평가가격 변동에 체결 수수료·펀딩을 반영한 미청산 손익임. ‘전량 청산 후’는 공통 종료시점의 실제 관측 호가 수량으로 모든 잔여 보유분을 청산하고 수수료까지 내는 가정임. 관측 호가가 실제 주문에도 그대로 유지된다는 보장은 없음.</p>
<p>기본 조건: 편도 수수료 9bp, 관측 수량 100%, 주문 전 계산 90초 + 전송 1초, 펀딩 수취 100%. 불리한 조건: 12bp, 관측 수량 50%, 계산 120초 + 전송 3초, 수취 50%. 펀딩 지급은 전액 부담함. 실행 루프 시작점을 0초/45초로 바꿔 비교했음. 4개 행은 <b>같은 가격 경로에 대한 민감도 분석</b>이며 독립 표본 4개가 아님.</p>
<img src="cost_robustness.png" alt="기존판 대비 순손익과 청산 후 손익 개선액">
<img src="pnl_paths.png" alt="네 체결 조건별 기존판, 신호 변경판, 실행 수정판의 손익 경로">
<h2>2. 무엇을 결합했는가</h2>
<ol><li><b>EMA·돌파·7일 되돌림·Chronos 보정 코어를 비교 기준 그대로 유지함.</b> 이번에 Chronos를 재학습하거나 새 모델로 바꾸지 않았음.</li>
<li><b>10종목 전체의 45개 페어를 평가함.</b> 최근 최대 720시간의 외부 기준시장 수익률로 관계와 헤지비율을 구함. 실제 거래 관측 120개 이상, 양의 상관 0.2 이상, 헤지비율 0.1~10 조건을 사용함. 종목 이름으로 배제하지 않음. 조건 미달 페어는 거래하지 않음.</li>
<li><b>수익 기회는 PAS 가격으로 판단함.</b> 선택된 후보는 log(PAS A 가격) − β × log(PAS B 가격)의 괴리가 다음 24시간에 얼마나 줄어들지 추정함. 최대 96시간 안의 유효한 인접 시간 간격 36개 이상을 사용하고, 회귀계수 불확실성만큼 기대 움직임을 줄임. 회귀가 확인되지 않거나 추정 반감기가 72시간을 넘으면 가격 회귀 신호를 0으로 둠.</li>
<li><b>예상 가격 이익 + 펀딩 − 왕복 비용이 유리한 방향만 허용함.</b> 부호가 반대면 반대편 롱·숏을 평가함. 수취 펀딩은 보수적으로 줄여 예상하고, 지급은 전액 반영함. 기본 예상 우위 43bp 필터를 넘더라도 현재 호가·수량·한쪽만 체결되는 위험·포트폴리오 한도를 다시 통과해야 함.</li>
<li><b>페어는 별도 내부 장부로 유지하고 최종 주문은 기존 보유분과 합쳐 냄.</b> 동시에 한 페어, 최대 명목금액 합계 6,000 USDP, 최대 보유 24시간임. 기존 장부의 ETH 숏과 페어 장부의 ETH 롱은 거래소에서 합산되므로 화면의 포지션과 가상 전략별 포지션은 다름.</li></ol>
<div class="card warn">이 후보는 ‘수익률 관계로 헤지하고 PAS 스프레드 회귀를 추정하는 상대가치 전략’임. 상관계수와 짧은 AR 추정만으로 장기 공적분이나 시장 중립성이 증명되지는 않음. ETH–GOLD 같은 서로 다른 자산군이 선택된 것도 검증 대상이며, 이름을 미리 지정한 결과는 아님.</div>
<h2>3. 발견하고 수정한 실행 문제</h2>
<p><b>판단 시각:</b> 기존 코드는 15분 구간의 첫 30초에만 페어를 검토했음. 90초 주기 루프가 45초 늦게 시작하면 검토 시각을 계속 놓쳤음. 연구 실행자에서는 각 15분 구간에서 한 번 검사하도록 수정했음.</p>
<p><b>포지션 반전:</b> 기존 ETH 숏을 닫은 직후 거래소 수량이 0이 되면 첫 매수는 중단되는데, 반대편 GOLD 주문은 목표량대로 나갔음. 이후 불균형을 줄이려고 GOLD를 되사며 비용을 냈음. 최종 수정은 ① 기존 반대 포지션 청산 체결 확인 → ② 원래 가격 한도 안에서 첫 다리 진입 → ③ 실제 체결 수량에 맞춰 반대 다리 주문 순서임. 미체결·부분 청산·응답 미확인 시 임의로 완료됐다고 처리하지 않음.</p>
<p>불리한 0초 조건에서는 ETH 청산이 일부만 체결돼 해당 시도가 정리되고, 이후 HYPE 롱–GOLD 숏이 체결됐음. 다른 세 조건에서는 ETH 롱–GOLD 숏으로 이어졌음. 단순히 원하는 목표량을 체결된 것으로 계산하지 않았음.</p>
{table(['조건','시작 지연','가격 손익','수수료','펀딩','최종 순손익','종료 명목금액 기존 → 후보'],decomp)}
<p>기본 조건의 종료 총명목금액이 약 1.23만에서 1.74만 USDP로 증가함. 같은 한도를 적용했지만 실제 노출이 같지는 않음. 따라서 손익 차이를 모두 예측력 개선으로 해석하거나 ‘같은 위험에서 개선됐다’고 주장하지 않음.</p>
<h2>4. 후보들을 어떻게 비교했는가</h2>
<p>10/19 21:00–10/24 21:00 KST 구간에서 5개 가설을 비교했음. 아래는 24시간 뒤 가격을 관측할 수 있는 신호의 경제성 진단임. 거래 체결을 가정한 계좌 수익률이 아니며, 시간상 겹치는 신호가 포함돼 있음. 시점은 3개 날짜에 집중되어 있음.</p>
{table(['후보','신호 수','기본 비용 후 평균','불리한 비용 후 평균'],[[models[r['arm']],r['signals'],fmt(None if r['base_mean'] is None else 100*r['base_mean'],4)+'%',fmt(None if r['stress_mean'] is None else 100*r['stress_mean'],4)+'%'] for r in diagnostic])}
<p>단순 외부 상대 추세보다 PAS 스프레드 회귀가 나았지만, 후자도 불리한 비용을 빼면 평균이 음수였음. 가장 덜 나쁜 후보를 골랐다는 사실과 수익성이 검증됐다는 주장은 구분해야 함. 회귀 후보를 선택한 뒤 10/24 21:00–10/26 21:00 KST를 시간순으로 평가했음. 이후 실행 수정은 이미 확인한 자료를 재사용한 개발임.</p>
<h2>5. 긴 구간 검증의 결과와 한계</h2>
{table(['구간','비용 조건','기존 손익','최종 후보 손익','차이','페어 체결 조각','완료 페어','펀딩 가격 대용 사용'],native)}
<p>development는 초기 120시간, later는 이후 48시간을 새 계좌로 시작, switch는 전체 168시간 중 처음 120시간을 기존판으로 운용한 뒤 보유분을 그대로 인계한 경우임. 세 체결 조건 모두 교체 이전 계좌 상태가 같은지 대조했음.</p>
<p><b>이 긴 구간은 PAS 분봉을 이용한 조건부 모의 체결임.</b> 호가가 없는 시점은 체결시키지 않고, 제한된 거래량·다음 분봉의 불리한 가격·원래 주문 가격 한도를 적용하니 기존판도 거래가 거의 발생하지 않았음. 이후 48시간의 개선액은 약 0.042 USDP에 그침. 수수료가 더 불리한 두 조건은 양쪽 모두 무체결임. 이를 ‘장기간 검증 통과’ 또는 ‘실대회 PnL 예측’으로 취급하지 않음. 펀딩 현금 단위가 없는 시점은 PAS 가격 대용을 썼고 그 수를 표에 공개했음.</p>
<h2>6. 현재 PAS와 외부 시장의 차이를 어떻게 반영했는가</h2>
<p>10/26 {kst(m['received_at']).split(' ')[1]} KST 공개 조회. 아래는 한 시점의 관측이며 이 값으로 과거 주문을 결정하지 않았음. 1bp = 0.01%.</p>
{table(['종목','PAS 평가가 / 오라클 괴리 bp','매수·매도 호가 차이 bp','시간당 펀딩 bp'],market)}
<p>BTC·ETH·HYPE 등은 음의 펀딩이 관측됐음. 음의 펀딩에서는 숏이 롱에게 지급하므로, 하락 방향을 맞혀도 가격 이익이 펀딩과 체결 비용을 넘지 못할 수 있음. 외부 시장은 관계 추정에만 쓰고, PAS 가격·공개 호가·수량·펀딩과 수신 시각을 주문 판단과 성과 평가에 사용했음. <a href="https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding">Hyperliquid 공식 펀딩 설명</a></p>
<h2>7. 과적합·오류 점검</h2>
<ul><li>현재 시점 뒤 가격·거래량을 크게 바꿔도 현재 예측이 바뀌지 않는 검사, 45페어 전수 평가, 누락 구간 연결 금지, 가격 단위 변경, 펀딩 방향, 주문 응답 미확인, 부분 청산, 원래 가격 한도, 15분 평가 주기 등 <b>22개 검사 통과</b>.</li>
<li>6개 예측 파일 묶음을 현재 코드로 다시 생성해 저장된 결과와 완전히 일치함을 확인했음. 원본 데이터 {audit['immutable_data_files_checked']}개 해시 대조, 자산·현금·수수료·펀딩 장부 오차 최대 {audit['maximum_accounting_error']:.3g} USDP, 위험 한도 위반·미확인 주문·미해결 주문 없음.</li>
<li>기존 source freeze가 v3 대신 v1 폴더를 가리키는 검증 보조 코드 오류도 발견했음. 원본 기록을 보존하고 폴더 선택과 누락 검사를 수정했으며, <b>평가 후 수정한 해시</b>라고 별도 기록했음. 이를 평가 전 코드 고정의 완전한 증명이라고 주장하지 않음. 주문 수와 체결 수를 혼동한 테스트 기대값 1개도 수정했고 원래 실패 로그를 남겼음.</li>
<li>후보 선택에 반복 사용한 과거 자료는 개발 자료임. 이후 48시간도 일부 실거래 상황이 이미 알려져 있으므로 완전히 손대지 않은 테스트로 인증하지 않음. 호가 구간을 보고 실행을 수정했으므로 그 결과 역시 개발 결과임. 과적합이 없다고 보증하지 않음.</li></ul>
<div class="card warn"><b>대회 점수 Q나 예상 순위는 산출하지 않았음.</b> 실호가 구간은 하루보다 짧아 완료된 일별 수익률 표준편차를 계산할 수 없음. 긴 분봉 모의평가의 무체결 Q=0.1 같은 수치도 참가 자격·실제 거래 성과를 증명하지 못하므로 성과 표에서 제외했음. 공식 평가는 12:00 UTC 일별 스냅숏에 기반함. <a href="https://perpetual-alpha26.github.io/leaderboard">공식 리더보드</a></div>
<h2>8. 실전 교체 판단</h2>
<p><b>이번에 확보한 것은 개선된 실행 코드와 개발 구간의 손익 개선임. 실전 교체를 뒷받침하는 완결된 검증은 아직 아님.</b> 다음 평가에서는 이 후보를 바꾸지 않고 새로운 PAS 기록으로 실제 최대 보유기간과 청산까지 확인해야 함. 비용·지연을 불리하게 해도 전량 청산 후 개선이 유지되는지, 여러 날짜·여러 페어에서 반복되는지, 늘어난 노출을 맞춰도 유리한지를 함께 봐야 함. 같은 6시간을 재차 맞추는 방식으로 ‘독립 검증 통과’를 만들지 않음.</p>
<details><summary>전체 버전 비교 — 실패 결과 포함</summary>{table(['조건','시작 지연','버전','순손익','전량 청산 후','수수료','펀딩','체결 조각'],[[r['scenario'],r['phase'],r['version'],*[fmt(r[k]) for k in ['net_pnl','terminal_liquidation_pnl','fees','funding']],r['fills']] for r in allrows])}</details>
<details><summary>최종 후보의 모든 모의 체결</summary>{table(['조건','시작 지연','KST','종목','행동','수량','가격','수수료'],trades)}</details>
<details><summary>논문에서 가져온 원칙과 구현 범위</summary><ul>
<li><a href="https://link.springer.com/article/10.1186/s40854-024-00702-7">Copula-based trading of cointegrated cryptocurrency pairs</a>: 관계 선택과 진입·청산 규칙, 거래 비용을 분리해 비교하는 관점을 참고함. 이 연구의 copula 모델을 재현했다고 주장하지 않음.</li>
<li><a href="https://arxiv.org/abs/2403.07998">Pairs Trading Using a Novel Graphical Matching Approach</a>: 여러 페어에 같은 자산이 중복돼 위험이 집중되는 문제를 참고함. 이번 후보는 동시에 한 페어를 보유하며 해당 그래프 최적화를 구현한 것은 아님.</li>
<li><a href="https://arxiv.org/abs/2609.35359">From Cointegration to Out-of-Sample Failure</a>: 형성 구간의 회귀가 이후 수익으로 이어지는지 따로 검증해야 한다는 사례로 참고한 2026년 preprint임. 심사 완료 논문이나 수익 보장으로 인용하지 않음.</li></ul></details>
<h2>재현 자료</h2><p><a href="comparisons.csv">최종 비교 CSV</a> · <a href="all_versions.csv">전체 버전 CSV</a> · <a href="summary.json">결론 JSON</a> · <a href="final_audit.json">검증 기록</a> · <a href="audit_corrections.json">검증 보조 코드 수정 기록</a> · <a href="verification_tests_final.log">22개 검사 로그</a> · <a href="pnl_paths.pdf">손익 그래프 PDF</a></p>
<p>모델: <code>ema_pairs_pas_20261026_v3/model.py</code><br>실행 수정: <code>pair_execution_sequence_20261026/account.py</code>, <code>coordinator.py</code><br>실험 실행: <code>.venv/bin/python -m pair_execution_sequence_20261026.study run</code><br>검증·보고서: <code>.venv/bin/python -m pair_execution_sequence_20261026.audit</code>, <code>.venv/bin/python -m pair_execution_sequence_20261026.report</code></p>
</main></html>'''
 doc=doc.replace('.venv/bin/python -m pair_execution_sequence_20261026.report','python3 -m pair_execution_sequence_20261026.report')
 for image in ['pnl_paths.png','cost_robustness.png']:
  doc=doc.replace('src="'+image+'"','src="data:image/png;base64,'+base64.b64encode((OUT/image).read_bytes()).decode()+'"')
 (OUT/'report.html').write_text(doc)
 print(OUT/'report.html')

if __name__=='__main__':main()
